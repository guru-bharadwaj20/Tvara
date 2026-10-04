"""Single-owner iteration scheduler; backend adapters own numerical execution."""

from dataclasses import dataclass, field
from enum import Enum
import time


class State(Enum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Request:
    identifier: str
    prompt: tuple[int, ...]
    max_tokens: int
    priority: int = 0
    state: State = State.QUEUED
    position: int = 0
    output: list[int] = field(default_factory=list)
    submitted: float = field(default_factory=time.perf_counter)
    token_times: list[float] = field(default_factory=list)
    error: str | None = None

    def __post_init__(self):
        self.prompt = tuple(self.prompt)
        if type(self.priority) is not int or not -100 <= self.priority <= 100:
            raise ValueError("Invalid priority")
        if (
            not self.identifier
            or not self.prompt
            or any(type(t) is not int or t < 0 for t in self.prompt)
            or type(self.max_tokens) is not int
            or self.max_tokens < 0
        ):
            raise ValueError("Invalid scheduler request")


class Scheduler:
    def __init__(self, backend, max_active=8, prefill_chunk=64, kv_tokens=4096):
        if type(max_active) is not int or max_active < 1:
            raise ValueError("Invalid active limit")
        if type(prefill_chunk) is not int or prefill_chunk < 1:
            raise ValueError("Invalid prefill chunk")
        if type(kv_tokens) is not int or kv_tokens < 1:
            raise ValueError("Invalid KV token budget")
        self.iteration = 0
        self.enqueued = {}
        self.kv_tokens = kv_tokens
        self.prefill_chunk = prefill_chunk
        self.backend = backend
        self.max_active = max_active
        self.requests = {}
        self.active = []
        self.queue = []

    def submit(self, request):
        if request.identifier in self.requests:
            raise ValueError("Duplicate request ID")
        if request.state is not State.QUEUED:
            raise ValueError("Request already started")
        if len(request.prompt) + request.max_tokens > self.kv_tokens:
            raise ValueError("Request exceeds KV reservation")
        self.requests[request.identifier] = request
        self.queue.append(request)
        self.enqueued[request.identifier] = self.iteration

    def admit(self):
        self.queue.sort(
            key=lambda r: (
                -(r.priority + (self.iteration - self.enqueued[r.identifier]) // 8),
                self.enqueued[r.identifier],
            )
        )
        while self.queue and len(self.active) < self.max_active:
            request = self.queue[0]
            reserved = sum(len(r.prompt) + r.max_tokens for r in self.active)
            if reserved + len(request.prompt) + request.max_tokens > self.kv_tokens:
                break
            self.queue.pop(0)
            if request.state is State.CANCELLED:
                continue
            try:
                self.backend.create(request.identifier)
            except Exception:
                request.state = State.FAILED
                request.error = "Backend admission failed"
                raise
            request.state = State.PREFILL
            self.active.append(request)

    def step(self):
        self.iteration += 1
        self.reap()
        self.admit()
        # Decode-ready requests advance while other requests prefill one chunk.
        for r in self.active:
            if r.state is State.PREFILL:
                replay = r.prompt + tuple(r.output)
                end = min(len(replay), r.position + self.prefill_chunk)
                self.backend.prefill(r.identifier, replay[r.position : end])
                r.position = end
                if end == len(replay):
                    r.state = State.DECODE
        batch = [r for r in self.active if r.state is State.DECODE and len(r.output) < r.max_tokens]
        if batch:
            tokens = list(self.backend.decode_batch([r.identifier for r in batch]))
            if len(tokens) != len(batch) or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("Invalid backend batch")
            for r, token in zip(batch, tokens):
                r.output.append(token)
                r.token_times.append(time.perf_counter())
        for r in self.active:
            if len(r.output) >= r.max_tokens:
                r.state = State.FINISHED
        return bool(self.active or self.queue)

    def reap(self):
        for r in list(self.active):
            if r.state in {State.FINISHED, State.CANCELLED, State.FAILED}:
                self.backend.drop(r.identifier)
                self.active.remove(r)

    def run(self, max_iterations=100000):
        for _ in range(max_iterations):
            self.step()
            self.reap()
            if not self.active and not self.queue:
                return
        raise RuntimeError("Scheduler iteration budget exhausted")

    def preempt(self, identifier):
        r = self.requests[identifier]
        if r not in self.active or r.state not in {State.PREFILL, State.DECODE}:
            raise ValueError("Request is not preemptible")
        self.backend.drop(identifier)
        self.active.remove(r)
        r.state = State.PREEMPTED
        r.position = 0

    def pressure(self, new_budget):
        if type(new_budget) is not int or new_budget < 1:
            raise ValueError("Invalid KV budget")
        while sum(len(r.prompt) + r.max_tokens for r in self.active) > new_budget:
            victim = min(self.active, key=lambda r: (r.priority, -self.enqueued[r.identifier]))
            self.preempt(victim.identifier)
        self.kv_tokens = new_budget

    def resume(self, identifier):
        r = self.requests[identifier]
        if r.state is not State.PREEMPTED:
            raise ValueError("Request is not preempted")
        if len(r.prompt) + r.max_tokens > self.kv_tokens:
            raise ValueError("Insufficient KV reservation")
        # Replay only tokens already emitted; never sample a second copy.
        r.state = State.QUEUED
        self.queue.append(r)
        self.enqueued[identifier] = self.iteration

    def cancel(self, identifier):
        r = self.requests[identifier]
        if r.state in {State.FINISHED, State.CANCELLED, State.FAILED}:
            return
        r.state = State.CANCELLED
        if r in self.queue:
            self.queue.remove(r)
        self.reap()


class NativeSchedulerBackend:
    """Serial native execution with iteration batching; no parallel GEMM claim."""

    def __init__(self, model):
        self.model = model
        self.sequences = {}
        self.pending = {}

    def create(self, key):
        self.sequences[key] = self.model._new_sequence()

    def prefill(self, key, tokens):
        for token in tokens:
            self.pending[key] = self.model.logits(self.sequences[key], token)

    def decode_batch(self, keys):
        result = []
        for key in keys:
            logits = self.pending[key]
            if not __import__("numpy").isfinite(logits).all():
                raise ValueError("Nonfinite model logits")
            token = int(logits.argmax())
            result.append(token)
            self.pending[key] = self.model.logits(self.sequences[key], token)
        return result

    def drop(self, key):
        self.model._drop_sequence(self.sequences.pop(key))
        self.pending.pop(key, None)


def conversation_load(count=16, seed=0, vocab=257):
    import random

    if count < 1 or vocab < 1:
        raise ValueError("Invalid synthetic workload")
    rng = random.Random(seed)
    return [
        Request(
            f"conversation-{i}",
            tuple(rng.randrange(vocab) for _ in range(rng.randint(1, 12))),
            rng.randint(1, 8),
        )
        for i in range(count)
    ]


def request_latency(request):
    return {
        "ttft_seconds": request.token_times[0] - request.submitted if request.token_times else None,
        "tokens": len(request.output),
        "state": request.state.value,
    }
