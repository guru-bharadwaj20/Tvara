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
    def __init__(self, backend, max_active=8, prefill_chunk=64):
        if type(max_active) is not int or max_active < 1:
            raise ValueError("Invalid active limit")
        if type(prefill_chunk) is not int or prefill_chunk < 1:
            raise ValueError("Invalid prefill chunk")
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
        self.requests[request.identifier] = request
        self.queue.append(request)

    def admit(self):
        while self.queue and len(self.active) < self.max_active:
            request = self.queue.pop(0)
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
        self.admit()
        for r in self.active:
            if r.state is State.PREFILL:
                end = min(len(r.prompt), r.position + self.prefill_chunk)
                self.backend.prefill(r.identifier, r.prompt[r.position : end])
                r.position = end
                if end == len(r.prompt):
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
