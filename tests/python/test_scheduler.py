from tvara.scheduler import Request, State, Scheduler
import pytest


def test_lifecycle_contract():
    assert Request("a", (1,), 2).state is State.QUEUED
    for args in [("", (1,), 2), ("a", (), 2), ("a", (-1,), 2), ("a", (1,), -1)]:
        with pytest.raises(ValueError):
            Request(*args)


class Backend:
    def __init__(self):
        self.live = {}
        self.batches = []

    def create(self, key):
        self.live[key] = []

    def prefill(self, key, tokens):
        self.live[key].extend(tokens)

    def decode_batch(self, keys):
        self.batches.append(tuple(keys))
        return [len(self.live[k]) % 7 for k in keys]

    def drop(self, key):
        del self.live[key]


def test_admission_limits_and_duplicates():
    s = Scheduler(Backend(), 1)
    a = Request("a", (1,), 2)
    s.submit(a)
    s.submit(Request("b", (2,), 1))
    s.admit()
    assert s.active == [a] and len(s.queue) == 1
    with pytest.raises(ValueError):
        s.submit(Request("a", (3,), 1))


def test_iteration_batch():
    b = Backend()
    s = Scheduler(b)
    for k in ["a", "b"]:
        s.submit(Request(k, (1, 2), 2))
    s.step()
    assert b.batches == [("a", "b")]
    assert all(len(r.output) == 1 for r in s.active)


def test_completion_releases_resources():
    b = Backend()
    s = Scheduler(b, 1)
    for i in range(3):
        s.submit(Request(str(i), (1,), i))
    s.run()
    assert not b.live and all(r.state is State.FINISHED for r in s.requests.values())


def test_prefill_chunk_bounds():
    b = Backend()
    s = Scheduler(b, prefill_chunk=2)
    r = Request("a", (1, 2, 3, 4, 5), 1)
    s.submit(r)
    s.step()
    assert r.position == 2 and not r.output
    s.step()
    assert r.position == 4
    s.step()
    assert r.state is State.FINISHED


def test_decode_progress_during_long_prefill():
    b = Backend()
    s = Scheduler(b, prefill_chunk=1)
    a = Request("a", (1,), 4)
    c = Request("c", (1,) * 10, 1)
    s.submit(a)
    s.step()
    s.submit(c)
    s.step()
    assert len(a.output) == 2 and c.position == 1 and not c.output


def test_kv_admission():
    s = Scheduler(Backend(), kv_tokens=4)
    s.submit(Request("a", (1, 2), 2))
    s.submit(Request("b", (1,), 1))
    s.admit()
    assert len(s.active) == 1
    s.run()
    assert all(r.state is State.FINISHED for r in s.requests.values())
    with pytest.raises(ValueError):
        s.submit(Request("c", (1,) * 5, 0))


def test_priority_and_aging():
    s = Scheduler(Backend(), 1)
    s.submit(Request("low", (1,), 1))
    s.submit(Request("high", (1,), 1, priority=2))
    s.admit()
    assert s.active[0].identifier == "high"
    s.iteration = 32
    s.reap()
    s.active[0].state = State.FINISHED
    s.reap()
    s.admit()
    assert s.active[0].identifier == "low"


def test_memory_pressure_preempts_and_releases():
    b = Backend()
    s = Scheduler(b, kv_tokens=8)
    s.submit(Request("a", (1,), 3))
    s.submit(Request("b", (2,), 3, priority=1))
    s.step()
    s.pressure(4)
    assert s.requests["a"].state is State.PREEMPTED and "a" not in b.live


def test_resume_and_cancel():
    b = Backend()
    s = Scheduler(b)
    r = Request("a", (1, 2), 3)
    s.submit(r)
    s.step()
    first = list(r.output)
    s.preempt("a")
    s.resume("a")
    s.step()
    assert r.output[:1] == first and len(r.output) == 2
    s.cancel("a")
    assert not b.live and r.state is State.CANCELLED


def test_seeded_conversation_load():
    from tvara.scheduler import conversation_load

    a = conversation_load()
    b = conversation_load()
    assert [(r.prompt, r.max_tokens) for r in a] == [(r.prompt, r.max_tokens) for r in b]
    s = Scheduler(Backend(), max_active=3, prefill_chunk=2)
    for r in a:
        s.submit(r)
    s.run()
    assert all(len(r.output) == r.max_tokens for r in a)


def test_ttft_matches_monotonic_timestamps():
    from tvara.scheduler import request_latency

    r = Request("a", (1,), 2)
    r.submitted = 10
    r.token_times = [10.5, 10.7]
    r.output = [2, 3]
    assert request_latency(r)["ttft_seconds"] == 0.5
    assert request_latency(Request("b", (1,), 0))["ttft_seconds"] is None


def test_intertoken_percentiles():
    from tvara.scheduler import latency_report

    r = Request("a", (1,), 3)
    r.submitted = 1
    r.token_times = [2, 2.2, 2.6]
    out = latency_report([r])
    assert out["inter_token"]["count"] == 2
    assert out["inter_token"]["p50_seconds"] == pytest.approx(0.3)
    assert latency_report([])["ttft"]["p99_seconds"] is None


def test_cancellation_and_starvation_under_load():
    from tvara.scheduler import conversation_load

    b = Backend()
    s = Scheduler(b, 2, prefill_chunk=1, kv_tokens=64)
    requests = conversation_load(40, seed=29)
    for r in requests:
        s.submit(r)
    s.step()
    cancelled = [requests[0], requests[20]]
    for r in cancelled:
        s.cancel(r.identifier)
    s.run(max_iterations=2000)
    assert not b.live and not s.queue and not s.active
    assert all(r.state is State.CANCELLED for r in cancelled)
    assert all(r.state is State.FINISHED for r in requests if r not in cancelled)


def test_backend_failure_releases_active_sequences():
    class Broken(Backend):
        def decode_batch(self, keys):
            raise RuntimeError("broken backend")

    b = Broken()
    s = Scheduler(b)
    r = Request("a", (1,), 2)
    s.submit(r)
    with pytest.raises(RuntimeError):
        s.step()
    assert r.state is State.FAILED and not b.live and not s.active


def test_admission_failure_keeps_explicit_terminal_state():
    class BrokenCreate(Backend):
        def create(self, key):
            raise RuntimeError("failed atomic create")

    s = Scheduler(BrokenCreate())
    r = Request("admission", (1,), 1)
    s.submit(r)
    with pytest.raises(RuntimeError):
        s.step()
    assert r.state is State.FAILED and not s.active and not s.queue


def test_native_resume_matches_uninterrupted_generation():
    from pathlib import Path
    from tvara.native import NativeModel
    from tvara.scheduler import NativeSchedulerBackend

    path = Path("models/api-fixture/model.gguf")
    if not path.exists():
        pytest.skip("Local native random fixture unavailable")
    with NativeModel(path) as model:
        expected = model.generate([1, 2, 3], max_new_tokens=5)
        s = Scheduler(NativeSchedulerBackend(model), prefill_chunk=2)
        r = Request("resume", (1, 2, 3), 5)
        s.submit(r)
        while not r.output:
            s.step()
        s.preempt(r.identifier)
        s.resume(r.identifier)
        s.run()
        assert r.output == expected
