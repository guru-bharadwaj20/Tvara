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
