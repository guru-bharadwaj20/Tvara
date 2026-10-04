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
