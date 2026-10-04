from tvara.scheduler import Request, State
import pytest


def test_lifecycle_contract():
    assert Request("a", (1,), 2).state is State.QUEUED
    for args in [("", (1,), 2), ("a", (), 2), ("a", (-1,), 2), ("a", (1,), -1)]:
        with pytest.raises(ValueError):
            Request(*args)
