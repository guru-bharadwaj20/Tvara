import numpy as np
from tvara.speculative import TokenizerIdentity, probabilities


class ToyModel:
    identity = TokenizerIdentity((b"a", b"b", b"c"), "fixture-v1")

    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)
        self.history = [0]

    def distribution(self):
        return self.p.copy()

    def advance(self, token):
        if not 0 <= token < len(self.p):
            raise ValueError("Invalid token")
        self.history.append(token)

    def checkpoint(self):
        return len(self.history)

    def rollback(self, checkpoint):
        if not 1 <= checkpoint <= len(self.history):
            raise ValueError("Invalid checkpoint")
        del self.history[checkpoint:]


def test_protocol_distribution_and_owned_state():
    model = ToyModel([2, 3, 5])
    np.testing.assert_allclose(probabilities(model.distribution()), [0.2, 0.3, 0.5])
    position = model.checkpoint()
    model.advance(1)
    model.rollback(position)
    assert model.history == [0]
