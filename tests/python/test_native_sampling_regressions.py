"""Regression tests for invalid sampler arithmetic before token emission."""

import numpy as np
import pytest
from tvara.native import NativeModel


class FakeNative:
    vocab_size = 3
    context_size = 16

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.drops = 0

    def _new_sequence(self):
        return 1

    def _drop_sequence(self, sequence):
        self.drops += 1

    def logits(self, sequence, token):
        return self.values


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("temperature", [0, 1])
def test_native_sampler_rejects_nonfinite_logits_and_releases_sequence(bad, temperature):
    model = FakeNative([1, bad, 2])
    with pytest.raises(ValueError, match="Nonfinite model logits"):
        list(NativeModel.iter_tokens(model, [0], max_new_tokens=1, temperature=temperature))
    assert model.drops == 1


@pytest.mark.parametrize("temperature", [0, 1])
def test_native_sampler_rejects_finite_penalty_overflow(temperature):
    model = FakeNative([1, 2, 3])
    with pytest.raises(ValueError, match="Nonfinite adjusted"):
        list(
            NativeModel.iter_tokens(
                model, [0, 0], max_new_tokens=1, temperature=temperature, frequency=1e308
            )
        )
    assert model.drops == 1


@pytest.mark.parametrize("eos", [1.5, True])
def test_native_sampler_rejects_noninteger_eos_before_allocation(eos):
    model = FakeNative([1, 2, 3])
    with pytest.raises(ValueError, match="Invalid EOS"):
        list(NativeModel.iter_tokens(model, [0], max_new_tokens=1, eos_id=eos))
    assert model.drops == 0


def test_float64_penalties_preserve_large_finite_representable_values():
    model = FakeNative([1, 2, 3])
    assert list(NativeModel.iter_tokens(model, [0, 0], max_new_tokens=1, frequency=1e38)) == [2]
    assert model.drops == 1
