"""Scalar sampler inputs must be validated before allocating native KV."""

import numpy as np
import pytest
from tvara.native import NativeModel


class FakeNative:
    vocab_size = 3
    context_size = 16

    def __init__(self):
        self.allocations = 0
        self.drops = 0

    def _new_sequence(self):
        self.allocations += 1
        return 1

    def _drop_sequence(self, sequence):
        self.drops += 1

    def logits(self, sequence, token):
        return np.array([1, 2, 3], dtype=np.float32)


@pytest.mark.parametrize(
    "field", ["temperature", "top_p", "min_p", "repetition", "frequency", "presence"]
)
@pytest.mark.parametrize("value", [np.array([1.0]), complex(0, 1), True, "1"])
def test_sampler_rejects_non_scalar_and_nonreal_controls_before_allocation(field, value):
    model = FakeNative()
    with pytest.raises(ValueError, match="real scalars"):
        list(NativeModel.iter_tokens(model, [0], max_new_tokens=1, **{field: value}))
    assert model.allocations == model.drops == 0


@pytest.mark.parametrize(
    "options",
    [{"top_k": True}, {"max_new_tokens": True}, {"seed": True}, {"seed": -1}, {"seed": 1.5}],
)
def test_sampler_rejects_boolean_counts_and_invalid_seeds(options):
    model = FakeNative()
    options = {"max_new_tokens": 1} | options
    with pytest.raises(ValueError):
        list(NativeModel.iter_tokens(model, [0], **options))
    assert model.allocations == model.drops == 0


@pytest.mark.parametrize("token", [True, np.bool_(False)])
def test_sampler_rejects_boolean_prompt_ids(token):
    model = FakeNative()
    with pytest.raises(ValueError, match="prompt token"):
        list(NativeModel.iter_tokens(model, [token], max_new_tokens=1))
    assert model.allocations == 0


def test_numpy_real_scalar_controls_remain_supported_and_reproducible():
    model = FakeNative()
    first = list(
        NativeModel.iter_tokens(
            model, [np.int64(0)], max_new_tokens=2, temperature=np.float64(1), seed=np.int64(123)
        )
    )
    second = list(
        NativeModel.iter_tokens(
            model, [np.int64(0)], max_new_tokens=2, temperature=np.float64(1), seed=np.int64(123)
        )
    )
    assert first == second and model.drops == 2


def test_direct_native_logits_rejects_boolean_before_buffer_creation():
    model = FakeNative()
    with pytest.raises(ValueError, match="token ID"):
        NativeModel.logits(model, 1, True)
