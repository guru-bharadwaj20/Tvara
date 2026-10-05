import numpy as np
import pytest
from tvara.quantized_kv import QuantizedKV


def test_round_trip_and_budget():
    rng = np.random.default_rng(251)
    keys = rng.normal(size=(17, 2, 32)).astype(np.float32)
    values = rng.normal(size=keys.shape).astype(np.float32)
    cache = QuantizedKV(2, 32, 17)
    for k, v in zip(keys, values, strict=True):
        cache.append(k, v)
    decoded_k, decoded_v = cache.decoded()
    assert np.all(np.abs(decoded_k - keys) <= cache.key_scales / 2 + 1e-6)
    assert np.all(np.abs(decoded_v - values) <= cache.value_scales / 2 + 1e-6)
    assert cache.active_bytes == 17 * 2 * (32 + 4) * 2
    assert cache.active_bytes < (keys.nbytes + values.nbytes) / 3
    with pytest.raises(ValueError):
        cache.append(keys[0], values[0])


def test_zero_invalid_and_atomic_append():
    cache = QuantizedKV(1, 4, 3)
    cache.append(np.zeros((1, 4)), np.zeros((1, 4)))
    np.testing.assert_array_equal(cache.decoded()[0], 0)
    with pytest.raises(ValueError):
        cache.append(np.ones((1, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError):
        cache.append(np.ones((1, 4)), np.full((1, 4), np.nan))
    assert cache.length == 1
