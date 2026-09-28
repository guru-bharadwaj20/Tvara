import numpy as np
import pytest
from tvara.reference.math import rms_norm


def test_rms_norm_matches_double_precision_scalar_calculation():
    x = np.array([[1, -2, 3], [0, 0, 0]], dtype=np.float32)
    w = np.array([0.5, 2, -1], dtype=np.float32)
    expected = np.array(
        [
            [
                float(a) * float(b) / (sum(float(t) ** 2 for t in row) / 3 + 1e-6) ** 0.5
                for a, b in zip(row, w)
            ]
            for row in x
        ]
    )
    np.testing.assert_allclose(rms_norm(x, w), expected, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError):
        rms_norm(x, np.ones(2))
    with pytest.raises(ValueError):
        rms_norm(x, w, 0)


def test_rope_zero_and_quarter_turn_preserve_norm():
    from tvara.reference.math import rope

    x = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    y = rope(x, [0, np.pi / 2])
    np.testing.assert_allclose(y[0], x[0])
    np.testing.assert_allclose(y[1], [[-4.0, 3.0]], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-6)
    with pytest.raises(ValueError):
        rope(np.zeros((2, 1, 3)), [0, 1])
