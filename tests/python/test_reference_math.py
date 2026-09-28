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


def test_grouped_attention_matches_scalar_and_is_causal():
    from tvara.reference.math import attention

    rng = np.random.default_rng(9)
    q = rng.normal(size=(3, 4, 2)).astype(np.float32)
    k = rng.normal(size=(3, 2, 2)).astype(np.float32)
    v = rng.normal(size=(3, 2, 2)).astype(np.float32)
    expected = np.zeros_like(q)
    for t in range(3):
        for h in range(4):
            scores = np.array(
                [np.dot(q[t, h].astype(float), k[j, h // 2]) / np.sqrt(2) for j in range(t + 1)]
            )
            prob = np.exp(scores - scores.max())
            prob /= prob.sum()
            expected[t, h] = sum(prob[j] * v[j, h // 2] for j in range(t + 1))
    np.testing.assert_allclose(attention(q, k, v), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(attention(q[-1:], k, v), expected[-1:], rtol=1e-5, atol=1e-6)
    v[2] *= 100
    np.testing.assert_allclose(attention(q, k, v)[0], expected[0], rtol=1e-5, atol=1e-6)


def test_swiglu_matches_direct_expression_and_extreme_values():
    from tvara.reference.math import swiglu

    g = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    u = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(swiglu(g, u), g / (1 + np.exp(-g)) * u, rtol=1e-6)
    assert np.isfinite(swiglu(np.array([-1000.0, 1000.0]), np.ones(2))).all()
    with pytest.raises(ValueError):
        swiglu(np.ones(2), np.ones(3))
