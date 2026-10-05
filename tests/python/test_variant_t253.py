import numpy as np
import pytest
from tvara.moe import route, mixture


def test_ties_and_selected_probability():
    ids, p = route([[1, 1, 1], [1000, -1000, 999]], 2)
    np.testing.assert_array_equal(ids, [[0, 1], [0, 2]])
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_allclose(p[0], [0.5, 0.5])
    assert np.isfinite(p).all()
    with pytest.raises(ValueError):
        route([[0, 1]], 3)


def test_mixture_independent_scalar_reference():
    rng = np.random.default_rng(253)
    x = rng.normal(size=(7, 4)).astype(np.float32)
    router = rng.normal(size=(3, 4)).astype(np.float32)
    experts = [
        tuple(rng.normal(size=s).astype(np.float32) for s in ((8, 4), (8, 4), (4, 8)))
        for _ in range(3)
    ]
    output, selected, weights = mixture(x, router, experts, 2)
    expected = np.zeros_like(x)
    for token in range(7):
        for slot in range(2):
            gate, up, down = experts[selected[token, slot]]
            g = x[token] @ gate.T
            u = x[token] @ up.T
            expected[token] += ((g / (1 + np.exp(-g)) * u) @ down.T) * weights[token, slot]
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)
    with pytest.raises(ValueError):
        mixture(x, router, [(np.zeros((3, 3)),) * 3] * 3, 2)
