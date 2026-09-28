import numpy as np
import pytest
from tvara.reference.metrics import compare_logits


def test_comparison_reports_disagreement_and_nonfinite_values():
    a = np.array([[0.0, 1.0, 2.0]])
    assert compare_logits(a, a)["within_tolerance"]
    b = a[:, ::-1]
    report = compare_logits(a, b)
    assert report["top1_agreement"] == 0 and report["max_absolute_error"] == 2
    with pytest.raises(ValueError):
        compare_logits(a, np.array([[np.nan, 1.0, 2.0]]))


def test_perplexity_uses_next_token_targets_and_stable_logsumexp():
    from tvara.reference.metrics import perplexity

    assert perplexity(np.zeros((4, 8)), [0, 1, 2, 3]) == pytest.approx(8.0)
    logits = np.array([[1000.0, 1001.0], [1001.0, 1000.0], [0.0, 0.0]])
    expected = 1 + np.exp(-1)
    assert perplexity(logits, [0, 1, 0]) == pytest.approx(expected)
    with pytest.raises(ValueError):
        perplexity(np.zeros((1, 8)), [0])
