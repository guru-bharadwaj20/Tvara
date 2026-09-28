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
