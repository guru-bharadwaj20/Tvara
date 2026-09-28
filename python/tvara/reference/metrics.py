"""Numerical agreement metrics with explicit precision thresholds."""

import numpy as np

FLOAT32_ATOL = 1e-5
FLOAT32_RTOL = 1e-4


def compare_logits(actual, expected, atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 2 or not actual.size:
        raise ValueError("Logit arrays must have identical nonempty [positions,vocabulary] shapes")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Logits must be finite")
    return {
        "max_absolute_error": float(np.max(np.abs(actual - expected))),
        "top1_agreement": float(np.mean(actual.argmax(-1) == expected.argmax(-1))),
        "within_tolerance": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
    }
