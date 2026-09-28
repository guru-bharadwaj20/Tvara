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


def perplexity(logits, tokens):
    logits = np.asarray(logits, dtype=np.float64)
    tokens = np.asarray(tokens)
    if logits.ndim != 2 or tokens.ndim != 1 or len(tokens) < 2 or logits.shape[0] != len(tokens):
        raise ValueError("Perplexity requires at least two aligned token positions")
    if tokens.dtype.kind not in "iu" or tokens.min() < 0 or tokens.max() >= logits.shape[1]:
        raise ValueError("Token IDs outside logit vocabulary")
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    scores = logits[:-1]
    maxima = scores.max(axis=-1)
    log_z = maxima + np.log(np.exp(scores - maxima[:, None]).sum(axis=-1))
    nll = log_z - scores[np.arange(len(tokens) - 1), tokens[1:]]
    return float(np.exp(nll.mean()))
