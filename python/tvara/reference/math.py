"""Portable float32 numerical reference primitives."""

import numpy as np


def rms_norm(x, weight, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if x.ndim < 1 or x.shape[-1] == 0 or weight.shape != (x.shape[-1],):
        raise ValueError("RMSNorm expects a nonempty last dimension and matching weight")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm epsilon must be positive and finite")
    variance = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x / np.sqrt(variance + np.float32(eps)) * weight).astype(np.float32)
