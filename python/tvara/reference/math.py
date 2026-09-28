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


def rope(x, positions, theta=10000.0):
    """Rotate [sequence, heads, head_dim] with Llama's split-half layout."""
    x = np.asarray(x, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] == 0 or x.shape[-1] % 2:
        raise ValueError("RoPE requires a positive even head dimension")
    if (
        positions.shape != (x.shape[0],)
        or not np.isfinite(positions).all()
        or (positions < 0).any()
    ):
        raise ValueError("RoPE positions must be nonnegative and match the sequence")
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("RoPE theta must be positive and finite")
    half = x.shape[-1] // 2
    inverse = 1.0 / np.float32(theta) ** (np.arange(half, dtype=np.float32) * 2 / x.shape[-1])
    angles = positions[:, None] * inverse[None, :]
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1).astype(np.float32)
