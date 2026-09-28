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


def attention(q, k, v):
    """Causal GQA over [sequence, heads, dim], with optional cached-prefix keys."""
    q, k, v = (np.asarray(a, dtype=np.float32) for a in (q, k, v))
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Attention expects matching rank-3 key/value arrays")
    if min(q.shape + k.shape) <= 0 or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
        raise ValueError("Invalid attention dimensions or GQA head ratio")
    if q.shape[0] > k.shape[0]:
        raise ValueError("Query sequence cannot exceed available keys")
    ratio = q.shape[1] // k.shape[1]
    keys = np.repeat(k, ratio, axis=1)
    values = np.repeat(v, ratio, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, keys) / np.float32(np.sqrt(q.shape[-1]))
    positions = np.arange(q.shape[0]) + k.shape[0] - q.shape[0]
    scores = np.where(
        np.arange(k.shape[0])[None, None, :] <= positions[None, :, None], scores, -np.inf
    )
    probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", probs, values).astype(np.float32)


def swiglu(gate, up):
    gate, up = (np.asarray(x, dtype=np.float32) for x in (gate, up))
    if gate.shape != up.shape:
        raise ValueError("SwiGLU branches must have identical shapes")
    small = np.exp(-np.abs(gate))
    sigmoid = np.where(gate >= 0, 1 / (1 + small), small / (1 + small))
    return (gate * sigmoid * up).astype(np.float32)
