"""Deterministic top-k routed expert prototype; no model-format claims."""

import numpy as np
from .reference.math import swiglu


def route(logits, k):
    logits = np.asarray(logits, dtype=np.float32)
    if (
        logits.ndim != 2
        or not logits.shape[0]
        or not logits.shape[1]
        or not np.isfinite(logits).all()
        or type(k) is not int
        or not 1 <= k <= logits.shape[1]
    ):
        raise ValueError("Invalid expert router shape or top-k")
    experts = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
    selected = np.take_along_axis(logits, experts, axis=-1)
    probability = np.exp(selected - selected.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)
    return experts, probability.astype(np.float32)


def mixture(x, router, experts, k):
    x = np.asarray(x, dtype=np.float32)
    router = np.asarray(router, dtype=np.float32)
    if (
        x.ndim != 2
        or not x.shape[0]
        or not x.shape[1]
        or router.ndim != 2
        or router.shape[1] != x.shape[1]
        or router.shape[0] != len(experts)
        or not np.isfinite(x).all()
        or not np.isfinite(router).all()
    ):
        raise ValueError("Invalid mixture inputs")
    validated = []
    for gate, up, down in experts:
        gate, up, down = (np.asarray(a, dtype=np.float32) for a in (gate, up, down))
        if (
            gate.ndim != 2
            or not gate.shape[0]
            or gate.shape != up.shape
            or gate.shape[1] != x.shape[1]
            or down.shape != (x.shape[1], gate.shape[0])
            or not all(np.isfinite(a).all() for a in (gate, up, down))
        ):
            raise ValueError("Invalid expert weights")
        validated.append((gate, up, down))
    selected, probability = route(x @ router.T, k)
    output = np.zeros_like(x)
    for expert, (gate, up, down) in enumerate(validated):
        tokens, slots = np.where(selected == expert)
        if tokens.size:
            value = swiglu(x[tokens] @ gate.T, x[tokens] @ up.T) @ down.T
            output[tokens] += value * probability[tokens, slots, None]
    return output, selected, probability
