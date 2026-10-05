"""Greedy verification of a finite draft tree against a target callback.

Nodes are (parent_index, token); -1 denotes the existing prefix. Parents must
precede children. The verifier evaluates each accepted prefix once and returns
the longest accepted path plus the target's next token. This is a correctness
prototype, without a fused tree-mask or batched native target implementation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TreeResult:
    tokens: tuple[int, ...]
    accepted_draft: int
    target_calls: int


def verify_tree(prefix, nodes, target, vocabulary, max_nodes=1024, max_depth=64):
    if (
        type(vocabulary) is not int
        or vocabulary <= 0
        or type(max_nodes) is not int
        or max_nodes <= 0
        or type(max_depth) is not int
        or max_depth <= 0
        or len(nodes) > max_nodes
    ):
        raise ValueError("Invalid tree budget or vocabulary")
    prefix = tuple(prefix)
    if any(type(t) is not int or not 0 <= t < vocabulary for t in prefix):
        raise ValueError("Invalid target prefix")
    paths = {-1: ()}
    seen = set()
    for i, (parent, token) in enumerate(nodes):
        if (
            type(parent) is not int
            or not -1 <= parent < i
            or type(token) is not int
            or not 0 <= token < vocabulary
            or (parent, token) in seen
        ):
            raise ValueError("Malformed, cyclic or duplicated tree edge")
        seen.add((parent, token))
        paths[i] = paths[parent] + (token,)
        if len(paths[i]) > max_depth:
            raise ValueError("Tree depth limit exceeded")
    predictions = {}
    accepted = {-1}
    best = -1

    def prediction(parent):
        if parent not in predictions:
            logits = np.asarray(target(prefix + paths[parent]))
            if logits.shape != (vocabulary,) or not np.isfinite(logits).all():
                raise ValueError("Invalid target tree logits")
            predictions[parent] = int(np.argmax(logits))
        return predictions[parent]

    for i, (parent, token) in enumerate(nodes):
        if parent in accepted and token == prediction(parent):
            accepted.add(i)
            if len(paths[i]) > len(paths[best]):
                best = i
    result = paths[best] + (prediction(best),)
    return TreeResult(result, len(paths[best]), len(predictions))
