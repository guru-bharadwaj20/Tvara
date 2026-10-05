import numpy as np
import pytest
from tvara.speculative_tree import verify_tree


def target(prefix):
    logits = np.full(11, -10.0)
    logits[(sum(prefix) + 3) % 11] = 10
    return logits


def test_longest_verified_branch_matches_target_greedy():
    prefix = (2,)
    nodes = [(-1, 4), (-1, 5), (1, 10), (1, 9), (2, 9), (2, 8)]
    result = verify_tree(prefix, nodes, target, 11)
    expected = []
    for _ in result.tokens:
        expected.append(int(np.argmax(target(prefix + tuple(expected)))))
    assert result.tokens == tuple(expected)
    assert result.accepted_draft == 3 and result.tokens == (5, 10, 9, 7)
    assert result.target_calls == 4


def test_empty_tree_and_invalid_edges():
    assert verify_tree((2,), [], target, 11).tokens == (5,)
    for nodes in [[(0, 1)], [(-1, 1), (-1, 1)], [(-1, 11)]]:
        with pytest.raises(ValueError):
            verify_tree((2,), nodes, target, 11)
    with pytest.raises(ValueError):
        verify_tree((2,), [], lambda _: np.zeros(3), 11)
