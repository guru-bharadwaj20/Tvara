import numpy as np
import pytest
from tvara.grammar import Grammar, GrammarState
from tvara import token_masks as m


def grammar(text):
    return Grammar.from_text("root ::= " + text)


def test_trie_shared_prefix_and_duplicates():
    trie = m.VocabularyTrie([b"a", b"ab", b"a", b""])
    assert trie.node_count == 3 and np.array(trie.vocabulary, dtype=object).shape == (4,)
    assert GrammarState.start(grammar('"a"')) is not None
    with pytest.raises(ValueError):
        m.VocabularyTrie([])
    assert trie.root.children[97].tokens == [0, 2]
    assert trie.root.children[97].children[98].tokens == [1]
