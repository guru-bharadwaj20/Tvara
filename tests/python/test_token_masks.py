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


def test_trie_grammar_traversal():
    g = grammar('"ab" | "ac"')
    state = m.ByteState(GrammarState.start(g))
    assert m.trie_allowed(m.VocabularyTrie([b"a", b"ab", b"ac", b"ad", b"b"]), g, state) == (
        0,
        1,
        2,
    )


def test_invalid_subtree_is_never_visited():
    g = grammar('"abc"')
    state = m.ByteState(GrammarState.start(g))
    stats = {}
    trie = m.VocabularyTrie([b"abc", b"axxxxxxxxx", b"yyyyyyyy"])
    assert m.trie_allowed(trie, g, state, stats) == (0,)
    assert stats["visited"] == 4 and stats["pruned"] == 2
    assert stats["visited"] < trie.node_count


def test_bitmask_boundary_and_invalid_ids():
    mask = m.TokenMask.from_ids(66, [0, 7, 8, 65])
    assert mask.ids() == (0, 7, 8, 65) and len(mask.packed()) == 9
    assert mask.packed()[0] == 129 and mask.packed()[-1] == 2
    with pytest.raises(ValueError):
        m.TokenMask.from_ids(3, [3])


def test_state_classification_is_conservative():
    g = grammar('"ab"')
    s = m.ByteState(GrammarState.start(g))
    assert m.state_kind(g, s) == "terminal-only"
    assert m.state_kind(g, m.ByteState(s.grammar_state, b"\xe2")) == "utf8-dependent"
    recursive = Grammar.from_text('root ::= "(" root ")" | "x"')
    assert m.state_kind(recursive, m.ByteState(GrammarState.start(recursive))) == "stack-dependent"


def test_precomputation_bound_and_eos():
    g = grammar('"a"')
    s = m.ByteState(GrammarState.start(g))
    e = m.MaskEngine([b"a", b""], g, eos_id=1, max_entries=1)
    e.precompute([s])
    assert e.mask(s).ids() == (0,)
    done = m.advance_byte(g, s, 97)
    assert e.compute(done).ids() == (1,)
    with pytest.raises(ValueError):
        e.precompute([done])


def test_lru_eviction_and_cache_identity():
    g = grammar('"ab"')
    s = m.ByteState(GrammarState.start(g))
    e = m.MaskEngine([b"a", b"b"], g, max_entries=1)
    first = e.mask(s)
    assert e.mask(s) is first and e.hits == 1
    after = m.advance_byte(g, s, 97)
    assert e.mask(after).ids() == (1,)
    assert len(e.cache) == 1 and s not in e.cache


def test_recursive_stack_masks_match_independent_vocabulary_scan():
    g = Grammar.from_text('root ::= "(" root ")" | "x"')
    vocab = [b"(", b"x", b")", b"(x)", b"xx", b"))"]
    e = m.MaskEngine(vocab, g)
    state = m.ByteState(GrammarState.start(g))
    for byte in b"((x))":
        assert e.mask(state).ids() == m.naive_allowed(vocab, g, state)
        state = m.advance_byte(g, state, byte)
    assert state.grammar_state.accepting(g)
    assert e.mask(state).ids() == ()
