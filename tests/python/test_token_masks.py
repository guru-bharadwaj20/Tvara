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


def test_split_utf8_and_malformed_sequences():
    g = grammar('"€"')
    e = m.MaskEngine([b"\xe2", b"\x82", b"\xac", b"\xe2\x82\xac", b"\xff", b""], g, eos_id=5)
    state = m.ByteState(GrammarState.start(g))
    assert e.mask(state).ids() == (0, 3)
    state = m.advance_byte(g, state, 0xE2)
    assert state.pending == b"\xe2"
    assert e.mask(state).ids() == (1,)
    state = m.advance_byte(g, state, 0x82)
    assert e.mask(state).ids() == (2,)
    state = m.advance_byte(g, state, 0xAC)
    assert e.mask(state).ids() == (5,)
    assert m.advance_byte(g, m.ByteState(GrammarState.start(g)), 0xC0) is None
    ascii_g = grammar('"a"')
    assert m.advance_byte(ascii_g, m.ByteState(GrammarState.start(ascii_g)), 0xE2) is None
    done_g = grammar('""')
    assert m.advance_byte(done_g, m.ByteState(GrammarState.start(done_g)), 0xE2) is None


def test_surrogate_prefixes_are_pruned_instead_of_raising():
    g = grammar("[^a]*")
    start = m.ByteState(GrammarState.start(g))
    lead = m.advance_byte(g, start, 0xED)
    assert lead.pending == bytes([0xED])
    assert m.advance_byte(g, lead, 0x9F).pending == bytes([0xED, 0x9F])
    for byte in (0xA0, 0xBF):
        assert m.advance_byte(g, lead, byte) is None
    vocabulary = [bytes([b]) for b in range(256)]
    allowed = m.MaskEngine(vocabulary, g).mask(lead).ids()
    assert allowed == tuple(range(0x80, 0xA0))


def test_constraint_dominates_logits_and_preserves_transactional_state():
    c = m.TokenConstraint(m.MaskEngine([b"a", b"b", b""], grammar('"a"'), eos_id=2))
    assert int(c.mask_logits(np.array([1.0, 999.0, 5.0])).argmax()) == 0
    before = c.state
    with pytest.raises(ValueError):
        c.accept(1)
    assert c.state == before
    c.accept(0)
    assert c.complete
    c.accept(2)
    assert c.ended
    with pytest.raises(ValueError):
        c.accept(2)


def test_schema_request_constraints_validate_real_json():
    from tvara.response_constraints import response_constraint
    from tvara.tokenizer import ByteBPE, byte_mapping
    from jsonschema import validate
    import json

    forward, _ = byte_mapping()
    tok = ByteBPE(list(forward.values()), [])
    schema = {
        "type": "object",
        "properties": {"ok": {"const": True}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    c = response_constraint({"type": "json_schema", "json_schema": {"schema": schema}}, tok)
    text = '{"ok":true}'
    for token in tok.encode(text):
        assert c.engine.mask(c.state).allows(token)
        c.accept(token)
    assert c.complete
    validate(json.loads(text), schema)
    with pytest.raises(ValueError):
        response_constraint({"type": "json_object"}, tok)


def test_gguf_chat_template_roles_and_sandbox():
    pytest.importorskip("jinja2")
    from tvara.chat_templates import render_chat_template

    class Tok:
        vocabulary = ("BOS", "EOS")

    meta = {
        "tokenizer.chat_template": "{{bos_token}}{% for m in messages %}{{m.role}}:{{m.content}};{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
        "tokenizer.ggml.bos_token_id": 0,
    }
    assert (
        render_chat_template(meta, [{"role": "user", "content": "Hi"}], Tok())
        == "BOSuser:Hi;assistant:"
    )
    meta["tokenizer.chat_template"] = "{{ messages.__class__.__mro__ }}"
    with pytest.raises(ValueError):
        render_chat_template(meta, [{"role": "user", "content": "Hi"}], Tok())


def test_tool_signatures_become_name_and_argument_constraints():
    from tvara.response_constraints import tool_response_format
    from tvara.schema import compile_schema
    from tvara.grammar import accepts

    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"enum": ["Paris"]}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    spec = tool_response_format(tools, "required")
    g = compile_schema(spec["json_schema"]["schema"])
    assert accepts(g, '{"name":"weather","arguments":{"city":"Paris"}}')
    assert not accepts(g, '{"name":"other","arguments":{"city":"Paris"}}')
    with pytest.raises(ValueError):
        tool_response_format(tools, "auto")


def test_seeded_trie_masks_match_naive_scan_after_unicode_transitions():
    rng = np.random.default_rng(225)
    vocab = [
        bytes(rng.integers(0, 256, size=int(rng.integers(1, 5)), dtype=np.uint8))
        for _ in range(200)
    ]
    vocab += [b"a", b"b", "€".encode(), b"\xe2", b"\x82", b"\xac"]
    g = grammar("[ab€]*")
    state = m.ByteState(GrammarState.start(g))
    trie = m.VocabularyTrie(vocab)
    for byte in "a€b".encode():
        assert m.trie_allowed(trie, g, state) == m.naive_allowed(vocab, g, state)
        state = m.advance_byte(g, state, byte)
        assert state is not None
