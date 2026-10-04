import pytest
from tvara.grammar import lex, CharSet


def test_grammar_lexer_handles_escaped_literals_ranges_and_comments():
    tokens = lex('root ::= "a" [0-9]+ # comment\n')
    assert tokens[0] == ("NAME", "root") and tokens[2] == ("LITERAL", "a")
    charclass = tokens[3][1]
    assert isinstance(charclass, CharSet) and charclass.matches("5") and not charclass.matches("x")
    with pytest.raises(ValueError):
        lex('root ::= "unfinished')


def test_parser_desugars_groups_and_repetition():
    from tvara.grammar import parse_grammar

    rules = parse_grammar('root = ("a" | "b")+ "c"? ;')
    assert "root" in rules and len(rules) > 1
    with pytest.raises(ValueError):
        parse_grammar('root = ("a";')


def test_grammar_owns_immutable_production_sequences():
    from tvara.grammar import Grammar

    grammar = Grammar.from_text('root ::= "a"')
    with pytest.raises(TypeError):
        grammar.rules["root"] = ()
    assert grammar.root == "root"


def test_validation_rejects_missing_and_left_recursive_rules():
    from tvara.grammar import Grammar

    for text in ["root ::= missing", 'root ::= root "a" | ""', 'root ::= ("")*']:
        with pytest.raises(ValueError):
            Grammar.from_text(text)
    Grammar.from_text('root ::= "(" root ")" | ""')


def test_pushdown_nesting_and_repetition_acceptance():
    from tvara.grammar import Grammar, accepts

    grammar = Grammar.from_text('root ::= "(" root ")" root | ""')
    for text in ["", "()", "(())()"]:
        assert accepts(grammar, text)
    for text in ["(", ")(", "(()"]:
        assert not accepts(grammar, text)
    repeated = Grammar.from_text('root ::= [a-c]+ "!"?')
    assert accepts(repeated, "abc!") and not accepts(repeated, "")


def test_json_primitives_reject_invalid_numbers_and_escapes():
    from tvara.grammar import json_primitive_grammar, accepts

    grammar = json_primitive_grammar()
    for text in ["null", " true ", "-1.2e+3", '"hello"', '"😀"', r'"a\nb"']:
        assert accepts(grammar, text), repr(text)
    for text in ["01", "+1", "NaN", '"unterminated', r'"\x00"', r'"\ud800"']:
        assert not accepts(grammar, text), repr(text)


def test_object_schema_required_optional_and_property_orders():
    from tvara.schema import compile_schema
    from tvara.grammar import accepts

    grammar = compile_schema(
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}},
            "required": ["a"],
            "additionalProperties": False,
        }
    )
    for text in ['{"a":1}', '{"b":true,"a":-2}', ' { "a": 3, "b": false } ']:
        assert accepts(grammar, text)
    for text in ["{}", '{"b":true}', '{"a":"x"}', '{"a":1,"a":2}']:
        assert not accepts(grammar, text)


def test_bounded_nested_array_schema():
    from tvara.schema import compile_schema
    from tvara.grammar import accepts

    grammar = compile_schema(
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            "minItems": 1,
            "maxItems": 2,
        }
    )
    assert accepts(grammar, '[{"x":1},{"x":2}]')
    assert not accepts(grammar, "[]") and not accepts(grammar, '[{"x":1},{"x":2},{"x":3}]')


def test_enum_and_anyof_restrict_to_valid_literals():
    from tvara.schema import compile_schema
    from tvara.grammar import accepts

    grammar = compile_schema({"anyOf": [{"enum": ["red", "blue"]}, {"type": "integer"}]})
    assert accepts(grammar, '"red"') and accepts(grammar, "42")
    assert not accepts(grammar, '"green"') and not accepts(grammar, "false")


def test_nested_local_refs_and_recursive_null_termination():
    from tvara.schema import compile_schema
    from tvara.grammar import accepts

    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                    "next": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]},
                },
                "required": ["value", "next"],
            }
        },
        "$ref": "#/$defs/Node",
    }
    grammar = compile_schema(schema)
    assert accepts(grammar, '{"value":1,"next":{"value":2,"next":null}}')
    assert not accepts(grammar, '{"value":1,"next":true}')


def test_string_lengths_and_declared_format_subset():
    from tvara.schema import compile_schema
    from tvara.grammar import accepts

    bounded = compile_schema({"type": "string", "minLength": 2, "maxLength": 3})
    assert accepts(bounded, '"ab"') and accepts(bounded, '"😀é"')
    assert not accepts(bounded, '"a"') and not accepts(bounded, '"abcd"')
    email = compile_schema({"type": "string", "format": "email"})
    assert accepts(email, '"tvara@example.org"') and not accepts(email, '"invalid"')
    with pytest.raises(ValueError):
        compile_schema({"type": "string", "format": "date"})
