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
