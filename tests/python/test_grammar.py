import pytest
from tvara.grammar import lex, CharSet


def test_grammar_lexer_handles_escaped_literals_ranges_and_comments():
    tokens = lex('root ::= "a" [0-9]+ # comment\n')
    assert tokens[0] == ("NAME", "root") and tokens[2] == ("LITERAL", "a")
    charclass = tokens[3][1]
    assert isinstance(charclass, CharSet) and charclass.matches("5") and not charclass.matches("x")
    with pytest.raises(ValueError):
        lex('root ::= "unfinished')
