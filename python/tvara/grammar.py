"""Bounded grammar compiler and nondeterministic pushdown reference engine."""

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class CharSet:
    ranges: tuple[tuple[int, int], ...]
    negate: bool = False

    def matches(self, char):
        inside = any(a <= ord(char) <= b for a, b in self.ranges)
        return not inside if self.negate else inside


def _escape(text, index):
    char = text[index]
    if char != "\\":
        return char, index + 1
    if index + 1 >= len(text):
        raise ValueError("Truncated character escape")
    code = text[index + 1]
    if code in "ux":
        width = 4 if code == "u" else 2
        try:
            value = chr(int(text[index + 2 : index + 2 + width], 16))
        except ValueError as error:
            raise ValueError("Invalid character escape") from error
        if len(text[index + 2 : index + 2 + width]) != width:
            raise ValueError("Truncated hex escape")
        return value, index + 2 + width
    return {"n": "\n", "r": "\r", "t": "\t"}.get(code, code), index + 2


def _character_class(text, index):
    index += 1
    negate = index < len(text) and text[index] == "^"
    if negate:
        index += 1
    ranges = []
    while index < len(text) and text[index] != "]":
        first, index = _escape(text, index)
        if index + 1 < len(text) and text[index] == "-" and text[index + 1] != "]":
            last, index = _escape(text, index + 1)
        else:
            last = first
        if ord(first) > ord(last):
            raise ValueError("Descending character range")
        ranges.append((ord(first), ord(last)))
    if index >= len(text) or not ranges:
        raise ValueError("Unterminated or empty character class")
    return CharSet(tuple(ranges), negate), index + 1


def lex(text):
    tokens = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in " \t\r":
            i += 1
            continue
        if char == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        if char == "\n":
            tokens.append(("NL", char))
            i += 1
            continue
        if text.startswith("::=", i):
            tokens.append(("ASSIGN", "::="))
            i += 3
            continue
        if char == "=":
            tokens.append(("ASSIGN", "="))
            i += 1
            continue
        if char in "|()*+?;":
            tokens.append((char, char))
            i += 1
            continue
        if char in "\"'":
            start = i
            quote = char
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    break
                i += 1
            if i >= len(text):
                raise ValueError("Unterminated grammar literal")
            i += 1
            try:
                value = ast.literal_eval(text[start:i])
            except (ValueError, SyntaxError) as error:
                raise ValueError("Invalid grammar literal") from error
            tokens.append(("LITERAL", value))
            continue
        if char == "[":
            value, i = _character_class(text, i)
            tokens.append(("CLASS", value))
            continue
        if char.isascii() and (char.isalpha() or char == "_"):
            start = i
            i += 1
            while i < len(text) and text[i].isascii() and (text[i].isalnum() or text[i] in "_-"):
                i += 1
            tokens.append(("NAME", text[start:i]))
            continue
        raise ValueError(f"Unsupported grammar character at offset {i}")
    return tokens + [("EOF", "")]


def literal(text):
    return tuple(CharSet(((ord(c), ord(c)),)) for c in text)


class Parser:
    def __init__(self, text):
        self.tokens = lex(text)
        self.position = 0
        self.rules = {}
        self.generated = 0

    def kind(self):
        return self.tokens[self.position][0]

    def take(self, kind):
        if self.kind() != kind:
            raise ValueError(f"Expected {kind}, received {self.kind()}")
        value = self.tokens[self.position][1]
        self.position += 1
        return value

    def define(self, productions):
        self.generated += 1
        name = f"__generated_{self.generated}"
        self.rules[name] = tuple(productions)
        return name

    def factor(self):
        kind = self.kind()
        if kind == "NAME":
            symbols = (self.take("NAME"),)
        elif kind == "LITERAL":
            symbols = literal(self.take("LITERAL"))
        elif kind == "CLASS":
            symbols = (self.take("CLASS"),)
        elif kind == "(":
            self.take("(")
            choices = self.expression(group=True)
            self.take(")")
            symbols = (self.define(choices),)
        else:
            raise ValueError("Expected a grammar symbol")
        if self.kind() in ("?", "*", "+"):
            operator = self.kind()
            self.take(operator)
            self.generated += 1
            name = f"__generated_{self.generated}"
            if operator == "?":
                choices = ((), symbols)
            elif operator == "*":
                choices = ((), symbols + (name,))
            else:
                choices = (symbols, symbols + (name,))
            self.rules[name] = choices
            symbols = (name,)
        return symbols

    def expression(self, group=False):
        choices = []
        sequence = ()
        while True:
            if group and self.kind() == "NL":
                self.take("NL")
                continue
            if self.kind() == "|":
                self.take("|")
                choices.append(sequence)
                sequence = ()
                continue
            if self.kind() in ("NAME", "LITERAL", "CLASS", "("):
                sequence += self.factor()
                continue
            break
        choices.append(sequence)
        return tuple(choices)

    def parse(self):
        while self.kind() != "EOF":
            if self.kind() in ("NL", ";"):
                self.position += 1
                continue
            name = self.take("NAME")
            self.take("ASSIGN")
            if name in self.rules:
                raise ValueError("Duplicate grammar rule")
            self.rules[name] = self.expression()
            if self.kind() not in ("EOF", "NL", ";"):
                raise ValueError("Expected rule separator")
        return self.rules


def parse_grammar(text):
    return Parser(text).parse()


@dataclass(frozen=True)
class Grammar:
    rules: dict
    root: str = "root"

    def __post_init__(self):
        from types import MappingProxyType

        copied = {name: tuple(tuple(p) for p in choices) for name, choices in self.rules.items()}
        object.__setattr__(self, "rules", MappingProxyType(copied))
        validate_grammar(self)

    @classmethod
    def from_text(cls, text, root="root"):
        return cls(parse_grammar(text), root)

    def with_root(self, root):
        return Grammar(dict(self.rules), root)


def validate_grammar(grammar):
    if grammar.root not in grammar.rules or len(grammar.rules) > 16384:
        raise ValueError("Missing root or excessive grammar rules")
    for choices in grammar.rules.values():
        for production in choices:
            for symbol in production:
                if (
                    not isinstance(symbol, (str, CharSet))
                    or isinstance(symbol, str)
                    and symbol not in grammar.rules
                ):
                    raise ValueError("Undefined or invalid grammar symbol")
    nullable = set()
    changed = True
    while changed:
        changed = False
        for name, choices in grammar.rules.items():
            if name not in nullable and any(
                all(isinstance(s, str) and s in nullable for s in p) for p in choices
            ):
                nullable.add(name)
                changed = True
    edges = {name: set() for name in grammar.rules}
    for name, choices in grammar.rules.items():
        for production in choices:
            for symbol in production:
                if isinstance(symbol, CharSet):
                    break
                edges[name].add(symbol)
                if symbol not in nullable:
                    break
    active = set()
    done = set()

    def visit(name):
        if name in active:
            raise ValueError("Left recursion or nullable expansion cycle")
        if name in done:
            return
        active.add(name)
        for child in edges[name]:
            visit(child)
        active.remove(name)
        done.add(name)

    for name in edges:
        visit(name)


class GrammarLimitError(ValueError):
    pass


def closure(grammar, stacks, max_depth=256, max_states=8192):
    pending = list(stacks)
    seen = set()
    frontier = set()
    while pending:
        stack = pending.pop()
        if stack in seen:
            continue
        seen.add(stack)
        if len(stack) > max_depth or len(seen) > max_states:
            raise GrammarLimitError("Grammar state/stack limit exceeded")
        if not stack or isinstance(stack[0], CharSet):
            frontier.add(stack)
            continue
        for production in grammar.rules[stack[0]]:
            pending.append(production + stack[1:])
    return frozenset(frontier)


@dataclass(frozen=True)
class GrammarState:
    stacks: frozenset

    @classmethod
    def start(cls, grammar):
        return cls(frozenset({(grammar.root,)}))

    def advance(self, grammar, char):
        if len(char) != 1:
            raise ValueError("Grammar transitions consume one character")
        remaining = frozenset(
            stack[1:] for stack in closure(grammar, self.stacks) if stack and stack[0].matches(char)
        )
        return GrammarState(remaining) if remaining else None

    def accepting(self, grammar):
        return () in closure(grammar, self.stacks)


def accepts(grammar, text):
    state = GrammarState.start(grammar)
    for char in text:
        state = state.advance(grammar, char)
        if state is None:
            return False
    return state.accepting(grammar)


JSON_PRIMITIVES = r"""
root ::= ws value ws
ws ::= [ \t\r\n]*
value ::= string | number | "true" | "false" | "null"
string ::= "\"" chars "\""
chars ::= char chars | ""
char ::= [^"\\\x00-\x1f\ud800-\udfff] | "\\" escape
escape ::= ["\\/bfnrt] | "u" unicode-safe
unicode-safe ::= [0-9a-cA-Ce-fE-F] hex hex hex | [dD] [0-7] hex hex
hex ::= [0-9a-fA-F]
number ::= "-"? integer fraction? exponent?
integer ::= "0" | [1-9] [0-9]*
fraction ::= "." [0-9]+
exponent ::= [eE] [+-]? [0-9]+
"""


def json_primitive_grammar():
    return Grammar.from_text(JSON_PRIMITIVES)
