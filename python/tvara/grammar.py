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
