# Grammar syntax

Rules use name ::= expression (GBNF) or name = expression ; (EBNF). A newline or
semicolon ends a top-level rule. Double/single-quoted literals, character classes,
inclusive ranges, negated classes, rule references, alternatives, grouped expressions,
and postfix ?, *, + are supported. Comments start with # outside literals/classes.
The default start rule is root. Parenthesized groups may span lines.

Terminals consume Unicode characters. Byte-level vocabulary handling will preserve
partial UTF-8 sequences separately. Left recursion and nullable repetition cycles
are rejected by this initial pushdown implementation. Right-recursive nesting is
supported within explicit stack/state limits. Unsupported syntax fails at compilation.
