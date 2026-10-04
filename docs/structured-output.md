# Structured-output compiler

The grammar compiler accepts the syntax in grammar-syntax.md and executes bounded
nondeterministic pushdown states over Unicode characters. Undefined rules, left
recursion, and nullable expansion cycles fail before generation. Runtime limits are
256 stack symbols and 8,192 closure states; reaching them is an error, not acceptance.

JSON Schema support includes objects with required/optional properties in any order,
nested arrays/objects, bounds, enum/const, type alternatives, anyOf, and recursive
local references. Objects support at most ten named properties and emit each once.
Additional properties may be permitted by the schema but are not generated. Array
and string bounds are limited to 64. Unbounded repetition is supported when no upper
bound is supplied. Enum values are filtered through an independent schema validator.

Email and UUID formats currently produce fixed valid exemplars rather than arbitrary
formatted strings. Other formats, numeric range/pattern constraints, uniqueItems,
external references, and anyOf/reference sibling constraints are rejected. Primitive
strings exclude surrogate code points and surrogate escapes; direct valid UTF-8
characters and safe JSON escapes are supported. This is a valid-output subset, not
a claim to implement all JSON Schema expressiveness.

Acceptance tests cover nesting, repetitions, malformed JSON primitives, required
properties, reordered keys, array/string bounds, local references, and independent
jsonschema validation. GrammarMachine allows EOS only in an accepting state and
keeps the old state when a text fragment is invalid. Vocabulary masks and byte-level
streaming integration are separate later tasks.
