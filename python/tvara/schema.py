"""Compile a documented JSON Schema subset into valid-output CFGs."""

import json
from .grammar import Grammar, JSON_PRIMITIVES, literal, parse_grammar


def extra_rules(text):
    rules = parse_grammar(text)
    prefix = "__extra_object_" if "generic-object" in text else "__extra_array_"
    names = {name: prefix + name for name in rules if name.startswith("__generated_")}
    return {
        names.get(name, name): tuple(
            tuple(names.get(s, s) if isinstance(s, str) else s for s in p) for p in choices
        )
        for name, choices in rules.items()
    }


class SchemaCompiler:
    def __init__(self, schema):
        self.source = schema
        self.rules = parse_grammar(JSON_PRIMITIVES)
        self.counter = 0
        self.memo = {}
        self.rules.update(
            extra_rules(r"""generic-object ::= "{" ws (string ws ":" ws value ws ("," ws string ws ":" ws value ws)*)? "}"
""")
        )
        self.rules.update(
            extra_rules(r"""generic-array ::= "[" ws (value ws ("," ws value ws)*)? "]"
""")
        )
        self.rules["value"] += (("generic-object",), ("generic-array",))

    def new(self, choices=()):
        self.counter += 1
        name = f"schema_{self.counter}"
        self.rules[name] = tuple(choices)
        return name

    def compile(self):
        target = self.node(self.source)
        self.rules["root"] = (("ws", target, "ws"),)
        return Grammar(self.rules)

    def node(self, schema):
        if schema is True:
            return "value"
        if schema is False:
            raise ValueError("False schema has no valid output")
        if not isinstance(schema, dict):
            raise ValueError("Schema must be an object or boolean")
        if id(schema) in self.memo:
            return self.memo[id(schema)]
        name = self.new()
        self.memo[id(schema)] = name
        kind = schema.get(
            "type", "object" if "properties" in schema or "required" in schema else None
        )
        if kind == "object":
            choices = self.object(schema)
        elif kind == "array":
            choices = self.array(schema)
        elif kind == "string":
            choices = (("string",),)
        elif kind == "integer":
            choices = (literal("-") + ("integer",), ("integer",))
        elif kind == "number":
            choices = (("number",),)
        elif kind == "boolean":
            choices = (literal("true"), literal("false"))
        elif kind == "null":
            choices = (literal("null"),)
        elif kind is None:
            choices = (("value",),)
        else:
            raise ValueError("Unsupported schema type")
        self.rules[name] = tuple(choices)
        return name

    def object(self, schema):
        properties = dict(schema.get("properties", {}))
        required = set(schema.get("required", []))
        for name in required:
            properties.setdefault(name, {})
        if len(properties) > 10:
            raise ValueError("Object schema supports at most ten named properties")
        names = list(properties)
        children = [self.node(properties[n]) for n in names]
        required_mask = sum(1 << i for i, name in enumerate(names) if name in required)
        states = {mask: self.new() for mask in range(1 << len(names))}
        minimum = schema.get("minProperties", 0)
        maximum = schema.get("maxProperties", len(names))
        for mask, state in states.items():
            choices = []
            if mask & required_mask == required_mask and minimum <= mask.bit_count() <= maximum:
                choices.append(())
            if mask.bit_count() < maximum:
                for i, key in enumerate(names):
                    if mask & (1 << i):
                        continue
                    prefix = literal(",") + ("ws",) if mask else ()
                    choices.append(
                        prefix
                        + literal(json.dumps(key, ensure_ascii=True))
                        + ("ws",)
                        + literal(":")
                        + ("ws", children[i], "ws", states[mask | (1 << i)])
                    )
            self.rules[state] = tuple(choices)
        return (literal("{") + ("ws", states[0]) + literal("}"),)

    def array(self, schema):
        minimum = schema.get("minItems", 0)
        maximum = schema.get("maxItems")
        if (
            not isinstance(minimum, int)
            or minimum < 0
            or minimum > 64
            or maximum is not None
            and (not isinstance(maximum, int) or maximum < minimum or maximum > 64)
        ):
            raise ValueError("Array bounds must be 0..64")
        child = self.node(schema.get("items", {}))
        if maximum is not None:
            choices = []
            for length in range(minimum, maximum + 1):
                sequence = ()
                for i in range(length):
                    sequence += (literal(",") + ("ws",) if i else ()) + (child, "ws")
                choices.append(sequence)
            content = self.new(choices)
        else:
            tail = self.new()
            self.rules[tail] = ((), literal(",") + ("ws", child, "ws", tail))
            prefix = ()
            for i in range(max(1, minimum)):
                prefix += (literal(",") + ("ws",) if i else ()) + (child, "ws")
            content = self.new(((), prefix + (tail,)) if minimum == 0 else (prefix + (tail,),))
        return (literal("[") + ("ws", content) + literal("]"),)


def compile_schema(schema):
    return SchemaCompiler(schema).compile()
