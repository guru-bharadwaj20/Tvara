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
        self.retained = []
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
        self.retained.append(schema)
        name = self.new()
        self.memo[id(schema)] = name
        if "$ref" in schema:
            if set(schema) - {
                "$ref",
                "$defs",
                "definitions",
                "title",
                "description",
                "$schema",
                "$id",
            }:
                raise ValueError("Reference sibling constraints unsupported")
            target = self.resolve(schema["$ref"])
            self.rules[name] = ((self.node(target),),)
            return name
        if "enum" in schema or "const" in schema:
            from jsonschema import Draft202012Validator

            candidate_schema = dict(schema)
            for key in ("$defs", "definitions"):
                if key in self.source:
                    candidate_schema.setdefault(key, self.source[key])
            validator = Draft202012Validator(candidate_schema)
            values = schema.get("enum", [schema.get("const")])
            valid = [value for value in values if validator.is_valid(value)]
            if not valid:
                raise ValueError("Enum has no valid values")
            self.rules[name] = tuple(
                literal(json.dumps(v, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
                for v in valid
            )
            return name
        if "anyOf" in schema or isinstance(schema.get("type"), list):
            if "anyOf" in schema:
                outer = set(schema) - {
                    "anyOf",
                    "title",
                    "description",
                    "$defs",
                    "definitions",
                    "$schema",
                    "$id",
                }
                if outer:
                    raise ValueError("anyOf with sibling constraints is unsupported")
                alternatives = schema["anyOf"]
            else:
                alternatives = [{**schema, "type": kind} for kind in schema["type"]]
            self.rules[name] = tuple((self.node(item),) for item in alternatives)
            return name
        kind = schema.get(
            "type", "object" if "properties" in schema or "required" in schema else None
        )
        if kind == "object":
            choices = self.object(schema)
        elif kind == "array":
            choices = self.array(schema)
        elif kind == "string":
            choices = self.string(schema)
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

    def resolve(self, reference):
        from urllib.parse import unquote

        if not isinstance(reference, str) or not reference.startswith("#"):
            raise ValueError("Only local schema references are supported")
        fragment = unquote(reference[1:])
        if fragment and not fragment.startswith("/"):
            raise ValueError("Invalid local reference")
        target = self.source
        for part in fragment.split("/")[1:] if fragment else []:
            key = part.replace("~1", "/").replace("~0", "~")
            try:
                target = target[int(key)] if isinstance(target, list) else target[key]
            except (KeyError, ValueError, IndexError, TypeError) as error:
                raise ValueError("Unresolved local schema reference") from error
        if not isinstance(target, (dict, bool)):
            raise ValueError("Reference target is not a schema")
        return target

    def string(self, schema):
        minimum = schema.get("minLength", 0)
        maximum = schema.get("maxLength")
        if (
            not isinstance(minimum, int)
            or minimum < 0
            or minimum > 64
            or maximum is not None
            and (not isinstance(maximum, int) or maximum < minimum or maximum > 64)
        ):
            raise ValueError("String bounds must be 0..64")
        if "format" in schema:
            if schema["format"] == "email":
                value = "tvara@example.org"
            elif schema["format"] == "uuid":
                value = "00000000-0000-4000-8000-000000000000"
            else:
                raise ValueError("Supported string formats are email and uuid")
            if len(value) < minimum or maximum is not None and len(value) > maximum:
                raise ValueError("Format with these length bounds is unsupported")
            return (literal(json.dumps(value)),)
        if maximum is None:
            content = self.new((("char",) * minimum + ("chars",),))
        else:
            content = self.new(tuple(("char",) * n for n in range(minimum, maximum + 1)))
        return (literal('"') + (content,) + literal('"'),)


def compile_schema(schema):
    validate_supported_schema(schema)
    return SchemaCompiler(schema).compile()


def validate_supported_schema(schema):
    from jsonschema import Draft202012Validator, SchemaError

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ValueError("Invalid JSON Schema") from error
    annotations = {
        "title",
        "description",
        "default",
        "examples",
        "$schema",
        "$id",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
    }
    supported = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "$defs",
        "definitions",
        "$ref",
        "anyOf",
        "enum",
        "const",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minProperties",
        "maxProperties",
        "format",
    }

    def check(node):
        if isinstance(node, bool):
            return
        if not isinstance(node, dict):
            raise ValueError("Invalid schema node")
        if "$ref" in node and not node["$ref"].startswith("#"):
            raise ValueError("External schema references are unsupported")
        if not ("enum" in node or "const" in node) and set(node) - supported - annotations:
            raise ValueError(
                "Unsupported schema keywords: "
                + ", ".join(sorted(set(node) - supported - annotations))
            )
        for key in ("properties", "$defs", "definitions"):
            for child in node.get(key, {}).values():
                check(child)
        for child in node.get("anyOf", []):
            check(child)
        if "items" in node:
            check(node["items"])
        if isinstance(node.get("additionalProperties"), dict):
            check(node["additionalProperties"])

    check(schema)
