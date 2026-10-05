"""Strict JSON Schema API request adaptation to token constraints."""

from .schema import compile_schema
from .token_masks import TokenConstraint, MaskEngine, tokenizer_bytes


def response_constraint(response_format, tokenizer, eos_id=None):
    if (
        not isinstance(response_format, dict)
        or set(response_format) != {"type", "json_schema"}
        or response_format["type"] != "json_schema"
    ):
        raise ValueError("Only explicit json_schema response_format is supported")
    spec = response_format["json_schema"]
    if (
        not isinstance(spec, dict)
        or "schema" not in spec
        or set(spec) - {"schema", "name", "description", "strict"}
    ):
        raise ValueError("Invalid JSON Schema response format")
    if spec.get("strict", True) is not True:
        raise ValueError("Structured output requires strict=true")
    grammar = compile_schema(spec["schema"])
    return TokenConstraint(MaskEngine(tokenizer_bytes(tokenizer), grammar, eos_id))
