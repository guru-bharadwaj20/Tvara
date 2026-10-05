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


def tool_response_format(tools, tool_choice):
    if not isinstance(tools, list) or not 1 <= len(tools) <= 16:
        raise ValueError("Invalid tool list")
    selected = None
    if isinstance(tool_choice, dict):
        if (
            set(tool_choice) != {"type", "function"}
            or tool_choice.get("type") != "function"
            or not isinstance(tool_choice["function"], dict)
            or set(tool_choice["function"]) != {"name"}
        ):
            raise ValueError("Invalid named tool choice")
        selected = tool_choice["function"]["name"]
    elif tool_choice != "required":
        raise ValueError("Tools require explicit required or named tool_choice")
    variants = []
    names = set()
    for tool in tools:
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "function"
            or set(tool) - {"type", "function"}
        ):
            raise ValueError("Invalid function tool")
        function = tool.get("function")
        if not isinstance(function, dict) or set(function) - {
            "name",
            "description",
            "parameters",
            "strict",
        }:
            raise ValueError("Invalid function signature")
        name = function.get("name")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 64
            or name in names
            or not all(c.isalnum() or c in "_-" for c in name)
        ):
            raise ValueError("Invalid or duplicate tool name")
        names.add(name)
        parameters = function.get("parameters", {"type": "object", "additionalProperties": False})
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError("Tool parameters must be an object schema")
        if selected is None or selected == name:
            variants.append(
                {
                    "type": "object",
                    "properties": {"name": {"const": name}, "arguments": parameters},
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                }
            )
    if not variants:
        raise ValueError("Named tool is not declared")
    schema = variants[0] if len(variants) == 1 else {"anyOf": variants}
    compile_schema(schema)
    return {
        "type": "json_schema",
        "json_schema": {"name": "tool_call", "schema": schema, "strict": True},
    }
