"""Final structured-output compliance evaluation through the HTTP API (T265).

Every request goes through the FastAPI application and the native C ABI backend, using a
random-weight byte-level fixture with a 512-token context. Finished responses
(finish_reason "stop") must parse as JSON and validate against the request schema with an
independent jsonschema Draft 2020-12 validator, including format checks. Length-limited
responses must be a valid grammar prefix. Streaming responses must reproduce the
non-streaming text for the same seed. Unsupported schemas must fail with HTTP 400 before
generation. The fixture's weights are random: this measures constraint enforcement, not
whether a trained model chooses semantically useful values.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_api_fixture import make_fixture  # noqa: E402
from tvara.native import NativeModel  # noqa: E402
from tvara.response_constraints import response_constraint, tool_response_format  # noqa: E402
from tvara.server import NativeBackend, ServerConfig, create_app  # noqa: E402
from tvara.tokenizer import ByteBPE  # noqa: E402

SHORT = {"type": "string", "maxLength": 12}
SCHEMAS = {
    "flat-object": {
        "type": "object",
        "properties": {
            "name": SHORT,
            "age": {"type": "integer"},
            "active": {"type": "boolean"},
        },
        "required": ["name", "age", "active"],
        "additionalProperties": False,
    },
    "enum-unicode": {
        "type": "object",
        "properties": {"city": {"enum": ["Paris", "東京", "São Paulo"]}, "ok": {"const": True}},
        "required": ["city", "ok"],
        "additionalProperties": False,
    },
    "nested-object": {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "email": {"type": "string", "format": "email"},
                },
                "required": ["id", "email"],
                "additionalProperties": False,
            },
            "score": {"type": "number"},
        },
        "required": ["user", "score"],
        "additionalProperties": False,
    },
    "bounded-array": {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": SHORT, "minItems": 1, "maxItems": 3},
        },
        "required": ["tags"],
        "additionalProperties": False,
    },
    "string-bounds": {
        "type": "object",
        "properties": {"code": {"type": "string", "minLength": 3, "maxLength": 5}},
        "required": ["code"],
        "additionalProperties": False,
    },
    "uuid-format": {
        "type": "object",
        "properties": {"id": {"type": "string", "format": "uuid"}},
        "required": ["id"],
        "additionalProperties": False,
    },
    "nullable-anyof": {
        "type": "object",
        "properties": {
            "value": {"anyOf": [{"type": "integer"}, {"type": "null"}, SHORT]},
            "flag": {"type": ["boolean", "null"]},
        },
        "required": ["value", "flag"],
        "additionalProperties": False,
    },
    "defs-ref": {
        "$defs": {
            "point": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            }
        },
        "type": "object",
        "properties": {"start": {"$ref": "#/$defs/point"}, "end": {"$ref": "#/$defs/point"}},
        "required": ["start", "end"],
        "additionalProperties": False,
    },
    "optional-properties": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}, "c": {"type": "null"}},
        "minProperties": 1,
        "maxProperties": 2,
        "additionalProperties": False,
    },
    "top-level-array": {"type": "array", "items": {"enum": [1, 2, 3]}, "maxItems": 4},
    "unconstrained-string": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    "tool-call": tool_response_format(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": SHORT, "unit": {"enum": ["c", "f"]}},
                        "required": ["city", "unit"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "parameters": {"type": "object", "additionalProperties": False},
                },
            },
        ],
        "required",
    )["json_schema"]["schema"],
}
UNSUPPORTED = {
    "pattern": {"type": "string", "pattern": "^a+$"},
    "numeric-minimum": {"type": "integer", "minimum": 3},
    "oneOf": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
    "external-ref": {"$ref": "https://example.com/schema.json"},
    "string-too-long": {"type": "string", "maxLength": 1000},
    "unknown-format": {"type": "string", "format": "date-time"},
    "false-schema": False,
    "not-strict": None,
}
SAMPLING = [{"temperature": 0, "seed": 0}] + [
    {"temperature": 1.0, "top_p": top_p, "seed": seed}
    for top_p in (1.0, 0.9)
    for seed in range(1, 11)
]


def request_format(schema):
    return {
        "type": "json_schema",
        "json_schema": {"name": "result", "strict": True, "schema": schema},
    }


def valid_prefix(text, schema, tokenizer, eos_id):
    constraint = response_constraint(request_format(schema), tokenizer, eos_id)
    try:
        for token in tokenizer.encode(text):
            constraint.accept(token)
    except ValueError:
        return False
    return True


def stream_text(client, body):
    pieces, reason = [], None
    with client.stream("POST", "/v1/completions", json={**body, "stream": True}) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choice = event["choices"][0]
            pieces.append(choice.get("text", ""))
            reason = choice.get("finish_reason") or reason
    return "".join(pieces), reason


def evaluate(client, tokenizer, eos_id, max_tokens, stream_every):
    rows, results = [], []
    for name, schema in SCHEMAS.items():
        validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
        counts = {
            "requests": 0,
            "stop": 0,
            "length": 0,
            "valid": 0,
            "invalid": 0,
            "valid_prefix": 0,
            "http_errors": 0,
            "stream_checked": 0,
            "stream_mismatch": 0,
        }
        tokens, seconds = [], []
        for index, sampling in enumerate(SAMPLING):
            body = {
                "model": "fixture",
                "prompt": f"Return {name} JSON:",
                "max_tokens": max_tokens,
                "response_format": request_format(schema),
                **sampling,
            }
            start = time.perf_counter()
            response = client.post("/v1/completions", json=body)
            seconds.append(time.perf_counter() - start)
            counts["requests"] += 1
            if response.status_code != 200:
                counts["http_errors"] += 1
                rows.append(
                    {
                        "schema": name,
                        **sampling,
                        "status": response.status_code,
                        "error": response.text[:200],
                    }
                )
                continue
            data = response.json()
            choice = data["choices"][0]
            text, reason = choice["text"], choice["finish_reason"]
            tokens.append(data["usage"]["completion_tokens"])
            row = {
                "schema": name,
                **sampling,
                "finish_reason": reason,
                "completion_tokens": data["usage"]["completion_tokens"],
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            counts[reason] += 1
            if reason == "stop":
                try:
                    errors = [e.message for e in validator.iter_errors(json.loads(text))]
                except json.JSONDecodeError as error:
                    errors = [f"JSON parse error: {error}"]
                row["errors"] = errors
                counts["valid" if not errors else "invalid"] += 1
                if errors:
                    row["text"] = text
            else:
                ok = valid_prefix(text, schema, tokenizer, eos_id)
                row["valid_prefix"] = ok
                counts["valid_prefix"] += ok
            if index % stream_every == 0:
                streamed, stream_reason = stream_text(client, body)
                counts["stream_checked"] += 1
                if streamed != text or stream_reason != reason:
                    counts["stream_mismatch"] += 1
                    row["stream_mismatch"] = True
            rows.append(row)
        results.append(
            {
                "schema": name,
                **counts,
                "compliance_of_finished": counts["valid"] / counts["stop"]
                if counts["stop"]
                else None,
                "completion_tokens_median": sorted(tokens)[len(tokens) // 2] if tokens else None,
                "request_seconds_median": sorted(seconds)[len(seconds) // 2],
            }
        )
        print(
            f"{name:<22} stop={counts['stop']:>2} valid={counts['valid']:>2} "
            f"length={counts['length']:>2} prefix_ok={counts['valid_prefix']:>2} "
            f"errors={counts['http_errors']} stream_mismatch={counts['stream_mismatch']}",
            flush=True,
        )
    return results, rows


def rejections(client):
    outcomes = []
    for name, schema in UNSUPPORTED.items():
        response_format = (
            {"type": "json_schema", "json_schema": {"schema": {"type": "integer"}, "strict": False}}
            if name == "not-strict"
            else request_format(schema)
        )
        response = client.post(
            "/v1/completions",
            json={
                "model": "fixture",
                "prompt": "x",
                "max_tokens": 4,
                "response_format": response_format,
            },
        )
        outcomes.append(
            {
                "case": name,
                "status": response.status_code,
                "rejected": response.status_code == 400,
                "message": response.json().get("error", {}).get("message")
                if response.headers.get("content-type", "").startswith("application/json")
                else None,
            }
        )
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="models/structured-fixture")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--stream-every", type=int, default=3)
    parser.add_argument("--output", default="bench/results/structured-output-compliance.json")
    args = parser.parse_args()
    directory = make_fixture(args.fixture, context=512)
    model_path = directory / "model.gguf"
    measured_at = datetime.now(timezone.utc).isoformat()
    tokenizer = ByteBPE.from_json(directory / "tokenizer.json")
    backend = NativeBackend(NativeModel(model_path), tokenizer)
    with TestClient(create_app(backend, ServerConfig(served_name="fixture"))) as client:
        results, rows = evaluate(
            client, tokenizer, backend.eos_id, args.max_tokens, args.stream_every
        )
        rejected = rejections(client)
    totals = {
        key: sum(r[key] for r in results)
        for key in (
            "requests",
            "stop",
            "length",
            "valid",
            "invalid",
            "valid_prefix",
            "http_errors",
            "stream_checked",
            "stream_mismatch",
        )
    }
    status = (
        "pass"
        if totals["invalid"] == 0
        and totals["http_errors"] == 0
        and totals["stream_mismatch"] == 0
        and totals["valid_prefix"] == totals["length"]
        and all(r["rejected"] for r in rejected)
        else "fail"
    )
    output = {
        "task": "T265",
        "measured_at_utc": measured_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "fixture": {
            "path": str(model_path).replace("\\", "/"),
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "vocabulary": 257,
            "context": 512,
            "eos_id": backend.eos_id,
            "weights": "random, seed 2026; not a trained model",
        },
        "protocol": {
            "endpoint": "POST /v1/completions via FastAPI TestClient and the native backend",
            "sampling": "greedy seed 0, then temperature 1.0 with top_p 1.0 and 0.9 for seeds 1-10",
            "max_tokens": args.max_tokens,
            "validator": "jsonschema Draft202012Validator with FORMAT_CHECKER",
            "length_check": "length-limited text is re-encoded and replayed through a fresh "
            "constraint; every byte must be grammar-accepted",
            "stream_check": f"every {args.stream_every}th request repeated with stream=true; "
            "text and finish_reason must match",
            "chat_and_tools": "the fixture has no chat template; tool schemas are evaluated "
            "through the same tool_response_format translation used by chat",
        },
        "status": status,
        "totals": totals,
        "schemas": results,
        "unsupported_rejections": rejected,
        "requests": rows,
        "scope": "constraint enforcement on random weights; not semantic accuracy, "
        "trained-model finish rates, or production latency",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, **totals}))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
