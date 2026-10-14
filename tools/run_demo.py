"""Reproducible end-to-end demo over real HTTP (T268).

Builds a seeded random-weight fixture with a chat template, starts `python -m tvara.server`
as a separate process, and drives every public endpoint: health, models, greedy and seeded
completions, SSE streaming, JSON-schema output, chat, a tool call and an explicit error.
The whole session runs twice against two fresh server processes; the normalized
transcripts (request IDs, timestamps and tool-call IDs removed) must be identical. The
weights are random, so generated text is arbitrary bytes: the demo shows the pipeline
and its determinism, not useful language output.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_api_fixture import make_fixture  # noqa: E402

CHAT_TEMPLATE = (
    "{% if tools %}<|tools|>{% for t in tools %}{{ t.function.name }} {% endfor %}\n{% endif %}"
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
NAME = "tvara-demo"
SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"enum": ["Paris", "Tokyo", "Lima"]},
        "days": {"type": "integer"},
        "indoor": {"type": "boolean"},
    },
    "required": ["city", "days", "indoor"],
    "additionalProperties": False,
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"enum": ["Paris", "Tokyo"]}, "unit": {"enum": ["c", "f"]}},
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
        },
    }
]


def steps():
    completion = {"model": NAME, "prompt": "Once upon a time", "max_tokens": 12}
    return [
        ("health", "GET", "/health", None),
        ("models", "GET", "/v1/models", None),
        ("greedy completion", "POST", "/v1/completions", completion),
        (
            "seeded sampling",
            "POST",
            "/v1/completions",
            {**completion, "temperature": 0.8, "top_p": 0.95, "seed": 7},
        ),
        (
            "streamed seeded sampling",
            "STREAM",
            "/v1/completions",
            {**completion, "temperature": 0.8, "top_p": 0.95, "seed": 7, "stream": True},
        ),
        (
            "JSON schema output",
            "POST",
            "/v1/completions",
            {
                "model": NAME,
                "prompt": "Plan a trip as JSON:",
                "max_tokens": 160,
                "temperature": 1.0,
                "seed": 3,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "trip", "strict": True, "schema": SCHEMA},
                },
            },
        ),
        (
            "chat",
            "POST",
            "/v1/chat/completions",
            {
                "model": NAME,
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Say hello."},
                ],
                "max_tokens": 12,
                "seed": 1,
                "temperature": 0.7,
            },
        ),
        (
            "tool call",
            "POST",
            "/v1/chat/completions",
            {
                "model": NAME,
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": TOOLS,
                "tool_choice": "required",
                "max_tokens": 160,
                "temperature": 1.0,
                "seed": 2,
            },
        ),
        (
            "unsupported schema is rejected",
            "POST",
            "/v1/completions",
            {
                "model": NAME,
                "prompt": "x",
                "max_tokens": 4,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": {"type": "string", "pattern": "^a+$"}},
                },
            },
        ),
    ]


def normalize(body):
    body = dict(body)
    for key in ("id", "created"):
        if key in body:
            body[key] = "<varies>"
    for choice in body.get("choices", []):
        for call in (choice.get("message") or {}).get("tool_calls") or []:
            call["id"] = "<varies>"
    return body


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def session(model, tokenizer):
    port = free_port()
    command = [
        sys.executable,
        "-m",
        "tvara.server",
        "--model",
        str(model),
        "--tokenizer",
        str(tokenizer),
        "--name",
        NAME,
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    base = f"http://127.0.0.1:{port}"
    transcript = []
    try:
        with httpx.Client(base_url=base, timeout=60) as client:
            deadline = time.monotonic() + 30
            while True:
                try:
                    if client.get("/ready").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if process.poll() is not None:
                    raise RuntimeError(process.stderr.read().decode(errors="replace"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Demo server did not become ready")
                time.sleep(0.1)
            for label, method, path, body in steps():
                if method == "GET":
                    response = client.get(path)
                    entry = {"status": response.status_code, "response": response.json()}
                elif method == "POST":
                    response = client.post(path, json=body)
                    entry = {"status": response.status_code, "response": normalize(response.json())}
                else:
                    with client.stream("POST", path, json=body) as response:
                        events = [
                            line[6:] for line in response.iter_lines() if line.startswith("data: ")
                        ]
                    chunks = [normalize(json.loads(e)) for e in events if e != "[DONE]"]
                    entry = {
                        "status": response.status_code,
                        "events": len(events),
                        "done": events[-1] == "[DONE]",
                        "text": "".join(c["choices"][0].get("text", "") for c in chunks),
                        "finish_reason": chunks[-1]["choices"][0]["finish_reason"],
                    }
                transcript.append(
                    {"step": label, "method": method, "path": path, "request": body} | entry
                )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.stderr:
            process.stderr.close()
    return transcript


def check(transcript):
    by_step = {entry["step"]: entry for entry in transcript}
    sampled = by_step["seeded sampling"]["response"]["choices"][0]
    streamed = by_step["streamed seeded sampling"]
    structured = by_step["JSON schema output"]["response"]["choices"][0]
    tool = by_step["tool call"]["response"]["choices"][0]
    checks = {
        "all expected statuses": all(
            entry["status"] == (400 if entry["step"].endswith("rejected") else 200)
            for entry in transcript
        ),
        "stream matches non-stream text": streamed["text"] == sampled["text"]
        and streamed["finish_reason"] == sampled["finish_reason"]
        and streamed["done"],
        "structured output is schema-valid": structured["finish_reason"] == "stop"
        and set(json.loads(structured["text"])) == set(SCHEMA["required"]),
        "tool call parsed": tool["finish_reason"] == "tool_calls"
        and tool["message"]["tool_calls"][0]["function"]["name"] == "get_weather",
    }
    from jsonschema import validate

    validate(json.loads(structured["text"]), SCHEMA)
    validate(
        json.loads(tool["message"]["tool_calls"][0]["function"]["arguments"]),
        TOOLS[0]["function"]["parameters"],
    )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="models/demo-fixture")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", default="bench/results/demo-transcript.json")
    args = parser.parse_args()
    directory = make_fixture(args.fixture, context=512, chat_template=CHAT_TEMPLATE)
    model, tokenizer = directory / "model.gguf", directory / "tokenizer.json"
    started = datetime.now(timezone.utc).isoformat()
    transcripts = [session(model, tokenizer) for _ in range(args.runs)]
    identical = all(t == transcripts[0] for t in transcripts)
    checks = check(transcripts[0]) | {f"{args.runs} server processes identical": identical}
    result = {
        "task": "T268",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "command": "python tools/run_demo.py",
        "fixture": {
            "path": str(model).replace("\\", "/"),
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            "context": 512,
            "vocabulary": 257,
            "weights": "random, seed 2026; not a trained model",
            "chat_template": CHAT_TEMPLATE,
        },
        "runs": args.runs,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "normalized": "id and created fields and tool-call IDs replaced with <varies>",
        "transcript": transcripts[0],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for entry in transcripts[0]:
        shown = entry.get("text")
        if shown is None and isinstance(entry["response"], dict):
            choices = entry["response"].get("choices") or [{}]
            shown = choices[0].get("text") or choices[0].get("message")
        print(f"{entry['status']} {entry['step']}: {json.dumps(shown)[:100]}")
    print(json.dumps({"status": result["status"], **checks}))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
