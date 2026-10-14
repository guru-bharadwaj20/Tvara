# End-to-end demo

`tools/run_demo.py` is a reproducible demonstration of the complete local pipeline:
GGUF loading, the C++ executor through the C ABI, byte-level tokenization, a GGUF chat
template, sampling, SSE streaming, JSON-schema constrained output, and tool calls, all
served over real HTTP by `python -m tvara.server`. The model is a seeded random-weight
fixture, so the generated text is arbitrary bytes. The demo shows that the pipeline works
and is deterministic, not that it produces useful language.

## Run it

From a clean clone, with Python 3.11+, a GCC or Clang C++20 compiler and a Make or Ninja
generator (Windows: MinGW-w64 GCC; see [development.md](development.md)):

```text
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,server]"
.venv/Scripts/cmake.exe --preset release -G "MinGW Makefiles"
.venv/Scripts/cmake.exe --build --preset release
.venv/Scripts/python.exe tools/run_demo.py
```

On Linux or macOS, use `.venv/bin/` and omit the `-G` option. The script writes the
fixture to `models/demo-fixture/` (ignored by Git), starts the server on a free local
port, and writes `bench/results/demo-transcript.json`. It exits non-zero if any check fails.

## What it does

| Step | Request | Checked |
|---|---|---|
| health, models | `GET /health`, `GET /v1/models` | ready, served name |
| greedy completion | `POST /v1/completions`, 12 tokens | status 200, token usage |
| seeded sampling | temperature 0.8, top_p 0.95, seed 7 | status 200 |
| streamed sampling | same request with `stream: true` | SSE text and finish reason equal the non-streaming response; ends with `[DONE]` |
| JSON schema output | strict schema with enum, integer and boolean fields | `finish_reason: "stop"`, validates with `jsonschema` |
| chat | system and user messages through the GGUF chat template | status 200 |
| tool call | `tools` with `tool_choice: "required"` | `finish_reason: "tool_calls"`; arguments validate against the declared parameters |
| explicit error | schema with `pattern` | HTTP 400, `Unsupported schema keywords: pattern` |

The full session then runs again against a second, freshly started server process. The
two transcripts must be identical after removing response IDs, creation times and
tool-call IDs, which are random by design.

## Recorded run

`bench/results/demo-transcript.json` holds the normalized transcript of the run recorded
on 2026-10-14 UTC (host B in [performance.md](performance.md)); all five checks passed.
Fixture SHA-256 `00c26880fca4cdc21645bcc19918f57455cb9a1e5f6f182afbef9a35c8ed5aa6`
(model) and `314d4ca982f7f93c74e8dac5c4b20d05f84be2ef2637f97267721d8d36d9b1a5` (tokenizer).
Selected responses:

- The structured request finished after 84 tokens with
  `{"city":"Paris","indoor":false,"days":6}` padded by the whitespace the JSON grammar
  allows between tokens.
- The tool call produced `get_weather` with schema-valid `city` and `unit` arguments
  after 117 tokens.
- Plain completions and chat return replacement characters, because random weights
  emit byte sequences that are not valid UTF-8.

Outputs depend on the native build only through floating-point results; a different
compiler, CPU or optimization level can change sampled tokens. Rerun the script to
regenerate the transcript for another machine instead of comparing text across hosts.
