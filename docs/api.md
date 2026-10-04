# Local CPU serving

Build the native library and install the server extra before starting the API:

```text
python -m pip install -e ".[server]"
cmake --preset debug
cmake --build --preset debug
python -m tvara.server --model models/model.gguf --tokenizer models/tokenizer.json --name tvara-local
```

The model must use a supported architecture and a tokenizer with matching vocabulary
IDs. GET /health, GET /ready, and GET /v1/models expose readiness and the served name.
POST /v1/completions accepts text or token IDs, model, max_tokens, temperature,
top_p, seed, and stream. POST /v1/chat/completions accepts role/content messages;
native chat requires a validated template, added in the chat-template milestone.

```python
import httpx
response = httpx.post("http://127.0.0.1:8000/v1/completions", json={
    "model": "tvara-local", "prompt": "Hello", "max_tokens": 16, "seed": 0
})
response.raise_for_status()
print(response.json()["choices"][0]["text"])
```

Set stream=true to receive SSE chunks followed by data: [DONE]. IDs, creation time,
model, choices, finish reason, and nonstreaming token usage follow the documented
compatibility subset. Unsupported fields and invalid model/context limits return
an error object. Disconnection closes generation iterators. Inference errors are
sanitized; this initial local server does not implement authentication or batching.

Reproduce the tested native HTTP path with tools/make_api_fixture.py followed by
tools/smoke_api.py. That fixture executes the real C++ model pipeline but has random
weights and does not demonstrate useful text generation. Wire-contract tests use a
separate fixture backend for deterministic text, chat, and SSE assertions. Results
in bench/results/api-smoke.json record the actual native smoke run.
