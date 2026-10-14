# Tvara

A CPU-only inference engine for small language models, built from scratch.

Tvara loads GGUF weights, executes Llama-family transformer models in C++20, and serves
text generation through an OpenAI-compatible API. The project focuses on
understanding each layer of inference and measuring correctness and performance
against Hugging Face Transformers and llama.cpp on the same hardware.

## Purpose and scope

Build and understand a complete local inference pipeline on CPUs. The work started
with a single Llama-family sequence and a float32 reference, and added quantization,
concurrency, and constrained decoding only after correctness was established. The
primary target is x86-64 with a scalar fallback and runtime-dispatched AVX2. GPU
execution and model training are outside the scope. This project reports measured
results rather than assumed speedups or promised parity with mature engines.

## Status

Version 0.1.0 is the first release; see the [release notes](docs/release-notes.md) for
setup, validation, and known limitations. It is a validated research baseline, not
an optimized engine: the executor runs float32 scalar matvec on one thread, and all
recorded results use seeded random-weight fixtures, not pretrained models.

- GGUF loading with F32, F16, Q4_0, Q8_0, and Q4_K tensors (dequantized to float32).
- A native Llama-family executor with paged KV storage and fused paged attention,
  checked against an independent NumPy reference and Transformers logits.
- Byte-level BPE and SentencePiece tokenizers, GGUF chat templates, and sampling
  with temperature, top-k, top-p, min-p, and repetition penalties.
- Grammar-constrained generation for GBNF and a strict JSON Schema subset, including
  tool calls.
- An OpenAI-compatible server with SSE streaming.
- Separately tested components not yet on the served path: radix-tree prefix reuse,
  an iteration-level scheduler, and exact speculative decoding.
- Prototypes: Qwen2 and Gemma 1 reference models, quantized KV, MoE routing, tree
  speculation, AVX-512 and NEON Q8_0 kernels.

## Quick start

```text
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,server]"
.venv/Scripts/cmake.exe --preset release -G "MinGW Makefiles"
.venv/Scripts/cmake.exe --build --preset release
.venv/Scripts/python.exe tools/run_demo.py
```

On Linux or macOS, use `.venv/bin/` and omit the generator option. The demo starts the
server against a generated fixture and exercises every endpoint
([demo.md](docs/demo.md)). See [local development](docs/development.md) for the full
test commands and [API setup](docs/api.md) for serving your own GGUF model.

## Documentation

- Design: [architecture](docs/architecture.md), [kernels](docs/kernels.md),
  [sampling](docs/sampling.md), and the original [proposal](docs/design.md).
- Results: [performance tables](docs/performance.md), [benchmark protocol](docs/benchmarks.md),
  [correctness gates](docs/correctness.md), and [reference validation](docs/reference-validation.md).
- Features: [structured output](docs/structured-output.md),
  [speculative decoding](docs/speculative-decoding.md), [scheduler](docs/scheduler.md),
  [prefix cache](docs/prefix-cache.md), and [model targets](docs/models.md).
- Release: [release notes](docs/release-notes.md) and [release review](docs/release-review.md).
- Contributions: [local development](docs/development.md) and the
  [pull request template](.github/pull_request_template.md) describe how changes are
  validated.

Model weights, datasets, and third-party dependencies retain their own licenses and
access requirements.

## License

Tvara's original code and documentation are available under the
[MIT License](LICENSE).
