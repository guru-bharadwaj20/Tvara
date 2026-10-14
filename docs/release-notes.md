# Tvara 0.1.0 release notes

Tvara 0.1.0 is the first tagged release of a CPU-only research inference engine for
small Llama-family language models. It completes the planned 18-phase roadmap as a
validated, measured baseline rather than an optimized product. Every performance and
quality number in the repository comes from seeded random-weight fixtures; no
pretrained-model quality or speed claim is made.

## Contents

- **Native core (C++20):** read-only GGUF v3 loading with F32, F16, Q4_0, Q8_0 and Q4_K
  tensors; a Llama-family executor with grouped-query attention, RoPE (split or
  interleaved), RMSNorm and SwiGLU; paged KV storage with fused paged attention; a radix
  prefix cache; scalar kernels with runtime-dispatched AVX2, opt-in AVX-512 and NEON
  Q8_0 dot products; a persistent thread pool; and a C ABI (`tvara_capi`).
- **Python package (`tvara`):** ctypes bindings, byte-level BPE and SentencePiece
  tokenizers, sampling (temperature, top-k, top-p, min-p, repetition, frequency and
  presence penalties), GBNF grammars and a strict JSON Schema subset compiled to token
  masks, GGUF chat templates in a sandboxed Jinja environment, tool calls, an
  iteration-level scheduler, exact speculative decoding with draft-model and
  prompt-lookup proposers, and an independent NumPy reference model.
- **Server:** an OpenAI-compatible subset (`/v1/completions`, `/v1/chat/completions`,
  `/v1/models`, `/health`, `/ready`) with SSE streaming, seeded sampling, structured
  output and required/named tool choice.
- **Evaluation:** the matched llama.cpp baseline, STREAM bandwidth, repeated timings,
  prefix-cache traces, scheduler concurrency, speculative speedups, structured-output
  compliance and the end-to-end demo, tabulated in [performance.md](performance.md).

Design notes: [architecture.md](architecture.md), [kernels.md](kernels.md),
[sampling.md](sampling.md). Licensing and release contents:
[release-review.md](release-review.md).

## Setup

Requirements: Python 3.11+, CMake 3.24+ (installed by the `dev` extra), a C++20 GCC or
Clang compiler and a Make or Ninja generator. On Windows, use MinGW-w64 GCC (validated
with GCC 15.2); MSVC builds compile only the scalar paths and are not validated.

```text
git clone https://github.com/guru-bharadwaj20/Tvara.git
cd Tvara
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,server]"
.venv/Scripts/cmake.exe --preset release -G "MinGW Makefiles"
.venv/Scripts/cmake.exe --build --preset release
.venv/Scripts/ctest.exe --preset release
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe tools/run_demo.py
```

On Linux or macOS, use `.venv/bin/` and omit `-G "MinGW Makefiles"`. On Windows, put
the MinGW `bin` directory used for the build first on `PATH` before running `ctest`:
the test executables link the GCC runtime dynamically, and an older `libstdc++` found
first (for example Git's `mingw64/bin`) fails with exit code `0xc0000139`. The
`tvara_capi` library links its runtime statically. Serve a GGUF model
whose tokenizer IDs match its `tokenizer.json`:

```text
python -m tvara.server --model path/to/model.gguf --tokenizer path/to/tokenizer.json
```

The optional `oracle` extra (torch, transformers, safetensors, tokenizers) and the
pinned checkpoints in `tests/fixtures/*.json` enable the Transformers and tokenizer
oracle tests; without them those tests skip explicitly. See
[development.md](development.md), [api.md](api.md) and [demo.md](demo.md).

## Validation for this release

Recorded on host B of [performance.md](performance.md) (Windows 11, 32 logical CPUs,
GCC 15.2, Python 3.13) from the tagged tree:

- `ctest`: 29 of 29 native tests pass in the debug and release presets.
- `pytest`: 220 passed and none skipped, with the `oracle` extra installed and the pinned
  tiny Llama checkpoint and Qwen2.5 tokenizer verified by SHA-256.
- `ruff check` and `ruff format --check` pass for `python`, `tests/python` and `tools`.
- `tools/render_performance_tables.py --check` confirms [performance.md](performance.md)
  matches the recorded results.
- `tools/run_demo.py` passes all checks, and its normalized transcript matched the
  committed `bench/results/demo-transcript.json` exactly.
- `tools/review_release.py --check` passes with no errors or warnings.
- Linux sanitizer builds and the ARM and AVX-512 portability jobs run in GitHub Actions
  and are not part of this local record.

## Known limitations

- **Not optimized for real models.** The executor dequantizes every weight to float32 at
  load and runs scalar matvec on one thread. Quantized and SIMD kernels exist but are
  not on the model path. Resident memory is the float32 copy.
- **Prefill is per token.** Each prompt token is one C ABI call; there is no batched
  prompt matmul.
- **Served path is single-sequence.** The HTTP server calls the executor directly and
  does not use the scheduler, prefix cache or speculative decoding. There is no
  admission control against the shared KV pool, authentication or rate limiting.
- **Speculative decoding is slower than plain decoding** in every measured
  configuration, because verification is serial and rollback re-decodes a token.
  Draft distributions apply temperature only, not the server's sampling filters.
- **Architectures:** native GGUF execution supports Llama only. Qwen2 and Gemma 1 exist
  as NumPy reference prototypes. Scaled RoPE, sliding-window attention and other
  architectures are rejected.
- **HTTP sampling controls:** `temperature`, `top_p` and `seed` only; top-k, min-p and
  penalties are Python API options.
- **Structured output:** a documented JSON Schema subset (no `pattern`, numeric bounds
  or `oneOf`); string and array bounds are limited to 64; `email` and `uuid` formats
  accept one canonical value each. JSON whitespace is unbounded, so weak models can pad
  output until the token limit. Tool calls require `tool_choice: "required"` or a named
  function and are not streamed.
- **Portability:** AVX-512 and ARM NEON code has not run on supporting hardware. MSVC
  and macOS builds are untested. Sanitizers run on Linux CI only.
- **Evidence:** all benchmarks use tiny random-weight fixtures on two uncontrolled
  Windows machines; tiny models exaggerate per-call overhead. Prototypes (quantized KV,
  MoE routing, tree speculation, WebAssembly) are experiments, not features.
