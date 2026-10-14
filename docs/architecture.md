# Architecture

This note describes the code as it exists, not the planned design in
[design.md](design.md). Kernel details are in [kernels.md](kernels.md) and the
sampling pipeline in [sampling.md](sampling.md).

## Layers

| Layer | Code | Language | Served by HTTP |
|---|---|---|---|
| File mapping, GGUF parse | [mapped_file.hpp](../core/include/tvara/mapped_file.hpp), [gguf.hpp](../core/include/tvara/gguf.hpp), [gguf_model.hpp](../core/include/tvara/gguf_model.hpp) | C++ | yes |
| Config and weight load | [model.hpp](../core/include/tvara/model.hpp) (`ModelConfig`, `load_weights`) | C++ | yes |
| Executor | [paged_model.hpp](../core/include/tvara/paged_model.hpp), [attention.hpp](../core/include/tvara/attention.hpp) | C++ | yes |
| Contiguous reference executor | `ModelExecutor` in [model.hpp](../core/include/tvara/model.hpp), [generation.hpp](../core/include/tvara/generation.hpp) | C++ | no (tests, parity) |
| Paged KV | [kv.hpp](../core/include/tvara/kv.hpp) | C++ | yes |
| Radix prefix cache | [radix.hpp](../core/include/tvara/radix.hpp) | C++ | no |
| C ABI | [c_api.h](../core/include/tvara/c_api.h), [c_api.cpp](../core/src/c_api.cpp) | C | yes |
| ctypes binding, sampling | [native.py](../python/tvara/native.py) | Python | yes |
| Tokenizer, chat template | [tokenizer.py](../python/tvara/tokenizer.py), [chat_templates.py](../python/tvara/chat_templates.py) | Python | yes |
| Grammar and schema masks | [grammar.py](../python/tvara/grammar.py), [schema.py](../python/tvara/schema.py), [token_masks.py](../python/tvara/token_masks.py), [response_constraints.py](../python/tvara/response_constraints.py) | Python | yes |
| Iteration scheduler | [scheduler.py](../python/tvara/scheduler.py) | Python | no |
| Speculative decoding | [speculative.py](../python/tvara/speculative.py), [speculative_tree.py](../python/tvara/speculative_tree.py) | Python | no |
| HTTP adapter | [server.py](../python/tvara/server.py) | Python | yes |

## Load path

`tvara_model_open` constructs a `PagedModelExecutor` from `load_weights`. That
maps the file read-only, parses the GGUF document, requires
`general.architecture == "llama"`, derives `LlamaShape` from `llama.*` keys and
the tokenizer vocabulary length, rejects scaled RoPE, and validates every tensor
name and shape. RoPE layout defaults to interleaved; `tvara.rope.layout` may set
`split`. Every tensor (F32, F16, Q4_0, Q8_0, Q4_K) is dequantized into owned
float32 vectors and checked for finiteness. The mapping is released when
`load_weights` returns; resident model memory is the float32 copy. A missing
`output.weight` means tied embeddings. The C ABI reopens the file once to
serialize all metadata as JSON for Python.

## Per-token execution

`PagedModelExecutor::decode` computes one token at a known position: embedding
row, then per layer RMSNorm, Q/K/V matvec, RoPE on Q and K, fused paged attention
over cached blocks plus the current K/V as an uncommitted tail, output projection,
residual, RMSNorm, SwiGLU feed-forward, residual. Final RMSNorm and the output
(or tied embedding) matrix produce float32 logits. New K/V for all layers is
appended only after every layer succeeds, so a failed decode leaves the
sequence unchanged. Prompt prefill is repeated single-token decode: the binding
issues one ABI call per prompt token. There is no batched prompt matmul or
cross-sequence batching in the executor.

## State ownership

| State | Owner | Lifetime |
|---|---|---|
| Float32 weights | `tvara_model` handle | open to close |
| KV block pool, block tables | `PagedCache` inside the handle | preallocated at open; `NativeModel` defaults 256 blocks x 16 tokens |
| Sequence ID | Python caller (generator, scheduler backend, speculative adapter) | create to drop; generators drop in `finally` |
| Last error string | `thread_local` in c_api.cpp | until the next ABI call on that thread |
| Tokenizer, grammar state, RNG, penalty history | Python, per request | one generation |
| `RadixCache` snapshots | C++ caller | not reachable through the C ABI |

The KV pool is shared by all sequences of one handle. The HTTP server has no
admission control against it: `CacheExhausted` during a request surfaces as an
inference error. See [kv-cache.md](kv-cache.md) and [prefix-cache.md](prefix-cache.md).

## Threading and locking

Model compute runs on the calling thread. [thread_pool.hpp](../core/include/tvara/thread_pool.hpp)
and [threaded_kernels.hpp](../core/include/tvara/threaded_kernels.hpp) exist and
are tested, but the executor does not use them. Each handle has one
`std::mutex`; create, decode, decode_batch, drop and truncate hold it for the
whole call. Vocabulary size, context size and metadata are immutable after open
and read without it. `NativeModel` additionally holds a `threading.RLock` around
each native call and the subsequent error read. FastAPI runs blocking
generation through `run_in_threadpool`/`iterate_in_threadpool`; concurrent
requests interleave at token granularity on the lock. `Scheduler`, `PagedCache`
and `RadixCache` are single-owner and not thread-safe.

## Error propagation

C++ validates shapes at public boundaries and throws `std::invalid_argument`,
`gguf::Error`, `CacheExhausted` or `std::logic_error`. The ABI catches all
exceptions, copies the message to the thread-local error, and returns `nullptr`,
0 or -1. `tvara_sequence_decode_batch` truncates the sequence to its entry
length on failure. Python raises `RuntimeError` with that message; Python-side
validation raises `ValueError`. The server maps `ValueError` to HTTP 400 with
the message, `RuntimeError` to 503 with a fixed "Inference failed" message, and
request-field validation errors to 400 with a generic message. Errors after streaming starts emit one SSE
error object followed by `data: [DONE]`. Client disconnects close the generator,
which drops the native sequence.

## Native, Python and reference code

Numerical execution, GGUF parsing and KV storage are C++. Tokenization, chat
templates, sampling, grammar masks, scheduling, speculation and HTTP are Python.
[python/tvara/reference](../python/tvara/reference/__init__.py) holds independent
NumPy float32 models, the Hugging Face oracle and metrics; it is used for
validation only and is never served. Prototypes, not integrated into serving:
MoE routing ([moe.py](../python/tvara/moe.py)), tree speculation
([speculative_tree.py](../python/tvara/speculative_tree.py)), quantized KV
([quantized_kv.py](../python/tvara/quantized_kv.py), [quantized-kv.md](quantized-kv.md)),
Qwen2/Gemma references ([variants.md](variants.md)) and WASM
([wasm-feasibility.md](wasm-feasibility.md)).

## Limits

* The server calls `NativeModel.iter_tokens` directly; it does not use
  `Scheduler`, `RadixCache` or speculative decoding.
* `NativeSchedulerBackend` is greedy-only and decodes a batch serially
  ([scheduler.md](scheduler.md)).
* The C ABI has truncate but no fork, so prefix reuse across sequences is not
  available to Python.
* Native execution is Llama-only and float32 in memory regardless of file type.
