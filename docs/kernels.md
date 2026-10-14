# Kernel design

Kernels are header-only C++20 under [core/include/tvara](../core/include/tvara/quant.hpp).
Scalar code is the required baseline; SIMD variants are optional and compared
against it. See [architecture.md](architecture.md) for where kernels sit.

## What the served executor uses

| Operation | `PagedModelExecutor` path | Also available, not used by the executor |
|---|---|---|
| Weight matvec | scalar float32 `matvec_f32`, double accumulation | Q8_0/Q4_0/Q4_K integer dots (scalar, AVX2), AVX-512 and NEON Q8_0 dots, `threaded_matvec` |
| Attention scores | `attention_dot`: AVX2 or scalar by dispatch | optional prefetch |
| RMSNorm, RoPE, SwiGLU | scalar | `rms_norm_avx2`, `rotary_avx2`, `swiglu_avx2` |
| Prompt projection | per-token decode | `prefill_blocked`, `prefill_q8`, `prefill_weight_prefetch` |
| Threads | caller thread only | `ThreadPool`, `SpinBarrier` |

The executor dequantizes all weights to float32 at load
([model.hpp](../core/include/tvara/model.hpp) `tensor_floats`), so quantized
files do not reduce resident memory and the integer kernels are not on the
served path. The only runtime-dispatched SIMD in served decode is the attention
score reduction.

## Block formats

| Type | GGUF id | Values/block | Bytes | Layout | Dequantization |
|---|---|---|---|---|---|
| F32 | 0 | 1 | 4 | little-endian float | as stored |
| F16 | 1 | 1 | 2 | half | `half_to_float` |
| Q4_0 | 2 | 32 | 18 | half d, 16 packed bytes | low nibble i, high nibble i+16: d*(q-8) |
| Q8_0 | 8 | 32 | 34 | half d, 32 int8 | d*q |
| Q4_K | 12 | 256 | 144 | half d, half dmin, 12 scale bytes, 128 nibble bytes | d*scale*q - dmin*min per 32-value group |

Q4_K groups use six-bit scales and minima (`q4k_scale_min`); group g reads bytes
`16+(g/2)*32`, low nibbles for even g and high for odd. Row widths must be
divisible by the block size ([gguf.hpp](../core/include/tvara/gguf.hpp)
`tensor_bytes`). Details and tests: [quantization.md](quantization.md).

## Dot products

Integer dots take Q8 activations: one float32 scale (max |x|/127) and 32 int8
values per block, rounded half away from zero (`quantize_activations`). Each
block accumulates int32 products, then adds `d * scale * sum` into a double.
Q4_K also sums activations per group for the `dmin*min` term.

* Scalar ([quant.hpp](../core/include/tvara/quant.hpp)): `dot_q8_0`, `dot_q4_0`,
  `dot_q4_k`, `dot_f32` (double accumulation), `matvec_quant`.
* AVX2 ([simd.hpp](../core/include/tvara/simd.hpp)): int8 or nibbles widened
  to int16, `_mm256_madd_epi16` to int32 lanes, horizontal sum through an
  aligned store. No `maddubs` or VNNI. Unaligned loads on inputs.
* AVX-512 ([avx512.hpp](../core/include/tvara/avx512.hpp)): Q8_0 only, AVX512F
  plus AVX512BW, one 32-lane int16 widen and `_mm512_madd_epi16`. Opt-in; no
  supported hardware has run it ([avx512-validation.md](avx512-validation.md)).
* NEON ([neon.hpp](../core/include/tvara/neon.hpp)): Q8_0 only, `vmull_s8` plus
  pairwise widening add; compiled when `__ARM_NEON` is defined, otherwise the
  same function uses a portable loop. Not validated on ARM hardware
  ([arm-validation.md](arm-validation.md)).

## Dispatch

AVX2 and AVX-512 code compiles only for GCC/Clang on x86, as functions with
`__attribute__((target(...)))`; MSVC and other targets get scalar code. No
global `-march=native` is used. `use_avx2` requires `__builtin_cpu_supports("avx2")`
and that neither the `force_scalar` argument nor `TVARA_FORCE_SCALAR=1` is set.
`KernelDispatch` picks AVX2 or scalar Q8_0/Q4_0/Q4_K pointers.
`OptionalKernelDispatch` additionally swaps the Q8_0 pointer to AVX-512 when
`TVARA_ENABLE_AVX512=1`, AVX2 is in use, and AVX512F/BW are present. NEON is a
standalone function, not a dispatch entry. Attention reads the environment on
each `fused_paged_attention` call.

## Threading

`ThreadPool` ([thread_pool.hpp](../core/include/tvara/thread_pool.hpp)) has 1-256
persistent workers. `execute` runs one job on every worker while the caller
blocks, serializes concurrent submissions, rethrows the first worker exception
and rejects nested execution from a worker. `partition_rows` gives contiguous,
non-overlapping row ranges with the remainder spread over the first workers.
`threaded_matvec` is float32 only. `SpinBarrier` yields 64 times before waiting
on a condition variable. None of this is called by the executor.

## Attention

`fused_paged_attention` ([attention.hpp](../core/include/tvara/attention.hpp))
walks a sequence's block table in place, without gathering K/V. Query head h
reads KV head `h / (heads / kv_heads)`. Scores are double dot products scaled by
`1/sqrt(head_dim)`; `OnlineAttention` keeps a running maximum, normalizer and
weighted value sum in double. The current token's K/V is passed as a tail
before it is committed. `fused_prompt_attention` evaluates a causal chunk over
cached positions but is used only by tests. The contiguous `ModelExecutor` uses
a two-pass softmax (`grouped_attention`) and serves as the parity reference.
Evidence: [optimizations.md](optimizations.md).

## Memory layout

Weights are row-major `[output, input]` float32 (`Matrix`); GGUF dimension 0 is
the input width. A KV block holds all layers for `tokens_per_block` positions,
indexed `((layer*2 + kind)*tokens_per_block + slot)*kv_width`, with kind 0 for
keys and 1 for values and `kv_width = kv_heads*head_dim`. Blocks are float32,
preallocated and zeroed on allocation ([kv-cache.md](kv-cache.md)).

## Limits

* Weight memory is float32; quantized kernels are not a low-memory model path.
* Weight matvec, normalization, RoPE and SwiGLU are scalar and single-threaded
  in served decode.
* Prompt prefill runs token by token; tiled and Q8 prefill are experiments.
* AVX-512 and NEON kernels are unvalidated on matching hardware.
* SIMD RMSNorm, RoPE and SwiGLU vectorize arithmetic only; `exp`, `sin`, `cos`
  and `pow` stay scalar.
* Recorded kernel timings ([kernel-overhead.json](../bench/results/kernel-overhead.json),
  [prefill results](../bench/results/prefill/T179.json)) are local
  microbenchmarks, not model throughput.
