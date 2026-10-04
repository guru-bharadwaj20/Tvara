# Prefill and attention optimization decisions

Measurements are captured by `tools/check_prefill.py` from freshly compiled C++
programs. Each JSON includes compiler, flags and platform. Timings are medians
after a warm-up, on synthetic deterministic inputs, and can change with CPU load.
Run an individual check with `.venv/Scripts/python tools/check_prefill.py
tests/core/prefill_t179.cpp bench/results/prefill/T179.json`.

## Accepted

* Fused paged attention is integrated in `PagedModelExecutor.decode`, including
  the uncommitted current token as a tail. It eliminates full-cache gathering
  and score-vector allocation. T171 checks cached and tail outputs, T173 checks
  causal chunk visibility, T174 poisons unused block slots, and T178 compares
  full two-layer logits against the independent contiguous executor.
* Online softmax maintains a running maximum and normalizer in double precision.
  T170 checks large scores and T177 compares contexts through 8192 tokens.
* AVX2 attention score reductions widen input lanes to doubles before products;
  runtime dispatch retains a scalar fallback and `TVARA_FORCE_SCALAR` support.
  T175 checks awkward dimensions and records dispatch overhead.
* Q8 projection accepts reusable activation tiles: quantization happens once
  per token, shared by all output rows. T168 compares to separately decoded
  weights and reconstructed activations, with separate quantization timings.

## Retained as explicit experiments

* Cache-blocked F32 prefill is an API, with explicit tile dimensions and boundary
  tests. T169 records four configurations and the plain reference; T179 records
  projection throughput for 1, 8, 32, 128 and 512 tokens. It is not enabled in the
  model executor: small synthetic shapes do not justify a global default.
* Weight and KV prefetch are optional and disabled by default. T176 measures
  both configurations. Benefits require broader CPU and model measurements.

## Limits

These are kernel and correctness measurements, not real-model speed claims.
The model loader still dequantizes weights into F32: Q8 tile reuse is an explicit
kernel path and does not reduce the executor's model memory. Prompt attention
is causal but the current model forward still executes tokens sequentially.
No claim of a production-quality tiled end-to-end prefill is made. Performance
acceptance means enabling the measured and validated fused attention path;
the other kernels remain callable experiments pending representative models.

## Recorded run

Full-model parity over 73 tokens had maximum absolute logit error 2.38419e-07. At this small shape, fused attention took 57.8 us versus 98.6 us for gathered unfused attention. The integrated path removes gathered buffers independent of the timing result. Broader end-to-end benchmarks remain necessary before performance claims.

Across contexts through 8192 positions, maximum fused/unfused absolute error was 1.78814e-07.

Reproducible inputs and measured raw evidence:

* [T168 JSON](../bench/results/prefill/T168.json)
* [T169 JSON](../bench/results/prefill/T169.json)
* [T175 JSON](../bench/results/prefill/T175.json)
* [T176 JSON](../bench/results/prefill/T176.json)
* [T177 JSON](../bench/results/prefill/T177.json)
* [T178 JSON](../bench/results/prefill/T178.json)
* [T179 JSON](../bench/results/prefill/T179.json)
