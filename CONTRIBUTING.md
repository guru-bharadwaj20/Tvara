# Contributing to Tvara

Tvara is in early implementation. Read [README.md](README.md) and the
[architecture proposal](docs/design.md) before choosing work.

## Workflow

1. Describe the problem, scope, and expected result in an issue or discussion.
2. Pick a small task whose prerequisites are complete and work on a branch.
3. Keep changes focused and explain design decisions in accompanying documentation.
4. For implementation changes, compare behavior with a reference and add relevant
   correctness tests. Report the commands and environment used for validation.
5. For performance changes, record hardware, model file, quantization, thread count,
   prompt lengths, warm-up, repetitions, and baseline configuration.
6. Submit a pull request describing the change and validation. Update task status
   only when the deliverable exists and the relevant checks pass.

Build, formatting, and test commands will be documented when the toolchain exists.
Do not commit model weights, secrets, private traces, or generated benchmark data
without an agreed storage plan. Respect third-party licenses and cite prior work.
Contributions to the project's original material are made under [MIT](LICENSE).

## Timeline and task status

The roadmap contains **270 subtasks across 18 phases**. Weeks are relative to the
start of implementation and follow the proposal's estimated 18-week, four-person
schedule. Phases overlap where contributors can work independently; later features
still depend on a validated model executor. Phase 17 includes optional stretch work
that may extend beyond week 18. Ownership and dates have not been assigned yet.

- ✅ Completed: the deliverable exists and its relevant validation has passed.
- ❌ Not completed: planned, unverified, or not yet implemented.

Current status: 250 tasks are completed and 20 are not completed.
A proposal describing a feature does not count as implementing it. When updating
this list, link the implementation or validation evidence and revise the totals.
The tracked design notes under docs/ replace the unavailable local proposal.
Completed tasks link to their deliverables; implementation tasks also require checks.

### Phase 01: Scope and documentation

Target: Week 1.

- ✅ T001 — Document the CPU-only project purpose. Evidence: [README purpose](README.md#purpose-and-scope).
- ✅ T002 — Write the initial architecture proposal. Evidence: [docs/design.md](docs/design.md).
- ✅ T003 — Identify the intended small-model size range. Evidence: [docs/models.md](docs/models.md).
- ✅ T004 — Outline planned inference components. Evidence: [docs/components.md](docs/components.md).
- ✅ T005 — Define reference-based correctness goals. Evidence: [docs/correctness.md](docs/correctness.md).
- ✅ T006 — Outline performance evaluation metrics. Evidence: [docs/benchmarks.md](docs/benchmarks.md).
- ✅ T007 — Record the proposed repository layout. Evidence: [docs/layout.md](docs/layout.md).
- ✅ T008 — Draft the team schedule and effort estimate. Evidence: [docs/schedule.md](docs/schedule.md).
- ✅ T009 — Record prior work and research references. Evidence: [docs/references.md](docs/references.md).
- ✅ T010 — Document stretch goals and known pitfalls. Evidence: [docs/risks.md](docs/risks.md).
- ✅ T011 — Publish a README with accurate implementation status. Evidence: [README](README.md).
- ✅ T012 — Create contribution workflow guidance. Evidence: [.github/pull_request_template.md](.github/pull_request_template.md).
- ✅ T013 — Add an MIT license for original project material. Evidence: [docs/licensing.md](docs/licensing.md).
- ✅ T014 — Publish the phased task timeline. Evidence: [tools/audit_roadmap.py](tools/audit_roadmap.py).
- ✅ T015 — Agree on acceptance criteria and owners for the first implementation milestone. Evidence: [docs/milestone-1.md](docs/milestone-1.md).

### Phase 02: Toolchain and repository setup

Target: Weeks 1-2.

- ✅ T016 — Choose C++20 or Rust for the core. Evidence: [docs/core-language.md](docs/core-language.md).
- ✅ T017 — Record the core-language decision and rationale. Evidence: [docs/decisions/0001-core.md](docs/decisions/0001-core.md).
- ✅ T018 — Choose the Python binding approach. Evidence: [docs/decisions/0002-bindings.md](docs/decisions/0002-bindings.md).
- ✅ T019 — Choose the HTTP serving framework. Evidence: [docs/decisions/0003-server.md](docs/decisions/0003-server.md).
- ✅ T020 — Create the core source directory structure. Evidence: [.gitignore](.gitignore), [core/include/tvara/tensor.hpp](core/include/tvara/tensor.hpp), [core/src/version.cpp](core/src/version.cpp), [core/include/tvara/version.hpp](core/include/tvara/version.hpp).
- ✅ T021 — Create the Python reference package. Evidence: [pyproject.toml](pyproject.toml), [python/tvara/__init__.py](python/tvara/__init__.py), [python/tvara/reference/__init__.py](python/tvara/reference/__init__.py).
- ✅ T022 — Create the test directory structure. Evidence: [tests/core/test_tensor.cpp](tests/core/test_tensor.cpp), [tests/python/test_package.py](tests/python/test_package.py).
- ✅ T023 — Configure a reproducible core build. Evidence: [CMakeLists.txt](CMakeLists.txt).
- ✅ T024 — Add a debug build configuration. Evidence: [CMakePresets.json](CMakePresets.json).
- ✅ T025 — Add an optimized release configuration. Evidence: [CMakePresets.json](CMakePresets.json).
- ✅ T026 — Configure formatting and linting. Evidence: [ruff.toml](ruff.toml).
- ✅ T027 — Add compiler warnings and sanitizer settings. Evidence: [CMakeLists.txt](CMakeLists.txt).
- ✅ T028 — Define supported operating systems and CPU targets. Evidence: [docs/platforms.md](docs/platforms.md).
- ✅ T029 — Add a minimal CI build and smoke check. Evidence: [.github/workflows/correctness.yml](.github/workflows/correctness.yml).
- ✅ T030 — Document local development setup commands. Evidence: [docs/development.md](docs/development.md).

### Phase 03: Reference model and correctness harness

Target: Weeks 1-2.

- ✅ T031 — Select the first Llama-family reference model. Evidence: [docs/reference-model.md](docs/reference-model.md).
- ✅ T032 — Record model revision and weight access requirements. Evidence: [tests/fixtures/reference-model.json](tests/fixtures/reference-model.json).
- ✅ T033 — Create a deterministic reference-loading script. Evidence: [python/tvara/reference/oracle.py](python/tvara/reference/oracle.py), [tests/python/test_oracle.py](tests/python/test_oracle.py).
- ✅ T034 — Implement NumPy RMSNorm. Evidence: [python/tvara/reference/math.py](python/tvara/reference/math.py), [tests/python/test_reference_math.py](tests/python/test_reference_math.py).
- ✅ T035 — Implement NumPy rotary position embeddings. Evidence: [python/tvara/reference/math.py](python/tvara/reference/math.py), [tests/python/test_reference_math.py](tests/python/test_reference_math.py).
- ✅ T036 — Implement NumPy grouped-query attention. Evidence: [python/tvara/reference/math.py](python/tvara/reference/math.py), [tests/python/test_reference_math.py](tests/python/test_reference_math.py).
- ✅ T037 — Implement NumPy SwiGLU. Evidence: [python/tvara/reference/math.py](python/tvara/reference/math.py), [tests/python/test_reference_math.py](tests/python/test_reference_math.py).
- ✅ T038 — Assemble a reference transformer block. Evidence: [python/tvara/reference/model.py](python/tvara/reference/model.py), [tests/python/test_reference_model.py](tests/python/test_reference_model.py).
- ✅ T039 — Assemble the full reference forward pass. Evidence: [python/tvara/reference/model.py](python/tvara/reference/model.py), [tests/python/test_reference_model.py](tests/python/test_reference_model.py).
- ✅ T040 — Capture reference logits for fixed prompts. Evidence: [tools/capture_reference.py](tools/capture_reference.py).
- ✅ T041 — Define absolute and relative numerical tolerances. Evidence: [python/tvara/reference/metrics.py](python/tvara/reference/metrics.py), [tests/python/test_metrics.py](tests/python/test_metrics.py).
- ✅ T042 — Compare reference logits with Hugging Face. Evidence: [tests/python/test_reference_logits.py](tests/python/test_reference_logits.py).
- ✅ T043 — Create a greedy-generation reference fixture. Evidence: [python/tvara/reference/generation.py](python/tvara/reference/generation.py), [tools/capture_greedy.py](tools/capture_greedy.py), [tests/python/test_greedy.py](tests/python/test_greedy.py).
- ✅ T044 — Create a perplexity evaluation fixture. Evidence: [python/tvara/reference/metrics.py](python/tvara/reference/metrics.py), [tests/python/test_metrics.py](tests/python/test_metrics.py).
- ✅ T045 — Document reference precision and reproducibility settings. Evidence: [docs/reference-validation.md](docs/reference-validation.md).

### Phase 04: GGUF loading and tensor storage

Target: Weeks 3-4.

- ✅ T046 — Create a bounds-checked binary reader. Evidence: [core/include/tvara/gguf.hpp](core/include/tvara/gguf.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp), [CMakeLists.txt](CMakeLists.txt).
- ✅ T047 — Validate GGUF magic and supported versions. Evidence: [core/include/tvara/gguf.hpp](core/include/tvara/gguf.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T048 — Parse GGUF header counts. Evidence: [core/include/tvara/gguf.hpp](core/include/tvara/gguf.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T049 — Parse scalar metadata types. Evidence: [core/include/tvara/gguf.hpp](core/include/tvara/gguf.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T050 — Parse strings and metadata arrays. Evidence: [core/include/tvara/gguf.hpp](core/include/tvara/gguf.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T051 — Parse tensor names and dimensions. Evidence: [core/include/tvara/gguf.hpp](core/include/tvara/gguf.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T052 — Parse tensor types and offsets. Evidence: [core/include/tvara/gguf.hpp](core/include/tvara/gguf.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T053 — Validate tensor alignment. Evidence: [core/include/tvara/gguf.hpp](core/include/tvara/gguf.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T054 — Detect truncated files and invalid offsets. Evidence: [core/include/tvara/gguf.hpp](core/include/tvara/gguf.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T055 — Implement read-only memory mapping. Evidence: [core/include/tvara/mapped_file.hpp](core/include/tvara/mapped_file.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T056 — Represent tensors as typed storage views. Evidence: [core/include/tvara/gguf_model.hpp](core/include/tvara/gguf_model.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T057 — Validate expected model tensor names. Evidence: [core/include/tvara/gguf_model.hpp](core/include/tvara/gguf_model.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T058 — Map architecture metadata to a model configuration. Evidence: [core/include/tvara/gguf_model.hpp](core/include/tvara/gguf_model.hpp), [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T059 — Test malformed GGUF fixtures. Evidence: [tests/core/test_gguf.cpp](tests/core/test_gguf.cpp).
- ✅ T060 — Document loader errors and supported formats. Evidence: [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), [docs/gguf.md](docs/gguf.md).

### Phase 05: Quantization and scalar kernels

Target: Weeks 3-5.

- ✅ T061 — Implement F16-to-F32 conversion. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp), [CMakeLists.txt](CMakeLists.txt).
- ✅ T062 — Implement Q8_0 reference dequantization. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T063 — Implement Q4_0 reference dequantization. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T064 — Decode Q4_K sub-block scales and minima. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T065 — Implement Q4_K reference dequantization. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T066 — Implement per-block activation quantization. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T067 — Implement scalar F32 dot products. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T068 — Implement scalar Q8_0 dot products. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T069 — Implement scalar Q4_0 dot products. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T070 — Implement scalar Q4_K dot products. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T071 — Implement scalar matrix-vector multiplication. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T072 — Implement scalar matrix-matrix multiplication. Evidence: [core/include/tvara/quant.hpp](core/include/tvara/quant.hpp), [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T073 — Test zero and extreme quantized values. Evidence: [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T074 — Compare quantized kernels with dequantized references. Evidence: [tests/core/test_quant.cpp](tests/core/test_quant.cpp).
- ✅ T075 — Document block layouts and error tolerances. Evidence: [docs/quantization.md](docs/quantization.md).

### Phase 06: Tokenizers and text handling

Target: Weeks 3-5.

- ✅ T076 — Extract tokenizer metadata from GGUF. Evidence: [python/tvara/tokenizer.py](python/tvara/tokenizer.py), [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T077 — Define tokenizer encode and decode interfaces. Evidence: [python/tvara/tokenizer.py](python/tvara/tokenizer.py), [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T078 — Implement byte-level token mapping. Evidence: [python/tvara/tokenizer.py](python/tvara/tokenizer.py), [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T079 — Implement BPE merge ranking. Evidence: [python/tvara/tokenizer.py](python/tvara/tokenizer.py), [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T080 — Implement BPE token encoding. Evidence: [python/tvara/tokenizer.py](python/tvara/tokenizer.py), [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py), [pyproject.toml](pyproject.toml).
- ✅ T081 — Implement SentencePiece vocabulary loading. Evidence: [python/tvara/tokenizer.py](python/tvara/tokenizer.py), [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T082 — Implement SentencePiece token encoding. Evidence: [python/tvara/tokenizer.py](python/tvara/tokenizer.py), [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T083 — Implement special-token recognition. Evidence: [python/tvara/tokenizer.py](python/tvara/tokenizer.py), [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T084 — Implement byte-fallback decoding. Evidence: [python/tvara/tokenizer.py](python/tvara/tokenizer.py), [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T085 — Handle leading spaces consistently. Evidence: [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T086 — Handle Unicode and split UTF-8 sequences. Evidence: [python/tvara/tokenizer.py](python/tvara/tokenizer.py), [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T087 — Define beginning and end token policies. Evidence: [python/tvara/tokenizer.py](python/tvara/tokenizer.py), [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T088 — Compare token IDs with reference tokenizers. Evidence: [tests/fixtures/tokenizer-model.json](tests/fixtures/tokenizer-model.json), [tests/python/test_tokenizer_oracle.py](tests/python/test_tokenizer_oracle.py).
- ✅ T089 — Test Indic text emoji and whitespace fixtures. Evidence: [tests/python/test_tokenizer.py](tests/python/test_tokenizer.py).
- ✅ T090 — Document tokenizer compatibility limitations. Evidence: [docs/tokenizers.md](docs/tokenizers.md).

### Phase 07: Model executor and sampling

Target: Weeks 4-6.

- ✅ T091 — Define model and layer configuration structures. Evidence: [core/include/tvara/model.hpp](core/include/tvara/model.hpp), [tests/core/test_model.cpp](tests/core/test_model.cpp), [CMakeLists.txt](CMakeLists.txt).
- ✅ T092 — Implement embedding lookup. Evidence: [core/include/tvara/model.hpp](core/include/tvara/model.hpp), [tests/core/test_model.cpp](tests/core/test_model.cpp).
- ✅ T093 — Implement scalar RMSNorm in the core. Evidence: [core/include/tvara/model.hpp](core/include/tvara/model.hpp), [tests/core/test_model.cpp](tests/core/test_model.cpp).
- ✅ T094 — Implement scalar RoPE in the core. Evidence: [core/include/tvara/model.hpp](core/include/tvara/model.hpp), [tests/core/test_model.cpp](tests/core/test_model.cpp).
- ✅ T095 — Implement query key and value projections. Evidence: [core/include/tvara/model.hpp](core/include/tvara/model.hpp), [tests/core/test_model.cpp](tests/core/test_model.cpp).
- ✅ T096 — Implement grouped-query causal attention. Evidence: [core/include/tvara/model.hpp](core/include/tvara/model.hpp), [tests/core/test_model.cpp](tests/core/test_model.cpp).
- ✅ T097 — Implement attention output projection. Evidence: [core/include/tvara/model.hpp](core/include/tvara/model.hpp), [tests/core/test_model.cpp](tests/core/test_model.cpp).
- ✅ T098 — Implement SwiGLU feed-forward layers. Evidence: [core/include/tvara/model.hpp](core/include/tvara/model.hpp), [tests/core/test_model.cpp](tests/core/test_model.cpp).
- ✅ T099 — Implement residual connections and final projection. Evidence: [core/include/tvara/model.hpp](core/include/tvara/model.hpp), [tests/core/test_model.cpp](tests/core/test_model.cpp).
- ✅ T100 — Implement single-sequence prefill and decode. Evidence: [core/include/tvara/model.hpp](core/include/tvara/model.hpp), [tests/core/test_model.cpp](tests/core/test_model.cpp), [core/src/model_check.cpp](core/src/model_check.cpp), [CMakeLists.txt](CMakeLists.txt).
- ✅ T101 — Implement greedy and temperature sampling. Evidence: [core/include/tvara/sampling.hpp](core/include/tvara/sampling.hpp), [tests/core/test_sampling.cpp](tests/core/test_sampling.cpp), [CMakeLists.txt](CMakeLists.txt).
- ✅ T102 — Implement top-k top-p and min-p filters. Evidence: [core/include/tvara/sampling.hpp](core/include/tvara/sampling.hpp), [tests/core/test_sampling.cpp](tests/core/test_sampling.cpp).
- ✅ T103 — Implement repetition and frequency penalties. Evidence: [core/include/tvara/sampling.hpp](core/include/tvara/sampling.hpp), [tests/core/test_sampling.cpp](tests/core/test_sampling.cpp).
- ✅ T104 — Implement seeded generation and stopping rules. Evidence: [core/include/tvara/generation.hpp](core/include/tvara/generation.hpp), [tests/core/test_model.cpp](tests/core/test_model.cpp).
- ✅ T105 — Validate core logits and greedy output against references. Evidence: [tools/convert_reference_gguf.py](tools/convert_reference_gguf.py), [tests/python/test_native_logits.py](tests/python/test_native_logits.py), [README.md](README.md).

### Phase 08: SIMD kernels and CPU threading

Target: Weeks 6-8.

- ✅ T106 — Implement runtime CPU feature detection. Evidence: [core/include/tvara/cpu.hpp](core/include/tvara/cpu.hpp), [tests/core/test_simd.cpp](tests/core/test_simd.cpp), [CMakeLists.txt](CMakeLists.txt).
- ✅ T107 — Retain a portable scalar dispatch path. Evidence: [core/include/tvara/simd.hpp](core/include/tvara/simd.hpp), [tests/core/test_simd.cpp](tests/core/test_simd.cpp).
- ✅ T108 — Implement AVX2 Q8_0 dot products. Evidence: [core/include/tvara/simd.hpp](core/include/tvara/simd.hpp), [tests/core/test_simd.cpp](tests/core/test_simd.cpp).
- ✅ T109 — Implement AVX2 Q4_0 dot products. Evidence: [core/include/tvara/simd.hpp](core/include/tvara/simd.hpp), [tests/core/test_simd.cpp](tests/core/test_simd.cpp).
- ✅ T110 — Implement AVX2 Q4_K dot products. Evidence: [core/include/tvara/simd.hpp](core/include/tvara/simd.hpp), [tests/core/test_simd.cpp](tests/core/test_simd.cpp).
- ✅ T111 — Vectorize normalization and activation operations. Evidence: [core/include/tvara/simd.hpp](core/include/tvara/simd.hpp), [tests/core/test_simd.cpp](tests/core/test_simd.cpp).
- ✅ T112 — Implement vectorized RoPE. Evidence: [core/include/tvara/simd.hpp](core/include/tvara/simd.hpp), [tests/core/test_simd.cpp](tests/core/test_simd.cpp).
- ✅ T113 — Design a persistent worker thread pool. Evidence: [core/include/tvara/thread_pool.hpp](core/include/tvara/thread_pool.hpp), [tests/core/test_threads.cpp](tests/core/test_threads.cpp), [CMakeLists.txt](CMakeLists.txt).
- ✅ T114 — Implement row partitioning for matrix operations. Evidence: [core/include/tvara/thread_pool.hpp](core/include/tvara/thread_pool.hpp), [core/include/tvara/threaded_kernels.hpp](core/include/tvara/threaded_kernels.hpp), [tests/core/test_threads.cpp](tests/core/test_threads.cpp).
- ✅ T115 — Implement a spin-then-sleep worker barrier. Evidence: [core/include/tvara/thread_pool.hpp](core/include/tvara/thread_pool.hpp), [tests/core/test_threads.cpp](tests/core/test_threads.cpp).
- ✅ T116 — Allow configurable worker counts. Evidence: [core/include/tvara/thread_pool.hpp](core/include/tvara/thread_pool.hpp), [tests/core/test_threads.cpp](tests/core/test_threads.cpp).
- ✅ T117 — Validate SIMD results against scalar kernels. Evidence: [tests/core/test_simd.cpp](tests/core/test_simd.cpp).
- ✅ T118 — Test concurrent worker startup and shutdown. Evidence: [tests/core/test_threads.cpp](tests/core/test_threads.cpp).
- ✅ T119 — Measure dispatch and synchronization overhead. Evidence: [bench/kernel_overhead.cpp](bench/kernel_overhead.cpp), [tools/run_kernel_overhead.py](tools/run_kernel_overhead.py), [CMakeLists.txt](CMakeLists.txt).
- ✅ T120 — Document alignment and instruction-set requirements. Evidence: [docs/cpu-kernels.md](docs/cpu-kernels.md).

### Phase 09: Paged KV cache

Target: Weeks 6-8.

- ✅ T121 — Define KV block geometry and storage layout. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp), [CMakeLists.txt](CMakeLists.txt).
- ✅ T122 — Implement a physical block allocator. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T123 — Implement per-sequence logical block tables. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T124 — Append keys and values to allocated blocks. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T125 — Read cached values through block tables. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T126 — Track block reference counts. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T127 — Release blocks at sequence completion. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T128 — Implement shared-block ownership. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T129 — Implement copy-on-write for divergent sequences. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T130 — Handle partially filled tail blocks. Evidence: [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T131 — Implement truncation for generation rollback. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T132 — Handle cache exhaustion predictably. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T133 — Test allocation and release invariants. Evidence: [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp), [tests/core/test_kv.cpp](tests/core/test_kv.cpp).
- ✅ T134 — Compare cached decode with uncached reference output. Evidence: [core/include/tvara/paged_model.hpp](core/include/tvara/paged_model.hpp), [core/src/model_check.cpp](core/src/model_check.cpp), [tests/python/test_native_logits.py](tests/python/test_native_logits.py).
- ✅ T135 — Document capacity calculations and cache lifecycle. Evidence: [docs/kv-cache.md](docs/kv-cache.md).

### Phase 10: Grammar compilation and automaton

Target: Weeks 6-9.

- ✅ T136 — Define the supported grammar syntax. Evidence: [docs/grammar-syntax.md](docs/grammar-syntax.md).
- ✅ T137 — Implement grammar lexing. Evidence: [python/tvara/grammar.py](python/tvara/grammar.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T138 — Implement EBNF or GBNF parsing. Evidence: [python/tvara/grammar.py](python/tvara/grammar.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T139 — Represent productions and terminals. Evidence: [python/tvara/grammar.py](python/tvara/grammar.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T140 — Validate undefined and recursive rules. Evidence: [python/tvara/grammar.py](python/tvara/grammar.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T141 — Implement pushdown-automaton state transitions. Evidence: [python/tvara/grammar.py](python/tvara/grammar.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T142 — Implement JSON primitive grammars. Evidence: [python/tvara/grammar.py](python/tvara/grammar.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T143 — Compile object properties and required fields. Evidence: [python/tvara/schema.py](python/tvara/schema.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T144 — Compile bounded arrays and nested objects. Evidence: [python/tvara/schema.py](python/tvara/schema.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T145 — Compile enums and alternatives. Evidence: [python/tvara/schema.py](python/tvara/schema.py), [tests/python/test_grammar.py](tests/python/test_grammar.py), [pyproject.toml](pyproject.toml).
- ✅ T146 — Resolve local schema references. Evidence: [python/tvara/schema.py](python/tvara/schema.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T147 — Define string-format support and unsupported cases. Evidence: [python/tvara/schema.py](python/tvara/schema.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T148 — Implement accepting-state and end-token checks. Evidence: [python/tvara/grammar.py](python/tvara/grammar.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T149 — Test valid and invalid character sequences. Evidence: [python/tvara/schema.py](python/tvara/schema.py), [tests/python/test_grammar.py](tests/python/test_grammar.py).
- ✅ T150 — Document supported JSON Schema features. Evidence: [docs/structured-output.md](docs/structured-output.md).

### Phase 11: Streaming server and Python integration

Target: Weeks 7-9.

- ✅ T151 — Expose model loading through Python bindings. Evidence: [core/include/tvara/c_api.h](core/include/tvara/c_api.h), [core/src/c_api.cpp](core/src/c_api.cpp), [python/tvara/native.py](python/tvara/native.py), [tests/python/test_native_bindings.py](tests/python/test_native_bindings.py), [CMakeLists.txt](CMakeLists.txt).
- ✅ T152 — Expose tokenization through Python bindings. Evidence: [core/include/tvara/c_api.h](core/include/tvara/c_api.h), [core/src/c_api.cpp](core/src/c_api.cpp), [python/tvara/native.py](python/tvara/native.py), [tests/python/test_native_bindings.py](tests/python/test_native_bindings.py).
- ✅ T153 — Expose generation through Python bindings. Evidence: [core/include/tvara/c_api.h](core/include/tvara/c_api.h), [core/src/c_api.cpp](core/src/c_api.cpp), [python/tvara/native.py](python/tvara/native.py), [tests/python/test_native_bindings.py](tests/python/test_native_bindings.py).
- ✅ T154 — Define server configuration and startup behavior. Evidence: [python/tvara/server.py](python/tvara/server.py), [tests/python/test_server.py](tests/python/test_server.py).
- ✅ T155 — Implement model health and readiness checks. Evidence: [python/tvara/server.py](python/tvara/server.py), [tests/python/test_server.py](tests/python/test_server.py).
- ✅ T156 — Implement the completions request contract. Evidence: [python/tvara/server.py](python/tvara/server.py), [tests/python/test_server.py](tests/python/test_server.py).
- ✅ T157 — Implement the chat-completions request contract. Evidence: [python/tvara/server.py](python/tvara/server.py), [tests/python/test_server.py](tests/python/test_server.py).
- ✅ T158 — Implement nonstreaming response serialization. Evidence: [python/tvara/server.py](python/tvara/server.py), [tests/python/test_server.py](tests/python/test_server.py).
- ✅ T159 — Implement server-sent event streaming. Evidence: [python/tvara/server.py](python/tvara/server.py), [tests/python/test_server.py](tests/python/test_server.py).
- ✅ T160 — Implement stream completion markers. Evidence: [python/tvara/server.py](python/tvara/server.py), [tests/python/test_server.py](tests/python/test_server.py).
- ✅ T161 — Validate request fields and generation limits. Evidence: [python/tvara/server.py](python/tvara/server.py), [tests/python/test_server.py](tests/python/test_server.py).
- ✅ T162 — Handle client disconnects and cancellation. Evidence: [python/tvara/server.py](python/tvara/server.py), [tests/python/test_server.py](tests/python/test_server.py).
- ✅ T163 — Return consistent request errors. Evidence: [python/tvara/server.py](python/tvara/server.py), [tests/python/test_server.py](tests/python/test_server.py).
- ✅ T164 — Add local client integration checks. Evidence: [tools/make_api_fixture.py](tools/make_api_fixture.py), [tools/smoke_api.py](tools/smoke_api.py).
- ✅ T165 — Document runnable API examples after validation. Evidence: [docs/api.md](docs/api.md), [README.md](README.md).

### Phase 12: Prefill and attention optimization

Target: Weeks 9-11.

- ✅ T166 — Create separate prefill and decode benchmark cases. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T166.json](bench/results/prefill/T166.json)..
- ✅ T167 — Implement cache-blocked prefill matrix multiplication. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T167.json](bench/results/prefill/T167.json)..
- ✅ T168 — Reuse quantized activation tiles. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T168.json](bench/results/prefill/T168.json)..
- ✅ T169 — Tune prefill tile dimensions. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T169.json](bench/results/prefill/T169.json)..
- ✅ T170 — Implement online softmax reference attention. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T170.json](bench/results/prefill/T170.json)..
- ✅ T171 — Implement fused paged attention. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T171.json](bench/results/prefill/T171.json)..
- ✅ T172 — Support grouped-query heads in fused attention. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T172.json](bench/results/prefill/T172.json)..
- ✅ T173 — Handle causal masks within prompt chunks. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T173.json](bench/results/prefill/T173.json)..
- ✅ T174 — Handle partial KV blocks in attention. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T174.json](bench/results/prefill/T174.json)..
- ✅ T175 — Vectorize attention reductions. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T175.json](bench/results/prefill/T175.json)..
- ✅ T176 — Add weight and cache prefetch experiments. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T176.json](bench/results/prefill/T176.json)..
- ✅ T177 — Measure numerical error across long contexts. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T177.json](bench/results/prefill/T177.json)..
- ✅ T178 — Compare fused attention with unfused output. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T178.json](bench/results/prefill/T178.json)..
- ✅ T179 — Measure prefill throughput across prompt lengths. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T179.json](bench/results/prefill/T179.json)..
- ✅ T180 — Record accepted optimizations with supporting measurements. Evidence: Native numerical assertions and measured evidence: [bench/results/prefill/T180.json](bench/results/prefill/T180.json)..

### Phase 13: Radix-tree prefix reuse

Target: Weeks 9-11.

- ✅ T181 — Define radix node and edge structures. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp), [CMakeLists.txt](CMakeLists.txt).
- ✅ T182 — Implement longest-prefix token lookup. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp).
- ✅ T183 — Implement insertion of token sequences. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp).
- ✅ T184 — Implement edge splitting at divergence. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp).
- ✅ T185 — Attach KV block references to cache nodes. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp).
- ✅ T186 — Reuse full shared prefix blocks. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp), [core/include/tvara/kv.hpp](core/include/tvara/kv.hpp).
- ✅ T187 — Handle partial-block reuse safely. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp).
- ✅ T188 — Pin nodes used by active sequences. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp).
- ✅ T189 — Implement leaf access tracking. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp).
- ✅ T190 — Implement LRU leaf eviction. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp).
- ✅ T191 — Release KV references after eviction. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp).
- ✅ T192 — Coordinate eviction with copy-on-write. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp).
- ✅ T193 — Test repeated and divergent conversations. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp).
- ✅ T194 — Measure hit rate and saved prefill tokens. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp), [bench/radix_prefix.cpp](bench/radix_prefix.cpp), [tools/run_radix_benchmark.py](tools/run_radix_benchmark.py).
- ✅ T195 — Document prefix isolation and cache reset behavior. Evidence: [core/include/tvara/radix.hpp](core/include/tvara/radix.hpp), [tests/core/test_radix.cpp](tests/core/test_radix.cpp), [docs/prefix-cache.md](docs/prefix-cache.md).

### Phase 14: Continuous batching and load evaluation

Target: Weeks 10-14.

- ✅ T196 — Define request lifecycle states. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T197 — Implement queued request admission. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T198 — Build iteration-level decode batches. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T199 — Remove completed sequences between iterations. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T200 — Implement configurable prefill chunks. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T201 — Interleave prefill chunks with decode. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T202 — Implement KV-aware admission control. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T203 — Define request fairness and priority rules. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T204 — Implement preemption under memory pressure. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T205 — Implement safe request resumption. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T206 — Build a synthetic conversation load generator. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T207 — Record time to first token. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T208 — Record inter-token latency percentiles. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py).
- ✅ T209 — Compare static and continuous batching throughput. Evidence: [tools/benchmark_scheduler.py](tools/benchmark_scheduler.py).
- ✅ T210 — Evaluate cancellation fairness and starvation under load. Evidence: [python/tvara/scheduler.py](python/tvara/scheduler.py), [tests/python/test_scheduler.py](tests/python/test_scheduler.py), [docs/scheduler.md](docs/scheduler.md).

### Phase 15: Grammar masks and structured API output

Target: Weeks 10-14.

- ✅ T211 — Build a trie over vocabulary token bytes. Evidence: [python/tvara/token_masks.py](python/tvara/token_masks.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T212 — Traverse the trie with grammar states. Evidence: [python/tvara/token_masks.py](python/tvara/token_masks.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T213 — Prune invalid vocabulary subtrees. Evidence: [python/tvara/token_masks.py](python/tvara/token_masks.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T214 — Represent allowed tokens as bitmasks. Evidence: [python/tvara/token_masks.py](python/tvara/token_masks.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T215 — Classify context-independent mask cases. Evidence: [python/tvara/token_masks.py](python/tvara/token_masks.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T216 — Precompute reusable grammar masks. Evidence: [python/tvara/token_masks.py](python/tvara/token_masks.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T217 — Cache masks by relevant automaton state. Evidence: [python/tvara/token_masks.py](python/tvara/token_masks.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T218 — Handle stack-dependent validity checks. Evidence: [python/tvara/token_masks.py](python/tvara/token_masks.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T219 — Preserve state across split UTF-8 tokens. Evidence: [python/tvara/token_masks.py](python/tvara/token_masks.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T220 — Integrate masks with sampling filters. Evidence: [python/tvara/native.py](python/tvara/native.py), [tests/python/test_native_bindings.py](tests/python/test_native_bindings.py), [python/tvara/token_masks.py](python/tvara/token_masks.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T221 — Implement JSON Schema response-format requests. Evidence: [tests/python/test_server.py](tests/python/test_server.py), [python/tvara/response_constraints.py](python/tvara/response_constraints.py), [python/tvara/server.py](python/tvara/server.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T222 — Apply GGUF chat templates. Evidence: [python/tvara/chat_templates.py](python/tvara/chat_templates.py), [python/tvara/server.py](python/tvara/server.py), [pyproject.toml](pyproject.toml), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T223 — Translate tool signatures into output constraints. Evidence: [python/tvara/response_constraints.py](python/tvara/response_constraints.py), [python/tvara/server.py](python/tvara/server.py), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).
- ✅ T224 — Run schema-compliance and mask-latency benchmarks. Evidence: [tools/benchmark_grammar_masks.py](tools/benchmark_grammar_masks.py), [docs/structured-output.md](docs/structured-output.md).
- ✅ T225 — Compare mask caching with a naive vocabulary scan. Evidence: [tools/benchmark_grammar_masks.py](tools/benchmark_grammar_masks.py), [docs/structured-output.md](docs/structured-output.md), [tests/python/test_token_masks.py](tests/python/test_token_masks.py).

### Phase 16: Speculative decoding

Target: Weeks 12-16.

- ✅ T226 — Define draft and target model interfaces. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py).
- ✅ T227 — Validate draft-target tokenizer compatibility. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py).
- ✅ T228 — Implement draft-token proposal generation. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py).
- ✅ T229 — Implement batched target verification. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py), [core/include/tvara/c_api.h](core/include/tvara/c_api.h), [core/src/c_api.cpp](core/src/c_api.cpp).
- ✅ T230 — Implement probabilistic draft acceptance. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py).
- ✅ T231 — Implement normalized residual sampling. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py).
- ✅ T232 — Handle zero-probability numerical edge cases. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py).
- ✅ T233 — Sample the extra target token after full acceptance. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py).
- ✅ T234 — Rollback target KV state after rejection. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py), [core/include/tvara/c_api.h](core/include/tvara/c_api.h), [core/src/c_api.cpp](core/src/c_api.cpp).
- ✅ T235 — Rollback draft KV state after rejection. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py).
- ✅ T236 — Integrate grammar validity with draft proposals. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py).
- ✅ T237 — Implement n-gram prompt-lookup proposals. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py).
- ✅ T238 — Implement adaptive draft length. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py).
- ✅ T239 — Run statistical distribution-equivalence checks. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py), [tools/check_speculative_distribution.py](tools/check_speculative_distribution.py).
- ✅ T240 — Measure acceptance rate and speed across task types. Evidence: [python/tvara/speculative.py](python/tvara/speculative.py), [tests/python/test_speculative.py](tests/python/test_speculative.py), [tools/bench_speculative.py](tools/bench_speculative.py), [docs/speculative-decoding.md](docs/speculative-decoding.md).

### Phase 17: Model variants and portability stretch work

Target: Weeks 15-16; stretch work after core validation.

- ✅ T241 — Implement Qwen-specific configuration mapping. Evidence: [python/tvara/reference/variants.py](python/tvara/reference/variants.py), [tests/python/test_variant_t241.py](tests/python/test_variant_t241.py).
- ✅ T242 — Validate Qwen normalization and rotary behavior. Evidence: [tests/python/test_variant_t242.py](tests/python/test_variant_t242.py).
- ✅ T243 — Validate Qwen tokenization and special tokens. Evidence: [python/tvara/qwen.py](python/tvara/qwen.py), [tests/python/test_variant_t243.py](tests/python/test_variant_t243.py).
- ✅ T244 — Implement Gemma-specific configuration mapping. Evidence: [python/tvara/reference/variants.py](python/tvara/reference/variants.py), [tests/python/test_variant_t244.py](tests/python/test_variant_t244.py).
- ✅ T245 — Implement Gemma-specific normalization behavior. Evidence: [python/tvara/reference/variants.py](python/tvara/reference/variants.py), [tests/python/test_variant_t245.py](tests/python/test_variant_t245.py).
- ✅ T246 — Validate Gemma logits against a reference. Evidence: [python/tvara/reference/variants.py](python/tvara/reference/variants.py), [docs/variants.md](docs/variants.md), [tests/python/test_variant_t246.py](tests/python/test_variant_t246.py).
- ✅ T247 — Implement optional AVX-512 kernel dispatch. Evidence: Executable numerical checks and capability evidence: [bench/results/variants/T247.json](bench/results/variants/T247.json)..
- ❌ T248 — Benchmark AVX-512 against AVX2 on supported hardware. Evidence: Pending: current host lacks AVX512F/BW. Guarded executable benchmark and capability result in [T248](bench/results/variants/T248.json); no supported-hardware measurement claimed..
- ✅ T249 — Implement ARM NEON quantized dot products. Evidence: Executable numerical checks and capability evidence: [bench/results/variants/T249.json](bench/results/variants/T249.json)..
- ❌ T250 — Validate an ARM build and reference output. Evidence: Pending: ARM compiler/emulator/hardware unavailable. [Validation harness](tests/core/variant_t250.cpp) and [local capability evidence](bench/results/variants/T250.json) document the blocker..
- ✅ T251 — Prototype quantized KV storage. Evidence: [python/tvara/quantized_kv.py](python/tvara/quantized_kv.py), [tests/python/test_variant_t251.py](tests/python/test_variant_t251.py).
- ✅ T252 — Measure quantized KV quality and memory tradeoffs. Evidence: [tools/bench_quantized_kv.py](tools/bench_quantized_kv.py), [docs/quantized-kv.md](docs/quantized-kv.md).
- ❌ T253 — Prototype mixture-of-experts routing.
- ❌ T254 — Prototype tree-based speculative verification.
- ❌ T255 — Assess WebAssembly SIMD build feasibility.

### Phase 18: Final evaluation documentation and release

Target: Weeks 17-18.

- ❌ T256 — Pin benchmark models and quantized file checksums.
- ❌ T257 — Record CPU memory OS and compiler configuration.
- ❌ T258 — Measure STREAM memory bandwidth on the test machine.
- ❌ T259 — Run matched llama.cpp prefill and decode baselines.
- ❌ T260 — Report repeated measurements and thermal conditions.
- ❌ T261 — Run the final perplexity and greedy regression suite.
- ❌ T262 — Measure prefix-cache effects on conversation traces.
- ❌ T263 — Measure scheduler latency at multiple concurrency levels.
- ❌ T264 — Measure speculative speedups and statistical results.
- ❌ T265 — Run final structured-output compliance evaluation.
- ❌ T266 — Publish performance tables with limitations.
- ❌ T267 — Write architecture kernel and sampling design notes.
- ❌ T268 — Record a reproducible end-to-end demo.
- ❌ T269 — Review dependency licenses and release contents.
- ❌ T270 — Tag a release with setup instructions and known limitations.
