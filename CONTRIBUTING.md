# Contributing to Tvara

Tvara is currently in the planning stage. Read [README.md](README.md) and the
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

Current status: 38 tasks are completed and 232 are not completed.
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
- ❌ T039 — Assemble the full reference forward pass.
- ❌ T040 — Capture reference logits for fixed prompts.
- ❌ T041 — Define absolute and relative numerical tolerances.
- ❌ T042 — Compare reference logits with Hugging Face.
- ❌ T043 — Create a greedy-generation reference fixture.
- ❌ T044 — Create a perplexity evaluation fixture.
- ❌ T045 — Document reference precision and reproducibility settings.

### Phase 04: GGUF loading and tensor storage

Target: Weeks 3-4.

- ❌ T046 — Create a bounds-checked binary reader.
- ❌ T047 — Validate GGUF magic and supported versions.
- ❌ T048 — Parse GGUF header counts.
- ❌ T049 — Parse scalar metadata types.
- ❌ T050 — Parse strings and metadata arrays.
- ❌ T051 — Parse tensor names and dimensions.
- ❌ T052 — Parse tensor types and offsets.
- ❌ T053 — Validate tensor alignment.
- ❌ T054 — Detect truncated files and invalid offsets.
- ❌ T055 — Implement read-only memory mapping.
- ❌ T056 — Represent tensors as typed storage views.
- ❌ T057 — Validate expected model tensor names.
- ❌ T058 — Map architecture metadata to a model configuration.
- ❌ T059 — Test malformed GGUF fixtures.
- ❌ T060 — Document loader errors and supported formats.

### Phase 05: Quantization and scalar kernels

Target: Weeks 3-5.

- ❌ T061 — Implement F16-to-F32 conversion.
- ❌ T062 — Implement Q8_0 reference dequantization.
- ❌ T063 — Implement Q4_0 reference dequantization.
- ❌ T064 — Decode Q4_K sub-block scales and minima.
- ❌ T065 — Implement Q4_K reference dequantization.
- ❌ T066 — Implement per-block activation quantization.
- ❌ T067 — Implement scalar F32 dot products.
- ❌ T068 — Implement scalar Q8_0 dot products.
- ❌ T069 — Implement scalar Q4_0 dot products.
- ❌ T070 — Implement scalar Q4_K dot products.
- ❌ T071 — Implement scalar matrix-vector multiplication.
- ❌ T072 — Implement scalar matrix-matrix multiplication.
- ❌ T073 — Test zero and extreme quantized values.
- ❌ T074 — Compare quantized kernels with dequantized references.
- ❌ T075 — Document block layouts and error tolerances.

### Phase 06: Tokenizers and text handling

Target: Weeks 3-5.

- ❌ T076 — Extract tokenizer metadata from GGUF.
- ❌ T077 — Define tokenizer encode and decode interfaces.
- ❌ T078 — Implement byte-level token mapping.
- ❌ T079 — Implement BPE merge ranking.
- ❌ T080 — Implement BPE token encoding.
- ❌ T081 — Implement SentencePiece vocabulary loading.
- ❌ T082 — Implement SentencePiece token encoding.
- ❌ T083 — Implement special-token recognition.
- ❌ T084 — Implement byte-fallback decoding.
- ❌ T085 — Handle leading spaces consistently.
- ❌ T086 — Handle Unicode and split UTF-8 sequences.
- ❌ T087 — Define beginning and end token policies.
- ❌ T088 — Compare token IDs with reference tokenizers.
- ❌ T089 — Test Indic text emoji and whitespace fixtures.
- ❌ T090 — Document tokenizer compatibility limitations.

### Phase 07: Model executor and sampling

Target: Weeks 4-6.

- ❌ T091 — Define model and layer configuration structures.
- ❌ T092 — Implement embedding lookup.
- ❌ T093 — Implement scalar RMSNorm in the core.
- ❌ T094 — Implement scalar RoPE in the core.
- ❌ T095 — Implement query key and value projections.
- ❌ T096 — Implement grouped-query causal attention.
- ❌ T097 — Implement attention output projection.
- ❌ T098 — Implement SwiGLU feed-forward layers.
- ❌ T099 — Implement residual connections and final projection.
- ❌ T100 — Implement single-sequence prefill and decode.
- ❌ T101 — Implement greedy and temperature sampling.
- ❌ T102 — Implement top-k top-p and min-p filters.
- ❌ T103 — Implement repetition and frequency penalties.
- ❌ T104 — Implement seeded generation and stopping rules.
- ❌ T105 — Validate core logits and greedy output against references.

### Phase 08: SIMD kernels and CPU threading

Target: Weeks 6-8.

- ❌ T106 — Implement runtime CPU feature detection.
- ❌ T107 — Retain a portable scalar dispatch path.
- ❌ T108 — Implement AVX2 Q8_0 dot products.
- ❌ T109 — Implement AVX2 Q4_0 dot products.
- ❌ T110 — Implement AVX2 Q4_K dot products.
- ❌ T111 — Vectorize normalization and activation operations.
- ❌ T112 — Implement vectorized RoPE.
- ❌ T113 — Design a persistent worker thread pool.
- ❌ T114 — Implement row partitioning for matrix operations.
- ❌ T115 — Implement a spin-then-sleep worker barrier.
- ❌ T116 — Allow configurable worker counts.
- ❌ T117 — Validate SIMD results against scalar kernels.
- ❌ T118 — Test concurrent worker startup and shutdown.
- ❌ T119 — Measure dispatch and synchronization overhead.
- ❌ T120 — Document alignment and instruction-set requirements.

### Phase 09: Paged KV cache

Target: Weeks 6-8.

- ❌ T121 — Define KV block geometry and storage layout.
- ❌ T122 — Implement a physical block allocator.
- ❌ T123 — Implement per-sequence logical block tables.
- ❌ T124 — Append keys and values to allocated blocks.
- ❌ T125 — Read cached values through block tables.
- ❌ T126 — Track block reference counts.
- ❌ T127 — Release blocks at sequence completion.
- ❌ T128 — Implement shared-block ownership.
- ❌ T129 — Implement copy-on-write for divergent sequences.
- ❌ T130 — Handle partially filled tail blocks.
- ❌ T131 — Implement truncation for generation rollback.
- ❌ T132 — Handle cache exhaustion predictably.
- ❌ T133 — Test allocation and release invariants.
- ❌ T134 — Compare cached decode with uncached reference output.
- ❌ T135 — Document capacity calculations and cache lifecycle.

### Phase 10: Grammar compilation and automaton

Target: Weeks 6-9.

- ❌ T136 — Define the supported grammar syntax.
- ❌ T137 — Implement grammar lexing.
- ❌ T138 — Implement EBNF or GBNF parsing.
- ❌ T139 — Represent productions and terminals.
- ❌ T140 — Validate undefined and recursive rules.
- ❌ T141 — Implement pushdown-automaton state transitions.
- ❌ T142 — Implement JSON primitive grammars.
- ❌ T143 — Compile object properties and required fields.
- ❌ T144 — Compile bounded arrays and nested objects.
- ❌ T145 — Compile enums and alternatives.
- ❌ T146 — Resolve local schema references.
- ❌ T147 — Define string-format support and unsupported cases.
- ❌ T148 — Implement accepting-state and end-token checks.
- ❌ T149 — Test valid and invalid character sequences.
- ❌ T150 — Document supported JSON Schema features.

### Phase 11: Streaming server and Python integration

Target: Weeks 7-9.

- ❌ T151 — Expose model loading through Python bindings.
- ❌ T152 — Expose tokenization through Python bindings.
- ❌ T153 — Expose generation through Python bindings.
- ❌ T154 — Define server configuration and startup behavior.
- ❌ T155 — Implement model health and readiness checks.
- ❌ T156 — Implement the completions request contract.
- ❌ T157 — Implement the chat-completions request contract.
- ❌ T158 — Implement nonstreaming response serialization.
- ❌ T159 — Implement server-sent event streaming.
- ❌ T160 — Implement stream completion markers.
- ❌ T161 — Validate request fields and generation limits.
- ❌ T162 — Handle client disconnects and cancellation.
- ❌ T163 — Return consistent request errors.
- ❌ T164 — Add local client integration checks.
- ❌ T165 — Document runnable API examples after validation.

### Phase 12: Prefill and attention optimization

Target: Weeks 9-11.

- ❌ T166 — Create separate prefill and decode benchmark cases.
- ❌ T167 — Implement cache-blocked prefill matrix multiplication.
- ❌ T168 — Reuse quantized activation tiles.
- ❌ T169 — Tune prefill tile dimensions.
- ❌ T170 — Implement online softmax reference attention.
- ❌ T171 — Implement fused paged attention.
- ❌ T172 — Support grouped-query heads in fused attention.
- ❌ T173 — Handle causal masks within prompt chunks.
- ❌ T174 — Handle partial KV blocks in attention.
- ❌ T175 — Vectorize attention reductions.
- ❌ T176 — Add weight and cache prefetch experiments.
- ❌ T177 — Measure numerical error across long contexts.
- ❌ T178 — Compare fused attention with unfused output.
- ❌ T179 — Measure prefill throughput across prompt lengths.
- ❌ T180 — Record accepted optimizations with supporting measurements.

### Phase 13: Radix-tree prefix reuse

Target: Weeks 9-11.

- ❌ T181 — Define radix node and edge structures.
- ❌ T182 — Implement longest-prefix token lookup.
- ❌ T183 — Implement insertion of token sequences.
- ❌ T184 — Implement edge splitting at divergence.
- ❌ T185 — Attach KV block references to cache nodes.
- ❌ T186 — Reuse full shared prefix blocks.
- ❌ T187 — Handle partial-block reuse safely.
- ❌ T188 — Pin nodes used by active sequences.
- ❌ T189 — Implement leaf access tracking.
- ❌ T190 — Implement LRU leaf eviction.
- ❌ T191 — Release KV references after eviction.
- ❌ T192 — Coordinate eviction with copy-on-write.
- ❌ T193 — Test repeated and divergent conversations.
- ❌ T194 — Measure hit rate and saved prefill tokens.
- ❌ T195 — Document prefix isolation and cache reset behavior.

### Phase 14: Continuous batching and load evaluation

Target: Weeks 10-14.

- ❌ T196 — Define request lifecycle states.
- ❌ T197 — Implement queued request admission.
- ❌ T198 — Build iteration-level decode batches.
- ❌ T199 — Remove completed sequences between iterations.
- ❌ T200 — Implement configurable prefill chunks.
- ❌ T201 — Interleave prefill chunks with decode.
- ❌ T202 — Implement KV-aware admission control.
- ❌ T203 — Define request fairness and priority rules.
- ❌ T204 — Implement preemption under memory pressure.
- ❌ T205 — Implement safe request resumption.
- ❌ T206 — Build a synthetic conversation load generator.
- ❌ T207 — Record time to first token.
- ❌ T208 — Record inter-token latency percentiles.
- ❌ T209 — Compare static and continuous batching throughput.
- ❌ T210 — Evaluate cancellation fairness and starvation under load.

### Phase 15: Grammar masks and structured API output

Target: Weeks 10-14.

- ❌ T211 — Build a trie over vocabulary token bytes.
- ❌ T212 — Traverse the trie with grammar states.
- ❌ T213 — Prune invalid vocabulary subtrees.
- ❌ T214 — Represent allowed tokens as bitmasks.
- ❌ T215 — Classify context-independent mask cases.
- ❌ T216 — Precompute reusable grammar masks.
- ❌ T217 — Cache masks by relevant automaton state.
- ❌ T218 — Handle stack-dependent validity checks.
- ❌ T219 — Preserve state across split UTF-8 tokens.
- ❌ T220 — Integrate masks with sampling filters.
- ❌ T221 — Implement JSON Schema response-format requests.
- ❌ T222 — Apply GGUF chat templates.
- ❌ T223 — Translate tool signatures into output constraints.
- ❌ T224 — Run schema-compliance and mask-latency benchmarks.
- ❌ T225 — Compare mask caching with a naive vocabulary scan.

### Phase 16: Speculative decoding

Target: Weeks 12-16.

- ❌ T226 — Define draft and target model interfaces.
- ❌ T227 — Validate draft-target tokenizer compatibility.
- ❌ T228 — Implement draft-token proposal generation.
- ❌ T229 — Implement batched target verification.
- ❌ T230 — Implement probabilistic draft acceptance.
- ❌ T231 — Implement normalized residual sampling.
- ❌ T232 — Handle zero-probability numerical edge cases.
- ❌ T233 — Sample the extra target token after full acceptance.
- ❌ T234 — Rollback target KV state after rejection.
- ❌ T235 — Rollback draft KV state after rejection.
- ❌ T236 — Integrate grammar validity with draft proposals.
- ❌ T237 — Implement n-gram prompt-lookup proposals.
- ❌ T238 — Implement adaptive draft length.
- ❌ T239 — Run statistical distribution-equivalence checks.
- ❌ T240 — Measure acceptance rate and speed across task types.

### Phase 17: Model variants and portability stretch work

Target: Weeks 15-16; stretch work after core validation.

- ❌ T241 — Implement Qwen-specific configuration mapping.
- ❌ T242 — Validate Qwen normalization and rotary behavior.
- ❌ T243 — Validate Qwen tokenization and special tokens.
- ❌ T244 — Implement Gemma-specific configuration mapping.
- ❌ T245 — Implement Gemma-specific normalization behavior.
- ❌ T246 — Validate Gemma logits against a reference.
- ❌ T247 — Implement optional AVX-512 kernel dispatch.
- ❌ T248 — Benchmark AVX-512 against AVX2 on supported hardware.
- ❌ T249 — Implement ARM NEON quantized dot products.
- ❌ T250 — Validate an ARM build and reference output.
- ❌ T251 — Prototype quantized KV storage.
- ❌ T252 — Measure quantized KV quality and memory tradeoffs.
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
