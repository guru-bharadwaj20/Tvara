# WebAssembly SIMD feasibility

The portable tensor, quantization and scalar reference kernels use C++20 spans
and standard containers. A standalone Q8 dot-product target is a practical
first port: compile with Emscripten `-std=c++20 -O3 -msimd128`, retain a scalar
reference, and compare signed-byte widening products and reductions before
timing. `core/include/tvara/simd.hpp` already excludes x86 intrinsics when the
target is not x86. An actual wasm SIMD implementation and emitted-instruction
audit are future work; native AVX2 tests cannot establish wasm correctness.

The complete runtime needs additional decisions: `mapped_file.hpp` only defines
Windows/POSIX mappings, which need replacement or virtual filesystem support;
threaded kernels need `-pthread` and browser SharedArrayBuffer availability
with cross-origin isolation; model downloads and linear-memory capacity need
bounded allocation and explicit user-facing progress. A scalar single-thread
build is the fallback when SIMD or browser threading is unavailable.

`tools/probe_wasm.py` records actual local Emscripten availability in
`bench/results/variants/wasm-feasibility.json`. No Emscripten toolchain is
installed on the current host, so this assessment does not claim a successful
WebAssembly build or browser benchmark. Next acceptance steps are a standalone
kernel cross-build, wasm runtime numerical comparison, SIMD opcode audit, and
browser measurement before attempting full inference.
