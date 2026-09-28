# Core language selection

Use C++20 for the core. Build one portable library and dispatch optional instruction
sets at runtime. Python remains the numerical oracle and API integration layer.
Keep ownership explicit with RAII, spans for borrowed contiguous data, and checked
dimension arithmetic at storage boundaries. Exceptions stop at the C ABI boundary.

This choice prioritizes direct SIMD intrinsics, predictable memory layout, and the
available GCC toolchain. Rust would also be viable, but maintaining two core
implementations would split validation effort. No Rust core is planned.
