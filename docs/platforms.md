# Platform targets

Initial targets are Windows x86-64 with MinGW GCC, Linux x86-64 with GCC/Clang,
and Windows x86-64 with MSVC. Local checks currently exercise MinGW only; CI will
exercise Linux GCC. MSVC is a target, not yet a locally validated build.

C++20 and CMake 3.24+ are required. Python tooling requires Python 3.11+ and NumPy.
No GPU, CUDA, or AVX2 support is required for portable tests. Future AVX2 and AVX-512
implementations must use runtime dispatch and retain a scalar fallback.

Address/undefined sanitizers are enabled explicitly on Linux GCC/Clang. Windows
sanitizer builds are rejected rather than silently running without instrumentation.
ARM NEON and WebAssembly remain stretch targets until tested builds exist.
