# ARM build and output validation

The NEON kernel uses baseline widening multiplication (`vmull_s8`) and pairwise
32-bit accumulation; it does not require ARM dot-product extensions. No sum is
accumulated in a 16-bit lane beyond a single signed-byte product.

On ARM with GCC/Clang and NEON enabled, run `python tools/check_prefill.py
tests/core/variant_t250.cpp bench/results/variants/T250.json`. To select a
compiler, set `CXX` to its executable. The test compares 500 adversarial and
deterministic int8 vectors and Q8 block outputs to portable scalar references.
For a complete project build, use the normal CMake build and CTest on ARM.

The available machine is Windows x86_64 and has no ARM compiler or emulator.
The local harness therefore records `blocked`; T250 remains incomplete.
The portable fallback's success is not presented as ARM instruction validation.
