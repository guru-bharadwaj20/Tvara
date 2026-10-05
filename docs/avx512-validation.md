# AVX512 validation

The optional Q8 kernel requires both AVX512F and AVX512BW. It is compiled as a
target-specific function and selected only with `TVARA_ENABLE_AVX512=1`, when
the host supports both instruction sets and scalar forcing is disabled.

Run `python tools/check_prefill.py tests/core/variant_t248.cpp
bench/results/variants/T248.json` on supported hardware. The harness compares
against AVX2 and scalar outputs before timing and records median microseconds.
On the current AVX2-only x86 host it records `blocked`, without running AVX512.
T248 remains incomplete until the supported-hardware benchmark actually runs.
