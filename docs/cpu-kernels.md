# CPU kernels and threading

The required path uses scalar C++20. GCC/Clang x86 builds detect AVX2 support at
runtime and isolate AVX2 instructions in target-annotated functions. Q8_0, Q4_0,
and Q4_K integer dot products have AVX2 implementations. Normalization, SwiGLU,
and RoPE vectorize arithmetic while retaining scalar tails and scalar transcendental
functions. An AVX512F/BW Q8_0 dot product is selected only by
OptionalKernelDispatch with TVARA_ENABLE_AVX512=1 on supporting hardware; it has
not yet run on such hardware ([avx512-validation.md](avx512-validation.md)).
A NEON Q8_0 dot product is a standalone function compiled when __ARM_NEON is
defined, unvalidated on ARM ([arm-validation.md](arm-validation.md)).

TVARA_FORCE_SCALAR=1 disables dispatch. Direct AVX2 functions require a successful
feature check; callers normally use KernelDispatch. Public inputs use unaligned
loads; temporary reduction buffers are 32-byte aligned. Non-x86 and unimplemented
compiler-dispatch paths retain scalar behavior. No global -march=native is used.

The persistent pool supports 1–256 workers, serializes simultaneous submissions,
captures worker exceptions, and rejects execution nested into the same pool.
Destruction requires callers to have finished submitting work. Matrix rows are
partitioned without overlaps, including when there are more workers than rows.
The reusable barrier spins for 64 yields before waiting on a condition variable.

Local checks compare SIMD with scalar outputs and exercise worker reuse, shutdown,
exception recovery, and barrier cycles. Kernel overhead results record actual UTC
execution time independently of maintainer-requested Git dates. They are local
microbenchmarks, not pretrained inference performance or hardware roofline results.
The current native model executor still dequantizes weight tensors to float32;
quantized SIMD kernels are available separately and not yet a low-memory model path.
The executor runs scalar float32 matvec, normalization, RoPE and SwiGLU on the
calling thread; only attention score reductions use AVX2 dispatch. The pool is
not used by the executor. See [kernels.md](kernels.md).
