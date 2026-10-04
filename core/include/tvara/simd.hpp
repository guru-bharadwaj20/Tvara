#pragma once
#include "tvara/cpu.hpp"
#include "tvara/model.hpp"
#include <cstdlib>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TVARA_AVX2_AVAILABLE 1
#define TVARA_AVX2 __attribute__((target("avx2")))
#else
#define TVARA_AVX2_AVAILABLE 0
#define TVARA_AVX2
#endif
namespace tvara {
inline bool use_avx2(bool force_scalar=false) {
    const auto setting=std::getenv("TVARA_FORCE_SCALAR");
    return TVARA_AVX2_AVAILABLE && !force_scalar && !(setting && std::string(setting)=="1") && cpu_features().avx2;
}
// SIMD_CODE
struct KernelDispatch {
    using Dot=float(*)(std::span<const std::byte>,std::span<const quant::Q8Activation>);
    Dot q8=quant::dot_q8_0,q4=quant::dot_q4_0,q4k=quant::dot_q4_k;
    bool avx2=false;
    explicit KernelDispatch(bool force_scalar=false) { (void)force_scalar; }
};
}
