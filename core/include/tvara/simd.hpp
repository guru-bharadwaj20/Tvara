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
#if TVARA_AVX2_AVAILABLE
TVARA_AVX2 __attribute__((always_inline)) inline std::int32_t sum_i32(__m256i value) {
    alignas(32) std::int32_t lanes[8];_mm256_store_si256(reinterpret_cast<__m256i*>(lanes),value);
    return std::accumulate(lanes,lanes+8,std::int32_t(0));
}
TVARA_AVX2 __attribute__((always_inline)) inline std::int32_t dot_i8_32(const std::int8_t* a,const std::int8_t* b) {
    auto a0=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
    auto b0=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
    auto a1=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+16)));
    auto b1=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b+16)));
    return sum_i32(_mm256_add_epi32(_mm256_madd_epi16(a0,b0),_mm256_madd_epi16(a1,b1)));
}
TVARA_AVX2 inline float dot_q8_avx2(std::span<const std::byte> w,std::span<const quant::Q8Activation> x) {
    if(w.size()%34 || w.size()/34!=x.size())throw std::invalid_argument("AVX2 Q8 dimensions mismatch");
    double sum=0;
    for(std::size_t b=0;b<x.size();++b)sum+=double(quant::half_to_float(quant::u16(w,b*34)))*x[b].scale*
        dot_i8_32(reinterpret_cast<const std::int8_t*>(w.data()+b*34+2),x[b].values.data());
    return static_cast<float>(sum);
}
#endif

#if TVARA_AVX2_AVAILABLE
TVARA_AVX2 inline float dot_q4_avx2(std::span<const std::byte> w,std::span<const quant::Q8Activation> x) {
    if(w.size()%18 || w.size()/18!=x.size())throw std::invalid_argument("AVX2 Q4 dimensions mismatch");
    double sum=0;const auto mask=_mm256_set1_epi16(15),offset=_mm256_set1_epi16(8);
    for(std::size_t b=0;b<x.size();++b) {
        auto packed=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w.data()+b*18+2)));
        auto low=_mm256_sub_epi16(_mm256_and_si256(packed,mask),offset);
        auto high=_mm256_sub_epi16(_mm256_srli_epi16(packed,4),offset);
        auto a0=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x[b].values.data())));
        auto a1=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x[b].values.data()+16)));
        auto products=_mm256_add_epi32(_mm256_madd_epi16(low,a0),_mm256_madd_epi16(high,a1));
        sum+=double(quant::half_to_float(quant::u16(w,b*18)))*x[b].scale*sum_i32(products);
    }
    return static_cast<float>(sum);
}
#endif

// SIMD_CODE
struct KernelDispatch {
    using Dot=float(*)(std::span<const std::byte>,std::span<const quant::Q8Activation>);
    Dot q8=quant::dot_q8_0,q4=quant::dot_q4_0,q4k=quant::dot_q4_k;
    bool avx2=false;
    explicit KernelDispatch(bool force_scalar=false) {
#if TVARA_AVX2_AVAILABLE
        if(use_avx2(force_scalar)){q8=dot_q8_avx2;q4=dot_q4_avx2;avx2=true;}
#else
        (void)force_scalar;
#endif
    }
};
}
