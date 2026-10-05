#pragma once
#include "tvara/simd.hpp"
namespace tvara {
#if TVARA_AVX2_AVAILABLE
#define TVARA_AVX512 __attribute__((target("avx512f,avx512bw")))
TVARA_AVX512 inline std::int32_t dot_i8_avx512(const std::int8_t* a,const std::int8_t* b){
    const auto x=_mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)));
    const auto y=_mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b)));
    const auto sums=_mm512_madd_epi16(x,y);std::int32_t lanes[16];_mm512_storeu_si512(lanes,sums);
    return std::accumulate(lanes,lanes+16,std::int32_t(0));
}
TVARA_AVX512 inline float dot_q8_avx512(std::span<const std::byte> w,std::span<const quant::Q8Activation> x){
    if(w.size()%34 || w.size()/34!=x.size())throw std::invalid_argument("AVX512 Q8 dimensions mismatch");
    double sum=0;for(std::size_t b=0;b<x.size();++b)sum+=double(quant::half_to_float(quant::u16(w,b*34)))*x[b].scale*
        dot_i8_avx512(reinterpret_cast<const std::int8_t*>(w.data()+b*34+2),x[b].values.data());
    return static_cast<float>(sum);
}
inline bool avx512_q8_available(){__builtin_cpu_init();return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");}
#else
inline bool avx512_q8_available(){return false;}
#endif
struct OptionalKernelDispatch:KernelDispatch {
    bool avx512=false;
    explicit OptionalKernelDispatch(bool force_scalar=false):KernelDispatch(force_scalar){
#if TVARA_AVX2_AVAILABLE
        const auto enabled=std::getenv("TVARA_ENABLE_AVX512");
        if(!force_scalar && use_avx2() && enabled && std::string(enabled)=="1" && avx512_q8_available()){
            q8=dot_q8_avx512;avx512=true;
        }
#endif
    }
};
}
