#pragma once
#include "tvara/quant.hpp"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TVARA_NEON_AVAILABLE 1
#else
#define TVARA_NEON_AVAILABLE 0
#endif
namespace tvara {
inline std::int32_t dot_i8_portable(const std::int8_t* a,const std::int8_t* b){std::int32_t sum=0;for(std::size_t i=0;i<32;++i)sum+=std::int32_t(a[i])*b[i];return sum;}
#if TVARA_NEON_AVAILABLE
inline std::int32_t dot_i8_neon(const std::int8_t* a,const std::int8_t* b){
    int32x4_t total=vdupq_n_s32(0);
    for(std::size_t i=0;i<32;i+=8){auto products=vmull_s8(vld1_s8(a+i),vld1_s8(b+i));total=vaddq_s32(total,vpaddlq_s16(products));}
    std::int32_t lanes[4];vst1q_s32(lanes,total);return lanes[0]+lanes[1]+lanes[2]+lanes[3];
}
#endif
inline float dot_q8_neon_or_scalar(std::span<const std::byte> w,std::span<const quant::Q8Activation> x){
    if(w.size()%34 || w.size()/34!=x.size())throw std::invalid_argument("NEON Q8 dimensions mismatch");
    double sum=0;for(std::size_t b=0;b<x.size();++b){auto a=reinterpret_cast<const std::int8_t*>(w.data()+b*34+2);
#if TVARA_NEON_AVAILABLE
        auto dot=dot_i8_neon(a,x[b].values.data());
#else
        auto dot=dot_i8_portable(a,x[b].values.data());
#endif
        sum+=double(quant::half_to_float(quant::u16(w,b*34)))*x[b].scale*dot;}
    return static_cast<float>(sum);
}
}
