#pragma once
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace tvara::quant {
inline float half_to_float(std::uint16_t h) {
    const std::uint32_t sign=std::uint32_t(h & 0x8000) << 16;
    std::uint32_t exponent=(h >> 10) & 31;
    std::uint32_t mantissa=h & 1023;
    std::uint32_t bits;
    if(exponent==0) {
        if(!mantissa) bits=sign;
        else {
            int e=-14;
            while((mantissa & 1024)==0) { mantissa <<= 1; --e; }
            bits=sign | (static_cast<std::uint32_t>(e+127)<<23) | ((mantissa & 1023)<<13);
        }
    } else if(exponent==31) bits=sign | 0x7f800000 | (mantissa<<13);
    else bits=sign | ((exponent+112)<<23) | (mantissa<<13);
    return std::bit_cast<float>(bits);
}
inline std::uint16_t u16(std::span<const std::byte> bytes,std::size_t offset=0) {
    if(offset>bytes.size() || bytes.size()-offset<2) throw std::invalid_argument("truncated half");
    return static_cast<std::uint16_t>(std::to_integer<unsigned>(bytes[offset]) |
                                   (std::to_integer<unsigned>(bytes[offset+1])<<8));
}
inline void require_layout(std::span<const std::byte> bytes,std::span<float> output,std::size_t width,std::size_t elements) {
    if(bytes.size()%width || bytes.size()/width>std::numeric_limits<std::size_t>::max()/elements ||
       output.size()!=bytes.size()/width*elements)
        throw std::invalid_argument("quantized storage/output size mismatch");
}
inline void dequantize_q8_0(std::span<const std::byte> bytes,std::span<float> output) {
    require_layout(bytes,output,34,32);
    for(std::size_t b=0;b<bytes.size()/34;++b) {
        const float scale=half_to_float(u16(bytes,b*34));
        for(std::size_t i=0;i<32;++i) {
            const auto q=std::bit_cast<std::int8_t>(std::to_integer<std::uint8_t>(bytes[b*34+2+i]));
            output[b*32+i]=scale*static_cast<float>(q);
        }
    }
}

inline void dequantize_q4_0(std::span<const std::byte> bytes,std::span<float> output) {
    require_layout(bytes,output,18,32);
    for(std::size_t b=0;b<bytes.size()/18;++b) {
        const auto scale=half_to_float(u16(bytes,b*18));
        for(std::size_t i=0;i<16;++i) {
            const auto q=std::to_integer<unsigned>(bytes[b*18+2+i]);
            output[b*32+i]=scale*static_cast<float>(static_cast<int>(q & 15)-8);
            output[b*32+i+16]=scale*static_cast<float>(static_cast<int>(q >> 4)-8);
        }
    }
}

inline std::array<unsigned,2> q4k_scale_min(std::span<const std::byte> scales,std::size_t index) {
    if(scales.size()!=12 || index>=8) throw std::invalid_argument("invalid Q4_K scale group");
    auto q=[&](std::size_t i){ return std::to_integer<unsigned>(scales[i]); };
    if(index<4) return {q(index)&63,q(index+4)&63};
    return {(q(index+4)&15)|((q(index-4)>>6)<<4), (q(index+4)>>4)|((q(index)>>6)<<4)};
}

inline void dequantize_q4_k(std::span<const std::byte> bytes,std::span<float> output) {
    require_layout(bytes,output,144,256);
    for(std::size_t b=0;b<bytes.size()/144;++b) {
        const auto block=bytes.subspan(b*144,144);
        const float d=half_to_float(u16(block)),dmin=half_to_float(u16(block,2));
        for(std::size_t group=0;group<8;++group) {
            const auto scale=q4k_scale_min(block.subspan(4,12),group);
            for(std::size_t i=0;i<32;++i) {
                const auto packed=std::to_integer<unsigned>(block[16+(group/2)*32+i]);
                const auto q=(group%2)?(packed>>4):(packed&15);
                output[b*256+group*32+i]=d*static_cast<float>(scale[0])*static_cast<float>(q)-dmin*static_cast<float>(scale[1]);
            }
        }
    }
}

struct Q8Activation { float scale=0; std::array<std::int8_t,32> values{}; };
inline std::vector<Q8Activation> quantize_activations(std::span<const float> x) {
    if(x.size()%32) throw std::invalid_argument("activation width must be divisible by 32");
    std::vector<Q8Activation> blocks(x.size()/32);
    for(std::size_t b=0;b<blocks.size();++b) {
        float maximum=0;
        for(std::size_t i=0;i<32;++i) {
            const auto value=x[b*32+i];
            if(!std::isfinite(value)) throw std::invalid_argument("nonfinite activation");
            maximum=std::max(maximum,std::abs(value));
        }
        auto& block=blocks[b]; block.scale=maximum/127.f;
        if(block.scale==0) continue;
        for(std::size_t i=0;i<32;++i)
            block.values[i]=static_cast<std::int8_t>(std::round(x[b*32+i]/block.scale));
    }
    return blocks;
}

inline float dot_f32(std::span<const float> a,std::span<const float> b) {
    if(a.size()!=b.size()) throw std::invalid_argument("dot-product dimension mismatch");
    double sum=0;
    for(std::size_t i=0;i<a.size();++i) sum+=double(a[i])*double(b[i]);
    return static_cast<float>(sum);
}

// QUANT_IMPLEMENTATIONS
}  // namespace tvara::quant
