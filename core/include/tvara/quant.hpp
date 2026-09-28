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

// QUANT_IMPLEMENTATIONS
}  // namespace tvara::quant
