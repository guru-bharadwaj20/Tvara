#include "tvara/quant.hpp"
#include <iostream>
#include <functional>
#include <random>

using namespace tvara::quant;
using Bytes=std::vector<std::byte>;
void check(bool condition) { if(!condition) throw std::runtime_error("quantization check failed"); }
void close(float a,float b,float tolerance=1e-4f) { check(std::abs(a-b)<=tolerance*(1+std::abs(b))); }
void rejects(const std::function<void()>& f) {
    bool bad=false; try { f(); } catch(const std::invalid_argument&) { bad=true; } check(bad);
}
void half(Bytes& bytes,std::uint16_t h,std::size_t offset=0) {
    bytes.at(offset)=std::byte(h & 255); bytes.at(offset+1)=std::byte(h >> 8);
}
void test_half() {
    close(half_to_float(0x3c00),1); close(half_to_float(0xc000),-2);
    check(half_to_float(1)==std::ldexp(1.0f,-24));
    check(half_to_float(0x7bff)==65504.f);
    check(std::isinf(half_to_float(0x7c00)) && std::isnan(half_to_float(0x7e00)));
    check(std::signbit(half_to_float(0x8000)));
}
void test_q8() {
    Bytes bytes(34); half(bytes,0x3800); std::vector<float> output(32);
    for(std::size_t i=0;i<32;++i) bytes[2+i]=std::byte(static_cast<unsigned char>(static_cast<int>(i)-16));
    dequantize_q8_0(bytes,output);
    for(std::size_t i=0;i<32;++i) close(output[i],0.5f*static_cast<float>(static_cast<int>(i)-16));
    rejects([&]{ dequantize_q8_0(std::span<const std::byte>(bytes).first(33),output); });
}

void test_q4() {
    Bytes bytes(18); half(bytes,0x3c00); std::vector<float> output(32);
    for(std::size_t i=0;i<16;++i) bytes[2+i]=std::byte(i | ((15-i)<<4));
    dequantize_q4_0(bytes,output);
    for(std::size_t i=0;i<16;++i) { close(output[i],static_cast<float>(static_cast<int>(i)-8));
        close(output[i+16],static_cast<float>(7-static_cast<int>(i))); }
}

void pack_scales(Bytes& bytes,const std::array<unsigned,8>& scales,const std::array<unsigned,8>& mins) {
    std::array<unsigned,12> encoded{};
    for(std::size_t i=0;i<4;++i) { encoded[i]=scales[i]&63; encoded[i+4]=mins[i]&63; }
    for(std::size_t i=4;i<8;++i) {
        encoded[i+4]=(scales[i]&15)|((mins[i]&15)<<4);
        encoded[i-4]|=(scales[i]>>4)<<6; encoded[i]|=(mins[i]>>4)<<6;
    }
    for(std::size_t i=0;i<12;++i) bytes[4+i]=std::byte(encoded[i]);
}
void test_q4k_scales() {
    std::array<unsigned,8> scales{0,1,15,16,31,32,47,63},mins{63,47,32,31,16,15,1,0};
    Bytes bytes(144); pack_scales(bytes,scales,mins);
    for(std::size_t i=0;i<8;++i) {
        auto decoded=q4k_scale_min(std::span<const std::byte>(bytes).subspan(4,12),i);
        check(decoded[0]==scales[i] && decoded[1]==mins[i]);
    }
}

void test_q4k() {
    Bytes bytes(144); half(bytes,0x3800); half(bytes,0x3400,2);
    std::array<unsigned,8> scales{1,2,3,4,5,6,7,8},mins{8,7,6,5,4,3,2,1};
    pack_scales(bytes,scales,mins);
    for(std::size_t i=0;i<128;++i) bytes[16+i]=std::byte((i%16)|(((15-i%16)<<4)));
    std::vector<float> out(256); dequantize_q4_k(bytes,out);
    for(std::size_t g=0;g<8;++g) for(std::size_t i=0;i<32;++i) {
        const float q=static_cast<float>(g%2?15-i%16:i%16);
        close(out[g*32+i],0.5f*static_cast<float>(scales[g])*q-0.25f*static_cast<float>(mins[g]));
    }
}

// EXTRA_TESTS
int main() {
    try {
        test_half();
        test_q8();
        test_q4();
        test_q4k_scales();
        test_q4k();
        // RUN_TESTS
        std::cout << "Quantization checks passed\n";
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
