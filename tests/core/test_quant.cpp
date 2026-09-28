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

// EXTRA_TESTS
int main() {
    try {
        test_half();
        test_q8();
        test_q4();
        // RUN_TESTS
        std::cout << "Quantization checks passed\n";
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
