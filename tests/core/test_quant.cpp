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
// EXTRA_TESTS
int main() {
    try {
        test_half();
        // RUN_TESTS
        std::cout << "Quantization checks passed\n";
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
