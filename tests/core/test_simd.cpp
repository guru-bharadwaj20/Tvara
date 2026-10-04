#include "tvara/cpu.hpp"
#include "tvara/simd.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
using namespace tvara;
void check(bool c){if(!c)throw std::runtime_error("SIMD check failed");}
void close(float a,float b){check(std::abs(a-b)<=1e-4f*(1+std::abs(b)));}
void test_detection(){auto features=cpu_features();std::cout<<"AVX2="<<features.avx2<<" AVX512="<<features.avx512<<'\n';}
void test_scalar_dispatch(){KernelDispatch dispatch(true);check(!dispatch.avx2 && dispatch.q8==quant::dot_q8_0);check(!use_avx2(true));}

void test_q8_simd(){
    std::vector<std::byte> w(34);w[0]=std::byte(0);w[1]=std::byte(0x3c);
    for(std::size_t i=0;i<32;++i)w[i+2]=std::byte(i*9);
    std::vector<float> x(32);for(std::size_t i=0;i<32;++i)x[i]=std::sin(static_cast<float>(i));
    auto a=quant::quantize_activations(x);KernelDispatch dispatch;
    close(dispatch.q8(w,a),quant::dot_q8_0(w,a));
}

// EXTRA_TESTS
int main(){try{test_detection();
    test_scalar_dispatch();
        test_q8_simd();
        // RUN_TESTS
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
