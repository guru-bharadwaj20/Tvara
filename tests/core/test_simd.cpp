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

void test_q4_simd(){
    std::vector<std::byte> w(18);w[0]=std::byte(0);w[1]=std::byte(0xbc);
    for(std::size_t i=0;i<16;++i)w[i+2]=std::byte(i|(i<<4));
    std::vector<float> x(32);for(std::size_t i=0;i<32;++i)x[i]=std::cos(static_cast<float>(i));
    auto a=quant::quantize_activations(x);KernelDispatch dispatch;close(dispatch.q4(w,a),quant::dot_q4_0(w,a));
}

void test_q4k_simd(){
    std::vector<std::byte> w(144);w[1]=std::byte(0x38);w[3]=std::byte(0x34);
    for(std::size_t i=4;i<144;++i)w[i]=std::byte(i*7 & 255);
    std::vector<float> x(256);for(std::size_t i=0;i<256;++i)x[i]=std::sin(static_cast<float>(i));
    auto a=quant::quantize_activations(x);KernelDispatch dispatch;close(dispatch.q4k(w,a),quant::dot_q4_k(w,a));
}

void test_vector_norm_activation(){
#if TVARA_AVX2_AVAILABLE
    if(!use_avx2())return;
    std::vector<float> x(35),w(35,1),up(35,2);for(std::size_t i=0;i<x.size();++i)x[i]=std::sin(static_cast<float>(i));
    auto reference=rms_norm(x,w,1e-6f),actual=rms_norm_avx2(x,w,1e-6f);for(std::size_t i=0;i<x.size();++i)close(actual[i],reference[i]);
    auto gate=x;swiglu_avx2(gate,up);for(std::size_t i=0;i<x.size();++i)close(gate[i],x[i]*2/(1+std::exp(-x[i])));
#endif
}

void test_vector_rope(){
#if TVARA_AVX2_AVAILABLE
    if(!use_avx2())return;
    for(bool layout:{false,true}){std::vector<float> x(72);for(std::size_t i=0;i<x.size();++i)x[i]=static_cast<float>(i)/10;
        auto ref=x;rotary(ref,2,36,7,10000,layout);rotary_avx2(x,2,36,7,10000,layout);
        for(std::size_t i=0;i<x.size();++i)close(x[i],ref[i]);}
#endif
}

// EXTRA_TESTS
int main(){try{test_detection();
    test_scalar_dispatch();
        test_q8_simd();
        test_q4_simd();
        test_q4k_simd();
        test_vector_norm_activation();
        test_vector_rope();
        // RUN_TESTS
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
