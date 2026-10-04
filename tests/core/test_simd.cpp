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

// EXTRA_TESTS
int main(){try{test_detection();
    test_scalar_dispatch();
        // RUN_TESTS
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
