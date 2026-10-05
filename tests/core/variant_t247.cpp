#include "prefill_support.hpp"
#include "tvara/avx512.hpp"
int main(){auto x=signal(64);auto activation=tvara::quant::quantize_activations(x);std::vector<std::byte> weights(68);
    for(std::size_t b=0;b<2;++b){weights[b*34]=std::byte(0);weights[b*34+1]=std::byte(0x3c);for(std::size_t k=0;k<32;++k)weights[b*34+2+k]=std::byte((k*7)%127);}
    tvara::OptionalKernelDispatch automatic,scalar(true);auto a=automatic.q8(weights,activation),b=scalar.q8(weights,activation);require(std::abs(a-b)<1e-5f*(1+std::abs(b)));
#if TVARA_AVX2_AVAILABLE
    if(tvara::avx512_q8_available()){auto value=tvara::dot_q8_avx512(weights,activation);require(std::abs(value-b)<1e-5f*(1+std::abs(b)));}
#endif
    require(!scalar.avx512);std::cout<<"{\"avx512_hardware\":"<<(tvara::avx512_q8_available()?"true":"false")<<",\"opt_in_active\":"<<(automatic.avx512?"true":"false")<<",\"fallback_error\":"<<std::abs(a-b)<<"}";
}
