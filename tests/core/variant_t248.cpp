#include "prefill_support.hpp"
#include "tvara/avx512.hpp"
#include <atomic>
int main(){if(!tvara::avx512_q8_available()){std::cout<<"{\"status\":\"blocked\",\"reason\":\"AVX512F and AVX512BW hardware unavailable\",\"avx2\":"<<(tvara::use_avx2()?"true":"false")<<"}";return 0;}
#if TVARA_AVX2_AVAILABLE
    auto x=signal(4096);auto activation=tvara::quant::quantize_activations(x);std::vector<std::byte> weights(activation.size()*34);
    for(std::size_t b=0;b<activation.size();++b){weights[b*34]=std::byte(0);weights[b*34+1]=std::byte(0x3c);for(std::size_t k=0;k<32;++k)weights[b*34+2+k]=std::byte((b+k)%127);}
    auto scalar=tvara::quant::dot_q8_0(weights,activation),avx2=tvara::dot_q8_avx2(weights,activation),avx512=tvara::dot_q8_avx512(weights,activation);require(std::abs(scalar-avx2)<1e-5f*(1+std::abs(scalar)));require(std::abs(scalar-avx512)<1e-5f*(1+std::abs(scalar)));
    volatile float sink=0;auto two=median_us([&]{for(int i=0;i<100;++i){std::atomic_signal_fence(std::memory_order_seq_cst);sink=tvara::dot_q8_avx2(weights,activation);}});auto five=median_us([&]{for(int i=0;i<100;++i){std::atomic_signal_fence(std::memory_order_seq_cst);sink=tvara::dot_q8_avx512(weights,activation);}});(void)sink;
    std::cout<<"{\"status\":\"validated\",\"avx2_us_100\":"<<two<<",\"avx512_us_100\":"<<five<<",\"max_error\":"<<std::max(std::abs(scalar-avx2),std::abs(scalar-avx512))<<"}";
#endif
}
