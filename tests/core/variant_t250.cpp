#include "prefill_support.hpp"
#include "tvara/neon.hpp"
int main(){
#if TVARA_NEON_AVAILABLE
    std::mt19937 rng(250);std::uniform_int_distribution<int> values(-128,127);std::int8_t a[32],b[32];volatile std::int32_t checksum=0;
    for(int trial=0;trial<500;++trial){for(std::size_t i=0;i<32;++i){a[i]=static_cast<std::int8_t>(values(rng));b[i]=static_cast<std::int8_t>(values(rng));}const auto actual=tvara::dot_i8_neon(a,b);checksum=actual;require(actual==tvara::dot_i8_portable(a,b));}
    auto x=signal(256);auto activation=tvara::quant::quantize_activations(x);std::vector<std::byte> weights(8*34);for(std::size_t block=0;block<8;++block){weights[block*34]=std::byte(0);weights[block*34+1]=std::byte(0x3c);for(std::size_t k=0;k<32;++k)weights[block*34+2+k]=std::byte((k*11+block)%255);}
    require(tvara::dot_q8_neon_or_scalar(weights,activation)==tvara::quant::dot_q8_0(weights,activation));(void)checksum;std::cout<<"{\"status\":\"validated\",\"neon_vectors\":500,\"q8_blocks\":8}";
#else
    std::cout<<"{\"status\":\"blocked\",\"reason\":\"ARM NEON build and execution unavailable on current x86 host\"}";
#endif
}
