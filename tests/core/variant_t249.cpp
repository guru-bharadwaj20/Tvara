#include "prefill_support.hpp"
#include "tvara/neon.hpp"
int main(){std::int8_t a[32],b[32];std::fill_n(a,32,std::int8_t(-128));std::fill_n(b,32,std::int8_t(-128));require(tvara::dot_i8_portable(a,b)==524288);
#if TVARA_NEON_AVAILABLE
    require(tvara::dot_i8_neon(a,b)==524288);
#endif
    auto x=signal(64);auto activation=tvara::quant::quantize_activations(x);std::vector<std::byte> weights(68);for(std::size_t block=0;block<2;++block){weights[block*34]=std::byte(0);weights[block*34+1]=std::byte(0x3c);for(std::size_t k=0;k<32;++k)weights[block*34+2+k]=std::byte((k*11)%255);}
    auto expected=tvara::quant::dot_q8_0(weights,activation),actual=tvara::dot_q8_neon_or_scalar(weights,activation);require(expected==actual);
    std::cout<<"{\"neon_compiled\":"<<(TVARA_NEON_AVAILABLE?"true":"false")<<",\"fallback_error\":"<<std::abs(actual-expected)<<",\"maximum_int8_sum\":524288}";
}
