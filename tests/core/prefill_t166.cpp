#include "prefill_support.hpp"
#include "tvara/prefill.hpp"
int main(){tvara::Matrix w(128,256);w.values=signal(w.values.size(),0.1f);auto x=signal(64*256);std::vector<float> out;
    auto decode=median_us([&]{out=tvara::prefill_reference(w,std::span<const float>(x).first(256),1);});
    auto prefill=median_us([&]{out=tvara::prefill_reference(w,x,64);});
    require(out.size()==64*128);std::cout<<"{\"tokens\":64,\"decode_us\":"<<decode<<",\"prefill_us\":"<<prefill<<",\"scope\":\"projection kernel, not end-to-end model\"}";
}
