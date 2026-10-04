#include "prefill_support.hpp"
#include "tvara/prefill.hpp"
#include "tvara/attention.hpp"
int main(){tvara::Matrix w(256,256);w.values=signal(w.values.size());auto x=signal(16*256);std::vector<float> out;auto ref=tvara::prefill_reference(w,x,16);
    auto plain=median_us([&]{out=tvara::prefill_weight_prefetch(w,x,16,false);});auto prefetched=median_us([&]{out=tvara::prefill_weight_prefetch(w,x,16,true);});close_vectors(out,ref);
    tvara::ModelConfig c{16,128,256,1,4,2,2048};tvara::PagedCache cache({1,2,32,16,128});auto seq=cache.create_sequence();auto k=signal(64),v=signal(64,2),q=signal(128,3);for(int i=0;i<1025;++i)cache.append(seq,k,v);
    auto expected=tvara::fused_paged_attention(q,cache,seq,0,c);auto cache_plain=median_us([&]{out=tvara::fused_paged_attention(q,cache,seq,0,c,{},{},1025,false,false);});auto cache_prefetch=median_us([&]{out=tvara::fused_paged_attention(q,cache,seq,0,c,{},{},1025,false,true);});close_vectors(out,expected);
    std::cout<<"{\"weight_plain_us\":"<<plain<<",\"weight_prefetch_us\":"<<prefetched<<",\"cache_plain_us\":"<<cache_plain<<",\"cache_prefetch_us\":"<<cache_prefetch<<",\"default_prefetch\":false}";
}
