#include "prefill_support.hpp"
#include "tvara/attention.hpp"
int main(){float error=0;for(std::size_t kv:{1,2,4,8}){tvara::ModelConfig c{16,64,128,1,8,kv,128};tvara::PagedCache cache({1,kv,8,16,8});auto seq=cache.create_sequence();auto k=signal(67*c.kv_width()),v=signal(k.size(),2),q=signal(c.hidden,3);
    for(std::size_t p=0;p<67;++p)cache.append(seq,std::span<const float>(k).subspan(p*c.kv_width(),c.kv_width()),std::span<const float>(v).subspan(p*c.kv_width(),c.kv_width()));auto out=tvara::fused_paged_attention(q,cache,seq,0,c),ref=tvara::grouped_attention(q,k,v,67,c);close_vectors(out,ref);error=std::max(error,max_error(out,ref));}
    std::cout<<"{\"head_ratios\":[8,4,2,1],\"max_error\":"<<error<<"}";
}
