#include "prefill_support.hpp"
#include "tvara/paged_model.hpp"
int main(){tvara::ModelConfig c{16,16,32,1,2,2,128};tvara::PagedCache cache({1,2,8,7,32});auto seq=cache.create_sequence();auto k=signal(53*c.kv_width()),v=signal(k.size(),2),q=signal(c.hidden,3);
    for(std::size_t p=0;p<53;++p)cache.append(seq,std::span<const float>(k).subspan(p*c.kv_width(),c.kv_width()),std::span<const float>(v).subspan(p*c.kv_width(),c.kv_width()));
    auto fused=tvara::fused_paged_attention(q,cache,seq,0,c);auto reference=tvara::grouped_attention(q,k,v,53,c);close_vectors(fused,reference);
    auto tail_k=signal(c.kv_width(),4),tail_v=signal(c.kv_width(),5);auto kt=k,vt=v;kt.insert(kt.end(),tail_k.begin(),tail_k.end());vt.insert(vt.end(),tail_v.begin(),tail_v.end());close_vectors(tvara::fused_paged_attention(q,cache,seq,0,c,tail_k,tail_v),tvara::grouped_attention(q,kt,vt,54,c));
    std::cout<<"{\"positions\":53,\"max_error\":"<<max_error(fused,reference)<<",\"gathered_cache_elements\":0}";
}
