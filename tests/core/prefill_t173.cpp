#include "prefill_support.hpp"
#include "tvara/attention.hpp"
int main(){tvara::ModelConfig c{16,32,64,1,4,1,128};tvara::PagedCache cache({1,1,8,7,20});auto seq=cache.create_sequence();auto k=signal(39*8),v=signal(k.size(),2),q=signal(13*32,3);
    for(std::size_t p=0;p<39;++p)cache.append(seq,std::span<const float>(k).subspan(p*8,8),std::span<const float>(v).subspan(p*8,8));auto out=tvara::fused_prompt_attention(q,cache,seq,0,11,13,c);
    for(std::size_t i=0;i<13;++i){auto ref=tvara::grouped_attention(std::span<const float>(q).subspan(i*32,32),std::span<const float>(k).first((12+i)*8),std::span<const float>(v).first((12+i)*8),12+i,c);close_vectors(std::span<const float>(out).subspan(i*32,32),ref);}
    auto changed=cache;auto last=changed.physical_block(seq,38);auto data=changed.pool().data(last);std::fill(data.begin(),data.end(),99999);close_vectors(out,tvara::fused_prompt_attention(q,changed,seq,0,11,13,c));rejects([&]{tvara::fused_prompt_attention(q,cache,seq,0,30,13,c);});
    std::cout<<"{\"chunk_start\":11,\"chunk_tokens\":13,\"future_mutation_invariant\":true}";
}
