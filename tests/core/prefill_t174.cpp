#include "prefill_support.hpp"
#include "tvara/attention.hpp"
int main(){float error=0;for(std::size_t len:{1,15,16,17,31,32,33}){tvara::ModelConfig c{16,32,64,1,4,2,128};tvara::PagedCache cache({1,2,8,16,8});auto old=cache.create_sequence();auto junk=signal(16,100);for(int i=0;i<48;++i)cache.append(old,junk,junk);cache.drop(old);auto seq=cache.create_sequence();auto k=signal(len*16),v=signal(k.size(),2),q=signal(32,3);
    for(std::size_t p=0;p<len;++p)cache.append(seq,std::span<const float>(k).subspan(p*16,16),std::span<const float>(v).subspan(p*16,16));
    auto last=cache.physical_block(seq,len-1);auto data=cache.pool().data(last);for(std::size_t slot=len%16;len%16 && slot<16;++slot){auto offset=tvara::kv_offset(cache.geometry(),0,0,slot);std::fill_n(data.begin()+static_cast<std::ptrdiff_t>(offset),16,std::numeric_limits<float>::quiet_NaN());}
    auto out=tvara::fused_paged_attention(q,cache,seq,0,c),ref=tvara::grouped_attention(q,k,v,len,c);close_vectors(out,ref);error=std::max(error,max_error(out,ref));cache.audit();}
    std::cout<<"{\"lengths\":[1,15,16,17,31,32,33],\"invalid_tail_poisoned\":true,\"max_error\":"<<error<<"}";
}
