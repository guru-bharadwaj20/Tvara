#include "prefill_support.hpp"
#include "tvara/attention.hpp"
int main(){std::cout<<"{\"contexts\":[";bool first=true;for(std::size_t length:{1,127,1024,4097,8192}){tvara::ModelConfig c{16,64,128,1,4,2,8192};tvara::PagedCache cache({1,2,16,16,512});auto seq=cache.create_sequence();auto k=signal(length*32,8),v=signal(k.size(),2),q=signal(64,6);
    for(std::size_t p=0;p<length;++p)cache.append(seq,std::span<const float>(k).subspan(p*32,32),std::span<const float>(v).subspan(p*32,32));auto fused=tvara::fused_paged_attention(q,cache,seq,0,c),online=tvara::online_attention(q,k,v,length,c),reference=tvara::grouped_attention(q,k,v,length,c);close_vectors(fused,reference,5e-5f);close_vectors(fused,online,1e-6f);
    if(!first)std::cout<<",";first=false;std::cout<<"{\"positions\":"<<length<<",\"fused_unfused_error\":"<<max_error(fused,reference)<<",\"fused_online_error\":"<<max_error(fused,online)<<"}";}std::cout<<"]}";
}
