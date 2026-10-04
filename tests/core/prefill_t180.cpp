#include "prefill_support.hpp"
#include "tvara/attention.hpp"
#include "tvara/prefill.hpp"
int main(){tvara::Matrix w(19,33);w.values=signal(w.values.size());auto x=signal(7*33);close_vectors(tvara::prefill_blocked(w,x,7),tvara::prefill_reference(w,x,7));
    tvara::ModelConfig c{16,24,48,1,4,2,32};tvara::PagedCache cache({1,2,6,5,8});auto seq=cache.create_sequence();auto k=signal(23*12),v=signal(k.size(),2),q=signal(24,3);for(std::size_t p=0;p<23;++p)cache.append(seq,std::span<const float>(k).subspan(p*12,12),std::span<const float>(v).subspan(p*12,12));
    auto scalar=tvara::fused_paged_attention(q,cache,seq,0,c,{},{},23,true),automatic=tvara::fused_paged_attention(q,cache,seq,0,c);close_vectors(scalar,automatic,1e-6f);std::cout<<"{\"accepted_fused_attention\":true,\"prefill_blocked_default\":false,\"prefetch_default\":false,\"max_dispatch_error\":"<<max_error(scalar,automatic)<<"}";
}
