#include "prefill_support.hpp"
#include "tvara/attention.hpp"
int main(){tvara::ModelConfig c{16,32,64,1,4,2,1024};auto q=signal(c.hidden,4),k=signal(731*c.kv_width(),7),v=signal(k.size());auto reference=tvara::grouped_attention(q,k,v,731,c);auto online=tvara::online_attention(q,k,v,731,c);close_vectors(online,reference,1e-5f);
    tvara::OnlineAttention state(2);std::vector<float> a{1,2},b{3,4};state.update(10000,a);state.update(10001,b);auto result=state.finish();require(std::isfinite(result[0]));rejects([&]{tvara::OnlineAttention(2).finish();});std::cout<<"{\"positions\":731,\"max_error\":"<<max_error(online,reference)<<"}";
}
