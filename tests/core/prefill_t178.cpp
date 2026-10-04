#include "prefill_support.hpp"
#include "tvara/paged_model.hpp"
int main(){tvara::ModelWeights w;w.config={23,16,32,2,4,2,128};w.embedding=tvara::Matrix(23,16);w.embedding.values=signal(w.embedding.values.size(),0.2f);w.final_norm.assign(16,1);
    for(int i=0;i<2;++i){tvara::LayerWeights l;l.attention_norm.assign(16,1);l.feedforward_norm.assign(16,1);l.query=tvara::Matrix(16,16);l.key=tvara::Matrix(8,16);l.value=tvara::Matrix(8,16);l.attention_output=tvara::Matrix(16,16);l.gate=tvara::Matrix(32,16);l.up=tvara::Matrix(32,16);l.down=tvara::Matrix(16,32);
        for(auto* m:{&l.query,&l.key,&l.value,&l.attention_output,&l.gate,&l.up,&l.down})m->values=signal(m->values.size(),0.03f);w.layers.push_back(l);}
    tvara::PagedModelExecutor executor(w,32,7);auto seq=executor.cache().create_sequence();tvara::SequenceState reference;reference.layers.resize(2);float error=0;
    for(std::size_t i=0;i<73;++i){auto a=executor.decode(seq,(i*7)%23),b=tvara::execute_token(w,reference,(i*7)%23);close_vectors(a,b,2e-5f);error=std::max(error,max_error(a,b));}executor.cache().audit();
    auto q=signal(w.config.hidden);std::vector<float> measured;
    auto fused_us=median_us([&]{measured=tvara::fused_paged_attention(q,executor.cache(),seq,0,w.config);});
    auto gathered_us=median_us([&]{auto k=executor.cache().read_all(seq,0),v=executor.cache().read_all(seq,0,true);measured=tvara::grouped_attention(q,k,v,73,w.config);});
    std::cout<<"{\"tokens\":73,\"layers\":2,\"max_logit_error\":"<<error<<",\"fused_attention_us\":"<<fused_us<<",\"gathered_unfused_attention_us\":"<<gathered_us<<"}";
}
