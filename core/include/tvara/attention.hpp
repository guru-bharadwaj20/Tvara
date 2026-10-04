#pragma once
#include "tvara/model.hpp"
#include "tvara/kv.hpp"
#include "tvara/simd.hpp"
namespace tvara {
struct OnlineAttention {
    explicit OnlineAttention(std::size_t dim):sum(dim){if(!dim)throw std::invalid_argument("empty attention head");}
    double maximum=-std::numeric_limits<double>::infinity(),normalizer=0;
    std::vector<double> sum;
    void update(double score,std::span<const float> value){
        if(value.size()!=sum.size() || !std::isfinite(score))throw std::invalid_argument("invalid online attention input");
        const auto next=std::max(maximum,score),old=std::exp(maximum-next),weight=std::exp(score-next);
        normalizer=normalizer*old+weight;for(std::size_t i=0;i<sum.size();++i)sum[i]=sum[i]*old+weight*value[i];maximum=next;
    }
    std::vector<float> finish()const{if(!(normalizer>0))throw std::invalid_argument("empty attention");std::vector<float> out(sum.size());for(std::size_t i=0;i<sum.size();++i)out[i]=static_cast<float>(sum[i]/normalizer);return out;}
};
inline std::vector<float> online_attention(std::span<const float> q,std::span<const float> keys,std::span<const float> values,std::size_t positions,const ModelConfig& c){
    c.validate();const auto dim=c.head_dim(),width=c.kv_width();
    if(!positions || positions>c.context || positions>std::numeric_limits<std::size_t>::max()/width || q.size()!=c.hidden || keys.size()!=positions*width || values.size()!=keys.size())throw std::invalid_argument("invalid online attention shapes");
    std::vector<float> output(c.hidden);
    for(std::size_t h=0;h<c.heads;++h){OnlineAttention state(dim);const auto kh=h/(c.heads/c.kv_heads);
        for(std::size_t p=0;p<positions;++p){double score=0;for(std::size_t i=0;i<dim;++i)score+=double(q[h*dim+i])*keys[p*width+kh*dim+i];state.update(score/std::sqrt(double(dim)),values.subspan(p*width+kh*dim,dim));}
        auto head=state.finish();std::copy(head.begin(),head.end(),output.begin()+static_cast<std::ptrdiff_t>(h*dim));}
    return output;
}
// ATTENTION_CODE
}
