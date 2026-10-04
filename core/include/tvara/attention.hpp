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
inline std::vector<float> fused_paged_attention(std::span<const float> q,const PagedCache& cache,std::uint64_t sequence,std::size_t layer,const ModelConfig& c,
    std::span<const float> tail_key={},std::span<const float> tail_value={},std::size_t visible=std::numeric_limits<std::size_t>::max(),bool scalar=false,bool prefetch=false) {
    c.validate();const auto& g=cache.geometry();const auto& table=cache.sequence(sequence);const auto width=c.kv_width(),dim=c.head_dim();
    if(q.size()!=c.hidden || g.kv_heads!=c.kv_heads || g.head_dim!=dim || layer>=g.layers || tail_key.size()!=tail_value.size() ||
       (!tail_key.empty() && tail_key.size()!=width) || table.length>c.context || (!tail_key.empty() && table.length==c.context))throw std::invalid_argument("invalid fused attention shape");
    const auto total=table.length+(tail_key.empty()?0:1);visible=std::min(visible,total);
    if(!visible)throw std::invalid_argument("empty fused attention");
    std::vector<float> output(c.hidden);const auto cached=std::min(visible,table.length);
    for(std::size_t h=0;h<c.heads;++h){OnlineAttention state(dim);const auto kh=h/(c.heads/c.kv_heads);auto query=q.subspan(h*dim,dim);
        for(std::size_t block=0;block<table.blocks.size() && block*g.tokens_per_block<cached;++block){auto data=cache.pool().data(table.blocks[block]);
            const auto count=std::min(g.tokens_per_block,cached-block*g.tokens_per_block);
            for(std::size_t slot=0;slot<count;++slot){auto key=data.subspan(kv_offset(g,layer,0,slot)+kh*dim,dim);auto value=data.subspan(kv_offset(g,layer,1,slot)+kh*dim,dim);
                // SCORE_CODE
                (void)scalar;(void)prefetch;double score=0;for(std::size_t i=0;i<dim;++i)score+=double(query[i])*key[i];
                state.update(score/std::sqrt(double(dim)),value);}}
        if(visible>table.length){double score=0;for(std::size_t i=0;i<dim;++i)score+=double(query[i])*tail_key[kh*dim+i];state.update(score/std::sqrt(double(dim)),tail_value.subspan(kh*dim,dim));}
        auto head=state.finish();std::copy(head.begin(),head.end(),output.begin()+static_cast<std::ptrdiff_t>(h*dim));}
    return output;
}

inline std::vector<float> fused_prompt_attention(std::span<const float> queries,const PagedCache& cache,std::uint64_t sequence,std::size_t layer,std::size_t start,std::size_t count,const ModelConfig& c){
    c.validate();const auto length=cache.sequence(sequence).length;
    if(!count || start>=length || count>length-start || count>std::numeric_limits<std::size_t>::max()/c.hidden || queries.size()!=count*c.hidden)throw std::invalid_argument("invalid causal prompt chunk");
    std::vector<float> output;output.reserve(queries.size());for(std::size_t i=0;i<count;++i){auto row=fused_paged_attention(queries.subspan(i*c.hidden,c.hidden),cache,sequence,layer,c,{},{},start+i+1);output.insert(output.end(),row.begin(),row.end());}return output;
}

// ATTENTION_CODE
}
