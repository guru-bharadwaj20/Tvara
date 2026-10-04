#pragma once
#include "tvara/model.hpp"
#include "tvara/kv.hpp"
namespace tvara {
class PagedModelExecutor {
public:
    explicit PagedModelExecutor(ModelWeights weights,std::size_t max_blocks=64,std::size_t block_tokens=16)
        :weights_(std::move(weights)),cache_({weights_.config.layers,weights_.config.kv_heads,weights_.config.head_dim(),block_tokens,max_blocks}){weights_.config.validate();}
    PagedCache& cache(){return cache_;}
    const ModelConfig& config()const{return weights_.config;}
    std::vector<float> decode(std::uint64_t sequence,std::size_t token){
        const auto& c=config();const auto position=cache_.sequence(sequence).length;
        if(token>=c.vocab || position>=c.context)throw std::invalid_argument("paged decode exceeds token/context limits");
        auto x=embedding_lookup(weights_.embedding,token);std::vector<float> new_keys(c.layers*c.kv_width()),new_values(new_keys.size());
        for(std::size_t i=0;i<c.layers;++i){const auto& layer=weights_.layers.at(i);
            auto norm=rms_norm(x,layer.attention_norm,c.rms_epsilon);auto projected=project_attention(layer,norm,c);
            rotary(projected.query,c.heads,c.head_dim(),position,c.rope_theta,c.interleaved_rope);
            rotary(projected.key,c.kv_heads,c.head_dim(),position,c.rope_theta,c.interleaved_rope);
            std::copy(projected.key.begin(),projected.key.end(),new_keys.begin()+static_cast<std::ptrdiff_t>(i*c.kv_width()));
            std::copy(projected.value.begin(),projected.value.end(),new_values.begin()+static_cast<std::ptrdiff_t>(i*c.kv_width()));
            auto keys=cache_.read_all(sequence,i),values=cache_.read_all(sequence,i,true);
            keys.insert(keys.end(),projected.key.begin(),projected.key.end());values.insert(values.end(),projected.value.begin(),projected.value.end());
            auto attended=grouped_attention(projected.query,keys,values,position+1,c);
            auto residual=project_attention_output(layer,attended,c);for(std::size_t j=0;j<c.hidden;++j)x[j]+=residual[j];
            norm=rms_norm(x,layer.feedforward_norm,c.rms_epsilon);residual=feedforward(layer,norm,c);for(std::size_t j=0;j<c.hidden;++j)x[j]+=residual[j];
        }
        auto norm=rms_norm(x,weights_.final_norm,c.rms_epsilon);auto logits=(weights_.output.rows?weights_.output:weights_.embedding).multiply(norm);
        cache_.append(sequence,new_keys,new_values);return logits;
    }
    std::vector<float> forward(std::span<const std::size_t> tokens){
        if(tokens.empty() || tokens.size()>config().context)throw std::invalid_argument("invalid paged forward prompt");
        const auto sequence=cache_.create_sequence();std::vector<float> output;
        try{for(auto token:tokens){auto logits=decode(sequence,token);output.insert(output.end(),logits.begin(),logits.end());}cache_.drop(sequence);return output;}
        catch(...){cache_.drop(sequence);throw;}
    }
private:
    ModelWeights weights_;PagedCache cache_;
};
}
