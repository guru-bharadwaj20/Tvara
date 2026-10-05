#pragma once
#include "tvara/gguf_model.hpp"
#include "tvara/quant.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace tvara {
struct ModelConfig {
    std::size_t vocab=0,hidden=0,intermediate=0,layers=0,heads=0,kv_heads=0,context=0;
    float rms_epsilon=1e-6f,rope_theta=10000.f;
    bool interleaved_rope=true;
    void validate() const {
        if(!vocab || !hidden || !intermediate || !layers || !heads || !kv_heads || !context ||
           hidden%heads || heads%kv_heads || (hidden/heads)%2 ||
           !std::isfinite(rms_epsilon) || rms_epsilon<=0 || !std::isfinite(rope_theta) || rope_theta<=0)
            throw std::invalid_argument("invalid native model configuration");
    }
    std::size_t head_dim() const { return hidden/heads; }
    std::size_t kv_width() const { return head_dim()*kv_heads; }
};
struct Matrix {
    std::size_t rows=0,cols=0;
    std::vector<float> values;
    Matrix()=default;
    Matrix(std::size_t r,std::size_t c) : rows(r),cols(c) {
        if(!r || !c || r>std::numeric_limits<std::size_t>::max()/c)
            throw std::invalid_argument("invalid matrix shape");
        values.resize(r*c);
    }
    std::vector<float> multiply(std::span<const float> x) const {
        std::vector<float> out(rows);
        quant::matvec_f32(values,rows,cols,x,out);
        return out;
    }
};
struct LayerWeights {
    std::vector<float> attention_norm,feedforward_norm;
    Matrix query,key,value,attention_output,gate,up,down;
};
struct ModelWeights {
    ModelConfig config;
    Matrix embedding,output;
    std::vector<float> final_norm;
    std::vector<LayerWeights> layers;
};
struct LayerCache { std::vector<float> keys,values; };
struct SequenceState { std::size_t length=0; std::vector<LayerCache> layers; };
inline std::vector<float> embedding_lookup(const Matrix& table,std::size_t token) {
    if(token>=table.rows || !table.cols || table.rows>std::numeric_limits<std::size_t>::max()/table.cols ||
       table.values.size()!=table.rows*table.cols)
        throw std::invalid_argument("invalid embedding token or storage");
    auto first=table.values.begin()+static_cast<std::ptrdiff_t>(token*table.cols);
    return {first,first+static_cast<std::ptrdiff_t>(table.cols)};
}

inline std::vector<float> rms_norm(std::span<const float> x,std::span<const float> weights,float epsilon) {
    if(x.empty() || x.size()!=weights.size() || !std::isfinite(epsilon) || epsilon<=0)
        throw std::invalid_argument("invalid RMSNorm dimensions or epsilon");
    double sum=0;for(auto value:x)sum+=double(value)*value;
    const float inverse=static_cast<float>(1/std::sqrt(sum/static_cast<double>(x.size())+epsilon));
    std::vector<float> out(x.size());
    for(std::size_t i=0;i<x.size();++i)out[i]=x[i]*inverse*weights[i];
    return out;
}

inline void rotary(std::span<float> x,std::size_t heads,std::size_t dim,std::size_t position,float theta,bool interleaved) {
    if(!heads || !dim || dim%2 || heads>std::numeric_limits<std::size_t>::max()/dim || x.size()!=heads*dim ||
       !std::isfinite(theta) || theta<=0)throw std::invalid_argument("invalid RoPE dimensions");
    for(std::size_t h=0;h<heads;++h)for(std::size_t i=0;i<dim/2;++i) {
        const float angle=static_cast<float>(position)/std::pow(theta,static_cast<float>(2*i)/static_cast<float>(dim));
        const float cosine=std::cos(angle),sine=std::sin(angle);
        const auto a=h*dim+(interleaved?2*i:i),b=h*dim+(interleaved?2*i+1:i+dim/2);
        const float left=x[a],right=x[b];x[a]=left*cosine-right*sine;x[b]=right*cosine+left*sine;
    }
}

struct Projections { std::vector<float> query,key,value; };
inline Projections project_attention(const LayerWeights& layer,std::span<const float> normalized,const ModelConfig& c) {
    c.validate();
    if(layer.query.rows!=c.hidden || layer.key.rows!=c.kv_width() || layer.value.rows!=c.kv_width())
        throw std::invalid_argument("attention projection output shape mismatch");
    return {layer.query.multiply(normalized),layer.key.multiply(normalized),layer.value.multiply(normalized)};
}

inline std::vector<float> grouped_attention(std::span<const float> q,std::span<const float> keys,
                                            std::span<const float> values,std::size_t positions,const ModelConfig& c) {
    c.validate();const auto dim=c.head_dim(),width=c.kv_width();
    if(!positions || positions>c.context || positions>std::numeric_limits<std::size_t>::max()/width ||
       q.size()!=c.hidden || keys.size()!=positions*width || values.size()!=keys.size())
        throw std::invalid_argument("invalid attention storage dimensions");
    std::vector<float> out(c.hidden),scores(positions);
    for(std::size_t h=0;h<c.heads;++h) {
        const auto kv_head=h/(c.heads/c.kv_heads);
        float maximum=-std::numeric_limits<float>::infinity();
        for(std::size_t p=0;p<positions;++p) {
            scores[p]=quant::dot_f32(q.subspan(h*dim,dim),keys.subspan(p*width+kv_head*dim,dim))/std::sqrt(static_cast<float>(dim));
            maximum=std::max(maximum,scores[p]);
        }
        double sum=0;for(auto& score:scores){score=std::exp(score-maximum);sum+=score;}
        for(std::size_t i=0;i<dim;++i) {
            double weighted=0;for(std::size_t p=0;p<positions;++p)weighted+=double(scores[p])*values[p*width+kv_head*dim+i];
            out[h*dim+i]=static_cast<float>(weighted/sum);
        }
    }
    return out;
}

inline std::vector<float> project_attention_output(const LayerWeights& layer,std::span<const float> attention,const ModelConfig& c) {
    if(layer.attention_output.rows!=c.hidden || layer.attention_output.cols!=c.hidden)
        throw std::invalid_argument("attention output projection shape mismatch");
    return layer.attention_output.multiply(attention);
}

inline std::vector<float> feedforward(const LayerWeights& layer,std::span<const float> normalized,const ModelConfig& c) {
    if(layer.gate.rows!=c.intermediate || layer.up.rows!=c.intermediate ||
       layer.down.rows!=c.hidden || layer.down.cols!=c.intermediate)
        throw std::invalid_argument("feed-forward matrix shape mismatch");
    auto gate=layer.gate.multiply(normalized),up=layer.up.multiply(normalized);
    for(std::size_t i=0;i<gate.size();++i) {
        const float small=std::exp(-std::abs(gate[i]));
        const float sigmoid=gate[i]>=0?1/(1+small):small/(1+small);
        gate[i]*=sigmoid*up[i];
    }
    return layer.down.multiply(gate);
}

inline std::vector<float> tensor_floats(const gguf::TensorView& tensor) {
    const auto& info=tensor.info();const auto bytes=tensor.bytes();
    std::size_t count=1;for(auto dim:info.dimensions)count*=static_cast<std::size_t>(dim);
    std::vector<float> out(count);
    if(info.type==0) {gguf::Reader r(bytes);for(auto& x:out)x=r.f32();}
    else if(info.type==1)for(std::size_t i=0;i<count;++i)out[i]=quant::half_to_float(quant::u16(bytes,2*i));
    else if(info.type==2)quant::dequantize_q4_0(bytes,out);
    else if(info.type==8)quant::dequantize_q8_0(bytes,out);
    else if(info.type==12)quant::dequantize_q4_k(bytes,out);
    else throw std::invalid_argument("unsupported native weight type");
    for(auto x:out)if(!std::isfinite(x))throw std::invalid_argument("nonfinite model weight");
    return out;
}
inline ModelWeights load_weights(const std::filesystem::path& path) {
    gguf::ModelFile file(path);const auto& doc=file.document();auto shape=gguf::llama_shape(doc);
    gguf::validate_llama_tensors(doc,shape);
    ModelWeights weights;
    weights.config={static_cast<std::size_t>(shape.vocab),static_cast<std::size_t>(shape.hidden),
                    static_cast<std::size_t>(shape.intermediate),static_cast<std::size_t>(shape.layers),
                    static_cast<std::size_t>(shape.heads),static_cast<std::size_t>(shape.kv_heads),
                    static_cast<std::size_t>(shape.context),static_cast<float>(shape.rms_epsilon),static_cast<float>(shape.rope_theta)};
    if(auto it=doc.metadata.find("tvara.rope.layout");it!=doc.metadata.end()) {
        auto layout=std::get_if<std::string>(&it->second.data);
        if(!layout || (*layout!="split" && *layout!="interleaved"))throw std::invalid_argument("invalid RoPE layout");
        weights.config.interleaved_rope=*layout=="interleaved";
    }
    weights.config.validate();
    auto matrix=[&](const std::string& name) {
        auto tensor=file.tensor(name);const auto& dims=tensor.info().dimensions;
        if(dims.size()!=2)throw std::invalid_argument("projection must have rank two");
        Matrix m(static_cast<std::size_t>(dims[1]),static_cast<std::size_t>(dims[0]));
        m.values=tensor_floats(tensor);return m;
    };
    weights.embedding=matrix("token_embd.weight");weights.final_norm=tensor_floats(file.tensor("output_norm.weight"));
    if(doc.tensors.contains("output.weight"))weights.output=matrix("output.weight");
    for(std::size_t i=0;i<weights.config.layers;++i) {
        auto p="blk."+std::to_string(i)+".";LayerWeights layer;
        layer.attention_norm=tensor_floats(file.tensor(p+"attn_norm.weight"));
        layer.feedforward_norm=tensor_floats(file.tensor(p+"ffn_norm.weight"));
        layer.query=matrix(p+"attn_q.weight");layer.key=matrix(p+"attn_k.weight");layer.value=matrix(p+"attn_v.weight");
        layer.attention_output=matrix(p+"attn_output.weight");layer.gate=matrix(p+"ffn_gate.weight");
        layer.up=matrix(p+"ffn_up.weight");layer.down=matrix(p+"ffn_down.weight");weights.layers.push_back(std::move(layer));
    }
    return weights;
}
inline std::vector<float> execute_token(const ModelWeights& w,SequenceState& state,std::size_t token) {
    const auto& c=w.config;c.validate();
    if(token>=c.vocab || state.length>=c.context || w.layers.size()!=c.layers || state.layers.size()!=c.layers)
        throw std::invalid_argument("invalid sequence token context or layer state");
    if(state.length>std::numeric_limits<std::size_t>::max()/c.kv_width())
        throw std::invalid_argument("layer KV length overflow");
    const auto cache_length=state.length*c.kv_width();
    for(const auto& cache:state.layers)
        if(cache.keys.size()!=cache_length || cache.values.size()!=cache_length)
            throw std::invalid_argument("inconsistent layer KV length");
    auto x=embedding_lookup(w.embedding,token);
    try {
        for(std::size_t i=0;i<c.layers;++i) {
            auto& cache=state.layers[i];const auto& layer=w.layers[i];
            auto normalized=rms_norm(x,layer.attention_norm,c.rms_epsilon);
            auto projected=project_attention(layer,normalized,c);
            rotary(projected.query,c.heads,c.head_dim(),state.length,c.rope_theta,c.interleaved_rope);
            rotary(projected.key,c.kv_heads,c.head_dim(),state.length,c.rope_theta,c.interleaved_rope);
            cache.keys.insert(cache.keys.end(),projected.key.begin(),projected.key.end());
            cache.values.insert(cache.values.end(),projected.value.begin(),projected.value.end());
            auto attention=grouped_attention(projected.query,cache.keys,cache.values,state.length+1,c);
            auto residual=project_attention_output(layer,attention,c);
            for(std::size_t j=0;j<c.hidden;++j)x[j]+=residual[j];
            normalized=rms_norm(x,layer.feedforward_norm,c.rms_epsilon);residual=feedforward(layer,normalized,c);
            for(std::size_t j=0;j<c.hidden;++j)x[j]+=residual[j];
        }
        auto normalized=rms_norm(x,w.final_norm,c.rms_epsilon);
        auto logits=(w.output.rows?w.output:w.embedding).multiply(normalized);
        ++state.length;return logits;
    } catch(...) {
        for(auto& cache:state.layers) {cache.keys.resize(cache_length);cache.values.resize(cache_length);}
        throw;
    }
}

class ModelExecutor {
public:
    explicit ModelExecutor(ModelWeights weights):weights_(std::move(weights)) {weights_.config.validate();}
    const ModelConfig& config() const {return weights_.config;}
    SequenceState new_sequence() const {return {0,std::vector<LayerCache>(config().layers)};}
    std::vector<float> decode(std::size_t token,SequenceState& state) const {return execute_token(weights_,state,token);}
    std::vector<float> prefill(std::span<const std::size_t> tokens,SequenceState& state) const {
        if(tokens.empty() || state.length>config().context || tokens.size()>config().context-state.length)
            throw std::invalid_argument("invalid prefill length");
        for(auto token:tokens)if(token>=config().vocab)throw std::invalid_argument("invalid prefill token");
        std::vector<float> logits;
        for(auto token:tokens)logits=decode(token,state);
        return logits;
    }
    std::vector<float> forward(std::span<const std::size_t> tokens) const {
        if(tokens.empty() || tokens.size()>config().context || tokens.size()>std::numeric_limits<std::size_t>::max()/config().vocab)
            throw std::invalid_argument("invalid full forward length");
        auto state=new_sequence();std::vector<float> out;out.reserve(tokens.size()*config().vocab);
        for(auto token:tokens){auto logits=decode(token,state);out.insert(out.end(),logits.begin(),logits.end());}
        return out;
    }
private:
    ModelWeights weights_;
};

// MODEL_CODE
}  // namespace tvara
