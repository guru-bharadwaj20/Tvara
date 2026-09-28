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

// MODEL_CODE
}  // namespace tvara
