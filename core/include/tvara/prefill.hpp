#pragma once
#include "tvara/model.hpp"
#include "tvara/simd.hpp"
namespace tvara {
// X is token-major [tokens, input]; W is output-major [output, input].
inline void prefill_shapes(const Matrix& weights,std::span<const float> x,std::size_t tokens) {
    if(!tokens || !weights.rows || !weights.cols || tokens>std::numeric_limits<std::size_t>::max()/weights.cols ||
       weights.rows>std::numeric_limits<std::size_t>::max()/weights.cols || weights.values.size()!=weights.rows*weights.cols ||
       x.size()!=tokens*weights.cols || tokens>std::numeric_limits<std::size_t>::max()/weights.rows)
        throw std::invalid_argument("invalid prefill shapes");
}
inline std::vector<float> prefill_reference(const Matrix& weights,std::span<const float> x,std::size_t tokens) {
    prefill_shapes(weights,x,tokens);std::vector<float> out(tokens*weights.rows);
    for(std::size_t t=0;t<tokens;++t)for(std::size_t r=0;r<weights.rows;++r){double sum=0;
        for(std::size_t k=0;k<weights.cols;++k)sum+=double(x[t*weights.cols+k])*weights.values[r*weights.cols+k];
        out[t*weights.rows+r]=static_cast<float>(sum);}
    return out;
}
struct PrefillTile {std::size_t tokens=8,outputs=16,inner=64;};
inline std::vector<float> prefill_blocked(const Matrix& weights,std::span<const float> x,std::size_t tokens,PrefillTile tile={}) {
    prefill_shapes(weights,x,tokens);if(!tile.tokens || !tile.outputs || !tile.inner)throw std::invalid_argument("zero prefill tile");
    tile.tokens=std::min(tile.tokens,tokens);tile.outputs=std::min(tile.outputs,weights.rows);tile.inner=std::min(tile.inner,weights.cols);
    std::vector<float> output(tokens*weights.rows);
    for(std::size_t tb=0;tb<tokens;tb+=tile.tokens)for(std::size_t rb=0;rb<weights.rows;rb+=tile.outputs){
        const auto nt=std::min(tile.tokens,tokens-tb),nr=std::min(tile.outputs,weights.rows-rb);
        std::vector<double> sums(nt*nr);
        for(std::size_t kb=0;kb<weights.cols;kb+=tile.inner){const auto end=std::min(weights.cols,kb+tile.inner);
            for(std::size_t t=0;t<nt;++t)for(std::size_t r=0;r<nr;++r){auto& sum=sums[t*nr+r];
                for(std::size_t k=kb;k<end;++k)sum+=double(x[(tb+t)*weights.cols+k])*weights.values[(rb+r)*weights.cols+k];}}
        for(std::size_t t=0;t<nt;++t)for(std::size_t r=0;r<nr;++r)output[(tb+t)*weights.rows+rb+r]=static_cast<float>(sums[t*nr+r]);}
    return output;
}

struct ActivationTiles {std::size_t tokens=0,cols=0;std::vector<quant::Q8Activation> blocks;};
inline ActivationTiles quantize_prefill(std::span<const float> x,std::size_t tokens,std::size_t cols) {
    if(!tokens || !cols || cols%32 || tokens>std::numeric_limits<std::size_t>::max()/cols || x.size()!=tokens*cols)throw std::invalid_argument("invalid quantized activation shape");
    ActivationTiles tile{tokens,cols,{}};tile.blocks.reserve(x.size()/32);
    for(std::size_t t=0;t<tokens;++t){auto row=quant::quantize_activations(x.subspan(t*cols,cols));tile.blocks.insert(tile.blocks.end(),row.begin(),row.end());}
    return tile;
}
inline std::vector<float> prefill_q8(std::span<const std::byte> weights,std::size_t rows,const ActivationTiles& tile,bool scalar=false) {
    if(!rows || !tile.tokens || !tile.cols || tile.cols%32 || rows>std::numeric_limits<std::size_t>::max()/(tile.cols/32)/34 ||
       weights.size()!=rows*(tile.cols/32)*34 || tile.tokens>std::numeric_limits<std::size_t>::max()/(tile.cols/32) ||
       tile.blocks.size()!=tile.tokens*(tile.cols/32) || rows>std::numeric_limits<std::size_t>::max()/tile.tokens)throw std::invalid_argument("invalid Q8 prefill shape");
    const auto blocks=tile.cols/32,bytes=blocks*34;KernelDispatch dispatch(scalar);std::vector<float> output(tile.tokens*rows);
    for(std::size_t t=0;t<tile.tokens;++t){auto activation=std::span<const quant::Q8Activation>(tile.blocks).subspan(t*blocks,blocks);
        for(std::size_t r=0;r<rows;++r)output[t*rows+r]=dispatch.q8(weights.subspan(r*bytes,bytes),activation);}
    return output;
}

inline std::vector<float> prefill_weight_prefetch(const Matrix& w,std::span<const float> x,std::size_t tokens,bool prefetch){
    prefill_shapes(w,x,tokens);std::vector<float> out(tokens*w.rows);
    for(std::size_t t=0;t<tokens;++t)for(std::size_t r=0;r<w.rows;++r){
#if defined(__GNUC__) || defined(__clang__)
        if(prefetch && r+1<w.rows)__builtin_prefetch(w.values.data()+(r+1)*w.cols,0,1);
#else
        (void)prefetch;
#endif
        double sum=0;for(std::size_t k=0;k<w.cols;++k)sum+=double(x[t*w.cols+k])*w.values[r*w.cols+k];out[t*w.rows+r]=static_cast<float>(sum);}
    return out;
}

// PREFILL_CODE
}
