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
// PREFILL_CODE
}
