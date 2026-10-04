#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
namespace tvara {
struct CacheGeometry {
    std::size_t layers=1,kv_heads=1,head_dim=4,tokens_per_block=16,max_blocks=64;
    std::size_t kv_width()const {
        if(!kv_heads || !head_dim || kv_heads>std::numeric_limits<std::size_t>::max()/head_dim)throw std::invalid_argument("KV width overflow");
        return kv_heads*head_dim;
    }
    std::size_t elements_per_block()const{
        std::size_t count=kv_width();for(auto dim:{layers,tokens_per_block,std::size_t(2)}){
            if(!dim || count>std::numeric_limits<std::size_t>::max()/dim)throw std::invalid_argument("KV block overflow");count*=dim;}
        return count;
    }
    void validate()const{
        auto elements=elements_per_block();
        if(!max_blocks || max_blocks>1000000 || elements>std::numeric_limits<std::size_t>::max()/sizeof(float)/max_blocks ||
           tokens_per_block>std::numeric_limits<std::size_t>::max()/max_blocks)throw std::invalid_argument("KV capacity overflow");
    }
};
inline std::size_t kv_offset(const CacheGeometry& g,std::size_t layer,std::size_t kind,std::size_t slot){
    if(layer>=g.layers || kind>1 || slot>=g.tokens_per_block)throw std::invalid_argument("KV position outside block");
    return ((layer*2+kind)*g.tokens_per_block+slot)*g.kv_width();
}
// KV_CODE
}
