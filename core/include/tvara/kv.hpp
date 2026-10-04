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
struct BlockHandle {std::size_t id=0;std::uint64_t generation=0;};
class BlockPool {
    struct Block {std::vector<float> values;bool in_use=false;std::uint64_t generation=0;};
public:
    explicit BlockPool(CacheGeometry geometry):geometry_(geometry){
        geometry.validate();blocks_.resize(geometry.max_blocks);free_.reserve(geometry.max_blocks);
        for(auto& block:blocks_)block.values.resize(geometry.elements_per_block());
        for(std::size_t i=geometry.max_blocks;i>0;--i)free_.push_back(i-1);
    }
    BlockHandle allocate(){
        if(free_.empty())throw std::runtime_error("KV cache exhausted");
        const auto id=free_.back();auto& block=blocks_[id];
        if(block.generation==std::numeric_limits<std::uint64_t>::max())throw std::runtime_error("KV generation exhausted");
        free_.pop_back();block.in_use=true;++block.generation;std::fill(block.values.begin(),block.values.end(),0);
        return {id,block.generation};
    }
    void free(BlockHandle handle){auto& block=validated(handle);block.in_use=false;free_.push_back(handle.id);}
    std::span<float> data(BlockHandle handle){return validated(handle).values;}
    std::span<const float> data(BlockHandle handle)const{return validated(handle).values;}
    const CacheGeometry& geometry()const{return geometry_;}
    std::size_t available()const{return free_.size();}
    // POOL_CODE
private:
    const Block& validated(BlockHandle handle)const{
        if(handle.id>=blocks_.size() || !blocks_[handle.id].in_use || blocks_[handle.id].generation!=handle.generation)
            throw std::invalid_argument("stale or invalid KV handle");
        return blocks_[handle.id];
    }
    Block& validated(BlockHandle handle){return const_cast<Block&>(std::as_const(*this).validated(handle));}
    CacheGeometry geometry_;std::vector<Block> blocks_;std::vector<std::size_t> free_;
};

// KV_CODE
}
