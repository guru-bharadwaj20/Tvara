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

struct BlockTable {std::vector<BlockHandle> blocks;std::size_t length=0;};
class PagedCache {
public:
    explicit PagedCache(CacheGeometry geometry):pool_(geometry){}
    std::uint64_t create_sequence(){if(next_==std::numeric_limits<std::uint64_t>::max())throw std::runtime_error("sequence ID exhausted");const auto id=++next_;sequences_.emplace(id,BlockTable{});return id;}
    const BlockTable& sequence(std::uint64_t id)const{
        auto it=sequences_.find(id);if(it==sequences_.end())throw std::invalid_argument("unknown KV sequence");return it->second;
    }
    BlockHandle physical_block(std::uint64_t id,std::size_t position)const{
        const auto& table=sequence(id);if(position>=table.length)throw std::invalid_argument("KV position beyond sequence");return table.blocks.at(position/geometry().tokens_per_block);
    }
    const CacheGeometry& geometry()const{return pool_.geometry();}
    BlockPool& pool(){return pool_;}
    const BlockPool& pool()const{return pool_;}
    void append(std::uint64_t id,std::span<const float> keys,std::span<const float> values){
        auto& table=mutable_sequence(id);const auto& g=geometry();
        if(keys.size()!=g.layers*g.kv_width() || values.size()!=keys.size() || table.length>=g.max_blocks*g.tokens_per_block)
            throw std::invalid_argument("invalid paged KV append dimensions");
        const auto slot=table.length%g.tokens_per_block;
        if(slot==0){auto block=pool_.allocate();try{table.blocks.push_back(block);}catch(...){pool_.free(block);throw;}}
        // COPY_ON_WRITE
        auto block=pool_.data(table.blocks.back());
        for(std::size_t layer=0;layer<g.layers;++layer){
            std::copy_n(keys.begin()+static_cast<std::ptrdiff_t>(layer*g.kv_width()),g.kv_width(),block.begin()+static_cast<std::ptrdiff_t>(kv_offset(g,layer,0,slot)));
            std::copy_n(values.begin()+static_cast<std::ptrdiff_t>(layer*g.kv_width()),g.kv_width(),block.begin()+static_cast<std::ptrdiff_t>(kv_offset(g,layer,1,slot)));
        }
        ++table.length;
    }

std::vector<float> read(std::uint64_t id,std::size_t layer,std::size_t position,bool value=false)const{
        const auto handle=physical_block(id,position);const auto offset=kv_offset(geometry(),layer,value?1:0,position%geometry().tokens_per_block);
        auto bytes=pool_.data(handle).subspan(offset,geometry().kv_width());return {bytes.begin(),bytes.end()};
    }
    std::vector<float> read_all(std::uint64_t id,std::size_t layer,bool value=false)const{
        const auto length=sequence(id).length;if(layer>=geometry().layers)throw std::invalid_argument("invalid KV layer");
        std::vector<float> output;output.reserve(length*geometry().kv_width());
        for(std::size_t position=0;position<length;++position){auto values=read(id,layer,position,value);output.insert(output.end(),values.begin(),values.end());}
        return output;
    }

// CACHE_CODE
private:
    BlockTable& mutable_sequence(std::uint64_t id){return const_cast<BlockTable&>(std::as_const(*this).sequence(id));}
    BlockPool pool_;std::map<std::uint64_t,BlockTable> sequences_;std::uint64_t next_=0;
};

// KV_CODE
}
