#pragma once
#include "tvara/kv.hpp"
#include <map>
#include <optional>
#include <string>
#include <vector>
namespace tvara {
struct RadixNode {
    std::uint64_t id=0,parent=0;
    std::vector<std::int32_t> edge;
    std::map<std::int32_t,std::uint64_t> children;
    std::optional<BlockTable> kv;
    std::size_t pins=0;
    std::uint64_t accessed=0;
};
struct PrefixMatch {std::uint64_t node=0;std::size_t tokens=0;};
class RadixCache {
public:
    explicit RadixCache(PagedCache& cache,std::string isolation):cache_(cache),isolation_(std::move(isolation)) {
        if(isolation_.empty())throw std::invalid_argument("prefix isolation key is required");
        nodes_.emplace(0,RadixNode{});
    }
    RadixCache(const RadixCache&)=delete;
    RadixCache& operator=(const RadixCache&)=delete;
    const RadixNode& node(std::uint64_t id)const{return nodes_.at(id);}
    std::size_t size()const{return nodes_.size()-1;}
    const std::string& isolation()const{return isolation_;}

    PrefixMatch lookup(std::span<const std::int32_t> tokens)const{
        std::uint64_t current=0;std::size_t offset=0;PrefixMatch result{};
        while(offset<tokens.size()){
            const auto& parent=nodes_.at(current);auto it=parent.children.find(tokens[offset]);
            if(it==parent.children.end())break;
            const auto& child=nodes_.at(it->second);std::size_t common=0;
            while(common<child.edge.size() && offset+common<tokens.size() && child.edge[common]==tokens[offset+common])++common;
            offset+=common;result={child.id,offset};
            if(common!=child.edge.size())break;current=child.id;
        }
        return result;
    }


    std::uint64_t insert(std::span<const std::int32_t> tokens){
        if(tokens.empty())throw std::invalid_argument("cannot cache empty prefix");
        std::uint64_t current=0;std::size_t offset=0;
        while(offset<tokens.size()){
            auto& parent=nodes_.at(current);auto found=parent.children.find(tokens[offset]);
            if(found==parent.children.end()){
                if(next_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("radix ID exhausted");const auto id=++next_;RadixNode child;child.id=id;child.parent=current;
                child.edge.assign(tokens.begin()+static_cast<std::ptrdiff_t>(offset),tokens.end());
                nodes_.emplace(id,std::move(child));parent.children.emplace(tokens[offset],id);return id;
            }
            auto& child=nodes_.at(found->second);std::size_t common=0;
            while(common<child.edge.size() && offset+common<tokens.size() && child.edge[common]==tokens[offset+common])++common;
            if(common!=child.edge.size()){
                if(next_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("radix ID exhausted");const auto split=++next_;RadixNode middle;middle.id=split;middle.parent=current;
                middle.pins=child.pins;middle.edge.assign(child.edge.begin(),child.edge.begin()+static_cast<std::ptrdiff_t>(common));
                child.edge.erase(child.edge.begin(),child.edge.begin()+static_cast<std::ptrdiff_t>(common));
                middle.children.emplace(child.edge.front(),child.id);child.parent=split;
                parent.children[tokens[offset]]=split;nodes_.emplace(split,std::move(middle));
                current=split;offset+=common;continue;
            }
            current=child.id;offset+=common;
        }
        return current;
    }


    void attach(std::uint64_t id,std::uint64_t sequence){
        auto& entry=nodes_.at(id);const auto source=cache_.sequence(sequence);
        std::size_t depth=0;for(auto cursor=id;cursor;cursor=nodes_.at(cursor).parent)depth+=nodes_.at(cursor).edge.size();
        if(source.length!=depth)throw std::invalid_argument("token and KV prefix lengths differ");
        std::size_t retained=0;
        try{for(auto block:source.blocks){cache_.pool().retain(block);++retained;}}
        catch(...){for(std::size_t i=0;i<retained;++i)cache_.pool().release(source.blocks[i]);throw;}
        if(entry.kv)for(auto block:entry.kv->blocks)cache_.pool().release(block);
        entry.kv=source;
    }
    ~RadixCache(){for(auto& [id,entry]:nodes_){(void)id;if(entry.kv)for(auto block:entry.kv->blocks)cache_.pool().release(block);}}
    std::vector<std::size_t> external_references()const{
        std::vector<std::size_t> counts(cache_.geometry().max_blocks);
        for(const auto& [id,entry]:nodes_){(void)id;if(entry.kv)for(auto block:entry.kv->blocks)++counts.at(block.id);}
        return counts;
    }


    std::pair<std::uint64_t,std::size_t> reuse(std::span<const std::int32_t> tokens,bool partial=false){
        auto match=lookup(tokens);
        if(!match.node || !nodes_.at(match.node).kv)return {cache_.create_sequence(),0};
        std::size_t length=match.tokens;
        if(!partial)length-=length%cache_.geometry().tokens_per_block;
        return {cache_.import_sequence(*nodes_.at(match.node).kv,length),length};
    }

// METHODS
private:
    PagedCache& cache_;
    std::string isolation_;
    std::map<std::uint64_t,RadixNode> nodes_;
    std::uint64_t next_=0,clock_=0;
};
}
