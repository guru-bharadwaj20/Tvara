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
                // SPLIT
                throw std::invalid_argument("divergent edge requires split support");
            }
            current=child.id;offset+=common;
        }
        return current;
    }

// METHODS
private:
    PagedCache& cache_;
    std::string isolation_;
    std::map<std::uint64_t,RadixNode> nodes_;
    std::uint64_t next_=0,clock_=0;
};
}
