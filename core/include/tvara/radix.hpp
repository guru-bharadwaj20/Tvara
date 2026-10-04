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
// METHODS
private:
    PagedCache& cache_;
    std::string isolation_;
    std::map<std::uint64_t,RadixNode> nodes_;
    std::uint64_t next_=0,clock_=0;
};
}
