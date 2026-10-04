#include "tvara/radix.hpp"
#include <cassert>
#include <iostream>
using namespace tvara;
int main(){
 CacheGeometry g{1,1,1,4,64};PagedCache cache(g);RadixCache tree(cache,"fixture-model/tokenizer/tenant-a");
 assert(tree.size()==0);assert(tree.node(0).edge.empty());
assert(tree.lookup(std::vector<std::int32_t>{1,2}).tokens==0);
const std::vector<std::int32_t> prompt{1,2,3,4,5,6};auto a=tree.insert(prompt);assert(tree.insert(prompt)==a);assert(tree.lookup(prompt).tokens==6);
auto b=tree.insert(std::vector<std::int32_t>{1,2,7});assert(b!=a);assert(tree.lookup(std::vector<std::int32_t>{1,2,8}).tokens==2);assert(tree.insert(std::vector<std::int32_t>{1,2})!=a);
auto seq=cache.create_sequence();for(int i=0;i<6;++i){float f=static_cast<float>(i);cache.append(seq,std::span(&f,1),std::span(&f,1));}tree.attach(a,seq);cache.audit(tree.external_references());
auto full=tree.reuse(prompt);assert(full.second==4);assert(cache.physical_block(full.first,0).id==cache.physical_block(seq,0).id);cache.drop(full.first);
auto partial=tree.reuse(prompt,true);assert(partial.second==6);auto old=cache.physical_block(seq,5);float nine=9;cache.append(partial.first,std::span(&nine,1),std::span(&nine,1));assert(cache.physical_block(partial.first,5).id!=old.id);assert(cache.read(seq,0,5)[0]==5);cache.drop(partial.first);cache.audit(tree.external_references());
tree.pin(a);assert(tree.node(a).pins==1);auto shorter=tree.insert(std::vector<std::int32_t>{1});assert(tree.node(shorter).pins==1);tree.unpin(a);assert(tree.node(shorter).pins==0);bool underflow=false;try{tree.unpin(a);}catch(const std::invalid_argument&){underflow=true;}assert(underflow);
tree.touch(a);tree.touch(b);assert(tree.node(a).accessed<tree.node(b).accessed);
assert(tree.eviction_candidate()==a);tree.pin(a);assert(tree.eviction_candidate()==b);tree.unpin(a);
tree.pin(b);assert(tree.evict_one());assert(!tree.node(b).edge.empty());tree.unpin(b);cache.audit(tree.external_references());
auto c=tree.insert(prompt);tree.attach(c,seq);auto survivor=tree.reuse(prompt,true);tree.touch(b);assert(tree.evict_one());assert(cache.sequence(survivor.first).length==6);cache.append(survivor.first,std::span(&nine,1),std::span(&nine,1));assert(cache.read(seq,0,5)[0]==5);cache.drop(survivor.first);cache.audit(tree.external_references());
// TESTS
 std::cout<<"radix OK\n";
}
