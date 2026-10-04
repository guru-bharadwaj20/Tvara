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
// TESTS
 std::cout<<"radix OK\n";
}
