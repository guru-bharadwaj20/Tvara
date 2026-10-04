#include "tvara/radix.hpp"
#include <cassert>
#include <iostream>
using namespace tvara;
int main(){
 CacheGeometry g{1,1,1,4,64};PagedCache cache(g);RadixCache tree(cache,"fixture-model/tokenizer/tenant-a");
 assert(tree.size()==0);assert(tree.node(0).edge.empty());
// TESTS
 std::cout<<"radix OK\n";
}
