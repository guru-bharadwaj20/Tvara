#include "tvara/kv.hpp"
#include <functional>
#include <iostream>
using namespace tvara;
void check(bool c){if(!c)throw std::runtime_error("KV check failed");}
void rejects(const std::function<void()>& f){bool failed=false;try{f();}catch(const std::invalid_argument&){failed=true;}check(failed);}
void test_geometry(){CacheGeometry g{2,2,4,4,8};g.validate();check(g.elements_per_block()==128 && kv_offset(g,1,1,3)==120);
    g.head_dim=std::numeric_limits<std::size_t>::max();rejects([&]{g.validate();});}
void test_allocator(){BlockPool pool({1,1,4,4,2});auto first=pool.allocate();pool.data(first)[0]=9;pool.free(first);
    rejects([&]{pool.data(first);});auto second=pool.allocate();check(second.id==first.id && second.generation!=first.generation && pool.data(second)[0]==0);pool.free(second);check(pool.available()==2);}

void test_tables(){PagedCache cache({1,1,4,4,4});auto a=cache.create_sequence(),b=cache.create_sequence();check(a!=b && cache.sequence(a).length==0);rejects([&]{cache.physical_block(a,0);});}

// EXTRA_TESTS
int main(){try{test_geometry();
    test_allocator();
        test_tables();
        // RUN_TESTS
    std::cout<<"Paged KV checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
