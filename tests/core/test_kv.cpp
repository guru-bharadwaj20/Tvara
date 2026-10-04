#include "tvara/kv.hpp"
#include <functional>
#include <iostream>
using namespace tvara;
void check(bool c){if(!c)throw std::runtime_error("KV check failed");}
void rejects(const std::function<void()>& f){bool failed=false;try{f();}catch(const std::invalid_argument&){failed=true;}check(failed);}
void test_geometry(){CacheGeometry g{2,2,4,4,8};g.validate();check(g.elements_per_block()==128 && kv_offset(g,1,1,3)==120);
    g.head_dim=std::numeric_limits<std::size_t>::max();rejects([&]{g.validate();});}
// EXTRA_TESTS
int main(){try{test_geometry();
    // RUN_TESTS
    std::cout<<"Paged KV checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
