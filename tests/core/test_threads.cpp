#include "tvara/thread_pool.hpp"
#include <iostream>
using namespace tvara;
void check(bool c){if(!c)throw std::runtime_error("thread-pool check failed");}
void test_pool(){ThreadPool pool(4);std::vector<int> seen(4);pool.execute([&](std::size_t i){seen[i]++;});for(auto v:seen)check(v==1);}
// EXTRA_TESTS
int main(){try{test_pool();
    // RUN_TESTS
    std::cout<<"Thread checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
