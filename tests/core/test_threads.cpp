#include "tvara/thread_pool.hpp"
#include "tvara/threaded_kernels.hpp"
#include <iostream>
using namespace tvara;
void check(bool c){if(!c)throw std::runtime_error("thread-pool check failed");}
void test_pool(){ThreadPool pool(4);std::vector<int> seen(4);pool.execute([&](std::size_t i){seen[i]++;});for(auto v:seen)check(v==1);}
void test_partitions(){
    std::vector<int> ownership(7);for(std::size_t worker=0;worker<4;++worker){auto [a,b]=partition_rows(7,worker,4);for(auto row=a;row<b;++row)ownership[row]++;}
    for(auto v:ownership)check(v==1);
    ThreadPool pool(4);std::vector<float> w(21,1),x(3,2),out(7);threaded_matvec(pool,w,7,3,x,out);for(auto v:out)check(v==6);
}

// EXTRA_TESTS
int main(){try{test_pool();
    test_partitions();
        // RUN_TESTS
    std::cout<<"Thread checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
