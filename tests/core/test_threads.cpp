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

void test_barrier(){
    ThreadPool pool(4);SpinBarrier barrier(4);std::atomic<int> count{0};
    pool.execute([&](std::size_t){for(int iteration=0;iteration<100;++iteration){count.fetch_add(1);barrier.arrive_and_wait();check(count.load()==(iteration+1)*4);barrier.arrive_and_wait();}});
    check(count.load()==400);
}

void test_worker_config(){check(parse_thread_count("3")==3);check(parse_thread_count("",0)==1);bool bad=false;try{parse_thread_count("-2");}catch(const std::invalid_argument&){bad=true;}check(bad);}

// EXTRA_TESTS
int main(){try{test_pool();
    test_partitions();
        test_barrier();
        test_worker_config();
        // RUN_TESTS
    std::cout<<"Thread checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
