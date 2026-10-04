#include "tvara/simd.hpp"
#include "tvara/thread_pool.hpp"
#include <chrono>
#include <iostream>
int main(){
    using clock=std::chrono::steady_clock;constexpr int repetitions=1000;tvara::ThreadPool pool(2);
    for(int i=0;i<20;++i)pool.execute([](std::size_t){});
    auto start=clock::now();for(int i=0;i<repetitions;++i)pool.execute([](std::size_t){});
    auto end=clock::now();double worker=std::chrono::duration<double,std::micro>(end-start).count()/repetitions;
    std::vector<std::byte> weights(34);weights[1]=std::byte(0x3c);std::vector<float>x(32,1);auto a=tvara::quant::quantize_activations(x);
    tvara::KernelDispatch dispatch;float checksum=0;start=clock::now();
    for(int i=0;i<100000;++i)checksum+=dispatch.q8(weights,a);end=clock::now();
    double dot=std::chrono::duration<double,std::nano>(end-start).count()/100000;
    std::cout<<"{\"workers\":2,\"repetitions\":"<<repetitions<<",\"noop_worker_us\":"<<worker
             <<",\"q8_dot_ns\":"<<dot<<",\"avx2_dispatch\":"<<(dispatch.avx2?"true":"false")<<",\"checksum\":"<<checksum<<"}\n";
}
