#include "prefill_support.hpp"
#include "tvara/attention.hpp"
int main(){double error=0;for(std::size_t dim:{2,6,8,14,32,64,128}){auto x=signal(dim,7),y=signal(dim,11);auto ref=tvara::attention_dot(x,y,true),out=tvara::attention_dot(x,y);require(std::abs(out-ref)<1e-9*(1+std::abs(ref)));error=std::max(error,std::abs(out-ref));}
    auto x=signal(128),y=signal(128,2);volatile double sink=0;auto scalar=median_us([&]{for(int i=0;i<10000;++i)sink=tvara::attention_dot(x,y,true);});auto vector=median_us([&]{for(int i=0;i<10000;++i)sink=tvara::attention_dot(x,y);});(void)sink;
    std::cout<<"{\"max_double_error\":"<<error<<",\"scalar_us_10000\":"<<scalar<<",\"dispatch_us_10000\":"<<vector<<",\"avx2\":"<<(tvara::use_avx2()?"true":"false")<<"}";
}
