#include "tvara/sampling.hpp"
#include <iostream>
using namespace tvara;
void check(bool c){if(!c)throw std::runtime_error("sampling check failed");}
void test_temperature(){
    std::vector<float> logits{1000,1001,-std::numeric_limits<float>::infinity()};
    auto p=probabilities(logits,1);check(std::abs(p[1]-1/(1+std::exp(-1)))<1e-12 && p[2]==0);
    std::mt19937_64 rng(1);check(sample(logits,0,rng)==1);
}
// EXTRA_TESTS
int main(){try{test_temperature();
    // RUN_TESTS
    std::cout<<"Sampling checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
