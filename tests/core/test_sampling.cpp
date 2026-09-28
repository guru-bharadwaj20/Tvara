#include "tvara/sampling.hpp"
#include <iostream>
using namespace tvara;
void check(bool c){if(!c)throw std::runtime_error("sampling check failed");}
void test_temperature(){
    std::vector<float> logits{1000,1001,-std::numeric_limits<float>::infinity()};
    auto p=probabilities(logits,1);check(std::abs(p[1]-1/(1+std::exp(-1)))<1e-12 && p[2]==0);
    std::mt19937_64 rng(1);check(sample(logits,0,rng)==1);
}
void test_filters(){
    std::vector<double> p{0.5,0.3,0.2};filter_probabilities(p,0,0.6,0);check(p[2]==0 && std::abs(p[0]-0.625)<1e-12);
    p={0.5,0.3,0.2};filter_probabilities(p,1,1,0);check(p[0]==1 && p[1]==0);
    p={0.5,0.3,0.2};filter_probabilities(p,0,1,0.5);check(p[2]==0 && p[1]>0);
}

// EXTRA_TESTS
int main(){try{test_temperature();
    test_filters();
        // RUN_TESTS
    std::cout<<"Sampling checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
