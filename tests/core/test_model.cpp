#include "tvara/model.hpp"
#include <functional>
#include <iostream>

using namespace tvara;
void check(bool condition) { if(!condition) throw std::runtime_error("native model check failed"); }
void close(float a,float b,float tolerance=1e-5f) { check(std::abs(a-b)<=tolerance*(1+std::abs(b))); }
void rejects(const std::function<void()>& fn) {
    bool failed=false;try{fn();}catch(const std::invalid_argument&){failed=true;}check(failed);
}
ModelConfig config() { return {16,8,12,1,2,1,32}; }
void test_configuration() {
    auto c=config();c.validate();check(c.head_dim()==4 && c.kv_width()==4);
    c.heads=3;rejects([&]{c.validate();});
    rejects([]{Matrix bad(std::numeric_limits<std::size_t>::max(),2);});
}
void test_embedding() {
    Matrix table(2,3);table.values={1,2,3,4,5,6};
    check(embedding_lookup(table,1)==std::vector<float>({4,5,6}));
    rejects([&]{embedding_lookup(table,2);});
}

void test_norm() {
    std::vector<float> x{1,-2,3},w{0.5f,2,-1};auto out=rms_norm(x,w,1e-6f);
    for(std::size_t i=0;i<3;++i)close(out[i],x[i]*w[i]/std::sqrt(14.f/3+1e-6f));
    std::fill(x.begin(),x.end(),0);out=rms_norm(x,w,1e-6f);for(auto v:out)check(v==0);
}

// EXTRA_TESTS
int main() {
    try {
        test_configuration();
        test_embedding();
        test_norm();
        // RUN_TESTS
        std::cout << "Native model checks passed\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
