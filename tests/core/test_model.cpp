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
// EXTRA_TESTS
int main() {
    try {
        test_configuration();
        // RUN_TESTS
        std::cout << "Native model checks passed\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
