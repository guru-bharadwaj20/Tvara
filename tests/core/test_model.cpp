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

void test_rope() {
    std::vector<float> x{1,2,3,4},before=x;rotary(x,1,4,0,10000,false);check(x==before);
    rotary(x,1,4,1,10000,false);
    close(x[0],std::cos(1.f)-3*std::sin(1.f));
    close(x[2],3*std::cos(1.f)+std::sin(1.f));
    std::vector<float> interleaved{1,3,2,4};rotary(interleaved,1,4,1,10000,true);
    close(interleaved[0],x[0]);close(interleaved[1],x[2]);close(interleaved[2],x[1]);close(interleaved[3],x[3]);
}

void test_projections() {
    auto c=config();LayerWeights layer;layer.query=Matrix(8,8);layer.key=Matrix(4,8);layer.value=Matrix(4,8);
    for(std::size_t i=0;i<8;++i)layer.query.values[i*8+i]=1;
    std::vector<float> x(8,2);auto projected=project_attention(layer,x,c);
    check(projected.query==x && projected.key.size()==4 && projected.value.size()==4);
}

void test_attention() {
    auto c=config();std::vector<float> q(8),keys(8),values{1,2,3,4,5,6,7,8};
    auto out=grouped_attention(q,keys,values,2,c);
    for(std::size_t h=0;h<2;++h)for(std::size_t i=0;i<4;++i)close(out[h*4+i],static_cast<float>(3+i));
    rejects([&]{grouped_attention(q,keys,values,0,c);});
}

void test_output_projection() {
    auto c=config();LayerWeights layer;layer.attention_output=Matrix(8,8);
    for(std::size_t i=0;i<8;++i)layer.attention_output.values[i*8+i]=2;
    std::vector<float> x(8,3);auto out=project_attention_output(layer,x,c);for(auto v:out)close(v,6);
}

void test_feedforward() {
    auto c=config();LayerWeights layer;layer.gate=Matrix(12,8);layer.up=Matrix(12,8);layer.down=Matrix(8,12);
    layer.gate.values[0]=1;layer.up.values[0]=2;layer.down.values[0]=3;
    std::vector<float> x(8,1);auto out=feedforward(layer,x,c);close(out[0],6/(1+std::exp(-1.f)));
    for(std::size_t i=1;i<out.size();++i)check(out[i]==0);
}

// EXTRA_TESTS
int main() {
    try {
        test_configuration();
        test_embedding();
        test_norm();
        test_rope();
        test_projections();
        test_attention();
        test_output_projection();
        test_feedforward();
        // RUN_TESTS
        std::cout << "Native model checks passed\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
