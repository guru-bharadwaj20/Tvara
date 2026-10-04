#include "prefill_support.hpp"
#include "tvara/prefill.hpp"
#include "tvara/paged_model.hpp"
#include "prefill_model_fixture.hpp"
int main(){tvara::Matrix w(128,256);w.values=signal(w.values.size(),0.1f);std::cout<<"{\"scope\":\"F32 projection kernel only; not end-to-end token generation\",\"prompts\":[";bool first=true;
    for(std::size_t length:{1,8,32,128,512}){auto x=signal(length*256);auto expected=tvara::prefill_reference(w,x,length);std::vector<float> out;
        auto scalar=median_us([&]{out=tvara::prefill_reference(w,x,length);},5);auto blocked=median_us([&]{out=tvara::prefill_blocked(w,x,length);},5);close_vectors(out,expected);
        if(!first)std::cout<<",";first=false;std::cout<<"{\"tokens\":"<<length<<",\"reference_us\":"<<scalar<<",\"blocked_us\":"<<blocked<<",\"blocked_tokens_per_second\":"<<double(length)*1e6/blocked<<"}";}
    std::cout<<"],\"end_to_end_scope\":\"two-layer synthetic F32 model, sequential prefill, reused executor weights\",\"model_prompts\":[";first=true;
    tvara::PagedModelExecutor executor(prefill_fixture(),64,16);
    for(std::size_t length:{1,8,32,128,512}){std::vector<std::size_t> tokens(length);for(std::size_t i=0;i<length;++i)tokens[i]=(i*7)%23;std::vector<float> out;
        auto us=median_us([&]{out=executor.forward(tokens);},5);require(out.size()==length*23);for(auto value:out)require(std::isfinite(value));executor.cache().audit();
        if(!first)std::cout<<",";first=false;std::cout<<"{\"tokens\":"<<length<<",\"model_prefill_us\":"<<us<<",\"model_tokens_per_second\":"<<double(length)*1e6/us<<"}";}
    std::cout<<"]}";
}
