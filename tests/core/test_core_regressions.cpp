#include "prefill_model_fixture.hpp"
#include "tvara/radix.hpp"
#include "tvara/sampling.hpp"
#include <cstdlib>
#include <new>
#include <string>

// Deterministic allocation failure specifically at radix snapshot publication.
// Armed only around attach; all setup and checks use normal allocation.
static int fail_allocation_in=-1;
#if defined(__GNUC__) || defined(__clang__)
#define TVARA_TEST_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define TVARA_TEST_NOINLINE __declspec(noinline)
#else
#define TVARA_TEST_NOINLINE
#endif
TVARA_TEST_NOINLINE void* operator new(std::size_t size){
    if(fail_allocation_in>0 && --fail_allocation_in==0)throw std::bad_alloc();
    if(auto data=std::malloc(size?size:1))return data;
    throw std::bad_alloc();
}
// The nothrow forms (used by std::stable_sort's temporary buffer) must also pair
// with malloc/free; otherwise sanitizers see the runtime's new freed by this free.
TVARA_TEST_NOINLINE void* operator new(std::size_t size,const std::nothrow_t&)noexcept{
    try{return ::operator new(size);}catch(...){return nullptr;}
}
TVARA_TEST_NOINLINE void operator delete(void* data)noexcept{std::free(data);}
TVARA_TEST_NOINLINE void operator delete(void* data,std::size_t)noexcept{std::free(data);}
TVARA_TEST_NOINLINE void operator delete(void* data,const std::nothrow_t&)noexcept{std::free(data);}

void sampler_overflow(){
    auto large=std::numeric_limits<double>::max()*0.75;
    std::vector<double> p{large,large};tvara::filter_probabilities(p);
    for(auto value:p)require(std::isfinite(value));
    require(std::abs(p[0]-0.5)<1e-12 && std::abs(p[1]-0.5)<1e-12);
    p={std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::denorm_min()};
    tvara::filter_probabilities(p);require(p[0]==0.5 && p[1]==0.5);
}
void malformed_state_preserved(){
    tvara::ModelExecutor executor(prefill_fixture(16));auto state=executor.new_sequence();executor.decode(1,state);
    state.layers[1].values.pop_back();auto before=state;
    bool rejected=false;try{executor.decode(2,state);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected && state.length==before.length);
    for(std::size_t i=0;i<state.layers.size();++i){require(state.layers[i].keys==before.layers[i].keys);require(state.layers[i].values==before.layers[i].values);}
}
void snapshot_publication_failure(){
    tvara::PagedCache cache({1,1,4,4,8});auto sequence=cache.create_sequence();std::vector<float> token(4,1);cache.append(sequence,token,token);
    tvara::RadixCache radix(cache,"regression-model");std::vector<std::int32_t> prefix{1};auto node=radix.insert(prefix);
    bool allocation_failed=false;fail_allocation_in=2;
    try{radix.attach(node,sequence);}catch(const std::bad_alloc&){allocation_failed=true;}
    fail_allocation_in=-1;
    // Fixed code moves the already-created snapshot, so the second allocation is
    // eliminated. If an implementation still allocates, failure must not leak.
    if(allocation_failed)require(!radix.node(node).kv);
    auto external=radix.external_references();cache.audit(external);
    if(!allocation_failed)require(radix.node(node).kv.has_value());
}
int main(int argc,char** argv){
    try{
        const std::string selected=argc>1?argv[1]:"all";
        if(selected=="all" || selected=="sampling")sampler_overflow();
        if(selected=="all" || selected=="state")malformed_state_preserved();
        if(selected=="all" || selected=="radix")snapshot_publication_failure();
        std::cout<<"{\"regressions\":\"passed\"}";
    }catch(const std::exception& error){fail_allocation_in=-1;std::cerr<<error.what()<<'\n';return 1;}
}
