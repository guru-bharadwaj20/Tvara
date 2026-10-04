#include "prefill_support.hpp"
#include "tvara/prefill.hpp"
int main(){tvara::Matrix w(128,256);w.values=signal(w.values.size(),0.1f);auto x=signal(64*256);auto ref=tvara::prefill_reference(w,x,64);std::vector<float> out;
    std::cout<<"{\"candidates\":[";bool first=true;for(auto tile:{tvara::PrefillTile{1,8,32},tvara::PrefillTile{4,16,64},tvara::PrefillTile{8,16,64},tvara::PrefillTile{16,32,128}}){auto us=median_us([&]{out=tvara::prefill_blocked(w,x,64,tile);});close_vectors(out,ref);if(!first)std::cout<<",";first=false;std::cout<<"{\"tokens\":"<<tile.tokens<<",\"outputs\":"<<tile.outputs<<",\"inner\":"<<tile.inner<<",\"us\":"<<us<<"}";}
    auto us=median_us([&]{out=tvara::prefill_reference(w,x,64);});std::cout<<"],\"reference_us\":"<<us<<",\"policy\":\"retain explicit tiles; no global winner from one shape\"}";
}
