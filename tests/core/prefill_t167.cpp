#include "prefill_support.hpp"
#include "tvara/prefill.hpp"
int main(){tvara::Matrix w(39,71);w.values=signal(w.values.size());auto x=signal(17*71);auto expected=tvara::prefill_reference(w,x,17);
    for(auto tile:{tvara::PrefillTile{1,1,1},tvara::PrefillTile{8,16,32},tvara::PrefillTile{100,100,100}})close_vectors(tvara::prefill_blocked(w,x,17,tile),expected);
    rejects([&]{tvara::prefill_blocked(w,x,17,{0,1,1});});std::vector<float> out;auto us=median_us([&]{out=tvara::prefill_blocked(w,x,17);});std::cout<<"{\"max_error\":"<<max_error(out,expected)<<",\"blocked_us\":"<<us<<"}";
}
