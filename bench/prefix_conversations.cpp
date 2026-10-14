#include "tvara/paged_model.hpp"
#include "tvara/radix.hpp"
#include <chrono>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=2)return 2;tvara::PagedModelExecutor executor(tvara::load_weights(argv[1]),512,4);tvara::RadixCache tree(executor.cache(),"benchmark-model/byte-tokenizer/tenant-a");std::vector<std::int32_t>prefix;for(int i=1;i<=16;++i)prefix.push_back(i);auto source=executor.cache().create_sequence();for(auto token:prefix)executor.decode(source,static_cast<std::size_t>(token));tree.attach(tree.insert(prefix),source);executor.cache().drop(source);
 std::cout<<"{\"prefix_tokens\":16,\"traces\":[";
 for(int i=0;i<12;++i){auto request=prefix;request.push_back(17+i);std::vector<std::size_t>cold_tokens(request.begin(),request.end());auto begin=std::chrono::steady_clock::now();auto cold=executor.forward(cold_tokens);double cold_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();begin=std::chrono::steady_clock::now();auto restored=tree.reuse(request,true);auto warm=executor.decode(restored.first,static_cast<std::size_t>(request.back()));double warm_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();executor.cache().drop(restored.first);if(restored.second!=16)return 3;for(std::size_t k=0;k<warm.size();++k)if(warm[k]!=cold[16*warm.size()+k])return 4;if(i)std::cout<<",";std::cout<<"{\"trace\":"<<i<<",\"saved_prefill_tokens\":"<<restored.second<<",\"cold_seconds\":"<<cold_seconds<<",\"cached_seconds\":"<<warm_seconds<<"}";}
 executor.cache().audit(tree.external_references());std::cout<<"]}\n";
}
