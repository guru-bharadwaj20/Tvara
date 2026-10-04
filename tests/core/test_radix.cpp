#include "tvara/radix.hpp"
#include "tvara/paged_model.hpp"
#include <cassert>
#include <iostream>
using namespace tvara;
int main(){
 CacheGeometry g{1,1,1,4,64};PagedCache cache(g);RadixCache tree(cache,"fixture-model/tokenizer/tenant-a");
 assert(tree.size()==0);assert(tree.node(0).edge.empty());
assert(tree.lookup(std::vector<std::int32_t>{1,2}).tokens==0);
const std::vector<std::int32_t> prompt{1,2,3,4,5,6};auto a=tree.insert(prompt);assert(tree.insert(prompt)==a);assert(tree.lookup(prompt).tokens==6);
auto b=tree.insert(std::vector<std::int32_t>{1,2,7});assert(b!=a);assert(tree.lookup(std::vector<std::int32_t>{1,2,8}).tokens==2);assert(tree.insert(std::vector<std::int32_t>{1,2})!=a);
auto seq=cache.create_sequence();for(int i=0;i<6;++i){float f=static_cast<float>(i);cache.append(seq,std::span(&f,1),std::span(&f,1));}tree.attach(a,seq);cache.audit(tree.external_references());
auto full=tree.reuse(prompt);assert(full.second==4);assert(cache.physical_block(full.first,0).id==cache.physical_block(seq,0).id);cache.drop(full.first);
auto partial=tree.reuse(prompt,true);assert(partial.second==6);auto old=cache.physical_block(seq,5);float nine=9;cache.append(partial.first,std::span(&nine,1),std::span(&nine,1));assert(cache.physical_block(partial.first,5).id!=old.id);assert(cache.read(seq,0,5)[0]==5);cache.drop(partial.first);cache.audit(tree.external_references());
tree.pin(a);assert(tree.node(a).pins==1);auto shorter=tree.insert(std::vector<std::int32_t>{1});assert(tree.node(shorter).pins==1);tree.unpin(a);assert(tree.node(shorter).pins==0);bool underflow=false;try{tree.unpin(a);}catch(const std::invalid_argument&){underflow=true;}assert(underflow);
tree.touch(a);tree.touch(b);assert(tree.node(a).accessed<tree.node(b).accessed);
assert(tree.eviction_candidate()==a);tree.pin(a);assert(tree.eviction_candidate()==b);tree.unpin(a);
tree.pin(b);assert(tree.evict_one());assert(!tree.node(b).edge.empty());tree.unpin(b);cache.audit(tree.external_references());
auto c=tree.insert(prompt);tree.attach(c,seq);auto survivor=tree.reuse(prompt,true);tree.touch(b);assert(tree.evict_one());assert(cache.sequence(survivor.first).length==6);cache.append(survivor.first,std::span(&nine,1),std::span(&nine,1));assert(cache.read(seq,0,5)[0]==5);cache.drop(survivor.first);cache.audit(tree.external_references());
for(int round=0;round<200;++round){std::vector<std::int32_t> path{10,11,round%7};auto id=tree.insert(path);auto s=cache.create_sequence();for(int i=0;i<3;++i){float f=static_cast<float>(i);cache.append(s,std::span(&f,1),std::span(&f,1));}tree.attach(id,s);auto restored=tree.reuse(path,true);assert(restored.second==3);assert(cache.read(restored.first,0,2)[0]==2);cache.drop(restored.first);cache.drop(s);cache.audit(tree.external_references());while(tree.size()>12)assert(tree.evict_one());}cache.drop(seq);cache.audit(tree.external_references());
ModelWeights weights;weights.config={16,8,12,1,2,1,32};weights.embedding=Matrix(16,8);weights.output=Matrix(16,8);weights.final_norm.assign(8,1);LayerWeights layer;layer.attention_norm.assign(8,1);layer.feedforward_norm.assign(8,1);layer.query=Matrix(8,8);layer.key=Matrix(4,8);layer.value=Matrix(4,8);layer.attention_output=Matrix(8,8);layer.gate=Matrix(12,8);layer.up=Matrix(12,8);layer.down=Matrix(8,12);for(auto matrix:{&weights.embedding,&weights.output,&layer.query,&layer.key,&layer.value,&layer.attention_output,&layer.gate,&layer.up,&layer.down})for(std::size_t i=0;i<matrix->values.size();++i)matrix->values[i]=0.1f*std::sin(static_cast<float>(i+1));weights.layers.push_back(std::move(layer));PagedModelExecutor executor(std::move(weights),32,4);RadixCache model_prefix(executor.cache(),"model-fixture/tenant-a");auto source=executor.cache().create_sequence();std::vector<std::int32_t> stem{1,2,3};for(auto token:stem)executor.decode(source,static_cast<std::size_t>(token));model_prefix.attach(model_prefix.insert(stem),source);for(std::size_t tail=4;tail<10;++tail){auto request=stem;request.push_back(static_cast<std::int32_t>(tail));auto restored=model_prefix.reuse(request,true);assert(restored.second==3);auto cached=executor.decode(restored.first,tail);std::vector<std::size_t> cold_tokens{1,2,3,tail};auto cold=executor.forward(cold_tokens);for(std::size_t i=0;i<cached.size();++i)assert(cached[i]==cold[3*cached.size()+i]);executor.cache().drop(restored.first);}executor.cache().drop(source);executor.cache().audit(model_prefix.external_references());
// TESTS
 std::cout<<"radix OK\n";
}
