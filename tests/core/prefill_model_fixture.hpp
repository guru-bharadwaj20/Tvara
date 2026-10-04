#pragma once
#include "prefill_support.hpp"
inline tvara::ModelWeights prefill_fixture(std::size_t context=512){
    tvara::ModelWeights w;w.config={23,16,32,2,4,2,context};w.embedding=tvara::Matrix(23,16);w.embedding.values=signal(w.embedding.values.size(),0.2f);w.final_norm.assign(16,1);
    for(int i=0;i<2;++i){tvara::LayerWeights l;l.attention_norm.assign(16,1);l.feedforward_norm.assign(16,1);l.query=tvara::Matrix(16,16);l.key=tvara::Matrix(8,16);l.value=tvara::Matrix(8,16);l.attention_output=tvara::Matrix(16,16);l.gate=tvara::Matrix(32,16);l.up=tvara::Matrix(32,16);l.down=tvara::Matrix(16,32);
        for(auto* m:{&l.query,&l.key,&l.value,&l.attention_output,&l.gate,&l.up,&l.down})m->values=signal(m->values.size(),0.03f);w.layers.push_back(l);}
    return w;
}
