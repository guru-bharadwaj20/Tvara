#include "llama.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
int main(int argc,char**argv){
 if(argc!=2)return 2;llama_log_set([](enum ggml_log_level level,const char*text,void*){if(level==GGML_LOG_LEVEL_ERROR)std::cerr<<text;},nullptr);llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
 auto cp=llama_context_default_params();cp.n_ctx=64;cp.n_batch=16;cp.n_threads=1;cp.n_threads_batch=1;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;auto*ctx=llama_init_from_model(model,cp);if(!ctx)return 4;
 auto decode=[&](const std::vector<llama_token>&tokens,int position){auto batch=llama_batch_init(static_cast<int>(tokens.size()),0,1);batch.n_tokens=static_cast<int>(tokens.size());for(int i=0;i<batch.n_tokens;++i){batch.token[i]=tokens[static_cast<std::size_t>(i)];batch.pos[i]=position+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=true;}auto status=llama_decode(ctx,batch);llama_batch_free(batch);if(status)throw std::runtime_error("llama decode failed");llama_synchronize(ctx);};
 std::vector<llama_token>prompt;for(int i=1;i<=16;++i)prompt.push_back(i);std::vector<double>pp,tg;std::vector<float>logits;
 for(int repeat=0;repeat<6;++repeat){llama_memory_clear(llama_get_memory(ctx),true);auto begin=std::chrono::steady_clock::now();decode(prompt,0);auto prefill=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();if(repeat==0){auto*p=llama_get_logits_ith(ctx,-1);logits.assign(p,p+llama_vocab_n_tokens(llama_model_get_vocab(model)));}begin=std::chrono::steady_clock::now();for(int i=0;i<8;++i)decode({17+i},16+i);auto generation=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();if(repeat){pp.push_back(prefill);tg.push_back(generation);}}
 auto array=[](const auto&values){std::cout<<"[";bool first=true;for(auto value:values){if(!first)std::cout<<",";first=false;std::cout<<value;}std::cout<<"]";};std::cout.precision(9);std::cout<<"{\"prefill_seconds\":";array(pp);std::cout<<",\"decode_seconds\":";array(tg);std::cout<<",\"prefill_logits\":";array(logits);std::cout<<"}\n";llama_free(ctx);llama_model_free(model);llama_backend_free();
}
