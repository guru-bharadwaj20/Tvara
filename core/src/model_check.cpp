#include "tvara/model.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc<4)throw std::invalid_argument("usage: tvara_model_check model.gguf output.bin token...");
        tvara::ModelExecutor model(tvara::load_weights(std::filesystem::path(argv[1])));
        std::vector<std::size_t> tokens;for(int i=3;i<argc;++i)tokens.push_back(std::stoull(argv[i]));
        auto logits=model.forward(tokens);std::ofstream out(argv[2],std::ios::binary);
        if(!out)throw std::runtime_error("cannot open logit output");
        out.write(reinterpret_cast<const char*>(logits.data()),static_cast<std::streamsize>(logits.size()*sizeof(float)));
        if(!out)throw std::runtime_error("failed to write logits");
        std::cout<<tokens.size()<<" positions, "<<model.config().vocab<<" vocabulary, CPU float32 output\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
