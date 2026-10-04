#include "tvara/model.hpp"
#include "tvara/paged_model.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc<4)throw std::invalid_argument("usage: tvara_model_check model.gguf output.bin token...");
        const bool paged=std::string(argv[3])=="--paged";
        std::vector<std::size_t> tokens;for(int i=paged?4:3;i<argc;++i)tokens.push_back(std::stoull(argv[i]));
        std::vector<float> logits;std::size_t vocabulary;
        if(paged){tvara::PagedModelExecutor model(tvara::load_weights(std::filesystem::path(argv[1])));logits=model.forward(tokens);vocabulary=model.config().vocab;}
        else{tvara::ModelExecutor model(tvara::load_weights(std::filesystem::path(argv[1])));logits=model.forward(tokens);vocabulary=model.config().vocab;}
        std::ofstream out(argv[2],std::ios::binary);
        if(!out)throw std::runtime_error("cannot open logit output");
        out.write(reinterpret_cast<const char*>(logits.data()),static_cast<std::streamsize>(logits.size()*sizeof(float)));
        if(!out)throw std::runtime_error("failed to write logits");
        std::cout<<tokens.size()<<" positions, "<<vocabulary<<" vocabulary, CPU float32 output\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
