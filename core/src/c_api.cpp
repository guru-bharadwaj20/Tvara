#include "tvara/c_api.h"
#include "tvara/paged_model.hpp"
#include <mutex>
#include <string>
#include <iomanip>
#include <sstream>
#include <type_traits>
struct tvara_model {
    tvara::PagedModelExecutor executor;
    std::mutex mutex;
    explicit tvara_model(const std::filesystem::path& path,std::size_t blocks,std::size_t tokens)
        :executor(tvara::load_weights(path),blocks,tokens){}
    std::string metadata;
    // HANDLE_MEMBERS
};
namespace {
thread_local std::string error_message;
void failure(const std::exception& error){error_message=error.what();}
std::string quoted(const std::string& text){
    std::ostringstream out;out<<'"';for(unsigned char c:text){
        if(c=='"' || c=='\\')out<<'\\'<<static_cast<char>(c);
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<static_cast<unsigned>(c)<<std::dec;
        else out<<static_cast<char>(c);
    }out<<'"';return out.str();
}
std::string json_value(const tvara::gguf::Value& value){
    return std::visit([](const auto& item)->std::string{
        using T=std::decay_t<decltype(item)>;
        if constexpr(std::is_same_v<T,std::string>)return quoted(item);
        else if constexpr(std::is_same_v<T,bool>)return item?"true":"false";
        else if constexpr(std::is_integral_v<T>)return std::to_string(item);
        else if constexpr(std::is_floating_point_v<T>){if(!std::isfinite(item))throw std::runtime_error("nonfinite metadata");std::ostringstream out;out<<std::setprecision(17)<<item;return out.str();}
        else{std::string out="[";bool first=true;for(const auto& element:item){if(!first)out+=",";first=false;out+=json_value(element);}return out+"]";}
    },value.data);
}
}
extern "C" {
tvara_model* tvara_model_open(const char* path,size_t blocks,size_t tokens){
    error_message.clear();try{if(!path || !*path)throw std::invalid_argument("empty model path");auto model=std::make_unique<tvara_model>(std::filesystem::u8path(path),blocks,tokens);
        tvara::gguf::ModelFile file(std::filesystem::u8path(path));model->metadata="{";bool first=true;
        for(const auto& [key,value]:file.document().metadata){if(!first)model->metadata+=",";first=false;model->metadata+=quoted(key)+":"+json_value(value);}model->metadata+="}";return model.release();}
    catch(const std::exception& error){failure(error);return nullptr;}catch(...){error_message="unknown native error";return nullptr;}
}
void tvara_model_close(tvara_model* model){delete model;}
size_t tvara_model_vocab_size(tvara_model* model){if(!model){error_message="closed model";return 0;}return model->executor.config().vocab;}
size_t tvara_model_context_size(tvara_model* model){if(!model){error_message="closed model";return 0;}return model->executor.config().context;}
const char* tvara_last_error(void){return error_message.c_str();}
const char* tvara_model_metadata(tvara_model* model){if(!model){error_message="closed model";return nullptr;}return model->metadata.c_str();}
uint64_t tvara_sequence_create(tvara_model* model){
    error_message.clear();try{if(!model)throw std::invalid_argument("closed model");std::lock_guard lock(model->mutex);return model->executor.cache().create_sequence();}
    catch(const std::exception& e){failure(e);return 0;}catch(...){error_message="unknown native error";return 0;}
}
int tvara_sequence_decode(tvara_model* model,uint64_t sequence,uint32_t token,float* output,size_t count){
    error_message.clear();try{if(!model || !output || count!=model->executor.config().vocab)throw std::invalid_argument("invalid logit output buffer");
        std::lock_guard lock(model->mutex);auto logits=model->executor.decode(sequence,token);std::copy(logits.begin(),logits.end(),output);return 0;}
    catch(const std::exception& e){failure(e);return -1;}catch(...){error_message="unknown native error";return -1;}
}
int tvara_sequence_drop(tvara_model* model,uint64_t sequence){
    error_message.clear();try{if(!model)throw std::invalid_argument("closed model");std::lock_guard lock(model->mutex);model->executor.cache().drop(sequence);return 0;}
    catch(const std::exception& e){failure(e);return -1;}catch(...){error_message="unknown native error";return -1;}
}
// ABI_IMPLEMENTATIONS
}
