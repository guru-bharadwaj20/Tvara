#include "tvara/c_api.h"
#include "tvara/paged_model.hpp"
#include <mutex>
#include <string>
struct tvara_model {
    tvara::PagedModelExecutor executor;
    std::mutex mutex;
    explicit tvara_model(const std::filesystem::path& path,std::size_t blocks,std::size_t tokens)
        :executor(tvara::load_weights(path),blocks,tokens){}
    // HANDLE_MEMBERS
};
namespace {
thread_local std::string error_message;
void failure(const std::exception& error){error_message=error.what();}
}
extern "C" {
tvara_model* tvara_model_open(const char* path,size_t blocks,size_t tokens){
    error_message.clear();try{if(!path || !*path)throw std::invalid_argument("empty model path");return new tvara_model(std::filesystem::u8path(path),blocks,tokens);}
    catch(const std::exception& error){failure(error);return nullptr;}catch(...){error_message="unknown native error";return nullptr;}
}
void tvara_model_close(tvara_model* model){delete model;}
size_t tvara_model_vocab_size(tvara_model* model){if(!model){error_message="closed model";return 0;}return model->executor.config().vocab;}
size_t tvara_model_context_size(tvara_model* model){if(!model){error_message="closed model";return 0;}return model->executor.config().context;}
const char* tvara_last_error(void){return error_message.c_str();}
// ABI_IMPLEMENTATIONS
}
