#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define TVARA_API __declspec(dllexport)
#else
#define TVARA_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct tvara_model tvara_model;
TVARA_API tvara_model* tvara_model_open(const char* utf8_path,size_t max_blocks,size_t block_tokens);
TVARA_API void tvara_model_close(tvara_model* model);
TVARA_API size_t tvara_model_vocab_size(tvara_model* model);
TVARA_API size_t tvara_model_context_size(tvara_model* model);
TVARA_API const char* tvara_last_error(void);
TVARA_API const char* tvara_model_metadata(tvara_model* model);
// ABI_FUNCTIONS
#ifdef __cplusplus
}
#endif
