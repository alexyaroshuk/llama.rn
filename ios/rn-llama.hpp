#pragma once

#include <exception>
#include <string>
#include "llama.h"

namespace rnllama {
    struct common_lora_adapter_info {
        const char* path;
        float scale;
    };

    class llama_rn_context {
    private:
        llama_model* model;
        llama_context* ctx;
        bool is_load_interrupted;
        int loading_progress;

    public:
        llama_rn_context() : model(nullptr), ctx(nullptr), is_load_interrupted(false), loading_progress(0) {}

        ~llama_rn_context() {
            if (ctx) llama_free(ctx);
            if (model) llama_free_model(model);
        }

        bool load_model(const char* path, NSDictionary* params, NSString** error) {
            try {
                // Your existing model loading code here
                return true;
            } catch (const std::exception& e) {
                if (error) {
                    *error = [NSString stringWithUTF8String:e.what()];
                }
                return false;
            }
        }

        bool init_metal(NSString** error) {
            try {
                // Your existing metal initialization code here
                return true;
            } catch (const std::exception& e) {
                if (error) {
                    *error = [NSString stringWithUTF8String:e.what()];
                }
                return false;
            }
        }

        // Add getters for private members
        bool get_is_load_interrupted() const { return is_load_interrupted; }
        void set_is_load_interrupted(bool value) { is_load_interrupted = value; }

        int get_loading_progress() const { return loading_progress; }
        void set_loading_progress(int value) { loading_progress = value; }

        llama_model* get_model() const { return model; }
    };
}