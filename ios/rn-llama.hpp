#pragma once

#include <string>
#include <exception>
#include <vector>
#include "llama.h"
#include "common.h"
#include "ggml.h"

namespace rnllama {

// Helper function to convert GGUF key-value to string
inline std::string lm_gguf_kv_to_str(struct lm_gguf_context* ctx, int i) {
    return std::string(lm_gguf_get_val_str(ctx, i));
}

struct completion_token_output {
    llama_token tok;
    std::vector<llama_token_data> probs;
};

class llama_rn_context {
public:  // Make these public since they're accessed from Objective-C
    llama_model* model;
    llama_context* ctx;
    bool is_load_interrupted;
    int loading_progress;
    bool is_predicting;
    std::string last_error;

public:
    llama_rn_context() : model(nullptr), ctx(nullptr), is_load_interrupted(false),
                        loading_progress(0), is_predicting(false) {}

    ~llama_rn_context() {
        if (ctx) llama_free(ctx);
        if (model) llama_free_model(model);
    }

    bool load_model(const common_params& params, NSString** error) {
        try {
            // Check file access
            FILE* f = fopen(params.model.c_str(), "rb");
            if (!f) {
                last_error = "Cannot open model file: " + std::string(strerror(errno));
                if (error) *error = [NSString stringWithUTF8String:last_error.c_str()];
                return false;
            }
            fclose(f);

            // Load model
            llama_model_params model_params = llama_model_default_params();
            model = llama_load_model_from_file(params.model.c_str(), model_params);

            if (!model) {
                last_error = "Failed to load model: llama_load_model_from_file returned null";
                if (error) *error = [NSString stringWithUTF8String:last_error.c_str()];
                return false;
            }

            // Initialize context
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx = params.n_ctx;
            ctx = llama_new_context_with_model(model, ctx_params);

            if (!ctx) {
                last_error = "Failed to create context";
                if (error) *error = [NSString stringWithUTF8String:last_error.c_str()];
                llama_free_model(model);
                model = nullptr;
                return false;
            }

            return true;

        } catch (const std::exception& e) {
            last_error = std::string("Exception during model loading: ") + e.what();
            if (error) *error = [NSString stringWithUTF8String:last_error.c_str()];
            return false;
        }
    }

    bool validateModelChatTemplate() const {
        // Add implementation based on your requirements
        return true;
    }

    int applyLoraAdapters(const std::vector<common_lora_adapter_info>& adapters) {
        // Add implementation based on your requirements
        return 0;
    }

    // Getter methods for compatibility
    llama_model* get_model() const { return model; }
    llama_context* get_context() const { return ctx; }
    bool get_is_load_interrupted() const { return is_load_interrupted; }
    void set_is_load_interrupted(bool value) { is_load_interrupted = value; }
    int get_loading_progress() const { return loading_progress; }
    void set_loading_progress(int value) { loading_progress = value; }
};

} // namespace rnllama