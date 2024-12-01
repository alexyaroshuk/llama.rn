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

// Helper function to format tokens to string
inline std::string tokens_to_output_formatted_string(llama_context* ctx, llama_token token) {
    const char* str = llama_token_to_str(ctx, token);
    return std::string(str ? str : "");
}

struct llama_token_data_with_prob {
    llama_token tok;
    float prob;
};

struct completion_token_output {
    llama_token tok;
    std::vector<llama_token_data_with_prob> probs;
};

// Sampling parameters structure
struct sampling_params {
    int32_t seed;
    int32_t n_predict;
    int32_t top_k;
    float top_p;
    float temp;
    float penalty_repeat;
    int32_t penalty_last_n;
    bool ignore_eos;
    std::vector<std::string> antiprompt;
};

// CPU parameters structure
struct cpu_params {
    int32_t n_threads;
};

// Parameters structure
struct params {
    std::string prompt;
    sampling_params sparams;
    cpu_params cpuparams;
    int32_t n_predict;
};

class llama_rn_context {
public:  // Make these public since they're accessed from Objective-C
    llama_model* model;
    llama_context* ctx;
    bool is_load_interrupted;
    int loading_progress;
    bool is_predicting;
    bool is_interrupted;
    bool has_next_token;
    std::string last_error;
    params params;

public:
    llama_rn_context() : model(nullptr), ctx(nullptr), is_load_interrupted(false),
                        loading_progress(0), is_predicting(false), is_interrupted(false),
                        has_next_token(false) {
        // Initialize default parameters
        params.sparams.seed = -1;
        params.sparams.n_predict = 128;
        params.sparams.top_k = 40;
        params.sparams.top_p = 0.95f;
        params.sparams.temp = 0.8f;
        params.sparams.penalty_repeat = 1.1f;
        params.sparams.penalty_last_n = 64;
        params.sparams.ignore_eos = false;
        params.cpuparams.n_threads = 4;
    }

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

    void rewind() {
        if (ctx) {
            llama_kv_cache_clear(ctx);
        }
    }

    bool initSampling() {
        if (!ctx || !model) return false;
        has_next_token = true;
        is_interrupted = false;
        return true;
    }

    void beginCompletion() {
        is_predicting = true;
    }

    void loadPrompt() {
        if (!ctx || params.prompt.empty()) return;
        std::vector<llama_token> tokens = llama_tokenize(ctx, params.prompt.c_str(), true);
        llama_eval(ctx, tokens.data(), tokens.size(), 0, params.cpuparams.n_threads);
    }

    bool validateModelChatTemplate() const {
        return true;
    }

    int applyLoraAdapters(const std::vector<common_lora_adapter_info>& adapters) {
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