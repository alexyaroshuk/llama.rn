@implementation RNLlamaContext {
    NSString *_lastErrorMessage;
}

+ (instancetype)initWithParams:(NSDictionary *)params onProgress:(void (^)(unsigned int progress))onProgress {
    RNLlamaContext *context = [[RNLlamaContext alloc] init];
    if (context) {
        context->onProgress = onProgress;
        context->is_metal_enabled = false;
        context->is_model_loaded = false;
        context->_lastErrorMessage = nil; // Initialize error message

        @try {
            // ... existing initialization code ...

            context->llama = new rnllama::llama_rn_context();
            if (!context->llama) {
                context->_lastErrorMessage = @"Failed to allocate llama context";
                return context;
            }

            NSString *model_path = params[@"model"];
            if (!model_path) {
                context->_lastErrorMessage = @"No model path provided";
                return context;
            }

            // Track Metal initialization errors
            NSString *metalError = nil;
            context->is_metal_enabled = context->llama->init_metal(&metalError);
            if (!context->is_metal_enabled && metalError) {
                context->reason_no_metal = metalError;
            }

            // Track model loading errors
            NSString *loadError = nil;
            context->is_model_loaded = context->llama->load_model([model_path UTF8String], params, &loadError);
            if (!context->is_model_loaded) {
                context->_lastErrorMessage = loadError ?: @"Unknown error during model loading";
            }

        } @catch (NSException *exception) {
            context->_lastErrorMessage = [NSString stringWithFormat:@"%@: %@",
                                       exception.name,
                                       exception.reason];
        }
    }
    return context;
}

- (NSString *)getLastError {
    return _lastErrorMessage;
}