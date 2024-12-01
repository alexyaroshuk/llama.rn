namespace rnllama {
    class llama_rn_context {
    public:
        // ... existing declarations ...

        bool load_model(const char* path, NSDictionary* params, NSString** error) {
            try {
                // Existing loading code...
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
                // Existing metal initialization...
                return true;
            } catch (const std::exception& e) {
                if (error) {
                    *error = [NSString stringWithUTF8String:e.what()];
                }
                return false;
            }
        }
    };
}