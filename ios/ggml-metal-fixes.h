#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// Helper macro for Metal API availability
#define GGML_METAL_CHECK_API_AVAILABLE @available(iOS 13.0, *)

// Helper function for Metal GPU family check
static inline BOOL ggml_metal_supports_family(id<MTLDevice> device, NSInteger family) {
    if (@available(iOS 13.0, *)) {
        return [device supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + family - 1)];
    }
    return NO;
}

// Helper function for Metal capture
static inline BOOL ggml_metal_start_capture(MTLCaptureDescriptor *descriptor, NSError **error) {
    if (@available(iOS 13.0, *)) {
        return [[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:error];
    }
    return NO;
}