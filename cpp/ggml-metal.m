#import "ggml-metal.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"
#import "ggml-metal-impl.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#import "ggml-metal-fixes.h"

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max memory buffers that can be mapped to the device
#define LM_GGML_METAL_MAX_BUFFERS 64

// max number of MTLCommandBuffer used to submit a graph for processing
#define LM_GGML_METAL_MAX_COMMAND_BUFFERS 8

#define UNUSED(x) (void)(x)

// globals

// overload of MTLGPUFamilyMetal3 (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;

// initialized in lm_ggml_backend_metal_reg
static struct lm_ggml_backend_reg    g_lm_ggml_backend_metal_reg;
static struct lm_ggml_backend_device g_lm_ggml_backend_metal_device;

// information about a Metal device
// note: assumes single GPU device - the default one
// TODO: support multiple GPU devices
static struct lm_ggml_backend_metal_device_context {
    id<MTLDevice> mtl_device;
    int           mtl_device_ref_count;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_bfloat;
    bool use_bfloat;

    char name[128];
} g_lm_ggml_ctx_dev_main = {
    /*.mtl_device              =*/ nil,
    /*.mtl_device_ref_count    =*/ 0,
    /*.has_simdgroup_reduction =*/ false,
    /*.has_simdgroup_mm        =*/ false,
    /*.has_bfloat              =*/ false,
    /*.use_bfloat              =*/ false,
    /*.name                    =*/ "",
};

// acquire
static id<MTLDevice> lm_ggml_backend_metal_device_acq(struct lm_ggml_backend_metal_device_context * ctx) {
    assert(ctx != NULL);

    if (ctx->mtl_device == nil) {
        ctx->mtl_device = MTLCreateSystemDefaultDevice();

        ctx->has_simdgroup_reduction  = ggml_metal_supports_family(ctx->mtl_device, MTLGPUFamilyApple7);
        ctx->has_simdgroup_reduction |= ggml_metal_supports_family(ctx->mtl_device, MTLGPUFamilyMetal3_GGML);

        ctx->has_simdgroup_mm = ggml_metal_supports_family(ctx->mtl_device, MTLGPUFamilyApple7);

        ctx->has_bfloat  = ggml_metal_supports_family(ctx->mtl_device, MTLGPUFamilyMetal3_GGML);
        ctx->has_bfloat |= ggml_metal_supports_family(ctx->mtl_device, MTLGPUFamilyApple6);

#if defined(LM_GGML_METAL_USE_BF16)
        ctx->use_bfloat = ctx->has_bfloat;
#else
        ctx->use_bfloat = false;
#endif

        strncpy(ctx->name, [[ctx->mtl_device name] UTF8String], sizeof(ctx->name) - 1);
    }

    ctx->mtl_device_ref_count++;

    return ctx->mtl_device;
}

// release
static void lm_ggml_backend_metal_device_rel(struct lm_ggml_backend_metal_device_context * ctx) {
    assert(ctx != NULL);
    assert(ctx->mtl_device_ref_count > 0);

    ctx->mtl_device_ref_count--;

    if (ctx->mtl_device_ref_count == 0) {
        [ctx->mtl_device release];
        ctx->mtl_device = nil;
    }
}

// kernels

struct lm_ggml_metal_kernel {
    id<MTLComputePipelineState> pipeline;
};

enum lm_ggml_metal_kernel_type {
    LM_GGML_METAL_KERNEL_TYPE_ADD,
    LM_GGML_METAL_KERNEL_TYPE_ADD_ROW,
    LM_GGML_METAL_KERNEL_TYPE_SUB,
    LM_GGML_METAL_KERNEL_TYPE_SUB_ROW,
    LM_GGML_METAL_KERNEL_TYPE_MUL,
    LM_GGML_METAL_KERNEL_TYPE_MUL_ROW,
    LM_GGML_METAL_KERNEL_TYPE_DIV,
    LM_GGML_METAL_KERNEL_TYPE_DIV_ROW,
    LM_GGML_METAL_KERNEL_TYPE_REPEAT_F32,
    LM_GGML_METAL_KERNEL_TYPE_REPEAT_F16,
    LM_GGML_METAL_KERNEL_TYPE_REPEAT_I32,
    LM_GGML_METAL_KERNEL_TYPE_REPEAT_I16,
    LM_GGML_METAL_KERNEL_TYPE_SCALE,
    LM_GGML_METAL_KERNEL_TYPE_SCALE_4,
    LM_GGML_METAL_KERNEL_TYPE_CLAMP,
    LM_GGML_METAL_KERNEL_TYPE_TANH,
    LM_GGML_METAL_KERNEL_TYPE_RELU,
    LM_GGML_METAL_KERNEL_TYPE_SIGMOID,
    LM_GGML_METAL_KERNEL_TYPE_GELU,
    LM_GGML_METAL_KERNEL_TYPE_GELU_4,
    LM_GGML_METAL_KERNEL_TYPE_GELU_QUICK,
    LM_GGML_METAL_KERNEL_TYPE_GELU_QUICK_4,
    LM_GGML_METAL_KERNEL_TYPE_SILU,
    LM_GGML_METAL_KERNEL_TYPE_SILU_4,
    LM_GGML_METAL_KERNEL_TYPE_ELU,
    LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F16,
    LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F16_4,
    LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F32,
    LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F32_4,
    LM_GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF,
    LM_GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF_8,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_F32,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_F16,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_BF16,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_0,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_1,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_0,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_1,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q8_0,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q2_K,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q3_K,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_K,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_K,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q6_K,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XXS,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XS,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_XXS,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_S,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_S,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_S,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_M,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_NL,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_XS,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_I32,
    LM_GGML_METAL_KERNEL_TYPE_RMS_NORM,
    LM_GGML_METAL_KERNEL_TYPE_GROUP_NORM,
    LM_GGML_METAL_KERNEL_TYPE_NORM,
    LM_GGML_METAL_KERNEL_TYPE_SSM_CONV_F32,
    LM_GGML_METAL_KERNEL_TYPE_SSM_SCAN_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F32_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_1ROW,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_L4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F16,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32_1ROW,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32_L4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_BF16,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q8_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q2_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q3_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q6_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_M_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_NL_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F32_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32,
  //LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32_1ROW,
  //LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32_L4,
  //LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F16,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_BF16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q8_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q2_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q3_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q6_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_M_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_NL_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_F32_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_F16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_BF16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q8_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q2_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q3_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q6_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_M_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_NL_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F32_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_BF16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q8_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q2_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q3_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q6_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_M_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_NL_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_ROPE_NORM_F32,
    LM_GGML_METAL_KERNEL_TYPE_ROPE_NORM_F16,
    LM_GGML_METAL_KERNEL_TYPE_ROPE_NEOX_F32,
    LM_GGML_METAL_KERNEL_TYPE_ROPE_NEOX_F16,
    LM_GGML_METAL_KERNEL_TYPE_IM2COL_F16,
    LM_GGML_METAL_KERNEL_TYPE_IM2COL_F32,
    LM_GGML_METAL_KERNEL_TYPE_IM2COL_EXT_F16,
    LM_GGML_METAL_KERNEL_TYPE_IM2COL_EXT_F32,
    LM_GGML_METAL_KERNEL_TYPE_UPSCALE_F32,
    LM_GGML_METAL_KERNEL_TYPE_PAD_F32,
    LM_GGML_METAL_KERNEL_TYPE_ARANGE_F32,
    LM_GGML_METAL_KERNEL_TYPE_TIMESTEP_EMBEDDING_F32,
    LM_GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_ASC,
    LM_GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_DESC,
    LM_GGML_METAL_KERNEL_TYPE_LEAKY_RELU_F32,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_F16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_BF16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F16_F16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F16_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_BF16_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_BF16_BF16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q8_0,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_0,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_1,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_0,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_1,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_IQ4_NL,
    LM_GGML_METAL_KERNEL_TYPE_CONCAT,
    LM_GGML_METAL_KERNEL_TYPE_SQR,
    LM_GGML_METAL_KERNEL_TYPE_SQRT,
    LM_GGML_METAL_KERNEL_TYPE_SIN,
    LM_GGML_METAL_KERNEL_TYPE_COS,
    LM_GGML_METAL_KERNEL_TYPE_SUM_ROWS,
    LM_GGML_METAL_KERNEL_TYPE_POOL_2D_AVG_F32,
    LM_GGML_METAL_KERNEL_TYPE_POOL_2D_MAX_F32,

    LM_GGML_METAL_KERNEL_TYPE_COUNT
};

struct lm_ggml_backend_metal_context {
    id<MTLCommandQueue> queue;

    dispatch_queue_t d_queue;

    struct lm_ggml_metal_kernel kernels[LM_GGML_METAL_KERNEL_TYPE_COUNT];

    // capture state
    bool capture_next_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct lm_ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    id<MTLCommandBuffer> command_buffers[LM_GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // abort lm_ggml_metal_graph_compute if callback returns true
    lm_ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

// MSL code
// TODO: move the contents here when ready
//       for now it is easier to work in a separate file
// static NSString * const msl_library_source = @"see metal.metal";

// Here to assist with NSBundle Path Hack
@interface LMGGMLMetalClass : NSObject
@end
@implementation LMGGMLMetalClass
@end

static void * lm_ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

#if TARGET_OS_OSX
    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        LM_GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }
#else
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        LM_GGML_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif

    return data;
}

static struct lm_ggml_backend_metal_context * lm_ggml_metal_init(lm_ggml_backend_dev_t dev) {
    LM_GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !LM_GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        LM_GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    struct lm_ggml_backend_metal_context * ctx = calloc(1, sizeof(struct lm_ggml_backend_metal_context));
    struct lm_ggml_backend_metal_device_context * ctx_dev = dev->context;

    id<MTLDevice> device = lm_ggml_backend_metal_device_acq(ctx_dev);
    LM_GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    ctx->queue  = [device newCommandQueue];
    ctx->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    id<MTLLibrary> metal_library;

    // load library
    //
    // - first check if the library is embedded
    // - then check if the library is in the bundle
    // - if not found, load the source and compile it
    // - if that fails, return NULL
    {
        NSBundle * bundle = nil;
#ifdef SWIFT_PACKAGE
        bundle = SWIFTPM_MODULE_BUNDLE;
#else
        bundle = [NSBundle bundleForClass:[LMGGMLMetalClass class]];
#endif

        NSError * error = nil;

#if LM_GGML_METAL_EMBED_LIBRARY
        const bool try_metallib = false;
#else
        const bool try_metallib = true;
#endif

        NSString * path_lib = [bundle pathForResource:@"ggml-llama" ofType:@"metallib"];
        if (try_metallib && path_lib != nil) {
            // pre-compiled library found
            NSURL * libURL = [NSURL fileURLWithPath:path_lib];
            LM_GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

            metal_library = [device newLibraryWithURL:libURL error:&error];
            if (error) {
                LM_GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return NULL;
            }
        } else {
#if LM_GGML_METAL_EMBED_LIBRARY
            LM_GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

            extern const char lm_ggml_metallib_start[];
            extern const char lm_ggml_metallib_end[];

            NSString * src = [[NSString alloc] initWithBytes:lm_ggml_metallib_start length:(lm_ggml_metallib_end-lm_ggml_metallib_start) encoding:NSUTF8StringEncoding];
#else
            LM_GGML_LOG_INFO("%s: default.metallib not found, loading from source\n", __func__);

            NSString * path_source;
            NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"LM_GGML_METAL_PATH_RESOURCES"];

            LM_GGML_LOG_INFO("%s: LM_GGML_METAL_PATH_RESOURCES = %s\n", __func__, path_resource ? [path_resource UTF8String] : "nil");

            if (path_resource) {
                path_source = [path_resource stringByAppendingPathComponent:@"ggml-metal.metal"];
            } else {
                path_source = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
            }

            if (path_source == nil) {
                LM_GGML_LOG_WARN("%s: error: could not use bundle path to find ggml-metal.metal, falling back to trying cwd\n", __func__);
                path_source = @"ggml-metal.metal";
            }

            LM_GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_source UTF8String]);

            NSString * src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
            if (error) {
                LM_GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return NULL;
            }
#endif // LM_GGML_METAL_EMBED_LIBRARY

            @autoreleasepool {
                // dictionary of preprocessor macros
                NSMutableDictionary * prep = [NSMutableDictionary dictionary];

                if (ctx_dev->use_bfloat) {
                    [prep setObject:@"1" forKey:@"LM_GGML_METAL_USE_BF16"];
                }

#if LM_GGML_METAL_EMBED_LIBRARY
                [prep setObject:@"1" forKey:@"LM_GGML_METAL_EMBED_LIBRARY"];
#endif

                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;

                //[options setFastMathEnabled:false];

                metal_library = [device newLibraryWithSource:src options:options error:&error];
                if (error) {
                    LM_GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    return NULL;
                }

#if !__has_feature(objc_arc)
                [options release];
#endif
            }
#if LM_GGML_METAL_EMBED_LIBRARY
            [src release];
#endif // LM_GGML_METAL_EMBED_LIBRARY
        }
    }

    // print MTL GPU family:
    LM_GGML_LOG_INFO("%s: GPU name:   %s\n", __func__, [[device name] UTF8String]);

    // determine max supported GPU family
    // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
    // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    {
        for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
            if (ggml_metal_supports_family(device, i - MTLGPUFamilyApple1 + 1)) {
                LM_GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, i - (int) MTLGPUFamilyApple1 + 1, i);
                break;
            }
        }

        for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
            if (ggml_metal_supports_family(device, i - MTLGPUFamilyCommon1 + 1)) {
                LM_GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                break;
            }
        }

        for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
            if (ggml_metal_supports_family(device, i - MTLGPUFamilyMetal3_GGML + 3)) {
                LM_GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                break;
            }
        }
    }

    LM_GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, ctx_dev->has_simdgroup_reduction     ? "true" : "false");
    LM_GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, ctx_dev->has_simdgroup_mm            ? "true" : "false");
    LM_GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, ctx_dev->has_bfloat                  ? "true" : "false");
    LM_GGML_LOG_INFO("%s: use bfloat            = %s\n", __func__, ctx_dev->use_bfloat                  ? "true" : "false");
    LM_GGML_LOG_INFO("%s: hasUnifiedMemory      = %s\n", __func__, ctx_dev->mtl_device.hasUnifiedMemory ? "true" : "false");

    ctx->capture_next_compute = false;
    ctx->capture_started = false;
    ctx->capture_scope = nil;

    ctx->gf = nil;
    ctx->encode_async = nil;
    for (int i = 0; i < LM_GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        ctx->command_buffers[i] = nil;
    }
