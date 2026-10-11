#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// DSV4_HC_POST with the SCALE -> SIGMOID -> SCALE chain that computes its post weights folded in
void ggml_cuda_op_dsv4_hc_post_scaled_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * scale0, const ggml_tensor * scale1);

// UNARY(SILU) of a SCALE, contiguous f32
void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * scale, ggml_tensor * silu);

