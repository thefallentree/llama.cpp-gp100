#pragma once

#include "common.cuh"

// Q4_1 mat-vec specialised for sm_60 (GP100): expands nibbles to half2 with LOP3 magic
// constants and accumulates with full-rate HFMA2, avoiding the DP4A emulation the generic
// mmvq path falls back to on this architecture.
bool ggml_cuda_mmvq_f16_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst);

void ggml_cuda_mmvq_f16_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst);
