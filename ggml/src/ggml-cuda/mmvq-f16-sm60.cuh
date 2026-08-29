#pragma once

#include "common.cuh"

// Q4_1 mat-vec for sm_60 using LOP3 nibble expansion and HFMA2.
bool ggml_cuda_mmvq_f16_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst);

void ggml_cuda_mmvq_f16_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst);
