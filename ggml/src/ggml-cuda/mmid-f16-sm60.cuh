#pragma once

#include "common.cuh"

// Q2_0 MUL_MAT_ID for batches on sm_60: weights decoded to half in shared memory, HFMA2 tile GEMM per expert.
bool ggml_cuda_mmid_f16_sm60_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);

void ggml_cuda_mmid_f16_sm60(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

// Q2_0 MUL_MAT_ID mat-vec (up to MMVQ_MAX_BATCH_SIZE tokens) on sm_60, same HFMA2 decode.
bool ggml_cuda_mmid_vec_f16_sm60_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);

void ggml_cuda_mmid_vec_f16_sm60(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
