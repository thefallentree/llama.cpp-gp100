#pragma once

#include "common.cuh"

// Q2_0 MUL_MAT_ID for batches on sm_60: weights decoded to half in shared memory, HFMA2 tile GEMM per expert.
bool ggml_cuda_mmid_f16_sm60_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);

void ggml_cuda_mmid_f16_sm60(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

// Q2_0 MUL_MAT_ID mat-vec (up to MMVQ_MAX_BATCH_SIZE tokens) on sm_60, same HFMA2 decode.
bool ggml_cuda_mmid_vec_f16_sm60_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);

// skip_cold: leave the rows of cold experts untouched (the host computes them, see moe-host.cuh)
void ggml_cuda_mmid_vec_f16_sm60(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
                                 bool skip_cold = false);

// A MUL_MAT_ID whose experts [src0->ne[2], src0->ne[2] + n_cold) live in pinned host memory (read over PCIe via UVA).
// llama sets op_params[12..15] = { magic, n_cold, data pointer }; only the kernels above understand it.
#define GGML_CUDA_EXPS_COLD_MAGIC 0x434f4c44
#define GGML_CUDA_EXPS_COLD_PARAM 12

static inline bool ggml_cuda_mmid_cold(const ggml_tensor * dst, const char ** data = nullptr, int * n_cold = nullptr) {
    if (dst->op != GGML_OP_MUL_MAT_ID || dst->op_params[GGML_CUDA_EXPS_COLD_PARAM] != GGML_CUDA_EXPS_COLD_MAGIC) {
        return false;
    }
    if (n_cold) {
        *n_cold = dst->op_params[GGML_CUDA_EXPS_COLD_PARAM + 1];
    }
    if (data) {
        memcpy(data, dst->op_params + GGML_CUDA_EXPS_COLD_PARAM + 2, sizeof(void *));
    }
    return true;
}

