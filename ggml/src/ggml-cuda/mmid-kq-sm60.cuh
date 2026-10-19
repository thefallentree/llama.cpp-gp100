#pragma once

#include "common.cuh"

// Q2_K / Q3_K MUL_MAT_ID mat-vec for up to MMVQ_MAX_BATCH_SIZE tokens on sm_60 (GP100: no DP4A, HFMA2 at twice the FP32 rate).
// The activations are converted once to half pairs, the 2-bit weight fields are masked into half subnormals and
// multiplied with HFMA2, the sub-block scales are applied in half2 per part of a super-block (mmid-kq-sm60-math.cuh).
// GGML_CUDA_MMID_KQ_SM60=0 turns it off (the generic mmvq path takes the op).
bool ggml_cuda_mmid_vec_kq_sm60_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);

// pairs whose expert is not in src0 (the cold experts of a hot/cold layer, computed by the host tier) are left untouched
// Switches for a model run: GGML_CUDA_MMID_KQ_SM60_NOCACHE=1 (no sharing of the converted rows between gate and up), _STATS=n (log the
// first n dispatches); GGML_CUDA_MMID_CHECK=n compares the pairs with a host reference (ggml-cuda.cu).
void ggml_cuda_mmid_vec_kq_sm60(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

// whether the last ggml_cuda_mmid_vec_kq_sm60 of the calling thread took its converted rows from the cache
bool ggml_cuda_mmid_vec_kq_sm60_last_hit();

// The same kernels for plain MUL_MAT (src1 F32 with up to MMVQ_MAX_BATCH_SIZE columns; src0 [k, m] or [k, m, g] with src1 [k, n, g*r]): the
// attention and dense Q2_K / Q3_K weights. The activations are converted per op (no cache).
// GGML_CUDA_MM_KQ_SM60=0 turns it off, GGML_CUDA_MM_KQ_SM60_MAXN=n limits it to batches of up to n columns.
bool ggml_cuda_mm_vec_kq_sm60_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mm_vec_kq_sm60(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
