#pragma once

#include "common.cuh"

// IQ2_S / IQ3_XXS / IQ3_S MUL_MAT_ID mat-vec for up to MMVQ_MAX_BATCH_SIZE tokens on sm_60 (GP100: no DP4A, HFMA2 at twice the
// FP32 rate): the hot expert pairs of Project Maya's files. The activations are the converted rows of the Q2_K / Q3_K kernel
// (mmid-kq-sm60.cu, shared between gate and up), the grid bytes are masked into half subnormals with their sign bits from a
// shared table and multiplied with HFMA2 (mmid-iq-sm60-math.cuh). GGML_CUDA_MMID_IQ_SM60=0 turns it off (the generic mmvq path
// with the emulated dp4a takes the op).
bool ggml_cuda_mmid_vec_iq_sm60_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);

// pairs whose expert is not in src0 (the cold experts of a hot/cold layer, computed by the host tier) are left untouched
void ggml_cuda_mmid_vec_iq_sm60(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

// whether the last ggml_cuda_mmid_vec_iq_sm60 of the calling thread took its converted rows from the cache
bool ggml_cuda_mmid_vec_iq_sm60_last_hit();
