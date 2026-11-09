#pragma once

#include "common.cuh"

// Sparse flash attention for GPUs without tensor cores (Pascal): FLASH_ATTN_EXT with the n_kv_max hint
// (op_params[4]) gathers only the mask's visible rows per query, so the cost is independent of the K/V length.
// D = DV = 512, groups of 32 heads per kv head (MQA / GQA with gqa % 32 == 0), K and V f16 (converted if needed).

// the shapes and the environment allow the sparse kernel (GGML_CUDA_FA_SPARSE_SM60=0 turns it off,
// GGML_CUDA_FA_SPARSE_SM60_MIN_Q sets the minimum number of queries, default 8)
bool ggml_cuda_fattn_sparse_sm60_shall_use(int cc, const ggml_tensor * dst);

void ggml_cuda_fattn_sparse_sm60(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
