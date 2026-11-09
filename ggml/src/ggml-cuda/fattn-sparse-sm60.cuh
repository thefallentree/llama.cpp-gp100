#pragma once

#include "common.cuh"

// Sparse flash attention for GPUs without tensor cores (Pascal): FLASH_ATTN_EXT with the n_kv_max hint
// (op_params[4]) gathers only the mask's visible rows per query, so the cost is independent of the K/V length.
// D = DV = 512, groups of 32 heads per kv head (MQA / GQA with gqa % 32 == 0), K and V f16 or q8_0 read in place (other types converted to f16).

// the shapes and the environment allow the sparse kernel (GGML_CUDA_FA_SPARSE_SM60=0 turns it off,
// GGML_CUDA_FA_SPARSE_SM60_MIN_Q sets the minimum number of queries, default 1)
bool ggml_cuda_fattn_sparse_sm60_shall_use(int cc, const ggml_tensor * dst);

// K and V (same type, f16 or q8_0, 16-byte aligned rows) need no f16 conversion
bool ggml_cuda_fattn_sparse_sm60_kv_native(const ggml_tensor * dst);

void ggml_cuda_fattn_sparse_sm60(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
