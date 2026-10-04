#pragma once

#include "common.cuh"

// Q2_0 MUL_MAT_ID for batches on sm_60: weights decoded to half in shared memory, HFMA2 tile GEMM per expert.
bool ggml_cuda_mmid_f16_sm60_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);

// skip_cold: leave the rows of cold experts untouched (the host computes them, see moe-host.cuh)
// cold_vram: all cold experts already copied to VRAM in the host tensor's layout (the prefetch below)
void ggml_cuda_mmid_f16_sm60(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
                             bool skip_cold = false, const char * cold_vram = nullptr);

// Prompt batches of a hot/cold MoE read every cold expert, so all of them are copied to VRAM: by DMA on a second
// stream, one MUL_MAT_ID ahead (two buffers), so the copy of the next op overlaps the GEMM of the current one.
// prepare: before the graph (and before a CUDA graph capture), collects the graph's prompt-sized cold MUL_MAT_IDs
//          and sizes the buffers; returns whether there are any.
// start:   at the graph start, forks the copy stream from the compute stream and issues the first two copies.
// run:     computes one of the ops from its prefetched buffer; returns false when the node is not one of them.
// finish:  at the graph end, joins the copy stream back (a CUDA graph capture must end joined).
// The copies only use stream-ordered calls, so a graph with them can be captured as a CUDA graph.
bool ggml_cuda_cold_prefetch_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph);
void ggml_cuda_cold_prefetch_start(ggml_backend_cuda_context & ctx);
bool ggml_cuda_cold_prefetch_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_cold_prefetch_finish(ggml_backend_cuda_context & ctx);
void ggml_cuda_cold_prefetch_free(ggml_backend_cuda_context & ctx);
// prompt-sized cold MUL_MAT_IDs (computed eagerly: the prefetch issues host-side copies and events)
bool ggml_cuda_mmid_cold_prompt(const ggml_tensor * node);

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

