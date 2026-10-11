#pragma once

#include "common.cuh"

// GGML_OP_FLASH_ATTN_SEL: attention of every token over its own selection of cache rows, see ggml.h.
// The kernels take a head size of 256, f16 or q8_0 keys and values and f32 queries (the QSA layers of
// Qwen3.8-Flash-Next).

#define FATTN_SEL_D         256
#define FATTN_SEL_CHUNK     64   // slots per chunk
#define FATTN_SEL_MAX_CHUNK 40   // chunks per token at most
#define FATTN_SEL_MAX_SEL   (FATTN_SEL_CHUNK*FATTN_SEL_MAX_CHUNK) // the cells of the indexer's top blocks (2048) and the tail

struct ggml_cuda_fattn_sel_args {
    const int32_t * sel;   // [nt][n_sel], s_sel apart: the cache rows of a token, a row outside [0, n_kv) is skipped
    const float *   q;     // [nt][n_head][D], sq_t and sq_h apart
    const char *    K;     // rows of D values (f16, or 8 q8_0 blocks), sk bytes apart, the kv heads skh bytes apart
    const char *    V;
    float *         out;   // [nt][n_head][D], so_t and so_h apart
    float *         part;  // [nt][n_head][n_cb][D + 4]: o, m, l of each chunk block, when n_cb > 1
    int             s_sel, sq_t, sq_h, so_t, so_h;
    int64_t         sk, skh, sv, svh;
    int             n_sel, n_kv, n_head, gqa;
    int             c_len, cpb, n_cb; // slots per chunk, chunks per block, chunk blocks per token (fattn_sel_plan)
    float           scale;
    bool            q8;    // the keys and values are q8_0 (both), else f16
};

// the chunking of the slots for nt tokens: a.c_len, a.cpb and a.n_cb; the caller provides a.part when n_cb > 1
void ggml_cuda_fattn_sel_plan(ggml_cuda_fattn_sel_args & a, int nt);

// all heads of nt tokens
void ggml_cuda_fattn_sel_launch(const ggml_cuda_fattn_sel_args & a, int nt, cudaStream_t stream);

bool ggml_cuda_flash_attn_sel_supported(const ggml_tensor * op);

void ggml_cuda_flash_attn_sel(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
