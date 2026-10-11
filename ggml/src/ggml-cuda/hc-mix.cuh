#pragma once

#include "common.cuh"

// Fused hyper-connection read of qwen4exp (build_hc_mix) for decode batches, Q8_0 weights.
//
// The graph computes, per token t and stream c < hc:
//   xn       = rms_norm(R) * w_norm                                (norm over each stream's n_embd values)
//   lo[k]    = silu(s * (w_down[k] . xn) + b)                      k < hc_lr
//   mixed[d] = scale * sum_c xn[c][d] * sigmoid(w_up[c*n_embd + d] . lo)
// with eight kernels (norm, two quantizations, two mat-vecs, scale+silu, the gated mean). Here it is two:
//   down: each warp of a block owns one stream; it sums the stream's squares and its part of the block's
//         rows' dot products in one pass over R (the norm factor is applied to the per-stream partial sums)
//   up  : each warp owns four columns d; the epilogue applies the sigmoid gate, writes xn (read later by the
//         inject mat-vec) and takes the mean over the streams
// Up to HC_MIX_MAX_T tokens share each weight read (a speculative verify window).

#define HC_MIX_MAX_T 4

struct ggml_cuda_hc_mix_args {
    const ggml_tensor * rms;      // RMS_NORM, src[0] = R [n_embd, hc, nt]
    const ggml_tensor * mul;      // MUL by w_norm: dst = xn
    const ggml_tensor * mm_down;  // MUL_MAT(w_down [hc*n_embd -> hc_lr], xn)
    const ggml_tensor * scale;    // SCALE (s, b)
    const ggml_tensor * silu;     // SILU: dst = lo
    const ggml_tensor * mm_up;    // MUL_MAT(w_up [hc_lr -> hc*n_embd], lo)
    const ggml_tensor * pre;      // DSV4_HC_PRE (gated): dst = mixed
};

// Matches the ten nodes from i (RMS_NORM, MUL, RESHAPE, RESHAPE, MUL_MAT, SCALE, SILU, MUL_MAT, RESHAPE,
// DSV4_HC_PRE) and checks that the kernels support them.
bool ggml_cuda_hc_mix_match(const ggml_cgraph * cgraph, int i, ggml_cuda_hc_mix_args & args);

void ggml_cuda_hc_mix(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & args);
