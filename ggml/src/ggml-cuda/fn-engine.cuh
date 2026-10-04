#pragma once

#include "common.cuh"
#include "hc-mix.cuh"

// Fused decode engine for sm_60 (GP100): planar weights and HFMA2 mat-vec kernels for windows of up to FN_MAX_T
// tokens (a speculative verify window).
//
// GP100 has no DP4A and no tensor cores; what it has is 600 GB/s of memory bandwidth and HFMA2 (two fp16 MACs per
// instruction at full rate). A decode window reads every dense weight once, so the kernels here are built to stream:
//   - weights are stored in planes (see below), so that every load is an aligned 16-byte transaction that
//     neighbouring lanes of a warp coalesce, with no realignment arithmetic;
//   - a block spans whole rows: a thread owns the same 16 columns of every row it reads, so its activations stay in
//     registers for the life of the block, and blocks are long-lived (a grid of resident blocks loops over row tiles
//     instead of one short block per tile);
//   - all loads of a tile are issued before the first is used (4-8 in flight per thread): a thread that waits for
//     one load at a time reaches half the bandwidth;
//   - activations are plain fp16 with one scale per token, the per-row partial sums stay in fp16 and are gathered
//     through shared memory (a warp shuffle reduction costs more instructions than the dot product).
// Measured on a P100 (40 MB matrix): 520/485/450 GB/s at 1/3/5 tokens, against 577 GB/s for reading the bytes.

#define FN_MAX_T 8

// Planar Q8_0 ("P8"): a [rows x cols] Q8_0 tensor repacked in place as
//   [rows*cols bytes: the quants, biased to unsigned (q ^ 0x80), row-major]
//   [rows*cols/32 halves: each block's scale relative to the largest scale of its row]
// plus a float scale per row in a side buffer. Same size as the Q8_0 blocks, same values (the relative scale is
// rounded to fp16). The tensor keeps its ggml type: only the kernels here may read it, see ggml_cuda_fn_planar().
struct ggml_cuda_fn_plane {
    float * rowscale = nullptr;
};

bool ggml_cuda_fn_enabled();

// Is the tensor's data in the planar layout? (looked up by its data pointer)
bool ggml_cuda_fn_planar(const ggml_tensor * t, ggml_cuda_fn_plane * plane = nullptr);

// graph_optimize: repacks the Q8_0 weights that the graph's MUL_MAT nodes read on this device. It runs before the
// first use of a graph, so no captured or in-flight launch still expects the old layout.
void ggml_cuda_fn_planes_optimize(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph);

// a device buffer is freed: forget the planar tensors in it
void ggml_cuda_fn_planes_release(const void * base, size_t size);

// all of a planar tensor as F16 / BF16 / F32 (prompt batches go through cuBLAS)
void ggml_cuda_fn_dequantize(const ggml_tensor * src0, void * dst, ggml_type dst_type, cudaStream_t stream);

// MUL_MAT with a planar src0: columns in passes of up to FN_MAX_T
bool ggml_cuda_fn_mul_mat_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_fn_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// the hyper-connection read (hc-mix.cuh) on planar weights, up to FN_MAX_T tokens
bool ggml_cuda_fn_hc_mix_supported(const ggml_cuda_hc_mix_args & args);
void ggml_cuda_fn_hc_mix(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & args);
