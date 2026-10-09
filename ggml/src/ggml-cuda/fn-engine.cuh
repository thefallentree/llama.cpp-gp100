#pragma once

#include "common.cuh"
#include "hc-mix.cuh"
#include "allreduce.cuh"
#include "shexp-fuse.cuh"

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
//   - activations are plain fp16 with one scale per token, converted from fp32 by the block itself; the per-row
//     partial sums stay in fp16 and are gathered through shared memory (a warp shuffle reduction costs more
//     instructions than the dot product).
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

// The logits of a router (the MUL_MAT node) are the node's data plus this, or null: where the router ran as two
// segments of a launch, the second one's output is here until the next router runs.
const float * ggml_cuda_fn_router_rest(ggml_backend_cuda_context & ctx, const ggml_tensor * logits);

// n >= 2 consecutive MUL_MAT nodes from node i that read the same vector with planar weights of the same row width
// are one launch (0: there is no such run); graph_optimize makes the MUL_MATs of a vector neighbours
int  ggml_cuda_fn_mul_mat_run_match(const ggml_cgraph * cgraph, int i);
void ggml_cuda_fn_mul_mat_run(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, int n);
void ggml_cuda_fn_reorder(ggml_cgraph * cgraph);

// the shared expert (shexp-fuse.cuh) on planar weights, up to FN_MAX_T tokens
bool ggml_cuda_fn_shexp_supported(const ggml_cuda_shexp_args & args);
void ggml_cuda_fn_shexp(ggml_backend_cuda_context & ctx, const ggml_cuda_shexp_args & args);

// Its last five nodes from the GLU node i (SwiGLU, down, gate scalar, sigmoid, product), where gate and up were
// computed before: ggml_cuda_fn_reorder moves them into the launch of the layer's router, which reads the same vector
// (GGML_CUDA_FN_ROUTER=0: the router stays an F32 mat-vec of its own).
bool ggml_cuda_fn_shexp_tail_match(const ggml_cgraph * cgraph, int i, ggml_cuda_shexp_args & args);
void ggml_cuda_fn_shexp_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_shexp_args & args);

// The output of a gated delta net layer before its projection, from node i: RMS_NORM, MUL (norm weights), RESHAPE,
// UNARY (SiLU of the gate), MUL, RESHAPE as one kernel that also leaves the activations for the projection.
// Returns the number of nodes to skip, 0 if the nodes are not that.
int ggml_cuda_fn_gdn_out(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// The output of an attention layer before its projection, from node i: CONT (of the gate), UNARY (sigmoid), MUL as
// one kernel that also leaves the activations for the projection. Returns the number of nodes to skip or 0.
int ggml_cuda_fn_gate_out(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// Do the n nodes from i have the ops `ops`, are they computed, and is every node but the last used only by nodes of
// the pattern? Bit k of `open` exempts node k from the last condition (a view that later nodes read as well: it has
// no kernel to skip). Unlike ggml_can_fuse_subgraph it accepts views of tensors outside the pattern and casts.
bool ggml_cuda_fn_pattern_closed(const ggml_cgraph * cgraph, int i, const ggml_op * ops, int n, uint64_t open);

// The input side of a recurrent (gated delta net) layer as one kernel: the conv input and its state snapshots, the
// convolution with its SiLU, the l2 norms of q and k and the gate (qwen4exp build_layer_attn_linear).
//   _begin: at the CONCAT of the conv input; matches the whole pattern, returns the number of nodes to skip or 0
//   the other: at the SSM_CONV that _begin matched; launches the kernel and returns the number of nodes to skip
int ggml_cuda_fn_gdn_pre_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_fn_gdn_pre(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// The pooled indexer keys of a QSA attention layer from node i (the SET_ROWS of the raw keys into the cache): the
// scatter of the raw keys and the mean of the members of the blocks to re-pool as one kernel. Returns the number of
// nodes to skip or 0.
int ggml_cuda_fn_qsa_pool(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// The selection of a QSA attention layer (qwen4exp build_qsa_sel) from node i, the first of the 16 nodes that map
// the cells of the top blocks and the tail and attend over them (FLASH_ATTN_SEL), for a window of up to FN_MAX_T
// tokens: the cells in one kernel, then the attention (fattn-sel.cuh). Returns the number of nodes to skip or 0.
int ggml_cuda_fn_qsa_attn(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// The routed experts of a MoE layer with Q2_0 weights, for a decode window of up to FN_MAX_T tokens:
//   up:   gate, up and SwiGLU of the (token, expert) pairs (MUL_MAT_ID, MUL_MAT_ID, GLU)
//   down: down and the sum of the token's pairs times their weights (MUL_MAT_ID, MUL, VIEWs, ADDs) -> dst
// Pairs whose expert is not in VRAM (ggml_cuda_mmid_cold) are skipped: the host threads compute them and
// ggml_cuda_moe_host_end_weighted adds their rows to dst.
struct ggml_cuda_fn_moe_route {
    int   e;  // position of the pair's expert
    float w;  // the pair's weight
};
bool ggml_cuda_fn_moe_supported(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
                                const ggml_tensor * weights, const ggml_tensor * dst);
void ggml_cuda_fn_moe_up(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * weights);
void ggml_cuda_fn_moe_down(ggml_backend_cuda_context & ctx, const ggml_tensor * down, ggml_tensor * dst);
// the pairs of the layer that ggml_cuda_fn_moe_up last computed on this context: [tokens][experts used]
const ggml_cuda_fn_moe_route * ggml_cuda_fn_moe_routes(const ggml_backend_cuda_context & ctx);

// the hyper-connection read (hc-mix.cuh) on planar weights, up to FN_MAX_T tokens
bool ggml_cuda_fn_hc_mix_supported(const ggml_cuda_hc_mix_args & args);
void ggml_cuda_fn_hc_mix(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & args);

// the read without its norm: MUL_MAT (down), SCALE, SILU, MUL_MAT (up), RESHAPE, DSV4_HC_PRE from node i, on the xn
// that an AllReduce epilogue has computed
bool ggml_cuda_fn_hc_tail_match(const ggml_cgraph * cgraph, int i, ggml_cuda_hc_mix_args & args);
void ggml_cuda_fn_hc_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & args);

// AllReduce epilogue (tensor-split windows): how many of the n_next nodes that follow the reduced tensor are computed
// together with its reduction (0: none), and the kernel that reduces and computes them on one device.
int  ggml_cuda_fn_ar_epilogue_match(const ggml_tensor * reduced, ggml_tensor ** next, int n_next);
void ggml_cuda_fn_ar_epilogue(ggml_backend_cuda_context & ctx, const ggml_tensor * reduced, ggml_tensor ** next, int n_next,
                              int n_fused, const ggml_cuda_ar_window_io & io);
