#pragma once

#include "common.cuh"

// Adaptive hot set of a hot/cold MoE layer (moe-host.cuh, ggml_cuda_mmid_cold).
//
// The file fixes which experts of a layer are in VRAM (the first n_hot of its hot-first order) and which are in
// pinned host memory. That order comes from routing counts of a calibration text; on other text 10-25% of the
// routed (token, expert) pairs are cold, almost every layer of a decode window has to wait for the host threads,
// and the host's memory bandwidth bounds the window. The experts that a text uses change slowly, though: a cache of
// the same size that follows the routing misses under 1% of the pairs.
//
// So the two tensors of a layer are treated as n_hot + n_cold slots whose contents are exchanged:
//   - the expert ids of a MUL_MAT_ID are replaced, once, by the positions of the experts (a table per layer on the
//     device): everything downstream keeps reading "position < n_hot is in VRAM, the others are host slots";
//   - the same kernel counts the routed experts;
//   - when the context's stream is synchronized (the window is done), the experts that were routed but cold take
//     the VRAM slots of the experts with the lowest decayed use count: one kernel exchanges the slices over PCIe
//     (the host slices are mapped), ordered on the stream before the next graph.
// The exchange is in place, in the model's weights: a layer is only changed by the context that registered it, and
// no more once a second context uses it.
//
// GGML_CUDA_EXPERT_CACHE=0 turns it off, GGML_CUDA_EXPERT_CACHE_SWAPS is the number of exchanges per update
// (default 16; after a prompt batch all the cold experts it routed to are brought in).

// before a graph is computed (and before a capture of it begins): sets up the tables if the graph has a hot/cold layer
void ggml_cuda_expert_cache_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph);

// is the layer of this expert tensor registered?
bool ggml_cuda_expert_cache_known(ggml_backend_cuda_context & ctx, const ggml_tensor * w);

// a hot/cold triple was identified (moe-host.cu): the layer's tensors on this device and their cold slices
void ggml_cuda_expert_cache_register(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
                                     const ggml_tensor * down, const char * c_up, const char * c_gate, const char * c_down,
                                     int n_cold);

// before a MUL_MAT_ID node is computed: its ids become positions (once per ids tensor and graph)
void ggml_cuda_expert_cache_remap(ggml_backend_cuda_context & ctx, const ggml_tensor * node);

// The same for a caller that writes the positions itself (ggml_cuda_moe_host_route): the layer's table of positions
// and its routing counts, indexed by expert. The node's ids then count as remapped. False: the layer is not registered.
bool ggml_cuda_expert_cache_take(ggml_backend_cuda_context & ctx, const ggml_tensor * node, const int32_t ** perm, uint32_t ** counts);

// after the context's stream was synchronized
void ggml_cuda_expert_cache_update(ggml_backend_cuda_context & ctx);

void ggml_cuda_expert_cache_context_free(ggml_backend_cuda_context & ctx);

// a device buffer is freed: forget the layers in it
void ggml_cuda_expert_cache_release(const void * base, size_t size);
