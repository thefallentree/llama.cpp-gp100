#pragma once

#include "common.cuh"

// Cold experts computed by host threads while the GPU computes the hot ones.
//
// A hot/cold split MoE layer keeps its most-routed experts in VRAM and the rest in pinned host memory
// (ggml_cuda_mmid_cold). For decode-sized batches the GPU reading a cold expert over PCIe costs more than
// the whole layer, so the cold (token, expert) pairs of the gate/up/swiglu/down triple are computed on
// the CPU instead, next to the memory that holds them, at the same time as the GPU computes the hot pairs
// (the design of Strata, github.com/Niko1221/Strata):
//
//   publish (GPU)  : the cold pairs and the input rows -> a mailbox in mapped pinned memory, then a doorbell
//   hot pairs (GPU): the up/gate/down mat-vec kernels skip the cold pairs
//   host threads   : spin on the doorbell, compute gate, up, swiglu and down of the cold pairs, write the rows
//   collect (GPU)  : waits for the host's answer and writes the rows into the down projection's output
//
// Everything is driven from the GPU side through kernels, so it works inside captured CUDA graphs.
// GGML_CUDA_MOE_HOST=0 turns it off (the cold experts are then staged to VRAM and computed on the GPU),
// GGML_CUDA_MOE_HOST_THREADS sets the host threads.

// Called before node i is computed. Opens a host triple at its first node and returns true for every
// node of the open triple: their mat-vec kernels have to skip the cold pairs.
bool ggml_cuda_moe_host_begin(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
// Called before the MUL_MAT_ID node i is computed, for batches of any size: a hot/cold triple that opens there is
// made known to the expert cache (expert-cache.cuh).
void ggml_cuda_moe_host_register(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// whether batches above MMVQ_MAX_BATCH_SIZE tokens go to the host too (opt-in)
bool ggml_cuda_moe_host_takes_prompts();

// Does node belong to the open host triple?
bool ggml_cuda_moe_host_active(const ggml_backend_cuda_context & ctx, const ggml_tensor * node);

// Called after a node is computed: after the triple's down projection, collects the host's rows.
void ggml_cuda_moe_host_end(ggml_backend_cuda_context & ctx, const ggml_tensor * node);

// The same when the triple's down projection was computed as the sum of the token's pairs times their weights
// (ggml_cuda_fn_moe_down): adds the host's rows to that sum.
void ggml_cuda_moe_host_end_weighted(ggml_backend_cuda_context & ctx, const ggml_tensor * node, ggml_tensor * dst);

void ggml_cuda_moe_host_free(ggml_backend_cuda_context & ctx);

// graph_optimize pass: moves work that only depends on a host triple's input (the shared expert) between the
// triple's first node and its down projection, so the GPU computes it while the host threads compute the cold
// experts instead of waiting for them.
void ggml_cuda_moe_host_reorder(ggml_cgraph * cgraph);
