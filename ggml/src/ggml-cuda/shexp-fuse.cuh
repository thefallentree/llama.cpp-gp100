#pragma once

#include "common.cuh"

// Fused shared expert of qwen4exp-style MoE layers for decode batches (Q8_0 weights, sm_60 HFMA2):
//   h   = silu(w_gate . x) * (w_up . x)               (MUL_MAT, MUL_MAT, SWIGLU)
//   y   = (w_down . h) * sigmoid(w_gate_inp . x)      (MUL_MAT, MUL_MAT, SIGMOID, MUL)
// as two kernels instead of seven plus two activation quantizations. Both kernels quantize their input to the
// half2 activation format in shared memory, so nothing else is launched.

#define SHEXP_FUSE_MAX_T 4

struct ggml_cuda_shexp_args {
    const ggml_tensor * gate;    // MUL_MAT(w_gate, x)
    const ggml_tensor * up;      // MUL_MAT(w_up,   x)
    const ggml_tensor * glu;     // SWIGLU(gate, up)
    const ggml_tensor * down;    // MUL_MAT(w_down, glu)
    const ggml_tensor * ginp;    // MUL_MAT(w_gate_inp, x): one value per token
    const ggml_tensor * sig;     // SIGMOID(ginp)
    const ggml_tensor * mul;     // MUL(down, sig): the output
};

// Matches the seven nodes from i and checks that the kernels support them. planar: the weights are in the planar
// layout of the fused engine, which then computes the nodes (ggml_cuda_fn_shexp).
bool ggml_cuda_shexp_match(const ggml_cgraph * cgraph, int i, ggml_cuda_shexp_args & args, bool * planar);

void ggml_cuda_shexp(ggml_backend_cuda_context & ctx, const ggml_cuda_shexp_args & args);
