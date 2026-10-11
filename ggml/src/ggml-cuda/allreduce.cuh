#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the GPU device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

// Window mode: the reductions of a decode window, issued between ggml_cuda_ar_window_begin and the end of the window,
// take their token from device memory instead of a launch argument and need no host-side pacing, so each device's
// part of the window can be captured into one CUDA graph and replayed (peer-memory transport only).
// max_bytes: the largest tensor reduced in the window.
bool ggml_cuda_ar_window_supported(const ggml_cuda_ar_pipeline * pipeline, size_t max_bytes);
void ggml_cuda_ar_window_begin(ggml_cuda_ar_pipeline * pipeline, ggml_backend_t * backends);
bool ggml_cuda_ar_window_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

// For kernels that reduce and go on computing (fn-engine.cuh): each device's end of the next reduction of the open
// window. The token of the reduction is ggml_cuda_ar_window_token(epoch, site), never 0 and different from the
// tokens of the reductions that used the same staging before. A kernel either writes it into arrival_mine once its
// data is in wire_mine and waits for the same token in arrival_other before it reads wire_other (up to
// GGML_CUDA_AR_WINDOW_BLOCKS blocks independently, block b with the tokens at b*GGML_CUDA_AR_WINDOW_ARRIVAL_INTS),
// or it puts the token into every unit it stores and polls the peer's units for it.
#define GGML_CUDA_AR_WINDOW_MAX_SITES    1024u
#define GGML_CUDA_AR_WINDOW_BLOCKS       8
#define GGML_CUDA_AR_WINDOW_ARRIVAL_INTS 16

struct ggml_cuda_ar_window_io {
    char *               wire_mine;     // in the peer's memory
    const char *         wire_other;    // in this device's memory, written by the peer
    size_t               wire_bytes;
    int *                arrival_mine;  // in the peer's memory
    const int *          arrival_other;
    const unsigned int * epoch;         // windows started on this device
    int                  site;          // index of the reduction in the window
};

static __device__ __forceinline__ int ggml_cuda_ar_window_token(const unsigned int * epoch, const int site) {
    return (int) (*epoch * GGML_CUDA_AR_WINDOW_MAX_SITES + (unsigned int) site);
}

static __device__ __forceinline__ void ggml_cuda_ar_window_signal_set(int * p, const int token) {
    *(volatile int *) p = token;
}

static __device__ __forceinline__ int ggml_cuda_ar_window_signal_get(const int * p) {
    return *(const volatile int *) p;
}

// The flag-in-data exchange: four values travel as one 16-byte unit, 24 bits each plus the token of the reduction,
// and the peer polls that unit until it carries the token. A 16-byte store arrives whole, so a unit with the token
// is complete: no token ring and no __threadfence_system() between the data and a separate signal (microseconds
// each on GP100). Both devices add the same two rounded values, so their results are identical.
// a float as its upper 24 bits, rounded to nearest
static __device__ __forceinline__ unsigned int ggml_cuda_ar_f24(const float x) {
    return (__float_as_uint(x) + 0x80u) >> 8;
}

static __device__ __forceinline__ float ggml_cuda_ar_f24_value(const unsigned int t) {
    return __uint_as_float(t << 8);
}

static __device__ __forceinline__ uint4 ggml_cuda_ar_pack(const unsigned int t0, const unsigned int t1, const unsigned int t2,
                                                          const unsigned int t3, const int token) {
    return make_uint4(t0 | (t1 << 24), (t1 >> 8) | (t2 << 16), (t2 >> 16) | (t3 << 8), (unsigned int) token);
}

static __device__ __forceinline__ float4 ggml_cuda_ar_unpack(const uint4 u) {
    return make_float4(ggml_cuda_ar_f24_value(u.x & 0xffffffu), ggml_cuda_ar_f24_value((u.x >> 24) | ((u.y & 0xffffu) << 8)),
                       ggml_cuda_ar_f24_value((u.y >> 16) | ((u.z & 0xffu) << 16)), ggml_cuda_ar_f24_value(u.z >> 8));
}

// io: one per device, in rank order
void ggml_cuda_ar_window_next(ggml_cuda_ar_pipeline * pipeline, ggml_cuda_ar_window_io * io);

// AllReduce fused with the residual ADD -> RMS_NORM -> MUL that follows every
// tensor-parallel projection in a decoder layer.  Writes the ADD output (the
// residual stream) and the MUL output; the reduced tensor and the RMS_NORM
// intermediate are not materialized.  One 1024-thread block per row keeps the
// arithmetic bit-identical to ggml_cuda_ar_kernel followed by rms_norm_f32<1024>
// with its fused pre-add and multiply.  Rows are limited by the per-block
// arrival ring; the caller checks *_supported() at graph-build time.
bool ggml_cuda_ar_allreduce_add_rms_norm_mul_supported(
    const ggml_cuda_ar_pipeline * pipeline, int64_t ncols, int64_t nrows);

bool ggml_cuda_ar_allreduce_add_rms_norm_mul(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors,       // partial sums, F32 [ncols, nrows] (only read)
    ggml_tensor           ** residuals,     // F32 [ncols, nrows]
    ggml_tensor           ** add_outputs,   // F32 [ncols, nrows] = reduced + residual
    ggml_tensor           ** norm_weights,  // F32 [ncols]
    ggml_tensor           ** norm_outputs,  // F32 [ncols, nrows] = rms_norm(add) * weight
    float                    eps);
