#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the CUDA device IDs in rank order.
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
