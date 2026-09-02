#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

// Packed max/location key: the float's total order in the high 32 bits, the
// bitwise-inverted column in the low 32 so that a max over keys prefers the
// lowest column on ties, matching argmax_f32.  A row that is entirely NaN
// yields key 0 (column ~0 == UINT32_MAX), which a consumer can detect.
static __device__ __forceinline__ uint64_t maxloc_key(float value, uint32_t col) {
    if (isnan(value)) {
        return 0;
    }
    const uint32_t bits    = __float_as_uint(value);
    const uint32_t ordered = (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
    return (uint64_t(ordered) << 32) | uint32_t(~col);
}

// One packed candidate per row.  Used when the row is a shard of a larger
// row that lives on several devices: the shards' keys can be max-merged by
// the reader without a second pass over the data.
static __global__ void argmax_maxloc_f32(const float * __restrict__ x, uint64_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row  = blockIdx.x;
    const float * rowx = x + row*ncols;

    uint64_t best = 0;
    for (uint32_t col = threadIdx.x; col < (uint32_t) ncols; col += blockDim.x) {
        best = max(best, maxloc_key(rowx[col], col));
    }
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        best = max(best, __shfl_xor_sync(0xFFFFFFFF, best, offset, WARP_SIZE));
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int max_warps = 1024 / WARP_SIZE;
        __shared__ uint64_t shared_best[max_warps];
        if (lane_id == 0) {
            shared_best[warp_id] = best;
        }
        __syncthreads();
        if (warp_id == 0) {
            best = lane_id < n_warps ? shared_best[lane_id] : 0;
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                best = max(best, __shfl_xor_sync(0xFFFFFFFF, best, offset, WARP_SIZE));
            }
        }
    }
    if (warp_id == 0 && lane_id == 0) {
        dst[row] = best;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32 || dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    if (dst->type == GGML_TYPE_I64) {
        argmax_maxloc_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, (uint64_t *) dst->data, ne00);
    } else {
        argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, (int32_t *) dst->data, ne00);
    }
}
