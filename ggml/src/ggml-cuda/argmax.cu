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
//
// The row is split over gridDim.x blocks that each reduce a chunk and
// max-merge it into dst[row] with a 64-bit atomicMax; dst is zeroed by the
// caller.  Every non-NaN key is positive, so 0 is the identity and the result
// is independent of the split.  One block per row was latency-bound: a
// 124k-column head shard took 62 us because each thread walked its columns
// with one outstanding load; split eight ways with float4 loads it is ~5 us.
static __global__ void argmax_maxloc_f32(const float * __restrict__ x, uint64_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row  = blockIdx.y;
    const float * rowx = x + row*ncols;

    const int64_t chunk = ((ncols + gridDim.x - 1)/gridDim.x + 3) & ~(int64_t) 3;
    const int64_t c0    = (int64_t) blockIdx.x*chunk;
    const int64_t c1    = min(c0 + chunk, ncols);

    uint64_t best = 0;
    if (c0 < c1) {
        const int64_t n4 = (c1 - c0) & ~(int64_t) 3;
        if ((((uintptr_t) (rowx + c0)) & 15) == 0) {
            const float4 * rowx4 = (const float4 *) (rowx + c0);
#pragma unroll 4
            for (int64_t i = threadIdx.x; i < n4/4; i += blockDim.x) {
                const float4 v = rowx4[i];
                const uint32_t col = (uint32_t) (c0 + 4*i);
                best = max(best, maxloc_key(v.x, col + 0));
                best = max(best, maxloc_key(v.y, col + 1));
                best = max(best, maxloc_key(v.z, col + 2));
                best = max(best, maxloc_key(v.w, col + 3));
            }
            for (int64_t col = c0 + n4 + threadIdx.x; col < c1; col += blockDim.x) {
                best = max(best, maxloc_key(rowx[col], (uint32_t) col));
            }
        } else {
#pragma unroll 4
            for (int64_t col = c0 + threadIdx.x; col < c1; col += blockDim.x) {
                best = max(best, maxloc_key(rowx[col], (uint32_t) col));
            }
        }
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
        atomicMax((unsigned long long *) &dst[row], (unsigned long long) best);
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
        // split long rows over several blocks; keys merge with atomicMax onto a zeroed dst
        const int64_t nsplit = std::max<int64_t>(1, std::min<int64_t>(16, ne00/8192));
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, nrows*sizeof(uint64_t), stream));
        argmax_maxloc_f32<<<dim3(nsplit, nrows, 1), blocks_dim, 0, stream>>>(src0_d, (uint64_t *) dst->data, ne00);
    } else {
        argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, (int32_t *) dst->data, ne00);
    }
}
