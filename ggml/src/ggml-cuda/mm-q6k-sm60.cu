// Q6_K MUL_MAT mat-vec for decode batches on sm_60 (GP100), one warp per row: the trunks of Project Maya's files.
// The math of a lane is in mm-q6k-sm60-math.cuh (a CPU test compiles it too); the activation records come from
// ggml_cuda_mmid_kq_rows (mmid-kq-sm60.cu), which also dispatches here for Q6_K.
//
// A block is KQ_NWARPS warps = rows; a warp loads the 53 words of a super-block of its row contiguously (lane l: word l,
// lanes 0..20: word 32 + l), shuffles the qh / scale / d words to the lanes that use them and multiplies its 8 weights with
// the activations of every column (up to MMVQ_MAX_BATCH_SIZE), so the weights are read once for all columns.

#include "mmid-kq-sm60.cuh"
#include "mmvq.cuh"

#include "mm-q6k-sm60-math.cuh"

// the per-column sums and record pointers of 4+ columns need more than the 64 registers of 8 resident blocks
template <int ncols>
static __global__ void __launch_bounds__(WARP_SIZE*KQ_NWARPS, ncols <= 3 ? 8 : 4)
mm_q6k_vec(
        const char * __restrict__ w, const char * __restrict__ x, float * __restrict__ dst, const int nsb, const int ne01,
        const int64_t nb01, const int64_t nb02, const int ne11, const int r2, const int64_t dst_nb1, const int64_t dst_nb2) {
#if defined(FP16_AVAILABLE)
    const int row  = blockIdx.x*KQ_NWARPS + threadIdx.y;
    const int i02  = blockIdx.z;
    const int lane = threadIdx.x;
    if (row >= ne01) {
        return;
    }
    const int n  = lane >> 4, i = lane & 15, j0 = i >> 3, ap = i & 7;
    const int ixw0 = q6_rec_xw_index(lane, j0), ixw1 = q6_rec_xw_index(lane, j0 + 2), ixx = q6_rec_xx_index(lane, j0)/2;
    // the columns: src1 rows i1 of the r2 channels that broadcast this matrix; the record of column c is row i2*ne11 + i1
    const char * xr[ncols];
    int          i1s[ncols], i2s[ncols];
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        const int i2r = r2 > 1 ? c / ne11 : 0;
        i1s[c] = c - i2r*ne11;
        i2s[c] = i02*r2 + i2r;
        xr[c]  = x + (int64_t) (i2s[c]*ne11 + i1s[c])*nsb*KQ_REC_BYTES;
    }
    float sum[ncols];
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        sum[c] = 0.0f;
    }
    const char * blk = w + (int64_t) i02*nb02 + (int64_t) row*nb01;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint32_t   mis = (uint32_t) (uintptr_t) blk & 3;   // the block starts 0 or 2 bytes into an aligned word
        const uint32_t * pb  = (const uint32_t *) (blk - mis);
        const uint32_t   sh  = 8*mis;
        const uint32_t w0 = __funnelshift_r(pb[lane], pb[lane + 1], sh);
        uint32_t w1 = 0;
        if (lane < 21) {
            const uint32_t hi = (lane == 20 && sh == 0) ? 0u : pb[33 + lane];   // the word after d belongs to the next block
            w1 = __funnelshift_r(pb[32 + lane], hi, sh);
        }
        q6_lane_words lw;
        lw.ql  = w0;
        lw.qh  = __shfl_sync(0xffffffff, w1, 8*n + ap);
        lw.sc  = __shfl_sync(0xffffffff, w1, 16 + 2*n);
        lw.sc2 = __shfl_sync(0xffffffff, w1, 17 + 2*n);
        lw.d16 = __shfl_sync(0xffffffff, w1, 20) & 0xFFFFu;
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            const char *  rec = xr[c] + (int64_t) sb*KQ_REC_BYTES;
            const uint2   xw0 = ((const uint2 *) rec)[ixw0];
            const uint2   xw1 = ((const uint2 *) rec)[ixw1];
            const uint32_t xx = ((const uint32_t *) rec)[ixx];
            const float    xs = ((const float *) rec)[kq_conv_xs_index(0, 0)];
            sum[c] = fmaf(xs, q6_lane_block(lw, lane, xw0, xw1, xx), sum[c]);
        }
        blk += 210;
    }
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            sum[c] += __shfl_xor_sync(0xffffffff, sum[c], off);
        }
        if (lane == 0) {
            *((float *) ((char *) dst + (int64_t) i2s[c]*dst_nb2 + (int64_t) i1s[c]*dst_nb1) + row) = sum[c];
        }
    }
#else
    GGML_UNUSED_VARS(w, x, dst, nsb, ne01, nb01, nb02, ne11, r2, dst_nb1, dst_nb2);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// x: the converted rows of src1 (ggml_cuda_mmid_kq_rows or a per-op conversion), in the order (i2, i1)
void ggml_cuda_mm_q6k_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * x) {
    cudaStream_t stream = ctx.stream();
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int     nsb  = src0->ne[0]/256;
    const int     r2   = ne12/ne02;
    const int     ncols = (int) (ne11*r2);
    GGML_ASSERT(ncols >= 1 && ncols <= MMVQ_MAX_BATCH_SIZE);
    const dim3 grid((ne01 + KQ_NWARPS - 1)/KQ_NWARPS, 1, ne02);
    const dim3 block(WARP_SIZE, KQ_NWARPS, 1);
    const char * w = (const char *) src0->data;
    float * d = (float *) dst->data;
#define Q6_LAUNCH(N) mm_q6k_vec<N><<<grid, block, 0, stream>>>(w, x, d, nsb, (int) ne01, src0->nb[1], src0->nb[2], (int) ne11, r2, dst->nb[1], dst->nb[2])
    switch (ncols) {
        case 1: Q6_LAUNCH(1); break;
        case 2: Q6_LAUNCH(2); break;
        case 3: Q6_LAUNCH(3); break;
        case 4: Q6_LAUNCH(4); break;
        case 5: Q6_LAUNCH(5); break;
        case 6: Q6_LAUNCH(6); break;
        case 7: Q6_LAUNCH(7); break;
        default: Q6_LAUNCH(8); break;
    }
#undef Q6_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
