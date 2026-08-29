// Q4_1 mat-vec for GP100 using LOP3 nibble expansion and HFMA2.

#include "mmvq-f16-sm60.cuh"
#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#define A16_QK     32 // elements per block, matching block_q4_1
#define A16_NWARPS  4
#define A16_ROWS    4 // output rows per CUDA block
#define A16_QG      8 // lanes per quantization group
#define A16_QBLOCK 128

// Quantize each 32-value block with eight lanes. This keeps the output permutation lane-local.
template <bool vec4>
static __global__ void quantize_a16(
        const float * __restrict__ x, half2 * __restrict__ aq, half2 * __restrict__ ads,
        const int s01, const int nblocks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL

    constexpr int G = A16_QG;       // lanes per block
    constexpr int E = A16_QK/G;     // elements per lane

    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int ib  = tid/G;          // block index within the row
    const int l   = tid % G;        // lane within the block
    const int row = blockIdx.y;
    if (ib >= nblocks) {
        return;
    }

    // Each group stays within one warp, so all shuffle lanes are active.
    const unsigned int gmask = ((1u << G) - 1) << ((threadIdx.x & (WARP_SIZE - 1)) & ~(G - 1));

    const float * xp = x + (size_t) row*s01 + ib*A16_QK + l*E;

    float v[E];
    float amax = 0.0f;
    if (vec4) {
#pragma unroll
        for (int i = 0; i < E/4; ++i) {
            const float4 t = ((const float4 *) xp)[i];
            v[4*i + 0] = t.x; v[4*i + 1] = t.y; v[4*i + 2] = t.z; v[4*i + 3] = t.w;
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
        }
    } else {
#pragma unroll
        for (int i = 0; i < E; ++i) {
            v[i] = xp[i];
            amax = fmaxf(amax, fabsf(v[i]));
        }
    }
#pragma unroll
    for (int s = G/2; s > 0; s >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(gmask, amax, s, G));
    }

    const float id = amax > 0.0f ? 1.0f/amax : 0.0f;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        v[i] *= id;
        sum  += v[i];
    }
#pragma unroll
    for (int s = G/2; s > 0; s >>= 1) {
        sum += __shfl_xor_sync(gmask, sum, s, G);
    }

    if (l == 0) {
        ads[row*nblocks + ib] = make_half2(__float2half(amax), __float2half(sum));
    }

    // Lanes holding elements 0..15 write outputs 4k+0 / 4k+2; those holding 16..31 write 4k+1 / 4k+3.
    const int off   = l >= G/2 ? 1 : 0;
    const int kbase = (E*l - off*16)/4;

    half2 * qs = aq + (size_t) row*nblocks*(A16_QK/2) + ib*(A16_QK/2);
#pragma unroll
    for (int e = 0; e < E/4; ++e) {
        const int k = kbase + e;
        qs[4*k + 0 + off] = make_half2(__float2half(v[4*e + 0]), __float2half(v[4*e + 2]));
        qs[4*k + 2 + off] = make_half2(__float2half(v[4*e + 1]), __float2half(v[4*e + 3]));
    }
#else
    GGML_UNUSED_VARS(x, aq, ads, s01, nblocks);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// Load activations before rows so register use does not grow with nrows_block.
// Keep the cross-block accumulator in float to avoid fp16 accumulation error.
template <int ncols_dst, int nwarps, int nrows_block = A16_ROWS, bool g64_fmt = false>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps,
        (ncols_dst >= 6 ? 2 : 4)*(A16_NWARPS/nwarps))
mul_mat_vec_q4_1_a16(
        const void * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int nblocks, const int stride_row_x,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    const int row0     = blockIdx.x*nrows_block;
    const int tid      = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nthreads = WARP_SIZE*nwarps;

    const block_q4_1 * x = (const block_q4_1 *) vx;

    const half2 magic_lo = __float2half2_rn(1024.0f);
    const half2 magic_hi = __float2half2_rn(  64.0f);

    float sumf[ncols_dst][nrows_block] = {{0.0f}};

    // Hoist row bases so the 20-byte Q4_1 stride does not add an XMAD chain inside the loop.
    // G64 passes byte strides because its 36-byte groups do not tile as block_q4_1.
    const char * const xbase = g64_fmt ? (const char *) vx + (size_t) row0*stride_row_x
                                       : (const char *) &x[(size_t) row0*stride_row_x];
    int rowoff[nrows_block];
#pragma unroll
    for (int i = 0; i < nrows_block; ++i) {
        rowoff[i] = g64_fmt ? i*stride_row_x : i*stride_row_x*(int) sizeof(block_q4_1);
    }

    for (int t = tid; t < 2*nblocks; t += nthreads) {
        const int kb = t >> 1;
        const int kh = t &  1;
        // G64 group: {half2 dm}{16-byte low half}{16-byte high half}.
        [[maybe_unused]] int koff_g = 0, dmoff_g = 0;
        if constexpr (g64_fmt) {
            const int g = kb >> 1;
            koff_g  = g*36 + 4 + (kb & 1)*16;
            dmoff_g = g*36;
        }
        const int koff = kb*(int) sizeof(block_q4_1);

        half2 a[ncols_dst][8];
        float yd[ncols_dst];
        float ys[ncols_dst];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const half2 * ap = aq + (size_t) j*stride_col_aq + kb*(A16_QK/2) + kh*8;
#pragma unroll
            for (int c2 = 0; c2 < 2; ++c2) {
                const int4 av = *((const int4 *) (ap + 4*c2));
                a[j][4*c2 + 0] = *((const half2 *) &av.x);
                a[j][4*c2 + 1] = *((const half2 *) &av.y);
                a[j][4*c2 + 2] = *((const half2 *) &av.z);
                a[j][4*c2 + 3] = *((const half2 *) &av.w);
            }
            const half2 ds = ads[j*stride_col_ads + kb];
            yd[j] = __low2float(ds);
            ys[j] = __high2float(ds)*0.5f;
        }

#pragma unroll
        for (int i = 0; i < nrows_block; ++i) {
            float2 dm;
            const int * wq;
            if constexpr (g64_fmt) {
                dm = __half22float2(*(const half2 *) (xbase + (rowoff[i] + dmoff_g)));
                wq = (const int *) (xbase + (rowoff[i] + koff_g));
            } else {
                const block_q4_1 * b = (const block_q4_1 *) (xbase + (rowoff[i] + koff));
                dm = __half22float2(b->dm);
                wq = (const int *) b->qs;
            }

            half2 w[8];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int w32  = wq[2*kh + k];
                const int w32s = w32 >> 8;
                int t0, t1, t2, t3;
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t0) : "r"(w32),  "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t1) : "r"(w32),  "n"(0x00f000f0), "n"(0x54005400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t2) : "r"(w32s), "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t3) : "r"(w32s), "n"(0x00f000f0), "n"(0x54005400));
                w[4*k + 0] = __hsub2(*((const half2 *) &t0), magic_lo);
                w[4*k + 1] = __hsub2(*((const half2 *) &t1), magic_hi);
                w[4*k + 2] = __hsub2(*((const half2 *) &t2), magic_lo);
                w[4*k + 3] = __hsub2(*((const half2 *) &t3), magic_hi);
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                half2 s = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int c2 = 0; c2 < 2; ++c2) {
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        s = __hfma2(w[4*c2 + c], a[j][4*c2 + c], s);
                    }
                }
                const float af = __half2float(__hadd(__low2half(s), __high2half(s)));
                sumf[j][i] += yd[j]*(dm.x*af + dm.y*ys[j]);
            }
        }
    }

    // XOR shuffles place the sum in every lane, so one warp needs no shared memory.
    if constexpr (nwarps == 1) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < nrows_block; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (threadIdx.x == i) {
                    dst[(size_t) j*stride_col_dst + row0 + i] = v;
                }
            }
        }
    } else {
        __shared__ float tmp[nwarps][ncols_dst][nrows_block];

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < nrows_block; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (threadIdx.x == 0) {
                    tmp[threadIdx.y][j][i] = v;
                }
            }
        }

        __syncthreads();

        if (threadIdx.y != 0) {
            return;
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < nrows_block; ++i) {
                if (threadIdx.x == i) {
                    float s = 0.0f;
#pragma unroll
                    for (int w = 0; w < nwarps; ++w) {
                        s += tmp[w][j][i];
                    }
                    dst[(size_t) j*stride_col_dst + row0 + i] = s;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

// Wide grids use fewer warps to reduce cross-warp work. Narrow grids use more to fill the SMs.
static int a16_pick_nwarps(const int nblocks, const int nblocks_grid) {
    int nw = nblocks_grid < 64 ? A16_NWARPS : 2;
    while (nw > 1 && 2*nblocks < WARP_SIZE*nw) {
        nw /= 2;
    }
    return nw;
}

template <int ncols_dst>
static void launch_a16(
        const void * vx, const half2 * aq, const half2 * ads, float * dst, const int nblocks,
        const int nrows, const int stride_row_x, const int stride_col_aq, const int stride_col_ads,
        const int stride_col_dst, cudaStream_t stream) {
    // Tall tensors use eight-row tiles to amortize scale loads and row addressing.
    if (nrows >= 1024 && nrows % 8 == 0) {
        const dim3 grid(nrows/8, 1, 1);
        switch (a16_pick_nwarps(nblocks, nrows/8)) {
            case 1:
                mul_mat_vec_q4_1_a16<ncols_dst, 1, 8><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                    vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
                return;
            case 2:
                mul_mat_vec_q4_1_a16<ncols_dst, 2, 8><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                    vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
                return;
            default:
                mul_mat_vec_q4_1_a16<ncols_dst, A16_NWARPS, 8><<<grid, dim3(WARP_SIZE, A16_NWARPS, 1), 0, stream>>>(
                    vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
                return;
        }
    }

    const dim3 grid(nrows/A16_ROWS, 1, 1);
    switch (a16_pick_nwarps(nblocks, nrows/A16_ROWS)) {
        case 1:
            mul_mat_vec_q4_1_a16<ncols_dst, 1><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
            return;
        case 2:
            mul_mat_vec_q4_1_a16<ncols_dst, 2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
            return;
        default:
            mul_mat_vec_q4_1_a16<ncols_dst, A16_NWARPS><<<grid, dim3(WARP_SIZE, A16_NWARPS, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
            return;
    }
}


template <int ncols_dst>
static void launch_a16_g64(
        const void * vx, const half2 * aq, const half2 * ads, float * dst, const int nblocks,
        const int nrows, const int stride_row_x, const int stride_col_aq, const int stride_col_ads,
        const int stride_col_dst, cudaStream_t stream) {
    const dim3 grid(nrows/8, 1, 1);
    switch (a16_pick_nwarps(nblocks, nrows/8)) {
        case 1:
            mul_mat_vec_q4_1_a16<ncols_dst, 1, 8, true><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
            return;
        case 2:
            mul_mat_vec_q4_1_a16<ncols_dst, 2, 8, true><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
            return;
        default:
            mul_mat_vec_q4_1_a16<ncols_dst, A16_NWARPS, 8, true><<<grid, dim3(WARP_SIZE, A16_NWARPS, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
            return;
    }
}

bool ggml_cuda_mmvq_f16_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst) {
    if (ids) {
        return false;
    }
    const bool is_g64  = src0->type == GGML_TYPE_Q4_1_G64;
    const bool is_q4_1 = src0->type == GGML_TYPE_Q4_1 || is_g64;
    if (!is_q4_1 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float)) {
        return false;
    }
    if (src1->ne[0] != src0->ne[0] || dst->ne[0] != src0->ne[1]) {
        return false;
    }
    // The generic kernel is faster outside GP100.
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc != GGML_CUDA_CC_PASCAL || !fp16_available(cc)) {
        return false;
    }
    // G64 folds [k, 1, nseq] activations into columns when the weight broadcasts over nseq.
    const bool g64_seq_fold = is_g64 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
                              src1->ne[1] == 1 && src1->ne[2] > 1 && src1->ne[2] <= 8 &&
                              src1->ne[3] == 1 && dst->ne[1] == 1 && dst->ne[2] == src1->ne[2] &&
                              dst->ne[3] == 1;
    if (!g64_seq_fold &&
            (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1)) {
        return false;
    }
    if (!g64_seq_fold &&
            (dst->ne[1] != src1->ne[1] || dst->ne[2] != 1 || dst->ne[3] != 1)) {
        return false;
    }
    if (src0->ne[0] % A16_QK != 0 || src0->ne[1] % A16_ROWS != 0 ||
            src0->ne[0]/A16_QK > INT_MAX/A16_QG || src0->ne[1] > INT_MAX || src0->nb[1] > INT_MAX) {
        return false;
    }
    // The kernel keeps offsets for an eight-row tile in 32-bit registers.  The
    // global row base remains size_t; only the seven-row local span is bounded.
    if (src0->nb[1] > INT_MAX/7) {
        return false;
    }
    const size_t row_stride = is_g64 ? src0->nb[1] : src0->nb[1]/sizeof(block_q4_1);
    if ((!is_g64 && src0->nb[1] % sizeof(block_q4_1) != 0) || row_stride > INT_MAX) {
        return false;
    }
    // Generic MMVQ is faster for Q4_1 width one and for widths above eight.
    // G64 has no generic CUDA vec_dot, so it also uses this path at width one.
    const int64_t eff_cols = src1->ne[1] == 1 && src1->ne[2] > 1 ? src1->ne[2] : src1->ne[1];
    if (eff_cols < (is_g64 ? 1 : 2) || eff_cols > 8) {
        return false;
    }
    if (is_g64 && (src0->ne[0] % 64 != 0 || src0->ne[1] % 8 != 0)) {
        return false;
    }
    const size_t src1_stride = g64_seq_fold ? src1->nb[2] : src1->nb[1];
    const size_t dst_stride  = g64_seq_fold ? dst->nb[2]  : dst->nb[1];
    if (src1_stride % sizeof(float) != 0 || src1_stride/sizeof(float) > INT_MAX ||
            dst_stride % sizeof(float) != 0 || dst_stride/sizeof(float) > INT_MAX) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

void ggml_cuda_mmvq_f16_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    // seq->columns fold (see supported()): [k, 1, nseq] activations become nseq columns.
    // Only Q4_1_G64 is ever dispatched here with ne[2] > 1.
    const bool    seq_fold = src1->ne[1] == 1 && src1->ne[2] > 1;
    const int64_t ne11     = seq_fold ? src1->ne[2] : src1->ne[1];
    const int nblocks  = ne00/A16_QK;

    // Store normalized half2 activations plus per-block scale and sum.
    const size_t aq_size  = (size_t) ne11*nblocks*(A16_QK/2)*sizeof(half2);
    const size_t ads_size = (size_t) ne11*nblocks*sizeof(half2);

    const int64_t s11 = (seq_fold ? src1->nb[2] : src1->nb[1])/ggml_type_size(src1->type);

    // Reuse quantized activations for adjacent projections on one stream.
    ggml_cuda_pool_alloc<half2> aq_scoped (ctx.pool());
    ggml_cuda_pool_alloc<half2> ads_scoped(ctx.pool());
    half2 * aq  = nullptr;
    half2 * ads = nullptr;
    const size_t total_size = aq_size + ads_size;

    const bool cacheable = ctx.curr_stream_no == 0 &&
                           (ctx.mmvq_f16_cache_stream == nullptr || ctx.mmvq_f16_cache_stream == stream);
    const bool hit = cacheable &&
                     ctx.mmvq_f16_cache_mem        != nullptr &&
                     ctx.mmvq_f16_cache_src        == src1 &&
                     ctx.mmvq_f16_cache_data       == src1->data &&
                     ctx.mmvq_f16_cache_stream     == stream &&
                     ctx.mmvq_f16_cache_aq_size    == aq_size &&
                     ctx.mmvq_f16_cache_stats_size == ads_size &&
                     ctx.mmvq_f16_cache_shape[0]   == nblocks &&
                     ctx.mmvq_f16_cache_shape[1]   == ne11 &&
                     ctx.mmvq_f16_cache_shape[2]   == s11;

    if (hit) {
        aq  = (half2 *) ctx.mmvq_f16_cache_mem;
        ads = (half2 *) (ctx.mmvq_f16_cache_mem + aq_size);
    } else if (!cacheable) {
        aq  = aq_scoped.alloc(aq_size/sizeof(half2));
        ads = ads_scoped.alloc(ads_size/sizeof(half2));
    } else {
        if (total_size > ctx.mmvq_f16_cache_cap) {
            ctx.mmvq_f16_cache_free();
            ggml_cuda_set_device(ctx.device);
            CUDA_CHECK(cudaMalloc((void **) &ctx.mmvq_f16_cache_mem, total_size));
            ctx.mmvq_f16_cache_cap = total_size;
        }
        aq  = (half2 *) ctx.mmvq_f16_cache_mem;
        ads = (half2 *) (ctx.mmvq_f16_cache_mem + aq_size);
    }

    if (!hit) {
        const int nthreads = nblocks*A16_QG;
        const dim3 grid((nthreads + A16_QBLOCK - 1)/A16_QBLOCK, ne11, 1);
        // Use float4 only when every activation row is 16-byte aligned.
        const bool vec4 = s11 % 4 == 0 && ((uintptr_t) src1->data) % 16 == 0;
        if (vec4) {
            quantize_a16<true> <<<grid, A16_QBLOCK, 0, stream>>>(
                (const float *) src1->data, aq, ads, s11, nblocks);
        } else {
            quantize_a16<false><<<grid, A16_QBLOCK, 0, stream>>>(
                (const float *) src1->data, aq, ads, s11, nblocks);
        }
        if (cacheable) {
            ctx.mmvq_f16_cache_src        = src1;
            ctx.mmvq_f16_cache_data       = src1->data;
            ctx.mmvq_f16_cache_stream     = stream;
            ctx.mmvq_f16_cache_aq_size    = aq_size;
            ctx.mmvq_f16_cache_stats_size = ads_size;
            ctx.mmvq_f16_cache_shape[0]   = nblocks;
            ctx.mmvq_f16_cache_shape[1]   = ne11;
            ctx.mmvq_f16_cache_shape[2]   = s11;
        }
    }

    if (src0->type == GGML_TYPE_Q4_1_G64) {
        const int  ne11_eff = (int) ne11;
        const int stride_row_g  = (int) src0->nb[1];
        const int stride_col_aqg  = nblocks*(A16_QK/2);
        const int stride_col_adsg = nblocks;
        const int stride_col_dstg = seq_fold ? (int) (dst->nb[2]/ggml_type_size(dst->type))
                                             : (int) (dst->nb[1]/ggml_type_size(dst->type));
        switch (ne11_eff) {
#define A16_G64_CASE(N)                                                                            \
            case N:                                                                                \
                launch_a16_g64<N>(src0->data, aq, ads, (float *) dst->data, nblocks, ne01,         \
                        stride_row_g, stride_col_aqg, stride_col_adsg, stride_col_dstg, stream);   \
                break;
            A16_G64_CASE(1) A16_G64_CASE(2) A16_G64_CASE(3) A16_G64_CASE(4)
            A16_G64_CASE(5) A16_G64_CASE(6) A16_G64_CASE(7) A16_G64_CASE(8)
#undef A16_G64_CASE
            default:
                GGML_ABORT("unsupported width for the sm_60 Q4_1_G64 mat-vec");
        }
        return;
    }

    const int stride_row_x   = src0->nb[1]/sizeof(block_q4_1);
    const int stride_col_aq  = nblocks*(A16_QK/2);
    const int stride_col_ads = nblocks;
    const int stride_col_dst = dst->nb[1]/ggml_type_size(dst->type);

#define A16_CASE(N)                                                                            \
    case N:                                                                                    \
        launch_a16<N>(src0->data, aq, ads, (float *) dst->data, nblocks, ne01,     \
                      stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst, stream);    \
        break;
    switch (ne11) {
        A16_CASE(2) A16_CASE(3) A16_CASE(4)
        A16_CASE(5) A16_CASE(6) A16_CASE(7) A16_CASE(8)
        default:
            GGML_ABORT("unsupported width for the sm_60 Q4_1 mat-vec");
    }
#undef A16_CASE
}

#else

bool ggml_cuda_mmvq_f16_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst) {
    GGML_UNUSED_VARS(src0, src1, ids, dst);
    return false;
}

void ggml_cuda_mmvq_f16_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    GGML_ABORT("sm_60 MMVQ is only available with the CUDA backend");
}

#endif
