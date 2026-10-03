// Q2_0 MUL_MAT_ID for prompt batches on sm_60 (GP100).
//
// GP100 has no DP4A, so MMQ emulates every 4-way int8 dot product with byte extracts and integer multiplies.
// HFMA2 does two FP16 MACs per instruction at full rate instead. Here each 64x(16*NJ) tile of one expert is
// computed with HFMA2: the 2-bit codes are decoded to half in shared memory with LOP3 magic constants (exact),
// activations are scaled per 64 values into [-1, 1] and stored as half, and each 64-value block is summed in
// half2 before its weight and activation scales are applied in FP32.
//
// Pair order: half2 pair j of a 16-value group g holds elements (16g + j, 16g + j + 8). This is the order the
// LOP3 decode produces; the activations are converted to the same order, so the dot product is unchanged.

#include "mmid-f16-sm60.cuh"
#include "mmid.cuh"

#define MMID16_TM      64
#define MMID16_THREADS 256
#define MMID16_LDS     36 // words per shared row: 32 half2 + 4 pad keeps LDS.128 aligned and conflict free

// Scale 64 activations into [-1, 1] and store them as half2 pairs in the decode order.
static __device__ __forceinline__ void mmid16_convert_block(const float * __restrict__ x, half2 * __restrict__ out, float * __restrict__ scale) {
    const float4 * xp = (const float4 *) x;
    float v[64];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float4 t = xp[i];
        v[4*i + 0] = t.x; v[4*i + 1] = t.y; v[4*i + 2] = t.z; v[4*i + 3] = t.w;
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
    }
    const float s  = amax > 0.0f ? amax : 1.0f;
    const float id = 1.0f/s;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[8*g + j] = make_half2(__float2half(v[16*g + j]*id), __float2half(v[16*g + j + 8]*id));
        }
    }
    *scale = s;
}

// one compact (expert-sorted) slot per block
static __global__ void mmid16_convert_act(
        const float * __restrict__ src1, const int32_t * __restrict__ ids_src1, half2 * __restrict__ xh,
        float * __restrict__ xs, const int64_t stride_row, const int nkb) {
    const int     c = blockIdx.x;
    const float * x = src1 + (int64_t) ids_src1[c]*stride_row;
    for (int kb = threadIdx.x; kb < nkb; kb += blockDim.x) {
        mmid16_convert_block(x + kb*64, xh + ((int64_t) c*nkb + kb)*32, xs + (int64_t) c*nkb + kb);
    }
}

// one src1 row (i11, i12) per block, in src1 order
static __global__ void mmid16_convert_rows(
        const float * __restrict__ src1, half2 * __restrict__ xh, float * __restrict__ xs,
        const int64_t s11, const int64_t s12, const int ne11, const int nkb) {
    const int     r = blockIdx.x;
    const float * x = src1 + (int64_t) (r / ne11)*s12 + (int64_t) (r % ne11)*s11;
    for (int kb = threadIdx.x; kb < nkb; kb += blockDim.x) {
        mmid16_convert_block(x + kb*64, xh + ((int64_t) r*nkb + kb)*32, xs + (int64_t) r*nkb + kb);
    }
}

// One block: exclusive scan of the per-expert tile counts, then a tile -> (expert, first slot) table.
template <int TN>
static __global__ void mmid16_tiles(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_expert,
        int32_t * __restrict__ tile_col0, const int n_experts) {
    __shared__ int scan[1024];
    const int e  = threadIdx.x;
    const int nt = e < n_experts ? (expert_bounds[e + 1] - expert_bounds[e] + TN - 1)/TN : 0;
    scan[e] = nt;
    __syncthreads();
    for (int off = 1; off < (int) blockDim.x; off <<= 1) {
        const int v = e >= off ? scan[e - off] : 0;
        __syncthreads();
        scan[e] += v;
        __syncthreads();
    }
    const int start = scan[e] - nt;
    for (int i = 0; i < nt; ++i) {
        tile_expert[start + i] = e;
        tile_col0[start + i]   = expert_bounds[e] + i*TN;
    }
}

static __device__ __forceinline__ half2 mmid16_h2(const uint32_t v) {
    return *((const half2 *) &v);
}

static __device__ __forceinline__ uint32_t mmid16_u32(const half2 v) {
    return *((const uint32_t *) &v);
}

// 16 Q2_0 codes (one 32-bit word) -> 8 half2 holding c - 1, pair j = elements (j, j + 8)
static __device__ __forceinline__ void mmid16_decode_q2_0(const uint32_t q, half2 * h) {
    const uint32_t qs = q >> 8;
    uint32_t t[8];
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[0]) : "r"(q),  "n"(0x00030003), "n"(0x64006400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[1]) : "r"(q),  "n"(0x000c000c), "n"(0x5c005c00));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[2]) : "r"(q),  "n"(0x00300030), "n"(0x54005400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[3]) : "r"(q),  "n"(0x00c000c0), "n"(0x4c004c00));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[4]) : "r"(qs), "n"(0x00030003), "n"(0x64006400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[5]) : "r"(qs), "n"(0x000c000c), "n"(0x5c005c00));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[6]) : "r"(qs), "n"(0x00300030), "n"(0x54005400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[7]) : "r"(qs), "n"(0x00c000c0), "n"(0x4c004c00));
    // code c sits on a magic base B (1024, 256, 64, 16 by bit position); B + 1 maps it to c - 1
    const half2 m0 = __float2half2_rn(1025.0f);
    const half2 m1 = __float2half2_rn( 257.0f);
    const half2 m2 = __float2half2_rn(  65.0f);
    const half2 m3 = __float2half2_rn(  17.0f);
    h[0] = __hsub2(mmid16_h2(t[0]), m0);
    h[1] = __hsub2(mmid16_h2(t[1]), m1);
    h[2] = __hsub2(mmid16_h2(t[2]), m2);
    h[3] = __hsub2(mmid16_h2(t[3]), m3);
    h[4] = __hsub2(mmid16_h2(t[4]), m0);
    h[5] = __hsub2(mmid16_h2(t[5]), m1);
    h[6] = __hsub2(mmid16_h2(t[6]), m2);
    h[7] = __hsub2(mmid16_h2(t[7]), m3);
}

// NJ token slots per thread, so a tile is 64 weight rows x 16*NJ slots.
template <int NJ>
static __global__ void __launch_bounds__(MMID16_THREADS, 2)
mmid16_q2_0_gemm(
        const char * __restrict__ w, const half2 * __restrict__ xh, const float * __restrict__ xs,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_expert, const int32_t * __restrict__ tile_col0,
        float * __restrict__ dst, const int nkb, const int64_t nb01, const int64_t nb02, const int64_t stride_dst) {
#if defined(FP16_AVAILABLE)
    constexpr int TN = 16*NJ;

    const int e = tile_expert[blockIdx.y];
    if (e < 0) {
        return;
    }
    const int c0   = tile_col0[blockIdx.y];
    const int cend = expert_bounds[e + 1];
    const int row0 = blockIdx.x*MMID16_TM;

    __shared__ __align__(16) uint32_t ws [MMID16_TM][MMID16_LDS];
    __shared__ __align__(16) uint32_t xsm[TN][MMID16_LDS];
    __shared__ float wd [MMID16_TM];
    __shared__ float xsc[TN];

    const int tid = threadIdx.x;
    const int tx  = tid % 16;
    const int ty  = tid / 16;

    const int lr = tid / 4; // weight row loaded by this thread
    const int lq = tid % 4; // 16-value group loaded by this thread
    const char * wrow = w + (int64_t) e*nb02 + (int64_t) (row0 + lr)*nb01;

    float acc[4][NJ];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int kb = 0; kb < nkb; ++kb) {
        {
            // block_q2_0 is 18 bytes, so the codes are only 2-byte aligned
            const char *     blk = wrow + (int64_t) kb*18;
            const uint16_t * q16 = (const uint16_t *) (blk + 2 + 4*lq);
            const uint32_t   q   = (uint32_t) q16[0] | ((uint32_t) q16[1] << 16);

            half2 h[8];
            mmid16_decode_q2_0(q, h);

            uint4 * wsp = (uint4 *) &ws[lr][8*lq];
            wsp[0] = make_uint4(mmid16_u32(h[0]), mmid16_u32(h[1]), mmid16_u32(h[2]), mmid16_u32(h[3]));
            wsp[1] = make_uint4(mmid16_u32(h[4]), mmid16_u32(h[5]), mmid16_u32(h[6]), mmid16_u32(h[7]));
            if (lq == 0) {
                wd[lr] = __half2float(*((const half *) blk));
            }
        }
#pragma unroll
        for (int i0 = 0; i0 < TN*8; i0 += MMID16_THREADS) {
            const int i    = i0 + tid;
            const int col  = i / 8;
            const int part = i % 8;
            const int c    = c0 + col;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (c < cend) {
                v = ((const uint4 *) (xh + ((int64_t) c*nkb + kb)*32))[part];
            }
            *((uint4 *) &xsm[col][4*part]) = v;
        }
        if (tid < TN) {
            const int c = c0 + tid;
            xsc[tid] = c < cend ? xs[(int64_t) c*nkb + kb] : 0.0f;
        }
        __syncthreads();

        half2 hacc[4][NJ];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                hacc[i][j] = make_half2(0.0f, 0.0f);
            }
        }
#pragma unroll
        for (int p4 = 0; p4 < 8; ++p4) {
            uint4 wv[4];
            uint4 xv[NJ];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wv[i] = *((const uint4 *) &ws[ty + 16*i][4*p4]);
            }
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                xv[j] = *((const uint4 *) &xsm[tx + 16*j][4*p4]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    hacc[i][j] = __hfma2(mmid16_h2(wv[i].x), mmid16_h2(xv[j].x), hacc[i][j]);
                    hacc[i][j] = __hfma2(mmid16_h2(wv[i].y), mmid16_h2(xv[j].y), hacc[i][j]);
                    hacc[i][j] = __hfma2(mmid16_h2(wv[i].z), mmid16_h2(xv[j].z), hacc[i][j]);
                    hacc[i][j] = __hfma2(mmid16_h2(wv[i].w), mmid16_h2(xv[j].w), hacc[i][j]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float dw = wd[ty + 16*i];
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                acc[i][j] += dw*xsc[tx + 16*j]*__half2float(__hadd(__low2half(hacc[i][j]), __high2half(hacc[i][j])));
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int c = c0 + tx + 16*j;
        if (c < cend) {
            float * d = dst + (int64_t) ids_dst[c]*stride_dst + row0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d[ty + 16*i] = acc[i][j];
            }
        }
    }
#else
    GGML_UNUSED_VARS(w, xh, xs, ids_dst, expert_bounds, tile_expert, tile_col0, dst, nkb, nb01, nb02, stride_dst);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// Mat-vec for up to MMVQ_MAX_BATCH_SIZE tokens: one token slot per blockIdx.y, MMID16V_RW rows per warp.
// A lane owns one 16-weight word of a block at a time for all rows of its warp, so the activation load is
// shared by the rows and every row adds independent loads in flight (the kernel is latency bound otherwise).
#define MMID16V_RW   8
#define MMID16V_ROWS (4*MMID16V_RW)
static __global__ void __launch_bounds__(128, 4)
mmid16_q2_0_vec(
        const char * __restrict__ w, const half2 * __restrict__ xh, const float * __restrict__ xs,
        const int32_t * __restrict__ ids, float * __restrict__ dst, const int nkb, const int64_t nb01, const int64_t nb02,
        const int ne11, const int n_used, const int si1, const int64_t stride_dst) {
#if defined(FP16_AVAILABLE)
    const int c = blockIdx.y;
    const int t = c / n_used;
    const int s = c % n_used;
    const int e = ids[t*si1 + s];
    const int r = t*ne11 + s % ne11;

    const half2 * xr   = xh + (int64_t) r*nkb*32;
    const float * xsr  = xs + (int64_t) r*nkb;
    const int     lane = threadIdx.x;
    const int     row0 = blockIdx.x*MMID16V_ROWS + threadIdx.y*MMID16V_RW;
    const char *  wr   = w + (int64_t) e*nb02 + (int64_t) row0*nb01;

    float acc[MMID16V_RW];
#pragma unroll
    for (int i = 0; i < MMID16V_RW; ++i) {
        acc[i] = 0.0f;
    }

    for (int it = lane; it < 4*nkb; it += WARP_SIZE) {
        const int kb = it >> 2;
        const int q  = it &  3;

        const uint4 * xp  = (const uint4 *) (xr + kb*32 + 8*q);
        const uint4   a0  = xp[0];
        const uint4   a1  = xp[1];
        const float   xsk = xsr[kb];

        uint32_t qq[MMID16V_RW];
        float    dd[MMID16V_RW];
#pragma unroll
        for (int i = 0; i < MMID16V_RW; ++i) {
            const char *     blk = wr + i*nb01 + kb*18;
            const uint16_t * q16 = (const uint16_t *) (blk + 2 + 4*q);
            qq[i] = (uint32_t) q16[0] | ((uint32_t) q16[1] << 16);
            dd[i] = __half2float(*(const half *) blk);
        }
#pragma unroll
        for (int i = 0; i < MMID16V_RW; ++i) {
            half2 h[8];
            mmid16_decode_q2_0(qq[i], h);
            half2 sacc = make_half2(0.0f, 0.0f);
            sacc = __hfma2(h[0], mmid16_h2(a0.x), sacc);
            sacc = __hfma2(h[1], mmid16_h2(a0.y), sacc);
            sacc = __hfma2(h[2], mmid16_h2(a0.z), sacc);
            sacc = __hfma2(h[3], mmid16_h2(a0.w), sacc);
            sacc = __hfma2(h[4], mmid16_h2(a1.x), sacc);
            sacc = __hfma2(h[5], mmid16_h2(a1.y), sacc);
            sacc = __hfma2(h[6], mmid16_h2(a1.z), sacc);
            sacc = __hfma2(h[7], mmid16_h2(a1.w), sacc);
            acc[i] += dd[i]*xsk*__half2float(__hadd(__low2half(sacc), __high2half(sacc)));
        }
    }

#pragma unroll
    for (int i = 0; i < MMID16V_RW; ++i) {
        const float v = warp_reduce_sum(acc[i]);
        if (lane == i) {
            dst[(int64_t) c*stride_dst + row0 + i] = v;
        }
    }
#else
    GGML_UNUSED_VARS(w, xh, xs, ids, dst, nkb, nb01, nb02, ne11, n_used, si1, stride_dst);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

bool ggml_cuda_mmid_f16_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_CUDA_SM60_MMID_F16") == nullptr || atoi(getenv("GGML_CUDA_SM60_MMID_F16")) != 0;
    if (!enabled || ids == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q2_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_PASCAL || cc >= GGML_CUDA_CC_DP4A) {
        return false;
    }
    if (src0->ne[0] % 64 != 0 || src0->ne[1] % MMID16_TM != 0 || src0->ne[2] > 1024 || src0->ne[3] != 1) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 ||
            src1->ne[3] != 1 || src1->nb[2] % src1->nb[1] != 0) {
        return false;
    }
    if (!ggml_is_contiguous(dst) || ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    return true;
}

void ggml_cuda_mmid_f16_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int64_t ne00     = src0->ne[0];
    const int64_t ne01     = src0->ne[1];
    const int64_t ne02     = src0->ne[2];
    const int64_t ne11     = src1->ne[1];
    const int64_t ne12     = src1->ne[2];
    const int     n_used   = ids->ne[0];
    const int64_t n_assign = ne12*n_used;
    const int     nkb      = ne00/64;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> bounds  (ctx.pool(), ne02 + 1);

    const int si1  = ids->nb[1]/sizeof(int32_t);
    const int sis1 = src1->nb[2]/src1->nb[1];
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
        ne02, ne12, n_used, ne11, si1, sis1, /*write_inverse =*/ false, stream);

    ggml_cuda_pool_alloc<half2> xh(ctx.pool(), n_assign*nkb*32);
    ggml_cuda_pool_alloc<float> xs(ctx.pool(), n_assign*nkb);
    mmid16_convert_act<<<n_assign, nkb <= 32 ? 32 : 64, 0, stream>>>(
        (const float *) src1->data, ids_src1.get(), xh.get(), xs.get(), src1->nb[1]/sizeof(float), nkb);

    // narrow tiles when experts see few slots each
    const bool    narrow = n_assign < 48*ne02;
    const int     tn     = narrow ? 32 : 64;
    const int64_t t_max  = (n_assign + tn - 1)/tn + ne02;
    GGML_ASSERT(t_max <= 65535);

    ggml_cuda_pool_alloc<int32_t> tile_expert(ctx.pool(), t_max);
    ggml_cuda_pool_alloc<int32_t> tile_col0  (ctx.pool(), t_max);
    CUDA_CHECK(cudaMemsetAsync(tile_expert.get(), 0xFF, t_max*sizeof(int32_t), stream));

    int nthr = 32;
    while (nthr < ne02) {
        nthr *= 2;
    }
    const dim3 grid(ne01/MMID16_TM, t_max, 1);
    if (narrow) {
        mmid16_tiles<32><<<1, nthr, 0, stream>>>(bounds.get(), tile_expert.get(), tile_col0.get(), ne02);
        mmid16_q2_0_gemm<2><<<grid, MMID16_THREADS, 0, stream>>>(
            (const char *) src0->data, xh.get(), xs.get(), ids_dst.get(), bounds.get(), tile_expert.get(), tile_col0.get(),
            (float *) dst->data, nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float));
    } else {
        mmid16_tiles<64><<<1, nthr, 0, stream>>>(bounds.get(), tile_expert.get(), tile_col0.get(), ne02);
        mmid16_q2_0_gemm<4><<<grid, MMID16_THREADS, 0, stream>>>(
            (const char *) src0->data, xh.get(), xs.get(), ids_dst.get(), bounds.get(), tile_expert.get(), tile_col0.get(),
            (float *) dst->data, nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float));
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mmid_vec_f16_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_CUDA_SM60_MMID_VEC_F16") == nullptr || atoi(getenv("GGML_CUDA_SM60_MMID_VEC_F16")) != 0;
    if (!enabled || !ggml_cuda_mmid_f16_sm60_supported(src0, src1, ids, dst)) {
        return false;
    }
    return src0->ne[1] % MMID16V_ROWS == 0 && src1->nb[2] % 16 == 0;
}

void ggml_cuda_mmid_vec_f16_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int64_t ne00   = src0->ne[0];
    const int64_t ne01   = src0->ne[1];
    const int64_t ne11   = src1->ne[1];
    const int64_t ne12   = src1->ne[2];
    const int     n_used = ids->ne[0];
    const int     nkb    = ne00/64;
    const int     si1    = ids->nb[1]/sizeof(int32_t);
    const int64_t s11    = src1->nb[1]/sizeof(float);
    const int64_t s12    = src1->nb[2]/sizeof(float);
    const int64_t sd     = dst->nb[1]/sizeof(float);
    const int64_t nrows1  = ne11*ne12;
    const size_t  xh_size = nrows1*nkb*32*sizeof(half2);
    const size_t  total   = xh_size + nrows1*nkb*sizeof(float);
    const int64_t key     = (nrows1 << 20) ^ ((int64_t) nkb << 8) ^ (s11 << 32) ^ s12;

    const bool hit = ctx.mmid16_cache_mem != nullptr && ctx.mmid16_cache_src1 == src1 &&
                     ctx.mmid16_cache_data == src1->data && ctx.mmid16_cache_key == key;
    if (!hit) {
        if (total > ctx.mmid16_cache_cap) {
            ctx.mmid16_cache_free();
            ggml_cuda_set_device(ctx.device);
            CUDA_CHECK(cudaMalloc((void **) &ctx.mmid16_cache_mem, total));
            ctx.mmid16_cache_cap = total;
        }
        mmid16_convert_rows<<<nrows1, nkb <= 32 ? 32 : 64, 0, stream>>>(
            (const float *) src1->data, (half2 *) ctx.mmid16_cache_mem, (float *) (ctx.mmid16_cache_mem + xh_size),
            s11, s12, ne11, nkb);
        ctx.mmid16_cache_src1 = src1;
        ctx.mmid16_cache_data = src1->data;
        ctx.mmid16_cache_key  = key;
    }
    const half2 * xh = (const half2 *) ctx.mmid16_cache_mem;
    const float * xs = (const float *) (ctx.mmid16_cache_mem + xh_size);

    const dim3 grid(ne01/MMID16V_ROWS, ne12*n_used, 1);
    mmid16_q2_0_vec<<<grid, dim3(WARP_SIZE, 4, 1), 0, stream>>>(
        (const char *) src0->data, xh, xs, (const int32_t *) ids->data, (float *) dst->data, nkb,
        src0->nb[1], src0->nb[2], ne11, n_used, si1, sd);
    CUDA_CHECK(cudaGetLastError());
}
