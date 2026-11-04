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
#include "moe-host.cuh"
#include "mmvq.cuh"
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

// one compact (expert-sorted) slot per block, one 64-value block per thread (GGML_CUDA_MMID16_CONV=1: strided
// 256-byte reads and 128-byte writes per thread, ~95 GB/s)
static __global__ void mmid16_convert_act_v1(
        const float * __restrict__ src1, const int32_t * __restrict__ ids_src1, half2 * __restrict__ xh,
        float * __restrict__ xs, const int64_t stride_row, const int nkb) {
    const int     c = blockIdx.x;
    const float * x = src1 + (int64_t) ids_src1[c]*stride_row;
    for (int kb = threadIdx.x; kb < nkb; kb += blockDim.x) {
        mmid16_convert_block(x + kb*64, xh + ((int64_t) c*nkb + kb)*32, xs + (int64_t) c*nkb + kb);
    }
}

static __device__ __forceinline__ uint32_t mmid16_u32(const half2 v) {
    return *((const uint32_t *) &v);
}

// one compact slot per block, one 64-value block per warp: the lanes read float2 (256 B coalesced), reduce the amax
// with shuffles, and the lanes with lane%8 < 4 write a uint2 of two half2 pairs (own elements 2l, 2l+1 as the low
// halves, lane l+4's as the high halves: pair 8g + j = elements (16g + j, 16g + j + 8)), 128 B coalesced. 4.9x the
// bandwidth of v1 (0.59 vs 2.9 ms for 15360 slots x 2560), bit-identical output.
static __global__ void mmid16_convert_act(
        const float * __restrict__ src1, const int32_t * __restrict__ ids_src1, half2 * __restrict__ xh,
        float * __restrict__ xs, const int64_t stride_row, const int nkb) {
    const int     c    = blockIdx.x;
    const float * x    = src1 + (int64_t) ids_src1[c]*stride_row;
    const int     lane = threadIdx.x % WARP_SIZE;
    const int     wid  = threadIdx.x / WARP_SIZE;
    const int     nw   = blockDim.x / WARP_SIZE;
    for (int kb = wid; kb < nkb; kb += nw) {
        const float2 v = ((const float2 *) (x + kb*64))[lane];
        float amax = fmaxf(fabsf(v.x), fabsf(v.y));
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        }
        const float    s   = amax > 0.0f ? amax : 1.0f;
        const float    id  = 1.0f/s;
        const uint32_t own = mmid16_u32(make_half2(__float2half(v.x*id), __float2half(v.y*id)));
        const uint32_t hi  = __shfl_down_sync(0xffffffff, own, 4);
        if ((lane & 7) < 4) {
            const int g = lane / 8, j = 2*(lane % 8);
            uint2 o;
            o.x = __byte_perm(own, hi, 0x5410);
            o.y = __byte_perm(own, hi, 0x7632);
            ((uint2 *) (xh + ((int64_t) c*nkb + kb)*32))[(8*g + j)/2] = o;
        }
        if (lane == 0) {
            xs[(int64_t) c*nkb + kb] = s;
        }
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
// MODE 0: ceil(count/TN) tiles per expert; MODE 1: only the full groups of TN slots (count/TN tiles); MODE 2: the
// remainder after full groups of 2*TN slots, in tiles of TN (the mixed launch: wide tiles on the full groups, narrow
// tiles on the remainder; the kernel bounds every tile by the expert's last slot)
template <int TN, int MODE = 0>
static __global__ void mmid16_tiles(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_expert,
        int32_t * __restrict__ tile_col0, const int n_experts) {
    __shared__ int scan[1024];
    const int e   = threadIdx.x;
    const int cnt = e < n_experts ? expert_bounds[e + 1] - expert_bounds[e] : 0;
    int nt  = (cnt + TN - 1)/TN;
    int off = 0;
    if (MODE == 1) {
        nt = cnt/TN;
    } else if (MODE == 2) {
        off = (cnt/(2*TN))*(2*TN);
        nt  = (cnt - off + TN - 1)/TN;
    }
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
        tile_col0[start + i]   = expert_bounds[e] + off + i*TN;
    }
}

static __device__ __forceinline__ half2 mmid16_h2(const uint32_t v) {
    return *((const half2 *) &v);
}

// Q4_0 (the MTP layer's experts, resident): 32-value blocks of 18 B (half d + 16 nibble bytes: element j = low
// nibble of byte j, j + 16 = high nibble), two blocks per 64-value k-block. Word w (0..7) of a k-block = block w/4,
// bytes 4(w%4)..+3, decoded to 4 half2 pairs t0 = (e, e+2) low nibbles of bytes 0/2, t1 = (e+16, e+18) high nibbles
// of bytes 0/2, t2 = (e+1, e+3), t3 = (e+17, e+19) with e = 32(w/4) + 4(w%4); activation pair 4w + t holds the same
// elements (mmid16_convert_act_q4). Bench: runs/fn-m2/qpf/bench/mmid_bench.cu (q4 argument).
static __device__ __forceinline__ void mmid16_decode_q4_0(const uint32_t q, half2 * h) {
    const uint32_t q4  = q >> 4;
    const uint32_t q8  = q >> 8;
    const uint32_t q12 = q >> 12;
    uint32_t t[4];
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[0]) : "r"(q),   "n"(0x000f000f), "n"(0x64006400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[1]) : "r"(q4),  "n"(0x000f000f), "n"(0x64006400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[2]) : "r"(q8),  "n"(0x000f000f), "n"(0x64006400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[3]) : "r"(q12), "n"(0x000f000f), "n"(0x64006400));
    const half2 m = __float2half2_rn(1032.0f); // 1024 + 8: code n -> n - 8
    h[0] = __hsub2(mmid16_h2(t[0]), m);
    h[1] = __hsub2(mmid16_h2(t[1]), m);
    h[2] = __hsub2(mmid16_h2(t[2]), m);
    h[3] = __hsub2(mmid16_h2(t[3]), m);
}

// one compact slot per block, one 64-value k-block per warp; even lane l pairs with lane l+1: A = (2l, 2l+2) and
// B = (2l+1, 2l+3); with 2l = 32b + 16h + 4m: A is pair t = h of word 4b + m, B is pair t = 2 + h
static __global__ void mmid16_convert_act_q4(
        const float * __restrict__ src1, const int32_t * __restrict__ ids_src1, half2 * __restrict__ xh,
        float * __restrict__ xs, const int64_t stride_row, const int nkb) {
    const int     c    = blockIdx.x;
    const float * x    = src1 + (int64_t) ids_src1[c]*stride_row;
    const int     lane = threadIdx.x % 32;
    const int     wid  = threadIdx.x / 32;
    const int     nw   = blockDim.x / 32;
    for (int kb = wid; kb < nkb; kb += nw) {
        const float2 v = ((const float2 *) (x + kb*64))[lane];
        float amax = fmaxf(fabsf(v.x), fabsf(v.y));
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        }
        const float    s   = amax > 0.0f ? amax : 1.0f;
        const float    id  = 1.0f/s;
        const uint32_t own = mmid16_u32(make_half2(__float2half(v.x*id), __float2half(v.y*id)));
        const uint32_t nbr = __shfl_down_sync(0xffffffff, own, 1);
        if ((lane & 1) == 0) {
            const int b = lane / 16, h = (lane / 8) & 1, m = (lane / 2) & 3;
            const int w = 4*b + m;
            uint32_t * o = (uint32_t *) (xh + ((int64_t) c*nkb + kb)*32);
            o[4*w + h]     = __byte_perm(own, nbr, 0x5410); // (own.x, nbr.x)
            o[4*w + 2 + h] = __byte_perm(own, nbr, 0x7632); // (own.y, nbr.y)
        }
        if (lane == 0) {
            xs[(int64_t) c*nkb + kb] = s;
        }
    }
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

// NT threads as 16 row groups of 4 rows x NT/16 slot groups of NJ slots: a tile is 64 weight rows x (NT/16)*NJ slots.
// (256, 2) = 32 slots, (256, 4) = 64, (128, 4) = 32 slots with 4 x 4 outputs per thread: 8 LDS.128 per 64 HFMA2 instead
// of 6 per 32 (the kernel is shared-memory bound)
template <int NJ, int NT = MMID16_THREADS>
static __global__ void __launch_bounds__(NT, 512/NT)
mmid16_q2_0_gemm(
        const char * __restrict__ w, const half2 * __restrict__ xh, const float * __restrict__ xs,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_expert, const int32_t * __restrict__ tile_col0,
        float * __restrict__ dst, const int nkb, const int64_t nb01, const int64_t nb02, const int64_t stride_dst,
        const char * __restrict__ w_cold, const int n_hot, const int e_lo, const int e_hi, const int cold_base) {
#if defined(FP16_AVAILABLE)
    constexpr int TX = NT/16;  // slot groups
    constexpr int TN = TX*NJ;  // slots per tile

    const int e = tile_expert[blockIdx.y];
    if (e < 0 || e < e_lo || e >= e_hi || (e >= n_hot && w_cold == nullptr)) {
        return; // no tile, an expert outside this launch's range, or a cold expert the host computes (moe-host.cuh)
    }
    const int c0   = tile_col0[blockIdx.y];
    const int cend = expert_bounds[e + 1];
    const int row0 = blockIdx.x*MMID16_TM;

    __shared__ __align__(16) uint32_t ws [MMID16_TM][MMID16_LDS];
    __shared__ __align__(16) uint32_t xsm[TN][MMID16_LDS];
    __shared__ float wd [MMID16_TM];
    __shared__ float xsc[TN];

    const int tid = threadIdx.x;
    const int tx  = tid % TX;
    const int ty  = tid / TX;

    const char * wexp = e < n_hot ? w + (int64_t) e*nb02 : w_cold + (int64_t) (e - cold_base)*nb02;

    float acc[4][NJ];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int kb = 0; kb < nkb; ++kb) {
#pragma unroll
        for (int w0 = 0; w0 < MMID16_TM*4; w0 += NT) {
            const int lr = (w0 + tid) / 4; // weight row loaded by this thread
            const int lq = (w0 + tid) % 4; // 16-value group loaded by this thread
            // block_q2_0 is 18 bytes, so the codes are only 2-byte aligned
            const char *     blk = wexp + (int64_t) (row0 + lr)*nb01 + (int64_t) kb*18;
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
        for (int i0 = 0; i0 < TN*8; i0 += NT) {
            const int i    = i0 + tid;
            const int col  = i / 8;
            const int part = i % 8;
            const int c    = c0 + col;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (i < TN*8 && c < cend) {
                v = ((const uint4 *) (xh + ((int64_t) c*nkb + kb)*32))[part];
            }
            if (i < TN*8) {
                *((uint4 *) &xsm[col][4*part]) = v;
            }
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
                xv[j] = *((const uint4 *) &xsm[tx + TX*j][4*p4]);
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
                acc[i][j] += dw*xsc[tx + TX*j]*__half2float(__hadd(__low2half(hacc[i][j]), __high2half(hacc[i][j])));
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int c = c0 + tx + TX*j;
        if (c < cend) {
            float * d = dst + (int64_t) ids_dst[c]*stride_dst + row0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d[ty + 16*i] = acc[i][j];
            }
        }
    }
#else
    GGML_UNUSED_VARS(w, xh, xs, ids_dst, expert_bounds, tile_expert, tile_col0, dst, nkb, nb01, nb02, stride_dst, w_cold, n_hot, e_lo, e_hi, cold_base);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// The pipelined narrow tile (the default since qp9, GGML_CUDA_MMID16_TILE=16): 64 rows x TN = (NT/(64/RPT))*NJ slots,
// RPT x NJ outputs per thread, the next k-block's global loads issued into registers before the compute of the
// current one (one shared buffer, two barriers per k-block). <64, 4, 4> = 16-slot tiles of 2 warps: at 20-30 slots
// per expert the 16-slot tiles fill 0.81 against 0.65 for 32-slot ones, and the hoisted loads hide the strided
// 2-byte weight loads: 3.14 vs 4.12 ms per gate op per card at 1536 tokens, 2.34 vs 3.06 at 1024 (bench:
// runs/fn-m2/qpf/bench/mmid_bench.cu; taller tiles and double-buffered shared memory measured slower there).
template <int NT, int RPT, int NJ>
struct mmid16_tile_cfg {
    static constexpr int TM   = MMID16_TM;
    static constexpr int RG   = TM/RPT;                 // row groups
    static constexpr int TX   = NT/RG;                  // slot groups
    static constexpr int TN   = TX*NJ;                  // slots per tile
    static constexpr int WI   = TM*4/NT;                // weight words per thread per k-block
    static constexpr int XI   = (TN*8 + NT - 1)/NT;     // activation uint4 per thread per k-block
    static constexpr int SMEM = TM*MMID16_LDS*4 + TN*MMID16_LDS*4 + TM*4 + TN*4;
    static constexpr int MINB = (65536/SMEM) < (512/NT) ? (65536/SMEM) : (512/NT);
    static_assert(TM*4 % NT == 0, "weight words per thread");
    static_assert(NT % RG == 0, "thread mapping");
};

template <int NT, int RPT, int NJ>
static __global__ void __launch_bounds__(NT, mmid16_tile_cfg<NT, RPT, NJ>::MINB)
mmid16_q2_0_gemm_p(
        const char * __restrict__ w, const half2 * __restrict__ xh, const float * __restrict__ xs,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_expert, const int32_t * __restrict__ tile_col0,
        float * __restrict__ dst, const int nkb, const int64_t nb01, const int64_t nb02, const int64_t stride_dst,
        const char * __restrict__ w_cold, const int n_hot, const int e_lo, const int e_hi, const int cold_base) {
#if defined(FP16_AVAILABLE)
    using C = mmid16_tile_cfg<NT, RPT, NJ>;
    constexpr int TM = C::TM, RG = C::RG, TX = C::TX, TN = C::TN, WI = C::WI, XI = C::XI;

    const int e = tile_expert[blockIdx.y];
    if (e < 0 || e < e_lo || e >= e_hi || (e >= n_hot && w_cold == nullptr)) {
        return; // no tile, an expert outside this launch's range, or a cold expert the host computes (moe-host.cuh)
    }
    const int c0   = tile_col0[blockIdx.y];
    const int cend = expert_bounds[e + 1];
    const int row0 = blockIdx.x*TM;

    __shared__ __align__(16) uint32_t ws [TM][MMID16_LDS];
    __shared__ __align__(16) uint32_t xsm[TN][MMID16_LDS];
    __shared__ float wd [TM];
    __shared__ float xsc[TN];

    const int tid = threadIdx.x;
    const int tx  = tid % TX;
    const int ty  = tid / TX;
    const char * wexp = e < n_hot ? w + (int64_t) e*nb02 : w_cold + (int64_t) (e - cold_base)*nb02;

    float acc[RPT][NJ];
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    uint32_t rq[WI];
    float    rd[WI];
    uint4    rx[XI];
    float    rxs = 0.0f;

    // Running pointers per thread, advanced by one k-block per load (weights 18 B, activations 128 B, scales 4 B),
    // with the slot predicates hoisted: the index form (expert row * nb01 + kb*18 etc. in every load) cost ~90
    // integer instructions per k-block next to the 512 HFMA2 (SASS count); this form 23 fewer than that, -4.5% on
    // the gate op and -3.9% on down at 1536 tokens (bench runs/fn-m2/qp7/ex/bench/mmid_bench.cu, variant P).
    // wq[i] points at the k-block's 18-byte block + 4*lq: the codes are wq[i][1..2], the scale at wq[i] - 2*lq.
    const uint16_t * wq[WI];
    const uint4 *    xp[XI];
    bool             xok[XI];
#pragma unroll
    for (int i = 0; i < WI; ++i) {
        const int lr = (i*NT + tid) / 4; const int lq = (i*NT + tid) % 4;
        wq[i] = (const uint16_t *) (wexp + (int64_t) (row0 + lr)*nb01 + 4*lq);
    }
#pragma unroll
    for (int i = 0; i < XI; ++i) {
        const int idx = i*NT + tid; const int col = idx / 8; const int part = idx % 8; const int c = c0 + col;
        xok[i] = idx < TN*8 && c < cend;
        xp[i]  = (const uint4 *) (xh + (int64_t) (xok[i] ? c : c0)*nkb*32) + part;
    }
    const bool    xsok = tid < TN && c0 + tid < cend;
    const float * xsp  = xs + (int64_t) (xsok ? c0 + tid : c0)*nkb;

    // the next k-block's global loads into registers
#define MMID16_LOAD() do { \
        _Pragma("unroll") for (int i = 0; i < WI; ++i) { \
            const int lq = (i*NT + tid) % 4; \
            rq[i] = (uint32_t) wq[i][1] | ((uint32_t) wq[i][2] << 16); \
            rd[i] = __half2float(*((const half *) (wq[i] - 2*lq))); \
            wq[i] += 9; \
        } \
        _Pragma("unroll") for (int i = 0; i < XI; ++i) { \
            rx[i] = xok[i] ? *xp[i] : make_uint4(0, 0, 0, 0); \
            xp[i] += 8; \
        } \
        rxs = xsok ? *xsp : 0.0f; xsp++; \
    } while (0)
    // the registers decoded into shared memory
#define MMID16_STORE() do { \
        _Pragma("unroll") for (int i = 0; i < WI; ++i) { \
            const int lr = (i*NT + tid) / 4; const int lq = (i*NT + tid) % 4; \
            half2 h[8]; mmid16_decode_q2_0(rq[i], h); \
            uint4 * wsp = (uint4 *) &ws[lr][8*lq]; \
            wsp[0] = make_uint4(mmid16_u32(h[0]), mmid16_u32(h[1]), mmid16_u32(h[2]), mmid16_u32(h[3])); \
            wsp[1] = make_uint4(mmid16_u32(h[4]), mmid16_u32(h[5]), mmid16_u32(h[6]), mmid16_u32(h[7])); \
            if (lq == 0) { wd[lr] = rd[i]; } \
        } \
        _Pragma("unroll") for (int i = 0; i < XI; ++i) { \
            const int idx = i*NT + tid; const int col = idx / 8; const int part = idx % 8; \
            if (idx < TN*8) { *((uint4 *) &xsm[col][4*part]) = rx[i]; } \
        } \
        if (tid < TN) { xsc[tid] = rxs; } \
    } while (0)

    MMID16_LOAD();
    for (int kb = 0; kb < nkb; ++kb) {
        MMID16_STORE();
        __syncthreads();
        if (kb + 1 < nkb) {
            MMID16_LOAD();
        }
        half2 hacc[RPT][NJ];
#pragma unroll
        for (int i = 0; i < RPT; ++i) {
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                hacc[i][j] = make_half2(0.0f, 0.0f);
            }
        }
#pragma unroll
        for (int p4 = 0; p4 < 8; ++p4) {
            uint4 wv[RPT];
            uint4 xv[NJ];
#pragma unroll
            for (int i = 0; i < RPT; ++i) {
                wv[i] = *((const uint4 *) &ws[ty + RG*i][4*p4]);
            }
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                xv[j] = *((const uint4 *) &xsm[tx + TX*j][4*p4]);
            }
#pragma unroll
            for (int i = 0; i < RPT; ++i) {
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
        for (int i = 0; i < RPT; ++i) {
            const float dw = wd[ty + RG*i];
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                acc[i][j] += dw*xsc[tx + TX*j]*__half2float(__hadd(__low2half(hacc[i][j]), __high2half(hacc[i][j])));
            }
        }
        __syncthreads();
    }
#undef MMID16_LOAD
#undef MMID16_STORE

#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int c = c0 + tx + TX*j;
        if (c < cend) {
            float * d = dst + (int64_t) ids_dst[c]*stride_dst + row0;
#pragma unroll
            for (int i = 0; i < RPT; ++i) {
                d[ty + RG*i] = acc[i][j];
            }
        }
    }
#else
    GGML_UNUSED_VARS(w, xh, xs, ids_dst, expert_bounds, tile_expert, tile_col0, dst, nkb, nb01, nb02, stride_dst, w_cold, n_hot, e_lo, e_hi, cold_base);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// The Q4_0 variant of the pipelined tile: 8 words and two scales per row per k-block, one half2 accumulator per
// 32-value block (GGML_CUDA_MMID16_Q4TILE selects the instantiation).
template <int NT, int RPT, int NJ, int MINB = mmid16_tile_cfg<NT, RPT, NJ>::MINB>
static __global__ void __launch_bounds__(NT, MINB)
mmid16_q4_0_gemm_p(
        const char * __restrict__ w, const half2 * __restrict__ xh, const float * __restrict__ xs,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_expert, const int32_t * __restrict__ tile_col0,
        float * __restrict__ dst, const int nkb, const int64_t nb01, const int64_t nb02, const int64_t stride_dst,
        const int e_lo, const int e_hi) {
#if defined(FP16_AVAILABLE)
    using C = mmid16_tile_cfg<NT, RPT, NJ>;
    constexpr int TM = C::TM, RG = C::RG, TX = C::TX, TN = C::TN, XI = C::XI;
    constexpr int WI = TM*8/NT; // 8 words per row per k-block
    static_assert(TM*8 % NT == 0, "weight words per thread");

    const int e = tile_expert[blockIdx.y];
    if (e < 0 || e < e_lo || e >= e_hi) {
        return;
    }
    const int c0   = tile_col0[blockIdx.y];
    const int cend = expert_bounds[e + 1];
    const int row0 = blockIdx.x*TM;

    __shared__ __align__(16) uint32_t ws [TM][MMID16_LDS];
    __shared__ __align__(16) uint32_t xsm[TN][MMID16_LDS];
    __shared__ float wd [2][TM];
    __shared__ float xsc[TN];

    const int tid = threadIdx.x;
    const int tx  = tid % TX;
    const int ty  = tid / TX;
    const char * wexp = w + (int64_t) e*nb02;

    float acc[RPT][NJ];
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    uint32_t rq[WI];
    float    rd[WI];
    uint4    rx[XI];
    float    rxs = 0.0f;

#define MMID16_LOADQ4(kb) do { \
        _Pragma("unroll") for (int i = 0; i < WI; ++i) { \
            const int lr = (i*NT + tid) / 8; const int lw = (i*NT + tid) % 8; \
            const char * blk = wexp + (int64_t) (row0 + lr)*nb01 + (int64_t) (kb)*36 + (lw/4)*18; \
            const uint16_t * q16 = (const uint16_t *) (blk + 2 + 4*(lw%4)); \
            rq[i] = (uint32_t) q16[0] | ((uint32_t) q16[1] << 16); \
            rd[i] = __half2float(*((const half *) blk)); \
        } \
        _Pragma("unroll") for (int i = 0; i < XI; ++i) { \
            const int idx = i*NT + tid; const int col = idx / 8; const int part = idx % 8; const int c = c0 + col; \
            rx[i] = make_uint4(0, 0, 0, 0); \
            if (idx < TN*8 && c < cend) { rx[i] = ((const uint4 *) (xh + ((int64_t) c*nkb + (kb))*32))[part]; } \
        } \
        if (tid < TN) { const int c = c0 + tid; rxs = c < cend ? xs[(int64_t) c*nkb + (kb)] : 0.0f; } \
    } while (0)
#define MMID16_STOREQ4() do { \
        _Pragma("unroll") for (int i = 0; i < WI; ++i) { \
            const int lr = (i*NT + tid) / 8; const int lw = (i*NT + tid) % 8; \
            half2 h[4]; mmid16_decode_q4_0(rq[i], h); \
            *((uint4 *) &ws[lr][4*lw]) = make_uint4(mmid16_u32(h[0]), mmid16_u32(h[1]), mmid16_u32(h[2]), mmid16_u32(h[3])); \
            if ((lw & 3) == 0) { wd[lw/4][lr] = rd[i]; } \
        } \
        _Pragma("unroll") for (int i = 0; i < XI; ++i) { \
            const int idx = i*NT + tid; const int col = idx / 8; const int part = idx % 8; \
            if (idx < TN*8) { *((uint4 *) &xsm[col][4*part]) = rx[i]; } \
        } \
        if (tid < TN) { xsc[tid] = rxs; } \
    } while (0)

    MMID16_LOADQ4(0);
    for (int kb = 0; kb < nkb; ++kb) {
        MMID16_STOREQ4();
        __syncthreads();
        if (kb + 1 < nkb) {
            MMID16_LOADQ4(kb + 1);
        }
        half2 hacc0[RPT][NJ];
        half2 hacc1[RPT][NJ];
#pragma unroll
        for (int i = 0; i < RPT; ++i) {
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                hacc0[i][j] = make_half2(0.0f, 0.0f);
                hacc1[i][j] = make_half2(0.0f, 0.0f);
            }
        }
#define MMID16_Q4_HALF(HACC, P0) do { \
        _Pragma("unroll") for (int p4 = P0; p4 < P0 + 4; ++p4) { \
            uint4 wv[RPT]; uint4 xv[NJ]; \
            _Pragma("unroll") for (int i = 0; i < RPT; ++i) { wv[i] = *((const uint4 *) &ws[ty + RG*i][4*p4]); } \
            _Pragma("unroll") for (int j = 0; j < NJ; ++j) { xv[j] = *((const uint4 *) &xsm[tx + TX*j][4*p4]); } \
            _Pragma("unroll") for (int i = 0; i < RPT; ++i) { \
                _Pragma("unroll") for (int j = 0; j < NJ; ++j) { \
                    HACC[i][j] = __hfma2(mmid16_h2(wv[i].x), mmid16_h2(xv[j].x), HACC[i][j]); \
                    HACC[i][j] = __hfma2(mmid16_h2(wv[i].y), mmid16_h2(xv[j].y), HACC[i][j]); \
                    HACC[i][j] = __hfma2(mmid16_h2(wv[i].z), mmid16_h2(xv[j].z), HACC[i][j]); \
                    HACC[i][j] = __hfma2(mmid16_h2(wv[i].w), mmid16_h2(xv[j].w), HACC[i][j]); \
                } \
            } \
        } \
    } while (0)
        MMID16_Q4_HALF(hacc0, 0);
        MMID16_Q4_HALF(hacc1, 4);
#undef MMID16_Q4_HALF
#pragma unroll
        for (int i = 0; i < RPT; ++i) {
            const float d0 = wd[0][ty + RG*i];
            const float d1 = wd[1][ty + RG*i];
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                acc[i][j] += xsc[tx + TX*j]*(d0*__half2float(__hadd(__low2half(hacc0[i][j]), __high2half(hacc0[i][j])))
                                           + d1*__half2float(__hadd(__low2half(hacc1[i][j]), __high2half(hacc1[i][j]))));
            }
        }
        __syncthreads();
    }
#undef MMID16_LOADQ4
#undef MMID16_STOREQ4

#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int c = c0 + tx + TX*j;
        if (c < cend) {
            float * d = dst + (int64_t) ids_dst[c]*stride_dst + row0;
#pragma unroll
            for (int i = 0; i < RPT; ++i) {
                d[ty + RG*i] = acc[i][j];
            }
        }
    }
#else
    GGML_UNUSED_VARS(w, xh, xs, ids_dst, expert_bounds, tile_expert, tile_col0, dst, nkb, nb01, nb02, stride_dst, e_lo, e_hi);
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
        const int ne11, const int n_used, const int si1, const int64_t stride_dst,
        const char * __restrict__ w_cold, const int n_hot, const int skip_cold) {
#if defined(FP16_AVAILABLE)
    const int c = blockIdx.y;
    const int t = c / n_used;
    const int s = c % n_used;
    const int e = ids[t*si1 + s];
    const int r = t*ne11 + s % ne11;
    if (skip_cold && e >= n_hot) {
        return;
    }

    const half2 * xr   = xh + (int64_t) r*nkb*32;
    const float * xsr  = xs + (int64_t) r*nkb;
    const int     lane = threadIdx.x;
    const int     row0 = blockIdx.x*MMID16V_ROWS + threadIdx.y*MMID16V_RW;
    const char *  wexp = e < n_hot ? w + (int64_t) e*nb02 : w_cold + (int64_t) c*nb02; // w_cold: per-slot staging
    const char *  wr   = wexp + (int64_t) row0*nb01;

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
    GGML_UNUSED_VARS(w, xh, xs, ids, dst, nkb, nb01, nb02, ne11, n_used, si1, stride_dst, w_cold, n_hot, skip_cold);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// Cold experts live in pinned host memory. Reading them in place from the mat-vec/GEMM kernels uses 2-byte loads
// spread over 18-byte blocks, which wastes most of each PCIe transaction (~2-3 GB/s measured), so the experts an op
// needs are first copied to VRAM with 16-byte loads. Per token slot for the mat-vec, per used cold expert for the GEMM.
// PCIe reads of pinned memory need many requests in flight: 4 independent 16-byte loads per thread before the stores
static __device__ __forceinline__ void mmid16_copy_pcie(const uint4 * __restrict__ src, uint4 * __restrict__ dst, const int64_t n16) {
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    for (int64_t i0 = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i0 < n16; i0 += 4*stride) {
        uint4 v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int64_t i = i0 + k*stride;
            if (i < n16) {
                v[k] = src[i];
            }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int64_t i = i0 + k*stride;
            if (i < n16) {
                dst[i] = v[k];
            }
        }
    }
}

static __global__ void mmid16_stage_cold_slots(
        const int32_t * __restrict__ ids, const int si1, const int n_used, const int n_hot,
        const uint4 * __restrict__ w_cold, const int64_t n16, uint4 * __restrict__ stage) {
    const int c = blockIdx.y;
    const int e = ids[(c / n_used)*si1 + c % n_used];
    if (e < n_hot) {
        return;
    }
    const uint4 * src = w_cold + (int64_t) (e - n_hot)*n16;
    uint4 *       dst = stage  + (int64_t) c*n16;
    mmid16_copy_pcie(src, dst, n16);
}

static __global__ void mmid16_stage_cold_experts(
        const int32_t * __restrict__ expert_bounds, const int n_hot,
        const uint4 * __restrict__ w_cold, const int64_t n16, uint4 * __restrict__ stage) {
    const int ec = blockIdx.y;
    if (expert_bounds[n_hot + ec + 1] == expert_bounds[n_hot + ec]) {
        return;
    }
    const uint4 * src = w_cold + (int64_t) ec*n16;
    uint4 *       dst = stage  + (int64_t) ec*n16;
    mmid16_copy_pcie(src, dst, n16);
}

bool ggml_cuda_mmid_f16_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_CUDA_SM60_MMID_F16") == nullptr || atoi(getenv("GGML_CUDA_SM60_MMID_F16")) != 0;
    if (!enabled || ids == nullptr) {
        return false;
    }
    // GGML_CUDA_SM60_MMID_Q4=0: the Q4_0 experts (the MTP layer's, all resident) stay on MMQ
    static const bool q4 = getenv("GGML_CUDA_SM60_MMID_Q4") == nullptr || atoi(getenv("GGML_CUDA_SM60_MMID_Q4")) != 0;
    if ((src0->type != GGML_TYPE_Q2_0 && !(q4 && src0->type == GGML_TYPE_Q4_0)) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_PASCAL || cc >= GGML_CUDA_CC_DP4A) {
        return false;
    }
    int n_cold = 0;
    ggml_cuda_mmid_cold(dst, nullptr, &n_cold);
    if (src0->type == GGML_TYPE_Q4_0 && n_cold > 0) {
        return false; // the cold staging and the mat-vec are Q2_0 only
    }
    if (src0->ne[0] % 64 != 0 || src0->ne[1] % MMID16_TM != 0 || src0->ne[2] + n_cold > 1024 || src0->ne[3] != 1) {
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

// The per-op setup of the GEMM (sorted ids, converted activations, the tile table), then one or more launches over
// expert ranges: the whole op at once, or a range per staged piece of the cold experts (the prefetch below).
struct mmid16_gemm_plan {
    // declared in allocation order: the pool wants its allocations released in reverse order (members are destroyed last first)
    ggml_cuda_pool_alloc<int32_t> ids_src1, ids_dst, bounds;
    ggml_cuda_pool_alloc<char>    stage;
    ggml_cuda_pool_alloc<half2>   xh;
    ggml_cuda_pool_alloc<float>   xs;
    ggml_cuda_pool_alloc<int32_t> tile_expert, tile_col0;
    ggml_cuda_pool_alloc<int32_t> tile_expert2, tile_col02; // the mixed launch's remainder tiles (narrow)
    const ggml_tensor * src0 = nullptr;
    ggml_tensor *       dst  = nullptr;
    const char * w_cold = nullptr;
    int     n_hot  = 0;
    int     n_cold = 0;
    int64_t ne02   = 0;
    int     nkb    = 0;
    bool    narrow = false;
    bool    mixed  = false; // wide tiles on the full 64-slot groups of every expert, narrow tiles on the remainders
    bool    narrow128 = false; // the narrow tiles with 128 threads
    bool    tile16 = false; // the pipelined 16-slot tiles (mmid16_q2_0_gemm_p<64, 4, 4>)
    int     q4     = 0;     // Q4_0 experts: 1 + the GGML_CUDA_MMID16_Q4TILE instantiation (0 = Q2_0)
    const half2 * xh_ptr = nullptr; // the converted activations: the pool allocation above or the context's cache
    const float * xs_ptr = nullptr;
    dim3    grid;
    dim3    grid2;

    mmid16_gemm_plan(ggml_backend_cuda_context & ctx) :
        ids_src1(ctx.pool()), ids_dst(ctx.pool()), bounds(ctx.pool()), stage(ctx.pool()),
        xh(ctx.pool()), xs(ctx.pool()), tile_expert(ctx.pool()), tile_col0(ctx.pool()),
        tile_expert2(ctx.pool()), tile_col02(ctx.pool()) {}
};

static void mmid16_gemm_setup(
        ggml_backend_cuda_context & ctx, mmid16_gemm_plan & p, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const bool skip_cold, const char * cold_vram) {
    cudaStream_t stream = ctx.stream();

    const int64_t ne00     = src0->ne[0];
    const int64_t ne01     = src0->ne[1];
    const char *  w_cold   = nullptr;
    int           n_cold   = 0;
    ggml_cuda_mmid_cold(dst, &w_cold, &n_cold);
    const int     n_hot    = src0->ne[2];
    const int64_t ne02     = src0->ne[2] + n_cold; // experts addressed by ids
    const int64_t ne11     = src1->ne[1];
    const int64_t ne12     = src1->ne[2];
    const int     n_used   = ids->ne[0];
    const int64_t n_assign = ne12*n_used;
    const int     nkb      = ne00/64;

    p.src0 = src0; p.dst = dst; p.n_hot = n_hot; p.n_cold = n_cold; p.ne02 = ne02; p.nkb = nkb;
    // GGML_CUDA_MMID16_Q4TILE: 0 = 64x16 tiles of 2 warps (4x4 per thread, 4 blocks per SM), 1 = 64x16 of 4 warps
    // (4x2), 2 = 64x32 of 4 warps (4x4, 3 blocks per SM)
    static const int q4tile = getenv("GGML_CUDA_MMID16_Q4TILE") != nullptr ? atoi(getenv("GGML_CUDA_MMID16_Q4TILE")) : 0;
    p.q4 = src0->type == GGML_TYPE_Q4_0 ? 1 + q4tile : 0;
    GGML_ASSERT(p.q4 == 0 || n_cold == 0);

    p.ids_src1.alloc(n_assign);
    p.ids_dst.alloc(n_assign);
    p.bounds.alloc(ne02 + 1);

    const int si1  = ids->nb[1]/sizeof(int32_t);
    const int sis1 = src1->nb[2]/src1->nb[1];
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, p.ids_src1.get(), p.ids_dst.get(), p.bounds.get(),
        ne02, ne12, n_used, ne11, si1, sis1, /*write_inverse =*/ false, stream);

    if (n_cold > 0 && skip_cold) {
        w_cold = nullptr; // the GEMM skips the cold experts' tiles
    } else if (n_cold > 0 && cold_vram != nullptr) {
        w_cold = cold_vram; // staged by the caller (the prefetch below hands a buffer per launch: MMID16_COLD_BY_LAUNCH)
    } else if (n_cold > 0) {
        GGML_ASSERT(src0->nb[2] % 16 == 0);
        const int64_t n16 = src0->nb[2]/16;
        p.stage.alloc((size_t) n_cold*src0->nb[2]);
        mmid16_stage_cold_experts<<<dim3(16, n_cold, 1), 256, 0, stream>>>(
            p.bounds.get(), n_hot, (const uint4 *) w_cold, n16, (uint4 *) p.stage.get());
        w_cold = p.stage.get();
    }
    p.w_cold = w_cold;

    // The converted activations: per op from the pool. GGML_CUDA_MMID16_SHARE=1 keeps them in the context instead
    // (cleared at every graph start, like the mat-vec's) so that up hits gate's conversion: off by default since
    // qp12, the buffer is a cudaMalloc outside the pool (81 MiB at 1536 tokens, +27 MiB for the MTP experts) for
    // ~0.6 ms per layer, and the gate run at -ub 1536 ran out of VRAM with it (qp8 itself ends a 200K prompt +
    // restore at 16.2 of 16.4 GB). GGML_CUDA_MMID16_CONV=1 takes the old strided conversion kernel.
    static const bool share    = getenv("GGML_CUDA_MMID16_SHARE") != nullptr && atoi(getenv("GGML_CUDA_MMID16_SHARE")) != 0;
    static const int  conv_env = getenv("GGML_CUDA_MMID16_CONV") != nullptr ? atoi(getenv("GGML_CUDA_MMID16_CONV")) : 2;
    const auto convert = [&](half2 * xh, float * xs) {
        if (p.q4) {
            mmid16_convert_act_q4<<<n_assign, 128, 0, stream>>>(
                (const float *) src1->data, p.ids_src1.get(), xh, xs, src1->nb[1]/sizeof(float), nkb);
        } else if (conv_env == 1) {
            mmid16_convert_act_v1<<<n_assign, nkb <= 32 ? 32 : 64, 0, stream>>>(
                (const float *) src1->data, p.ids_src1.get(), xh, xs, src1->nb[1]/sizeof(float), nkb);
        } else {
            mmid16_convert_act<<<n_assign, 128, 0, stream>>>(
                (const float *) src1->data, p.ids_src1.get(), xh, xs, src1->nb[1]/sizeof(float), nkb);
        }
    };
    if (share) {
        const size_t  xh_size = (size_t) n_assign*nkb*32*sizeof(half2);
        const size_t  total   = xh_size + (size_t) n_assign*nkb*sizeof(float);
        const int64_t key     = (n_assign << 20) ^ ((int64_t) nkb << 8) ^ ((int64_t) src1->nb[1] << 32) ^ (int64_t) ids->nb[1] ^ ((int64_t) p.q4 << 60);
        const bool hit = ctx.mmid16_pcache_mem != nullptr && ctx.mmid16_pcache_src1 == src1 && ctx.mmid16_pcache_data == src1->data &&
                         ctx.mmid16_pcache_ids == ids->data && ctx.mmid16_pcache_key == key;
        if (!hit) {
            if (total > ctx.mmid16_pcache_cap) {
                ctx.retire_mem(ctx.mmid16_pcache_mem, ctx.mmid16_pcache_cap);
                ctx.mmid16_pcache_src1 = nullptr;
                ggml_cuda_set_device(ctx.device);
                CUDA_CHECK(cudaMalloc((void **) &ctx.mmid16_pcache_mem, total));
                ctx.mmid16_pcache_cap = total;
            }
            convert((half2 *) ctx.mmid16_pcache_mem, (float *) (ctx.mmid16_pcache_mem + xh_size));
            ctx.mmid16_pcache_src1 = src1;
            ctx.mmid16_pcache_data = src1->data;
            ctx.mmid16_pcache_ids  = ids->data;
            ctx.mmid16_pcache_key  = key;
        }
        p.xh_ptr = (const half2 *) ctx.mmid16_pcache_mem;
        p.xs_ptr = (const float *) (ctx.mmid16_pcache_mem + xh_size);
    } else {
        p.xh.alloc(n_assign*nkb*32);
        p.xs.alloc(n_assign*nkb);
        convert(p.xh.get(), p.xs.get());
        p.xh_ptr = p.xh.get();
        p.xs_ptr = p.xs.get();
    }

    // GGML_CUDA_MMID16_MIXED=1: every expert's full groups of 64 slots take wide tiles (NJ=4: more HFMA2 per shared
    // load), the remainder narrow ones; else one tile width for the op: narrow when experts see few slots each
    static const bool mixed = getenv("GGML_CUDA_MMID16_MIXED") != nullptr && atoi(getenv("GGML_CUDA_MMID16_MIXED")) != 0;
    // GGML_CUDA_MMID16_NARROW128=1: the narrow (32-slot) tiles with 128 threads of 4 x 4 outputs
    static const bool narrow128 = getenv("GGML_CUDA_MMID16_NARROW128") != nullptr && atoi(getenv("GGML_CUDA_MMID16_NARROW128")) != 0;
    // GGML_CUDA_MMID16_TILE=16 (default): the pipelined 16-slot tiles instead of the 32-slot ones in the narrow case
    static const int tile_env = getenv("GGML_CUDA_MMID16_TILE") != nullptr ? atoi(getenv("GGML_CUDA_MMID16_TILE")) : 16;
    p.mixed     = mixed;
    p.narrow128 = narrow128;
    p.tile16    = !mixed && tile_env == 16 && n_assign < 48*ne02;
    p.narrow    = !mixed && !p.tile16 && n_assign < 48*ne02;
    if (p.q4) {
        p.mixed = false; p.narrow = p.q4 == 3; p.tile16 = p.q4 != 3; p.narrow128 = false;
    }
    const int     tn     = p.tile16 ? 16 : p.narrow ? 32 : 64;
    const int64_t t_max  = (n_assign + tn - 1)/tn + ne02;
    GGML_ASSERT(t_max <= 65535);

    p.tile_expert.alloc(t_max);
    p.tile_col0.alloc(t_max);
    ggml_cuda_fill_async(p.tile_expert.get(), 0xFF, t_max*sizeof(int32_t), stream); // a kernel, not a memset node (PP1 finding)

    int nthr = 32;
    while (nthr < ne02) {
        nthr *= 2;
    }
    p.grid = dim3(ne01/MMID16_TM, t_max, 1);
    if (mixed) {
        // wide tiles: at most n_assign/64 of them; narrow remainder tiles: at most 2 per expert
        const int64_t t_max2 = 2*ne02;
        p.tile_expert2.alloc(t_max2);
        p.tile_col02.alloc(t_max2);
        ggml_cuda_fill_async(p.tile_expert2.get(), 0xFF, t_max2*sizeof(int32_t), stream);
        p.grid2 = dim3(ne01/MMID16_TM, t_max2, 1);
        mmid16_tiles<64, 1><<<1, nthr, 0, stream>>>(p.bounds.get(), p.tile_expert.get(),  p.tile_col0.get(),  ne02);
        mmid16_tiles<32, 2><<<1, nthr, 0, stream>>>(p.bounds.get(), p.tile_expert2.get(), p.tile_col02.get(), ne02);
    } else if (p.tile16) {
        mmid16_tiles<16><<<1, nthr, 0, stream>>>(p.bounds.get(), p.tile_expert.get(), p.tile_col0.get(), ne02);
    } else if (p.narrow) {
        mmid16_tiles<32><<<1, nthr, 0, stream>>>(p.bounds.get(), p.tile_expert.get(), p.tile_col0.get(), ne02);
    } else {
        mmid16_tiles<64><<<1, nthr, 0, stream>>>(p.bounds.get(), p.tile_expert.get(), p.tile_col0.get(), ne02);
    }
    CUDA_CHECK(cudaGetLastError());
}

// a non-null cold_vram for mmid16_gemm_setup that means "the launches bring their own staged cold experts"
#define MMID16_COLD_BY_LAUNCH ((const char *) 1)

// the tiles of experts [e_lo, e_hi); cold expert e is read at w_cold + (e - cold_base)*nb02
static void mmid16_gemm_launch(
        ggml_backend_cuda_context & ctx, mmid16_gemm_plan & p, const int e_lo, const int e_hi,
        const char * w_cold, const int cold_base) {
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * src0 = p.src0;
    ggml_tensor *       dst  = p.dst;
    if (p.q4) {
        switch (p.q4) {
            case 2:
                mmid16_q4_0_gemm_p<128, 4, 2><<<p.grid, 128, 0, stream>>>(
                    (const char *) src0->data, p.xh_ptr, p.xs_ptr, p.ids_dst.get(), p.bounds.get(), p.tile_expert.get(), p.tile_col0.get(),
                    (float *) dst->data, p.nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float), e_lo, e_hi);
                break;
            case 3:
                mmid16_q4_0_gemm_p<128, 4, 4, 3><<<p.grid, 128, 0, stream>>>(
                    (const char *) src0->data, p.xh_ptr, p.xs_ptr, p.ids_dst.get(), p.bounds.get(), p.tile_expert.get(), p.tile_col0.get(),
                    (float *) dst->data, p.nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float), e_lo, e_hi);
                break;
            default:
                mmid16_q4_0_gemm_p<64, 4, 4, 4><<<p.grid, 64, 0, stream>>>(
                    (const char *) src0->data, p.xh_ptr, p.xs_ptr, p.ids_dst.get(), p.bounds.get(), p.tile_expert.get(), p.tile_col0.get(),
                    (float *) dst->data, p.nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float), e_lo, e_hi);
                break;
        }
        GGML_UNUSED(w_cold); GGML_UNUSED(cold_base);
    } else if (p.mixed) {
        mmid16_q2_0_gemm<4><<<p.grid, MMID16_THREADS, 0, stream>>>(
            (const char *) src0->data, p.xh_ptr, p.xs_ptr, p.ids_dst.get(), p.bounds.get(), p.tile_expert.get(), p.tile_col0.get(),
            (float *) dst->data, p.nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float), w_cold, p.n_hot, e_lo, e_hi, cold_base);
        if (p.narrow128) {
            mmid16_q2_0_gemm<4, 128><<<p.grid2, 128, 0, stream>>>(
                (const char *) src0->data, p.xh_ptr, p.xs_ptr, p.ids_dst.get(), p.bounds.get(), p.tile_expert2.get(), p.tile_col02.get(),
                (float *) dst->data, p.nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float), w_cold, p.n_hot, e_lo, e_hi, cold_base);
        } else {
            mmid16_q2_0_gemm<2><<<p.grid2, MMID16_THREADS, 0, stream>>>(
                (const char *) src0->data, p.xh_ptr, p.xs_ptr, p.ids_dst.get(), p.bounds.get(), p.tile_expert2.get(), p.tile_col02.get(),
                (float *) dst->data, p.nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float), w_cold, p.n_hot, e_lo, e_hi, cold_base);
        }
    } else if (p.tile16) {
        mmid16_q2_0_gemm_p<64, 4, 4><<<p.grid, 64, 0, stream>>>(
            (const char *) src0->data, p.xh_ptr, p.xs_ptr, p.ids_dst.get(), p.bounds.get(), p.tile_expert.get(), p.tile_col0.get(),
            (float *) dst->data, p.nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float), w_cold, p.n_hot, e_lo, e_hi, cold_base);
    } else if (p.narrow && p.narrow128) {
        mmid16_q2_0_gemm<4, 128><<<p.grid, 128, 0, stream>>>(
            (const char *) src0->data, p.xh_ptr, p.xs_ptr, p.ids_dst.get(), p.bounds.get(), p.tile_expert.get(), p.tile_col0.get(),
            (float *) dst->data, p.nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float), w_cold, p.n_hot, e_lo, e_hi, cold_base);
    } else if (p.narrow) {
        mmid16_q2_0_gemm<2><<<p.grid, MMID16_THREADS, 0, stream>>>(
            (const char *) src0->data, p.xh_ptr, p.xs_ptr, p.ids_dst.get(), p.bounds.get(), p.tile_expert.get(), p.tile_col0.get(),
            (float *) dst->data, p.nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float), w_cold, p.n_hot, e_lo, e_hi, cold_base);
    } else {
        mmid16_q2_0_gemm<4><<<p.grid, MMID16_THREADS, 0, stream>>>(
            (const char *) src0->data, p.xh_ptr, p.xs_ptr, p.ids_dst.get(), p.bounds.get(), p.tile_expert.get(), p.tile_col0.get(),
            (float *) dst->data, p.nkb, src0->nb[1], src0->nb[2], dst->nb[1]/sizeof(float), w_cold, p.n_hot, e_lo, e_hi, cold_base);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mmid_f16_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const bool skip_cold, const char * cold_vram) {
    mmid16_gemm_plan p(ctx);
    mmid16_gemm_setup(ctx, p, src0, src1, ids, dst, skip_cold, cold_vram);
    mmid16_gemm_launch(ctx, p, 0, (int) p.ne02, p.w_cold, p.n_hot);
}

bool ggml_cuda_mmid_vec_f16_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_CUDA_SM60_MMID_VEC_F16") == nullptr || atoi(getenv("GGML_CUDA_SM60_MMID_VEC_F16")) != 0;
    if (!enabled || src0->type != GGML_TYPE_Q2_0 || !ggml_cuda_mmid_f16_sm60_supported(src0, src1, ids, dst)) {
        return false;
    }
    return src0->ne[1] % MMID16V_ROWS == 0 && src1->nb[2] % 16 == 0;
}

void ggml_cuda_mmid_vec_f16_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const bool skip_cold) {
    cudaStream_t stream = ctx.stream();

    const int64_t ne00   = src0->ne[0];
    const int64_t ne01   = src0->ne[1];
    const int64_t ne11   = src1->ne[1];
    const int64_t ne12   = src1->ne[2];
    const int     n_used = ids->ne[0];
    const int     nkb    = ne00/64;
    const int     si1    = ids->nb[1]/sizeof(int32_t);
    const char *  w_cold = nullptr;
    ggml_cuda_pool_alloc<char> stage(ctx.pool());
    if (ggml_cuda_mmid_cold(dst, &w_cold) && !skip_cold) {
        GGML_ASSERT(src0->nb[2] % 16 == 0);
        const int64_t n16    = src0->nb[2]/16;
        const int     nslots = ne12*n_used;
        stage.alloc((size_t) nslots*src0->nb[2]);
        mmid16_stage_cold_slots<<<dim3(64, nslots, 1), 256, 0, stream>>>(
            (const int32_t *) ids->data, si1, n_used, (int) src0->ne[2], (const uint4 *) w_cold, n16, (uint4 *) stage.get());
        w_cold = stage.get();
    }
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
            ctx.retire_mem(ctx.mmid16_cache_mem, ctx.mmid16_cache_cap);
            ctx.mmid16_cache_clear();
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
        src0->nb[1], src0->nb[2], ne11, n_used, si1, sd, w_cold, (int) src0->ne[2], (int) skip_cold);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------
// prefetch of the cold experts for prompt batches

bool ggml_cuda_mmid_cold_prompt(const ggml_tensor * node) {
    // Q2_K / Q3_K cold experts have their own staging (cold-kq.cuh), not through the sm_60 Q2_0 GEMM
    return node->op == GGML_OP_MUL_MAT_ID && ggml_cuda_mmid_cold(node) && node->ne[2] > MMVQ_MAX_BATCH_SIZE &&
           !ggml_cuda_moe_host_ktype(node->src[0]->type);
}

static bool ggml_cuda_cold_prefetch_enabled() {
    static const bool enabled = getenv("GGML_CUDA_COLD_PREFETCH") == nullptr || atoi(getenv("GGML_CUDA_COLD_PREFETCH")) != 0;
    return enabled;
}

// GGML_CUDA_COLD_PREFETCH_SLOTS=n: slots of the ring (n x the largest op's cold bytes: 63-69 MiB per card at 272 cold
// experts of a 2560 x 640 Q2_0 slice); 2 = one op ahead within a graph (the depth before the ring ran across graphs)
static int cold_n_slots() {
    static const int n = [] {
        const char * e = getenv("GGML_CUDA_COLD_PREFETCH_SLOTS");
        return std::max(2, e != nullptr ? atoi(e) : 3);
    }();
    return n;
}

// GGML_CUDA_COLD_PREFETCH_AHEAD=n: pieces of later graphs a graph with cold ops of its own issues (after its GEMMs free
// slots), default 1: the FFN graph of a layer copies the first piece of the next layer, the attention graph in between
// (no cold ops, copy engine idle) the rest; a graph without own ops fills every free slot. 0 = no copies across graphs.
static int cold_ahead_own() {
    static const int n = [] {
        const char * e = getenv("GGML_CUDA_COLD_PREFETCH_AHEAD");
        return std::max(0, e != nullptr ? atoi(e) : 1);
    }();
    return n;
}

// GGML_CUDA_COLD_COPY_CHUNK_MB=n: a piece is copied as ceil(bytes / n MiB) commands (the AllReduce chunks and the other
// small copies interleave at that granularity instead of waiting behind a 7 ms piece); 0 = one command
static size_t cold_chunk_bytes() {
    static const size_t bytes = [] {
        const char * e = getenv("GGML_CUDA_COLD_COPY_CHUNK_MB");
        return (size_t) std::max(0, e != nullptr ? atoi(e) : 8)*1048576;
    }();
    return bytes;
}

static bool cold_check_enabled() {
    static const bool f = getenv("GGML_CUDA_COLD_PREFETCH_CHECK") != nullptr && atoi(getenv("GGML_CUDA_COLD_PREFETCH_CHECK")) != 0;
    return f;
}

static ggml_cuda_cold_prefetch::piece cold_piece_of(const ggml_tensor * op) {
    const char * host = nullptr;
    int n_cold = 0;
    ggml_cuda_mmid_cold(op, &host, &n_cold);
    return { host, (size_t) n_cold*op->src[0]->nb[2] };
}

static bool operator==(const ggml_cuda_cold_prefetch::piece & a, const ggml_cuda_cold_prefetch::piece & b) {
    return a.host == b.host && a.bytes == b.bytes;
}

// every 64th 16-byte word of a slot against the host piece (pinned, read over PCIe): a stale or torn slot counts
static __global__ void cold_check_kernel(const uint4 * __restrict__ dev, const uint4 * __restrict__ host, const size_t n16, int * __restrict__ bad) {
    const size_t i = ((size_t) blockIdx.x*blockDim.x + threadIdx.x)*64;
    if (i >= n16) {
        return;
    }
    const uint4 a = dev[i];
    const uint4 b = host[i];
    if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) {
        atomicAdd(bad, 1);
    }
}

// forget the filled slots (bookkeeping only: a later copy into a slot is stream-ordered after every earlier copy and
// read of it, as the graphs of one stream serialize and each graph joins its copy stream)
static void cold_ring_reset(ggml_cuda_cold_prefetch & pf) {
    for (auto & s : pf.slots) {
        s.filled = false;
    }
    pf.n_filled   = 0;
    pf.seq_issued = pf.seq_pos;
}

// copy piece seq[seq_issued] into the slot after the filled ones; the slot's last reader in this graph gates it
static void cold_issue_next(ggml_cuda_cold_prefetch & pf) {
    GGML_ASSERT(pf.n_filled < (int) pf.slots.size() && pf.seq_issued < (int) pf.seq.size());
    const int si = (pf.head + pf.n_filled) % (int) pf.slots.size();
    ggml_cuda_cold_prefetch::slot & s = pf.slots[si];
    const ggml_cuda_cold_prefetch::piece & pc = pf.seq[pf.seq_issued];
    if (s.freed_in == pf.graph_no) {
        CUDA_CHECK(cudaStreamWaitEvent(pf.copy, s.freed, 0));
    }
    const size_t chunk = cold_chunk_bytes() > 0 ? cold_chunk_bytes() : pc.bytes;
    for (size_t off = 0; off < pc.bytes; off += chunk) {
        CUDA_CHECK(cudaMemcpyAsync(s.buf + off, pc.host + off, std::min(chunk, pc.bytes - off), cudaMemcpyHostToDevice, pf.copy));
    }
    CUDA_CHECK(cudaEventRecord(s.ready, pf.copy));
    s.key       = pc;
    s.filled    = true;
    s.issued_in = pf.graph_no;
    pf.n_filled++;
    pf.seq_issued++;
}

static void cold_issue_ahead(ggml_cuda_cold_prefetch & pf) {
    while (pf.budget > 0 && pf.n_filled < (int) pf.slots.size() && pf.seq_issued < (int) pf.seq.size()) {
        cold_issue_next(pf);
        pf.budget--;
    }
}

bool ggml_cuda_cold_prefetch_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    ggml_cuda_cold_prefetch & pf = ctx.cold_pf;
    pf.active  = false;
    pf.started = false;
    pf.next    = 0;
    pf.budget  = 0;
    pf.ops.clear();
    if (!ggml_cuda_cold_prefetch_enabled()) {
        return false;
    }
    static const int64_t cold_prefetch_min_tok = getenv("GGML_CUDA_COLD_PREFETCH_MIN_TOK") != nullptr ? atoll(getenv("GGML_CUDA_COLD_PREFETCH_MIN_TOK")) : 1024;
    size_t need = pf.cap;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        // the prompt batches the host tier takes (GGML_CUDA_MOE_HOST_PROMPT=1, up to its MAX_TOK) need no prefetch; the
        // larger ones keep it (it was off for every batch with the host prompts on: 16K cold prefill +8-11%)
        // a batch below GGML_CUDA_COLD_PREFETCH_MIN_TOK tokens (default 1024) stages only the cold experts it routes to
        // (mmid16_stage_cold_experts) instead of copying every cold expert of the op: 269-616-token prompt batches took
        // 14-18% less that way on Qwen3.8 (the prefetch wins at 1536 tokens, where nearly every expert is routed to)
        if (ggml_cuda_mmid_cold_prompt(n) && (n->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            !ggml_cuda_moe_host_takes_prompt((int) n->src[2]->ne[1]) && n->src[2]->ne[1] >= cold_prefetch_min_tok &&
            ggml_cuda_mmid_f16_sm60_supported(n->src[0], n->src[1], n->src[2], n)) {
            pf.ops.push_back(n);
            need = std::max(need, cold_piece_of(n).bytes);
        }
    }
    if (pf.ops.empty() && (!pf.in_ubatch || pf.seq_issued >= (int) pf.seq.size() || pf.cap == 0)) {
        return false;
    }
    ggml_cuda_set_device(ctx.device);
    if (pf.copy == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&pf.copy, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&pf.fork, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&pf.join, cudaEventDisableTiming));
        pf.slots.resize(cold_n_slots());
        for (auto & s : pf.slots) {
            CUDA_CHECK(cudaEventCreateWithFlags(&s.ready, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.freed, cudaEventDisableTiming));
        }
        if (cold_check_enabled()) {
            CUDA_CHECK(cudaMalloc((void **) &pf.check, sizeof(int)));
            CUDA_CHECK(cudaMemset(pf.check, 0, sizeof(int)));
        }
    }
    if (need > pf.cap) {
        // larger buffers are only needed at the first prompt: wait for the old ones' copies and users
        CUDA_CHECK(cudaStreamSynchronize(pf.copy));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        for (auto & s : pf.slots) {
            if (s.buf != nullptr) {
                CUDA_CHECK(cudaFree(s.buf));
                s.buf = nullptr;
            }
        }
        pf.cap = 0;
        cold_ring_reset(pf);
        for (auto & s : pf.slots) {
            if (cudaMalloc((void **) &s.buf, need) != cudaSuccess) {
                (void) cudaGetLastError();
                GGML_LOG_WARN("%s: no VRAM for the cold-expert prefetch (%d x %.1f MiB), staging per op\n", __func__, (int) pf.slots.size(), need/1048576.0);
                for (auto & c : pf.slots) {
                    if (c.buf != nullptr) {
                        CUDA_CHECK(cudaFree(c.buf));
                        c.buf = nullptr;
                    }
                }
                pf.ops.clear();
                pf.seq.clear();
                pf.seq_pos = pf.seq_issued = 0;
                return false;
            }
        }
        pf.cap = need;
        static bool once = false;
        if (!once) {
            once = true;
            GGML_LOG_WARN("%s: dev%d cold-expert prefetch ring %d x %.1f MiB (%d ops in this graph, chunks of %.1f MiB, %d ahead)\n", __func__,
                          ctx.device, (int) pf.slots.size(), need/1048576.0, (int) pf.ops.size(), cold_chunk_bytes()/1048576.0, cold_ahead_own());
        }
    }
    // the graph's ops against the sequence: in place, or resynchronized (an ubatch boundary, a changed graph), or learned
    if (!pf.ops.empty()) {
        pf.in_ubatch = true;
        pf.n_own_graphs++;
        pf.n_pieces += (int64_t) pf.ops.size();
        pf.ntok = pf.ops[0]->ne[2];
        const ggml_cuda_cold_prefetch::piece first = cold_piece_of(pf.ops[0]);
        bool in_place = pf.seq_pos < (int) pf.seq.size() && pf.seq[pf.seq_pos] == first;
        if (!in_place) {
            int j = 0;
            while (j < (int) pf.seq.size() && !(pf.seq[j] == first)) {
                j++;
            }
            if (j < (int) pf.seq.size()) {
                pf.seq_pos = j;               // the sequence starts over (next ubatch) or skipped
                pf.n_resync++;
            } else if (pf.seq_pos < (int) pf.seq.size()) {
                pf.seq.clear();               // an op the sequence does not know mid-way: a changed graph, learn it again
                pf.seq_pos = 0;
                pf.n_relearn++;
            }
            cold_ring_reset(pf);
        }
        for (size_t k = 0; k < pf.ops.size(); ++k) {
            const ggml_cuda_cold_prefetch::piece pc = cold_piece_of(pf.ops[k]);
            const int at = pf.seq_pos + (int) k;
            if (at == (int) pf.seq.size()) {
                pf.seq.push_back(pc);
            } else if (!(pf.seq[at] == pc)) {
                // the tail of the known sequence differs from here on: cut it and learn the rest
                pf.seq.resize(at);
                pf.seq.push_back(pc);
                if (pf.seq_issued > at) {
                    cold_ring_reset(pf);
                }
            }
        }
        // the ring must hold exactly the next pieces of the sequence, in order
        for (int k = 0; k < pf.n_filled; ++k) {
            const auto & s = pf.slots[(pf.head + k) % (int) pf.slots.size()];
            if (pf.seq_pos + k >= (int) pf.seq.size() || !(s.key == pf.seq[pf.seq_pos + k])) {
                cold_ring_reset(pf);
                break;
            }
        }
        if (pf.seq_issued != pf.seq_pos + pf.n_filled) {
            cold_ring_reset(pf);
        }
        pf.budget = cold_ahead_own();
    } else {
        pf.budget = (int) pf.slots.size();
    }
    pf.active = !pf.ops.empty() || (pf.n_filled < (int) pf.slots.size() && pf.seq_issued < (int) pf.seq.size());
    if (pf.active && pf.ops.empty()) {
        pf.n_ahead_graphs++;
    }
    return pf.active;
}

void ggml_cuda_cold_prefetch_start(ggml_backend_cuda_context & ctx) {
    ggml_cuda_cold_prefetch & pf = ctx.cold_pf;
    if (!pf.active || pf.started) {
        return;
    }
    // the copy stream follows everything issued before this graph, so the slots nobody fills in this graph are free
    // (also when captured: a CUDA graph starts after the previous work on its stream, and the fork makes the copy
    // stream part of it)
    CUDA_CHECK(cudaEventRecord(pf.fork, ctx.stream()));
    CUDA_CHECK(cudaStreamWaitEvent(pf.copy, pf.fork, 0));
    // the graph's own pieces that no earlier graph copied, as far as the slots go (the rest as the GEMMs free slots)
    const int own_end = pf.seq_pos + (int) pf.ops.size();
    while (pf.seq_issued < own_end && pf.n_filled < (int) pf.slots.size()) {
        cold_issue_next(pf);
    }
    if (pf.ops.empty()) {
        cold_issue_ahead(pf);
    }
    pf.started = true;
}

void ggml_cuda_cold_prefetch_finish(ggml_backend_cuda_context & ctx) {
    ggml_cuda_cold_prefetch & pf = ctx.cold_pf;
    if (!pf.started) {
        return;
    }
    CUDA_CHECK(cudaEventRecord(pf.join, pf.copy));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), pf.join, 0));
    pf.started = false;
    pf.active  = false;
    pf.graph_no++;
}

bool ggml_cuda_cold_prefetch_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_cold_prefetch & pf = ctx.cold_pf;
    if (!pf.started) {
        return false;
    }
    int j = pf.next;
    while (j < (int) pf.ops.size() && pf.ops[j] != dst) {
        j++;
    }
    if (j == (int) pf.ops.size()) {
        return false;
    }
    // ops of this graph skipped (not computed): their pieces are consumed unread (the copies are stream-ordered
    // before any later copy into the slot; the issuing below refills the ring)
    for (; pf.next < j; pf.next++) {
        GGML_ASSERT(pf.n_filled > 0);
        pf.slots[pf.head].filled = false;
        pf.head = (pf.head + 1) % (int) pf.slots.size();
        pf.n_filled--;
        pf.seq_pos++;
    }
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_cold_prefetch::piece pc = cold_piece_of(dst);
    if (!(pf.n_filled > 0 && pf.slots[pf.head].key == pc)) {
        // the ring holds something else (a graph that changed under the sequence): drop it, re-learn from here and copy
        // this graph's remaining pieces now (their slots' last readers in this graph gate the copies)
        static int n_warn = 0;
        pf.n_oos++;
        if (n_warn++ < 4) {
            GGML_LOG_WARN("%s: dev%d cold-expert prefetch ring out of sequence at piece %d of %d: copying per op\n", __func__,
                          ctx.device, pf.seq_pos, (int) pf.seq.size());
        }
        cold_ring_reset(pf);
        pf.seq.resize(pf.seq_pos);
        for (size_t k = j; k < pf.ops.size(); ++k) {
            pf.seq.push_back(cold_piece_of(pf.ops[k]));
        }
        while (pf.seq_issued < (int) pf.seq.size() && pf.n_filled < (int) pf.slots.size()) {
            cold_issue_next(pf);
        }
    }
    ggml_cuda_cold_prefetch::slot & s = pf.slots[pf.head];
    mmid16_gemm_plan p(ctx);
    // the cold experts come from the prefetched slot: the setup must not stage them again (it did in qp1-qp3:
    // a second copy of every op's cold experts through UVA loads, +22% prefill time and +228 MiB of pool)
    mmid16_gemm_setup(ctx, p, dst->src[0], dst->src[1], dst->src[2], dst, /*skip_cold =*/ false, MMID16_COLD_BY_LAUNCH);
    if (s.issued_in == pf.graph_no) {
        // copied in this graph: wait for it (a copy of an earlier graph is done: that graph joined its copy stream)
        CUDA_CHECK(cudaStreamWaitEvent(stream, s.ready, 0));
    }
    if (pf.check != nullptr) {
        const size_t n16 = pc.bytes/16;
        cold_check_kernel<<<(unsigned) ((n16/64 + 255)/256), 256, 0, stream>>>((const uint4 *) s.buf, (const uint4 *) pc.host, n16, pf.check);
        CUDA_CHECK(cudaGetLastError());
        pf.n_checked++;
    }
    const int n_hot = (int) dst->src[0]->ne[2];
    int n_cold = 0;
    ggml_cuda_mmid_cold(dst, nullptr, &n_cold);
    mmid16_gemm_launch(ctx, p, 0, n_hot + n_cold, s.buf, n_hot);
    CUDA_CHECK(cudaEventRecord(s.freed, stream));
    s.freed_in = pf.graph_no;
    s.filled   = false;
    pf.head    = (pf.head + 1) % (int) pf.slots.size();
    pf.n_filled--;
    pf.seq_pos++;
    pf.next = j + 1;
    // the rest of this graph's own pieces first, then the ones ahead within the budget
    const int own_end = pf.seq_pos + (int) pf.ops.size() - pf.next;
    while (pf.seq_issued < own_end && pf.n_filled < (int) pf.slots.size()) {
        cold_issue_next(pf);
    }
    if (pf.seq_issued >= own_end) {
        cold_issue_ahead(pf);
    }
    return true;
}

void ggml_cuda_cold_prefetch_sync(ggml_backend_cuda_context & ctx, const bool invalidate) {
    ggml_cuda_cold_prefetch & pf = ctx.cold_pf;
    if (invalidate && pf.n_filled > 0) {
        cold_ring_reset(pf);
    }
    if (pf.in_ubatch) {
        // the ubatch ended: the sequence is one ubatch's piece order. A ubatch that ends mid-sequence (a shorter layer
        // range than the one learned: the load-time warm-up runs several ranges, a pipeline head runs its first
        // layers) cuts it there, or the next ubatch's resynchronization would land on an earlier copy of its
        // first piece inside the concatenated ranges and leave seq_pos mid-sequence after it (then every decode step
        // refilled the ring: 3 x 63 MiB per step, +17 ms; PP8_PROGRESS.md)
        if (pf.seq_pos > 0 && pf.seq_pos < (int) pf.seq.size()) {
            pf.seq.resize(pf.seq_pos);
            pf.n_trunc++;
        }
        cold_ring_reset(pf);
        pf.in_ubatch = false;
    }
    if (pf.check != nullptr && pf.n_checked > 0) {
        int bad = 0;
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMemcpy(&bad, pf.check, sizeof(int), cudaMemcpyDeviceToHost));
        if (bad != 0) {
            GGML_LOG_ERROR("%s: dev%d cold-expert prefetch: %d stale 16-byte words in %lld checked pieces\n", __func__, ctx.device, bad, (long long) pf.n_checked);
            CUDA_CHECK(cudaMemset(pf.check, 0, sizeof(int)));
        }
        pf.n_checked = 0;
    }
}

void ggml_cuda_cold_prefetch_free(ggml_backend_cuda_context & ctx) {
    ggml_cuda_cold_prefetch & pf = ctx.cold_pf;
    if (pf.n_own_graphs > 0) {
        GGML_LOG_WARN("cold-prefetch: dev%d graphs with cold ops %lld (%lld pieces), copy-ahead graphs %lld, resync %lld, relearn %lld, truncate %lld, out-of-sequence %lld, sequence %d\n",
                      ctx.device, (long long) pf.n_own_graphs, (long long) pf.n_pieces, (long long) pf.n_ahead_graphs, (long long) pf.n_resync,
                      (long long) pf.n_relearn, (long long) pf.n_trunc, (long long) pf.n_oos, (int) pf.seq.size());
    }
    if (pf.copy != nullptr) {
        cudaStreamSynchronize(pf.copy);
    }
    for (auto & s : pf.slots) {
        if (s.buf != nullptr) {
            cudaFree(s.buf);
        }
        if (s.ready != nullptr) {
            cudaEventDestroy(s.ready);
        }
        if (s.freed != nullptr) {
            cudaEventDestroy(s.freed);
        }
    }
    if (pf.check != nullptr) {
        cudaFree(pf.check);
    }
    if (pf.fork != nullptr) {
        cudaEventDestroy(pf.fork);
    }
    if (pf.join != nullptr) {
        cudaEventDestroy(pf.join);
    }
    if (pf.copy != nullptr) {
        cudaStreamDestroy(pf.copy);
    }
    pf = {};
}
