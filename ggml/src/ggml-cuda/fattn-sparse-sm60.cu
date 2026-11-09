#include "fattn-common.cuh"
#include "fattn-sparse-sm60.cuh"
#include "convert.cuh"

// Sparse attention for Pascal (no tensor cores, F2F f16->f32 conversion at 1/4 of the FFMA rate):
// a block takes one query and 32 heads of one kv head and walks the query's visible rows (the mask compaction's
// index list, <= n_kv_max) in chunks of 16. The 16 gathered rows are converted once and kept in shared memory as
// FP32 (32 KB) for both the scores and the values (K == V in deepseek41; otherwise the V rows are gathered after
// the scores). K/V rows are read in their cache type: f16 rows are converted from the gather registers, q8_0 rows
// (544 bytes) are staged raw in shared memory and dequantized from there, so nothing of the K/V tensor beyond the
// selected rows is touched (the whole-tensor f16 conversion of the other kernels grows with the context).
// Q (32 heads x 512, FP32) lives in registers: a thread holds 4 heads x 16 columns. All arithmetic is FP32: the
// tile kernel's 16-element fp16 windows already cost acceptance on this model.
//
//   scores: S[32 h][16 r] = sum_d Q[h][d] K[r][d]; a warp owns 4 heads, its 32 lanes own 16-column slices
//           (float4 j, j + 32, j + 64, j + 96 of the row), a pass takes 4 rows: 4 h x 4 r x 16 d per lane, then
//           the 16 partial sums are reduce-scattered over the 32 lanes with shuffles. The heads and rows a lane
//           accumulates are permuted by the lane index (slot s holds (head, row) = s ^ (lane >> 1)) so that each
//           butterfly stage adds a fixed register of the partner: no selects.
//   softmax: online per head over the 16 rows, the sinks join the denominator at the end
//   values: O[32 h][512 d] += P[h][r] V[r][d], a thread owns 8 heads x (4 + 4) columns
#define FSS_D   512
#define FSS_HB  32   // heads per block
#define FSS_RC  16   // rows per chunk
#define FSS_NT  256  // threads
#define FSS_PS  40   // row stride of the score tile (floats): 16-byte aligned, conflict-free stores
#define FSS_Q8_ROW (FSS_D/QK8_0*(int) sizeof(block_q8_0)) // 544 bytes per q8_0 row
#define FSS_Q8_U4  (FSS_RC*FSS_Q8_ROW/16)                 // 544 uint4 per chunk

struct fattn_sparse_sm60_args {
    const char *    Q;      // f32, strides in bytes
    int64_t         nbq1, nbq2, nbq3;
    const char *    K;      // f16 or q8_0 rows
    int64_t         nbk1, nbk2, nbk3;
    const char *    V;      // same type as K
    int64_t         nbv1, nbv2, nbv3;
    const half *    mask;
    int64_t         sm1, sm3; // elements
    const int32_t * idx;    // [seq][query][n_kv_max]
    const int32_t * counts; // [seq][query]
    const float *   sinks;  // [n_head] or nullptr
    float *         dst;    // [DV][n_head][n_q][n_seq]
    float *         part;   // n_split > 1: per block the unnormalized O[32][512], then m[32], l[32]
    int             n_split; // blocks per (query, head group): each takes cpb chunks of the row list
    int             cpb;
    int             n_q;
    int             n_head;
    int             n_kv_max;
    int             gqa;
    int             kv_same;
    float           scale;
};

static __device__ __forceinline__ void fss_cvt8(const uint4 u, float4 & lo, float4 & hi) {
    const float2 a = __half22float2(*(const half2 *) &u.x);
    const float2 b = __half22float2(*(const half2 *) &u.y);
    const float2 c = __half22float2(*(const half2 *) &u.z);
    const float2 d = __half22float2(*(const half2 *) &u.w);
    lo = make_float4(a.x, a.y, b.x, b.y);
    hi = make_float4(c.x, c.y, d.x, d.y);
}

static __device__ __forceinline__ int fss_row_id(const int32_t * idx, const int count, const int rsel) {
    const int id = rsel < count ? idx[rsel] : -1;
    return id >= 0 ? id : 0; // a dead row reads row 0, its score is -inf
}

// f16 rows: the 4 row pieces of a thread for chunk c: piece i is row (tid>>6) + 4i, 16-byte column (tid & 63)
static __device__ __forceinline__ void fss_gather_f16(
        const char * base, const int64_t nb1, const int32_t * idx, const int count, const int c, const int tid, uint4 (&v)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int id = fss_row_id(idx, count, c*FSS_RC + (tid >> 6) + 4*i);
        v[i] = __ldg((const uint4 *) (base + (int64_t) id*nb1) + (tid & 63));
    }
}

static __device__ __forceinline__ void fss_store_rows_f16(float (* s_k)[FSS_D], const uint4 (&v)[4], const int tid) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r  = (tid >> 6) + 4*i;
        const int qd = tid & 63;
        float4 lo, hi;
        fss_cvt8(v[i], lo, hi);
        *(float4 *) (s_k[r] + 8*qd)     = lo;
        *(float4 *) (s_k[r] + 8*qd + 4) = hi;
    }
}

// q8_0 rows: the chunk's 16 rows are 544 contiguous bytes each = 34 uint4; piece p = tid + 256 i is
// row p / 34, uint4 p % 34 of the row
static __device__ __forceinline__ void fss_gather_q8(
        const char * base, const int64_t nb1, const int32_t * idx, const int count, const int c, const int tid, uint4 (&v)[3]) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int p = tid + FSS_NT*i;
        if (p < FSS_Q8_U4) {
            const int r  = p / (FSS_Q8_ROW/16);
            const int u  = p - r*(FSS_Q8_ROW/16);
            const int id = fss_row_id(idx, count, c*FSS_RC + r);
            v[i] = __ldg((const uint4 *) (base + (int64_t) id*nb1) + u);
        }
    }
}

static __device__ __forceinline__ void fss_stage_q8(uint4 * s_raw, const uint4 (&v)[3], const int tid) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int p = tid + FSS_NT*i;
        if (p < FSS_Q8_U4) {
            s_raw[p] = v[i];
        }
    }
}

// thread (row tid >> 4, block tid & 15): the 32 values of the staged block, one float4 at a time in an order
// rotated by the block index so that the 8 lanes of a store phase hit 8 different bank groups (and only 4 values
// are live next to the Q and O registers)
static __device__ __forceinline__ void fss_dequant_q8(float (* s_k)[FSS_D], const uint4 * s_raw, const int tid) {
    const int r   = tid >> 4;
    const int b   = tid & 15;
    const uint16_t * src = (const uint16_t *) s_raw + r*(FSS_Q8_ROW/2) + b*((int) sizeof(block_q8_0)/2);
    const float d   = __half2float(__ushort_as_half(src[0]));
    const float off = -8388736.0f*d; // (2^23 + 128) d: the byte u = q + 128 sits in the mantissa of 2^23 + u
    float * dst = s_k[r] + QK8_0*b;
#pragma unroll
    for (int t = 0; t < QK8_0/4; ++t) {
        const int j = (t + b) & 7;
        const uint32_t w0 = src[1 + 2*j];
        const uint32_t w1 = src[2 + 2*j];
        float4 f;
        f.x = fmaf(__uint_as_float((w0 & 0xff) ^ 0x4B000080u), d, off);
        f.y = fmaf(__uint_as_float((w0 >> 8)   ^ 0x4B000080u), d, off);
        f.z = fmaf(__uint_as_float((w1 & 0xff) ^ 0x4B000080u), d, off);
        f.w = fmaf(__uint_as_float((w1 >> 8)   ^ 0x4B000080u), d, off);
        *(float4 *) (dst + 4*j) = f;
    }
}

// kv_same: V is K (deepseek41): the chunk's rows serve both the scores and the values.
// With n_split > 1 (few queries) a block takes cpb chunks of the list and leaves its softmax state and
// unnormalized sum to part; fattn_sparse_sm60_join adds the blocks up.
template <ggml_type type_KV, bool kv_same>
__global__ void __launch_bounds__(FSS_NT, 1)
fattn_sparse_sm60_d512(const fattn_sparse_sm60_args a) {
    constexpr bool q8 = type_KV == GGML_TYPE_Q8_0;
    static_assert(q8 || type_KV == GGML_TYPE_F16, "K/V type");

    __shared__ __align__(16) float s_k[FSS_RC][FSS_D];      // the chunk's rows
    __shared__ __align__(16) uint4 s_raw[q8 ? FSS_Q8_U4 : 1]; // the chunk's raw q8_0 rows
    __shared__ __align__(16) float s_p[FSS_RC][FSS_PS];     // scores, then weights: [row][head]
    __shared__ float s_m[FSS_HB];
    __shared__ float s_l[FSS_HB];
    __shared__ float s_alpha[FSS_HB];
    __shared__ float s_mv[FSS_RC];                           // mask values of the chunk's rows

    const int tid = threadIdx.x;
    const int q   = blockIdx.x;
    const int hg  = blockIdx.y;
    const int seq   = blockIdx.z / a.n_split;
    const int split = blockIdx.z - seq*a.n_split;
    const int h0  = hg*FSS_HB;
    const int kvh = h0 / a.gqa;

    const char *    Qb    = a.Q + seq*a.nbq3 + (int64_t) q*a.nbq1 + (int64_t) h0*a.nbq2;
    const char *    Kb    = a.K + seq*a.nbk3 + (int64_t) kvh*a.nbk2;
    const char *    Vb    = a.V + seq*a.nbv3 + (int64_t) kvh*a.nbv2;
    const half *    mrow  = a.mask + seq*a.sm3 + (int64_t) q*a.sm1;
    const int32_t * idx   = a.idx + ((int64_t) seq*a.n_q + q)*a.n_kv_max;
    const int       count = a.counts[(int64_t) seq*a.n_q + q];
    const int       n_ch  = (count + FSS_RC - 1)/FSS_RC;
    const int       c0    = split*a.cpb;
    const int       c1    = min(c0 + a.cpb, n_ch);

    if (tid < FSS_HB) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }

    // values: 8 heads x (4 + 4) columns per thread
    const int pv_hg = tid >> 6;  // heads 8*pv_hg ..
    const int pv_dg = tid & 63;  // columns 4*pv_dg .. and 256 + 4*pv_dg ..
    float o[8][8];
#pragma unroll
    for (int h = 0; h < 8; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[h][i] = 0.0f;
        }
    }

    // scores: warp w owns heads 4w .. 4w+3, lane l the float4 columns l, l + 32, l + 64, l + 96;
    // register slot hs holds head 4w + (hs ^ (l >> 3)), row slot rs of a pass holds row 4p + (rs ^ ((l >> 1) & 3))
    const int qk_w  = tid >> 5;
    const int qk_l  = tid & 31;
    const int qk_hx = qk_l >> 3;
    const int qk_rx = (qk_l >> 1) & 3;
    float4 qr[4][4];
#pragma unroll
    for (int hs = 0; hs < 4; ++hs) {
        const char * Qh = Qb + (int64_t) (4*qk_w + (hs ^ qk_hx))*a.nbq2;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            qr[hs][k] = __ldg((const float4 *) Qh + qk_l + 32*k);
        }
    }

    uint4 kpre[q8 ? 3 : 4];
    if (c0 < c1) {
        if constexpr (q8) {
            fss_gather_q8(Kb, a.nbk1, idx, count, c0, tid, kpre);
            fss_stage_q8(s_raw, kpre, tid);
        } else {
            fss_gather_f16(Kb, a.nbk1, idx, count, c0, tid, kpre);
        }
    }
    __syncthreads();

    for (int c = c0; c < c1; ++c) {
        if constexpr (q8) {
            fss_dequant_q8(s_k, s_raw, tid);
        } else {
            fss_store_rows_f16(s_k, kpre, tid);
        }
        if (tid < FSS_RC) {
            const int rsel = c*FSS_RC + tid;
            const int id = rsel < count ? idx[rsel] : -1;
            s_mv[tid] = id >= 0 ? __half2float(mrow[id]) : -INFINITY;
        }
        __syncthreads();

        // the next chunk's K rows land in registers while this chunk computes
        uint4 vpre[4];
        if constexpr (kv_same || q8) {
            if (c + 1 < c1) {
                if constexpr (q8) {
                    fss_gather_q8(Kb, a.nbk1, idx, count, c + 1, tid, kpre);
                } else {
                    fss_gather_f16(Kb, a.nbk1, idx, count, c + 1, tid, kpre);
                }
            }
        } else {
            fss_gather_f16(Vb, a.nbv1, idx, count, c, tid, vpre);
        }

#ifndef FSS_SKIP_QK
#pragma unroll 1
        for (int p = 0; p < FSS_RC/4; ++p) {
            float acc[16];
#pragma unroll
            for (int s = 0; s < 16; ++s) {
                acc[s] = 0.0f;
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float4 kq[4];
#pragma unroll
                for (int rs = 0; rs < 4; ++rs) {
                    kq[rs] = *(const float4 *) (s_k[4*p + (rs ^ qk_rx)] + 4*(qk_l + 32*k));
                }
#pragma unroll
                for (int hs = 0; hs < 4; ++hs) {
#pragma unroll
                    for (int rs = 0; rs < 4; ++rs) {
                        acc[4*hs + rs] += qr[hs][k].x*kq[rs].x;
                        acc[4*hs + rs] += qr[hs][k].y*kq[rs].y;
                        acc[4*hs + rs] += qr[hs][k].z*kq[rs].z;
                        acc[4*hs + rs] += qr[hs][k].w*kq[rs].w;
                    }
                }
            }
            // butterfly: lane bit 4 <-> slot bit 3, 3 <-> 2, 2 <-> 1, 1 <-> 0; the partner's slot s | c holds the
            // same (head, row) as this lane's slot s
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                acc[s] += __shfl_xor_sync(0xffffffff, acc[s + 8], 16);
            }
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                acc[s] += __shfl_xor_sync(0xffffffff, acc[s + 4], 8);
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                acc[s] += __shfl_xor_sync(0xffffffff, acc[s + 2], 4);
            }
            acc[0] += __shfl_xor_sync(0xffffffff, acc[1], 2);
            acc[0] += __shfl_xor_sync(0xffffffff, acc[0], 1);
            // slot 0 of lane l: head 4w + (l >> 3), row 4p + ((l >> 1) & 3); lanes l and l ^ 1 agree
            if ((qk_l & 1) == 0) {
                const int r = 4*p + qk_rx;
                s_p[r][4*qk_w + qk_hx] = acc[0]*a.scale + s_mv[r];
            }
        }
#else
        if (tid < FSS_RC*FSS_HB) {
            s_p[tid >> 5][tid & 31] = s_mv[tid >> 5];
        }
#endif
        __syncthreads();

        if constexpr (!kv_same) {
            // the scores are done with the K rows: V takes their place
            if constexpr (q8) {
                uint4 vraw[3];
                fss_gather_q8(Vb, a.nbv1, idx, count, c, tid, vraw);
                fss_stage_q8(s_raw, vraw, tid);
                __syncthreads();
                fss_dequant_q8(s_k, s_raw, tid);
            } else {
                fss_store_rows_f16(s_k, vpre, tid);
                if (c + 1 < c1) {
                    fss_gather_f16(Kb, a.nbk1, idx, count, c + 1, tid, kpre);
                }
            }
        }

        // online softmax: 8 lanes per head, 2 rows each
        {
            const int h  = tid >> 3;
            const int l8 = tid & 7;
            const float v0 = s_p[2*l8][h];
            const float v1 = s_p[2*l8 + 1][h];
            float mx = fmaxf(v0, v1);
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            }
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, mx);
            float p0 = 0.0f, p1 = 0.0f, alpha = 1.0f;
            if (m_new > -INFINITY) {
                p0    = v0 > -INFINITY ? expf(v0 - m_new) : 0.0f;
                p1    = v1 > -INFINITY ? expf(v1 - m_new) : 0.0f;
                alpha = m_old > -INFINITY ? expf(m_old - m_new) : 0.0f;
            }
            float l = p0 + p1;
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                l += __shfl_xor_sync(0xffffffff, l, off);
            }
            s_p[2*l8][h]     = p0;
            s_p[2*l8 + 1][h] = p1;
            if (l8 == 0) {
                s_alpha[h] = alpha;
                s_l[h]     = s_l[h]*alpha + l;
                s_m[h]     = m_new;
            }
        }
        __syncthreads();

        // values
#pragma unroll
        for (int h = 0; h < 8; ++h) {
            const float alpha = s_alpha[pv_hg*8 + h];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                o[h][i] *= alpha;
            }
        }
#ifndef FSS_SKIP_PV
#pragma unroll 4
        for (int r = 0; r < FSS_RC; ++r) {
            const float4 p0 = *(const float4 *) (s_p[r] + pv_hg*8);
            const float4 p1 = *(const float4 *) (s_p[r] + pv_hg*8 + 4);
            const float4 v0 = *(const float4 *) (s_k[r] + 4*pv_dg);
            const float4 v1 = *(const float4 *) (s_k[r] + FSS_D/2 + 4*pv_dg);
            const float p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
            for (int h = 0; h < 8; ++h) {
                o[h][0] += p[h]*v0.x; o[h][1] += p[h]*v0.y; o[h][2] += p[h]*v0.z; o[h][3] += p[h]*v0.w;
                o[h][4] += p[h]*v1.x; o[h][5] += p[h]*v1.y; o[h][6] += p[h]*v1.z; o[h][7] += p[h]*v1.w;
            }
        }
#endif
        if constexpr (q8) {
            // every thread passed this chunk's dequantization: the staging takes the next rows
            if (c + 1 < c1) {
                fss_stage_q8(s_raw, kpre, tid);
            }
        }
        __syncthreads(); // the next chunk overwrites the rows and the weights
    }

    if (a.n_split > 1) {
        // the block's part: the sinks join in fattn_sparse_sm60_join
        float * part = a.part + ((((int64_t) seq*a.n_q + q)*(a.n_head/FSS_HB) + hg)*a.n_split + split)*(FSS_HB*FSS_D + 2*FSS_HB);
        float * po   = part + (int64_t) pv_hg*8*FSS_D;
#pragma unroll
        for (int h = 0; h < 8; ++h) {
            float * oh = po + (int64_t) h*FSS_D;
            *(float4 *) (oh + 4*pv_dg)           = make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
            *(float4 *) (oh + FSS_D/2 + 4*pv_dg) = make_float4(o[h][4], o[h][5], o[h][6], o[h][7]);
        }
        if (tid < FSS_HB) {
            part[FSS_HB*FSS_D + tid]          = s_m[tid];
            part[FSS_HB*FSS_D + FSS_HB + tid] = s_l[tid];
        }
        return;
    }

    // the sinks: one more softmax term without a value
    if (tid < FSS_HB) {
        float l = s_l[tid];
        if (a.sinks) {
            const float m = s_m[tid];
            l += m > -INFINITY ? expf(a.sinks[h0 + tid] - m) : 0.0f;
        }
        s_alpha[tid] = l > 0.0f ? 1.0f/l : 0.0f;
    }
    __syncthreads();

    float * out = a.dst + (((int64_t) seq*a.n_q + q)*a.n_head + h0 + pv_hg*8)*FSS_D;
#pragma unroll
    for (int h = 0; h < 8; ++h) {
        const float inv = s_alpha[pv_hg*8 + h];
        float * oh = out + (int64_t) h*FSS_D;
        *(float4 *) (oh + 4*pv_dg)           = make_float4(o[h][0]*inv, o[h][1]*inv, o[h][2]*inv, o[h][3]*inv);
        *(float4 *) (oh + FSS_D/2 + 4*pv_dg) = make_float4(o[h][4]*inv, o[h][5]*inv, o[h][6]*inv, o[h][7]*inv);
    }
}

// block (head, query, seq), 128 threads x float4: out = sum_s exp(m_s - m) O_s / (sum_s exp(m_s - m) l_s + exp(sink - m))
static __global__ void fattn_sparse_sm60_join(const fattn_sparse_sm60_args a) {
    const int h   = blockIdx.x;
    const int q   = blockIdx.y;
    const int seq = blockIdx.z;
    const int hg  = h / FSS_HB;
    const int hh  = h - hg*FSS_HB;
    const float * part = a.part + ((((int64_t) seq*a.n_q + q)*(a.n_head/FSS_HB) + hg)*a.n_split)*(FSS_HB*FSS_D + 2*FSS_HB);
    constexpr int ps = FSS_HB*FSS_D + 2*FSS_HB;
    float m = -INFINITY;
    for (int sp = 0; sp < a.n_split; ++sp) {
        m = fmaxf(m, part[(int64_t) sp*ps + FSS_HB*FSS_D + hh]);
    }
    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float  l = 0.0f;
    if (m > -INFINITY) {
        for (int sp = 0; sp < a.n_split; ++sp) {
            const float * p = part + (int64_t) sp*ps;
            const float ms = p[FSS_HB*FSS_D + hh];
            const float w  = ms > -INFINITY ? expf(ms - m) : 0.0f;
            const float4 v = *(const float4 *) (p + (int64_t) hh*FSS_D + 4*threadIdx.x);
            o.x += w*v.x; o.y += w*v.y; o.z += w*v.z; o.w += w*v.w;
            l   += w*p[FSS_HB*FSS_D + FSS_HB + hh];
        }
        if (a.sinks) {
            l += expf(a.sinks[h] - m);
        }
    }
    const float inv = l > 0.0f ? 1.0f/l : 0.0f;
    float * out = a.dst + (((int64_t) seq*a.n_q + q)*a.n_head + h)*FSS_D;
    *(float4 *) (out + 4*threadIdx.x) = make_float4(o.x*inv, o.y*inv, o.z*inv, o.w*inv);
}

static int fss_env_int(const char * name, const int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

// K and V rows as the kernel reads them without a conversion: both f16 or both q8_0, rows (and the head / sequence
// strides) 16-byte aligned. Decided from the strides only: the scratch size is queried before the buffers exist,
// the launch asserts the data alignment (buffers are 128-byte aligned, cache views sit at row multiples).
bool ggml_cuda_fattn_sparse_sm60_kv_native(const ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (K->type != V->type || (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0)) {
        return false;
    }
    for (const ggml_tensor * t : {K, V}) {
        if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0 || t->nb[3] % 16 != 0) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_fattn_sparse_sm60_shall_use(const int cc, const ggml_tensor * dst) {
    static const int  enabled = fss_env_int("GGML_CUDA_FA_SPARSE_SM60", 1);
    static const int  min_q   = fss_env_int("GGML_CUDA_FA_SPARSE_SM60_MIN_Q", 1);
    if (!enabled || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc >= GGML_CUDA_CC_VOLTA) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (dst->src[5] != nullptr || mask == nullptr || mask->type != GGML_TYPE_F16) {
        return false;
    }
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    if (n_kv_max <= 0 || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (Q->ne[0] != FSS_D || K->ne[0] != FSS_D || V->ne[0] != FSS_D || Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t gqa = Q->ne[2] / K->ne[2];
    if (Q->ne[2] % K->ne[2] != 0 || gqa % FSS_HB != 0) {
        return false;
    }
    if (Q->ne[1] < min_q || Q->ne[1] > INT32_MAX || K->ne[1] < 2*(int64_t) n_kv_max || K->ne[1] > INT32_MAX) {
        return false;
    }
    if (mask->ne[0] != K->ne[1] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != Q->ne[3] ||
            K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3] || V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1]) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 || Q->nb[3] % 16 != 0) {
        return false;
    }
    if (ggml_cuda_fattn_sparse_sm60_kv_native(dst)) {
        return true;
    }
    for (const ggml_tensor * t : {K, V}) {
        if (t->nb[0] != ggml_type_size(t->type)) {
            return false;
        }
        if (t->type == GGML_TYPE_F16 && (t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0 || t->nb[3] % 16 != 0)) {
            return false;
        }
        if (t->type != GGML_TYPE_F16 && !ggml_is_contiguously_allocated(t) && ggml_get_to_fp16_nc_cuda(t->type) == nullptr) {
            return false;
        }
        if (t->type != GGML_TYPE_F16 && ggml_is_contiguously_allocated(t) && ggml_get_to_fp16_cuda(t->type) == nullptr) {
            return false;
        }
    }
    return true;
}

// K or V as f16 rows: the tensor itself, or its conversion into the scratch after dst (see
// ggml_cuda_flash_attn_ext_get_f16_extra_data); the strides come back in bytes
static void fss_kv_f16(const ggml_tensor * T, const uintptr_t scratch, cudaStream_t stream,
        const char *& data, int64_t & nb1, int64_t & nb2, int64_t & nb3) {
    data = (const char *) T->data;
    nb1  = T->nb[1];
    nb2  = T->nb[2];
    nb3  = T->nb[3];
    if (T->type == GGML_TYPE_F16) {
        return;
    }
    GGML_ASSERT(scratch != 0);
    half * f16 = (half *) scratch;
    const size_t bs = ggml_blck_size(T->type);
    const size_t ts = ggml_type_size(T->type);
    if (ggml_is_contiguously_allocated(T)) {
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(T->type);
        to_fp16(data, f16, ggml_nelements(T), stream);
        nb1 = nb1*bs*sizeof(half)/ts;
        nb2 = nb2*bs*sizeof(half)/ts;
        nb3 = nb3*bs*sizeof(half)/ts;
    } else {
        to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(T->type);
        to_fp16(data, f16, T->ne[0], T->ne[1], T->ne[2], T->ne[3], nb1/ts, nb2/ts, nb3/ts, stream);
        nb1 = T->ne[0]*sizeof(half);
        nb2 = T->ne[1]*nb1;
        nb3 = T->ne[2]*nb2;
    }
    data = (const char *) f16;
}

void ggml_cuda_fattn_sparse_sm60(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    cudaStream_t stream = ctx.stream();

    const bool native = ggml_cuda_fattn_sparse_sm60_kv_native(dst);
    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, !native, !native);
    const bool V_is_K_view = V == K || (V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs)));

    fattn_sparse_sm60_args a;
    if (native) {
        GGML_ASSERT((uintptr_t) K->data % 16 == 0 && (uintptr_t) V->data % 16 == 0);
        a.K    = (const char *) K->data;
        a.nbk1 = K->nb[1];
        a.nbk2 = K->nb[2];
        a.nbk3 = K->nb[3];
        a.V    = (const char *) V->data;
        a.nbv1 = V->nb[1];
        a.nbv2 = V->nb[2];
        a.nbv3 = V->nb[3];
    } else {
        fss_kv_f16(K, f16_extra.K, stream, a.K, a.nbk1, a.nbk2, a.nbk3);
        if (V_is_K_view && V->type == K->type && V->nb[1] == K->nb[1] && V->nb[2] == K->nb[2] && V->nb[3] == K->nb[3]) {
            a.V    = a.K;
            a.nbv1 = a.nbk1;
            a.nbv2 = a.nbk2;
            a.nbv3 = a.nbk3;
        } else {
            fss_kv_f16(V, f16_extra.V, stream, a.V, a.nbv1, a.nbv2, a.nbv3);
        }
    }
    a.kv_same = a.V == a.K && a.nbv1 == a.nbk1 && a.nbv2 == a.nbk2 && a.nbv3 == a.nbk3;
    GGML_ASSERT(a.nbk1 % 16 == 0 && a.nbk2 % 16 == 0 && a.nbk3 % 16 == 0);
    GGML_ASSERT(a.nbv1 % 16 == 0 && a.nbv2 % 16 == 0 && a.nbv3 % 16 == 0);

    a.n_q      = (int) Q->ne[1];
    a.n_head   = (int) Q->ne[2];
    a.n_kv_max = ggml_get_op_params_i32(dst, 4);
    a.gqa      = (int) (Q->ne[2] / K->ne[2]);
    memcpy(&a.scale, (const float *) dst->op_params + 0, sizeof(float));

    // one index list per query: the visible rows, -1 padded, and the count
    const int64_t n_lists = (int64_t) a.n_q*mask->ne[3];
    ggml_cuda_pool_alloc<int32_t> lists(ctx.pool(), n_lists*a.n_kv_max + n_lists);
    a.idx    = lists.ptr;
    a.counts = lists.ptr + n_lists*a.n_kv_max;
    ggml_cuda_flash_attn_ext_compact_mask(mask, lists.ptr, lists.ptr + n_lists*a.n_kv_max, a.n_q, 1, a.n_kv_max, stream);

    a.Q     = (const char *) Q->data;
    a.nbq1  = Q->nb[1];
    a.nbq2  = Q->nb[2];
    a.nbq3  = Q->nb[3];
    a.mask  = (const half *) mask->data;
    a.sm1   = mask->nb[1]/sizeof(half);
    a.sm3   = mask->nb[3]/sizeof(half);
    a.sinks = sinks ? (const float *) sinks->data : nullptr;
    a.dst   = (float *) dst->data;

    // few queries: the chunks of a list go to several blocks (a block per chunk at the extreme), so that the device
    // is busy; many queries: a block per (query, head group) walks the whole list
    static const int min_blocks = fss_env_int("GGML_CUDA_FA_SPARSE_SM60_MIN_BLOCKS", 2*ggml_cuda_info().devices[ggml_cuda_get_device()].nsm);
    const int     n_ch_max = (a.n_kv_max + FSS_RC - 1)/FSS_RC;
    const int64_t n_blocks = (int64_t) a.n_q*(a.n_head/FSS_HB)*Q->ne[3];
    a.n_split = (int) std::min<int64_t>(n_ch_max, std::max<int64_t>(1, (min_blocks + n_blocks - 1)/n_blocks));
    a.cpb     = (n_ch_max + a.n_split - 1)/a.n_split;
    a.n_split = (n_ch_max + a.cpb - 1)/a.cpb;
    a.part    = nullptr;
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (a.n_split > 1) {
        a.part = part.alloc((size_t) n_blocks*a.n_split*(FSS_HB*FSS_D + 2*FSS_HB));
    }

    const dim3 grid((unsigned) a.n_q, (unsigned) (a.n_head/FSS_HB), (unsigned) (Q->ne[3]*a.n_split));
    const bool q8 = native && K->type == GGML_TYPE_Q8_0;
    if (q8) {
        if (a.kv_same) {
            fattn_sparse_sm60_d512<GGML_TYPE_Q8_0, true><<<grid, FSS_NT, 0, stream>>>(a);
        } else {
            fattn_sparse_sm60_d512<GGML_TYPE_Q8_0, false><<<grid, FSS_NT, 0, stream>>>(a);
        }
    } else {
        if (a.kv_same) {
            fattn_sparse_sm60_d512<GGML_TYPE_F16, true><<<grid, FSS_NT, 0, stream>>>(a);
        } else {
            fattn_sparse_sm60_d512<GGML_TYPE_F16, false><<<grid, FSS_NT, 0, stream>>>(a);
        }
    }
    if (a.n_split > 1) {
        fattn_sparse_sm60_join<<<dim3((unsigned) a.n_head, (unsigned) a.n_q, (unsigned) Q->ne[3]), FSS_D/4, 0, stream>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
}
