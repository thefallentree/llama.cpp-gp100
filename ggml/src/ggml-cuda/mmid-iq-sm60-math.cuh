// Per-lane math of the sm_60 IQ2_S / IQ3_XXS / IQ3_S MUL_MAT_ID mat-vec (mmid-iq-sm60.cu), on the activation records of
// mmid-kq-sm60-math.cuh (the same converted rows as the Q2_K / Q3_K lanes, shared between gate and up through the cache).
// No includes: a CPU test compiles this file against an emulation of half2, uint4, __byte_perm and __funnelshift_r.
//
// The i-quants are codebook types: every 8 (IQ2_S) or 4 (IQ3_*) weights are a grid entry of small odd magnitudes (one byte
// each, at most 62) with one sign bit per weight. A grid word's bytes (b0, b2) and (b1, b3) masked with 0x003F003F are the
// half subnormals b*2^-24 of two weights; the sign bits come from a 256-entry table of half2 sign masks (by the sign byte of
// a group of 8), OR'ed into the same LOP3. The activation record holds the pairs (4a, 4a+2), (4a+1, 4a+3) of every 16-weight
// sub-block (n, k, h) scaled by 2^(15 - 2k), so the HFMA2 products are b*x*2^(-9-2k): one half2 accumulator per scale group
// (16 weights in IQ2_S, 32 in IQ3_*), one HFMA2 with the exact half2 integer (2s + 1) into the plane's half2 sum, then the
// two planes of the lane go to float with their 4^k.
//
// A lane handles 64 consecutive weights of its row per super-block: lane l = ib32 groups 2l and 2l+1, i.e. n = l >> 1 and the
// planes k = 2(l & 1), 2(l & 1) + 1 of the record (parts p = 2n + h). The 4 lanes of a row read the block's bytes once.

#pragma once

#include "mmid-kq-sm60-math.cuh"

#ifndef IQ_DEV
#define IQ_DEV static __device__ __forceinline__
#endif

// the grid tables are __device__ constants: read through the read-only path on the GPU, plain on the CPU test
#ifndef IQ_LDG32
#define IQ_LDG32(p) __ldg(p)
#define IQ_LDG64(p) __ldg((const uint2 *) (p))
#endif

#define IQ_FIELD 0x003F003Fu   // the grid bytes are at most 62

// a 4-byte field of a block at a 2-byte aligned address, without reading past it when it is aligned
IQ_DEV uint32_t iq_ld32(const char * p) {
    const uint32_t * a  = (const uint32_t *) ((uintptr_t) p & ~(uintptr_t) 3);
    const uint32_t   sh = ((uint32_t) (uintptr_t) p & 2u) << 3;   // 0 or 16, the same for the warp
    const uint32_t   lo = a[0];
    const uint32_t   hi = sh != 0 ? a[1] : 0u;
    return __funnelshift_r(lo, hi, sh);
}

// the half2 sign masks of the pairs (0, 2), (1, 3), (4, 6), (5, 7) of a group of 8 weights, by its sign byte (bit j = weight j negative)
IQ_DEV uint4 iq_sign_masks(const uint32_t s) {
    return make_uint4(((s & 0x01u) << 15) | ((s & 0x04u) << 29), ((s & 0x02u) << 14) | ((s & 0x08u) << 28),
                      ((s & 0x10u) << 11) | ((s & 0x40u) << 25), ((s & 0x20u) << 10) | ((s & 0x80u) << 24));
}

// the 4 HFMA2 of a group of 8 weights: grid words g0 (weights 0..3) and g1 (4..7), their sign masks m, the activation uint4 xa of
// the record (pairs (0, 2), (1, 3), (4, 6), (5, 7))
IQ_DEV half2 iq_group8(half2 acc, const uint32_t g0, const uint32_t g1, const uint4 m, const uint4 xa) {
    acc = __hfma2(kq_h2((g0 & IQ_FIELD) | m.x),        kq_h2(xa.x), acc);
    acc = __hfma2(kq_h2(((g0 >> 8) & IQ_FIELD) | m.y), kq_h2(xa.y), acc);
    acc = __hfma2(kq_h2((g1 & IQ_FIELD) | m.z),        kq_h2(xa.z), acc);
    acc = __hfma2(kq_h2(((g1 >> 8) & IQ_FIELD) | m.w), kq_h2(xa.w), acc);
    return acc;
}

// the exact half2 (v, v) of an integer v < 2048
IQ_DEV half2 iq_int_h2(const uint32_t v) {
    return __hsub2(kq_h2(__byte_perm(v | 0x6400u, 0u, 0x1010)), kq_h2(0x64006400u));
}

// the two planes k0, k0 + 1 of a lane's super-block as one float: the products carry 2^(-9-2k), so plane k0 + 1 counts 4x
IQ_DEV float iq_planes(const half2 (&P)[2]) {
    const float2 p0 = __half22float2(P[0]);
    const float2 p1 = __half22float2(P[1]);
    return fmaf(4.0f, p1.x + p1.y, p0.x + p0.y);
}

// d * the record's block scale * 2^9 * 4^k0 (2^9 undoes the products' 2^-9, 4^k0 the plane k0's 2^-2k0)
IQ_DEV float iq_block_scale(const char * __restrict__ rec, const int p0, const uint32_t d16, const int k0) {
    const uint4 xp = ((const uint4 *) (rec + 512))[p0];
    const float d  = __half22float2(kq_h2(d16)).x;
    return d*__uint_as_float(xp.z)*(float) ((1 << KQ_SCALE_SHIFT) << (2*k0));
}

// ---------------------------------------------------------------------------------------------------------------
// IQ2_S: 82 bytes = d, qs[32] grid index low bytes, qs[32] sign bytes, qh[8] (2 index high bits per group of 8), scales[8]
// (a nibble per 16 weights); weight = d*0.125*(2s + 1)*grid*sign. sgn: the 256-entry table of iq_sign_masks.
IQ_DEV float iq2s_lane(const char * __restrict__ row, const int nsb, const int l, const char * __restrict__ x, const uint4 * __restrict__ sgn,
        const uint64_t * __restrict__ grid) {
    const int   p0 = 2*(l >> 1);
    const int   k0 = 2*(l & 1);
    const char * blk = row;
    const char * rec = x;
    float sum = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint32_t d16 = (uint32_t) *((const uint16_t *) blk);
        const uint32_t qh  = (uint32_t) *((const uint16_t *) (blk + 66 + 2*l));
        const uint32_t sc  = (uint32_t) *((const uint16_t *) (blk + 74 + 2*l));
        half2 P[2] = { kq_h2(0u), kq_h2(0u) };
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int      k   = k0 + kk;
            const uint32_t qw  = iq_ld32(blk + 2 + 8*l + 4*kk);    // the 4 grid index bytes of ib32 2l + kk
            const uint32_t sw  = iq_ld32(blk + 34 + 8*l + 4*kk);   // its 4 sign bytes
            const uint32_t qhb = qh >> (8*kk);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                half2 acc = kq_h2(0u);
#pragma unroll
                for (int e2 = 0; e2 < 2; ++e2) {
                    const int      j   = 2*h + e2;
                    const uint4    xa  = ((const uint4 *) rec)[(2*k + e2)*4 + p0 + h];
                    const uint32_t idx = ((qw >> (8*j)) & 0xFFu) | (((qhb >> (2*j)) & 3u) << 8);
                    const uint2    g   = IQ_LDG64(grid + idx);
                    const uint4    m   = sgn[(sw >> (8*j)) & 0xFFu];
                    acc = iq_group8(acc, g.x, g.y, m, xa);
                }
                const uint32_t s4 = (sc >> (8*kk + 4*h)) & 0xFu;
                P[kk] = __hfma2(acc, iq_int_h2(2u*s4 + 1u), P[kk]);
            }
        }
        sum = fmaf(iq_block_scale(rec, p0, d16, k0)*0.125f, iq_planes(P), sum);
        blk += 82;
        rec += KQ_REC_BYTES;
    }
    return sum;
}

// ---------------------------------------------------------------------------------------------------------------
// IQ3_XXS: 98 bytes = d, qs[64] grid index bytes (a byte per 4 weights), 8 x u32 (4 x 7-bit sign index of the groups of 8,
// the scale in bits 28..31); weight = d*0.25*(2s + 1)*grid*sign. sgn7: the 128-entry table of iq_sign_masks(ksigns_iq2xs[i]).
IQ_DEV float iq3xxs_lane(const char * __restrict__ row, const int nsb, const int l, const char * __restrict__ x, const uint4 * __restrict__ sgn7,
        const uint32_t * __restrict__ grid) {
    const int   p0 = 2*(l >> 1);
    const int   k0 = 2*(l & 1);
    const char * blk = row;
    const char * rec = x;
    float sum = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint32_t d16 = (uint32_t) *((const uint16_t *) blk);
        half2 P[2] = { kq_h2(0u), kq_h2(0u) };
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int      k   = k0 + kk;
            const uint32_t qa  = iq_ld32(blk + 2 + 16*l + 8*kk);
            const uint32_t qb  = iq_ld32(blk + 6 + 16*l + 8*kk);
            const uint32_t aux = iq_ld32(blk + 66 + 8*l + 4*kk);
            half2 acc = kq_h2(0u);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int e2 = 0; e2 < 2; ++e2) {
                    const int      j  = 2*h + e2;
                    const uint4    xa = ((const uint4 *) rec)[(2*k + e2)*4 + p0 + h];
                    const uint32_t w  = j < 2 ? qa : qb;
                    const uint32_t g0 = IQ_LDG32(grid + ((w >> (16*(j & 1))) & 0xFFu));
                    const uint32_t g1 = IQ_LDG32(grid + ((w >> (16*(j & 1) + 8)) & 0xFFu));
                    const uint4    m  = sgn7[(aux >> (7*j)) & 127u];
                    acc = iq_group8(acc, g0, g1, m, xa);
                }
            }
            P[kk] = __hfma2(acc, iq_int_h2(2u*(aux >> 28) + 1u), P[kk]);
        }
        sum = fmaf(iq_block_scale(rec, p0, d16, k0)*0.25f, iq_planes(P), sum);
        blk += 98;
        rec += KQ_REC_BYTES;
    }
    return sum;
}

// ---------------------------------------------------------------------------------------------------------------
// IQ3_S: 110 bytes = d, qs[64] grid index bytes, qh[8] (the 9th index bit of the 8 groups of 4 of an ib32), signs[32] (a byte
// per group of 8), scales[4] (a nibble per 32 weights); weight = d*(2s + 1)*grid*sign. sgn: the 256-entry table.
IQ_DEV float iq3s_lane(const char * __restrict__ row, const int nsb, const int l, const char * __restrict__ x, const uint4 * __restrict__ sgn,
        const uint32_t * __restrict__ grid) {
    const int   p0 = 2*(l >> 1);
    const int   k0 = 2*(l & 1);
    const char * blk = row;
    const char * rec = x;
    float sum = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint32_t d16 = (uint32_t) *((const uint16_t *) blk);
        const uint32_t qh2 = (uint32_t) *((const uint16_t *) (blk + 66 + 2*l));
        const uint32_t scb = (uint32_t) *((const uint8_t *) (blk + 106 + l));
        half2 P[2] = { kq_h2(0u), kq_h2(0u) };
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int      k   = k0 + kk;
            const uint32_t qa  = iq_ld32(blk + 2 + 16*l + 8*kk);
            const uint32_t qb  = iq_ld32(blk + 6 + 16*l + 8*kk);
            const uint32_t sw  = iq_ld32(blk + 74 + 8*l + 4*kk);
            const uint32_t qhb = qh2 >> (8*kk);
            half2 acc = kq_h2(0u);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int e2 = 0; e2 < 2; ++e2) {
                    const int      j  = 2*h + e2;
                    const uint4    xa = ((const uint4 *) rec)[(2*k + e2)*4 + p0 + h];
                    const uint32_t w  = j < 2 ? qa : qb;
                    const uint32_t g0 = IQ_LDG32(grid + (((w >> (16*(j & 1))) & 0xFFu)     | (((qhb >> (2*j)) & 1u) << 8)));
                    const uint32_t g1 = IQ_LDG32(grid + (((w >> (16*(j & 1) + 8)) & 0xFFu) | (((qhb >> (2*j + 1)) & 1u) << 8)));
                    const uint4    m  = sgn[(sw >> (8*j)) & 0xFFu];
                    acc = iq_group8(acc, g0, g1, m, xa);
                }
            }
            P[kk] = __hfma2(acc, iq_int_h2(2u*((scb >> (4*kk)) & 0xFu) + 1u), P[kk]);
        }
        sum = fmaf(iq_block_scale(rec, p0, d16, k0), iq_planes(P), sum);
        blk += 110;
        rec += KQ_REC_BYTES;
    }
    return sum;
}

// ---------------------------------------------------------------------------------------------------------------
// the mat-vec: a block is (rows, slot c = t*n_used + s) with KQ_NWARPS warps of 8 rows (4 lanes per row), as kq_vec_lane.
// IQ_T: 0 = IQ2_S, 1 = IQ3_XXS, 2 = IQ3_S. False: the block has nothing to do (the expert is not in w: the host tier computes it).
template <int IQ_T>
IQ_DEV bool iq_vec_lane(const char * __restrict__ w, const char * __restrict__ x, const int32_t * __restrict__ ids,
        const int bx, const int c, const int warp, const int lane, const int nsb, const int ne01, const int64_t nb01, const int64_t nb02,
        const int ne11, const int n_used, const int si1, const int n_hot, const uint4 * __restrict__ sgn, const void * __restrict__ grid,
        float & sum, int & t, int & s, int & row0) {
    t = c / n_used;
    s = c % n_used;
    const int e = ids[t*si1 + s];
    if (e < 0 || e >= n_hot) {
        return false;
    }
    const int r = t*ne11 + s % ne11;
    row0 = bx*KQ_ROWS + warp*8 + (lane >> 2);
    const int row = row0 < ne01 ? row0 : ne01 - 1; // rows past the end read the last row and do not store
    const char * wr = w + (int64_t) e*nb02 + (int64_t) row*nb01;
    const char * xr = x + (int64_t) r*nsb*KQ_REC_BYTES;
    sum = IQ_T == 0 ? iq2s_lane(wr, nsb, lane & 3, xr, sgn, (const uint64_t *) grid) :
          IQ_T == 1 ? iq3xxs_lane(wr, nsb, lane & 3, xr, sgn, (const uint32_t *) grid) :
                      iq3s_lane(wr, nsb, lane & 3, xr, sgn, (const uint32_t *) grid);
    return true;
}
