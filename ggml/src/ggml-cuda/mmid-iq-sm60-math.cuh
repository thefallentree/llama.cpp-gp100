// Per-lane math of the sm_60 IQ2_S / IQ3_XXS / IQ3_S / IQ2_XXS / IQ2_XS / IQ1_S / IQ1_M MUL_MAT_ID mat-vec (mmid-iq-sm60.cu), on the activation records of
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
// IQ2_XXS: 66 bytes = d, then per 32 weights two u32: 4 grid index bytes (iq2xxs_grid, 8 weights each) and 4 x 7-bit sign
// indices with the scale in bits 28..31; weight = d*0.125*(2s + 1)*grid*sign. sgn7: the 128-entry table (by ksigns_iq2xs).
IQ_DEV float iq2xxs_lane(const char * __restrict__ row, const int nsb, const int l, const char * __restrict__ x, const uint4 * __restrict__ sgn7,
        const uint64_t * __restrict__ grid) {
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
            const uint32_t qw  = iq_ld32(blk + 2 + 16*l + 8*kk);   // the 4 grid index bytes of ib32 2l + kk
            const uint32_t aux = iq_ld32(blk + 6 + 16*l + 8*kk);   // its sign indices and scale
            half2 acc = kq_h2(0u);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int e2 = 0; e2 < 2; ++e2) {
                    const int      j  = 2*h + e2;
                    const uint4    xa = ((const uint4 *) rec)[(2*k + e2)*4 + p0 + h];
                    const uint2    g  = IQ_LDG64(grid + ((qw >> (8*j)) & 0xFFu));
                    const uint4    m  = sgn7[(aux >> (7*j)) & 127u];
                    acc = iq_group8(acc, g.x, g.y, m, xa);
                }
            }
            P[kk] = __hfma2(acc, iq_int_h2(2u*(aux >> 28) + 1u), P[kk]);
        }
        sum = fmaf(iq_block_scale(rec, p0, d16, k0)*0.125f, iq_planes(P), sum);
        blk += 66;
        rec += KQ_REC_BYTES;
    }
    return sum;
}

// ---------------------------------------------------------------------------------------------------------------
// IQ2_XS: 74 bytes = d, qs[32] u16 (9-bit grid index into iq2xs_grid, 7-bit sign index), scales[8] (a nibble per 16
// weights); weight = d*0.125*(2s + 1)*grid*sign. sgn7: the 128-entry table.
IQ_DEV float iq2xs_lane(const char * __restrict__ row, const int nsb, const int l, const char * __restrict__ x, const uint4 * __restrict__ sgn7,
        const uint64_t * __restrict__ grid) {
    const int   p0 = 2*(l >> 1);
    const int   k0 = 2*(l & 1);
    const char * blk = row;
    const char * rec = x;
    float sum = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint32_t d16 = (uint32_t) *((const uint16_t *) blk);
        const uint32_t sc  = (uint32_t) *((const uint16_t *) (blk + 66 + 2*l));   // the scale bytes of ib32 2l, 2l + 1
        half2 P[2] = { kq_h2(0u), kq_h2(0u) };
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int      k   = k0 + kk;
            const uint32_t q01 = iq_ld32(blk + 2 + 16*l + 8*kk);   // groups 0, 1 of ib32 2l + kk
            const uint32_t q23 = iq_ld32(blk + 6 + 16*l + 8*kk);   // groups 2, 3
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                half2 acc = kq_h2(0u);
                const uint32_t qq = h == 0 ? q01 : q23;
#pragma unroll
                for (int e2 = 0; e2 < 2; ++e2) {
                    // group j = 2h + e2 of the ib32
                    const uint4    xa = ((const uint4 *) rec)[(2*k + e2)*4 + p0 + h];
                    const uint32_t q  = (qq >> (16*e2)) & 0xFFFFu;
                    const uint2    g  = IQ_LDG64(grid + (q & 511u));
                    const uint4    m  = sgn7[q >> 9];
                    acc = iq_group8(acc, g.x, g.y, m, xa);
                }
                const uint32_t s4 = (sc >> (8*kk + 4*h)) & 0xFu;
                P[kk] = __hfma2(acc, iq_int_h2(2u*s4 + 1u), P[kk]);
            }
        }
        sum = fmaf(iq_block_scale(rec, p0, d16, k0)*0.125f, iq_planes(P), sum);
        blk += 74;
        rec += KQ_REC_BYTES;
    }
    return sum;
}

// ---------------------------------------------------------------------------------------------------------------
// IQ1_S: 50 bytes = d, qs[32] (grid index low bytes, a byte per 8 weights), qh[8] u16 per 32 weights (3 index high bits per
// group of 8, the scale in bits 12..14, the delta's sign in bit 15); weight = d*(2s + 1)*(grid + delta), grid in {-1, 0, 1},
// delta = +-IQ1S_DELTA. iq1s_grid_gpu holds grid + 1 as nibbles (weights 0..3 in the low, 4..7 in the high nibbles of the
// bytes): shifted to the high nibble a byte is 16*(grid + 1) <= 32, a half subnormal like the other grids, no sign bits.
// The -1 + delta of the 32 weights of a scale group is one product with their activation sum (the record's X_k of the two
// halves), in float.
#ifndef IQ1S_DELTA
#define IQ1S_DELTA 0.125f
#endif
IQ_DEV float iq1s_lane(const char * __restrict__ row, const int nsb, const int l, const char * __restrict__ x,
        const uint32_t * __restrict__ grid) {
    const int   p0 = 2*(l >> 1);
    const int   k0 = 2*(l & 1);
    const char * blk = row;
    const char * rec = x;
    const uint4 m0 = make_uint4(0u, 0u, 0u, 0u);
    float sum = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const float     d   = __half22float2(kq_h2((uint32_t) *((const uint16_t *) blk))).x;
        const uint4     xp0 = ((const uint4 *) (rec + 512))[p0];
        const uint4     xp1 = ((const uint4 *) (rec + 512))[p0 + 1];
        float t = 0.0f;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int      k   = k0 + kk;
            const int      ib  = 2*l + kk;
            const uint32_t qw  = iq_ld32(blk + 2 + 4*ib);
            const uint32_t qh  = (uint32_t) *((const uint16_t *) (blk + 34 + 2*ib));
            half2 acc = kq_h2(0u);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int e2 = 0; e2 < 2; ++e2) {
                    const int      j  = 2*h + e2;
                    const uint4    xa = ((const uint4 *) rec)[(2*k + e2)*4 + p0 + h];
                    const uint32_t gw = IQ_LDG32(grid + (((qw >> (8*j)) & 0xFFu) | (((qh >> (3*j)) & 7u) << 8)));
                    acc = iq_group8(acc, (gw << 4) & 0xF0F0F0F0u, gw & 0xF0F0F0F0u, m0, xa);
                }
            }
            // a product is (16*(grid + 1)*2^-24)*(x*2^(15 - 2k)) = (grid + 1)*x*2^(-5 - 2k)
            const float2 a  = __half22float2(acc);
            const float  sv = (a.x + a.y)*(float) (32 << (2*k));
            // X_k of the halves h = 0, 1 (parts p0, p0 + 1): half (k >> 1) of word x (k even) or y (k odd)
            const uint32_t w0 = (k & 1) ? xp0.y : xp0.x;
            const uint32_t w1 = (k & 1) ? xp1.y : xp1.x;
            const float2 x0 = __half22float2(kq_h2(w0));
            const float2 x1 = __half22float2(kq_h2(w1));
            const float  X  = (k >> 1) ? x0.y + x1.y : x0.x + x1.x;
            const float  dl = (float) (2u*((qh >> 12) & 7u) + 1u);
            const float  dt = (qh & 0x8000u) ? -1.0f - IQ1S_DELTA : -1.0f + IQ1S_DELTA;
            t = fmaf(dl, fmaf(dt, X, sv), t);
        }
        sum = fmaf(d*__uint_as_float(xp0.z), t, sum);
        blk += 50;
        rec += KQ_REC_BYTES;
    }
    return sum;
}

// ---------------------------------------------------------------------------------------------------------------
// IQ1_M: 56 bytes = qs[32] (grid index low bytes, a byte per 8 weights), qh[16] (a nibble per 8 weights: 3 index high bits, the
// delta's sign in bit 3), scales[4] u16 (a 3-bit scale per 16 weights: sub-block i = word i >> 2, bits 3*(i & 3); the fp16
// super-scale d in the four top nibbles); weight = d*(2s + 1)*(grid + delta), grid in {-1, 0, 1} (iq1s_grid_gpu, as IQ1_S),
// delta = +-IQ1M_DELTA per 8 weights. The delta goes into the grid bytes: 16*(grid + 1), plus 32*IQ1M_DELTA where the sign bit
// is clear, is 16*(grid + delta) + 16*(1 + IQ1M_DELTA) <= 36, a half subnormal; the bias leaves as one product with the 16
// weights' activation sum (the record's X_k of the part), in float. One half2 accumulator per 16 weights (its scale).
#ifndef IQ1M_DELTA
#define IQ1M_DELTA 0.125f
#endif
IQ_DEV float iq1m_lane(const char * __restrict__ row, const int nsb, const int l, const char * __restrict__ x,
        const uint32_t * __restrict__ grid) {
    const int   p0 = 2*(l >> 1);
    const int   k0 = 2*(l & 1);
    const char * blk = row;
    const char * rec = x;
    const uint4    m0   = make_uint4(0u, 0u, 0u, 0u);
    const uint32_t dadd = (uint32_t) (32.0f*IQ1M_DELTA)*0x01010101u;
    const float    bias = 1.0f + IQ1M_DELTA;
    float sum = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint32_t s01 = iq_ld32(blk + 48);
        const uint32_t s23 = iq_ld32(blk + 52);
        const uint32_t d16 = ((s01 >> 12) & 0x000Fu) | ((s01 >> 24) & 0x00F0u) | ((s23 >> 4) & 0x0F00u) | ((s23 >> 16) & 0xF000u);
        const float    d   = __half22float2(kq_h2(d16)).x;
        const uint32_t scw = (uint32_t) *((const uint16_t *) (blk + 48 + 2*l));   // the scales of ib32 2l, 2l + 1
        const uint4    xp0 = ((const uint4 *) (rec + 512))[p0];
        const uint4    xp1 = ((const uint4 *) (rec + 512))[p0 + 1];
        float t = 0.0f;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int      k   = k0 + kk;
            const int      ib  = 2*l + kk;
            const uint32_t qw  = iq_ld32(blk + 4*ib);
            const uint32_t qh  = (uint32_t) *((const uint16_t *) (blk + 32 + 2*ib));
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                half2 acc = kq_h2(0u);
#pragma unroll
                for (int e2 = 0; e2 < 2; ++e2) {
                    const int      j  = 2*h + e2;
                    const uint4    xa = ((const uint4 *) rec)[(2*k + e2)*4 + p0 + h];
                    const uint32_t gw = IQ_LDG32(grid + (((qw >> (8*j)) & 0xFFu) | (((qh >> (4*j)) & 7u) << 8)));
                    const uint32_t dv = ((qh >> (4*j + 3)) & 1u) ? 0u : dadd;
                    acc = iq_group8(acc, ((gw << 4) & 0xF0F0F0F0u) | dv, (gw & 0xF0F0F0F0u) | dv, m0, xa);
                }
                // a product is (u*2^-24)*(x*2^(15 - 2k)), u = 16*(grid + delta + bias): the sum of (grid + delta + bias)*x
                const float2 a  = __half22float2(acc);
                const float  sv = (a.x + a.y)*(float) (32 << (2*k));
                // X_k of part p0 + h: half (k >> 1) of word x (k even) or y (k odd)
                const uint32_t wx = h ? ((k & 1) ? xp1.y : xp1.x) : ((k & 1) ? xp0.y : xp0.x);
                const float2   xx = __half22float2(kq_h2(wx));
                const float    X  = (k >> 1) ? xx.y : xx.x;
                const float    dl = (float) (2u*((scw >> (6*kk + 3*h)) & 7u) + 1u);
                t = fmaf(dl, fmaf(-bias, X, sv), t);
            }
        }
        sum = fmaf(d*__uint_as_float(xp0.z), t, sum);
        blk += 56;
        rec += KQ_REC_BYTES;
    }
    return sum;
}

// ---------------------------------------------------------------------------------------------------------------
// the mat-vec: a block is (rows, slot c = t*n_used + s) with KQ_NWARPS warps of 8 rows (4 lanes per row), as kq_vec_lane.
// IQ_T: 0 = IQ2_S, 1 = IQ3_XXS, 2 = IQ3_S, 3 = IQ2_XXS, 4 = IQ2_XS, 5 = IQ1_S, 6 = IQ1_M. False: the block has nothing to do (the expert
// is not in w: the host tier computes it).
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
          IQ_T == 2 ? iq3s_lane(wr, nsb, lane & 3, xr, sgn, (const uint32_t *) grid) :
          IQ_T == 3 ? iq2xxs_lane(wr, nsb, lane & 3, xr, sgn, (const uint64_t *) grid) :
          IQ_T == 4 ? iq2xs_lane(wr, nsb, lane & 3, xr, sgn, (const uint64_t *) grid) :
          IQ_T == 5 ? iq1s_lane(wr, nsb, lane & 3, xr, (const uint32_t *) grid) :
                      iq1m_lane(wr, nsb, lane & 3, xr, (const uint32_t *) grid);
    return true;
}
