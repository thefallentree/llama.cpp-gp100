// Per-lane math of the sm_60 Q6_K MUL_MAT mat-vec with one warp per row (mm-q6k-sm60.cu), on the activation records of
// mmid-kq-sm60-math.cuh. No includes: a CPU test compiles this file against an emulation of half2 and __byte_perm.
//
// The 4-lanes-per-row layout of the K-quant lanes streams 8 rows per warp with scattered 4-byte loads, which the trunk's big
// Q6_K matrices turn into L1 misses (the lane kernel was 1.4x slower than the generic mmvq). Here the 32 lanes of a warp load
// the 53 words of a super-block of ONE row contiguously (lane l: word l = its 4 ql bytes; lanes 0..20: word 32 + l = the qh,
// scale and d words), the qh and scale words a lane needs come from the lanes that loaded them (3 shuffles), and the lane
// multiplies its 4 weights of the two planes j0 = (l >> 3) & 1 and j0 + 2 (the two nibbles of its ql bytes) for every column.
//
// Lane l: n = l >> 4 (half of the super-block), i = l & 15, j0 = i >> 3, a' = i & 7 (ql word within the chunk), so its weights are
// y = 128n + 32j + 4a' + b (b = 0..3) for j in {j0, j0 + 2}: the record's sub-block (n, k = j, h = a' >> 2), word a = a' & 3.
// ql word l holds the nibble j >> 1 of each byte; its qh word is 8n + a' (bits 2j, 2j + 1); the scale of plane j is byte
// 8n + 2j + h of the 16 int8 scales (word 48 + 2n + (j >> 1), byte (2j + h) & 3). As in the K-quant lanes, the fields stay in
// place: the nibble of plane j at bit 4(j >> 1) (shifted up by 2 for odd j) and the high bits at bit 2j, joined by one HFMA2
// with 16, so that every plane's product with the record's 2^(15 - 2j) activations carries 2^-9. The -32 offset of Q6_K is
// 32*sc*X per 16-weight sub-block, of which each of its 4 lanes takes a quarter from the record's sub-block sums.

#pragma once

#include "mmid-kq-sm60-math.cuh"

#ifndef Q6_DEV
#define Q6_DEV static __device__ __forceinline__
#endif

#define Q6_WORDS 53   // 210 bytes of block + the half word of the next block (never used: the d word's upper half)

// the words a lane needs of one super-block, as loaded and shuffled by the warp (CPU test: picked from the 53 words)
struct q6_lane_words {
    uint32_t ql;    // word l: 4 ql bytes
    uint32_t qh;    // word 32 + 8n + a'
    uint32_t sc;    // word 48 + 2n: the scales of the planes 0, 1 (bytes 2j + h)
    uint32_t sc2;   // word 49 + 2n: the scales of the planes 2, 3
    uint32_t d16;   // bytes 208..209 (word 52, low half)
};

// the lane's words from the 53 words of a block (CPU emulation of the loads + shuffles)
Q6_DEV void q6_pick_words(const uint32_t * w, const int lane, q6_lane_words & o) {
    const int n = lane >> 4, i = lane & 15, ap = i & 7;
    o.ql  = w[lane];
    o.qh  = w[32 + 8*n + ap];
    o.sc  = w[48 + 2*n];
    o.sc2 = w[49 + 2*n];
    o.d16 = w[52] & 0xFFFFu;
}

// the lane's partial dot product of one super-block for one column: xw0, xw1: the record's uint2 (pairs (b0, b2), (b1, b3)) of
// the sub-blocks (n, j0, h) and (n, j0 + 2, h) at word a, xx: the half2 (X_j0, X_j0+2) of those sub-blocks
Q6_DEV float q6_lane_block(const q6_lane_words & w, const int lane, const uint2 xw0, const uint2 xw1, const uint32_t xx) {
    const int      i  = lane & 15;
    const int      j0 = i >> 3;
    const int      h  = (i & 7) >> 2;
    const half2    f16 = kq_h2(0x4C004C00u); // 16
    const uint32_t cw = w.ql, cs = w.ql >> 8;
    const uint32_t gw = w.qh, gs = w.qh >> 8;
    // plane j0: nibble 0 (bits 0..3, shifted up by 2 for j0 = 1), high bits at 2*j0; plane j0 + 2: nibble 1 (bits 4..7, + 2), bits 2*j0 + 4
    const int      sh  = 2*j0;
    const uint32_t mq0 = 0x000F000Fu << sh, mq1 = 0x00F000F0u << sh;
    const uint32_t mh0 = 0x00030003u << sh, mh1 = 0x00300030u << sh;
    half2 acc0 = kq_h2(0u), acc1 = kq_h2(0u);
    acc0 = __hfma2(__hfma2(kq_h2(gw & mh0), f16, kq_h2((cw << sh) & mq0)), kq_h2(xw0.x), acc0);
    acc0 = __hfma2(__hfma2(kq_h2(gs & mh0), f16, kq_h2((cs << sh) & mq0)), kq_h2(xw0.y), acc0);
    acc1 = __hfma2(__hfma2(kq_h2(gw & mh1), f16, kq_h2((cw << sh) & mq1)), kq_h2(xw1.x), acc1);
    acc1 = __hfma2(__hfma2(kq_h2(gs & mh1), f16, kq_h2((cs << sh) & mq1)), kq_h2(xw1.y), acc1);
    // the int8 scales of the planes j0 and j0 + 2: bytes (2*j0 + h) of w.sc and w.sc2; biased by 128 into exact half2 integers
    const uint32_t sel = (uint32_t) (2*j0 + h) | ((uint32_t) (4 + 2*j0 + h) << 8);   // byte 0 <- sc byte, byte 2 <- sc2 byte (1, 3: junk)
    const uint32_t v2  = ((__byte_perm(w.sc, w.sc2, sel) & 0x00FF00FFu) ^ 0x00800080u) | 0x64006400u;   // 1024 + (sc + 128) per half
    const half2 scv = __hsub2(kq_h2(v2), kq_h2(0x64806480u));
    // (sum_j0, sum_j0+2) of the products, times the scales, minus 8*X*sc (a quarter of the sub-block's 32*X*sc)
    const half2  P  = __hmul2(kq_fold2(acc0, acc1), scv);
    const half2  M  = __hmul2(kq_h2(xx), scv);
    const float2 pf = __half22float2(P);
    const float2 mf = __half22float2(M);
    const float  d  = __half22float2(kq_h2(w.d16)).x;
    return d*fmaf((float) (1 << KQ_SCALE_SHIFT), pf.x + pf.y, -8.0f*(mf.x + mf.y));
}

// where the lane's activation words sit in a record (uint2 index of the pairs of plane j at word a, half index of X_j)
Q6_DEV int q6_rec_xw_index(const int lane, const int j) {
    const int n = lane >> 4, i = lane & 15, ap = i & 7, h = ap >> 2, a = ap & 3;
    return ((2*j + (a >> 1))*4 + 2*n + h)*2 + (a & 1);
}

Q6_DEV int q6_rec_xx_index(const int lane, const int j) {
    const int n = lane >> 4, i = lane & 15, ap = i & 7, h = ap >> 2;
    return 256 + (2*n + h)*8 + 2*(j & 1) + (j >> 1);
}
