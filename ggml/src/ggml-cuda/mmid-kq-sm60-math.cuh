// Per-lane math of the sm_60 Q2_K / Q3_K MUL_MAT_ID mat-vec (mmid-kq-sm60.cu).
// No includes: a CPU test compiles this file against an emulation of half2, uint4, __byte_perm and __funnelshift_r.
//
// The activations of a row (K = 256*nsb values) are converted once per op (mmid_kq_convert_rows) into one record of
// KQ_REC_BYTES per super-block (the 4 parts p = 2*n + h of the two halves n of a super-block, h = 16-value half of a plane):
//   uint4 xh[8][4]  8 groups g = 2*k + e2 (k = plane, e2 = word pair), then the part: the activation half2 of the weight decode
//   uint4 xp[4]     per part: half2 (X0, X2), half2 (X1, X3), float xs, unused; X_k = sum of the 16 values of sub-block (n, k, h)
// The values are divided by the block scale xs (the max of the 256 values). A super-block value y (0..255) has n = y >> 7,
// k = (y >> 5) & 3, h = (y >> 4) & 1, a = (y >> 2) & 3, b = y & 3. It is a half in uint4 xh[2*k + (a >> 1)][p], word
// 2*(a & 1) + (b & 1), half (b >> 1), scaled by 2^(15 - 2k). So the two weights of a half2 of the decode (bytes b0, b2 or
// b1, b3 of a word) meet their two activations in one register.
//
// A part is a 16-byte chunk of the qs bytes of one half n of the super-block (Q2_K: 4 weight planes of 2 bits, one sub-block
// of 16 weights per plane). One lane handles one part of one row. A 2-bit field c at bit 2k of a 16-bit lane is a half
// subnormal c*2^(2k-24), so a LOP3 mask is the whole decode: the products carry c*x*2^-9 and no subtraction is needed.

#pragma once

#ifndef KQ_DEV
#define KQ_DEV static __device__ __forceinline__
#endif

#define KQ_SCALE_SHIFT 9 // the half2 partial sums are scaled by 2^-KQ_SCALE_SHIFT

KQ_DEV half2 kq_h2(const uint32_t v) {
    return *((const half2 *) &v);
}

KQ_DEV uint32_t kq_u32(const half2 v) {
    return *((const uint32_t *) &v);
}

// the sums (a.lo + a.hi) of two half2 pairs as one half2: (a0.lo + a0.hi, a1.lo + a1.hi)
KQ_DEV half2 kq_fold2(const half2 a0, const half2 a1) {
    return __hadd2(kq_h2(__byte_perm(kq_u32(a0), kq_u32(a1), 0x5410)), kq_h2(__byte_perm(kq_u32(a0), kq_u32(a1), 0x7632)));
}

#define KQ_REC_BYTES 576

// The converter (mmid_kq_convert_rows) works on a warp per super-block: lane l holds the float4 at y = 128q + 4l, q = 0, 1,
// so n = q, k = l >> 3, h = (l >> 2) & 1, a = l & 3 (the four lanes of a sub-block are consecutive).
// The two half2 pairs (b0, b2), (b1, b3) of the float4, divided by the block scale (inv) and scaled by 2^(15 - 2k)
KQ_DEV uint2 kq_conv_pairs(const float4 v, const float inv, const int lane) {
    const float f = inv*__int_as_float((127 + 15 - 2*(lane >> 3)) << 23);
    return make_uint2(kq_u32(__floats2half2_rn(v.x*f, v.z*f)), kq_u32(__floats2half2_rn(v.y*f, v.w*f)));
}

// index of that uint2 in the uint2 view of the converted row
KQ_DEV int kq_conv_xh_index(const int sb, const int q, const int lane) {
    const int k = lane >> 3, h = (lane >> 2) & 1, a = lane & 3;
    return sb*(KQ_REC_BYTES/8) + ((2*k + (a >> 1))*4 + 2*q + h)*2 + (a & 1);
}

// index of the half X_k of the sub-block of that lane in the half view of the converted row, in the order (X0, X2, X1, X3) of a part
KQ_DEV int kq_conv_xx_index(const int sb, const int q, const int lane) {
    const int k = lane >> 3, h = (lane >> 2) & 1;
    return sb*(KQ_REC_BYTES/2) + 256 + (2*q + h)*8 + 2*(k & 1) + (k >> 1);
}

// index of the float xs of part p in the float view of the converted row
KQ_DEV int kq_conv_xs_index(const int sb, const int part) {
    return sb*(KQ_REC_BYTES/4) + 128 + 4*part + 2;
}

// ---------------------------------------------------------------------------------------------------------------
// Q2_K: super-block 84 bytes = scales[16] (4 bit scale, 4 bit min), qs[64], half d, half dmin; weight = d*sc*q - dmin*m

// the contribution of part p of one super-block to the dot product, without the block scale xs
// c: the 4 words of the part's qs bytes, s0, s1: scale bytes 8n..8n+7, dm: (d, dmin), xa: the 8 activation groups of the part,
// xx: its (X0, X2), (X1, X3), sel: 0x6420 + 0x1111*h gathers the 4 scale bytes of the part
KQ_DEV float kq_q2k_chunk(const uint32_t (&c)[4], const uint32_t s0, const uint32_t s1, const uint32_t dm,
        const uint4 (&xa)[8], const uint2 xx, const uint32_t sel) {
    half2 acc[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        acc[k] = kq_h2(0u);
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const uint32_t cw = c[a];
        const uint32_t cs = cw >> 8;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t m  = 0x00030003u << (2*k);
            const uint4    xv = xa[2*k + (a >> 1)];
            const uint32_t xe = (a & 1) ? xv.z : xv.x;
            const uint32_t xo = (a & 1) ? xv.w : xv.y;
            acc[k] = __hfma2(kq_h2(cw & m), kq_h2(xe), acc[k]);
            acc[k] = __hfma2(kq_h2(cs & m), kq_h2(xo), acc[k]);
        }
    }

    // scale and min bytes of the planes 0..3 in bytes 0..3; pairs (0, 2) and (1, 3) as exact half2 integers
    const uint32_t v4  = __byte_perm(s0, s1, sel);
    const uint32_t v4s = v4 >> 8;
    const half2 b1024 = kq_h2(0x64006400u);
    const half2 b64   = kq_h2(0x54005400u);
    const half2 sc02  = __hsub2(kq_h2((v4  & 0x000F000Fu) | 0x64006400u), b1024);
    const half2 sc13  = __hsub2(kq_h2((v4s & 0x000F000Fu) | 0x64006400u), b1024);
    const half2 m02   = __hsub2(kq_h2((v4  & 0x00F000F0u) | 0x54005400u), b64);
    const half2 m13   = __hsub2(kq_h2((v4s & 0x00F000F0u) | 0x54005400u), b64);

    const half2 P = __hfma2(kq_fold2(acc[0], acc[2]), sc02, __hmul2(kq_fold2(acc[1], acc[3]), sc13));
    const half2 M = __hfma2(kq_h2(xx.x), m02, __hmul2(kq_h2(xx.y), m13));
    const float2 pm  = __half22float2(__hadd2(kq_h2(__byte_perm(kq_u32(P), kq_u32(M), 0x5410)), kq_h2(__byte_perm(kq_u32(P), kq_u32(M), 0x7632))));
    const float2 dmf = __half22float2(kq_h2(dm));
    return fmaf(dmf.x*(float) (1 << KQ_SCALE_SHIFT), pm.x, -dmf.y*pm.y);
}

// part p of every super-block of one row; row points at the row's first block (4-byte aligned), x at the converted row
KQ_DEV float kq_q2k_lane(const char * __restrict__ row, const int nsb, const int part, const char * __restrict__ x) {
    const int      n   = part >> 1;
    const uint32_t sel = 0x6420u + 0x1111u*(uint32_t) (part & 1);
    const char *   blk = row;
    const char *   rec = x;
    float sum = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint32_t * qp = (const uint32_t *) (blk + 16 + 16*part);
        const uint32_t c[4] = { qp[0], qp[1], qp[2], qp[3] };
        const uint32_t s0 = *((const uint32_t *) (blk + 8*n));
        const uint32_t s1 = *((const uint32_t *) (blk + 8*n + 4));
        const uint32_t dm = *((const uint32_t *) (blk + 80));
        uint4 xa[8];
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            xa[g] = ((const uint4 *) rec)[g*4 + part];
        }
        const uint4 xp = ((const uint4 *) (rec + 512))[part];
        sum = fmaf(__uint_as_float(xp.z), kq_q2k_chunk(c, s0, s1, dm, xa, make_uint2(xp.x, xp.y), sel), sum);
        blk += 84;
        rec += KQ_REC_BYTES;
    }
    return sum;
}

// ---------------------------------------------------------------------------------------------------------------
// Q3_K: super-block 110 bytes (2-byte aligned) = hmask[32], qs[64], scales[12] (6 bit), half d; weight = d*(sc - 32)*(q + 4*h - 4)
// weight y = 128n + 32j + l of the super-block: q = bits 2j of qs[32n + l], h = bit 4n + j of hmask[l], l = 16h' + 4a + b

// c: qs words of the part, g: its hmask words (hmask bytes 16h'.., shifted right by 4n: the bit j of a byte is the high bit of plane j),
// sw0..sw2: scale bytes 0..11, dbits: d, sel_lo = 0x6420 + 0x1111*h', sel_hi = 0x2020 + 0x1111*h'
KQ_DEV float kq_q3k_chunk(const uint32_t (&c)[4], const uint32_t (&g)[4], const uint32_t sw0, const uint32_t sw1, const uint32_t sw2,
        const uint32_t dbits, const uint4 (&xa)[8], const uint2 xx, const int n, const uint32_t sel_lo, const uint32_t sel_hi) {
    // u = q + 4h in a subnormal: (q field at bit 2j, 2^(2j-24) per unit) + 4*(h bit at bit j, 2^(j-24) per unit)*2^(j+2) -> u*2^(2j-24)
    const half2 f0 = kq_h2(0x44004400u); // 4
    const half2 f1 = kq_h2(0x48004800u); // 8
    const half2 f2 = kq_h2(0x4C004C00u); // 16
    const half2 f3 = kq_h2(0x50005000u); // 32
    half2 acc[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        acc[k] = kq_h2(0u);
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const uint32_t cw = c[a];
        const uint32_t cs = cw >> 8;
        const uint32_t gw = g[a];
        const uint32_t gs = gw >> 8;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const half2    f  = k == 0 ? f0 : k == 1 ? f1 : k == 2 ? f2 : f3;
            const uint32_t mq = 0x00030003u << (2*k);
            const uint32_t mh = 0x00010001u << k;
            const uint4    xv = xa[2*k + (a >> 1)];
            const uint32_t xe = (a & 1) ? xv.z : xv.x;
            const uint32_t xo = (a & 1) ? xv.w : xv.y;
            const half2 we = __hfma2(kq_h2(gw & mh), f, kq_h2(cw & mq));
            const half2 wo = __hfma2(kq_h2(gs & mh), f, kq_h2(cs & mq));
            acc[k] = __hfma2(we, kq_h2(xe), acc[k]);
            acc[k] = __hfma2(wo, kq_h2(xo), acc[k]);
        }
    }

    // scales of the planes 0..3 of the part: low 4 bits from bytes 2j + h' (nibble n), high 2 bits from bytes 8 + h', 10 + h'
    const uint32_t A  = __byte_perm(sw0, sw1, sel_lo);
    const uint32_t H  = __byte_perm(sw2, 0u, sel_hi);
    const uint32_t lo = (A >> (4*n)) & 0x0F0F0F0Fu;
    const uint32_t hi = ((H >> (4*n)) & 0x00000303u) | ((H >> (4*n + 2)) & 0x03030000u);
    const uint32_t us = lo | (hi << 4);
    const half2 b1056 = kq_h2(0x64206420u); // 1024 + 32
    const half2 sc02  = __hsub2(kq_h2((us      & 0x003F003Fu) | 0x64006400u), b1056);
    const half2 sc13  = __hsub2(kq_h2(((us >> 8) & 0x003F003Fu) | 0x64006400u), b1056);

    const half2 P = __hfma2(kq_fold2(acc[0], acc[2]), sc02, __hmul2(kq_fold2(acc[1], acc[3]), sc13));
    const half2 M = __hfma2(kq_h2(xx.x), sc02, __hmul2(kq_h2(xx.y), sc13));
    const float2 pm = __half22float2(__hadd2(kq_h2(__byte_perm(kq_u32(P), kq_u32(M), 0x5410)), kq_h2(__byte_perm(kq_u32(P), kq_u32(M), 0x7632))));
    const float2 df = __half22float2(kq_h2(dbits));
    return df.x*fmaf((float) (1 << KQ_SCALE_SHIFT), pm.x, -4.0f*pm.y);
}

// part p of every super-block of one row; row points at the row's first block (2-byte aligned), x at the converted row
KQ_DEV float kq_q3k_lane(const char * __restrict__ row, const int nsb, const int part, const char * __restrict__ x) {
    const int      n       = part >> 1;
    const int      h       = part & 1;
    const uint32_t sel_lo  = 0x6420u + 0x1111u*(uint32_t) h;
    const uint32_t sel_hi  = 0x2020u + 0x1111u*(uint32_t) h;
    const char *   blk = row;
    const char *   rec = x;
    float sum = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint32_t mis = (uint32_t) (uintptr_t) blk & 3; // the block starts 0 or 2 bytes into an aligned word
        const uint32_t * pb = (const uint32_t *) (blk - mis);
        const uint32_t   sh = 8*mis;

        uint32_t c[4], g[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            c[i] = __funnelshift_r(pb[8 + 4*part + i], pb[8 + 4*part + i + 1], sh);
            g[i] = __funnelshift_r(pb[4*h + i],        pb[4*h + i + 1],        sh + 4*n);
        }
        const uint32_t sw0 = __funnelshift_r(pb[24], pb[25], sh);
        const uint32_t sw1 = __funnelshift_r(pb[25], pb[26], sh);
        const uint32_t sw2 = __funnelshift_r(pb[26], pb[26 + (sh >> 4)], sh); // the word 27 only belongs to the block for sh = 16
        const uint32_t dbits = (uint32_t) *((const uint16_t *) (blk + 108));

        uint4 xa[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            xa[q] = ((const uint4 *) rec)[q*4 + part];
        }
        const uint4 xp = ((const uint4 *) (rec + 512))[part];
        sum = fmaf(__uint_as_float(xp.z), kq_q3k_chunk(c, g, sw0, sw1, sw2, dbits, xa, make_uint2(xp.x, xp.y), n, sel_lo, sel_hi), sum);
        blk += 110;
        rec += KQ_REC_BYTES;
    }
    return sum;
}

// ---------------------------------------------------------------------------------------------------------------
// the mat-vec: a block is (rows, slot c = t*n_used + s) with KQ_NWARPS warps of 8 rows (4 lanes per row)

#define KQ_NWARPS 4
#define KQ_ROWS   (8*KQ_NWARPS)

// The work of one lane of block (bx, c): its part of its row of expert ids[t*si1 + s]. False: the block has nothing to do
// (the expert is not in w: the host tier computes it). Then sum is this lane's part of the row row0 of dst[., s, t].
template <bool q3>
KQ_DEV bool kq_vec_lane(const char * __restrict__ w, const char * __restrict__ x, const int32_t * __restrict__ ids,
        const int bx, const int c, const int warp, const int lane, const int nsb, const int ne01, const int64_t nb01, const int64_t nb02,
        const int ne11, const int n_used, const int si1, const int n_hot, float & sum, int & t, int & s, int & row0) {
    t = c / n_used;
    s = c % n_used;
    const int e = ids[t*si1 + s];
    if (e < 0 || e >= n_hot) {
        return false;
    }
    const int r    = t*ne11 + s % ne11;
    const int part = lane & 3;
    row0 = bx*KQ_ROWS + warp*8 + (lane >> 2);
    const int row = row0 < ne01 ? row0 : ne01 - 1; // rows past the end read the last row and do not store
    const char * wr = w + (int64_t) e*nb02 + (int64_t) row*nb01;
    const char * xr = x + (int64_t) r*nsb*KQ_REC_BYTES;
    sum = q3 ? kq_q3k_lane(wr, nsb, part, xr) : kq_q2k_lane(wr, nsb, part, xr);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// MUL_MAT (no ids): a block is (column c, row tile, weight matrix i02) with KQ_NWARPS warps of 8 rows. The ncols = ne11*r2 columns of
// a matrix are the src1 rows i1 of the r2 channels i2 = i02*r2 + i2r that broadcast it (c = i2r*ne11 + i1). They are the fastest grid
// index, so the blocks that read the same weights run together (one DRAM read, the rest hit L2). The converted rows are in the order
// (i2, i1): the row of a column is i02*ncols + c.

// The work of one lane: its part of row row0 of dst[., i1, i2]
template <bool q3>
KQ_DEV float kq_mm_lane(const char * __restrict__ w, const char * __restrict__ x, const int c, const int tile, const int i02,
        const int warp, const int lane, const int nsb, const int ne01, const int64_t nb01, const int64_t nb02, const int ne11, const int r2,
        int & i1, int & i2, int & row0) {
    const int i2r = r2 > 1 ? c / ne11 : 0;
    i1   = c - i2r*ne11;
    i2   = i02*r2 + i2r;
    row0 = tile*KQ_ROWS + warp*8 + (lane >> 2);
    const int row = row0 < ne01 ? row0 : ne01 - 1; // rows past the end read the last row and do not store
    const char * wr = w + (int64_t) i02*nb02 + (int64_t) row*nb01;
    const char * xr = x + (int64_t) (i2*ne11 + i1)*nsb*KQ_REC_BYTES;
    return q3 ? kq_q3k_lane(wr, nsb, lane & 3, xr) : kq_q2k_lane(wr, nsb, lane & 3, xr);
}
