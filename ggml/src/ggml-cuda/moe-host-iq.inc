// The "k" kernels of the i-quant cold experts (Project Maya's GLM-5.3-Flash and DeepSeek-V4.1-Flash quants): IQ2_XXS, IQ2_S,
// IQ3_XXS, IQ3_S and IQ1_S gate/up/down rows against one activation row in k blocks (kact_block, see moe-host-cpu.cpp).
// Included inside the anonymous namespace of moe-host-cpu.cpp, in the MH_X86 section, after kdot_q40_avx2.
//
// Every i-quant block is 256 weights = 8 groups of 32, and the k activation block keeps each group of 32 contiguous, so a
// group is: the grid lookups (an index per 8 or 4 weights, gathered), the signs, one int8 multiply-add against the group's
// activation values, one integer scale per group (two for IQ2_S: 16 weights each) and the group's float activation scale.
//   IQ2_XXS 66 bytes: d, 8 x { 4 grid bytes, u32 (4 x 7-bit sign index, scale in bits 28..31) }   w = d*0.125*(2s+1)*grid*sign
//   IQ2_S   82 bytes: d, qs[32] grid low bytes, qs[32] sign bytes, qh[8], scales[8] (two nibbles)  w = d*0.125*(2s+1)*grid*sign
//   IQ3_XXS 98 bytes: d, qs[64] grid bytes, 8 x u32 (as IQ2_XXS)                                   w = d*0.25*(2s+1)*grid*sign
//   IQ3_S  110 bytes: d, qs[64], qh[8], signs[32], scales[4] (a nibble per group)                  w = d*(2s+1)*grid*sign
//   IQ1_S   50 bytes: d, qs[32], qh[8] x u16 (3 index bits x4, scale bits 12..14, delta sign bit 15) w = d*(2s+1)*(grid + delta)
// The grids are ggml's (ggml-common.h) and the arithmetic is ggml's vec_dot (generic) with the activation of the k block,
// float rounding aside.

#define MH_IQ_YOFF(g) (64*((g) & 3) + 32*((g) >> 2))

static inline uint32_t iq_ld32(const uint8_t * p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline uint16_t iq_ld16(const uint8_t * p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

// the sign bytes as the AVX2 kernels apply them (vpsignb): 0xff where a sign bit is set, 0x01 elsewhere
struct iq_sign_tabs {
    uint64_t s8[256];   // by the sign mask byte
    uint64_t s7[128];   // by a 7-bit sign index of IQ2_XXS / IQ3_XXS (ksigns_iq2xs adds the parity bit)
    iq_sign_tabs() {
        for (int m = 0; m < 256; ++m) {
            uint64_t v = 0;
            for (int j = 0; j < 8; ++j) {
                v |= (uint64_t) ((m >> j) & 1 ? 0xff : 0x01) << (8*j);
            }
            s8[m] = v;
        }
        for (int i = 0; i < 128; ++i) {
            s7[i] = s8[ksigns_iq2xs[i]];
        }
    }
};

static const iq_sign_tabs & iq_tabs() {
    static const iq_sign_tabs t;
    return t;
}

// the four 8-bit sign masks of a group of IQ2_XXS / IQ3_XXS (a1: bits 0..27 = 4 x 7-bit index) as one 32-bit mask
static inline uint32_t iq_mask7(uint32_t a1) {
    return (uint32_t) ksigns_iq2xs[a1 & 127] | (uint32_t) ksigns_iq2xs[(a1 >> 7) & 127] << 8 |
           (uint32_t) ksigns_iq2xs[(a1 >> 14) & 127] << 16 | (uint32_t) ksigns_iq2xs[(a1 >> 21) & 127] << 24;
}

// ======================================================= AVX2 (one group of 32 per step)

#define MH_IQ_AVX2 __attribute__((target("avx2,f16c,fma")))

MH_IQ_AVX2 static inline __m256i iq_sgn7_avx2(const iq_sign_tabs & T, uint32_t a1) {
    return _mm256_set_epi64x((long long) T.s7[(a1 >> 21) & 127], (long long) T.s7[(a1 >> 14) & 127], (long long) T.s7[(a1 >> 7) & 127], (long long) T.s7[a1 & 127]);
}

MH_IQ_AVX2 static inline __m256i iq_sgn8_avx2(const iq_sign_tabs & T, const uint8_t * s4) {
    return _mm256_set_epi64x((long long) T.s8[s4[3]], (long long) T.s8[s4[2]], (long long) T.s8[s4[1]], (long long) T.s8[s4[0]]);
}

// sum += (grid . (y signed)) * scale * the group's activation scale
MH_IQ_AVX2 static inline __m256 iq_acc_avx2(__m256 sb, __m256i grid, __m256i sg, __m256i y, __m256i sc, const float * asg) {
    const __m256i pr = _mm256_madd_epi16(_mm256_maddubs_epi16(grid, _mm256_sign_epi8(y, sg)), sc);
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(pr), _mm256_broadcast_ss(asg), sb);
}

template <int NR>
MH_IQ_AVX2 static void kdot_iq2xxs_avx2(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const iq_sign_tabs & T = iq_tabs();
    __m256 acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm256_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p = w[r] + (size_t) k*66;
            const uint8_t * q = p + 2;
            __m256 sb = _mm256_setzero_ps();
#pragma GCC unroll 8
            for (int g = 0; g < 8; ++g) {
                const __m256i y  = _mm256_load_si256((const __m256i *) (ab.q + MH_IQ_YOFF(g)));
                const uint32_t i4 = iq_ld32(q + 8*g), a1 = iq_ld32(q + 8*g + 4);
                const __m256i grid = _mm256_i32gather_epi64((const long long *) iq2xxs_grid, _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) i4)), 8);
                sb = iq_acc_avx2(sb, grid, iq_sgn7_avx2(T, a1), y, _mm256_set1_epi16((short) (2*(a1 >> 28) + 1)), ab.asg + g);
            }
            acc[r] = _mm256_fmadd_ps(sb, _mm256_set1_ps(_cvtsh_ss(iq_ld16(p))*0.125f), acc[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = kh256(acc[r]);
    }
}

template <int NR>
MH_IQ_AVX2 static void kdot_iq2s_avx2(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const iq_sign_tabs & T = iq_tabs();
    const __m128i qsh = _mm_setr_epi32(8, 6, 4, 2);
    const __m128i m3  = _mm_set1_epi32(0x300);
    __m256 acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm256_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*82;
            const uint8_t * qs = p + 2, * sg = p + 34, * qh = p + 66, * sc = p + 74;
            __m256 sb = _mm256_setzero_ps();
#pragma GCC unroll 8
            for (int g = 0; g < 8; ++g) {
                const __m256i y   = _mm256_load_si256((const __m256i *) (ab.q + MH_IQ_YOFF(g)));
                const __m128i hi  = _mm_and_si128(_mm_sllv_epi32(_mm_set1_epi32(qh[g]), qsh), m3);
                const __m128i idx = _mm_or_si128(_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) iq_ld32(qs + 4*g))), hi);
                const __m256i grid = _mm256_i32gather_epi64((const long long *) iq2s_grid, idx, 8);
                const __m256i scl = _mm256_set_m128i(_mm_set1_epi16((short) (2*(sc[g] >> 4) + 1)), _mm_set1_epi16((short) (2*(sc[g] & 15) + 1)));
                sb = iq_acc_avx2(sb, grid, iq_sgn8_avx2(T, sg + 4*g), y, scl, ab.asg + g);
            }
            acc[r] = _mm256_fmadd_ps(sb, _mm256_set1_ps(_cvtsh_ss(iq_ld16(p))*0.125f), acc[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = kh256(acc[r]);
    }
}

template <int NR>
MH_IQ_AVX2 static void kdot_iq3xxs_avx2(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const iq_sign_tabs & T = iq_tabs();
    __m256 acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm256_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*98;
            const uint8_t * qs = p + 2, * gas = p + 66;
            __m256 sb = _mm256_setzero_ps();
#pragma GCC unroll 8
            for (int g = 0; g < 8; ++g) {
                const __m256i y  = _mm256_load_si256((const __m256i *) (ab.q + MH_IQ_YOFF(g)));
                const uint32_t a1 = iq_ld32(gas + 4*g);
                const __m256i grid = _mm256_i32gather_epi32((const int *) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) (qs + 8*g))), 4);
                sb = iq_acc_avx2(sb, grid, iq_sgn7_avx2(T, a1), y, _mm256_set1_epi16((short) (2*(a1 >> 28) + 1)), ab.asg + g);
            }
            acc[r] = _mm256_fmadd_ps(sb, _mm256_set1_ps(_cvtsh_ss(iq_ld16(p))*0.25f), acc[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = kh256(acc[r]);
    }
}

template <int NR>
MH_IQ_AVX2 static void kdot_iq3s_avx2(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const iq_sign_tabs & T = iq_tabs();
    const __m256i qsh = _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1);
    const __m256i m1  = _mm256_set1_epi32(256);
    __m256 acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm256_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*110;
            const uint8_t * qs = p + 2, * qh = p + 66, * sg = p + 74, * sc = p + 106;
            __m256 sb = _mm256_setzero_ps();
#pragma GCC unroll 8
            for (int g = 0; g < 8; ++g) {
                const __m256i y   = _mm256_load_si256((const __m256i *) (ab.q + MH_IQ_YOFF(g)));
                const __m256i hi  = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[g]), qsh), m1);
                const __m256i idx = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) (qs + 8*g))), hi);
                const __m256i grid = _mm256_i32gather_epi32((const int *) iq3s_grid, idx, 4);
                const int ls = 2*((sc[g >> 1] >> (4*(g & 1))) & 15) + 1;
                sb = iq_acc_avx2(sb, grid, iq_sgn8_avx2(T, sg + 4*g), y, _mm256_set1_epi16((short) ls), ab.asg + g);
            }
            acc[r] = _mm256_fmadd_ps(sb, _mm256_set1_ps(_cvtsh_ss(iq_ld16(p))), acc[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = kh256(acc[r]);
    }
}

// IQ1_S: the grid is ternary (-1, 0, 1) and the group has a shift delta = +-1/8 on every weight: the grid plus one is unsigned
// for vpmaddubsw, and the one (with the delta) leaves as (1 - delta)*scale*(sum of the group's activations), computed for the 8
// groups of a block at once (a branch on the delta sign mispredicts half the time)
template <int NR>
MH_IQ_AVX2 static void kdot_iq1s_avx2(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const __m128i qsh = _mm_setr_epi32(0, 3, 6, 9);
    const __m128i m7  = _mm_set1_epi32(7);
    const __m256i one = _mm256_set1_epi8(1);
    const __m256 fd1  = _mm256_set1_ps(2.0f*IQ1S_DELTA), fd0 = _mm256_set1_ps(1.0f - IQ1S_DELTA);
    __m256 acc[NR], cor[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm256_setzero_ps();
        cor[r] = _mm256_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        // the groups' activation sums times their scales
        const __m256 asb = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_load_si256((const __m256i *) ab.bs), _mm256_set1_epi16(1))), _mm256_load_ps(ab.asg));
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*50;
            const uint8_t * qs = p + 2, * qhp = p + 34;
            __m256 sb = _mm256_setzero_ps();
#pragma GCC unroll 8
            for (int g = 0; g < 8; ++g) {
                const __m256i y   = _mm256_load_si256((const __m256i *) (ab.q + MH_IQ_YOFF(g)));
                const uint32_t qh = iq_ld16(qhp + 2*g);
                const __m128i hi  = _mm_slli_epi32(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) qh), qsh), m7), 8);
                const __m128i idx = _mm_or_si128(_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) iq_ld32(qs + 4*g))), hi);
                const __m256i grid = _mm256_add_epi8(_mm256_i32gather_epi64((const long long *) iq1s_grid, idx, 8), one);
                const int   ls = 2*((qh >> 12) & 7) + 1;
                const __m256i pr = _mm256_madd_epi16(_mm256_maddubs_epi16(grid, y), _mm256_set1_epi16((short) ls));
                sb = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pr), _mm256_broadcast_ss(ab.asg + g), sb);
            }
            const __m256i qhv = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) qhp));
            const __m256  lsf = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qhv, 12), _mm256_set1_epi32(7)), 1), _mm256_set1_epi32(1)));
            const __m256  fv  = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(qhv, 15)), fd1, fd0);
            const __m256  d   = _mm256_set1_ps(_cvtsh_ss(iq_ld16(p)));
            acc[r] = _mm256_fmadd_ps(sb, d, acc[r]);
            cor[r] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_mul_ps(lsf, fv), asb), d, cor[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = kh256(acc[r]) - kh256(cor[r]);
    }
}

// ======================================================= AVX-512 (two groups of 64 per step: the groups j and j + 4 of
// a k block share a 64-byte vector of activations; signs by mask registers; the gathers take both groups' indices)
//
// What a group needs besides the lookups (sign masks, scales, the high index bits) is computed for the 8 groups of a block at
// once as vectors and handed to the pair steps by one permute each: scalar code for it was two thirds of the instructions.

#define MH_IQ_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,f16c,fma")))

// two ymm halves into one zmm
MH_IQ_AVX512 static inline __m512i iq_cat(__m256i lo, __m256i hi) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

// acc += (grid . (y negated where the mask has a bit)) * scale (32 int16: group a in 0..15, group b in 16..31) * asg
MH_IQ_AVX512 static inline __m512 iq_acc512(__m512 sb, __m512i grid, __mmask64 m, __m512i y, __m512i sc, __m512 as) {
    const __m512i ys = _mm512_mask_sub_epi8(y, m, _mm512_setzero_si512(), y);
    const __m512i pr = _mm512_madd_epi16(_mm512_maddubs_epi16(grid, ys), sc);
    return _mm512_fmadd_ps(_mm512_cvtepi32_ps(pr), as, sb);
}

// per k block, shared by the rows: the 4 vectors of activations (groups j and j + 4) and their float scales
struct iq_act512 {
    __m512i y[4];
    __m512  as[4];
};

MH_IQ_AVX512 static inline void iq_load_act512(const kact_block & ab, iq_act512 & a) {
    for (int j = 0; j < 4; ++j) {
        a.y[j]  = _mm512_load_si512((const void *) (ab.q + 64*j));
        a.as[j] = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(ab.asg[j])), _mm256_set1_ps(ab.asg[j + 4]), 1);
    }
}

// vpermw indexes of a pair step: 32 words = [word a0 x8, word a1 x8, word b0 x8, word b1 x8] of a vector of per-group scales
MH_IQ_AVX512 static inline __m512i iq_wperm(int a0, int a1, int b0, int b1) {
    short idx[32];
    for (int i = 0; i < 8; ++i) {
        idx[i]      = (short) a0;
        idx[8 + i]  = (short) a1;
        idx[16 + i] = (short) b0;
        idx[24 + i] = (short) b1;
    }
    return _mm512_loadu_si512((const void *) idx);
}

// the sign masks of the 8 groups of IQ2_XXS / IQ3_XXS from their words (4 x 7-bit sign index each, bits 0..27): one 32-bit mask
// per group (ksigns_iq2xs of each index, the parity bit from a nibble lookup)
MH_IQ_AVX512 static inline __m256i iq_masks7_vec(__m256i a1) {
    const __m256i m127 = _mm256_set1_epi32(127), m15 = _mm256_set1_epi8(15);
    const __m256i par  = _mm256_broadcastsi128_si256(_mm_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0));
    __m256i out = _mm256_setzero_si256();
#pragma GCC unroll 4
    for (int l = 0; l < 4; ++l) {
        const __m256i s = _mm256_and_si256(_mm256_srli_epi32(a1, 7*l), m127);
        const __m256i x = _mm256_and_si256(_mm256_xor_si256(s, _mm256_srli_epi32(s, 4)), m15);
        out = _mm256_or_si256(out, _mm256_slli_epi32(_mm256_or_si256(s, _mm256_shuffle_epi8(par, x)), 8*l));
    }
    return out;
}

// 8 dwords [m0 .. m7] -> 4 qwords [m_j | m_{j+4} << 32]
MH_IQ_AVX512 static inline __m256i iq_pairq(__m256i v) {
    return _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

template <int NR>
MH_IQ_AVX512 static void kdot_iq2xxs_avx512(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    __m512 acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm512_setzero_ps();
    }
    const __m512i wps[4] = { iq_wperm(0, 0, 4, 4), iq_wperm(1, 1, 5, 5), iq_wperm(2, 2, 6, 6), iq_wperm(3, 3, 7, 7) };
    for (int k = 0; k < nsb; ++k) {
        iq_act512 A;
        iq_load_act512(a[k], A);
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p = w[r] + (size_t) k*66;
            const __m512i v   = _mm512_loadu_si512((const void *) (p + 2));       // 8 groups x (4 grid bytes, u32 signs + scale)
            const __m256i a1  = _mm512_cvtepi64_epi32(_mm512_srli_epi64(v, 32));
            alignas(32) uint64_t mq[4], iq[4];
            _mm256_store_si256((__m256i *) mq, iq_pairq(iq_masks7_vec(a1)));
            _mm256_store_si256((__m256i *) iq, iq_pairq(_mm512_cvtepi64_epi32(v)));
            const __m128i ls = _mm_add_epi16(_mm_slli_epi16(_mm_packs_epi32(_mm256_castsi256_si128(_mm256_srli_epi32(a1, 28)), _mm256_extracti128_si256(_mm256_srli_epi32(a1, 28), 1)), 1), _mm_set1_epi16(1));
            const __m512i ls512 = _mm512_castsi128_si512(ls);
            __m512 sb = _mm512_setzero_ps();
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const __m256i idx  = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) iq[j]));
                const __m512i grid = _mm512_i32gather_epi64(idx, (const void *) iq2xxs_grid, 8);
                sb = iq_acc512(sb, grid, (__mmask64) mq[j], A.y[j], _mm512_permutexvar_epi16(wps[j], ls512), A.as[j]);
            }
            acc[r] = _mm512_fmadd_ps(sb, _mm512_set1_ps(_cvtsh_ss(iq_ld16(p))*0.125f), acc[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = _mm512_reduce_add_ps(acc[r]);
    }
}

template <int NR>
MH_IQ_AVX512 static void kdot_iq2s_avx512(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const __m256i qsh = _mm256_setr_epi32(8, 6, 4, 2, 8, 6, 4, 2);
    const __m256i m3  = _mm256_set1_epi32(0x300);
    __m512 acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm512_setzero_ps();
    }
    // the scale words of group g: [lo nibble x8, hi nibble x8] = words 2g, 2g + 1 of the interleaved nibbles, 8 lanes each
    const __m512i wps[4] = { iq_wperm(0, 1, 8, 9), iq_wperm(2, 3, 10, 11), iq_wperm(4, 5, 12, 13), iq_wperm(6, 7, 14, 15) };
    for (int k = 0; k < nsb; ++k) {
        iq_act512 A;
        iq_load_act512(a[k], A);
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*82;
            const uint8_t * qs = p + 2, * sg = p + 34, * qh = p + 66, * sc = p + 74;
            const __m128i s8   = _mm_loadl_epi64((const __m128i *) sc);
            const __m128i nib  = _mm_unpacklo_epi8(_mm_and_si128(s8, _mm_set1_epi8(15)), _mm_and_si128(_mm_srli_epi16(s8, 4), _mm_set1_epi8(15)));   // 16 bytes: g0 lo, g0 hi, g1 lo ...
            const __m256i ls16 = _mm256_add_epi16(_mm256_slli_epi16(_mm256_cvtepu8_epi16(nib), 1), _mm256_set1_epi16(1));
            const __m512i ls512 = _mm512_castsi256_si512(ls16);
            const __m256i qhv  = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) qh));
            __m512 sb = _mm512_setzero_ps();
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const int ga = j, gb = j + 4;
                const __m256i qpair = _mm256_permutevar8x32_epi32(qhv, _mm256_setr_epi32(ga, ga, ga, ga, gb, gb, gb, gb));
                const __m256i hi  = _mm256_and_si256(_mm256_sllv_epi32(qpair, qsh), m3);
                const __m256i idx = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_unpacklo_epi32(_mm_cvtsi32_si128((int) iq_ld32(qs + 4*ga)), _mm_cvtsi32_si128((int) iq_ld32(qs + 4*gb)))), hi);
                const __m512i grid = _mm512_i32gather_epi64(idx, (const void *) iq2s_grid, 8);
                const __mmask64 m = (__mmask64) iq_ld32(sg + 4*ga) | (__mmask64) iq_ld32(sg + 4*gb) << 32;
                sb = iq_acc512(sb, grid, m, A.y[j], _mm512_permutexvar_epi16(wps[j], ls512), A.as[j]);
            }
            acc[r] = _mm512_fmadd_ps(sb, _mm512_set1_ps(_cvtsh_ss(iq_ld16(p))*0.125f), acc[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = _mm512_reduce_add_ps(acc[r]);
    }
}

template <int NR>
MH_IQ_AVX512 static void kdot_iq3xxs_avx512(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    __m512 acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm512_setzero_ps();
    }
    const __m512i wps[4] = { iq_wperm(0, 0, 4, 4), iq_wperm(1, 1, 5, 5), iq_wperm(2, 2, 6, 6), iq_wperm(3, 3, 7, 7) };
    for (int k = 0; k < nsb; ++k) {
        iq_act512 A;
        iq_load_act512(a[k], A);
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*98;
            const uint8_t * qs = p + 2;
            const __m256i a1   = _mm256_loadu_si256((const __m256i *) (p + 66));
            alignas(32) uint64_t mq[4];
            _mm256_store_si256((__m256i *) mq, iq_pairq(iq_masks7_vec(a1)));
            const __m256i l28  = _mm256_srli_epi32(a1, 28);
            const __m128i ls   = _mm_add_epi16(_mm_slli_epi16(_mm_packs_epi32(_mm256_castsi256_si128(l28), _mm256_extracti128_si256(l28, 1)), 1), _mm_set1_epi16(1));
            const __m512i ls512 = _mm512_castsi128_si512(ls);
            __m512 sb = _mm512_setzero_ps();
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const __m128i i16 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *) (qs + 8*j)), _mm_loadl_epi64((const __m128i *) (qs + 8*(j + 4))));
                const __m512i grid = _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(i16), (const void *) iq3xxs_grid, 4);
                sb = iq_acc512(sb, grid, (__mmask64) mq[j], A.y[j], _mm512_permutexvar_epi16(wps[j], ls512), A.as[j]);
            }
            acc[r] = _mm512_fmadd_ps(sb, _mm512_set1_ps(_cvtsh_ss(iq_ld16(p))*0.25f), acc[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = _mm512_reduce_add_ps(acc[r]);
    }
}

template <int NR>
MH_IQ_AVX512 static void kdot_iq3s_avx512(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const __m512i qsh = _mm512_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1);
    const __m512i m1  = _mm512_set1_epi32(256);
    __m512 acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm512_setzero_ps();
    }
    const __m512i wps[4] = { iq_wperm(0, 0, 4, 4), iq_wperm(1, 1, 5, 5), iq_wperm(2, 2, 6, 6), iq_wperm(3, 3, 7, 7) };
    const __m512i qperm[4] = {
        _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4), _mm512_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5),
        _mm512_setr_epi32(2, 2, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6), _mm512_setr_epi32(3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7) };
    for (int k = 0; k < nsb; ++k) {
        iq_act512 A;
        iq_load_act512(a[k], A);
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*110;
            const uint8_t * qs = p + 2, * qh = p + 66, * sg = p + 74, * sc = p + 106;
            const __m128i s4   = _mm_cvtsi32_si128((int) iq_ld32(sc));
            const __m128i nib  = _mm_unpacklo_epi8(_mm_and_si128(s4, _mm_set1_epi8(15)), _mm_and_si128(_mm_srli_epi16(s4, 4), _mm_set1_epi8(15)));   // groups 0..7
            const __m128i ls   = _mm_add_epi16(_mm_slli_epi16(_mm_cvtepu8_epi16(nib), 1), _mm_set1_epi16(1));
            const __m512i ls512 = _mm512_castsi128_si512(ls);
            const __m512i qhv  = _mm512_castsi256_si512(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) qh)));
            __m512 sb = _mm512_setzero_ps();
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const int ga = j, gb = j + 4;
                const __m512i hi  = _mm512_and_si512(_mm512_sllv_epi32(_mm512_permutexvar_epi32(qperm[j], qhv), qsh), m1);
                const __m128i i16 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *) (qs + 8*ga)), _mm_loadl_epi64((const __m128i *) (qs + 8*gb)));
                const __m512i grid = _mm512_i32gather_epi32(_mm512_or_si512(_mm512_cvtepu8_epi32(i16), hi), (const void *) iq3s_grid, 4);
                const __mmask64 m = (__mmask64) iq_ld32(sg + 4*ga) | (__mmask64) iq_ld32(sg + 4*gb) << 32;
                sb = iq_acc512(sb, grid, m, A.y[j], _mm512_permutexvar_epi16(wps[j], ls512), A.as[j]);
            }
            acc[r] = _mm512_fmadd_ps(sb, _mm512_set1_ps(_cvtsh_ss(iq_ld16(p))), acc[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = _mm512_reduce_add_ps(acc[r]);
    }
}

template <int NR>
MH_IQ_AVX512 static void kdot_iq1s_avx512(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const __m256i qsh = _mm256_setr_epi32(0, 3, 6, 9, 0, 3, 6, 9);
    const __m256i m7  = _mm256_set1_epi32(7);
    const __m512i one = _mm512_set1_epi8(1);
    const __m256 fd1  = _mm256_set1_ps(2.0f*IQ1S_DELTA), fd0 = _mm256_set1_ps(1.0f - IQ1S_DELTA);
    const __m512i wps[4] = { iq_wperm(0, 0, 4, 4), iq_wperm(1, 1, 5, 5), iq_wperm(2, 2, 6, 6), iq_wperm(3, 3, 7, 7) };
    __m512 acc[NR];
    __m256 cor[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm512_setzero_ps();
        cor[r] = _mm256_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        iq_act512 A;
        iq_load_act512(ab, A);
        const __m256 asb = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_load_si256((const __m256i *) ab.bs), _mm256_set1_epi16(1))), _mm256_load_ps(ab.asg));
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*50;
            const uint8_t * qs = p + 2, * qhp = p + 34;
            const __m256i qhv = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) qhp));
            const __m256i l3  = _mm256_and_si256(_mm256_srli_epi32(qhv, 12), m7);
            const __m128i ls  = _mm_add_epi16(_mm_slli_epi16(_mm_packs_epi32(_mm256_castsi256_si128(l3), _mm256_extracti128_si256(l3, 1)), 1), _mm_set1_epi16(1));
            const __m512i ls512 = _mm512_castsi128_si512(ls);
            __m512 sb = _mm512_setzero_ps();
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const int ga = j, gb = j + 4;
                const __m256i qpair = _mm256_permutevar8x32_epi32(qhv, _mm256_setr_epi32(ga, ga, ga, ga, gb, gb, gb, gb));
                const __m256i hi  = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qpair, qsh), m7), 8);
                const __m256i idx = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_unpacklo_epi32(_mm_cvtsi32_si128((int) iq_ld32(qs + 4*ga)), _mm_cvtsi32_si128((int) iq_ld32(qs + 4*gb)))), hi);
                const __m512i grid = _mm512_add_epi8(_mm512_i32gather_epi64(idx, (const void *) iq1s_grid, 8), one);
                const __m512i pr = _mm512_madd_epi16(_mm512_maddubs_epi16(grid, A.y[j]), _mm512_permutexvar_epi16(wps[j], ls512));
                sb = _mm512_fmadd_ps(_mm512_cvtepi32_ps(pr), A.as[j], sb);
            }
            const __m256  lsf = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(l3, 1), _mm256_set1_epi32(1)));
            const __m256  fv  = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(qhv, 15)), fd1, fd0);
            const float   d   = _cvtsh_ss(iq_ld16(p));
            acc[r] = _mm512_fmadd_ps(sb, _mm512_set1_ps(d), acc[r]);
            cor[r] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_mul_ps(lsf, fv), asb), _mm256_set1_ps(d), cor[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = _mm512_reduce_add_ps(acc[r]) - kh256(cor[r]);
    }
}

// ======================================================= IQ2_XS and Q4_K (AVX2 only: an AVX-512 machine runs them on these)
//   IQ2_XS  74 bytes: d, qs[32] u16 (9-bit grid index, 7-bit sign index above), scales[8] (two nibbles per group)   w = d*0.125*(2s+1)*grid*sign
//   Q4_K   144 bytes: d, dmin, scales[12] (8 x 6-bit scale + 8 x 6-bit min), qs[128] (group 2j: low nibbles, 2j + 1: high nibbles of 32 bytes)

template <int NR>
MH_IQ_AVX2 static void kdot_iq2xs_avx2(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const iq_sign_tabs & T = iq_tabs();
    const __m128i m511 = _mm_set1_epi32(511);
    __m256 acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm256_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*74;
            const uint8_t * qs = p + 2, * sc = p + 66;
            __m256 sb = _mm256_setzero_ps();
#pragma GCC unroll 8
            for (int g = 0; g < 8; ++g) {
                const __m256i y   = _mm256_load_si256((const __m256i *) (ab.q + MH_IQ_YOFF(g)));
                const __m128i c   = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *) (qs + 8*g)));
                const __m256i grid = _mm256_i32gather_epi64((const long long *) iq2xs_grid, _mm_and_si128(c, m511), 8);
                const __m128i si  = _mm_srli_epi32(c, 9);
                const __m256i sg  = _mm256_set_epi64x((long long) T.s7[_mm_extract_epi32(si, 3)], (long long) T.s7[_mm_extract_epi32(si, 2)],
                                                      (long long) T.s7[_mm_extract_epi32(si, 1)], (long long) T.s7[_mm_cvtsi128_si32(si)]);
                const __m256i scl = _mm256_set_m128i(_mm_set1_epi16((short) (2*(sc[g] >> 4) + 1)), _mm_set1_epi16((short) (2*(sc[g] & 15) + 1)));
                sb = iq_acc_avx2(sb, grid, sg, y, scl, ab.asg + g);
            }
            acc[r] = _mm256_fmadd_ps(sb, _mm256_set1_ps(_cvtsh_ss(iq_ld16(p))*0.125f), acc[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = kh256(acc[r]);
    }
}

template <int NR>
MH_IQ_AVX2 static void kdot_q4k_avx2(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    __m256 acc[NR];
    float  cor[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm256_setzero_ps();
        cor[r] = 0.0f;
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p = w[r] + (size_t) k*144;
            const float d = _cvtsh_ss(iq_ld16(p)), dmin = _cvtsh_ss(iq_ld16(p + 2));
            const uint8_t * q = p + 4;
            int scl[8], mn[8];
            for (int j = 0; j < 4; ++j) {
                scl[j] = q[j] & 63;
                mn[j]  = q[j + 4] & 63;
                scl[j + 4] = (q[j + 8] & 15) | ((q[j] >> 6) << 4);
                mn[j + 4]  = (q[j + 8] >> 4) | ((q[j + 4] >> 6) << 4);
            }
            __m256 sb = _mm256_setzero_ps();
            float  cm = 0.0f;
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const __m256i qv = _mm256_loadu_si256((const __m256i *) (p + 16 + 32*j));
                const int ga = 2*j, gb = 2*j + 1;
                const __m256i ya = _mm256_load_si256((const __m256i *) (ab.q + MH_IQ_YOFF(ga)));
                const __m256i yb = _mm256_load_si256((const __m256i *) (ab.q + MH_IQ_YOFF(gb)));
                const __m256i pa = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(qv, m4), ya), _mm256_set1_epi16((short) scl[ga]));
                const __m256i pb = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(qv, 4), m4), yb), _mm256_set1_epi16((short) scl[gb]));
                sb = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pa), _mm256_broadcast_ss(ab.asg + ga), sb);
                sb = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pb), _mm256_broadcast_ss(ab.asg + gb), sb);
                cm += (float) mn[ga]*ab.asg[ga]*(float) (ab.bs[2*ga] + ab.bs[2*ga + 1]) + (float) mn[gb]*ab.asg[gb]*(float) (ab.bs[2*gb] + ab.bs[2*gb + 1]);
            }
            acc[r] = _mm256_fmadd_ps(sb, _mm256_set1_ps(d), acc[r]);
            cor[r] += cm*dmin;
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = kh256(acc[r]) - cor[r];
    }
}

// ======================================================= decoded rows (AVX-512): a prompt batch routes several tokens to the
// same cold expert, and the gathers, sign masks and scale words above are the same for every token. kdec_* decode a row's
// blocks once into the gathered grid bytes (the activation's order: bytes 64j.. = groups j, j + 4), the sign masks of the 4
// pair steps and the 16 scale words (the IQ2_S layout: group g lo, hi; IQ3_*: both words the group's), and kdot_dec_avx512
// is the dot of decoded rows with a token (one load, one masked negate, maddubs, madd, cvt, fma per pair step).

struct alignas(64) kdec_block {
    uint8_t  g[256];
    uint64_t m[4];
    int16_t  ls[16];
};
static_assert(sizeof(kdec_block) == 320, "unexpected kdec block size");

typedef void (*kdec_fn)(const uint8_t * row, int nsb, kdec_block * out, float * d);
typedef void (*kdotdec_fn)(const kdec_block * const * w, const float * const * d, const kact_block * a, int nsb, float * out);

MH_IQ_AVX512 static void kdec_iq2s_avx512(const uint8_t * row, int nsb, kdec_block * out, float * d) {
    const __m256i qsh = _mm256_setr_epi32(8, 6, 4, 2, 8, 6, 4, 2);
    const __m256i m3  = _mm256_set1_epi32(0x300);
    for (int k = 0; k < nsb; ++k) {
        const uint8_t * p  = row + (size_t) k*82;
        const uint8_t * qs = p + 2, * sg = p + 34, * qh = p + 66, * sc = p + 74;
        const __m128i s8   = _mm_loadl_epi64((const __m128i *) sc);
        const __m128i nib  = _mm_unpacklo_epi8(_mm_and_si128(s8, _mm_set1_epi8(15)), _mm_and_si128(_mm_srli_epi16(s8, 4), _mm_set1_epi8(15)));
        _mm256_storeu_si256((__m256i *) out[k].ls, _mm256_add_epi16(_mm256_slli_epi16(_mm256_cvtepu8_epi16(nib), 1), _mm256_set1_epi16(1)));
        const __m256i qhv  = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) qh));
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            const int ga = j, gb = j + 4;
            const __m256i qpair = _mm256_permutevar8x32_epi32(qhv, _mm256_setr_epi32(ga, ga, ga, ga, gb, gb, gb, gb));
            const __m256i hi  = _mm256_and_si256(_mm256_sllv_epi32(qpair, qsh), m3);
            const __m256i idx = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_unpacklo_epi32(_mm_cvtsi32_si128((int) iq_ld32(qs + 4*ga)), _mm_cvtsi32_si128((int) iq_ld32(qs + 4*gb)))), hi);
            _mm512_store_si512((void *) (out[k].g + 64*j), _mm512_i32gather_epi64(idx, (const void *) iq2s_grid, 8));
            out[k].m[j] = (uint64_t) iq_ld32(sg + 4*ga) | (uint64_t) iq_ld32(sg + 4*gb) << 32;
        }
        d[k] = _cvtsh_ss(iq_ld16(p))*0.125f;
    }
}

MH_IQ_AVX512 static void kdec_iq3xxs_avx512(const uint8_t * row, int nsb, kdec_block * out, float * d) {
    for (int k = 0; k < nsb; ++k) {
        const uint8_t * p  = row + (size_t) k*98;
        const uint8_t * qs = p + 2;
        const __m256i a1   = _mm256_loadu_si256((const __m256i *) (p + 66));
        _mm256_storeu_si256((__m256i *) out[k].m, iq_pairq(iq_masks7_vec(a1)));
        const __m256i l28  = _mm256_srli_epi32(a1, 28);
        const __m128i ls   = _mm_add_epi16(_mm_slli_epi16(_mm_packs_epi32(_mm256_castsi256_si128(l28), _mm256_extracti128_si256(l28, 1)), 1), _mm_set1_epi16(1));
        _mm_storeu_si128((__m128i *) out[k].ls,       _mm_unpacklo_epi16(ls, ls));
        _mm_storeu_si128((__m128i *) (out[k].ls + 8), _mm_unpackhi_epi16(ls, ls));
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            const __m128i i16 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *) (qs + 8*j)), _mm_loadl_epi64((const __m128i *) (qs + 8*(j + 4))));
            _mm512_store_si512((void *) (out[k].g + 64*j), _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(i16), (const void *) iq3xxs_grid, 4));
        }
        d[k] = _cvtsh_ss(iq_ld16(p))*0.25f;
    }
}

MH_IQ_AVX512 static void kdec_iq3s_avx512(const uint8_t * row, int nsb, kdec_block * out, float * d) {
    const __m512i qsh = _mm512_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1);
    const __m512i m1  = _mm512_set1_epi32(256);
    const __m512i qperm[4] = {
        _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4), _mm512_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5),
        _mm512_setr_epi32(2, 2, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6), _mm512_setr_epi32(3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7) };
    for (int k = 0; k < nsb; ++k) {
        const uint8_t * p  = row + (size_t) k*110;
        const uint8_t * qs = p + 2, * qh = p + 66, * sg = p + 74, * sc = p + 106;
        const __m128i s4   = _mm_cvtsi32_si128((int) iq_ld32(sc));
        const __m128i nib  = _mm_unpacklo_epi8(_mm_and_si128(s4, _mm_set1_epi8(15)), _mm_and_si128(_mm_srli_epi16(s4, 4), _mm_set1_epi8(15)));
        const __m128i ls   = _mm_add_epi16(_mm_slli_epi16(_mm_cvtepu8_epi16(nib), 1), _mm_set1_epi16(1));
        _mm_storeu_si128((__m128i *) out[k].ls,       _mm_unpacklo_epi16(ls, ls));
        _mm_storeu_si128((__m128i *) (out[k].ls + 8), _mm_unpackhi_epi16(ls, ls));
        const __m512i qhv  = _mm512_castsi256_si512(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) qh)));
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            const int ga = j, gb = j + 4;
            const __m512i hi  = _mm512_and_si512(_mm512_sllv_epi32(_mm512_permutexvar_epi32(qperm[j], qhv), qsh), m1);
            const __m128i i16 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *) (qs + 8*ga)), _mm_loadl_epi64((const __m128i *) (qs + 8*gb)));
            _mm512_store_si512((void *) (out[k].g + 64*j), _mm512_i32gather_epi32(_mm512_or_si512(_mm512_cvtepu8_epi32(i16), hi), (const void *) iq3s_grid, 4));
            out[k].m[j] = (uint64_t) iq_ld32(sg + 4*ga) | (uint64_t) iq_ld32(sg + 4*gb) << 32;
        }
        d[k] = _cvtsh_ss(iq_ld16(p));
    }
}

template <int NR>
MH_IQ_AVX512 static void kdot_dec_avx512(const kdec_block * const * w, const float * const * d, const kact_block * a, int nsb, float * out) {
    const __m512i wps[4] = { iq_wperm(0, 1, 8, 9), iq_wperm(2, 3, 10, 11), iq_wperm(4, 5, 12, 13), iq_wperm(6, 7, 14, 15) };
    __m512 acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm512_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        iq_act512 A;
        iq_load_act512(a[k], A);
        for (int r = 0; r < NR; ++r) {
            const kdec_block & b = w[r][k];
            const __m512i ls512 = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) b.ls));
            __m512 sb = _mm512_setzero_ps();
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                sb = iq_acc512(sb, _mm512_load_si512((const void *) (b.g + 64*j)), (__mmask64) b.m[j], A.y[j], _mm512_permutexvar_epi16(wps[j], ls512), A.as[j]);
            }
            acc[r] = _mm512_fmadd_ps(sb, _mm512_set1_ps(d[r][k]), acc[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = _mm512_reduce_add_ps(acc[r]);
    }
}

#undef MH_IQ_AVX2
#undef MH_IQ_AVX512
