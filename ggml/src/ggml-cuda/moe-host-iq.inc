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
// for vpmaddubsw, and the one (with the delta) leaves as (1 - delta)*scale*(sum of the group's activations)
template <int NR>
MH_IQ_AVX2 static void kdot_iq1s_avx2(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const __m128i qsh = _mm_setr_epi32(0, 3, 6, 9);
    const __m128i m7  = _mm_set1_epi32(7);
    const __m256i one = _mm256_set1_epi8(1);
    __m256 acc[NR];
    float  cor[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm256_setzero_ps();
        cor[r] = 0.0f;
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*50;
            const uint8_t * qs = p + 2, * qhp = p + 34;
            __m256 sb = _mm256_setzero_ps();
            float  cb = 0.0f;
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
                cb += ab.asg[g]*(float) ls*((qh & 0x8000 ? 1.0f + IQ1S_DELTA : 1.0f - IQ1S_DELTA))*(float) (ab.bs[2*g] + ab.bs[2*g + 1]);
            }
            const float d = _cvtsh_ss(iq_ld16(p));
            acc[r] = _mm256_fmadd_ps(sb, _mm256_set1_ps(d), acc[r]);
            cor[r] += cb*d;
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = kh256(acc[r]) - cor[r];
    }
}

// ======================================================= AVX-512 (two groups of 64 per step: the groups j and j + 4 of
// a k block share a 64-byte vector of activations; signs by mask registers; the gathers take both groups' indices)

#define MH_IQ_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,f16c,fma")))

// two ymm halves into one zmm
MH_IQ_AVX512 static inline __m512i iq_cat(__m256i lo, __m256i hi) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

MH_IQ_AVX512 static inline __m512 iq_asg2(const float * asg, int a, int b) {
    return _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(asg[a])), _mm256_set1_ps(asg[b]), 1);
}

// acc += (grid . (y negated where the mask has a bit)) * scale (16 int32 lanes: group a in 0..7, group b in 8..15) * asg
MH_IQ_AVX512 static inline __m512 iq_acc512(__m512 sb, __m512i grid, __mmask64 m, __m512i y, __m512i sc, __m512 as) {
    const __m512i ys = _mm512_mask_sub_epi8(y, m, _mm512_setzero_si512(), y);
    const __m512i pr = _mm512_madd_epi16(_mm512_maddubs_epi16(grid, ys), sc);
    return _mm512_fmadd_ps(_mm512_cvtepi32_ps(pr), as, sb);
}

template <int NR>
MH_IQ_AVX512 static void kdot_iq2xxs_avx512(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    __m512 acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm512_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p = w[r] + (size_t) k*66;
            const uint8_t * q = p + 2;
            __m512 sb = _mm512_setzero_ps();
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const uint32_t ia = iq_ld32(q + 8*j), aa = iq_ld32(q + 8*j + 4);
                const uint32_t ib = iq_ld32(q + 8*(j + 4)), ab1 = iq_ld32(q + 8*(j + 4) + 4);
                const __m256i idx = _mm256_cvtepu8_epi32(_mm_unpacklo_epi32(_mm_cvtsi32_si128((int) ia), _mm_cvtsi32_si128((int) ib)));
                const __m512i grid = _mm512_i32gather_epi64(idx, (const void *) iq2xxs_grid, 8);
                const __mmask64 m = (__mmask64) iq_mask7(aa) | (__mmask64) iq_mask7(ab1) << 32;
                const __m512i sc = iq_cat(_mm256_set1_epi16((short) (2*(aa >> 28) + 1)), _mm256_set1_epi16((short) (2*(ab1 >> 28) + 1)));
                sb = iq_acc512(sb, grid, m, _mm512_load_si512((const void *) (ab.q + 64*j)), sc, iq_asg2(ab.asg, j, j + 4));
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
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*82;
            const uint8_t * qs = p + 2, * sg = p + 34, * qh = p + 66, * sc = p + 74;
            __m512 sb = _mm512_setzero_ps();
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const int ga = j, gb = j + 4;
                const __m256i hi  = _mm256_and_si256(_mm256_sllv_epi32(_mm256_inserti128_si256(_mm256_set1_epi32(qh[ga]), _mm_set1_epi32(qh[gb]), 1), qsh), m3);
                const __m256i idx = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_unpacklo_epi32(_mm_cvtsi32_si128((int) iq_ld32(qs + 4*ga)), _mm_cvtsi32_si128((int) iq_ld32(qs + 4*gb)))), hi);
                const __m512i grid = _mm512_i32gather_epi64(idx, (const void *) iq2s_grid, 8);
                const __mmask64 m = (__mmask64) iq_ld32(sg + 4*ga) | (__mmask64) iq_ld32(sg + 4*gb) << 32;
                const __m256i sa = _mm256_set_m128i(_mm_set1_epi16((short) (2*(sc[ga] >> 4) + 1)), _mm_set1_epi16((short) (2*(sc[ga] & 15) + 1)));
                const __m256i sbv = _mm256_set_m128i(_mm_set1_epi16((short) (2*(sc[gb] >> 4) + 1)), _mm_set1_epi16((short) (2*(sc[gb] & 15) + 1)));
                sb = iq_acc512(sb, grid, m, _mm512_load_si512((const void *) (ab.q + 64*j)), iq_cat(sa, sbv), iq_asg2(ab.asg, ga, gb));
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
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*98;
            const uint8_t * qs = p + 2, * gas = p + 66;
            __m512 sb = _mm512_setzero_ps();
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const uint32_t aa = iq_ld32(gas + 4*j), ab1 = iq_ld32(gas + 4*(j + 4));
                const __m128i i16 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *) (qs + 8*j)), _mm_loadl_epi64((const __m128i *) (qs + 8*(j + 4))));
                const __m512i grid = _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(i16), (const void *) iq3xxs_grid, 4);
                const __mmask64 m = (__mmask64) iq_mask7(aa) | (__mmask64) iq_mask7(ab1) << 32;
                const __m512i sc = iq_cat(_mm256_set1_epi16((short) (2*(aa >> 28) + 1)), _mm256_set1_epi16((short) (2*(ab1 >> 28) + 1)));
                sb = iq_acc512(sb, grid, m, _mm512_load_si512((const void *) (ab.q + 64*j)), sc, iq_asg2(ab.asg, j, j + 4));
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
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*110;
            const uint8_t * qs = p + 2, * qh = p + 66, * sg = p + 74, * sc = p + 106;
            __m512 sb = _mm512_setzero_ps();
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const int ga = j, gb = j + 4;
                const __m512i hi  = _mm512_and_si512(_mm512_sllv_epi32(iq_cat(_mm256_set1_epi32(qh[ga]), _mm256_set1_epi32(qh[gb])), qsh), m1);
                const __m128i i16 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *) (qs + 8*ga)), _mm_loadl_epi64((const __m128i *) (qs + 8*gb)));
                const __m512i grid = _mm512_i32gather_epi32(_mm512_or_si512(_mm512_cvtepu8_epi32(i16), hi), (const void *) iq3s_grid, 4);
                const __mmask64 m = (__mmask64) iq_ld32(sg + 4*ga) | (__mmask64) iq_ld32(sg + 4*gb) << 32;
                const int la = 2*((sc[ga >> 1] >> (4*(ga & 1))) & 15) + 1, lb = 2*((sc[gb >> 1] >> (4*(gb & 1))) & 15) + 1;
                sb = iq_acc512(sb, grid, m, _mm512_load_si512((const void *) (ab.q + 64*j)),
                               iq_cat(_mm256_set1_epi16((short) la), _mm256_set1_epi16((short) lb)), iq_asg2(ab.asg, ga, gb));
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
    __m512 acc[NR];
    float  cor[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm512_setzero_ps();
        cor[r] = 0.0f;
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*50;
            const uint8_t * qs = p + 2, * qhp = p + 34;
            __m512 sb = _mm512_setzero_ps();
            float  cb = 0.0f;
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const int ga = j, gb = j + 4;
                const uint32_t qa = iq_ld16(qhp + 2*ga), qb = iq_ld16(qhp + 2*gb);
                const __m256i hi  = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(_mm256_inserti128_si256(_mm256_set1_epi32((int) qa), _mm_set1_epi32((int) qb), 1), qsh), m7), 8);
                const __m256i idx = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_unpacklo_epi32(_mm_cvtsi32_si128((int) iq_ld32(qs + 4*ga)), _mm_cvtsi32_si128((int) iq_ld32(qs + 4*gb)))), hi);
                const __m512i grid = _mm512_add_epi8(_mm512_i32gather_epi64(idx, (const void *) iq1s_grid, 8), one);
                const int la = 2*((qa >> 12) & 7) + 1, lb = 2*((qb >> 12) & 7) + 1;
                const __m512i pr = _mm512_madd_epi16(_mm512_maddubs_epi16(grid, _mm512_load_si512((const void *) (ab.q + 64*j))),
                                                     iq_cat(_mm256_set1_epi16((short) la), _mm256_set1_epi16((short) lb)));
                sb = _mm512_fmadd_ps(_mm512_cvtepi32_ps(pr), iq_asg2(ab.asg, ga, gb), sb);
                cb += ab.asg[ga]*(float) la*((qa & 0x8000 ? 1.0f + IQ1S_DELTA : 1.0f - IQ1S_DELTA))*(float) (ab.bs[2*ga] + ab.bs[2*ga + 1]) +
                      ab.asg[gb]*(float) lb*((qb & 0x8000 ? 1.0f + IQ1S_DELTA : 1.0f - IQ1S_DELTA))*(float) (ab.bs[2*gb] + ab.bs[2*gb + 1]);
            }
            const float d = _cvtsh_ss(iq_ld16(p));
            acc[r] = _mm512_fmadd_ps(sb, _mm512_set1_ps(d), acc[r]);
            cor[r] += cb*d;
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = _mm512_reduce_add_ps(acc[r]) - cor[r];
    }
}

#undef MH_IQ_AVX2
#undef MH_IQ_AVX512
