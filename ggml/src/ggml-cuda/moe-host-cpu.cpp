// Host worker pool of the cold-expert path: spinning threads compute the cold experts of a hot/cold
// MUL_MAT_ID triple (gate, up, swiglu, down) from pinned host memory while the GPU computes the hot ones.
// See moe-host.cuh.

#include "moe-host-cpu.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <ctime>

#if defined(__linux__)
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sched.h>
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define MH_X86 1
#else
#define MH_X86 0
#endif

namespace {

constexpr int MH_MAX_DEVICES = 16;
constexpr int QK = 64; // Q2_0 block: half d + 16 bytes of 2-bit codes, element e at bits 2*(e%4) of byte e/4

struct q2_block {
    uint16_t d;
    uint8_t  qs[16];
};
static_assert(sizeof(q2_block) == 18, "unexpected Q2_0 block size");

typedef mh_act_block act_block;

// v: the 64 rounded values of the block, d: the scales of its two halves
static void pack_act(const int32_t * v, const float * d, act_block & o) {
    int sum[2] = {0, 0};
    for (int i = 0; i < 64; ++i) {
        sum[i/32] += v[i];
    }
    for (int L = 0; L < 4; ++L) {
        for (int j = 0; j < 16; ++j) {
            o.q[16*L + j] = (int8_t) v[4*j + L];
        }
        o.s[4*L + 0] = d[0];
        o.s[4*L + 1] = d[0];
        o.s[4*L + 2] = d[1];
        o.s[4*L + 3] = d[1];
    }
    o.corr = d[0]*(float) sum[0] + d[1]*(float) sum[1];
}

static void quantize_act_generic(const float * x, int n, act_block * out) {
    for (int b = 0; b < n/QK; ++b) {
        const float * xb = x + b*QK;
        int32_t v[64];
        float   d[2];
        for (int h = 0; h < 2; ++h) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = std::max(amax, std::fabs(xb[32*h + i]));
            }
            d[h] = amax/127.0f;
            const float id = d[h] != 0.0f ? 1.0f/d[h] : 0.0f;
            for (int i = 0; i < 32; ++i) {
                v[32*h + i] = (int32_t) nearbyintf(xb[32*h + i]*id);
            }
        }
        pack_act(v, d, out[b]);
    }
}

#if MH_X86
__attribute__((target("avx2,fma")))
static void quantize_act_avx2(const float * x, int n, act_block * out) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int b = 0; b < n/QK; ++b) {
        const float * xb = x + b*QK;
        alignas(32) int32_t v[64];
        float d[2];
        for (int h = 0; h < 2; ++h) {
            const __m256 x0 = _mm256_loadu_ps(xb + 32*h +  0);
            const __m256 x1 = _mm256_loadu_ps(xb + 32*h +  8);
            const __m256 x2 = _mm256_loadu_ps(xb + 32*h + 16);
            const __m256 x3 = _mm256_loadu_ps(xb + 32*h + 24);
            __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, x0), _mm256_andnot_ps(sign, x1)),
                                     _mm256_max_ps(_mm256_andnot_ps(sign, x2), _mm256_andnot_ps(sign, x3)));
            __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
            m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
            d[h] = _mm_cvtss_f32(m4)/127.0f;
            const __m256 id = _mm256_set1_ps(d[h] != 0.0f ? 1.0f/d[h] : 0.0f);
            // cvtps rounds to nearest even, like nearbyintf in the default rounding mode
            _mm256_store_si256((__m256i *) (v + 32*h +  0), _mm256_cvtps_epi32(_mm256_mul_ps(x0, id)));
            _mm256_store_si256((__m256i *) (v + 32*h +  8), _mm256_cvtps_epi32(_mm256_mul_ps(x1, id)));
            _mm256_store_si256((__m256i *) (v + 32*h + 16), _mm256_cvtps_epi32(_mm256_mul_ps(x2, id)));
            _mm256_store_si256((__m256i *) (v + 32*h + 24), _mm256_cvtps_epi32(_mm256_mul_ps(x3, id)));
        }
        pack_act(v, d, out[b]);
    }
}
#endif // MH_X86

static inline float fp16_to_fp32(uint16_t h) {
    return GGML_FP16_TO_FP32(h);
}

// two weight rows against one activation
static void dot2_generic(const q2_block * w0, const q2_block * w1, const act_block * a, int nb, float * s0, float * s1) {
    float r0 = 0.0f, r1 = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int acc0[16] = {0}, acc1[16] = {0};
        for (int L = 0; L < 4; ++L) {
            for (int j = 0; j < 16; ++j) {
                const int y  = a[b].q[16*L + j];
                const int c0 = (w0[b].qs[j] >> (2*L)) & 3;
                const int c1 = (w1[b].qs[j] >> (2*L)) & 3;
                acc0[4*L + j/4] += c0*y;
                acc1[4*L + j/4] += c1*y;
            }
        }
        float f0 = 0.0f, f1 = 0.0f;
        for (int i = 0; i < 16; ++i) {
            f0 += a[b].s[i]*(float) acc0[i];
            f1 += a[b].s[i]*(float) acc1[i];
        }
        const float d0 = fp16_to_fp32(w0[b].d), d1 = fp16_to_fp32(w1[b].d);
        r0 += d0*(f0 - a[b].corr);
        r1 += d1*(f1 - a[b].corr);
    }
    *s0 = r0;
    *s1 = r1;
}

#if MH_X86
__attribute__((target("avx512f,avx512bw,avx512dq,f16c,fma")))
static void dot2_avx512(const q2_block * w0, const q2_block * w1, const act_block * a, int nb, float * s0, float * s1) {
    const __m512i shifts = _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0);
    const __m512i m3     = _mm512_set1_epi8(3);
    const __m512i ones   = _mm512_set1_epi16(1);
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    float  c0 = 0.0f, c1 = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const __m512i y = _mm512_loadu_si512((const void *) a[b].q);
        const __m512  s = _mm512_loadu_ps(a[b].s);
        const float  d0 = _cvtsh_ss(w0[b].d);
        const float  d1 = _cvtsh_ss(w1[b].d);
        __m512i q0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) w0[b].qs));
        __m512i q1 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) w1[b].qs));
        q0 = _mm512_and_si512(_mm512_srlv_epi64(q0, shifts), m3);
        q1 = _mm512_and_si512(_mm512_srlv_epi64(q1, shifts), m3);
        const __m512i p0 = _mm512_madd_epi16(_mm512_maddubs_epi16(q0, y), ones);
        const __m512i p1 = _mm512_madd_epi16(_mm512_maddubs_epi16(q1, y), ones);
        acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p0), _mm512_mul_ps(s, _mm512_set1_ps(d0)), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p1), _mm512_mul_ps(s, _mm512_set1_ps(d1)), acc1);
        c0 += d0*a[b].corr;
        c1 += d1*a[b].corr;
    }
    *s0 = _mm512_reduce_add_ps(acc0) - c0;
    *s1 = _mm512_reduce_add_ps(acc1) - c1;
}

// Four weight rows against one activation (the down projection's rows of n_ff = 320 are five blocks: two rows per call
// spent a quarter of the time in the call and its two horizontal sums): the activation loads are shared and the four
// sums reduced together. The sums are those of dot2_avx512 up to the order of the last additions (~1e-5 relative).
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,f16c,fma")))
static void dot4_avx512(const q2_block * w0, const q2_block * w1, const q2_block * w2, const q2_block * w3, const act_block * a,
                        int nb, float * out) {
    const __m512i shifts = _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0);
    const __m512i m3     = _mm512_set1_epi8(3);
    const __m512i ones   = _mm512_set1_epi16(1);
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps(), acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    __m128 cs = _mm_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const __m512i y  = _mm512_loadu_si512((const void *) a[b].q);
        const __m512  s  = _mm512_loadu_ps(a[b].s);
        const __m128  dd = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, (short) w3[b].d, (short) w2[b].d, (short) w1[b].d, (short) w0[b].d));
        cs = _mm_fmadd_ps(dd, _mm_set1_ps(a[b].corr), cs);
        const __m512i q0 = _mm512_and_si512(_mm512_srlv_epi64(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) w0[b].qs)), shifts), m3);
        const __m512i q1 = _mm512_and_si512(_mm512_srlv_epi64(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) w1[b].qs)), shifts), m3);
        const __m512i q2 = _mm512_and_si512(_mm512_srlv_epi64(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) w2[b].qs)), shifts), m3);
        const __m512i q3 = _mm512_and_si512(_mm512_srlv_epi64(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) w3[b].qs)), shifts), m3);
        const __m512  f0 = _mm512_cvtepi32_ps(_mm512_madd_epi16(_mm512_maddubs_epi16(q0, y), ones));
        const __m512  f1 = _mm512_cvtepi32_ps(_mm512_madd_epi16(_mm512_maddubs_epi16(q1, y), ones));
        const __m512  f2 = _mm512_cvtepi32_ps(_mm512_madd_epi16(_mm512_maddubs_epi16(q2, y), ones));
        const __m512  f3 = _mm512_cvtepi32_ps(_mm512_madd_epi16(_mm512_maddubs_epi16(q3, y), ones));
        const __m512  dz = _mm512_castps128_ps512(dd);
        acc0 = _mm512_fmadd_ps(f0, _mm512_mul_ps(s, _mm512_permutexvar_ps(_mm512_set1_epi32(0), dz)), acc0);
        acc1 = _mm512_fmadd_ps(f1, _mm512_mul_ps(s, _mm512_permutexvar_ps(_mm512_set1_epi32(1), dz)), acc1);
        acc2 = _mm512_fmadd_ps(f2, _mm512_mul_ps(s, _mm512_permutexvar_ps(_mm512_set1_epi32(2), dz)), acc2);
        acc3 = _mm512_fmadd_ps(f3, _mm512_mul_ps(s, _mm512_permutexvar_ps(_mm512_set1_epi32(3), dz)), acc3);
    }
    // 512 -> 256 halves added, then a 4-way hadd tree (a macro: a lambda would not inherit the target attribute)
#define MH_HALF(v) _mm256_add_ps(_mm512_castps512_ps256(v), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1)))
    const __m256 ab   = _mm256_hadd_ps(MH_HALF(acc0), MH_HALF(acc1));
    const __m256 cd   = _mm256_hadd_ps(MH_HALF(acc2), MH_HALF(acc3));
#undef MH_HALF
    const __m256 abcd = _mm256_hadd_ps(ab, cd);
    _mm_storeu_ps(out, _mm_sub_ps(_mm_add_ps(_mm256_castps256_ps128(abcd), _mm256_extractf128_ps(abcd, 1)), cs));
}

__attribute__((target("avx2,f16c,fma")))
static void dot2_avx2(const q2_block * w0, const q2_block * w1, const act_block * a, int nb, float * s0, float * s1) {
    // lanes 0,1 of the 512-bit layout in the low ymm, lanes 2,3 in the high one
    const __m256i sh_lo = _mm256_set_epi64x(2, 2, 0, 0);
    const __m256i sh_hi = _mm256_set_epi64x(6, 6, 4, 4);
    const __m256i m3    = _mm256_set1_epi8(3);
    const __m256i ones  = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float  c0 = 0.0f, c1 = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const __m256i ylo = _mm256_loadu_si256((const __m256i *) (a[b].q +  0));
        const __m256i yhi = _mm256_loadu_si256((const __m256i *) (a[b].q + 32));
        const __m256  slo = _mm256_loadu_ps(a[b].s + 0);
        const __m256  shi = _mm256_loadu_ps(a[b].s + 8);
        const float   d0  = _cvtsh_ss(w0[b].d);
        const float   d1  = _cvtsh_ss(w1[b].d);
        const __m256i q0  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) w0[b].qs));
        const __m256i q1  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) w1[b].qs));
        const __m256i p0lo = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srlv_epi64(q0, sh_lo), m3), ylo), ones);
        const __m256i p0hi = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srlv_epi64(q0, sh_hi), m3), yhi), ones);
        const __m256i p1lo = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srlv_epi64(q1, sh_lo), m3), ylo), ones);
        const __m256i p1hi = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srlv_epi64(q1, sh_hi), m3), yhi), ones);
        const __m256 vd0 = _mm256_set1_ps(d0);
        const __m256 vd1 = _mm256_set1_ps(d1);
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0lo), _mm256_mul_ps(slo, vd0), acc0);
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0hi), _mm256_mul_ps(shi, vd0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1lo), _mm256_mul_ps(slo, vd1), acc1);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1hi), _mm256_mul_ps(shi, vd1), acc1);
        c0 += d0*a[b].corr;
        c1 += d1*a[b].corr;
    }
    __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
    __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(acc1), _mm256_extractf128_ps(acc1, 1));
    r0 = _mm_hadd_ps(r0, r0); r0 = _mm_hadd_ps(r0, r0);
    r1 = _mm_hadd_ps(r1, r1); r1 = _mm_hadd_ps(r1, r1);
    *s0 = _mm_cvtss_f32(r0) - c0;
    *s1 = _mm_cvtss_f32(r1) - c1;
}
#endif // MH_X86

// ---- Q2_K cold experts (glm5-next): super-blocks of 256 = 16 sub-blocks of 16 with a 4-bit scale and a 4-bit min
// each (scales[s] & 0xF, scales[s] >> 4) under fp16 d/dmin; element e: half h = e/128, shift j = (e%128)/32,
// byte l = e%32: code = (qs[32h + l] >> 2j) & 3, value = d*sc*code - dmin*min. The activations come in natural
// order with two scales per 64 (one per 32) and the sums of each 16 (for the min term).
constexpr int QK_K_ = 256;

struct q2k_block {
    uint8_t  scales[16];
    uint8_t  qs[64];
    uint16_t d;
    uint16_t dmin;
};
static_assert(sizeof(q2k_block) == 84, "unexpected Q2_K block size");

struct alignas(32) nat_block {
    int8_t  q[64];
    float   s[2];
    int32_t bsum[4];
    int32_t pad[2];
};
static_assert(sizeof(nat_block) == 96, "unexpected nat block size");

typedef void (*dot2k_fn)(const uint8_t *, const uint8_t *, const nat_block *, int, float *, float *);
typedef void (*quantn_fn)(const float *, int, nat_block *);

// the GPU's act blocks (element e at q[16*(e%4) + e/4], scales s[0] and s[2]) -> natural order
static void act_to_nat(const act_block * a, int nb, nat_block * o) {
    for (int b = 0; b < nb; ++b) {
        int sum[4] = {0, 0, 0, 0};
        for (int e = 0; e < 64; ++e) {
            const int8_t v = a[b].q[16*(e % 4) + e/4];
            o[b].q[e] = v;
            sum[e/16] += v;
        }
        o[b].s[0] = a[b].s[0];
        o[b].s[1] = a[b].s[2];
        for (int i = 0; i < 4; ++i) {
            o[b].bsum[i] = sum[i];
        }
    }
}

static void quantize_nat_generic(const float * x, int n, nat_block * out) {
    for (int b = 0; b < n/QK; ++b) {
        const float * xb = x + b*QK;
        int sum[4] = {0, 0, 0, 0};
        for (int h = 0; h < 2; ++h) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = std::max(amax, std::fabs(xb[32*h + i]));
            }
            const float d  = amax/127.0f;
            const float id = d != 0.0f ? 1.0f/d : 0.0f;
            out[b].s[h] = d;
            for (int i = 0; i < 32; ++i) {
                const int v = (int) nearbyintf(xb[32*h + i]*id);
                out[b].q[32*h + i] = (int8_t) v;
                sum[2*h + i/16] += v;
            }
        }
        for (int i = 0; i < 4; ++i) {
            out[b].bsum[i] = sum[i];
        }
    }
}

static void dot2_q2k_generic(const uint8_t * w0b, const uint8_t * w1b, const nat_block * a, int nb64, float * s0, float * s1) {
    const q2k_block * w0 = (const q2k_block *) w0b;
    const q2k_block * w1 = (const q2k_block *) w1b;
    float r0 = 0.0f, r1 = 0.0f;
    for (int k = 0; k < nb64/4; ++k) {
        const q2k_block & x0 = w0[k];
        const q2k_block & x1 = w1[k];
        const float d0 = fp16_to_fp32(x0.d), m0 = fp16_to_fp32(x0.dmin);
        const float d1 = fp16_to_fp32(x1.d), m1 = fp16_to_fp32(x1.dmin);
        for (int g = 0; g < 8; ++g) {   // 32-element groups: half g/4, shift g%4; act block 4k + g/2, its half g%2
            const nat_block & ab = a[4*k + g/2];
            const int8_t  * y  = ab.q + 32*(g % 2);
            const uint8_t * q0 = x0.qs + 32*(g/4);
            const uint8_t * q1 = x1.qs + 32*(g/4);
            const int sh = 2*(g % 4);
            int A0a = 0, A0b = 0, A1a = 0, A1b = 0;
            for (int l = 0; l < 16; ++l) {
                A0a += ((q0[l] >> sh) & 3)*y[l];
                A1a += ((q1[l] >> sh) & 3)*y[l];
            }
            for (int l = 16; l < 32; ++l) {
                A0b += ((q0[l] >> sh) & 3)*y[l];
                A1b += ((q1[l] >> sh) & 3)*y[l];
            }
            const int   Ba = ab.bsum[2*(g % 2)], Bb = ab.bsum[2*(g % 2) + 1];
            const int   sa = 2*g, sb = 2*g + 1;
            const float as = ab.s[g % 2];
            r0 += as*(d0*(float) ((x0.scales[sa] & 15)*A0a + (x0.scales[sb] & 15)*A0b) - m0*(float) ((x0.scales[sa] >> 4)*Ba + (x0.scales[sb] >> 4)*Bb));
            r1 += as*(d1*(float) ((x1.scales[sa] & 15)*A1a + (x1.scales[sb] & 15)*A1b) - m1*(float) ((x1.scales[sa] >> 4)*Ba + (x1.scales[sb] >> 4)*Bb));
        }
    }
    *s0 = r0;
    *s1 = r1;
}

// ---- Q3_K cold experts (deepseek41's down projections): super-blocks of 256 = 16 sub-blocks of 16 with a 6-bit
// scale each (packed in scales[12], value - 32) under fp16 d; element e: half h = e/128, shift j = (e%128)/32,
// byte l = e%32: code = ((qs[32h + l] >> 2j) & 3) | (((hmask[l] >> (4h + j)) & 1) << 2), value = d*sc*(code - 4).
struct q3k_block {
    uint8_t  hmask[32];
    uint8_t  qs[64];
    uint8_t  scales[12];
    uint16_t d;
};
static_assert(sizeof(q3k_block) == 110, "unexpected Q3_K block size");

// the 16 sub-block scales of a Q3_K block, 6 bits each as ggml packs them, minus the 32 offset
static inline void q3k_scales(const uint8_t * sc12, int8_t out[16]) {
    uint32_t aux[4];
    memcpy(aux, sc12, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
    const int8_t * sc = (const int8_t *) aux;
    for (int i = 0; i < 16; ++i) {
        out[i] = (int8_t) (sc[i] - 32);
    }
}

static void dot2_q3k_generic(const uint8_t * w0b, const uint8_t * w1b, const nat_block * a, int nb64, float * s0, float * s1) {
    const q3k_block * w0 = (const q3k_block *) w0b;
    const q3k_block * w1 = (const q3k_block *) w1b;
    float r0 = 0.0f, r1 = 0.0f;
    for (int k = 0; k < nb64/4; ++k) {
        const q3k_block & x0 = w0[k];
        const q3k_block & x1 = w1[k];
        const float d0 = fp16_to_fp32(x0.d), d1 = fp16_to_fp32(x1.d);
        int8_t sc0[16], sc1[16];
        q3k_scales(x0.scales, sc0);
        q3k_scales(x1.scales, sc1);
        for (int g = 0; g < 8; ++g) {   // 32-element groups: half g/4, shift g%4; act block 4k + g/2, its half g%2
            const nat_block & ab = a[4*k + g/2];
            const int8_t  * y  = ab.q + 32*(g % 2);
            const uint8_t * q0 = x0.qs + 32*(g/4);
            const uint8_t * q1 = x1.qs + 32*(g/4);
            const int sh = 2*(g % 4), hb = 4*(g/4) + g % 4;
            int A0a = 0, A0b = 0, A1a = 0, A1b = 0;
            for (int l = 0; l < 16; ++l) {
                A0a += (((q0[l] >> sh) & 3) | (((x0.hmask[l] >> hb) & 1) << 2))*y[l];
                A1a += (((q1[l] >> sh) & 3) | (((x1.hmask[l] >> hb) & 1) << 2))*y[l];
            }
            for (int l = 16; l < 32; ++l) {
                A0b += (((q0[l] >> sh) & 3) | (((x0.hmask[l] >> hb) & 1) << 2))*y[l];
                A1b += (((q1[l] >> sh) & 3) | (((x1.hmask[l] >> hb) & 1) << 2))*y[l];
            }
            const int   Ba = ab.bsum[2*(g % 2)], Bb = ab.bsum[2*(g % 2) + 1];
            const int   sa = 2*g, sb = 2*g + 1;
            const float as = ab.s[g % 2];
            r0 += as*d0*(float) (sc0[sa]*(A0a - 4*Ba) + sc0[sb]*(A0b - 4*Bb));
            r1 += as*d1*(float) (sc1[sa]*(A1a - 4*Ba) + sc1[sb]*(A1b - 4*Bb));
        }
    }
    *s0 = r0;
    *s1 = r1;
}

#if MH_X86
__attribute__((target("avx2,f16c,fma")))
static void quantize_nat_avx2(const float * x, int n, nat_block * out) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int b = 0; b < n/QK; ++b) {
        const float * xb = x + b*QK;
        alignas(32) int32_t v[64];
        for (int h = 0; h < 2; ++h) {
            const __m256 x0 = _mm256_loadu_ps(xb + 32*h +  0);
            const __m256 x1 = _mm256_loadu_ps(xb + 32*h +  8);
            const __m256 x2 = _mm256_loadu_ps(xb + 32*h + 16);
            const __m256 x3 = _mm256_loadu_ps(xb + 32*h + 24);
            __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, x0), _mm256_andnot_ps(sign, x1)),
                                     _mm256_max_ps(_mm256_andnot_ps(sign, x2), _mm256_andnot_ps(sign, x3)));
            __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
            m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
            const float d = _mm_cvtss_f32(m4)/127.0f;
            out[b].s[h] = d;
            const __m256 id = _mm256_set1_ps(d != 0.0f ? 1.0f/d : 0.0f);
            _mm256_store_si256((__m256i *) (v + 32*h +  0), _mm256_cvtps_epi32(_mm256_mul_ps(x0, id)));
            _mm256_store_si256((__m256i *) (v + 32*h +  8), _mm256_cvtps_epi32(_mm256_mul_ps(x1, id)));
            _mm256_store_si256((__m256i *) (v + 32*h + 16), _mm256_cvtps_epi32(_mm256_mul_ps(x2, id)));
            _mm256_store_si256((__m256i *) (v + 32*h + 24), _mm256_cvtps_epi32(_mm256_mul_ps(x3, id)));
        }
        int sum[4] = {0, 0, 0, 0};
        for (int e = 0; e < 64; ++e) {
            out[b].q[e] = (int8_t) v[e];
            sum[e/16] += v[e];
        }
        for (int i = 0; i < 4; ++i) {
            out[b].bsum[i] = sum[i];
        }
    }
}

__attribute__((target("avx2,f16c,fma")))
static inline float mh_hsum256(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,f16c,fma")))
static void dot2_q2k_avx2(const uint8_t * w0b, const uint8_t * w1b, const nat_block * a, int nb64, float * s0, float * s1) {
    const q2k_block * w0 = (const q2k_block *) w0b;
    const q2k_block * w1 = (const q2k_block *) w1b;
    const __m256i m3 = _mm256_set1_epi8(3);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float  mins0 = 0.0f, mins1 = 0.0f;
    for (int k = 0; k < nb64/4; ++k) {
        const q2k_block & x0 = w0[k];
        const q2k_block & x1 = w1[k];
        const float d0 = fp16_to_fp32(x0.d), m0 = fp16_to_fp32(x0.dmin);
        const float d1 = fp16_to_fp32(x1.d), m1 = fp16_to_fp32(x1.dmin);
        for (int h = 0; h < 2; ++h) {
            const __m256i qh0 = _mm256_loadu_si256((const __m256i *) (x0.qs + 32*h));
            const __m256i qh1 = _mm256_loadu_si256((const __m256i *) (x1.qs + 32*h));
            for (int j = 0; j < 4; ++j) {
                const int g = 4*h + j;
                const nat_block & ab = a[4*k + g/2];
                const __m256i y  = _mm256_loadu_si256((const __m256i *) (ab.q + 32*(g % 2)));
                const __m256i q0 = _mm256_and_si256(_mm256_srli_epi16(qh0, 2*j), m3);
                const __m256i q1 = _mm256_and_si256(_mm256_srli_epi16(qh1, 2*j), m3);
                const int sa = 2*g, sb = 2*g + 1;
                const __m256i sc0 = _mm256_set_m128i(_mm_set1_epi16(x0.scales[sb] & 15), _mm_set1_epi16(x0.scales[sa] & 15));
                const __m256i sc1 = _mm256_set_m128i(_mm_set1_epi16(x1.scales[sb] & 15), _mm_set1_epi16(x1.scales[sa] & 15));
                const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(q0, y), sc0);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q1, y), sc1);
                const float as = ab.s[g % 2];
                acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(as*d0), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(as*d1), acc1);
                const int Ba = ab.bsum[2*(g % 2)], Bb = ab.bsum[2*(g % 2) + 1];
                mins0 += as*m0*(float) ((x0.scales[sa] >> 4)*Ba + (x0.scales[sb] >> 4)*Bb);
                mins1 += as*m1*(float) ((x1.scales[sa] >> 4)*Ba + (x1.scales[sb] >> 4)*Bb);
            }
        }
    }
    *s0 = mh_hsum256(acc0) - mins0;
    *s1 = mh_hsum256(acc1) - mins1;
}

// 64 codes per op: lanes 0-31 = half 0 of the super-block at shift j (elements 32j..), lanes 32-63 = half 1 at the
// same shift (elements 128 + 32j..); the activations of those come from the nat blocks j/2 and 2 + j/2, half j%2.
// The min term of a super-block is one 16-lane fmadd over its sub-blocks.
__attribute__((target("avx512f,avx512bw,avx512dq,avx2,f16c,fma")))
static void dot2_q2k_avx512(const uint8_t * w0b, const uint8_t * w1b, const nat_block * a, int nb64, float * s0, float * s1) {
    const q2k_block * w0 = (const q2k_block *) w0b;
    const q2k_block * w1 = (const q2k_block *) w1b;
    const __m512i m3 = _mm512_set1_epi8(3);
    // lane permutes: int16 lanes 0-7 <- sub-block 2j, 8-15 <- 2j+1, 16-23 <- 8+2j, 24-31 <- 8+2j+1 (the scales of
    // the two 32-element groups at shift j in the two halves); float lanes 0-7 <- group j, 8-15 <- group 4+j
    static const __m512i sidx[4] = {
        _mm512_set_epi16( 9, 9, 9, 9, 9, 9, 9, 9,  8, 8, 8, 8, 8, 8, 8, 8,  1, 1, 1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0, 0),
        _mm512_set_epi16(11,11,11,11,11,11,11,11, 10,10,10,10,10,10,10,10,  3, 3, 3, 3, 3, 3, 3, 3,  2, 2, 2, 2, 2, 2, 2, 2),
        _mm512_set_epi16(13,13,13,13,13,13,13,13, 12,12,12,12,12,12,12,12,  5, 5, 5, 5, 5, 5, 5, 5,  4, 4, 4, 4, 4, 4, 4, 4),
        _mm512_set_epi16(15,15,15,15,15,15,15,15, 14,14,14,14,14,14,14,14,  7, 7, 7, 7, 7, 7, 7, 7,  6, 6, 6, 6, 6, 6, 6, 6),
    };
    static const __m512i aidx[4] = {
        _mm512_set_epi32( 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
        _mm512_set_epi32(10,10,10,10,10,10,10,10, 2, 2, 2, 2, 2, 2, 2, 2),
        _mm512_set_epi32(12,12,12,12,12,12,12,12, 4, 4, 4, 4, 4, 4, 4, 4),
        _mm512_set_epi32(14,14,14,14,14,14,14,14, 6, 6, 6, 6, 6, 6, 6, 6),
    };
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 accm0 = _mm512_setzero_ps(), accm1 = _mm512_setzero_ps();
    for (int k = 0; k < nb64/4; ++k) {
        const q2k_block & x0 = w0[k];
        const q2k_block & x1 = w1[k];
        const nat_block * ab = a + 4*k;
        const __m512 d0 = _mm512_set1_ps(fp16_to_fp32(x0.d)), dm0 = _mm512_set1_ps(fp16_to_fp32(x0.dmin));
        const __m512 d1 = _mm512_set1_ps(fp16_to_fp32(x1.d)), dm1 = _mm512_set1_ps(fp16_to_fp32(x1.dmin));
        const __m512i q0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) x0.qs)), _mm256_loadu_si256((const __m256i *) (x0.qs + 32)), 1);
        const __m512i q1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) x1.qs)), _mm256_loadu_si256((const __m256i *) (x1.qs + 32)), 1);
        // the 16 sub-block scales (low nibbles) and mins (high nibbles) of each row, int16 lanes 0..15 (16..31 zero)
        const __m512i sc0 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) x0.scales)));
        const __m512i sc1 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) x1.scales)));
        const __m512i lo0 = _mm512_and_si512(sc0, _mm512_set1_epi16(0x0f)), hi0 = _mm512_srli_epi16(sc0, 4);
        const __m512i lo1 = _mm512_and_si512(sc1, _mm512_set1_epi16(0x0f)), hi1 = _mm512_srli_epi16(sc1, 4);
        // the activations' sums per sub-block and scales per sub-block (s/4 -> nat block, (s%4)/2 -> half)
        const __m512 bsum = _mm512_cvtepi32_ps(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(
                          _mm_loadu_si128((const __m128i *) ab[0].bsum)), _mm_loadu_si128((const __m128i *) ab[1].bsum), 1),
                          _mm_loadu_si128((const __m128i *) ab[2].bsum), 2), _mm_loadu_si128((const __m128i *) ab[3].bsum), 3));
        const __m512 as16 = _mm512_set_ps(ab[3].s[1], ab[3].s[1], ab[3].s[0], ab[3].s[0], ab[2].s[1], ab[2].s[1], ab[2].s[0], ab[2].s[0],
                                          ab[1].s[1], ab[1].s[1], ab[1].s[0], ab[1].s[0], ab[0].s[1], ab[0].s[1], ab[0].s[0], ab[0].s[0]);
        const __m512 ba = _mm512_mul_ps(bsum, as16);
        accm0 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(hi0))), dm0), ba, accm0);
        accm1 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(hi1))), dm1), ba, accm1);
        for (int j = 0; j < 4; ++j) {
            const __m512i c0 = _mm512_and_si512(_mm512_srli_epi16(q0, 2*j), m3);
            const __m512i c1 = _mm512_and_si512(_mm512_srli_epi16(q1, 2*j), m3);
            const __m512i y  = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) (ab[j/2].q + 32*(j % 2)))),
                                                  _mm256_loadu_si256((const __m256i *) (ab[2 + j/2].q + 32*(j % 2))), 1);
            const __m512i p0 = _mm512_madd_epi16(_mm512_maddubs_epi16(c0, y), _mm512_permutexvar_epi16(sidx[j], lo0));
            const __m512i p1 = _mm512_madd_epi16(_mm512_maddubs_epi16(c1, y), _mm512_permutexvar_epi16(sidx[j], lo1));
            const __m512  asv = _mm512_permutexvar_ps(aidx[j], as16);
            acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p0), _mm512_mul_ps(asv, d0), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p1), _mm512_mul_ps(asv, d1), acc1);
        }
    }
    *s0 = _mm512_reduce_add_ps(acc0) - _mm512_reduce_add_ps(accm0);
    *s1 = _mm512_reduce_add_ps(acc1) - _mm512_reduce_add_ps(accm1);
}

__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,f16c,fma")))
static void dot2_q3k_avx512(const uint8_t * w0b, const uint8_t * w1b, const nat_block * a, int nb64, float * s0, float * s1) {
    const q3k_block * w0 = (const q3k_block *) w0b;
    const q3k_block * w1 = (const q3k_block *) w1b;
    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i m1 = _mm512_set1_epi8(1);
    // the same lane permutes as the Q2_K kernel: int16 lanes 0-7 <- sub-block 2j, 8-15 <- 2j+1, 16-23 <- 8+2j,
    // 24-31 <- 8+2j+1; float lanes 0-7 <- group j, 8-15 <- group 4+j
    static const __m512i sidx[4] = {
        _mm512_set_epi16( 9, 9, 9, 9, 9, 9, 9, 9,  8, 8, 8, 8, 8, 8, 8, 8,  1, 1, 1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0, 0),
        _mm512_set_epi16(11,11,11,11,11,11,11,11, 10,10,10,10,10,10,10,10,  3, 3, 3, 3, 3, 3, 3, 3,  2, 2, 2, 2, 2, 2, 2, 2),
        _mm512_set_epi16(13,13,13,13,13,13,13,13, 12,12,12,12,12,12,12,12,  5, 5, 5, 5, 5, 5, 5, 5,  4, 4, 4, 4, 4, 4, 4, 4),
        _mm512_set_epi16(15,15,15,15,15,15,15,15, 14,14,14,14,14,14,14,14,  7, 7, 7, 7, 7, 7, 7, 7,  6, 6, 6, 6, 6, 6, 6, 6),
    };
    static const __m512i aidx[4] = {
        _mm512_set_epi32( 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
        _mm512_set_epi32(10,10,10,10,10,10,10,10, 2, 2, 2, 2, 2, 2, 2, 2),
        _mm512_set_epi32(12,12,12,12,12,12,12,12, 4, 4, 4, 4, 4, 4, 4, 4),
        _mm512_set_epi32(14,14,14,14,14,14,14,14, 6, 6, 6, 6, 6, 6, 6, 6),
    };
    // the high bit of the codes: byte l of half h at shift j sits at bit 4h + j of hmask[l]; the 512-bit hmask
    // vector carries the 32 bytes twice, the lanes of half 1 shifted by 4 more
    static const __m512i hshift[4] = {
        _mm512_set_epi16(4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
        _mm512_set_epi16(5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1),
        _mm512_set_epi16(6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2),
        _mm512_set_epi16(7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7, 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3),
    };
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 accm0 = _mm512_setzero_ps(), accm1 = _mm512_setzero_ps();
    for (int k = 0; k < nb64/4; ++k) {
        const q3k_block & x0 = w0[k];
        const q3k_block & x1 = w1[k];
        const nat_block * ab = a + 4*k;
        const __m512 d0 = _mm512_set1_ps(fp16_to_fp32(x0.d)), d1 = _mm512_set1_ps(fp16_to_fp32(x1.d));
        const __m512i q0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) x0.qs)), _mm256_loadu_si256((const __m256i *) (x0.qs + 32)), 1);
        const __m512i q1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) x1.qs)), _mm256_loadu_si256((const __m256i *) (x1.qs + 32)), 1);
        const __m256i hm0h = _mm256_loadu_si256((const __m256i *) x0.hmask);
        const __m256i hm1h = _mm256_loadu_si256((const __m256i *) x1.hmask);
        const __m512i hm0 = _mm512_inserti64x4(_mm512_castsi256_si512(hm0h), hm0h, 1);
        const __m512i hm1 = _mm512_inserti64x4(_mm512_castsi256_si512(hm1h), hm1h, 1);
        // the 16 sub-block scales (6 bits - 32, signed) of each row, int16 lanes 0..15
        alignas(16) int8_t s0b[16], s1b[16];
        q3k_scales(x0.scales, s0b);
        q3k_scales(x1.scales, s1b);
        const __m512i sc0 = _mm512_cvtepi8_epi16(_mm256_castsi128_si256(_mm_load_si128((const __m128i *) s0b)));
        const __m512i sc1 = _mm512_cvtepi8_epi16(_mm256_castsi128_si256(_mm_load_si128((const __m128i *) s1b)));
        // the activations' sums per sub-block and scales per sub-block (s/4 -> nat block, (s%4)/2 -> half)
        const __m512 bsum = _mm512_cvtepi32_ps(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(
                          _mm_loadu_si128((const __m128i *) ab[0].bsum)), _mm_loadu_si128((const __m128i *) ab[1].bsum), 1),
                          _mm_loadu_si128((const __m128i *) ab[2].bsum), 2), _mm_loadu_si128((const __m128i *) ab[3].bsum), 3));
        const __m512 as16 = _mm512_set_ps(ab[3].s[1], ab[3].s[1], ab[3].s[0], ab[3].s[0], ab[2].s[1], ab[2].s[1], ab[2].s[0], ab[2].s[0],
                                          ab[1].s[1], ab[1].s[1], ab[1].s[0], ab[1].s[0], ab[0].s[1], ab[0].s[1], ab[0].s[0], ab[0].s[0]);
        // the - 4 of every code: 4 * bsum * scale * d per sub-block
        const __m512 ba4 = _mm512_mul_ps(_mm512_mul_ps(bsum, as16), _mm512_set1_ps(4.0f));
        accm0 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(sc0))), d0), ba4, accm0);
        accm1 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(sc1))), d1), ba4, accm1);
        for (int j = 0; j < 4; ++j) {
            const __m512i h0 = _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(hm0, hshift[j]), m1), 2);
            const __m512i h1 = _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(hm1, hshift[j]), m1), 2);
            const __m512i c0 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(q0, 2*j), m3), h0);
            const __m512i c1 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(q1, 2*j), m3), h1);
            const __m512i y  = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) (ab[j/2].q + 32*(j % 2)))),
                                                  _mm256_loadu_si256((const __m256i *) (ab[2 + j/2].q + 32*(j % 2))), 1);
            const __m512i p0 = _mm512_madd_epi16(_mm512_maddubs_epi16(c0, y), _mm512_permutexvar_epi16(sidx[j], sc0));
            const __m512i p1 = _mm512_madd_epi16(_mm512_maddubs_epi16(c1, y), _mm512_permutexvar_epi16(sidx[j], sc1));
            const __m512  asv = _mm512_permutexvar_ps(aidx[j], as16);
            acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p0), _mm512_mul_ps(asv, d0), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p1), _mm512_mul_ps(asv, d1), acc1);
        }
    }
    *s0 = _mm512_reduce_add_ps(acc0) - _mm512_reduce_add_ps(accm0);
    *s1 = _mm512_reduce_add_ps(acc1) - _mm512_reduce_add_ps(accm1);
}
#endif // MH_X86

// ---- the "k" kernels of Q2_K / Q3_K cold experts (GGML_CUDA_MOE_HOST_KERNEL, default new). The partial sums of a vector
// are int32 and are scaled by the sub-block scales with vpmaddwd, so one convert and one fma per 32 weights stay in the
// float domain (the old kernels converted and scaled twice per 32 and permuted the scales and the activation scales).
// The activations come in blocks of 256 laid out as the vectors read them:
//   q:   vector j (j = 0..3) is 64 bytes: the values 32j..32j+31, then 128+32j..128+32j+31 (the two groups of 32 that sit
//        at 2-bit shift j in the two halves of a super-block)
//   as:  vector j's float scale for each of its 16 int32 lanes (lanes 0..7 the first group, 8..15 the second)
//   bs:  the sum of every 16 values (the sub-blocks in element order), asg: the scale of every 32 values (g = 4*half + j)
struct alignas(64) kact_block {
    int8_t  q[256];
    float   as[64];
    int16_t bs[16];
    float   asg[8];
};
static_assert(sizeof(kact_block) == 576, "unexpected k act block size");

#if MH_X86
static inline int kact_sum16(const int8_t * q) {
    // sad works on unsigned bytes: bias by 128 and take it off again
    const __m128i v = _mm_xor_si128(_mm_loadu_si128((const __m128i *) q), _mm_set1_epi8((char) 0x80));
    const __m128i s = _mm_sad_epu8(v, _mm_setzero_si128());
    return _mm_cvtsi128_si32(s) + _mm_cvtsi128_si32(_mm_srli_si128(s, 8)) - 16*128;
}
#else
static inline int kact_sum16(const int8_t * q) {
    int s = 0;
    for (int i = 0; i < 16; ++i) {
        s += q[i];
    }
    return s;
}
#endif

// fills as and bs from q and asg
static void kact_finish(kact_block & o) {
    for (int j = 0; j < 4; ++j) {
        for (int t = 0; t < 8; ++t) {
            o.as[16*j + t]     = o.asg[j];
            o.as[16*j + 8 + t] = o.asg[4 + j];
        }
    }
    for (int g = 0; g < 8; ++g) {
        const int8_t * q = o.q + 64*(g % 4) + 32*(g/4);
        o.bs[2*g + 0] = (int16_t) kact_sum16(q);
        o.bs[2*g + 1] = (int16_t) kact_sum16(q + 16);
    }
}

// the GPU's act blocks (64 values, lane order, scales s[0] and s[2]) of nsb super-blocks -> k blocks
static void kact_from_act_generic(const act_block * a, int nsb, kact_block * o) {
    for (int k = 0; k < nsb; ++k) {
        for (int ab = 0; ab < 4; ++ab) {
            const act_block & x = a[4*k + ab];
            for (int e = 0; e < 64; ++e) {
                o[k].q[64*(2*(ab % 2) + e/32) + 32*(ab/2) + e % 32] = x.q[16*(e % 4) + e/4];
            }
            o[k].asg[2*ab + 0] = x.s[0];
            o[k].asg[2*ab + 1] = x.s[2];
        }
        kact_finish(o[k]);
    }
}

// the same values, rounding like quantize_nat_*: 32 values share a scale
static void kact_quantize_generic(const float * x, kact_block & o) {
    for (int g = 0; g < 8; ++g) {
        const float * xg = x + 32*g;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            amax = std::max(amax, std::fabs(xg[i]));
        }
        const float d  = amax/127.0f;
        const float id = d != 0.0f ? 1.0f/d : 0.0f;
        o.asg[g] = d;
        int8_t * q = o.q + 64*(g % 4) + 32*(g/4);
        for (int i = 0; i < 32; ++i) {
            q[i] = (int8_t) nearbyintf(xg[i]*id);
        }
    }
    kact_finish(o);
}

#if MH_X86
// the 4x16 byte transpose of the lane order is four unpacks per 32 values (SSE2)
static void kact_from_act_sse(const act_block * a, int nsb, kact_block * o) {
    for (int k = 0; k < nsb; ++k) {
        for (int ab = 0; ab < 4; ++ab) {
            const act_block & x = a[4*k + ab];
            const __m128i r0 = _mm_loadu_si128((const __m128i *) (x.q +  0));
            const __m128i r1 = _mm_loadu_si128((const __m128i *) (x.q + 16));
            const __m128i r2 = _mm_loadu_si128((const __m128i *) (x.q + 32));
            const __m128i r3 = _mm_loadu_si128((const __m128i *) (x.q + 48));
            const __m128i t01l = _mm_unpacklo_epi8(r0, r1), t01h = _mm_unpackhi_epi8(r0, r1);
            const __m128i t23l = _mm_unpacklo_epi8(r2, r3), t23h = _mm_unpackhi_epi8(r2, r3);
            int8_t * d0 = o[k].q + 64*(2*(ab % 2) + 0) + 32*(ab/2);
            int8_t * d1 = o[k].q + 64*(2*(ab % 2) + 1) + 32*(ab/2);
            _mm_store_si128((__m128i *) (d0 +  0), _mm_unpacklo_epi16(t01l, t23l));
            _mm_store_si128((__m128i *) (d0 + 16), _mm_unpackhi_epi16(t01l, t23l));
            _mm_store_si128((__m128i *) (d1 +  0), _mm_unpacklo_epi16(t01h, t23h));
            _mm_store_si128((__m128i *) (d1 + 16), _mm_unpackhi_epi16(t01h, t23h));
            o[k].asg[2*ab + 0] = x.s[0];
            o[k].asg[2*ab + 1] = x.s[2];
        }
        kact_finish(o[k]);
    }
}

__attribute__((target("avx2,fma")))
static void kact_quantize_avx2(const float * x, kact_block & o) {
    const __m256  sign = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int g = 0; g < 8; ++g) {
        const float * xg = x + 32*g;
        const __m256 x0 = _mm256_loadu_ps(xg +  0);
        const __m256 x1 = _mm256_loadu_ps(xg +  8);
        const __m256 x2 = _mm256_loadu_ps(xg + 16);
        const __m256 x3 = _mm256_loadu_ps(xg + 24);
        const __m256 m  = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, x0), _mm256_andnot_ps(sign, x1)),
                                        _mm256_max_ps(_mm256_andnot_ps(sign, x2), _mm256_andnot_ps(sign, x3)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float d = _mm_cvtss_f32(m4)/127.0f;
        o.asg[g] = d;
        const __m256 id = _mm256_set1_ps(d != 0.0f ? 1.0f/d : 0.0f);
        const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(x0, id));
        const __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(x1, id));
        const __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(x2, id));
        const __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(x3, id));
        const __m256i b  = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        _mm256_store_si256((__m256i *) (o.q + 64*(g % 4) + 32*(g/4)), _mm256_permutevar8x32_epi32(b, perm));
    }
    kact_finish(o);
}

// vpshufb index: 16-bit lane t of 128-bit lane L takes scale byte k_L (the byte lands in the low half of the lane, or in the
// high half with hi: then the lane is scale*256, which carries the sign of a signed scale)
static inline uint32_t kshuf_word(int k, bool hi) {
    return hi ? 0x00800080u + (uint32_t) k*0x01000100u : 0x80008000u + (uint32_t) k*0x00010001u;
}

__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,f16c,fma")))
static inline __m512i kshuf512(int j, bool hi) {
    // 128-bit lanes: sub-blocks 2j, 2j+1 (first half of the vector) and 8+2j, 9+2j (second half)
    const int a = (int) kshuf_word(2*j, hi), b = (int) kshuf_word(2*j + 1, hi);
    const int c = (int) kshuf_word(8 + 2*j, hi), d = (int) kshuf_word(9 + 2*j, hi);
    return _mm512_set_epi32(d, d, d, d, c, c, c, c, b, b, b, b, a, a, a, a);
}

__attribute__((target("avx2,f16c,fma")))
static inline __m256i kshuf256(int g, bool hi) {
    // group g = 4*half + j: sub-blocks 2g and 2g+1
    const int a = (int) kshuf_word(2*g, hi), b = (int) kshuf_word(2*g + 1, hi);
    return _mm256_set_epi32(b, b, b, b, a, a, a, a);
}

__attribute__((target("avx2,f16c,fma")))
static inline float kh256(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

// ---- Q2_K, NR rows against one activation (nsb super-blocks): out[r] = sum of the dequantized row r times the activation
template <int NR>
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,f16c,fma")))
static void kdot_q2k_avx512(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const __m512i m3  = _mm512_set1_epi8(3);
    const __m512i m0f = _mm512_set1_epi8(0x0f);
    const __m128i n0f = _mm_set1_epi8(0x0f);
    const __m512i ix0 = kshuf512(0, false), ix1 = kshuf512(1, false), ix2 = kshuf512(2, false), ix3 = kshuf512(3, false);
    __m512 acc[NR];
    __m256 am[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm512_setzero_ps();
        am[r]  = _mm256_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        const __m256  asg = _mm256_load_ps(ab.asg);
        const __m256i bs  = _mm256_load_si256((const __m256i *) ab.bs);
        const __m512i y0 = _mm512_load_si512((const void *) (ab.q +   0)), y1 = _mm512_load_si512((const void *) (ab.q +  64));
        const __m512i y2 = _mm512_load_si512((const void *) (ab.q + 128)), y3 = _mm512_load_si512((const void *) (ab.q + 192));
        const __m512 as0 = _mm512_load_ps(ab.as +  0), as1 = _mm512_load_ps(ab.as + 16);
        const __m512 as2 = _mm512_load_ps(ab.as + 32), as3 = _mm512_load_ps(ab.as + 48);
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*84;
            const __m128i  sc  = _mm_loadu_si128((const __m128i *) p);
            const __m512i  qv  = _mm512_loadu_si512((const void *) (p + 16));
            int32_t dd;
            memcpy(&dd, p + 80, 4);
            const __m128 df = _mm_cvtph_ps(_mm_cvtsi32_si128(dd));      // d, dmin
            const __m512i scl = _mm512_and_si512(_mm512_broadcast_i32x4(sc), m0f);
            const __m512i c0 = _mm512_and_si512(qv, m3);
            const __m512i c1 = _mm512_and_si512(_mm512_srli_epi16(qv, 2), m3);
            const __m512i c2 = _mm512_and_si512(_mm512_srli_epi16(qv, 4), m3);
            const __m512i c3 = _mm512_and_si512(_mm512_srli_epi16(qv, 6), m3);
            const __m512i s0 = _mm512_madd_epi16(_mm512_maddubs_epi16(c0, y0), _mm512_shuffle_epi8(scl, ix0));
            const __m512i s1 = _mm512_madd_epi16(_mm512_maddubs_epi16(c1, y1), _mm512_shuffle_epi8(scl, ix1));
            const __m512i s2 = _mm512_madd_epi16(_mm512_maddubs_epi16(c2, y2), _mm512_shuffle_epi8(scl, ix2));
            const __m512i s3 = _mm512_madd_epi16(_mm512_maddubs_epi16(c3, y3), _mm512_shuffle_epi8(scl, ix3));
            __m512 sb = _mm512_mul_ps(_mm512_cvtepi32_ps(s0), as0);
            sb = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s1), as1, sb);
            sb = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s2), as2, sb);
            sb = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s3), as3, sb);
            acc[r] = _mm512_fmadd_ps(sb, _mm512_broadcastss_ps(df), acc[r]);
            // the mins: dmin * sum over groups of scale * (min_a * sum_a + min_b * sum_b)
            const __m256i mn = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sc, 4), n0f));
            const __m256  mf = _mm256_cvtepi32_ps(_mm256_madd_epi16(mn, bs));
            const __m256  dm = _mm256_broadcastss_ps(_mm_shuffle_ps(df, df, 0x55));
            am[r] = _mm256_fmadd_ps(_mm256_mul_ps(mf, dm), asg, am[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = _mm512_reduce_add_ps(acc[r]) - kh256(am[r]);
    }
}

template <int NR>
__attribute__((target("avx2,f16c,fma")))
static void kdot_q2k_avx2(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const __m256i m3  = _mm256_set1_epi8(3);
    const __m128i n0f = _mm_set1_epi8(0x0f);
    __m256i ix[8];
    for (int g = 0; g < 8; ++g) {
        ix[g] = kshuf256(g, false);
    }
    __m256 acc[NR];
    __m256 am[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm256_setzero_ps();
        am[r]  = _mm256_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        const __m256  asg = _mm256_load_ps(ab.asg);
        const __m256i bs  = _mm256_load_si256((const __m256i *) ab.bs);
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*84;
            const __m128i  sc  = _mm_loadu_si128((const __m128i *) p);
            const __m256i  scl = _mm256_broadcastsi128_si256(_mm_and_si128(sc, n0f));
            int32_t dd;
            memcpy(&dd, p + 80, 4);
            const __m128 df = _mm_cvtph_ps(_mm_cvtsi32_si128(dd));
            __m256 sb = _mm256_setzero_ps();
#pragma GCC unroll 2
            for (int h = 0; h < 2; ++h) {
                const __m256i qh = _mm256_loadu_si256((const __m256i *) (p + 16 + 32*h));
#pragma GCC unroll 4
                for (int j = 0; j < 4; ++j) {
                    const int g = 4*h + j;
                    const __m256i c  = _mm256_and_si256(j == 0 ? qh : _mm256_srli_epi16(qh, 2*j), m3);
                    const __m256i y  = _mm256_load_si256((const __m256i *) (ab.q + 64*j + 32*h));
                    const __m256i pr = _mm256_madd_epi16(_mm256_maddubs_epi16(c, y), _mm256_shuffle_epi8(scl, ix[g]));
                    sb = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pr), _mm256_broadcast_ss(ab.asg + g), sb);
                }
            }
            acc[r] = _mm256_fmadd_ps(sb, _mm256_broadcastss_ps(df), acc[r]);
            const __m256i mn = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sc, 4), n0f));
            const __m256  mf = _mm256_cvtepi32_ps(_mm256_madd_epi16(mn, bs));
            const __m256  dm = _mm256_broadcastss_ps(_mm_shuffle_ps(df, df, 0x55));
            am[r] = _mm256_fmadd_ps(_mm256_mul_ps(mf, dm), asg, am[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = kh256(acc[r]) - kh256(am[r]);
    }
}

// the 16 signed sub-block scales of a Q3_K block (12 bytes of 6-bit scales as ggml packs them, minus 32), as bytes
__attribute__((target("avx2,f16c,fma")))
static inline __m128i kq3_scales(const uint8_t * sc12) {
    int64_t lo;
    int32_t hi;
    memcpy(&lo, sc12, 8);
    memcpy(&hi, sc12 + 8, 4);
    // dwords 0,1 take the low nibbles of scales 0..3 and 4..7, dwords 2,3 the high nibbles; the top two bits of every
    // scale come from the 4 bytes at 8..11: dword n takes bits 2n, 2n+1 of each byte, shifted to bits 4,5
    const __m128i a = _mm_shuffle_epi32(_mm_cvtsi64_si128(lo), 0x44);
    const __m128i x = _mm_and_si128(_mm_srlv_epi32(a, _mm_set_epi32(4, 4, 0, 0)), _mm_set1_epi8(0x0f));
    const __m128i t = _mm_slli_epi32(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(hi), _mm_set_epi32(6, 4, 2, 0)), _mm_set1_epi8(3)), 4);
    return _mm_sub_epi8(_mm_or_si128(x, t), _mm_set1_epi8(32));
}

// ---- Q3_K: code = low 2 bits | (high bit << 2), value = d * scale * (code - 4); the - 4 rides on the activation sums
template <int NR>
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,f16c,fma")))
static void kdot_q3k_avx512(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i m4 = _mm512_set1_epi8(4);
    // the high bits of the second half of a super-block sit 4 bits higher: shift that half down once per block
    const __m512i hsh = _mm512_set_epi32(0x00040004, 0x00040004, 0x00040004, 0x00040004, 0x00040004, 0x00040004, 0x00040004, 0x00040004, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i ix0 = kshuf512(0, true), ix1 = kshuf512(1, true), ix2 = kshuf512(2, true), ix3 = kshuf512(3, true);
    __m512 acc[NR];
    __m256 am[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm512_setzero_ps();
        am[r]  = _mm256_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        const __m256  asg = _mm256_load_ps(ab.asg);
        const __m256i bs  = _mm256_load_si256((const __m256i *) ab.bs);
        const __m512i y0 = _mm512_load_si512((const void *) (ab.q +   0)), y1 = _mm512_load_si512((const void *) (ab.q +  64));
        const __m512i y2 = _mm512_load_si512((const void *) (ab.q + 128)), y3 = _mm512_load_si512((const void *) (ab.q + 192));
        const __m512 as0 = _mm512_load_ps(ab.as +  0), as1 = _mm512_load_ps(ab.as + 16);
        const __m512 as2 = _mm512_load_ps(ab.as + 32), as3 = _mm512_load_ps(ab.as + 48);
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*110;
            const __m512i  hm  = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *) p));
            const __m512i  hv  = _mm512_srlv_epi16(hm, hsh);
            const __m512i  qv  = _mm512_loadu_si512((const void *) (p + 32));
            const __m128i  sc  = kq3_scales(p + 96);
            uint16_t dh;
            memcpy(&dh, p + 108, 2);
            const float d = _cvtsh_ss(dh);
            const __m512i scl = _mm512_broadcast_i32x4(sc);
            // code j: low bits of qv >> 2j, bit j of the high byte moved to bit 2
            const __m512i k0 = _mm512_ternarylogic_epi32(_mm512_and_si512(qv, m3), _mm512_slli_epi16(hv, 2), m4, 0xF8);
            const __m512i k1 = _mm512_ternarylogic_epi32(_mm512_and_si512(_mm512_srli_epi16(qv, 2), m3), _mm512_slli_epi16(hv, 1), m4, 0xF8);
            const __m512i k2 = _mm512_ternarylogic_epi32(_mm512_and_si512(_mm512_srli_epi16(qv, 4), m3), hv, m4, 0xF8);
            const __m512i k3 = _mm512_ternarylogic_epi32(_mm512_and_si512(_mm512_srli_epi16(qv, 6), m3), _mm512_srli_epi16(hv, 1), m4, 0xF8);
            const __m512i s0 = _mm512_madd_epi16(_mm512_maddubs_epi16(k0, y0), _mm512_shuffle_epi8(scl, ix0));
            const __m512i s1 = _mm512_madd_epi16(_mm512_maddubs_epi16(k1, y1), _mm512_shuffle_epi8(scl, ix1));
            const __m512i s2 = _mm512_madd_epi16(_mm512_maddubs_epi16(k2, y2), _mm512_shuffle_epi8(scl, ix2));
            const __m512i s3 = _mm512_madd_epi16(_mm512_maddubs_epi16(k3, y3), _mm512_shuffle_epi8(scl, ix3));
            __m512 sb = _mm512_mul_ps(_mm512_cvtepi32_ps(s0), as0);
            sb = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s1), as1, sb);
            sb = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s2), as2, sb);
            sb = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s3), as3, sb);
            // the scales sat in the high byte of the lanes: the sums are 256 times too big
            acc[r] = _mm512_fmadd_ps(sb, _mm512_set1_ps(d*(1.0f/256.0f)), acc[r]);
            const __m256 mf = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_cvtepi8_epi16(sc), bs));
            am[r] = _mm256_fmadd_ps(_mm256_mul_ps(mf, _mm256_set1_ps(-4.0f*d)), asg, am[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = _mm512_reduce_add_ps(acc[r]) + kh256(am[r]);
    }
}

template <int NR>
__attribute__((target("avx2,f16c,fma")))
static void kdot_q3k_avx2(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m4 = _mm256_set1_epi8(4);
    __m256i ix[8];
    for (int g = 0; g < 8; ++g) {
        ix[g] = kshuf256(g, true);
    }
    __m256 acc[NR];
    __m256 am[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm256_setzero_ps();
        am[r]  = _mm256_setzero_ps();
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
        const __m256  asg = _mm256_load_ps(ab.asg);
        const __m256i bs  = _mm256_load_si256((const __m256i *) ab.bs);
        for (int r = 0; r < NR; ++r) {
            const uint8_t * p  = w[r] + (size_t) k*110;
            const __m256i  hm  = _mm256_loadu_si256((const __m256i *) p);
            const __m128i  sc  = kq3_scales(p + 96);
            const __m256i  scl = _mm256_broadcastsi128_si256(sc);
            uint16_t dh;
            memcpy(&dh, p + 108, 2);
            const float d = _cvtsh_ss(dh);
            __m256 sb = _mm256_setzero_ps();
#pragma GCC unroll 2
            for (int h = 0; h < 2; ++h) {
                const __m256i qh = _mm256_loadu_si256((const __m256i *) (p + 32 + 32*h));
#pragma GCC unroll 4
                for (int j = 0; j < 4; ++j) {
                    const int g = 4*h + j;
                    // bit g of the high mask to bit 2
                    const __m256i hb = g < 2 ? _mm256_slli_epi16(hm, 2 - g) : g == 2 ? hm : _mm256_srli_epi16(hm, g - 2);
                    const __m256i c  = _mm256_or_si256(_mm256_and_si256(j == 0 ? qh : _mm256_srli_epi16(qh, 2*j), m3), _mm256_and_si256(hb, m4));
                    const __m256i y  = _mm256_load_si256((const __m256i *) (ab.q + 64*j + 32*h));
                    const __m256i pr = _mm256_madd_epi16(_mm256_maddubs_epi16(c, y), _mm256_shuffle_epi8(scl, ix[g]));
                    sb = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pr), _mm256_broadcast_ss(ab.asg + g), sb);
                }
            }
            acc[r] = _mm256_fmadd_ps(sb, _mm256_set1_ps(d*(1.0f/256.0f)), acc[r]);
            const __m256 mf = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_cvtepi8_epi16(sc), bs));
            am[r] = _mm256_fmadd_ps(_mm256_mul_ps(mf, _mm256_set1_ps(-4.0f*d)), asg, am[r]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = kh256(acc[r]) + kh256(am[r]);
    }
}

// ---- Q4_0 (glm5-next's NextN block): blocks of 32 = fp16 d + 16 bytes (element i: low nibble of byte i, element 16 + i:
// high nibble), value = d*(code - 8). One block per activation group of 32 (scale asg[G] in element order G, the codes
// against the group's int8 values, the -8 on the sums of its two 16-halves).
template <int NR>
__attribute__((target("avx2,f16c,fma")))
static void kdot_q40_avx2(const uint8_t * const * w, const kact_block * a, int nsb, float * out) {
    const __m128i m4   = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[NR];
    float  cor[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = _mm256_setzero_ps();
        cor[r] = 0.0f;
    }
    for (int k = 0; k < nsb; ++k) {
        const kact_block & ab = a[k];
#pragma GCC unroll 8
        for (int G = 0; G < 8; ++G) {
            const __m256i y  = _mm256_load_si256((const __m256i *) (ab.q + 64*(G & 3) + 32*(G >> 2)));
            const float   as = ab.asg[G];
            const int     bs = ab.bs[2*G] + ab.bs[2*G + 1];
            for (int r = 0; r < NR; ++r) {
                const uint8_t * p = w[r] + (size_t) (8*k + G)*18;
                uint16_t dh;
                memcpy(&dh, p, 2);
                const __m128i qs = _mm_loadu_si128((const __m128i *) (p + 2));
                const __m128i lo = _mm_and_si128(qs, m4);
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), m4);
                const __m256i c  = _mm256_set_m128i(hi, lo);
                const __m256i pr = _mm256_madd_epi16(_mm256_maddubs_epi16(c, y), ones);
                const float   da = _cvtsh_ss(dh)*as;
                acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pr), _mm256_set1_ps(da), acc[r]);
                cor[r] += da*8.0f*(float) bs;
            }
        }
    }
    for (int r = 0; r < NR; ++r) {
        out[r] = kh256(acc[r]) - cor[r];
    }
}
#endif // MH_X86

typedef void (*kdot_fn)(const uint8_t * const *, const kact_block *, int, float *);

static inline int64_t mh_realtime_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t) ts.tv_sec*1000000000LL + ts.tv_nsec;
}

static inline void mh_prefetch(const void * p) {
#if MH_X86
    _mm_prefetch((const char *) p, _MM_HINT_T0);
#else
    __builtin_prefetch(p);
#endif
}

#if MH_X86
// A core leaves the frequency license of wide vector multiplies about 2 ms after the last one, and the next ones then
// run throttled for up to half a millisecond: longer than a job. The spinning threads keep the license with these.
__attribute__((target("avx512f")))
static void keep_warm_avx512() {
    static thread_local float v = 1.0f;
    const __m512 a = _mm512_set1_ps(v);
    v = _mm512_cvtss_f32(_mm512_fmadd_ps(a, _mm512_set1_ps(0.999f), _mm512_set1_ps(0.001f)));
}

__attribute__((target("avx2,fma")))
static void keep_warm_avx2() {
    static thread_local float v = 1.0f;
    const __m256 a = _mm256_set1_ps(v);
    v = _mm256_cvtss_f32(_mm256_fmadd_ps(a, _mm256_set1_ps(0.999f), _mm256_set1_ps(0.001f)));
}
#endif

typedef void (*dot2_fn)(const q2_block *, const q2_block *, const act_block *, int, float *, float *);
typedef void (*dot4_fn)(const q2_block *, const q2_block *, const q2_block *, const q2_block *, const act_block *, int, float *);
typedef void (*quant_fn)(const float *, int, act_block *);
typedef void (*kfrom_fn)(const act_block *, int, kact_block *);
typedef void (*kquant_fn)(const float *, kact_block &);

// GGML_CUDA_MOE_HOST_ISA=generic|avx2 forces a narrower kernel
static void pick_kernels(dot2_fn & dot2, quant_fn & quant, dot2k_fn & dot2k, dot2k_fn & dot2k3, quantn_fn & quantn, const char *& name) {
    dot2   = dot2_generic;
    quant  = quantize_act_generic;
    dot2k  = dot2_q2k_generic;
    dot2k3 = dot2_q3k_generic;
    quantn = quantize_nat_generic;
    name   = "generic";
#if MH_X86
    const char * force = getenv("GGML_CUDA_MOE_HOST_ISA");
    __builtin_cpu_init();
    const bool has512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq");
    const bool has2   = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    if (force != nullptr && strcmp(force, "generic") == 0) {
        return;
    }
    if (has2) {
        quant  = quantize_act_avx2;
        dot2   = dot2_avx2;
        dot2k  = dot2_q2k_avx2;
        quantn = quantize_nat_avx2;
        name   = "AVX2";
    }
    if (has512 && has2 && !(force != nullptr && strcmp(force, "avx2") == 0)) {
        dot2   = dot2_avx512;
        dot2k  = dot2_q2k_avx512;
        dot2k3 = dot2_q3k_avx512;
        name   = "AVX-512";
    }
#endif
}

// the k kernels (2 rows) and activation converters: AVX-512 where the old kernels are, AVX2 below; false: the old path only
static bool pick_kkernels(kdot_fn & q2, kdot_fn & q3, kdot_fn & q4, kfrom_fn & from, kquant_fn & quant, const char *& name) {
#if MH_X86
    const char * force = getenv("GGML_CUDA_MOE_HOST_ISA");
    __builtin_cpu_init();
    const bool has512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl");
    const bool has2   = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    if (!has2 || (force != nullptr && strcmp(force, "generic") == 0)) {
        return false;
    }
    from  = kact_from_act_sse;
    quant = kact_quantize_avx2;
    q2    = kdot_q2k_avx2<2>;
    q3    = kdot_q3k_avx2<2>;
    q4    = kdot_q40_avx2<2>;
    name  = "AVX2";
    if (has512 && !(force != nullptr && strcmp(force, "avx2") == 0)) {
        q2   = kdot_q2k_avx512<2>;
        q3   = kdot_q3k_avx512<2>;
        name = "AVX-512";
    }
    return true;
#else
    (void) q2; (void) q3; (void) q4; (void) from; (void) quant; (void) name;
    return false;
#endif
}

struct group {
    int c;                  // cold expert
    int n;                  // pairs in the group
    int first;              // its pairs: pair_of[first .. first + n), tok_of[...]
};

// the layers, shared by the pools
struct slot_table {
    std::mutex       mtx;
    std::atomic<int> n { 0 };
    mh_slot_desc     tab[1024];
};

slot_table g_slots;

// GGML_CUDA_MOE_HOST_NUMA_SPLIT=1: the NUMA node that holds each cold expert's gate and down rows of a layer (asked of the
// kernel with move_pages when the layer is registered), so that the threads of a socket take the experts of their own
// node first (work2). Indexed like g_slots.tab.
struct slot_nodes {
    std::vector<int8_t> gate, down;
};
std::vector<slot_nodes> g_slot_nodes(1024);

bool numa_split_enabled() {
    static const bool on = getenv("GGML_CUDA_MOE_HOST_NUMA_SPLIT") != nullptr && atoi(getenv("GGML_CUDA_MOE_HOST_NUMA_SPLIT")) != 0;
    return on;
}

std::vector<int8_t> expert_nodes(const uint8_t * base, const size_t nb2, const int n) {
    std::vector<int8_t> r((size_t) n, 0);
#if defined(__linux__)
    std::vector<void *> pages((size_t) n);
    std::vector<int>    status((size_t) n, -1);
    for (int e = 0; e < n; ++e) {
        pages[e] = (void *) (((uintptr_t) (base + (size_t) e*nb2 + nb2/2)) & ~(uintptr_t) 4095);
    }
    if (syscall(SYS_move_pages, 0, (unsigned long) n, pages.data(), nullptr, status.data(), 0) == 0) {
        for (int e = 0; e < n; ++e) {
            r[e] = (int8_t) (status[e] >= 0 && status[e] < 64 ? status[e] : 0);
        }
    }
#else
    GGML_UNUSED(base); GGML_UNUSED(nb2);
#endif
    return r;
}

// the NUMA node (CPU package) of the calling thread's CPU
int thread_node() {
    static thread_local int node = -2;
    if (node == -2) {
        node = 0;
#if defined(__linux__)
        const int cpu = sched_getcpu();
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        if (FILE * f = fopen(path, "r")) {
            int id = 0;
            if (fscanf(f, "%d", &id) == 1 && id >= 0 && id < 8) {
                node = id;
            }
            fclose(f);
        }
        GGML_LOG_WARN("moe-host: a pool thread on cpu %d is on node %d\n", cpu, node);
#endif
    }
    return node;
}

std::mutex       g_pinned_mtx;
std::vector<int> g_pinned_cpus; // mh_pinned_cpus

// The CPUs this process may run on, one per physical core first (hyperthreads of cores already listed last).
std::vector<int> cpu_list() {
    std::vector<int> first, rest;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return first;
    }
    std::vector<std::pair<int, int>> seen; // (package, core)
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &set)) {
            continue;
        }
        int ids[2] = { -1, -1 };
        const char * files[2] = { "physical_package_id", "core_id" };
        for (int k = 0; k < 2; ++k) {
            char path[128];
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", c, files[k]);
            if (FILE * f = fopen(path, "r")) {
                if (fscanf(f, "%d", &ids[k]) != 1) {
                    ids[k] = -1;
                }
                fclose(f);
            }
        }
        const std::pair<int, int> key(ids[0], ids[1]);
        if (ids[1] >= 0 && std::find(seen.begin(), seen.end(), key) != seen.end()) {
            rest.push_back(c);
        } else {
            seen.push_back(key);
            first.push_back(c);
        }
    }
#endif
    first.insert(first.end(), rest.begin(), rest.end());
    return first;
}

struct pool {
    int          n_threads = 0;
    int          id        = 0;
    dot2_fn      dot2  = nullptr;
    dot4_fn      dot4  = nullptr;   // the down projection's rows four at a time (AVX-512), or null
    quant_fn     quant = nullptr;
    dot2k_fn     dot2k  = nullptr;   // Q2_K cold experts (natural-order activations)
    dot2k_fn     dot2k3 = nullptr;   // Q3_K cold experts (the down of deepseek41's Q2_K files)
    quantn_fn    quantn = nullptr;
    const char * isa   = nullptr;
    void (*keep_warm)() = nullptr; // called by the spinning threads
    bool spin_pause = true;
    // the helpers spin for this long after the last job before they sleep (GGML_CUDA_MOE_HOST_HOT_MS, default 50): a
    // node that serves a slice of the layers over RPC idles for the other nodes' slices between its bursts
    int  hot_ms = 50;

    std::vector<std::thread> threads;
    mh_mailbox * mb[MH_MAX_DEVICES] = {};
    uint32_t     seen[MH_MAX_DEVICES] = {};
    std::atomic<int> n_mb { 0 };

    // the current job, written by the leader before job_gen is bumped
    const mh_slot_desc * js = nullptr;
    mh_mailbox *         jm = nullptr;
    int                  n_groups = 0;
    std::vector<group>   groups;         // the cold experts of the job
    std::vector<int>     pair_of;        // pairs grouped by expert: pair index p (into the mailbox y)
    std::vector<int>     tok_of;         // ... and its token
    std::vector<int>     grp_of_exp;     // scratch: expert -> group
    const act_block *    jxq = nullptr;  // [n_tokens][n_embd/64], in the mailbox
    float *              jy  = nullptr;  // [n_pairs][n_embd], in the mailbox
    std::vector<float>     h;    // [n_pairs][n_ff]: the swiglu outputs
    std::vector<act_block> hq;   // [n_pairs][n_ff/64]: ... quantized
    std::vector<nat_block> xqn;  // Q2_K layers: the input rows in natural order, [n_tokens][n_embd/64]
    std::vector<nat_block> hqn;  // ... and the swiglu outputs, [n_pairs][n_ff/64]
    // the k kernels (GGML_CUDA_MOE_HOST_KERNEL=old turns them off): Q2_K gate/up with a Q2_K or Q3_K down, blocks of 256
    bool         knew     = false;
    int          jmode    = 0;           // this job: 0 old kernels, 2 k kernels
    kdot_fn      kd_q2k   = nullptr;     // 2 rows
    kdot_fn      kd_q3k   = nullptr;
    kdot_fn      kd_q40   = nullptr;     // Q4_0 gate/up/down (wtype 3)
    kfrom_fn     kfrom    = nullptr;
    kquant_fn    kquant   = nullptr;
    const char * kisa     = nullptr;
    int          CH3k     = 64;          // down rows per chunk of the k path
    std::vector<kact_block> xk;          // the input rows in k format, [n_tokens][n_embd/256]
    std::vector<kact_block> hk;          // the swiglu outputs, [n_pairs][n_ff/256]
    size_t                 cap_blk = 0;
    std::unique_ptr<std::atomic<int>[]> blk_cnt;   // [n_groups][n_ff/64]: finished CH1-row chunks per expert block
    std::unique_ptr<std::atomic<int>[]> grp_ready; // [n_groups]: quantized activation blocks per expert

    alignas(64) std::atomic<uint32_t> job_gen { 0 };
    alignas(64) std::atomic<int>      chunk1 { 0 };
    alignas(64) std::atomic<int>      chunk3 { 0 };
    // NUMA split of the k path's queue: per node, its gate/up chunks, then its down chunks (ids as in the single queue);
    // threads take their node's gate/up chunks, then the other node's, then the down chunks the same way
    static constexpr int MH_NODES = 2;
    bool                              nsplit = false;
    std::vector<int>                  qgu[MH_NODES], qdn[MH_NODES];
    struct alignas(64) counter { std::atomic<int> v { 0 }; };
    std::atomic<int64_t>              st_own { 0 }, st_steal { 0 };   // chunks taken from the thread's own node's queue / the other's
    counter                           qcnt_gu[MH_NODES], qcnt_dn[MH_NODES];
    alignas(64) std::atomic<int>      bar_count { 0 };
    alignas(64) std::atomic<uint32_t> bar_gen { 0 };
    alignas(64) std::atomic<bool>     hot { false };
    std::atomic<bool>       quit { false };
    std::mutex              cv_mtx;
    std::condition_variable cv;

    void barrier() {
        const uint32_t gen = bar_gen.load(std::memory_order_acquire);
        if (bar_count.fetch_add(1, std::memory_order_acq_rel) == n_threads - 1) {
            bar_count.store(0, std::memory_order_relaxed);
            bar_gen.store(gen + 1, std::memory_order_release);
        } else {
            while (bar_gen.load(std::memory_order_acquire) == gen) {
#if MH_X86
                _mm_pause();
#endif
            }
        }
    }

    // phase 1: gate/up in chunks of CH1 rows (small, so that the threads finish together); the thread that completes
    // a block of 64 rows of h = silu(gate) * up quantizes it, so phase 3 reads ready activations.
    // phase 3: down in chunks of CH3 rows.
    int CH1 = 16;
    int CH3 = 160;
    int pf_rows = 0; // gate/up: prefetch this many rows ahead (down: 16 times as many); measured: no gain

#ifdef MH_TRACE
    // per-thread timeline of a job, relative to its start: when the thread saw the job, its time in gate/up chunks,
    // waiting for an expert's activations, in down chunks, and when it reached the barrier
    struct trace { double seen = 0, p1 = 0, wait = 0, p3 = 0, end = 0; long n = 0; };
    static constexpr int MAX_TR = 64;
    trace            tr[MAX_TR];
    std::atomic<int> tr_ids { 0 };
    std::atomic<int64_t> t_job_ns { 0 };
    static int64_t tns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    int tr_id() { static thread_local int id = -1; static thread_local pool * owner = nullptr; if (owner != this) { owner = this; id = tr_ids.fetch_add(1) % MAX_TR; } return id; }
    void tr_print() {
        for (int i = 0; i < std::min(MAX_TR, tr_ids.load()); ++i) {
            const trace & t = tr[i];
            if (t.n > 0) {
                fprintf(stderr, "trace pool %d thread %2d: %ld jobs, saw the job at %6.1f us, gate/up %6.1f us, waiting %6.1f us, down %6.1f us, at the barrier at %6.1f us\n",
                        id, i, t.n, t.seen/t.n/1e3, t.p1/t.n/1e3, t.wait/t.n/1e3, t.p3/t.n/1e3, t.end/t.n/1e3);
            }
        }
    }
#define MH_TR(x) x
#else
#define MH_TR(x)
#endif

    void work() {
        const mh_slot_desc & s = *js;
        MH_TR(trace & T = tr[tr_id()]; const int64_t t_job = t_job_ns.load(); int64_t t_prev = tns(); T.seen += t_prev - t_job; T.n++;)
        const int nb_e = s.n_embd/QK;
        const int nb_f = s.n_ff/QK;
        const int rb1  = s.n_ff/CH1;
        const int n1   = n_groups*rb1;
        const int ch3  = s.n_embd % CH3 == 0 ? CH3 : 32;
        const int rb3  = s.n_embd/ch3;
        const int n3   = n_groups*rb3;
        // one queue: the gate/up chunks of all experts, then their down chunks. A down chunk waits for its own
        // expert's activations only, so the threads that finish early start on the down projections of the experts
        // that are complete instead of waiting for all of them.
        for (int c; (c = chunk1.fetch_add(1, std::memory_order_relaxed)) < n1 + n3; ) {
            if (c < n1) {
                const int       gi = c/rb1;
                const group &   g  = groups[gi];
                const int       r0 = (c % rb1)*CH1;
                const uint8_t * gate = s.gate + (size_t) g.c*s.gate_nb2;
                const uint8_t * up   = s.up   + (size_t) g.c*s.up_nb2;
                const int * gp = pair_of.data() + g.first;
                const int * gt = tok_of.data()  + g.first;
                // the rows of a chunk are collected here: neighbouring chunks of h are other threads' and share its cache lines
                float hl[64];
                for (int i = 0; i < g.n; ++i) {
                    const act_block * xq  = jxq + (size_t) gt[i]*nb_e;
                    const nat_block * xqk = s.wtype == 1 ? xqn.data() + (size_t) gt[i]*nb_e : nullptr;
                    for (int r = r0; r < r0 + CH1; ++r) {
                        const q2_block * wg = (const q2_block *) (gate + (size_t) r*s.gate_nb1);
                        const q2_block * wu = (const q2_block *) (up   + (size_t) r*s.up_nb1);
                        if (i == 0 && pf_rows > 0 && r + pf_rows < r0 + CH1) {
                            // the rows are in memory that no cache holds, and 4 KB pages stop the hardware prefetchers
                            const char * pg = (const char *) wg + (size_t) pf_rows*s.gate_nb1;
                            const char * pu = (const char *) wu + (size_t) pf_rows*s.up_nb1;
                            for (size_t o = 0; o < s.gate_nb1; o += 64) {
                                mh_prefetch(pg + o);
                                mh_prefetch(pu + o);
                            }
                        }
                        float vg, vu;
                        if (s.wtype == 1) {
                            dot2k((const uint8_t *) wg, (const uint8_t *) wu, xqk, nb_e, &vg, &vu);
                        } else {
                            dot2(wg, wu, xq, nb_e, &vg, &vu);
                        }
                        if (s.clamp > 0.0f) {   // swiglu_clamp(limit): silu(min(g, limit)) * clamp(u, -limit, limit)
                            vg = std::min(vg, s.clamp);
                            vu = std::max(std::min(vu, s.clamp), -s.clamp);
                        }
                        hl[r - r0] = vg/(1.0f + expf(-vg))*vu;
                    }
                    memcpy(h.data() + (size_t) gp[i]*s.n_ff + r0, hl, CH1*sizeof(float));
                }
                MH_TR({ const int64_t t = tns(); T.p1 += t - t_prev; t_prev = t; })
                const int kb = r0/QK;
                if (blk_cnt[(size_t) gi*nb_f + kb].fetch_add(1, std::memory_order_acq_rel) == QK/CH1 - 1) {
                    for (int i = 0; i < g.n; ++i) {
                        const int p = gp[i];
                        if (s.wtype == 1) {
                            quantn(h.data() + (size_t) p*s.n_ff + kb*QK, QK, hqn.data() + (size_t) p*nb_f + kb);
                        } else {
                            quant(h.data() + (size_t) p*s.n_ff + kb*QK, QK, hq.data() + (size_t) p*nb_f + kb);
                        }
                    }
                    grp_ready[gi].fetch_add(1, std::memory_order_release);
                    if (stat_every > 0 && p1_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        t_p1_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
                    }
                }
                continue;
            }
            const int       c3 = c - n1;
            const int       gi = c3/rb3;
            const group &   g  = groups[gi];
            const int       r0 = (c3 % rb3)*ch3;
            const uint8_t * down = s.down + (size_t) g.c*s.down_nb2;
            MH_TR({ const int64_t t = tns(); T.p1 += t - t_prev; t_prev = t; })
            while (grp_ready[gi].load(std::memory_order_acquire) < nb_f) {
#if MH_X86
                _mm_pause();
#endif
            }
            MH_TR({ const int64_t t = tns(); T.wait += t - t_prev; t_prev = t; })
            const int * gp = pair_of.data() + g.first;
            if (s.wtype == 0 && dot4 != nullptr && ch3 % 4 == 0) {
                for (int r = r0; r < r0 + ch3; r += 4) {
                    const q2_block * w0 = (const q2_block *) (down + (size_t) (r + 0)*s.down_nb1);
                    const q2_block * w1 = (const q2_block *) (down + (size_t) (r + 1)*s.down_nb1);
                    const q2_block * w2 = (const q2_block *) (down + (size_t) (r + 2)*s.down_nb1);
                    const q2_block * w3 = (const q2_block *) (down + (size_t) (r + 3)*s.down_nb1);
                    for (int i = 0; i < g.n; ++i) {
                        const int p = gp[i];
                        dot4(w0, w1, w2, w3, hq.data() + (size_t) p*nb_f, nb_f, jy + (size_t) p*s.n_embd + r);
                    }
                }
                MH_TR({ const int64_t t = tns(); T.p3 += t - t_prev; t_prev = t; })
                continue;
            }
            for (int r = r0; r < r0 + ch3; r += 2) {
                const q2_block * w0 = (const q2_block *) (down + (size_t) (r + 0)*s.down_nb1);
                const q2_block * w1 = (const q2_block *) (down + (size_t) (r + 1)*s.down_nb1);
                if (pf_rows > 0) {
                    const char * pd = (const char *) w0 + (size_t) 16*pf_rows*s.down_nb1;
                    for (size_t o = 0; o < 2*s.down_nb1; o += 64) {
                        mh_prefetch(pd + o);
                    }
                }
                for (int i = 0; i < g.n; ++i) {
                    const int p = gp[i];
                    float v0, v1;
                    if (s.wtype == 1) {
                        (s.wtype_down == 2 ? dot2k3 : dot2k)((const uint8_t *) w0, (const uint8_t *) w1, hqn.data() + (size_t) p*nb_f, nb_f, &v0, &v1);
                    } else {
                        dot2(w0, w1, hq.data() + (size_t) p*nb_f, nb_f, &v0, &v1);
                    }
                    float * y = jy + (size_t) p*s.n_embd;
                    y[r + 0] = v0;
                    y[r + 1] = v1;
                }
            }
            MH_TR({ const int64_t t = tns(); T.p3 += t - t_prev; t_prev = t; })
        }
        MH_TR(T.end += tns() - t_job;)
    }

    // Every thread of the k path works on private copies of the activation rows. Read from the shared buffers (the input rows
    // that the leader wrote, the swiglu rows that whichever thread completed a block quantized) every kernel call took 3x
    // longer on the 2-socket Xeon; with a copy per thread and job the same kernel runs at the memory limit.
    struct kscratch {
        std::vector<kact_block> x;         // the input row of token x_tok
        std::vector<kact_block> h;         // the quantized swiglu row of pair h_pair
        uint32_t x_job = 0, h_job = 0;
        int      x_tok = -1, h_pair = -1;
    };
    uint32_t job_id = 0;                   // counts the jobs of this pool: the scratch of a thread is valid for one job

    // The job of the k path, same queue as work(): the gate/up chunks of every expert, then their down chunks. The thread that
    // completes the chunks of a block of 256 rows of h quantizes it for all the pairs of the expert.
    void work2() {
        static thread_local kscratch ks;
        const uint32_t job = job_id;
        const mh_slot_desc & s = *js;
        MH_TR(trace & T = tr[tr_id()]; const int64_t t_job = t_job_ns.load(); int64_t t_prev = tns(); T.seen += t_prev - t_job; T.n++;)
        const int nsb_e = s.n_embd/256;
        const int nsb_f = s.n_ff/256;
        const int ch1 = CH1;
        const int rb1 = s.n_ff/ch1;
        const int n1  = n_groups*rb1;
        const int ch3 = s.n_embd % CH3k == 0 ? CH3k : 32;
        const int rb3 = s.n_embd/ch3;
        const int n3  = n_groups*rb3;
        const kdot_fn kdn = s.wtype_down == 3 ? kd_q40 : s.wtype_down == 2 ? kd_q3k : kd_q2k;
        const kdot_fn kdg = s.wtype == 3 ? kd_q40 : kd_q2k;
        float * hbase = (float *) (((uintptr_t) h.data() + 63) & ~(uintptr_t) 63);   // 64-byte lines: chunks of 16 rows own whole lines
        const bool split = nsplit;
        const int  my    = std::min(thread_node(), MH_NODES - 1);
        int        phase = 0;   // NUMA split: 0 own node's gate/up chunks, 1 the other's, 2 own down chunks, 3 the other's
        auto next_chunk = [&]() -> int {
            if (!split) {
                const int c = chunk1.fetch_add(1, std::memory_order_relaxed);
                return c < n1 + n3 ? c : -1;
            }
            for (; phase < 4; ++phase) {
                const int k = (phase & 1) ? 1 - my : my;
                const std::vector<int> & q = phase < 2 ? qgu[k] : qdn[k];
                std::atomic<int> & cnt = phase < 2 ? qcnt_gu[k].v : qcnt_dn[k].v;
                if (cnt.load(std::memory_order_relaxed) < (int) q.size()) {
                    const int j = cnt.fetch_add(1, std::memory_order_relaxed);
                    if (j < (int) q.size()) {
                        if (stat_every > 0) {
                            (k == my ? st_own : st_steal).fetch_add(1, std::memory_order_relaxed);
                        }
                        return q[j];
                    }
                }
            }
            return -1;
        };
        for (int c; (c = next_chunk()) >= 0; ) {
            if (c < n1) {
                const int       gi = c/rb1;
                const group &   g  = groups[gi];
                const int       r0 = (c % rb1)*ch1;
                const uint8_t * gate = s.gate + (size_t) g.c*s.gate_nb2;
                const uint8_t * up   = s.up   + (size_t) g.c*s.up_nb2;
                const int * gp = pair_of.data() + g.first;
                const int * gt = tok_of.data()  + g.first;
                float hl[64];
                for (int i = 0; i < g.n; ++i) {
                    if (ks.x_job != job || ks.x_tok != gt[i]) {
                        if (ks.x.size() < (size_t) nsb_e) {
                            ks.x.resize(nsb_e);
                        }
                        memcpy(ks.x.data(), xk.data() + (size_t) gt[i]*nsb_e, (size_t) nsb_e*sizeof(kact_block));
                        ks.x_job = job;
                        ks.x_tok = gt[i];
                    }
                    for (int r = r0; r < r0 + ch1; ++r) {
                        const uint8_t * pw[2] = { gate + (size_t) r*s.gate_nb1, up + (size_t) r*s.up_nb1 };
                        float v[2];
                        kdg(pw, ks.x.data(), nsb_e, v);
                        float vg = v[0], vu = v[1];
                        if (s.clamp > 0.0f) {   // swiglu_clamp(limit): silu(min(g, limit)) * clamp(u, -limit, limit)
                            vg = std::min(vg, s.clamp);
                            vu = std::max(std::min(vu, s.clamp), -s.clamp);
                        }
                        hl[r - r0] = vg/(1.0f + expf(-vg))*vu;
                    }
                    memcpy(hbase + (size_t) gp[i]*s.n_ff + r0, hl, ch1*sizeof(float));
                }
                MH_TR({ const int64_t t = tns(); T.p1 += t - t_prev; t_prev = t; })
                const int kb = r0/256;
                if (blk_cnt[(size_t) gi*nsb_f + kb].fetch_add(1, std::memory_order_acq_rel) == 256/ch1 - 1) {
                    for (int i = 0; i < g.n; ++i) {
                        const int p = gp[i];
                        kquant(hbase + (size_t) p*s.n_ff + (size_t) kb*256, hk[(size_t) p*nsb_f + kb]);
                    }
                    grp_ready[gi].fetch_add(1, std::memory_order_release);
                    if (stat_every > 0 && p1_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        t_p1_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
                    }
                }
                continue;
            }
            const int       c3 = c - n1;
            const int       gi = c3/rb3;
            const group &   g  = groups[gi];
            const int       r0 = (c3 % rb3)*ch3;
            const uint8_t * down = s.down + (size_t) g.c*s.down_nb2;
            MH_TR({ const int64_t t = tns(); T.p1 += t - t_prev; t_prev = t; })
            while (grp_ready[gi].load(std::memory_order_acquire) < nsb_f) {
#if MH_X86
                _mm_pause();
#endif
            }
            MH_TR({ const int64_t t = tns(); T.wait += t - t_prev; t_prev = t; })
            const int * gp = pair_of.data() + g.first;
            for (int i = 0; i < g.n; ++i) {
                const int p = gp[i];
                if (ks.h_job != job || ks.h_pair != p) {
                    if (ks.h.size() < (size_t) nsb_f) {
                        ks.h.resize(nsb_f);
                    }
                    memcpy(ks.h.data(), hk.data() + (size_t) p*nsb_f, (size_t) nsb_f*sizeof(kact_block));
                    ks.h_job  = job;
                    ks.h_pair = p;
                }
                float * y = jy + (size_t) p*s.n_embd;
                for (int r = r0; r < r0 + ch3; r += 2) {
                    const uint8_t * pw[2] = { down + (size_t) (r + 0)*s.down_nb1, down + (size_t) (r + 1)*s.down_nb1 };
                    float v[2];
                    kdn(pw, ks.h.data(), nsb_f, v);
                    y[r + 0] = v[0];
                    y[r + 1] = v[1];
                }
            }
            MH_TR({ const int64_t t = tns(); T.p3 += t - t_prev; t_prev = t; })
        }
        MH_TR(T.end += tns() - t_job;)
    }

    void run_work() {
        if (jmode == 2) {
            work2();
        } else {
            work();
        }
    }

    void helper() {
        // job_gen starts at 0: a helper that the scheduler starts late must still join a job that is already published
        uint32_t last = 0;
        while (!quit.load(std::memory_order_relaxed)) {
            const uint32_t j = job_gen.load(std::memory_order_acquire);
            if (j == last) {
                if (!hot.load(std::memory_order_relaxed)) {
                    std::unique_lock<std::mutex> lock(cv_mtx);
                    cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                        return hot.load() || quit.load() || job_gen.load() != last;
                    });
                } else {
#if MH_X86
                    if (spin_pause) {
                        _mm_pause();
                    }
#endif
                    if (keep_warm != nullptr) {
                        keep_warm();
                    }
                }
                continue;
            }
            last = j;
            run_work();
            barrier();
        }
    }

    // GGML_CUDA_MOE_HOST_STATS=N: every N jobs, print the mean job time, cold pairs and distinct experts per job
    int64_t stat_every = 0, stat_jobs = 0, stat_pairs = 0, stat_groups = 0;
    int     n_dumped = 0;   // the first jobs of this pool are logged (parameters from the mailbox)
    int64_t clk_off = 0, stat_seen_ns = 0, stat_seen_n = 0; // FN_WAITS transport timing (leader())
    double  stat_us = 0.0, stat_quant_us = 0.0, stat_p1_us = 0.0, stat_work_us = 0.0;
    std::atomic<int>     p1_left { 0 };
    std::atomic<int64_t> t_p1_ns { 0 };

    void run_job(mh_mailbox * m) {
        const auto t_start = std::chrono::steady_clock::now();
        std::atomic_thread_fence(std::memory_order_acquire);
        const int slot = m->slot;
        if (slot < 0 || slot >= g_slots.n.load(std::memory_order_acquire)) {
            GGML_LOG_ERROR("%s: bad slot %d\n", __func__, slot);
            return;
        }
        js = &g_slots.tab[slot];
        jm = m;
        jmode = knew && ((js->wtype == 1 && (js->wtype_down == 1 || js->wtype_down == 2)) || (js->wtype == 3 && js->wtype_down == 3)) &&
                js->n_embd % 256 == 0 && js->n_ff % 256 == 0 ? 2 : 0;
        GGML_ASSERT((js->wtype != 3 || jmode == 2) && "Q4_0 cold experts need the k kernels (GGML_CUDA_MOE_HOST_KERNEL, AVX2)");
        const int n_used  = m->n_used;
        const int n_pairs = std::min<int>(m->n_pairs, m->cap_pairs);
        const int nb_f    = js->n_ff/QK;
        const int32_t * pidx = mh_pair_idx(m);
        const int32_t * pexp = mh_pair_exp(m);

        // group the pairs by cold expert (counting sort, stable in pair order)
        int max_c = 0, min_c = 0;
        for (int p = 0; p < n_pairs; ++p) {
            max_c = std::max(max_c, (int) pexp[p]);
            min_c = std::min(min_c, (int) pexp[p]);
        }
        {
            int max_i = 0, min_i = 0;
            for (int p = 0; p < n_pairs; ++p) {
                max_i = std::max(max_i, (int) pidx[p]);
                min_i = std::min(min_i, (int) pidx[p]);
            }
            const bool bad = min_c < 0 || (js->n_cold > 0 && max_c >= js->n_cold) || m->n_tokens <= 0 || n_used <= 0 ||
                             min_i < 0 || max_i >= m->n_tokens*n_used || js->n_ff <= 0 || js->n_embd <= 0;
            if (n_dumped < 40 || bad) {
                n_dumped++;
                char pairs[256]; int o = 0;
                for (int p = 0; p < std::min(n_pairs, 8) && o < 200; ++p) {
                    o += snprintf(pairs + o, sizeof(pairs) - o, " (%d,%d)", (int) pidx[p], (int) pexp[p]);
                }
                GGML_LOG_WARN("moe-host: pool %d job slot %d: n_tokens %d n_used %d n_pairs %d (cap %d) experts %d..%d of %d, pairs %d..%d, n_ff %d n_embd %d wtype %d/%d, "
                              "gate %p/%zu/%zu up %p/%zu/%zu down %p/%zu/%zu xq %p:%s%s\n",
                              id, slot, (int) m->n_tokens, n_used, n_pairs, (int) m->cap_pairs, min_c, max_c, js->n_cold, min_i, max_i, js->n_ff, js->n_embd,
                              js->wtype, js->wtype_down, (const void *) js->gate, js->gate_nb1, js->gate_nb2, (const void *) js->up, js->up_nb1, js->up_nb2,
                              (const void *) js->down, js->down_nb1, js->down_nb2, (const void *) mh_xq(m), pairs, bad ? "  <- OUT OF RANGE, job dropped" : "");
            }
            if (bad) {
                return;
            }
        }
        grp_of_exp.assign((size_t) max_c + 1, -1);
        groups.clear();
        for (int p = 0; p < n_pairs; ++p) {
            int & gi = grp_of_exp[pexp[p]];
            if (gi < 0) {
                gi = (int) groups.size();
                groups.push_back({ (int) pexp[p], 0, 0 });
            }
            groups[gi].n++;
        }
        n_groups = (int) groups.size();
        for (int gi = 0, first = 0; gi < n_groups; ++gi) {
            groups[gi].first = first;
            first += groups[gi].n;
            groups[gi].n = 0;
        }
        pair_of.resize(n_pairs);
        tok_of.resize(n_pairs);
        for (int p = 0; p < n_pairs; ++p) {
            group & g = groups[grp_of_exp[pexp[p]]];
            pair_of[g.first + g.n] = p;
            tok_of [g.first + g.n] = pidx[p]/n_used;
            g.n++;
        }
        const int n_blk = jmode == 2 ? js->n_ff/256 : nb_f;   // blocks of h per expert: 256 rows (k path) or 64
        if (h.size() < (size_t) n_pairs*js->n_ff + 16) {
            h.resize((size_t) n_pairs*js->n_ff + 16);
        }
        if (jmode == 2) {
            if (hk.size() < (size_t) n_pairs*n_blk) {
                hk.resize((size_t) n_pairs*n_blk);
            }
        } else {
            if (hq.size() < (size_t) n_pairs*nb_f) {
                hq.resize((size_t) n_pairs*nb_f);
            }
            if (js->wtype == 1 && hqn.size() < (size_t) n_pairs*nb_f) {
                hqn.resize((size_t) n_pairs*nb_f);
            }
        }
        if (cap_blk < (size_t) n_groups*nb_f) {
            cap_blk = (size_t) n_groups*nb_f;
            blk_cnt.reset(new std::atomic<int>[cap_blk]);
            grp_ready.reset(new std::atomic<int>[cap_blk]);
        }
        for (size_t i = 0; i < (size_t) n_groups*n_blk; ++i) {
            blk_cnt[i].store(0, std::memory_order_relaxed);
        }
        for (int gi = 0; gi < n_groups; ++gi) {
            grp_ready[gi].store(0, std::memory_order_relaxed);
        }
        // the GPU quantized the input rows (mh_publish)
        jxq = mh_xq(m);
        jy  = mh_y(m);
        if (jmode == 2) {
            const size_t n = (size_t) m->n_tokens*(js->n_embd/256);
            if (xk.size() < n) {
                xk.resize(n);
            }
            kfrom(jxq, (int) n, xk.data());
        } else if (js->wtype == 1) {
            // Q2_K: the dot wants the activations in natural order with their 16-sums
            const int    nb_e = js->n_embd/QK;
            const size_t n    = (size_t) m->n_tokens*nb_e;
            if (xqn.size() < n) {
                xqn.resize(n);
            }
            act_to_nat(jxq, (int) n, xqn.data());
        }
        const auto t_quant = std::chrono::steady_clock::now();
        p1_left.store(n_groups*n_blk, std::memory_order_relaxed);
        MH_TR(t_job_ns.store(tns());)
        job_id++;
        nsplit = jmode == 2 && numa_split_enabled() && slot < (int) g_slot_nodes.size() && !g_slot_nodes[slot].gate.empty();
        if (nsplit) {
            const int ch1 = CH1;
            const int rb1 = js->n_ff/ch1;
            const int n1  = n_groups*rb1;
            const int ch3 = js->n_embd % CH3k == 0 ? CH3k : 32;
            const int rb3 = js->n_embd/ch3;
            for (int k = 0; k < MH_NODES; ++k) {
                qgu[k].clear();
                qdn[k].clear();
                qcnt_gu[k].v.store(0, std::memory_order_relaxed);
                qcnt_dn[k].v.store(0, std::memory_order_relaxed);
            }
            for (int gi = 0; gi < n_groups; ++gi) {
                const int c  = groups[gi].c;
                const int kg = std::min<int>(g_slot_nodes[slot].gate[c], MH_NODES - 1);
                const int kd = std::min<int>(g_slot_nodes[slot].down[c], MH_NODES - 1);
                for (int r = 0; r < rb1; ++r) {
                    qgu[kg].push_back(gi*rb1 + r);
                }
                for (int r = 0; r < rb3; ++r) {
                    qdn[kd].push_back(n1 + gi*rb3 + r);
                }
            }
        }
        chunk1.store(0, std::memory_order_relaxed);
        // GGML_CUDA_MOE_HOST_DIAG_EMPTY=1: DIAGNOSTICS ONLY (wrong results): the job answers without computing, so the GPU's
        // collect wait shows the transport latency alone
        static const bool diag_empty = getenv("GGML_CUDA_MOE_HOST_DIAG_EMPTY") != nullptr && atoi(getenv("GGML_CUDA_MOE_HOST_DIAG_EMPTY")) != 0;
        if (diag_empty) {
            std::atomic_thread_fence(std::memory_order_release);
            return;
        }
        job_gen.fetch_add(1, std::memory_order_acq_rel);
        run_work();
        const auto t_work = std::chrono::steady_clock::now();
        barrier();
        std::atomic_thread_fence(std::memory_order_release);
        if (stat_every > 0) {
            const auto t_end = std::chrono::steady_clock::now();
            stat_us       += std::chrono::duration<double, std::micro>(t_end - t_start).count();
            stat_quant_us += std::chrono::duration<double, std::micro>(t_quant - t_start).count();
            stat_work_us  += std::chrono::duration<double, std::micro>(t_work - t_start).count();
            stat_p1_us    += (t_p1_ns.load() - std::chrono::duration_cast<std::chrono::nanoseconds>(t_start.time_since_epoch()).count())/1e3;
            stat_pairs    += n_pairs;
            stat_groups   += n_groups;
            if (++stat_jobs == stat_every) {
                fprintf(stderr, "moe-host: pool %d: %lld jobs: %.1f us/job (setup %.1f, gate/up done at %.1f, leader's work done at %.1f us), %.2f cold pairs, %.2f experts per job, %s kernels%s\n",
                              id, (long long) stat_jobs, stat_us/stat_jobs, stat_quant_us/stat_jobs, stat_p1_us/stat_jobs, stat_work_us/stat_jobs,
                              (double) stat_pairs/stat_jobs, (double) stat_groups/stat_jobs, jmode == 2 ? "k" : "old",
                              stat_seen_n > 0 ? (", request seen " + std::to_string((double) stat_seen_ns/1e3/stat_seen_n) + " us after the GPU's fence (offset by the clock estimate)").c_str() : "");
                stat_seen_ns = stat_seen_n = 0;
                if (nsplit) {
                    fprintf(stderr, "moe-host: pool %d: NUMA split: %lld chunks from the thread's own node, %lld from the other\n", id,
                            (long long) st_own.exchange(0), (long long) st_steal.exchange(0));
                }
                stat_jobs = stat_pairs = stat_groups = 0;
                stat_us = stat_quant_us = stat_p1_us = stat_work_us = 0.0;
            }
        }
    }

    void leader() {
        auto last_job = std::chrono::steady_clock::now();
        int  nap_us   = 0;
        while (!quit.load(std::memory_order_relaxed)) {
            bool found = false;
            const int n = n_mb.load(std::memory_order_acquire);
            for (int d = 0; d < n; ++d) {
                mh_mailbox * m = mb[d];
                if (m == nullptr) {
                    continue;
                }
                const uint32_t r = m->req;
                if (r == seen[d]) {
                    continue;
                }
                seen[d] = r;
                // GGML_CUDA_FN_WAITS: the request's flight (the GPU wrote its %globaltimer before the fences); the first one
                // sets this pool's estimate of the GPU clock (host ns - GPU ns), so the figures carry that request's latency
                const uint64_t t_req = m->t_req;
                if (t_req != 0) {
                    const int64_t h1 = mh_realtime_ns();
                    if (clk_off == 0) {
                        clk_off = h1 - (int64_t) t_req;
                    }
                    stat_seen_ns += h1 - (int64_t) t_req - clk_off;
                    stat_seen_n++;
                }
                if (!hot.load(std::memory_order_relaxed)) {
                    hot.store(true);
                    std::lock_guard<std::mutex> lock(cv_mtx);
                    cv.notify_all();
                }
                run_job(m);
                if (clk_off != 0) {
                    m->t_done = (uint64_t) (mh_realtime_ns() - clk_off);
                }
                m->done = r;
                found    = true;
                last_job = std::chrono::steady_clock::now();
                nap_us   = 0;
            }
            if (found) {
                continue;
            }
            if (hot.load(std::memory_order_relaxed)) {
                if (std::chrono::steady_clock::now() - last_job > std::chrono::milliseconds(hot_ms)) {
                    hot.store(false);
                }
#if MH_X86
                if (spin_pause) {
                    _mm_pause();
                }
#endif
                if (keep_warm != nullptr) {
                    keep_warm();
                }
            } else {
                // idle: nap, so an idle server does not burn a core; the first layer after it waits for the nap
                nap_us = std::min(nap_us + 10, 200);
                std::this_thread::sleep_for(std::chrono::microseconds(nap_us));
            }
        }
    }

    // n_pools pools share the threads: pool i serves the mailboxes i, i + n_pools, ...
    void start(const int pool_id, const int n_pools) {
        id = pool_id;
        pick_kernels(dot2, quant, dot2k, dot2k3, quantn, isa);
#if MH_X86
        // GGML_CUDA_MOE_HOST_DOT4=0: the down rows two at a time as before
        if (dot2 == dot2_avx512 && (getenv("GGML_CUDA_MOE_HOST_DOT4") == nullptr || atoi(getenv("GGML_CUDA_MOE_HOST_DOT4")) != 0)) {
            dot4 = dot4_avx512;
        }
#endif
        stat_every = getenv("GGML_CUDA_MOE_HOST_STATS") ? atoll(getenv("GGML_CUDA_MOE_HOST_STATS")) : 0;
        const char * env = getenv("GGML_CUDA_MOE_HOST_THREADS");
        const int hw = (int) std::thread::hardware_concurrency();
        const int total = env != nullptr ? atoi(env) : std::max(1, std::min(16, hw - 4));
        n_threads = std::max(1, total/n_pools);
        // GGML_CUDA_MOE_HOST_POOL_CPUS=c,c,...:c,c,... (separator : or | or ;): the CPUs of each pool (the NUMA-pool layout: pool i = device i's
        // socket); the list of a pool sets its thread count
        std::vector<int> pool_cpus;
        if (const char * e = getenv("GGML_CUDA_MOE_HOST_POOL_CPUS")) {
            int i = 0;
            for (const char * p = e; *p != 0 && i <= pool_id;) {
                char * end = nullptr;
                const long c = strtol(p, &end, 10);
                if (end == p) {
                    break;
                }
                if (i == pool_id) {
                    pool_cpus.push_back((int) c);
                }
                p = end;
                if (*p == ',') {
                    ++p;
                } else if (*p == ';' || *p == '|' || *p == ':') {
                    ++p;
                    ++i;
                }
            }
            if (!pool_cpus.empty()) {
                n_threads = (int) pool_cpus.size();
            }
        }
        if (const char * e = getenv("GGML_CUDA_MOE_HOST_CH1")) {
            const int v = atoi(e);
            if (v == 4 || v == 8 || v == 16 || v == 32 || v == 64) {
                CH1 = v;
            }
        }
        if (const char * e = getenv("GGML_CUDA_MOE_HOST_PREFETCH")) {
            pf_rows = std::max(0, atoi(e));
        }
        if (const char * e = getenv("GGML_CUDA_MOE_HOST_CH3")) {
            const int v = atoi(e);
            if (v >= 2 && v % 2 == 0) {
                CH3 = v;
            }
        }
        // GGML_CUDA_MOE_HOST_KERNEL=old: the previous kernels and job loop for every layer (an A/B switch)
        {
            const char * ke = getenv("GGML_CUDA_MOE_HOST_KERNEL");
            knew = (ke == nullptr || strcmp(ke, "old") != 0) && pick_kkernels(kd_q2k, kd_q3k, kd_q40, kfrom, kquant, kisa);
            if (const char * e = getenv("GGML_CUDA_MOE_HOST_CH3")) {
                const int v = atoi(e);
                if (v >= 2 && v % 2 == 0) {
                    CH3k = v;
                }
            }
        }
        spin_pause = getenv("GGML_CUDA_MOE_HOST_SPIN_PAUSE") == nullptr || atoi(getenv("GGML_CUDA_MOE_HOST_SPIN_PAUSE")) != 0;
        hot_ms     = getenv("GGML_CUDA_MOE_HOST_HOT_MS") != nullptr ? std::max(1, atoi(getenv("GGML_CUDA_MOE_HOST_HOT_MS"))) : 50;
#if MH_X86
        if (getenv("GGML_CUDA_MOE_HOST_KEEP_WARM") == nullptr || atoi(getenv("GGML_CUDA_MOE_HOST_KEEP_WARM")) != 0) {
            keep_warm = dot2 == dot2_avx512 ? keep_warm_avx512 : dot2 == dot2_avx2 ? keep_warm_avx2 : nullptr;
        }
#endif
        threads.emplace_back(&pool::leader, this);
        for (int t = 1; t < n_threads; ++t) {
            threads.emplace_back(&pool::helper, this);
        }
#if defined(__linux__)
        // GGML_CUDA_MOE_HOST_PIN=1: one core per thread, so that two spinning threads never share a core;
        // GGML_CUDA_MOE_HOST_CPUS=c,c,...: the threads on these CPUs (one each, in order; the GPUs' NUMA node where
        // the process may run on both: the cold slices are read from memory next to it, GGML_CUDA_HOST_NUMA)
        std::vector<int> cpus;
        if (!pool_cpus.empty()) {
            cpus = pool_cpus;
        } else if (const char * e = getenv("GGML_CUDA_MOE_HOST_CPUS")) {
            for (const char * p = e; *p != 0;) {
                char * end = nullptr;
                const long c = strtol(p, &end, 10);
                if (end == p) {
                    break;
                }
                cpus.push_back((int) c);
                p = *end == ',' ? end + 1 : end;
            }
        } else if (getenv("GGML_CUDA_MOE_HOST_PIN") != nullptr && atoi(getenv("GGML_CUDA_MOE_HOST_PIN")) != 0) {
            cpus = cpu_list();
        }
        for (int t = 0; t < n_threads && !cpus.empty(); ++t) {
            const int cpu = cpus[(size_t) ((pool_cpus.empty() ? pool_id*n_threads : 0) + t) % cpus.size()];
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            pthread_setaffinity_np(threads[t].native_handle(), sizeof(set), &set);
            std::lock_guard<std::mutex> lock(g_pinned_mtx);
            g_pinned_cpus.push_back(cpu);
        }
#endif
        GGML_LOG_INFO("%s: pool %d of %d: %d host threads compute the cold experts (%s; %s%s)\n", __func__, pool_id, n_pools, n_threads, isa,
                      knew ? "Q2_K/Q3_K layers on the k kernels: " : "old kernels", knew ? kisa : "");
    }

    ~pool() {
        MH_TR(tr_print();)
        quit.store(true);
        {
            std::lock_guard<std::mutex> lock(cv_mtx);
            cv.notify_all();
        }
        for (auto & t : threads) {
            t.join();
        }
    }
};

constexpr int MH_MAX_POOLS = 4;

pool * g_pool(const int i) {
    static pool p[MH_MAX_POOLS];
    return &p[i];
}

std::mutex g_start_mtx;
int        g_n_pools = 0;

} // namespace

bool mh_cpu_supported() {
    return true;
}

std::vector<int> mh_pinned_cpus() {
    std::lock_guard<std::mutex> lock(g_pinned_mtx);
    return g_pinned_cpus;
}

void mh_pool_attach(int index, mh_mailbox * m, int n_devices, int device) {
    GGML_ASSERT(index >= 0 && index < MH_MAX_POOLS*MH_MAX_DEVICES);
    std::lock_guard<std::mutex> lock(g_start_mtx);
    if (g_n_pools == 0) {
        // The devices of a tensor split publish the cold pairs of a layer at the same time: with one pool the second
        // device waits for the job of the first, and the first then waits for it at the AllReduce.
        const char * env = getenv("GGML_CUDA_MOE_HOST_POOLS");
        g_n_pools = std::max(1, std::min(MH_MAX_POOLS, env != nullptr ? atoi(env) : n_devices));
    }
    const int ip = device >= 0 ? device % g_n_pools : index % g_n_pools;
    pool * p = g_pool(ip);
    const int im = device >= 0 ? p->n_mb.load() : index / g_n_pools;
    GGML_ASSERT(im < MH_MAX_DEVICES);
    if (p->threads.empty()) {
        p->start(ip, g_n_pools);
    }
    if (p->mb[im] == nullptr) {
        p->seen[im] = m->req;
        p->mb[im]   = m;
        p->n_mb.store(std::max(p->n_mb.load(), im + 1), std::memory_order_release);
    }
}

int mh_pool_register(const mh_slot_desc & d) {
    std::lock_guard<std::mutex> lock(g_slots.mtx);
    const int n = g_slots.n.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i) {
        if (g_slots.tab[i].down == d.down) {
            return i;
        }
    }
    GGML_ASSERT(n < (int) (sizeof(g_slots.tab)/sizeof(g_slots.tab[0])));
    GGML_ASSERT(d.n_embd % 64 == 0 && d.n_embd <= MH_MAX_EMBD && d.n_ff % 64 == 0 && d.n_ff <= MH_MAX_FF);
    g_slots.tab[n] = d;
    if (numa_split_enabled() && n < (int) g_slot_nodes.size()) {
        g_slot_nodes[n].gate = expert_nodes(d.gate, d.gate_nb2, d.n_cold);
        g_slot_nodes[n].down = expert_nodes(d.down, d.down_nb2, d.n_cold);
        int c[8] = {};
        for (const int8_t v : g_slot_nodes[n].gate) { c[v & 7]++; }
        GGML_LOG_WARN("moe-host: slot %d: gate experts by node: %d %d %d\n", n, c[0], c[1], c[2]);
    }
    g_slots.n.store(n + 1, std::memory_order_release);
    return n;
}
