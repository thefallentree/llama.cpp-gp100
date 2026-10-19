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

#if defined(__linux__)
#include <pthread.h>
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
typedef void (*quant_fn)(const float *, int, act_block *);

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
    quant_fn     quant = nullptr;
    dot2k_fn     dot2k  = nullptr;   // Q2_K cold experts (natural-order activations)
    dot2k_fn     dot2k3 = nullptr;   // Q3_K cold experts (the down of deepseek41's Q2_K files)
    quantn_fn    quantn = nullptr;
    const char * isa   = nullptr;
    void (*keep_warm)() = nullptr; // called by the spinning threads
    bool spin_pause = true;

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
    size_t                 cap_blk = 0;
    std::unique_ptr<std::atomic<int>[]> blk_cnt;   // [n_groups][n_ff/64]: finished CH1-row chunks per expert block
    std::unique_ptr<std::atomic<int>[]> grp_ready; // [n_groups]: quantized activation blocks per expert

    alignas(64) std::atomic<uint32_t> job_gen { 0 };
    alignas(64) std::atomic<int>      chunk1 { 0 };
    alignas(64) std::atomic<int>      chunk3 { 0 };
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

    void helper() {
        uint32_t last = job_gen.load(std::memory_order_acquire);
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
            work();
            barrier();
        }
    }

    // GGML_CUDA_MOE_HOST_STATS=N: every N jobs, print the mean job time, cold pairs and distinct experts per job
    int64_t stat_every = 0, stat_jobs = 0, stat_pairs = 0, stat_groups = 0;
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
            static std::atomic<int> n_logged{0};
            const bool bad = min_c < 0 || (js->n_cold > 0 && max_c >= js->n_cold) || m->n_tokens <= 0 || n_used <= 0;
            if (n_logged.load() < 3 || bad) {
                n_logged++;
                char pairs[256]; int o = 0;
                for (int p = 0; p < std::min(n_pairs, 8) && o < 200; ++p) {
                    o += snprintf(pairs + o, sizeof(pairs) - o, " (%d,%d)", (int) pidx[p], (int) pexp[p]);
                }
                GGML_LOG_WARN("moe-host: job slot %d: n_tokens %d n_used %d n_pairs %d (cap %d) experts %d..%d of %d, n_ff %d n_embd %d wtype %d/%d:%s%s\n",
                              slot, (int) m->n_tokens, n_used, n_pairs, (int) m->cap_pairs, min_c, max_c, js->n_cold, js->n_ff, js->n_embd,
                              js->wtype, js->wtype_down, pairs, bad ? "  <- OUT OF RANGE, job dropped" : "");
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
        if (h.size() < (size_t) n_pairs*js->n_ff) {
            h.resize((size_t) n_pairs*js->n_ff);
            hq.resize((size_t) n_pairs*nb_f);
        }
        if (js->wtype == 1 && hqn.size() < (size_t) n_pairs*nb_f) {
            hqn.resize((size_t) n_pairs*nb_f);
        }
        if (cap_blk < (size_t) n_groups*nb_f) {
            cap_blk = (size_t) n_groups*nb_f;
            blk_cnt.reset(new std::atomic<int>[cap_blk]);
            grp_ready.reset(new std::atomic<int>[cap_blk]);
        }
        for (size_t i = 0; i < (size_t) n_groups*nb_f; ++i) {
            blk_cnt[i].store(0, std::memory_order_relaxed);
        }
        for (int gi = 0; gi < n_groups; ++gi) {
            grp_ready[gi].store(0, std::memory_order_relaxed);
        }
        // the GPU quantized the input rows (mh_publish)
        jxq = mh_xq(m);
        jy  = mh_y(m);
        if (js->wtype == 1) {
            // Q2_K: the dot wants the activations in natural order with their 16-sums
            const int    nb_e = js->n_embd/QK;
            const size_t n    = (size_t) m->n_tokens*nb_e;
            if (xqn.size() < n) {
                xqn.resize(n);
            }
            act_to_nat(jxq, (int) n, xqn.data());
        }
        const auto t_quant = std::chrono::steady_clock::now();
        p1_left.store(n_groups*nb_f, std::memory_order_relaxed);
        MH_TR(t_job_ns.store(tns());)
        chunk1.store(0, std::memory_order_relaxed);
        job_gen.fetch_add(1, std::memory_order_acq_rel);
        work();
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
                fprintf(stderr, "moe-host: pool %d: %lld jobs: %.1f us/job (setup %.1f, gate/up done at %.1f, leader's work done at %.1f us), %.2f cold pairs, %.2f experts per job\n",
                              id, (long long) stat_jobs, stat_us/stat_jobs, stat_quant_us/stat_jobs, stat_p1_us/stat_jobs, stat_work_us/stat_jobs,
                              (double) stat_pairs/stat_jobs, (double) stat_groups/stat_jobs);
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
                if (!hot.load(std::memory_order_relaxed)) {
                    hot.store(true);
                    std::lock_guard<std::mutex> lock(cv_mtx);
                    cv.notify_all();
                }
                run_job(m);
                m->done = r;
                found    = true;
                last_job = std::chrono::steady_clock::now();
                nap_us   = 0;
            }
            if (found) {
                continue;
            }
            if (hot.load(std::memory_order_relaxed)) {
                if (std::chrono::steady_clock::now() - last_job > std::chrono::milliseconds(50)) {
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
        stat_every = getenv("GGML_CUDA_MOE_HOST_STATS") ? atoll(getenv("GGML_CUDA_MOE_HOST_STATS")) : 0;
        const char * env = getenv("GGML_CUDA_MOE_HOST_THREADS");
        const int hw = (int) std::thread::hardware_concurrency();
        const int total = env != nullptr ? atoi(env) : std::max(1, std::min(16, hw - 4));
        n_threads = std::max(1, total/n_pools);
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
        spin_pause = getenv("GGML_CUDA_MOE_HOST_SPIN_PAUSE") == nullptr || atoi(getenv("GGML_CUDA_MOE_HOST_SPIN_PAUSE")) != 0;
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
        if (const char * e = getenv("GGML_CUDA_MOE_HOST_CPUS")) {
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
            const int cpu = cpus[(size_t) (pool_id*n_threads + t) % cpus.size()];
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            pthread_setaffinity_np(threads[t].native_handle(), sizeof(set), &set);
            std::lock_guard<std::mutex> lock(g_pinned_mtx);
            g_pinned_cpus.push_back(cpu);
        }
#endif
        GGML_LOG_INFO("%s: pool %d of %d: %d host threads compute the cold experts (%s)\n", __func__, pool_id, n_pools, n_threads, isa);
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

void mh_pool_attach(int index, mh_mailbox * m, int n_devices) {
    GGML_ASSERT(index >= 0 && index < MH_MAX_POOLS*MH_MAX_DEVICES);
    std::lock_guard<std::mutex> lock(g_start_mtx);
    if (g_n_pools == 0) {
        // The devices of a tensor split publish the cold pairs of a layer at the same time: with one pool the second
        // device waits for the job of the first, and the first then waits for it at the AllReduce.
        const char * env = getenv("GGML_CUDA_MOE_HOST_POOLS");
        g_n_pools = std::max(1, std::min(MH_MAX_POOLS, env != nullptr ? atoi(env) : n_devices));
    }
    const int ip = index % g_n_pools;
    const int im = index / g_n_pools;
    pool * p = g_pool(ip);
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
    g_slots.n.store(n + 1, std::memory_order_release);
    return n;
}
