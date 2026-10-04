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

static inline void mh_prefetch(const void * p) {
#if MH_X86
    _mm_prefetch((const char *) p, _MM_HINT_T0);
#else
    __builtin_prefetch(p);
#endif
}

typedef void (*dot2_fn)(const q2_block *, const q2_block *, const act_block *, int, float *, float *);
typedef void (*quant_fn)(const float *, int, act_block *);

// GGML_CUDA_MOE_HOST_ISA=generic|avx2 forces a narrower kernel
static void pick_kernels(dot2_fn & dot2, quant_fn & quant, const char *& name) {
    dot2  = dot2_generic;
    quant = quantize_act_generic;
    name  = "generic";
#if MH_X86
    const char * force = getenv("GGML_CUDA_MOE_HOST_ISA");
    __builtin_cpu_init();
    const bool has512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq");
    const bool has2   = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    if (force != nullptr && strcmp(force, "generic") == 0) {
        return;
    }
    if (has2) {
        quant = quantize_act_avx2;
        dot2  = dot2_avx2;
        name  = "AVX2";
    }
    if (has512 && has2 && !(force != nullptr && strcmp(force, "avx2") == 0)) {
        dot2 = dot2_avx512;
        name = "AVX-512";
    }
#endif
}

struct group {
    int c;                  // cold expert
    int n;                  // pairs in the group
    int pair[MH_MAX_TOK];   // pair index p (into mailbox y)
    int tok[MH_MAX_TOK];    // token of each pair
};

struct pool {
    int          n_threads = 0;
    dot2_fn      dot2  = nullptr;
    quant_fn     quant = nullptr;
    const char * isa   = nullptr;

    std::vector<std::thread> threads;
    mh_mailbox * mb[MH_MAX_DEVICES] = {};
    uint32_t     seen[MH_MAX_DEVICES] = {};
    std::atomic<int> n_mb { 0 };

    std::mutex                slots_mtx;
    std::atomic<int>          n_slots { 0 };
    mh_slot_desc              slot_tab[1024];

    // the current job, written by the leader before job_gen is bumped
    const mh_slot_desc * js = nullptr;
    mh_mailbox *         jm = nullptr;
    int                  n_groups = 0;
    group                groups[MH_MAX_PAIRS];
    const act_block *    jxq = nullptr; // [n_tokens][n_embd/64], in the mailbox
    std::vector<float>     h;    // [MH_MAX_PAIRS][n_ff]: the swiglu outputs
    std::vector<act_block> hq;   // [MH_MAX_PAIRS][n_ff/64]: ... quantized
    std::unique_ptr<std::atomic<int>[]> blk_cnt;   // [MH_MAX_PAIRS][MH_MAX_FF/64]: finished CH1-row chunks per expert block
    std::unique_ptr<std::atomic<int>[]> grp_ready; // [MH_MAX_PAIRS]: quantized activation blocks per expert

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
    static constexpr int CH1 = 16;
    static constexpr int CH3 = 32;

    void work() {
        const mh_slot_desc & s = *js;
        const int nb_e = s.n_embd/QK;
        const int nb_f = s.n_ff/QK;
        const int rb1  = s.n_ff/CH1;
        const int n1   = n_groups*rb1;
        const int rb3  = s.n_embd/CH3;
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
                for (int r = r0; r < r0 + CH1; ++r) {
                    const q2_block * wg = (const q2_block *) (gate + (size_t) r*s.gate_nb1);
                    const q2_block * wu = (const q2_block *) (up   + (size_t) r*s.up_nb1);
                    for (int i = 0; i < g.n; ++i) {
                        float vg, vu;
                        dot2(wg, wu, jxq + (size_t) g.tok[i]*nb_e, nb_e, &vg, &vu);
                        h[(size_t) g.pair[i]*s.n_ff + r] = vg/(1.0f + expf(-vg))*vu;
                    }
                }
                const int kb = r0/QK;
                if (blk_cnt[(size_t) gi*(MH_MAX_FF/QK) + kb].fetch_add(1, std::memory_order_acq_rel) == QK/CH1 - 1) {
                    for (int i = 0; i < g.n; ++i) {
                        const int p = g.pair[i];
                        quant(h.data() + (size_t) p*s.n_ff + kb*QK, QK, hq.data() + (size_t) p*nb_f + kb);
                    }
                    grp_ready[gi].fetch_add(1, std::memory_order_release);
                }
                continue;
            }
            const int       c3 = c - n1;
            const int       gi = c3/rb3;
            const group &   g  = groups[gi];
            const int       r0 = (c3 % rb3)*CH3;
            const uint8_t * down = s.down + (size_t) g.c*s.down_nb2;
            while (grp_ready[gi].load(std::memory_order_acquire) < nb_f) {
#if MH_X86
                _mm_pause();
#endif
            }
            for (int r = r0; r < r0 + CH3; r += 2) {
                const q2_block * w0 = (const q2_block *) (down + (size_t) (r + 0)*s.down_nb1);
                const q2_block * w1 = (const q2_block *) (down + (size_t) (r + 1)*s.down_nb1);
                for (int i = 0; i < g.n; ++i) {
                    const int p = g.pair[i];
                    float v0, v1;
                    dot2(w0, w1, hq.data() + (size_t) p*nb_f, nb_f, &v0, &v1);
                    float * y = jm->y + (size_t) p*s.n_embd;
                    y[r + 0] = v0;
                    y[r + 1] = v1;
                }
            }
        }
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
                    _mm_pause();
#endif
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
    double  stat_us = 0.0, stat_quant_us = 0.0;

    void run_job(mh_mailbox * m) {
        const auto t_start = std::chrono::steady_clock::now();
        std::atomic_thread_fence(std::memory_order_acquire);
        const int slot = m->slot;
        if (slot < 0 || slot >= n_slots.load(std::memory_order_acquire)) {
            GGML_LOG_ERROR("%s: bad slot %d\n", __func__, slot);
            return;
        }
        js = &slot_tab[slot];
        jm = m;
        const int n_used  = m->n_used;
        const int n_pairs = std::min<int>(m->n_pairs, MH_MAX_PAIRS);
        const int nb_e    = js->n_embd/QK;

        n_groups = 0;
        bool tok_used[MH_MAX_TOK] = {};
        for (int p = 0; p < n_pairs; ++p) {
            const int c = m->pair_exp[p];
            const int t = m->pair_idx[p]/n_used;
            tok_used[t] = true;
            int gi = 0;
            while (gi < n_groups && groups[gi].c != c) {
                ++gi;
            }
            if (gi == n_groups) {
                groups[n_groups].c = c;
                groups[n_groups].n = 0;
                ++n_groups;
            }
            group & g = groups[gi];
            g.pair[g.n] = p;
            g.tok[g.n]  = t;
            ++g.n;
        }
        // the GPU quantized the input rows (mh_publish)
        jxq = m->xq;
        for (int gi = 0; gi < n_groups; ++gi) {
            for (int kb = 0; kb < js->n_ff/QK; ++kb) {
                blk_cnt[(size_t) gi*(MH_MAX_FF/QK) + kb].store(0, std::memory_order_relaxed);
            }
            grp_ready[gi].store(0, std::memory_order_relaxed);
        }
        GGML_UNUSED(tok_used);
        GGML_UNUSED(nb_e);
        const auto t_quant = std::chrono::steady_clock::now();
        chunk1.store(0, std::memory_order_relaxed);
        job_gen.fetch_add(1, std::memory_order_acq_rel);
        work();
        barrier();
        std::atomic_thread_fence(std::memory_order_release);
        if (stat_every > 0) {
            const auto t_end = std::chrono::steady_clock::now();
            stat_us       += std::chrono::duration<double, std::micro>(t_end - t_start).count();
            stat_quant_us += std::chrono::duration<double, std::micro>(t_quant - t_start).count();
            stat_pairs    += n_pairs;
            stat_groups   += n_groups;
            if (++stat_jobs == stat_every) {
                fprintf(stderr, "moe-host: %lld jobs: %.1f us/job (setup %.1f us), %.2f cold pairs, %.2f experts per job\n",
                              (long long) stat_jobs, stat_us/stat_jobs, stat_quant_us/stat_jobs,
                              (double) stat_pairs/stat_jobs, (double) stat_groups/stat_jobs);
                stat_jobs = stat_pairs = stat_groups = 0;
                stat_us = stat_quant_us = 0.0;
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
                _mm_pause();
#endif
            } else {
                // idle: nap, so an idle server does not burn a core; the first layer after it waits for the nap
                nap_us = std::min(nap_us + 10, 200);
                std::this_thread::sleep_for(std::chrono::microseconds(nap_us));
            }
        }
    }

    void start() {
        pick_kernels(dot2, quant, isa);
        stat_every = getenv("GGML_CUDA_MOE_HOST_STATS") ? atoll(getenv("GGML_CUDA_MOE_HOST_STATS")) : 0;
        const char * env = getenv("GGML_CUDA_MOE_HOST_THREADS");
        const int hw = (int) std::thread::hardware_concurrency();
        n_threads = env != nullptr ? atoi(env) : std::max(1, std::min(16, hw - 4));
        n_threads = std::max(1, n_threads);
        h.resize((size_t) MH_MAX_PAIRS*MH_MAX_FF);
        hq.resize((size_t) MH_MAX_PAIRS*(MH_MAX_FF/QK));
        blk_cnt.reset(new std::atomic<int>[(size_t) MH_MAX_PAIRS*(MH_MAX_FF/QK)]);
        grp_ready.reset(new std::atomic<int>[MH_MAX_PAIRS]);
        threads.emplace_back(&pool::leader, this);
        for (int t = 1; t < n_threads; ++t) {
            threads.emplace_back(&pool::helper, this);
        }
        GGML_LOG_INFO("%s: %d host threads compute the cold experts (%s)\n", __func__, n_threads, isa);
    }

    ~pool() {
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

pool * g_pool() {
    static pool p;
    return &p;
}

std::mutex g_start_mtx;

} // namespace

bool mh_cpu_supported() {
    return true;
}

void mh_pool_attach(int device, mh_mailbox * m) {
    GGML_ASSERT(device >= 0 && device < MH_MAX_DEVICES);
    pool * p = g_pool();
    std::lock_guard<std::mutex> lock(g_start_mtx);
    if (p->threads.empty()) {
        p->start();
    }
    if (p->mb[device] == nullptr) {
        p->seen[device] = m->req;
        p->mb[device]   = m;
        p->n_mb.store(std::max(p->n_mb.load(), device + 1), std::memory_order_release);
    }
}

int mh_pool_register(const mh_slot_desc & d) {
    pool * p = g_pool();
    std::lock_guard<std::mutex> lock(p->slots_mtx);
    const int n = p->n_slots.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i) {
        if (p->slot_tab[i].down == d.down) {
            return i;
        }
    }
    GGML_ASSERT(n < (int) (sizeof(p->slot_tab)/sizeof(p->slot_tab[0])));
    GGML_ASSERT(d.n_embd % 64 == 0 && d.n_embd <= MH_MAX_EMBD && d.n_ff % 64 == 0 && d.n_ff <= MH_MAX_FF);
    p->slot_tab[n] = d;
    p->n_slots.store(n + 1, std::memory_order_release);
    return n;
}
