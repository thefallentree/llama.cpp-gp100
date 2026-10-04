#pragma once

// temporary host-side timers (FN_PROF=1): per translation unit, printed at exit

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif
// 0: target context, 1: draft/MTP context (set by llama_context::decode)
#if defined(_WIN32)
extern int ggml_fn_prof_tag;
#else
extern __attribute__((visibility("default"))) int ggml_fn_prof_tag;
#endif
#ifdef __cplusplus
}
#endif

struct fn_prof_t {
    enum { MAXT = 96, RING = 64 };
    const char * tu;
    bool         on;
    int          cnt = 0;
    const char * names[MAXT];
    int          tags[MAXT];
    int64_t      sum[MAXT];
    int64_t      n[MAXT];
    int64_t      ring[MAXT][RING];

    fn_prof_t(const char * tu) : tu(tu) {
        on = getenv("FN_PROF") != nullptr;
    }
    ~fn_prof_t() {
        if (!on) {
            return;
        }
        for (int i = 0; i < cnt; ++i) {
            const int m = (int) std::min<int64_t>(n[i], RING);
            int64_t tmp[RING];
            memcpy(tmp, ring[i], m*sizeof(int64_t));
            std::sort(tmp, tmp + m);
            fprintf(stderr, "fn-prof %-8s tag%d %-28s n %7lld total %10.3f ms median(last %d) %9.1f us\n", tu, tags[i], names[i],
                    (long long) n[i], sum[i]/1e6, m, m > 0 ? tmp[m/2]/1e3 : 0.0);
        }
    }
    int id(const char * name) {
        const int tag = ggml_fn_prof_tag;
        for (int i = 0; i < cnt; ++i) {
            if (tags[i] == tag && (names[i] == name || strcmp(names[i], name) == 0)) {
                return i;
            }
        }
        if (cnt == MAXT) {
            return MAXT - 1;
        }
        names[cnt] = name;
        tags[cnt]  = tag;
        sum[cnt]   = 0;
        n[cnt]     = 0;
        return cnt++;
    }
    void add(const char * name, int64_t ns) {
        const int i = id(name);
        ring[i][n[i] % RING] = ns;
        sum[i] += ns;
        n[i]++;
    }
};

static inline int64_t fn_prof_now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#define FN_PROF_DECL(tu) static fn_prof_t g_fn_prof(tu)
#define FN_PROF_T(var) const int64_t var = g_fn_prof.on ? fn_prof_now() : 0
#define FN_PROF_ADD(name, t0) do { if (g_fn_prof.on) { g_fn_prof.add(name, fn_prof_now() - (t0)); } } while (0)
