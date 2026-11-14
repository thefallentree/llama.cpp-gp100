// regression test for the server's host prompt cache (server_prompt_cache, tools/server/server-task.cpp): a restored
// conversation must have every cell of its prompt in the attention cache (CL1, cross-session leak, 2026-11-11)
//
// - delta save: a drafter context whose cells lag the target (the MTP drafter) must not make save() drop the kept
//   segments after the first one and start the new chunk at the common prefix (the dropped cells were in no segment)
// - load guard: an entry whose segments do not cover its prompt is dropped instead of restored
//
// - HC1 (2026-11-12): eviction is least-recently-used (a restored entry survives a later save that needs room), older
//   checkpoints of unowned entries are shed before whole entries go, and find_better() names the entry a returning
//   conversation would restore while the slot holds a short other conversation sharing the system prefix
// - HC2: a delta save keeps the earlier segments although the drafter lags the target (only the new cells are saved)
// - HC3 (gpu-50 wedge, 2026-11-12): a save whose buffer fits no free block while the pool's slab prefetch is in flight
//   must not hang (the pre-HC3 alloc() re-entered itself holding the pool mutex: a self-deadlock of the main thread);
//   a save the pool cannot hold (past its capacity cap, nothing to evict) is skipped and the next one works.
//   A watchdog fails a scenario that does not return in 60 s.
// - TF1 (audit for "an entry evicted during its own pass", 2026-11-14):
//   K: a delta save over the size limit is skipped and must leave the slot's origin entry whole (it kept only the
//      segments before the common prefix under the entry's full token list: a hole the next restore dropped the
//      entry for, after wiping the slot's cells)
//   L: three slots on one context pair under a tiny size limit and a capped pool: every save evicts or trims, delta
//      saves, divergent tails, idle saves, clears; after every step no entry a slot holds was freed or handed to
//      another slot, every entry tiles its prompt, every pool block belongs to exactly one live buffer
//      (build with -DLLAMA_SANITIZE_ADDRESS=ON to catch a use-after-free too)
//
// CPU only, a tiny generated model (test-generate-models): test-server-prompt-cache -m <model.gguf>

#include "server-task.h"
#include "server-queue.h"
#include "llama.h"
#include "log.h"
#include "../src/llama-ext.h" // llama_memory_seq_pos_gap
#include "../ggml/src/ggml-backend-impl.h" // a slow host buffer type (pinning takes 0.4 s per 512 MiB on the nodes)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHECK(cond) do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return false; \
        } \
    } while (0)

static const llama_seq_id SEQ = 0;

// the drafter's cells end this many tokens before the target's at every save
static const int LAG = 8;

static llama_token conv_token(int32_t n_vocab, llama_pos pos) {
    return (llama_token) ((11*(uint32_t) pos + 3) % (uint32_t) n_vocab);
}

static bool decode_range(llama_context * ctx, const llama_tokens & conv, llama_pos p0, llama_pos p1, llama_seq_id seq = SEQ) {
    if (p0 >= p1) {
        return true;
    }
    llama_batch batch = llama_batch_init(p1 - p0, 0, 1);
    for (llama_pos p = p0; p < p1; ++p) {
        batch.token   [batch.n_tokens] = conv[p];
        batch.pos     [batch.n_tokens] = p;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id  [batch.n_tokens][0] = seq;
        batch.logits  [batch.n_tokens] = p + 1 == p1;
        batch.n_tokens++;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

// the slot: its prompt, the cache entry it is (server_slot::prompt_origin), the two contexts
struct slot {
    llama_context * tgt = nullptr;
    llama_context * dft = nullptr;
    server_prompt prompt;
    server_prompt_cache_state * origin = nullptr;

    // the slot's cells go, the entry stays in the cache (server_slot::prompt_clear)
    void clear() {
        llama_memory_seq_rm(llama_get_memory(tgt), SEQ, -1, -1);
        llama_memory_seq_rm(llama_get_memory(dft), SEQ, -1, -1);
        prompt.clear();
        if (origin != nullptr) {
            origin->owner = -1;
            origin = nullptr;
        }
    }

    // decode the conversation up to n tokens on the target and up to n - LAG on the drafter, from what each holds
    bool extend(const llama_tokens & conv, int n) {
        const llama_pos t0 = llama_memory_seq_pos_max(llama_get_memory(tgt), SEQ) + 1;
        const llama_pos d0 = llama_memory_seq_pos_max(llama_get_memory(dft), SEQ) + 1;
        CHECK(decode_range(tgt, conv, t0, n));
        CHECK(decode_range(dft, conv, d0, n - LAG));
        prompt.tokens = server_tokens(llama_tokens(conv.begin(), conv.begin() + n), false);
        return true;
    }
};

// one turn of a conversation: restore the entry (when the slot is empty), decode the new tokens, save
static bool turn(server_prompt_cache & cache, slot & s, const llama_tokens & conv, int n) {
    if (s.prompt.n_tokens() == 0) {
        server_tokens tokens_new(llama_tokens(conv.begin(), conv.begin() + n), false);
        CHECK(cache.load(s.prompt, tokens_new, s.tgt, s.dft, nullptr, SEQ, nullptr, &s.origin));
    }
    const int n_cached = s.prompt.n_tokens();
    CHECK(n_cached <= n);
    // the restored prefix has no hole (the slot would reuse it)
    CHECK(llama_memory_seq_pos_gap(s.tgt, SEQ, 0) >= n_cached);
    CHECK(s.extend(conv, n));
    auto * e = cache.save(s.prompt, s.tgt, s.dft, nullptr, SEQ, &s.origin, true);
    CHECK(e != nullptr);
    CHECK(s.origin == e);
    return true;
}

// scenario A: three turns with delta saves and a lagging drafter, then a restore of the final entry must cover the
// whole prompt (pre-fix: the second segment was dropped at the third save, the entry had a hole [N1, N2))
static bool scenario_save_keeps_every_cell(server_prompt_cache & cache, slot & s, const llama_tokens & conv) {
    const int N1 = 40;
    const int N2 = 72;
    const int N3 = 104;

    CHECK(turn(cache, s, conv, N1));
    CHECK(cache.states.size() == 1);
    CHECK(s.origin->data.segs.size() == 1);
    s.clear();

    CHECK(turn(cache, s, conv, N2));
    CHECK(cache.states.size() == 1);
    CHECK(s.origin->data.segs.size() == 2);
    s.clear();

    CHECK(turn(cache, s, conv, N3));
    CHECK(cache.states.size() == 1);
    const auto * e = s.origin;
    s.clear();

    // the entry's segments must tile [0, N3): every position of the prompt is in one of them
    llama_pos covered = 0;
    for (const auto & seg : e->data.segs) {
        fprintf(stderr, "  segment [%d, %d) drafter [%d, %d)\n", seg.p0, seg.p1, seg.d0, seg.d1);
        CHECK(seg.p0 <= covered);
        covered = seg.p1;
    }
    CHECK(covered >= N3);

    // restore into an empty sequence: the attention cache must hold [0, N3) without a hole
    server_tokens tokens_new(llama_tokens(conv.begin(), conv.begin() + N3), false);
    const bool loaded = cache.load(s.prompt, tokens_new, s.tgt, s.dft, nullptr, SEQ, nullptr, &s.origin);
    const llama_pos gap = llama_memory_seq_pos_gap(s.tgt, SEQ, 0);
    fprintf(stderr, "  restore of the %d-token entry: load %d, attention cells contiguous up to pos %d\n", N3, (int) loaded, gap);
    CHECK(loaded);
    CHECK(s.prompt.n_tokens() == N3);
    CHECK(gap >= N3);
    s.clear();
    return true;
}

// scenario B: an entry that lost its first segment must not be restored (load fails, the entry is dropped)
static bool scenario_load_refuses_a_hole(server_prompt_cache & cache, slot & s, const llama_tokens & conv) {
    const int N1 = 40;
    const int N2 = 72;

    cache.states.clear();
    CHECK(turn(cache, s, conv, N1));
    s.clear();
    CHECK(turn(cache, s, conv, N2));
    auto * e = s.origin;
    s.clear();
    CHECK(cache.states.size() == 1);
    CHECK(e->data.segs.size() == 2);

    // the cells [0, N1) are in no segment of the entry now
    e->data.segs.erase(e->data.segs.begin());

    server_tokens tokens_new(llama_tokens(conv.begin(), conv.begin() + N2), false);
    const bool loaded = cache.load(s.prompt, tokens_new, s.tgt, s.dft, nullptr, SEQ, nullptr, &s.origin);
    const llama_pos gap = llama_memory_seq_pos_gap(s.tgt, SEQ, 0);
    fprintf(stderr, "  restore of the %d-token entry without its first segment: load %d, attention cells contiguous up to pos %d, %zu entries left\n",
            N2, (int) loaded, gap, cache.states.size());
    CHECK(!loaded);
    CHECK(cache.states.empty());
    CHECK(s.origin == nullptr);
    CHECK(s.prompt.n_tokens() == 0);
    // a failed restore leaves the sequence empty
    CHECK(llama_memory_seq_pos_max(llama_get_memory(s.tgt), SEQ) == -1);
    s.clear();
    return true;
}


// a conversation sharing the first n_shared tokens (the "system prompt") with every other one
static llama_tokens make_conv(int32_t n_vocab, int id, int n_shared, int n) {
    llama_tokens conv;
    for (llama_pos p = 0; p < n; ++p) {
        conv.push_back(p < n_shared ? conv_token(n_vocab, p) : (llama_token) ((7*(uint32_t) p + 101*(uint32_t) id + 5) % (uint32_t) n_vocab));
    }
    return conv;
}

static common_prompt_checkpoint fake_ckpt(int64_t n_tokens, size_t bytes) {
    common_prompt_checkpoint c;
    c.n_tokens = n_tokens;
    c.pos_min  = (llama_pos) n_tokens - 1;
    c.pos_max  = (llama_pos) n_tokens - 1;
    c.data_tgt.assign(bytes, 1);
    return c;
}

static const server_prompt_cache_state * entry_of(const server_prompt_cache & cache, const llama_tokens & conv) {
    for (const auto & st : cache.states) {
        if (st.prompt.n_tokens() > 0 && st.prompt.tokens.get_common_prefix(server_tokens(conv, false)) == (size_t) st.prompt.n_tokens()) {
            return &st;
        }
    }
    return nullptr;
}

// scenario C: LRU eviction. Entries A, B, C fill the cache; A is restored (used again); D needs room: B (the least
// recently used) goes, A stays. Before HC1 eviction followed the first save: A went although it was just used.
static bool scenario_lru(server_prompt_cache & cache, slot & s, int32_t n_vocab) {
    cache.states.clear();
    const int N = 48;
    llama_tokens A = make_conv(n_vocab, 1, 8, N), B = make_conv(n_vocab, 2, 8, N), C = make_conv(n_vocab, 3, 8, N), D = make_conv(n_vocab, 4, 8, N);
    const size_t limit0 = cache.limit_size;
    cache.limit_size = 0;
    CHECK(turn(cache, s, A, N)); s.clear();
    const size_t sz = cache.size();
    CHECK(turn(cache, s, B, N)); s.clear();
    CHECK(turn(cache, s, C, N)); s.clear();
    CHECK(cache.states.size() == 3);
    // room for three entries and a half
    cache.limit_size = sz * 7 / 2;
    // A is used again: restored and saved with a few more tokens
    llama_tokens A2 = A;
    for (int i = 0; i < 8; ++i) A2.push_back(conv_token(n_vocab, 200 + i));
    CHECK(turn(cache, s, A2, N + 8)); s.clear();
    CHECK(turn(cache, s, D, N)); s.clear();
    fprintf(stderr, "  entries after D: %zu (A %d B %d C %d D %d)\n", cache.states.size(), entry_of(cache, A2) != nullptr,
            entry_of(cache, B) != nullptr, entry_of(cache, C) != nullptr, entry_of(cache, D) != nullptr);
    CHECK(entry_of(cache, A2) != nullptr);
    CHECK(entry_of(cache, B) == nullptr);
    CHECK(entry_of(cache, D) != nullptr);
    cache.limit_size = limit0;
    return true;
}

// scenario D: under pressure an unowned entry sheds its older checkpoints (the newest stay) before any entry is evicted
static bool scenario_trim(server_prompt_cache & cache, slot & s, int32_t n_vocab) {
    cache.states.clear();
    const int N = 48;
    llama_tokens A = make_conv(n_vocab, 5, 8, N), B = make_conv(n_vocab, 6, 8, N);
    const size_t limit0 = cache.limit_size;
    cache.limit_size = 0;
    CHECK(turn(cache, s, A, N));
    const size_t sz = cache.size();
    s.clear();
    // A carries 8 checkpoints of 1/2 entry each
    auto * ea = const_cast<server_prompt_cache_state *>(entry_of(cache, A));
    CHECK(ea != nullptr);
    for (int i = 0; i < 8; ++i) {
        ea->prompt.checkpoints.push_back(fake_ckpt(5 + 5*i, sz / 2));
    }
    // room for two plain entries + 1.5 checkpoints: B's save must trim A to 3 checkpoints, not evict it
    cache.limit_size = 2*sz + 3*sz/4 + 3*(sz/2);
    CHECK(turn(cache, s, B, N)); s.clear();
    ea = const_cast<server_prompt_cache_state *>(entry_of(cache, A));
    fprintf(stderr, "  after B: %zu entries, A %s with %zu checkpoints\n", cache.states.size(), ea ? "kept" : "evicted", ea ? ea->prompt.checkpoints.size() : 0);
    CHECK(ea != nullptr);
    CHECK(ea->prompt.checkpoints.size() == 3);
    // the newest three stay
    int64_t lo = INT64_MAX;
    for (const auto & c : ea->prompt.checkpoints) lo = std::min(lo, c.n_tokens);
    CHECK(lo == 5 + 5*5);
    CHECK(entry_of(cache, B) != nullptr);
    cache.limit_size = limit0;
    return true;
}

// scenario E: the slot holds a short conversation S that shares the system prefix with A; A's long entry is in the
// cache. f_keep of the slot for A's next request is high (S is mostly the shared prefix), yet find_better names A's
// entry (the server then saves S and restores A instead of re-prefilling A from the shared prefix)
static bool scenario_find_better(server_prompt_cache & cache, slot & s, int32_t n_vocab) {
    cache.states.clear();
    const int NS = 40, NA = 120, SH = 24; // SH/NA < 0.25: S's first turn does not restore A
    llama_tokens A = make_conv(n_vocab, 7, SH, NA + 8), S = make_conv(n_vocab, 8, SH, NS);
    CHECK(turn(cache, s, A, NA)); s.clear();
    CHECK(turn(cache, s, S, NS));   // the slot keeps S (and owns its entry)
    server_tokens next(llama_tokens(A.begin(), A.begin() + NA + 8), false);
    const int lcp_slot = (int) s.prompt.tokens.get_common_prefix(next);
    const float f_keep = float(lcp_slot) / s.prompt.n_tokens();
    const auto * better = cache.find_better(s.prompt, next, SEQ, 16);
    fprintf(stderr, "  slot f_keep %.2f for A's request; find_better -> %s\n", f_keep, better ? "A's entry" : "none");
    CHECK(f_keep >= 0.5f);
    CHECK(better != nullptr && better->prompt.n_tokens() == NA);
    CHECK(cache.find_better(s.prompt, next, SEQ, NA) == nullptr);
    s.clear();
    return true;
}
// scenario F (HC2): a delta save keeps every earlier segment and saves only the new cells. The drafter lags the target
// (LAG), so each segment's drafter part ends before the target part; the next segment's drafter part starts there, which
// is no hole. Before HC2 the save took that lag for a hole and re-saved everything after the first segment on every turn
// (O(conversation) bytes per save); LLAMA_CACHE_HOLE_OLD=1 restores that rule and fails this scenario.
static bool scenario_delta_keeps_segments(server_prompt_cache & cache, slot & s, int32_t n_vocab) {
    cache.states.clear();
    const llama_tokens conv = make_conv(n_vocab, 9, 8, 200);
    const int N[] = {40, 72, 104, 136};
    for (int i = 0; i < 4; ++i) {
        CHECK(turn(cache, s, conv, N[i]));
        const auto * e = s.origin;
        CHECK(cache.states.size() == 1);
        fprintf(stderr, "  turn %d (%d tokens): %zu segments:", i, N[i], e->data.segs.size());
        for (const auto & seg : e->data.segs) {
            fprintf(stderr, " [%d,%s) d[%d,%d)", seg.p0, seg.p1 == INT32_MAX ? "inf" : std::to_string(seg.p1).c_str(), seg.d0, seg.d1);
        }
        fprintf(stderr, "\n");
        // one segment per turn, each starting where the previous turn ended
        CHECK(e->data.segs.size() == (size_t) i + 1);
        CHECK(e->data.segs.back().p0 == (i == 0 ? 0 : N[i - 1]));
        s.clear();
    }
    // the restore still covers the whole prompt (target and drafter)
    server_tokens tokens_new(llama_tokens(conv.begin(), conv.begin() + N[3]), false);
    CHECK(cache.load(s.prompt, tokens_new, s.tgt, s.dft, nullptr, SEQ, nullptr, &s.origin));
    const llama_pos gap_t = llama_memory_seq_pos_gap(s.tgt, SEQ, 0);
    const llama_pos gap_d = llama_memory_seq_pos_gap(s.dft, SEQ, 0);
    fprintf(stderr, "  restore: target contiguous up to %d, drafter up to %d (drafter cells end at %d)\n", gap_t, gap_d, N[3] - LAG);
    CHECK(s.prompt.n_tokens() == N[3]);
    CHECK(gap_t >= N[3]);
    CHECK(gap_d == N[3] - LAG);
    s.clear();
    return true;
}

// a host buffer type that takes SLOW_MS per allocation (cudaMallocHost of a slab): the pool's slab prefetch thread is then
// still running when the next allocation of a save looks for room
static const int SLOW_MS = 300;
static ggml_backend_buffer_t slow_alloc_buffer(ggml_backend_buffer_type_t, size_t size) {
    std::this_thread::sleep_for(std::chrono::milliseconds(SLOW_MS));
    return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
}
static const char * slow_get_name(ggml_backend_buffer_type_t) { return "slow_host"; }
static size_t slow_get_alignment(ggml_backend_buffer_type_t) { return 64; }
static bool slow_is_host(ggml_backend_buffer_type_t) { return true; }
static ggml_backend_buffer_type slow_buft = {
    /* .iface = */ {
        /* .get_name         = */ slow_get_name,
        /* .alloc_buffer     = */ slow_alloc_buffer,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ slow_get_alignment,
        /* .get_max_size     = */ nullptr,
        /* .get_alloc_size   = */ nullptr,
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ slow_is_host,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

// scenario G (HC3): the pool alone. A 40 KiB allocation makes the first 64 KiB slab and starts the prefetch of the next
// (less than half a slab is free); a 100 KiB allocation right after fits no free block, joins the prefetch and finds
// its slab pending. Pre-HC3: alloc() called itself with the pool mutex held and the inner call locked it again.
static bool scenario_pool_alloc_during_prefetch(server_prompt_cache &, slot &, int32_t) {
    server_state_pool pool;
    pool.buft      = &slow_buft;
    pool.slab_size = 64*1024;
    int32_t si = -1;
    uint8_t * a = pool.alloc(40*1024, si);
    CHECK(a != nullptr && si == 0);
    const int32_t sa = si;
    uint8_t * b = pool.alloc(100*1024, si);
    fprintf(stderr, "  pool: %zu slabs, %.0f KiB, %.0f KiB in use\n", pool.slabs.size(), pool.capacity() / 1024.0, pool.n_used / 1024.0);
    CHECK(b != nullptr);
    CHECK(pool.slabs.size() == 3); // the first, the prefetched one, the 100 KiB one
    pool.free(si, b, 100*1024);
    pool.free(sa, a, 40*1024);
    CHECK(pool.n_used == 0);
    return true;
}

// scenario H (HC3, the gpu-50 wedge through the server's path): a save into a pool with a slow host buffer type and
// slabs smaller than any buffer of the save. The target segment gets a slab of its own, which starts the prefetch of
// another slab; the drafter segment then fits nowhere, the prefetch is joined and its slab is pending: pre-HC3 the
// main thread hung right there (gdb: server_state_pool::alloc <- server_state_pool::alloc <- server_state_buf::alloc
// <- server_prompt_cache::save <- server_slot::prompt_save <- get_available_slot)
static bool scenario_save_during_prefetch(server_prompt_cache &, slot & s, int32_t n_vocab) {
    server_prompt_cache cache(/*limit_size_mib=*/ 64, /*limit_tokens=*/ 0, &slow_buft);
    cache.pool.slab_size = 256;
    const llama_tokens conv = make_conv(n_vocab, 10, 8, 128);
    const int64_t t0 = ggml_time_us();
    CHECK(turn(cache, s, conv, 96));
    fprintf(stderr, "  save of 96 tokens: %zu segments, %.1f KiB, pool %zu slabs (%.0f ms)\n", s.origin->data.segs.size(),
            s.origin->size() / 1024.0, cache.pool.slabs.size(), (ggml_time_us() - t0) / 1e3);
    CHECK(s.origin->data.segs.size() == 1);
    CHECK(s.origin->data.segs[0].main.size() > 256 && s.origin->data.segs[0].drft.size() > 256);
    s.clear();
    // and the entry restores
    server_tokens tokens_new(llama_tokens(conv.begin(), conv.begin() + 96), false);
    CHECK(cache.load(s.prompt, tokens_new, s.tgt, s.dft, nullptr, SEQ, nullptr, &s.origin));
    CHECK(s.prompt.n_tokens() == 96);
    s.clear();
    return true;
}

// scenario I (HC3): the pool may not grow past its cap. A save whose buffers fit no free block evicts the least recently
// used unowned entry and retries; with nothing left to evict the save is skipped (nullptr, a WARN), the cache limit
// stays (the out-of-memory path cuts it to 40%), and the next save after the pressure is gone works
static bool scenario_save_skipped_when_pool_full(server_prompt_cache &, slot & s, int32_t n_vocab) {
    server_prompt_cache cache(/*limit_size_mib=*/ 64, /*limit_tokens=*/ 0, nullptr);
    cache.pool.slab_size = 4096;
    const llama_tokens A = make_conv(n_vocab, 11, 8, 128), B = make_conv(n_vocab, 12, 8, 128), C = make_conv(n_vocab, 13, 8, 128);
    CHECK(turn(cache, s, A, 64)); s.clear();
    const size_t limit0 = cache.limit_size;

    // the pool is capped at what it holds now (a slab prefetched after A's save joins first) and every free byte is taken
    if (cache.pool.th.joinable()) {
        cache.pool.th.join();
    }
    cache.pool.adopt_pending();
    cache.pool.cap_max = cache.pool.capacity();
    std::vector<server_state_buf> filler;
    auto fill = [&]() {
        for (auto & sl : cache.pool.slabs) {
            while (!sl.free_blocks.empty()) {
                const size_t sz = sl.free_blocks.begin()->second;
                filler.emplace_back();
                if (!filler.back().alloc(cache.pool, sz)) {
                    return false;
                }
            }
        }
        return true;
    };
    CHECK(fill());

    // B evicts A (unowned, least recently used) and takes its blocks
    CHECK(turn(cache, s, B, 64));
    fprintf(stderr, "  B saved into a full pool: %zu entries (A %s)\n", cache.states.size(), entry_of(cache, A) ? "kept" : "evicted");
    CHECK(entry_of(cache, A) == nullptr);
    CHECK(entry_of(cache, B) != nullptr);
    s.clear();

    // nothing to evict: C's save is skipped, the server goes on
    cache.states.clear();
    CHECK(fill());
    CHECK(s.extend(C, 64));
    auto * e = cache.save(s.prompt, s.tgt, s.dft, nullptr, SEQ, &s.origin, true);
    fprintf(stderr, "  C into a full pool with nothing to evict: %s, %zu entries, limit %.1f MiB\n", e ? "saved" : "skipped",
            cache.states.size(), cache.limit_size / (1024.0 * 1024.0));
    CHECK(e == nullptr);
    CHECK(s.origin == nullptr);
    CHECK(cache.states.empty());
    CHECK(cache.limit_size == limit0);
    CHECK(!cache.pool.cap_refused);

    // the pressure goes: the same save works
    filler.clear();
    e = cache.save(s.prompt, s.tgt, s.dft, nullptr, SEQ, &s.origin, true);
    CHECK(e != nullptr);
    CHECK(s.origin == e);
    s.clear();
    return true;
}

// scenario J (HC3): the task loop's watchdog numbers. A task that keeps the loop for 1.5 s shows as a stall of more than
// a second with the next task queued behind it; once the loop is back waiting the stall is 0 (/health turns 503 on
// these when the stall passes LLAMA_SRV_WATCHDOG_S)
static bool scenario_queue_watchdog(server_prompt_cache &, slot &, int32_t) {
    server_queue q;
    std::atomic<int> n_done{0};
    q.on_new_task([&](server_task && task, bool) {
        if (task.id == 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        }
        n_done++;
        return true;
    });
    q.on_update_slots([]() {});
    std::thread loop([&]() { q.start_loop(-1); });
    CHECK(q.stall_ms() == 0 || q.stall_ms() < 100);
    server_task t1(SERVER_TASK_TYPE_METRICS); t1.id = 1;
    server_task t2(SERVER_TASK_TYPE_METRICS); t2.id = 2;
    q.post(std::move(t1));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    q.post(std::move(t2));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    const int64_t stall = q.stall_ms();
    const int64_t queued = q.n_queued_nowait();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    const int64_t stall_after = q.stall_ms();
    fprintf(stderr, "  stall during the slow task %lld ms (%lld queued), after %lld ms, %d tasks done\n",
            (long long) stall, (long long) queued, (long long) stall_after, n_done.load());
    q.terminate();
    loop.join();
    CHECK(stall >= 1000);
    CHECK(queued == 1);
    CHECK(stall_after == 0);
    CHECK(n_done == 2);
    return true;
}

// the cells of seq from position keep on go; a recurrent (hybrid) memory cannot cut its state there: the sequence is
// decoded again up to keep instead (what the server gets from a context checkpoint)
static bool truncate_seq(llama_context * ctx, llama_seq_id seq, const llama_tokens & conv, llama_pos keep) {
    if (llama_memory_seq_pos_max(llama_get_memory(ctx), seq) < keep || llama_memory_seq_rm(llama_get_memory(ctx), seq, keep, -1)) {
        return true;
    }
    llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
    return decode_range(ctx, conv, 0, keep, seq);
}

// scenario K (TF1): the slot holds entry E (3 segments, 104 tokens) and its prompt now ends in another tail (a re-rendered
// answer: 90 tokens shared, 200 in all). The delta save is over the size limit and skipped: E must stay whole and restore.
// Pre-fix: the segments past the common prefix were already freed when the limit check skipped the save, E kept 2 of its
// 3 segments under its 104 tokens and the restore failed (CL1 guard) and dropped it.
static bool scenario_over_limit_keeps_origin(server_prompt_cache & cache, slot & s, int32_t n_vocab) {
    cache.states.clear();
    const size_t limit0 = cache.limit_size;
    const llama_tokens conv = make_conv(n_vocab, 14, 8, 200);
    const int N[] = {40, 72, 104};
    for (int i = 0; i < 3; ++i) {
        CHECK(turn(cache, s, conv, N[i]));
        if (i < 2) {
            s.clear();
        }
    }
    auto * e = s.origin;
    CHECK(e != nullptr && e->data.segs.size() == 3);
    const size_t size_e = e->size();

    // the slot's tail changes after 90 tokens
    llama_tokens conv2(conv.begin(), conv.begin() + 90);
    for (int p = 90; p < 200; ++p) {
        conv2.push_back((llama_token) ((13*(uint32_t) p + 17) % (uint32_t) n_vocab));
    }
    CHECK(truncate_seq(s.tgt, SEQ, conv, 90));
    CHECK(truncate_seq(s.dft, SEQ, conv, 90));
    CHECK(s.extend(conv2, 200));

    cache.limit_size = size_e; // the kept part + the new cells do not fit
    auto * r = cache.save(s.prompt, s.tgt, s.dft, nullptr, SEQ, &s.origin, true);
    fprintf(stderr, "  over-limit delta save: %s; entry: %zu segments, %d tokens, owner %d\n", r ? "saved" : "skipped",
            e->data.segs.size(), e->prompt.n_tokens(), e->owner);
    CHECK(r == nullptr);
    CHECK(s.origin == nullptr);
    CHECK(cache.states.size() == 1 && &cache.states.front() == e);
    CHECK(e->owner == -1);
    CHECK(e->data.segs.size() == 3);
    CHECK(e->size() == size_e);
    cache.limit_size = limit0;
    s.clear();

    // the conversation comes back: E restores in full
    server_tokens tokens_new(llama_tokens(conv.begin(), conv.begin() + 104), false);
    const bool loaded = cache.load(s.prompt, tokens_new, s.tgt, s.dft, nullptr, SEQ, nullptr, &s.origin);
    const llama_pos gap = llama_memory_seq_pos_gap(s.tgt, SEQ, 0);
    fprintf(stderr, "  restore after the skipped save: load %d, %d tokens, cells contiguous up to %d\n", (int) loaded, s.prompt.n_tokens(), gap);
    CHECK(loaded);
    CHECK(s.prompt.n_tokens() == 104);
    CHECK(gap >= 104);
    s.clear();
    return true;
}

// scenario L (TF1): the stress. Three slots (seq = slot id, as in the server) share a context pair
static llama_context * g_tgt3 = nullptr;
static llama_context * g_dft3 = nullptr;

struct mslot {
    llama_seq_id seq = 0;
    server_prompt prompt;
    server_prompt_cache_state * origin = nullptr;
    const llama_tokens * conv = nullptr; // the conversation the slot holds (for the step's bookkeeping only)

    void clear() {
        conv = nullptr;
        llama_memory_seq_rm(llama_get_memory(g_tgt3), seq, -1, -1);
        llama_memory_seq_rm(llama_get_memory(g_dft3), seq, -1, -1);
        prompt.clear();
        if (origin != nullptr) {
            origin->owner = -1;
            origin = nullptr;
        }
    }
};

// a fourth sequence restores every entry on the side: a segment's p1 is open-ended, so only a restore shows its cells
static const llama_seq_id SEQ_PROBE = 3;

static bool probe_entry(const server_prompt_cache_state & st, int step) {
    auto * mem = llama_get_memory(g_tgt3);
    llama_memory_seq_rm(mem, SEQ_PROBE, -1, -1);
    const auto & segs = st.data.segs;
    const llama_pos n_e = st.prompt.tokens.pos_next();
    for (size_t i = 0; i < segs.size(); ++i) {
        if (!segs[i].main.empty()) {
            CHECK(llama_state_seq_set_data_range(g_tgt3, segs[i].main.data(), segs[i].main.size(), SEQ_PROBE) == segs[i].main.size());
        }
        const llama_pos next_p0 = i + 1 < segs.size() ? segs[i + 1].p0 : n_e;
        if (segs[i].p1 > next_p0) {
            llama_memory_seq_rm(mem, SEQ_PROBE, next_p0, -1);
        }
    }
    const llama_pos gap = llama_memory_seq_pos_gap(g_tgt3, SEQ_PROBE, 0);
    llama_memory_seq_rm(mem, SEQ_PROBE, -1, -1);
    if (gap >= 0 && gap < n_e) {
        fprintf(stderr, "FAIL step %d: entry of %d tokens (owner %d, %zu segments) restores only up to pos %d\n", step, st.prompt.n_tokens(),
                st.owner, segs.size(), gap);
        return false;
    }
    return true;
}

static uint32_t g_rng = 12345;
static uint32_t rnd(uint32_t n) {
    g_rng = g_rng*1664525u + 1013904223u;
    return (g_rng >> 8) % n;
}

// the invariants of the cache against the slots: what a slot holds is alive, its own, and whole
static bool check_invariants(server_prompt_cache & cache, std::vector<mslot> & slots, int step) {
    auto in_cache = [&](const server_prompt_cache_state * p) {
        for (const auto & st : cache.states) {
            if (&st == p) {
                return true;
            }
        }
        return false;
    };
    for (auto & ms : slots) {
        if (ms.origin != nullptr) {
            if (!in_cache(ms.origin)) {
                fprintf(stderr, "FAIL step %d: slot %d's origin %p was freed\n", step, ms.seq, (void *) ms.origin);
                return false;
            }
            CHECK(ms.origin->owner == ms.seq);
        }
    }
    struct blk { int32_t slab; size_t off, n; };
    std::vector<blk> blks;
    size_t n_used = 0;
    auto add_buf = [&](const server_state_buf & b) -> bool {
        if (b.p == nullptr) {
            CHECK(b.cap == 0 || b.pool == nullptr);
            return true;
        }
        CHECK(b.pool == &cache.pool);
        CHECK(b.slab >= 0 && (size_t) b.slab < cache.pool.slabs.size());
        const auto & sl = cache.pool.slabs[b.slab];
        CHECK(b.p >= sl.base && b.p + server_state_pool::round(b.cap) <= sl.base + sl.cap);
        blks.push_back({b.slab, (size_t) (b.p - sl.base), server_state_pool::round(b.cap)});
        n_used += server_state_pool::round(b.cap);
        return true;
    };
    for (const auto & st : cache.states) {
        // an owned entry belongs to exactly the slot whose origin it is
        if (st.owner >= 0) {
            CHECK((size_t) st.owner < slots.size());
            CHECK(slots[st.owner].origin == &st);
        }
        // the segments tile [0, pos_next) for the target, the drafter parts leave no hole
        const auto & segs = st.data.segs;
        CHECK(!segs.empty());
        CHECK(segs[0].p0 == 0 && segs[0].d0 == 0);
        for (size_t i = 0; i + 1 < segs.size(); ++i) {
            if (!(segs[i].p1 >= segs[i + 1].p0 && segs[i].d1 >= segs[i + 1].d0 && segs[i + 1].d0 <= segs[i + 1].p0)) {
                fprintf(stderr, "FAIL step %d: entry of %d tokens: segment %zu [%d,%d) d[%d,%d) then [%d,%d) d[%d,%d)\n", step,
                        st.prompt.n_tokens(), i, segs[i].p0, segs[i].p1, segs[i].d0, segs[i].d1, segs[i + 1].p0, segs[i + 1].p1, segs[i + 1].d0, segs[i + 1].d1);
                return false;
            }
        }
        if (segs.back().p1 < st.prompt.tokens.pos_next()) {
            fprintf(stderr, "FAIL step %d: entry of %d tokens: segments end at %d (%zu segments)\n", step, st.prompt.n_tokens(), segs.back().p1, segs.size());
            return false;
        }
        for (const auto & seg : segs) {
            CHECK(add_buf(seg.main));
            CHECK(add_buf(seg.drft));
        }
        CHECK(add_buf(st.data.recr));
    }
    // every block in use is one live buffer's, no two overlap
    if (n_used != cache.pool.n_used) {
        fprintf(stderr, "FAIL step %d: the entries hold %zu bytes of the pool, the pool counts %zu in use\n", step, n_used, cache.pool.n_used);
        return false;
    }
    std::sort(blks.begin(), blks.end(), [](const blk & a, const blk & b) { return a.slab != b.slab ? a.slab < b.slab : a.off < b.off; });
    for (size_t i = 0; i + 1 < blks.size(); ++i) {
        if (blks[i].slab == blks[i + 1].slab) {
            CHECK(blks[i].off + blks[i].n <= blks[i + 1].off);
        }
    }
    // and no free block overlaps a used one
    for (size_t si = 0; si < cache.pool.slabs.size(); ++si) {
        for (const auto & [off, n] : cache.pool.slabs[si].free_blocks) {
            for (const auto & b : blks) {
                if (b.slab == (int32_t) si) {
                    CHECK(off + n <= b.off || b.off + b.n <= off);
                }
            }
        }
    }
    return true;
}

static bool scenario_stress(server_prompt_cache &, slot &, int32_t n_vocab) {
    const int N_SLOTS = 3, N_CONV = 7, N_MAX = 230, STEPS = 400;
    server_prompt_cache cache(/*limit_size_mib=*/ 64, /*limit_tokens=*/ 0, nullptr);
    cache.pool.slab_size = 16*1024;

    std::vector<mslot> slots(N_SLOTS);
    for (int i = 0; i < N_SLOTS; ++i) {
        slots[i].seq = i;
        slots[i].clear();
    }
    std::vector<llama_tokens> convs(N_CONV);
    std::vector<int> len(N_CONV, 0);
    int next_id = 100;
    for (int c = 0; c < N_CONV; ++c) {
        convs[c] = make_conv(n_vocab, next_id++, 24, N_MAX);
    }

    int n_over = 0, n_over_stale = 0, n_stale = 0;
    int n_saves = 0, n_saved = 0, n_loads = 0, n_restored = 0, n_failed_loads = 0, n_idle = 0, n_clears = 0, n_back = 0;
    size_t max_entries = 0;
    bool limited = false;
    size_t limit_lo = 0, limit_hi = 0;
    for (int step = 0; step < STEPS; ++step) {
        // phases where a long conversation alone is over the limit: its saves are skipped (the entry it extends stays)
        if (limited) {
            // (low: three quarters of the biggest entry so far; the recurrent state makes the size per token misleading)
            if (step % 60 == 30) {
                size_t max_e = 0;
                for (const auto & st : cache.states) {
                    max_e = std::max(max_e, st.size());
                }
                limit_lo = std::max(limit_lo, max_e * 3 / 4);
            }
            cache.limit_size = step % 60 >= 30 ? limit_lo : limit_hi;
        }
        // once the first entries show the size per token: room for about two conversations, the pool capped near it
        if (!limited && cache.n_tokens() > 0 && step > 6) {
            const size_t per_tok = std::max<size_t>(1, cache.size() / cache.n_tokens());
            cache.limit_size = per_tok * 2 * N_MAX;
            cache.pool.cap_max = std::max<size_t>(cache.pool.capacity(), cache.limit_size + 2*cache.pool.slab_size);
            limited = true;
            limit_hi = cache.limit_size;
            limit_lo = 0;
            fprintf(stderr, "  step %d: limit %.1f KiB (%zu B/token), pool cap %.1f KiB\n", step, cache.limit_size / 1024.0, per_tok,
                    cache.pool.cap_max / 1024.0);
        }

        const int op = (int) rnd(10);
        mslot * pms = &slots[rnd(N_SLOTS)];
        if (op == 0) {
            mslot & ms = *pms;
            // prompt_clear (a child or stateless task, the unified-KV purge)
            ms.clear();
            n_clears++;
        } else if (op == 1) {
            mslot & ms = *pms;
            // cache_idle_slots: an idle slot saves (checkpoints copied), the slot keeps its prompt and the entry
            if (ms.prompt.n_tokens() > 0) {
                n_saves++;
                if (cache.save(ms.prompt, g_tgt3, g_dft3, nullptr, ms.seq, &ms.origin, false) != nullptr) {
                    n_saved++;
                    CHECK(ms.origin != nullptr);
                }
                cache.update();
                n_idle++;
            }
        } else {
            // a request: conversation c grows by a turn; sometimes its last answer comes back re-rendered (a new tail)
            const int c = (int) rnd(N_CONV);
            // the server prefers the slot that holds the conversation (LCP similarity): its next save is a delta
            for (auto & o : slots) {
                if (o.conv == &convs[c] && o.prompt.n_tokens() > 0 && rnd(3) != 0) {
                    pms = &o;
                }
            }
            mslot & ms = *pms;
            if (len[c] + 32 > N_MAX) {
                convs[c] = make_conv(n_vocab, next_id++, 24, N_MAX);
                len[c] = 0;
            }
            if (len[c] > 40 && rnd(2) == 0) {
                for (int p = len[c] - 1 - (int) rnd(12); p < N_MAX; ++p) {
                    convs[c][p] = (llama_token) ((5*(uint32_t) p + 31*(uint32_t) step + 9) % (uint32_t) n_vocab);
                }
            }
            const int n = std::min(N_MAX, len[c] + 8 + (int) rnd(24));
            len[c] = n;
            const llama_tokens & conv = convs[c];
            server_tokens tokens_new(llama_tokens(conv.begin(), conv.begin() + n), false);

            // the server's slot switch (get_available_slot): find_better's pointer is used right away only
            if (const auto * better = cache.find_better(ms.prompt, tokens_new, ms.seq, 16)) {
                CHECK(better->owner < 0 || better->owner == ms.seq);
            }
            server_prompt_cache_state * saved = nullptr;
            if (ms.prompt.n_tokens() > 0) {
                n_saves++;
                auto * held = ms.origin;
                const bool stale = held != nullptr && (int) held->prompt.tokens.get_common_prefix(ms.prompt.tokens) < held->prompt.n_tokens();
                const size_t held_segs = held != nullptr ? held->data.segs.size() : 0;
                saved = cache.save(ms.prompt, g_tgt3, g_dft3, nullptr, ms.seq, &ms.origin, true);
                n_saved += saved != nullptr;
                n_over += saved == nullptr && held != nullptr && cache.limit_size == limit_lo;
                n_over_stale += saved == nullptr && stale && held_segs > 1;
                n_stale += stale;
                CHECK(saved == ms.origin);
                CHECK(check_invariants(cache, slots, step));
            }
            n_loads++;
            const int n_before = ms.prompt.n_tokens();
            if (!cache.load(ms.prompt, tokens_new, g_tgt3, g_dft3, nullptr, ms.seq, saved, &ms.origin)) {
                n_failed_loads++;
                ms.clear();
            } else if (ms.origin != nullptr && ms.origin != saved) {
                n_restored++;
                // a restored prompt has every cell
                CHECK(llama_memory_seq_pos_gap(g_tgt3, ms.seq, 0) >= ms.prompt.n_tokens());
            }
            cache.update();
            CHECK(check_invariants(cache, slots, step));

            // prompt reuse: the common prefix stays; the checkpoints past it go back to the origin entry or away
            const int lcp = (int) ms.prompt.tokens.get_common_prefix(tokens_new);
            CHECK(truncate_seq(g_tgt3, ms.seq, conv, lcp));
            CHECK(truncate_seq(g_dft3, ms.seq, conv, std::min(lcp, n - LAG)));
            ms.prompt.tokens.keep_first(lcp);
            for (auto it = ms.prompt.checkpoints.begin(); it != ms.prompt.checkpoints.end();) {
                if (it->n_tokens > lcp) {
                    auto * origin = ms.origin;
                    if (origin != nullptr && it->n_tokens <= origin->prompt.n_tokens() &&
                            (origin->prompt.checkpoints.empty() || origin->prompt.checkpoints.back().n_tokens < it->n_tokens)) {
                        auto next = std::next(it);
                        origin->prompt.checkpoints.splice(origin->prompt.checkpoints.end(), ms.prompt.checkpoints, it);
                        it = next;
                        n_back++;
                    } else {
                        it = ms.prompt.checkpoints.erase(it);
                    }
                } else {
                    ++it;
                }
            }

            // the prefill and a few checkpoints on the way (one per ~32 tokens, the newest five kept)
            const llama_pos t0 = llama_memory_seq_pos_max(llama_get_memory(g_tgt3), ms.seq) + 1;
            const llama_pos d0 = llama_memory_seq_pos_max(llama_get_memory(g_dft3), ms.seq) + 1;
            if (t0 != lcp) {
                fprintf(stderr, "FAIL step %d slot %d: cells end at %d, prompt %d tokens, lcp %d (restored %d, origin %p saved %p, n_before %d)\n",
                        step, ms.seq, t0, ms.prompt.n_tokens(), lcp, (int) (ms.origin != saved), (void *) ms.origin, (void *) saved, n_before);
                return false;
            }
            CHECK(decode_range(g_tgt3, conv, t0, n, ms.seq));
            CHECK(decode_range(g_dft3, conv, d0, n - LAG, ms.seq));
            ms.prompt.tokens = server_tokens(llama_tokens(conv.begin(), conv.begin() + n), false);
            for (int k = lcp / 32 + 1; k * 32 <= n; ++k) {
                ms.prompt.checkpoints.push_back(fake_ckpt(k * 32, 512 + 64 * (size_t) rnd(16)));
            }
            while (ms.prompt.checkpoints.size() > 5) {
                ms.prompt.checkpoints.pop_front();
            }
            ms.conv = &conv;
        }
        CHECK(check_invariants(cache, slots, step));
        for (const auto & st : cache.states) {
            CHECK(probe_entry(st, step));
        }
        max_entries = std::max(max_entries, cache.states.size());
    }
    fprintf(stderr, "  %d saves of a held entry skipped in the low-limit phases (%d with a stale tail; %d delta saves onto a stale tail)\n",
            n_over, n_over_stale, n_stale);
    fprintf(stderr, "  %d steps: %d saves (%d kept), %d loads (%d restores, %d failed), %d idle saves, %d clears, %d checkpoints back; "
            "max %zu entries, end %zu entries %.1f KiB (limit %.1f KiB), pool %zu slabs %.1f KiB\n",
            STEPS, n_saves, n_saved, n_loads, n_restored, n_failed_loads, n_idle, n_clears, n_back, max_entries, cache.states.size(),
            cache.size() / 1024.0, cache.limit_size / 1024.0, cache.pool.slabs.size(), cache.pool.capacity() / 1024.0);
    // a restore never fails here: every entry tiles its prompt
    CHECK(n_failed_loads == 0);
    CHECK(n_restored > 20);
    for (auto & ms : slots) {
        ms.clear();
    }
    return true;
}

// the watchdog: a scenario that does not return within this deadline hung (the process exits with 2)
static std::atomic<int64_t> g_deadline_us{0};
static std::atomic<const char *> g_scenario{""};

int main(int argc, char ** argv) {
    std::string model_path;
    std::string only; // --only CDGH: run only the table scenarios with these letters (A and B always run)
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::string(argv[i]) == "--only" && i + 1 < argc) {
            only = argv[++i];
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m <model.gguf>\n", argv[0]);
        return 1;
    }

    // the save rules without the short-tail re-save (seg_min): the drafter-hole rule decides alone
    setenv("LLAMA_CACHE_SEG_MIN", "0", 0);
    // no pinned slabs up front
    setenv("LLAMA_CACHE_PREALLOC_MIB", "0", 0);

    llama_backend_init();
    common_log_set_verbosity_thold(0);

    std::thread([]() {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const int64_t d = g_deadline_us.load();
            if (d > 0 && ggml_time_us() > d) {
                fprintf(stderr, "FAIL: scenario %s did not return within 60 s (hung)\n", g_scenario.load());
                fflush(stderr);
                _exit(2);
            }
        }
    }).detach();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx     = 256;
    cparams.n_batch   = 256;
    cparams.n_ubatch  = 256;
    cparams.n_seq_max = 1;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    llama_context * ctx_tgt = llama_init_from_model(model, cparams);
    llama_context * ctx_dft = llama_init_from_model(model, cparams);
    if (ctx_tgt == nullptr || ctx_dft == nullptr) {
        fprintf(stderr, "failed to create the contexts\n");
        return 1;
    }

    // scenario L: three slots on one context pair (seq = slot id)
    {
        auto cp3 = cparams;
        cp3.n_ctx     = 4*1024; // room for every slot's cells and the ghosts of the saved ones
        cp3.n_seq_max = 4; // the slots' 3 + the probe
        g_tgt3 = llama_init_from_model(model, cp3);
        g_dft3 = llama_init_from_model(model, cp3);
        if (g_tgt3 == nullptr || g_dft3 == nullptr) {
            fprintf(stderr, "failed to create the 3-sequence contexts\n");
            return 1;
        }
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_tokens conv;
    for (llama_pos p = 0; p < 128; ++p) {
        conv.push_back(conv_token(n_vocab, p));
    }

    server_prompt_cache cache(/*limit_size_mib=*/ 64, /*limit_tokens=*/ 0, nullptr);

    slot s;
    s.tgt = ctx_tgt;
    s.dft = ctx_dft;

    int n_fail = 0;

    fprintf(stderr, "scenario A: delta saves with a lagging drafter keep every cell\n");
    if (scenario_save_keeps_every_cell(cache, s, conv)) {
        fprintf(stderr, "scenario A: OK\n");
    } else {
        fprintf(stderr, "scenario A: FAILED\n");
        n_fail++;
    }
    s.clear();

    fprintf(stderr, "scenario B: an entry with a hole is not restored\n");
    if (scenario_load_refuses_a_hole(cache, s, conv)) {
        fprintf(stderr, "scenario B: OK\n");
    } else {
        fprintf(stderr, "scenario B: FAILED\n");
        n_fail++;
    }
    s.clear();

    struct { const char * name; bool (*fn)(server_prompt_cache &, slot &, int32_t); } hc1[] = {
        {"C: least recently used entry is evicted", scenario_lru},
        {"D: older checkpoints are shed before an entry is evicted", scenario_trim},
        {"E: find_better names the returning conversation's entry", scenario_find_better},
        {"F: a delta save keeps the earlier segments (HC2)", scenario_delta_keeps_segments},
        {"G: a pool allocation while the slab prefetch runs does not hang (HC3)", scenario_pool_alloc_during_prefetch},
        {"H: a save that fits no free block while the prefetch runs does not hang (HC3)", scenario_save_during_prefetch},
        {"I: a save the capped pool cannot hold is skipped, the next works (HC3)", scenario_save_skipped_when_pool_full},
        {"J: the task loop watchdog sees a stuck task (HC3)", scenario_queue_watchdog},
        {"K: an over-limit delta save leaves the slot's entry whole (TF1)", scenario_over_limit_keeps_origin},
        {"L: three slots under a tiny limit and a capped pool: nothing in use is freed (TF1)", scenario_stress},
    };
    for (const auto & sc : hc1) {
        if (!only.empty() && only.find(sc.name[0]) == std::string::npos) {
            continue;
        }
        fprintf(stderr, "scenario %s\n", sc.name);
        g_scenario = sc.name;
        g_deadline_us = ggml_time_us() + 60*1000*1000;
        const bool ok = sc.fn(cache, s, n_vocab);
        g_deadline_us = 0;
        if (ok) {
            fprintf(stderr, "  OK\n");
        } else {
            fprintf(stderr, "  FAILED\n");
            n_fail++;
        }
        s.clear();
    }

    cache.states.clear();
    llama_free(g_dft3);
    llama_free(g_tgt3);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model);
    llama_backend_free();

    return n_fail == 0 ? 0 : 1;
}
