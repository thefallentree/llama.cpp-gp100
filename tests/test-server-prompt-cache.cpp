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

static bool decode_range(llama_context * ctx, const llama_tokens & conv, llama_pos p0, llama_pos p1) {
    if (p0 >= p1) {
        return true;
    }
    llama_batch batch = llama_batch_init(p1 - p0, 0, 1);
    for (llama_pos p = p0; p < p1; ++p) {
        batch.token   [batch.n_tokens] = conv[p];
        batch.pos     [batch.n_tokens] = p;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id  [batch.n_tokens][0] = SEQ;
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
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model);
    llama_backend_free();

    return n_fail == 0 ? 0 : 1;
}
