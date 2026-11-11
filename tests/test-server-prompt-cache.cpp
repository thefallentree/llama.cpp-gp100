// regression test for the server's host prompt cache (server_prompt_cache, tools/server/server-task.cpp): a restored
// conversation must have every cell of its prompt in the attention cache (CL1, cross-session leak, 2026-11-11)
//
// - delta save: a drafter context whose cells lag the target (the MTP drafter) must not make save() drop the kept
//   segments after the first one and start the new chunk at the common prefix (the dropped cells were in no segment)
// - load guard: an entry whose segments do not cover its prompt is dropped instead of restored
//
// CPU only, a tiny generated model (test-generate-models): test-server-prompt-cache -m <model.gguf>

#include "server-task.h"
#include "llama.h"
#include "log.h"
#include "../src/llama-ext.h" // llama_memory_seq_pos_gap

#include <cstdio>
#include <cstdlib>
#include <string>
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

int main(int argc, char ** argv) {
    std::string model_path;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-m" && i + 1 < argc) {
            model_path = argv[++i];
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

    cache.states.clear();
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model);
    llama_backend_free();

    return n_fail == 0 ? 0 : 1;
}
