// the server's loop guard (tools/server/server-loop-guard.h): synthetic loops trigger it, varied text does not
//
// test-loop-guard                          synthetic cases
// test-loop-guard -v VOCAB.gguf FILE...    false-positive scan: each file is one reply, tokenized with the vocab;
//                                          fails if the guard (server defaults) triggers on any of them

#include "server-loop-guard.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

// the token index at which the guard triggers, -1 = never
static int run_guard(const std::vector<llama_token> & toks, int32_t * period = nullptr, int span = 512, int reps = 8, int pmax = 512) {
    server_loop_guard g;
    g.init(span, reps, pmax);
    for (size_t i = 0; i < toks.size(); i++) {
        const int32_t p = g.push(toks[i]);
        if (p > 0) {
            if (period) {
                *period = p;
            }
            return (int) i;
        }
    }
    return -1;
}

static uint32_t rng_state = 12345;
static llama_token rnd(uint32_t n) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (llama_token) ((rng_state >> 8) % n);
}

static void test_synthetic() {
    // varied text: 20000 tokens from a 30000-token vocabulary
    std::vector<llama_token> text;
    for (int i = 0; i < 20000; i++) {
        text.push_back(rnd(30000));
    }
    CHECK(run_guard(text) == -1);

    // "1 * 1 * 1 * ..." after some text: period 4 ("1", " *", " ", ...), cut once 512 tokens repeat
    {
        std::vector<llama_token> t(text.begin(), text.begin() + 1000);
        for (int i = 0; i < 4000; i++) {
            t.push_back(100 + i % 4);
        }
        int32_t p = 0;
        const int at = run_guard(t, &p);
        CHECK(p == 4);
        CHECK(at >= 1000 + 512 && at <= 1000 + 520);
    }

    // one token repeated (period 1)
    {
        std::vector<llama_token> t(text.begin(), text.begin() + 50);
        t.insert(t.end(), 600, 7);
        int32_t p = 0;
        CHECK(run_guard(t, &p) >= 0 && p == 1);
    }

    // a 300-token paragraph repeated: triggers after 8 repetitions (7 x 300 matching tokens)
    {
        std::vector<llama_token> para(text.begin() + 5000, text.begin() + 5300);
        std::vector<llama_token> t(text.begin(), text.begin() + 200);
        for (int r = 0; r < 12; r++) {
            t.insert(t.end(), para.begin(), para.end());
        }
        int32_t p = 0;
        const int at = run_guard(t, &p);
        CHECK(p == 300);
        CHECK(at == 200 + 8 * 300 - 1);
    }

    // a period longer than period_max is not seen (the limit of the detector)
    {
        std::vector<llama_token> para(text.begin() + 5000, text.begin() + 5600);
        std::vector<llama_token> t;
        for (int r = 0; r < 10; r++) {
            t.insert(t.end(), para.begin(), para.end());
        }
        CHECK(run_guard(t) == -1);
    }

    // a repetition shorter than the span is normal text (a table separator, a short refrain)
    {
        std::vector<llama_token> t(text.begin(), text.begin() + 1000);
        for (int i = 0; i < 400; i++) {
            t.push_back(200 + i % 2);
        }
        t.insert(t.end(), text.begin() + 2000, text.begin() + 3000);
        CHECK(run_guard(t) == -1);
    }

    // code with a counter: "a[i] = i;\n" for i = 0..2999, the number tokens differ on every line
    {
        std::vector<llama_token> t;
        for (int i = 0; i < 3000; i++) {
            const llama_token num = 1000 + i;
            for (llama_token tok : { 10, 11, num, 12, 13, num, 14, 15 }) {
                t.push_back(tok);
            }
        }
        CHECK(run_guard(t) == -1);
    }

    // reset() forgets the history: the same loop needs a full span again
    {
        server_loop_guard g;
        g.init(512, 8, 512);
        int first = -1;
        int second = -1;
        for (int i = 0; i < 3000; i++) {
            if (g.push(100 + i % 4) > 0) {
                if (first < 0) {
                    first = i;
                    g.reset();
                } else if (second < 0) {
                    second = i;
                }
            }
        }
        CHECK(first > 0 && second - first >= 512);
    }

    // period_max 0 = off
    {
        std::vector<llama_token> t(2000, 5);
        CHECK(run_guard(t, nullptr, 512, 8, 0) == -1);
    }
}

static int scan_files(const char * vocab_path, int argc, char ** argv, int first) {
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(vocab_path, mparams);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", vocab_path);
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    size_t n_tok = 0;
    int    n_hit = 0;
    int    n_files = 0;
    for (int i = first; i < argc; i++) {
        std::ifstream f(argv[i], std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string text = ss.str();
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true);
        if (n < 0) {
            toks.resize(-n);
            n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true);
        }
        toks.resize(std::max(n, 0));
        n_tok += toks.size();
        n_files++;
        int32_t p = 0;
        const int at = run_guard(toks, &p);
        if (at >= 0) {
            n_hit++;
            printf("TRIGGER %s: period %d at token %d of %zu\n", argv[i], p, at, toks.size());
        }
    }
    printf("scanned %d replies, %zu tokens, %d triggers\n", n_files, n_tok, n_hit);
    llama_model_free(model);
    llama_backend_free();
    return n_hit == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc >= 3 && strcmp(argv[1], "-v") == 0) {
        return scan_files(argv[2], argc, argv, 3);
    }
    test_synthetic();
    if (n_fail) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }
    printf("loop guard: all synthetic cases passed\n");
    return 0;
}
