// llama_tokenize_with_literals(): no special token, not even a user-defined one, is matched inside the literal
// ranges, while the text runs keep their whole-text tokenization across the range borders.
// usage: test-tokenize-literal <vocab.gguf>...
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

static bool is_special(const llama_vocab * vocab, llama_token id) {
    return llama_vocab_get_attr(vocab, id) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED);
}

static int count_special(const llama_vocab * vocab, const std::vector<llama_token> & ids) {
    int n = 0;
    for (auto id : ids) {
        n += is_special(vocab, id);
    }
    return n;
}

static std::string detok(const llama_vocab * vocab, const std::vector<llama_token> & ids) {
    std::string out;
    for (auto id : ids) {
        out += common_token_to_piece(vocab, id, true);
    }
    return out;
}

static void test_vocab(const char * path) {
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", path);
        n_fail++;
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // the special and user-defined tokens of the vocab with a plain text
    std::vector<std::pair<std::string, llama_token>> specials;
    for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); id++) {
        if (!is_special(vocab, id)) {
            continue;
        }
        std::string text = common_token_to_piece(vocab, id, true);
        if (text.size() < 3 || text.find('[') == 0) {
            continue; // skip single bytes and [PAD..] fillers
        }
        // only tokens that match as a whole in a plain tokenization (no strip attributes around them)
        if (llama_vocab_get_attr(vocab, id) & (LLAMA_TOKEN_ATTR_LSTRIP | LLAMA_TOKEN_ATTR_RSTRIP)) {
            continue;
        }
        auto ids = common_tokenize(vocab, "x" + text + "y", false, true);
        bool whole = false;
        for (auto t : ids) {
            whole |= t == id;
        }
        if (whole) {
            specials.emplace_back(text, id);
        }
    }
    int n_user_defined = 0;
    for (const auto & [text, id] : specials) {
        n_user_defined += (llama_vocab_get_attr(vocab, id) & LLAMA_TOKEN_ATTR_USER_DEFINED) != 0;
    }
    printf("%s: %zu special texts (%d user-defined)\n", path, specials.size(), n_user_defined);
    CHECK(!specials.empty(), "%s has no special tokens to test with", path);

    int n_ud_matched = 0;
    for (const auto & [text, id] : specials) {
        const std::string tmpl_a = "<<A>>\n";
        const std::string input  = "file: " + text + " end " + text + "\n";
        const std::string tmpl_b = text + "\n";
        const std::string all    = tmpl_a + input + tmpl_b;
        const std::vector<std::pair<size_t, size_t>> literal = { { tmpl_a.size(), tmpl_a.size() + input.size() } };

        // today: the special token is matched in the input
        auto today = common_tokenize(vocab, all, false, true);
        CHECK(count_special(vocab, today) >= 3, "%s: expected the input text to be special today", text.c_str());

        // literal: only the template's own occurrence is the special token
        auto lit = common_tokenize_with_literals(vocab, all, literal, false, true);
        int n_id = 0;
        for (auto t : lit) {
            n_id += t == id;
        }
        CHECK(n_id == 1, "%s: %d special ids with the input as literal, expected 1", text.c_str(), n_id);
        CHECK(detok(vocab, lit) == all, "%s: the tokens do not give back the text", text.c_str());

        // parse_special == false still matches user-defined tokens, the literal range never does
        auto ps_false = common_tokenize(vocab, input, false, false);
        auto ps_lit   = common_tokenize_with_literals(vocab, input, { { 0, input.size() } }, false, false);
        CHECK(count_special(vocab, ps_lit) == 0, "%s: a special id inside a literal range", text.c_str());
        if ((llama_vocab_get_attr(vocab, id) & LLAMA_TOKEN_ATTR_USER_DEFINED) && count_special(vocab, ps_false) > 0) {
            n_ud_matched++; // what the literal ranges add over parse_special == false
        }

        // an occurrence straddling the range border is text too
        {
            const size_t cut = text.size() / 2;
            const std::string s = "a" + text + "b";
            auto t = common_tokenize_with_literals(vocab, s, { { 1 + cut, s.size() } }, false, true);
            CHECK(count_special(vocab, t) == 0, "%s: an occurrence straddling the literal border was matched", text.c_str());
        }
    }

    printf("  %d user-defined tokens are matched by llama_tokenize(parse_special=false), none inside a literal range\n", n_ud_matched);

    // without special texts, the literal ranges change nothing: the text runs are tokenized as a whole
    {
        const std::string tmpl_a = "<|x|>user\n<parameter=";
        const std::string input  = "hello world, the parameter=path is here\n  indented: 123456 \xe4\xbd\xa0\xe5\xa5\xbd";
        const std::string tmpl_b = "\n\nmore template text";
        const std::string all    = tmpl_a + input + tmpl_b;
        auto whole = common_tokenize(vocab, all, true, true);
        auto lit   = common_tokenize_with_literals(vocab, all, { { tmpl_a.size(), tmpl_a.size() + input.size() } }, true, true);
        CHECK(whole == lit, "literal ranges without special texts changed the tokens");
        // tokenizing the parts on their own does change them at the borders (the reason for the ranges)
        auto parts = common_tokenize(vocab, tmpl_a, true, true);
        for (const auto & s : { input, tmpl_b }) {
            auto t = common_tokenize(vocab, s, false, true);
            parts.insert(parts.end(), t.begin(), t.end());
        }
        printf("  part-wise tokenization %s the whole-text tokens (%zu vs %zu tokens)\n", parts == whole ? "matches" : "differs from", parts.size(), whole.size());
    }

    // invalid ranges are refused
    {
        llama_token buf[64];
        const char * s = "hello";
        int32_t r1[] = { 3, 2 };
        int32_t r2[] = { 0, 9 };
        int32_t r3[] = { 2, 4, 1, 3 };
        CHECK(llama_tokenize_with_literals(vocab, s, 5, r1, 1, buf, 64, false, true) == INT32_MIN, "reversed range accepted");
        CHECK(llama_tokenize_with_literals(vocab, s, 5, r2, 1, buf, 64, false, true) == INT32_MIN, "range past the text accepted");
        CHECK(llama_tokenize_with_literals(vocab, s, 5, r3, 2, buf, 64, false, true) == INT32_MIN, "overlapping ranges accepted");
    }

    llama_model_free(model);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s vocab.gguf...\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);
    for (int i = 1; i < argc; i++) {
        test_vocab(argv[i]);
    }
    llama_backend_free();
    if (n_fail > 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
