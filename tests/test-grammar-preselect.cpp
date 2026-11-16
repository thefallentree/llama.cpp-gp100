// GS1: the grammar over the backend's pre-selected candidates (common_sampler_sample_preselected) against the full
// row: over random logits and a tool-call grammar, the token is always grammar-valid, equals the full-row path (greedy
// and top-k/temperature sampling with the same seed), and the whole-row fallback runs when too few candidates are valid.
// usage: test-grammar-preselect <vocab.gguf>
#include "llama.h"
#include "common.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static const char * TOOL_CALL_GRAMMAR = R"(
root ::= "<tool_call>" ws "{" ws "\"name\"" ws ":" ws name ws "," ws "\"arguments\"" ws ":" ws args ws "}" ws "</tool_call>"
name ::= "\"" [a-z_]+ "\""
args ::= "{" ws ( kv ( ws "," ws kv )* )? ws "}"
kv ::= "\"" [a-z_]+ "\"" ws ":" ws value
value ::= "\"" [a-zA-Z0-9 _./-]* "\"" | [0-9]+ | "true" | "false"
ws ::= [ \t\n]*
)";

struct cfg {
    const char * name;
    float temp;
    int   top_k;
    float top_p;
};

static std::vector<llama_token_data> full_row(const std::vector<float> & logits) {
    std::vector<llama_token_data> cur(logits.size());
    for (size_t t = 0; t < logits.size(); ++t) {
        cur[t] = { (llama_token) t, logits[t], 0.0f };
    }
    return cur;
}

static std::vector<llama_token_data> top_k_candidates(const std::vector<float> & logits, int k) {
    std::vector<llama_token_data> cur = full_row(logits);
    std::partial_sort(cur.begin(), cur.begin() + k, cur.end(), [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
    cur.resize(k);
    return cur;
}

// the grammar's validity mask of the whole row at the sampler's current state (one application)
static std::vector<char> valid_mask(struct common_sampler * gsmpl, int n_vocab) {
    std::vector<llama_token_data> cur(n_vocab);
    for (int t = 0; t < n_vocab; ++t) {
        cur[t] = { (llama_token) t, 0.0f, 0.0f };
    }
    llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
    common_sampler_apply_grammar(gsmpl, &arr);
    std::vector<char> valid(n_vocab, 0);
    for (size_t i = 0; i < arr.size; ++i) {
        valid[arr.data[i].id] = arr.data[i].logit != -INFINITY ? 1 : 0;
    }
    return valid;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <vocab.gguf>\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const cfg cfgs[] = {
        { "greedy", 0.0f, 40, 0.95f },
        { "top_k20_temp0.7", 0.7f, 20, 0.8f },
        { "greedy_budget_nogrammar", 0.0f, 40, 0.95f },   // a reasoning budget (think tags) without a grammar: the budget sampler alone
    };
    int n_fail = 0, n_fallback_total = 0;
    for (const auto & c : cfgs) {
        common_params_sampling params;
        const bool budget_only = std::string(c.name) == "greedy_budget_nogrammar";
        if (budget_only) {
            params.reasoning_budget_start  = common_tokenize(vocab, "<think>", false, true);
            params.reasoning_budget_end    = { common_tokenize(vocab, "</think>", false, true) };
            params.reasoning_budget_forced = params.reasoning_budget_end[0];
            params.reasoning_budget_tokens = 0;
        } else {
            params.grammar          = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, TOOL_CALL_GRAMMAR);
        }
        params.backend_sampling = true;
        params.temp             = c.temp;
        params.top_k            = c.top_k;
        params.top_p            = c.top_p;
        params.min_p            = 0.0f;
        params.seed             = 42;
        params.samplers         = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TEMPERATURE };
        common_params_sampling params_ref = params;
        params_ref.backend_sampling = false;

        struct common_sampler * gs  = common_sampler_init(model, params);
        struct common_sampler * ref = common_sampler_init(model, params_ref);
        struct common_sampler * chk = common_sampler_init(model, params_ref);   // the validity probe (its own grammar state)
        int32_t needed = 0;
        const bool pre = common_sampler_grammar_preselect(gs, &needed);
        const int32_t k_pre = llama_sampler_chain_get_preselect_k(common_sampler_get(gs));
        printf("%s: grammar preselect %d, k_pre %d, needed %d\n", c.name, (int) pre, k_pre, needed);
        if (!pre || k_pre <= 0) {
            fprintf(stderr, "FAIL %s: the grammar preselect is off\n", c.name);
            n_fail++;
            common_sampler_free(gs); common_sampler_free(ref); common_sampler_free(chk);
            continue;
        }

        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        int n_steps = 0, n_fallback = 0, n_equal = 0, n_valid = 0;
        for (int seq = 0; seq < 12; ++seq) {
            common_sampler_reset(gs); common_sampler_reset(ref); common_sampler_reset(chk);
            for (int step = 0; step < 24; ++step) {
                std::vector<float> logits(n_vocab);
                for (int t = 0; t < n_vocab; ++t) {
                    logits[t] = nd(rng);
                }
                // a few spikes, so that the top of the row is sharp
                for (int s = 0; s < 8; ++s) {
                    logits[rng() % n_vocab] += 6.0f + (rng() % 5);
                }
                const int kind = step % 3;   // 0: plain random, 1: valid tokens spiked (the sharp rows of a real model), 2: forced fallback
                const std::vector<char> valid = valid_mask(chk, n_vocab);
                if (kind == 1) {
                    int placed = 0;
                    for (int t = 0; t < n_vocab && placed < 3; ++t) {
                        if (valid[t]) {
                            logits[t] = 30.0f + (float) placed * 0.5f;
                            placed++;
                        }
                    }
                }
                if (kind == 2) {
                    // the top k_pre logits go to tokens the grammar rejects at this state
                    int placed = 0;
                    for (int t = 0; t < n_vocab && placed < k_pre; ++t) {
                        if (!valid[t]) {
                            logits[t] = 100.0f + (float) placed * 0.01f;
                            placed++;
                        }
                    }
                }
                // the reference: the whole row under the grammar, then the chain
                common_sampler_set_candidates(ref, full_row(logits));
                const llama_token id_ref = common_sampler_sample_preselected(ref, nullptr, [&]() {});
                // the path under test: the top-k_pre candidates, the raw row as the fallback
                common_sampler_set_candidates(gs, top_k_candidates(logits, k_pre));
                bool fallback = false;
                const llama_token id = common_sampler_sample_preselected(gs, &fallback, [&]() { common_sampler_set_candidates(gs, full_row(logits)); });
                const bool is_valid = id >= 0 && id < n_vocab && valid[id];
                n_steps++; n_fallback += fallback ? 1 : 0; n_equal += id == id_ref ? 1 : 0; n_valid += is_valid ? 1 : 0;
                // greedy and the fallback are exact; a sampled row that ignored a far tail (kind 1 without a fallback) may differ
                const bool must_equal = c.temp == 0.0f || fallback || kind != 1;
                if (!is_valid || (must_equal && id != id_ref)) {
                    fprintf(stderr, "FAIL %s seq %d step %d kind %d: id %d (valid %d, fallback %d) ref %d\n", c.name, seq, step, kind, id, (int) is_valid, (int) fallback, id_ref);
                    n_fail++;
                }
                if (kind == 2 && !fallback) {
                    fprintf(stderr, "FAIL %s seq %d step %d: the forced fallback did not trigger\n", c.name, seq, step);
                    n_fail++;
                }
                common_sampler_accept(gs, id, true);
                common_sampler_accept(ref, id, true);
                common_sampler_accept(chk, id, true);
                if (llama_vocab_is_eog(vocab, id)) {
                    break;
                }
            }
        }
        printf("%s: %d steps, %d equal to the full row, %d valid, %d fallbacks\n", c.name, n_steps, n_equal, n_valid, n_fallback);
        n_fallback_total += n_fallback;
        common_sampler_free(gs); common_sampler_free(ref); common_sampler_free(chk);
    }
    llama_model_free(model);
    llama_backend_free();
    printf("%s (%d failures, %d fallbacks)\n", n_fail == 0 ? "OK" : "FAILED", n_fail, n_fallback_total);
    return n_fail == 0 ? 0 : 1;
}
