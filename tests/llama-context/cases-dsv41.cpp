// DeepSeek-V4.1 cases

#include "test-case.h"

#include "test-log.h"
#include "test-util.h"

#include "../../src/llama-ext.h" // llama_set_prompt_end

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// The exact long-prompt layer skip (LLAMA_DSV41_LAYER_SKIP, llama_set_prompt_end): the ubatches wholly before the last
// tail rows of a prompt run the layers up to the last compressed-stream source and nothing past it, and the logits of
// the last row and of the tokens decoded after it are the ones of the full work. The model has 8 layers, the last
// source is layer 4 and the window is 32 rows: the tail is (8 - 4)(32 - 1) + 32 = 156 rows. The bound that the walk
// back of the graph constructor derives (one window less than the tail used) is (8 - 4 - 1)(32 - 1) + 32 = 125 rows and is
// exact here too (measured: 123 is the last tail that is bit-identical on this model, 122 is not). A tail that is
// too short leaves a stale K/V cell in the window of a row the logits read, so those must differ: that is what makes
// the pass of the others mean something. Run with the ubatch a row long, so that no ubatch straddles the cut.
struct case_dsv41_skip : ctx_test {
    const char * name() const override { return "dsv41skip"; }

    const char * description() const override {
        return "DeepSeek-V4.1 exact long-prompt layer skip: the logits of the last row and of the decode after it are the full work's";
    }

    // the rings the skip leaves stale must not matter whatever they hold
    bool uses_fill() const override { return true; }

    std::string skip_reason(model_run & mr) const override {
        char buf[64] = {};
        if (llama_model_meta_val_str(mr.model, "general.architecture", buf, sizeof(buf)) < 0 || std::string(buf) != "deepseek41") {
            return "not a DeepSeek-V4.1 model";
        }

        return ctx_test::skip_reason(mr);
    }

    // logits of the prompt's last row, then of n_gen tokens decoded after it; the skip when tail >= 0
    static bool run_once(model_run & mr, int32_t tail_override, bool skip, int n_prompt, int n_gen, int32_t & tail_out,
            std::vector<std::vector<float>> & logits, uint32_t n_ubatch = 1) {
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        ctx_opts opts;
        opts.n_ctx     = 512;
        opts.n_batch   = 256;
        opts.n_ubatch  = n_ubatch;
        opts.n_seq_max = 1;
        opts.fill      = mr.fill;

        llama_context_ptr ctx = mr.ctx(opts);
        if (!ctx) {
            return false;
        }

        if (tail_override >= 0) {
            setenv("LLAMA_DSV41_SKIP_TAIL", std::to_string(tail_override).c_str(), 1);
        } else {
            unsetenv("LLAMA_DSV41_SKIP_TAIL");
        }

        tail_out = skip ? llama_set_prompt_end(ctx.get(), 0, n_prompt) : 0;
        unsetenv("LLAMA_DSV41_SKIP_TAIL");

        // the prompt in one batch, its last row alone carries the output, as the server's does
        {
            common_batch batch(ctx.get());
            for (int i = 0; i < n_prompt; ++i) {
                batch.add(gen_token(n_vocab, 0, i), i, 0, i + 1 == n_prompt);
            }
            if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                return false;
            }
        }

        logits.clear();
        logits.emplace_back();
        if (!get_current_logits(ctx.get(), logits.back())) {
            return false;
        }

        for (int i = 0; i < n_gen; ++i) {
            if (!decode_one(ctx.get(), gen_token(n_vocab, 0, n_prompt + i), n_prompt + i)) {
                return false;
            }
            logits.emplace_back();
            if (!get_current_logits(ctx.get(), logits.back())) {
                return false;
            }
        }

        return true;
    }

    static logits_diff compare_runs(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
        logits_diff diff;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
            diff.add(a[i].data(), b[i].data(), a[i].size());
        }
        return diff;
    }

    case_result run(model_run & mr) const override {
        setenv("LLAMA_DSV41_LAYER_SKIP", "1", 1);

        const int n_prompt = 236;
        const int n_gen    = 6;

        const int32_t n_layer = llama_model_n_layer(mr.model);
        const int32_t n_swa   = llama_model_n_ctx_train(mr.model)/8; // the window of the test models (test-llama-archs)
        const int32_t tail_expect = (n_layer - 4)*(n_swa - 1) + n_swa;
        const int32_t tail_tight  = (n_layer - 4 - 1)*(n_swa - 1) + n_swa;

        std::vector<std::vector<float>> full, skipped, tight, one_less, far_too_short;
        int32_t tail = 0;
        int32_t tail_dummy = 0;

        CTX_REQUIRE(run_once(mr, -1, false, n_prompt, n_gen, tail_dummy, full), "the full run failed");
        CTX_REQUIRE(run_once(mr, -1, true, n_prompt, n_gen, tail, skipped), "the skipping run failed");
        CTX_REQUIRE(tail == tail_expect, "the tail is " + std::to_string(tail) + " rows, expected " + std::to_string(tail_expect));

        const logits_diff d_skip = compare_runs(full, skipped);
        TLOG_INF("%s: tail %d: nmse %.3e, max diff %.3e\n", __func__, tail, d_skip.nmse(), (double) d_skip.diff_max);
        CTX_REQUIRE(d_skip.n_nan == 0 && d_skip.nmse() <= 1e-12,
                "the skip changed the logits: nmse " + std::to_string(d_skip.nmse()) + ", max diff " + std::to_string(d_skip.diff_max));

        // ubatches of 16 rows (the cut falls inside one of them: it and the rows after it run in full), and of 64
        for (uint32_t n_ub : { 16u, 64u }) {
            std::vector<std::vector<float>> full_ub, skipped_ub;
            CTX_REQUIRE(run_once(mr, -1, false, n_prompt, n_gen, tail_dummy, full_ub, n_ub), "the full run failed");
            CTX_REQUIRE(run_once(mr, -1, true, n_prompt, n_gen, tail_dummy, skipped_ub, n_ub), "the skipping run failed");
            const logits_diff d_ub = compare_runs(full_ub, skipped_ub);
            TLOG_INF("%s: ubatch %u: nmse %.3e, max diff %.3e\n", __func__, n_ub, d_ub.nmse(), (double) d_ub.diff_max);
            CTX_REQUIRE(d_ub.n_nan == 0 && d_ub.nmse() <= 1e-12, "the skip changed the logits with ubatches of " + std::to_string(n_ub));
        }

        // the tightest cut is exact too ...
        CTX_REQUIRE(run_once(mr, tail_tight, true, n_prompt, n_gen, tail_dummy, tight), "the tight run failed");
        const logits_diff d_tight = compare_runs(full, tight);
        TLOG_INF("%s: tail %d: nmse %.3e, max diff %.3e\n", __func__, tail_tight, d_tight.nmse(), (double) d_tight.diff_max);
        CTX_REQUIRE(d_tight.n_nan == 0 && d_tight.nmse() <= 1e-12,
                "the tightest cut changed the logits: nmse " + std::to_string(d_tight.nmse()));

        // ... and a tail a few rows short and a much shorter one are not: the test sees a stale ring
        CTX_REQUIRE(run_once(mr, n_swa, true, n_prompt, n_gen, tail_dummy, far_too_short), "the short tail run failed");
        const logits_diff d_short = compare_runs(full, far_too_short);
        TLOG_INF("%s: tail %d: nmse %.3e, max diff %.3e\n", __func__, n_swa, d_short.nmse(), (double) d_short.diff_max);
        CTX_REQUIRE(d_short.nmse() > 1e-8, "a tail of one window left the logits as they were: the skip is not taken");


        CTX_REQUIRE(run_once(mr, tail_tight - 3, true, n_prompt, n_gen, tail_dummy, one_less), "the short run failed");
        const logits_diff d_less = compare_runs(full, one_less);
        TLOG_INF("%s: tail %d: nmse %.3e, max diff %.3e\n", __func__, tail_tight - 3, d_less.nmse(), (double) d_less.diff_max);
        CTX_REQUIRE(d_less.diff_max > 0.0f, "a tail three rows short of the bound left the logits as they were: the test does not see a stale ring");

        unsetenv("LLAMA_DSV41_LAYER_SKIP");

        return case_result::pass();
    }
};

std::vector<ctx_test_ptr> make_dsv41_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_dsv41_skip>());

    return cases;
}
