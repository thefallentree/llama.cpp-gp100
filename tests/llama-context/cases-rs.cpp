// recurrent state cases

#include "test-case.h"

#include "test-log.h"
#include "test-util.h"

#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// roll back multiple sequences, then replay them in a single batch whose per-seq token count
// exceeds n_ubatch: each seq's replay spans several ubatches while its rollback restore is
// still pending. Compared against a reference context that never advanced past the rollback
// point and decodes the identical replay batch.
struct case_multi_seq_split_replay : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "replay"; }

    const char * description() const override {
        return "replay rolled-back sequences in a batch wider than n_ubatch, against a reference context";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t  n_seqs     = 2;
        constexpr uint32_t  n_ubatch   = 16;
        constexpr uint32_t  n_prompt   = 20;
        constexpr uint32_t  n_rollback = 3;
        constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
        constexpr llama_pos p0         = n_prompt - n_rollback;

        // both contexts have the same layout, only what they decode differs
        const auto make_ctx_multi = [&]() {
            ctx_opts opts;
            opts.n_ctx      = 256;
            opts.n_batch    = 256;
            opts.n_ubatch   = n_ubatch;
            opts.n_seq_max  = n_seqs;
            opts.n_rs_seq   = 8;
            opts.kv_unified = 0;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_roll = make_ctx_multi();
        llama_context_ptr ctx_ref  = make_ctx_multi();
        if (!ctx_roll || !ctx_ref) {
            return case_result::fail();
        }

        if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        const auto tok = [&](uint32_t seq, llama_pos pos) {
            return gen_token(n_vocab, (llama_seq_id) seq, pos);
        };

        bool ok = true;

        // keep the anchor at p0 - 1 when rolling back the tail
        for (uint32_t s = 0; s < n_seqs && ok; ++s) {
            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);

            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) n_prompt);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) p0);

            ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

            // a second partial removal while one is pending must be refused
            ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
        }
        CTX_REQUIRE(ok, "the multi-seq prefill / rollback failed");

        // all seqs replay in a single batch
        const auto decode_replay = [&](llama_context * ctx) {
            common_batch batch(ctx);
            for (uint32_t s = 0; s < n_seqs; ++s) {
                for (uint32_t i = 0; i < n_replay; ++i) {
                    const llama_pos pos = p0 + (llama_pos) i;
                    batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
                }
            }
            return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
        };

        ok = decode_replay(ctx_roll.get()) && decode_replay(ctx_ref.get());
        CTX_REQUIRE(ok, "the multi-seq replay decode failed");

        // both contexts decode identical batches, so the logits should match;
        // random dummy models can still drift up to ~1.7e-5, so the bound is 1e-4
        constexpr float nmse_eps = 1e-4f;

        logits_diff diff;
        for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
                return case_result::fail();
            }

            diff.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(diff.nmse() <= nmse_eps,
                "multi-seq split replay logits mismatch (max diff " + std::to_string(diff.diff_max) +
                ", nmse " + std::to_string(diff.nmse()) + ", first difference at index " + std::to_string(diff.i_first) + ")");

        TLOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff.diff_max, diff.nmse());

        // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
        //   in ctx_ref only, then compare identical seq-1-only continuations
        constexpr uint32_t n_tail = 4;

        {
            common_batch batch_tail(ctx_ref.get());
            for (uint32_t i = 0; i < n_tail; ++i) {
                const llama_pos pos = p0 + (llama_pos) (n_replay + i);
                batch_tail.add(tok(0, pos + 7), pos, 0, false);
            }
            ok = llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch_tail.get()) == 0;
        }

        logits_diff diff_tail;
        for (uint32_t i = 0; i < n_tail && ok; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            ok = decode_one(ctx_roll.get(), tok(1, pos), pos, 1) &&
                 decode_one(ctx_ref.get(),  tok(1, pos), pos, 1);
            if (!ok) {
                break;
            }

            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing tail logits at step %u\n", __func__, i);
                return case_result::fail();
            }

            diff_tail.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(ok && diff_tail.nmse() <= nmse_eps,
                "a seq-1-only decode leaked the state of seq 0 (nmse " + std::to_string(diff_tail.nmse()) + ")");

        TLOG_INF("%s: seq-1-only decode is independent of seq 0 (max diff %g, nmse %g)\n",
                __func__, (double) diff_tail.diff_max, diff_tail.nmse());

        return case_result::pass();
    }
};

// save a rolled-back single-seq state, restore it into a fresh and into a dirty context, and
// verify exact logit matches on replay
struct case_rollback : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "rollback"; }

    const char * description() const override {
        return "a rolled-back recurrent state restores into fresh and dirty contexts with identical replay";
    }

    case_result run(model_run & mr) const override {
        const llama_vocab * vocab   = llama_model_get_vocab(mr.model);
        const int           n_vocab = llama_vocab_n_tokens(vocab);

        const auto make_rs_ctx = [&]() {
            ctx_opts opts;
            opts.n_seq_max = 1;
            opts.n_rs_seq  = 8;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_src = make_rs_ctx();
        llama_context_ptr ctx_dst = make_rs_ctx();
        if (!ctx_src || !ctx_dst) {
            return case_result::fail();
        }

        const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
        if (n_rs_seq == 0) {
            return case_result::skip("n_rs_seq is disabled");
        }

        constexpr uint32_t n_rollback = 3;
        if (n_rs_seq < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        llama_tokens tokens;
        if (!mr.has_vocab()) {
            tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        } else {
            tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
        }
        if (tokens.empty()) {
            TLOG_ERR("%s: not enough prompt tokens\n", __func__);
            return case_result::fail();
        }
        tokens.resize(n_rs_seq + 1, tokens.back());

        const llama_pos rollback_pos = (llama_pos) tokens.size() - n_rollback;

        // Decode the full prompt on the source, then roll back three positions.
        // Replaying them crosses DSV4's ratio-4 compressor boundary.
        // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
        if (!decode_tokens(ctx_src.get(), tokens)) {
            TLOG_ERR("%s: failed to decode the prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: rollback failed\n", __func__);
            return case_result::fail();
        }

        // Save the rolled-back state and restore it into a fresh context.
        common_prompt_checkpoint ckpt;
        ckpt.update_tgt(ctx_src.get(), 0, 0);
        ckpt.load_tgt(ctx_dst.get(), 0, 0);

        // exact match - the same context configuration replaying the same tokens
        constexpr float nmse_eps = 0.0;

        std::vector<std::vector<float>> logits_src_replay(n_rollback);
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                TLOG_ERR("%s: missing replay logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);

            const logits_diff diff = compare_logits(logits_src, logits_dst, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "replay logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
        //       this is not the case here. add asserts and guardrails to prevent such attempts
        //if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1) ||
        //    !llama_memory_seq_rm(llama_get_memory(ctx_dst.get()), 0, rollback_pos, -1)) {
        //    TLOG_ERR("%s: partial rollback failed\n", __func__);
        //    return case_result::fail();
        //}

        //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        //common_prompt_checkpoint ckpt_partial;
        //ckpt_partial.update_tgt(ctx_src.get(), 0, partial_flags);
        //ckpt_partial.load_tgt(ctx_dst.get(), 0, partial_flags);

        // Repeat the load into a context that already has its own rollback state:
        // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
        // non-zero at load time. The restore must wipe that state and still match.
        llama_context_ptr ctx_dirty = make_rs_ctx();
        if (!ctx_dirty) {
            return case_result::fail();
        }

        llama_tokens noise = tokens;
        for (auto & tok : noise) {
            tok = (tok + 1) % n_vocab;
            if (tok < 0) {
                tok = 0;
            }
        }
        if (!decode_tokens(ctx_dirty.get(), noise)) {
            TLOG_ERR("%s: failed to decode the dirty prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: dirty rollback failed\n", __func__);
            return case_result::fail();
        }

        ckpt.load_tgt(ctx_dirty.get(), 0, 0);

        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
            if (logits_dirty == nullptr) {
                TLOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const logits_diff diff = compare_logits(logits_src_replay[i].data(), logits_dirty, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "dirty-ctx logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        TLOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);

        return case_result::pass();
    }
};

// decode a prompt into seq 0, share its cells with a second sequence, then keep decoding both
struct case_shared_seq_reserve : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "shared"; }

    const char * description() const override {
        return "sequences sharing cells keep decoding correctly after a seq_cp";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the shared-seq batch is a multi-seq graph";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t n_seqs     = 2;
        constexpr uint32_t n_prompt   = 128;
        constexpr uint32_t n_continue = 32;

        ctx_opts opts;
        opts.n_ctx      = 512;
        opts.n_batch    = 256;
        opts.n_ubatch   = 64;
        opts.n_seq_max  = n_seqs;
        opts.kv_unified = 1; // only a unified cache shares cells on seq_cp

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        if (!decode_gen(ctx.get(), n_vocab, 0, 0, (llama_pos) n_prompt)) {
            TLOG_ERR("%s: prompt decode failed\n", __func__);
            return case_result::fail();
        }

        // this is what llama-batched-bench does for -pps
        llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, -1, -1);

        for (uint32_t i = 0; i < n_continue; ++i) {
            const llama_pos pos = (llama_pos) (n_prompt + i);

            common_batch batch(ctx.get());
            for (uint32_t s = 0; s < n_seqs; ++s) {
                batch.add(gen_token(n_vocab, (llama_seq_id) s, pos), pos, (llama_seq_id) s, true);
            }
            CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                    "shared-seq decode failed at step " + std::to_string(i));

            for (uint32_t s = 0; s < n_seqs; ++s) {
                const float * logits = llama_get_logits_ith(ctx.get(), (int) s);
                if (logits == nullptr) {
                    TLOG_ERR("%s: missing shared-seq logits at index %u\n", __func__, s);
                    return case_result::fail();
                }

                bool finite = true;
                for (int t_ = 0; t_ < n_vocab; ++t_) {
                    finite = finite && std::isfinite(logits[t_]);
                }

                CTX_REQUIRE(finite, "non-finite shared-seq logit at step " + std::to_string(i) + ", seq " + std::to_string(s));
            }
        }

        TLOG_INF("%s: shared-seq decode succeeded (%u tokens after seq_cp)\n", __func__, n_continue*n_seqs);

        return case_result::pass();
    }
};

// the speculative rounds of the server on a cache with rollback
// the prompt is cut like the server does (its last n_rs_seq + 1 tokens come last), then each round decodes [last token, draft...] with an output on every row and removes the rejected tail
// every row with a true history must match a reference context that decodes the stream as one batch
// the same rounds run on a second context with the greedy sampler of the server on the backend, whose sampled token of a row must be a maximum of the reference logits
// the prompt lengths and the accepted counts cycle through the alignments of the compressor blocks and of the ubatch splits
struct case_spec_rounds : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "spec"; }

    const char * description() const override {
        return "speculative verify rounds with partial rejection match a plain batch of the same tokens";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t n_draft_max = 3;
        constexpr uint32_t n_total     = 96; // positions of the stream, a few blocks of every compressor ratio of a small model
        constexpr double   nmse_eps    = 1e-4;

        ctx_opts opts_spec;
        opts_spec.n_ctx         = 256;
        opts_spec.n_batch       = 256;
        opts_spec.n_ubatch      = 64;
        opts_spec.n_seq_max     = 1;
        opts_spec.n_rs_seq      = n_draft_max;
        opts_spec.n_graph_slots = 8;
        opts_spec.n_outputs_max         = 2*(n_draft_max + 1);
        opts_spec.n_outputs_max_per_seq = n_draft_max + 1;

        llama_context_ptr ctx_spec = mr.ctx(opts_spec);
        llama_context_ptr ctx_bs   = mr.ctx(opts_spec);
        if (!ctx_spec || !ctx_bs) {
            return case_result::fail();
        }
        if (llama_n_rs_seq(ctx_spec.get()) < n_draft_max) {
            return case_result::skip("n_rs_seq is disabled");
        }

        // the greedy sampler of the server (-bs) on the backend: one sampled token per output row of the verification batch, no logits
        llama_sampler_ptr smpl_backend(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(smpl_backend.get(), llama_sampler_init_greedy());
        const bool on_backend = llama_set_sampler(ctx_bs.get(), 0, smpl_backend.get());

        // the reference: the whole stream in one batch, a logit row per position
        ctx_opts opts_ref;
        opts_ref.n_ctx     = 256;
        opts_ref.n_batch   = 256;
        opts_ref.n_ubatch  = 128;
        opts_ref.n_seq_max = 1;

        llama_context_ptr ctx_ref = mr.ctx(opts_ref);
        if (!ctx_ref) {
            return case_result::fail();
        }

        const auto tok = [&](llama_pos pos) {
            return gen_token(n_vocab, 0, pos);
        };

        {
            common_batch batch(ctx_ref.get());
            for (uint32_t i = 0; i < n_total; ++i) {
                batch.add(tok((llama_pos) i), (llama_pos) i, 0, true);
            }
            CTX_REQUIRE(llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "the reference decode failed");
        }

        std::vector<float> logits_ref((size_t) n_total*n_vocab);
        for (uint32_t i = 0; i < n_total; ++i) {
            const float * l = llama_get_logits_ith(ctx_ref.get(), (int) i);
            CTX_REQUIRE(l != nullptr, "missing reference logits at position " + std::to_string(i));
            std::copy(l, l + n_vocab, logits_ref.begin() + (size_t) i*n_vocab);
        }

        struct row_t {
            llama_token tok;
            llama_pos   pos;
            bool        output;
        };

        // the rows go to both contexts
        const auto decode_rows = [&](const std::vector<row_t> & rows) {
            for (llama_context * ctx : { ctx_spec.get(), ctx_bs.get() }) {
                common_batch batch(ctx);
                for (const auto & r : rows) {
                    batch.add(r.tok, r.pos, 0, r.output);
                }
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                    return false;
                }
            }
            return true;
        };

        const auto check_row = [&](int i_row, llama_pos pos) {
            const float * l = llama_get_logits_ith(ctx_spec.get(), i_row);

            return l != nullptr ? compare_logits(l, logits_ref.data() + (size_t) pos*n_vocab, n_vocab).nmse() : INFINITY;
        };

        // the backend sampled token of a row is a maximum of the reference logits of its position (up to the noise of another batch shape)
        const auto sampled_ok = [&](int i_row, llama_pos pos) {
            if (!on_backend) {
                return true;
            }
            const llama_token t = llama_get_sampled_token_ith(ctx_bs.get(), i_row);
            if (t < 0 || t >= n_vocab) {
                return false;
            }
            const float * l = logits_ref.data() + (size_t) pos*n_vocab;
            return l[t] >= *std::max_element(l, l + n_vocab) - 1e-3f;
        };

        uint32_t n_rounds = 0;
        uint32_t n_rolled = 0;

        for (uint32_t n_prompt : { 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 11u, 13u, 16u, 21u, 33u }) {
            // the rollback state of a previous stream must not leak into the next
            llama_memory_clear(llama_get_memory(ctx_spec.get()), true);
            llama_memory_clear(llama_get_memory(ctx_bs.get()), true);

            // the server cuts the last n_rs_seq + 1 prompt tokens off, to checkpoint before them
            std::vector<uint32_t> bounds = { 0 };
            if (n_prompt > n_draft_max + 1) {
                bounds.push_back(n_prompt - (n_draft_max + 1));
            }
            bounds.push_back(n_prompt);

            for (size_t c = 0; c + 1 < bounds.size(); ++c) {
                std::vector<row_t> rows;
                for (uint32_t i = bounds[c]; i < bounds[c + 1]; ++i) {
                    rows.push_back({ tok((llama_pos) i), (llama_pos) i, i + 1 == n_prompt });
                }
                CTX_REQUIRE(decode_rows(rows), "prompt chunk decode failed, n_prompt " + std::to_string(n_prompt) + ", chunk " + std::to_string(c));
            }

            const double nmse_p = check_row(-1, (llama_pos) n_prompt - 1); // the last output row
            CTX_REQUIRE(nmse_p <= nmse_eps, "prompt logits mismatch, n_prompt " + std::to_string(n_prompt) + ", nmse " + std::to_string(nmse_p));
            CTX_REQUIRE(sampled_ok(-1, (llama_pos) n_prompt - 1), "the sampled token of the prompt is not a maximum, n_prompt " + std::to_string(n_prompt));

            llama_pos pos = (llama_pos) n_prompt; // the true token at pos is the next one to decode, the way the server's sampled token is

            for (uint32_t r = 0; pos + (llama_pos) n_draft_max < (llama_pos) n_total; ++r) {
                // a plain decode now and then, and a window padded to the full width (launch-ahead of the server, with a draft
                // shorter than n_draft_max): the padding rows leave with the rejected ones in one removal
                const uint32_t pad = r % 4 == 1 ? 1 : 0;
                const uint32_t k   = n_draft_max - (r % 4 == 3 ? 3 : 0) - pad;
                const uint32_t a   = (r*5 + n_prompt) % (k + 1); // accepted drafts

                std::vector<row_t> rows = { { tok(pos), pos, true } };
                for (uint32_t i = 1; i <= k; ++i) {
                    const llama_token t = i <= a ? tok(pos + (llama_pos) i) : (llama_token) ((tok(pos + (llama_pos) i) + i) % n_vocab);
                    rows.push_back({ t, pos + (llama_pos) i, true });
                }
                for (uint32_t i = k + 1; i <= k + pad; ++i) {
                    rows.push_back({ rows.back().tok, pos + (llama_pos) i, true });
                }
                CTX_REQUIRE(decode_rows(rows), "verify decode failed, n_prompt " + std::to_string(n_prompt) + ", round " + std::to_string(r));

                for (uint32_t i = 0; i <= a; ++i) {
                    const double nmse = check_row((int) i, pos + (llama_pos) i);
                    CTX_REQUIRE(nmse <= nmse_eps,
                            "verify logits mismatch at position " + std::to_string(pos + (llama_pos) i) + " (n_prompt " + std::to_string(n_prompt) +
                            ", round " + std::to_string(r) + ", " + std::to_string(a) + "/" + std::to_string(k) + " accepted), nmse " + std::to_string(nmse));
                    CTX_REQUIRE(sampled_ok((int) i, pos + (llama_pos) i),
                            "the sampled token of row " + std::to_string(i) + " is not a maximum at position " + std::to_string(pos + (llama_pos) i) +
                            " (n_prompt " + std::to_string(n_prompt) + ", round " + std::to_string(r) + ")");
                }

                if (a < k + pad) {
                    for (llama_context * ctx : { ctx_spec.get(), ctx_bs.get() }) {
                        CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx), 0, pos + (llama_pos) a + 1, -1),
                                "the rollback of " + std::to_string(k + pad - a) + " tokens was refused");
                    }
                    n_rolled++;
                }

                pos += (llama_pos) a + 1;
                n_rounds++;
            }
        }

        TLOG_INF("%s: %u speculative rounds (%u rolled back) matched the plain batch, backend sampling %s\n", __func__, n_rounds, n_rolled, on_backend ? "on" : "off");

        return case_result::pass();
    }
};

// requests one after the other on the same contexts, the way the server serves a client that repeats a prompt:
//  - a request makes its own sampler chain (the old one is freed before the context learns of the new one), bare greedy or the pre-selection chain of --ignore-eos
//  - its memory is emptied (cache_prompt: false) or restored from the checkpoint taken 4 tokens before the end of the prompt (a prompt cache hit)
//  - then the verification rounds of case_spec_rounds
// a repeat of a request must give the rows of the first one up to rounding, and every decision of the backend sampler must be a maximum of the plain batch
struct case_spec_requests : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "specreq"; }

    const char * description() const override {
        return "a repeated prompt after a speculative request (new sampler chain, emptied or restored memory) reproduces the first request";
    }

    // --spec-draft-n-max 3 and 5 of the 3-node run: verification batches of 4 and 6 rows, n_rs_seq 3 and 5
    case_result run(model_run & mr) const override {
        for (const uint32_t n_draft_max : { 3u, 5u }) {
            case_result res = run_n_max(mr, n_draft_max);
            if (res.status != test_status::PASS) {
                res.reason = "n_draft_max " + std::to_string(n_draft_max) + ": " + res.reason;
                return res;
            }
        }

        return case_result::pass();
    }

    static case_result run_n_max(model_run & mr, const uint32_t n_draft_max) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t n_ubatch    = 64;
        constexpr uint32_t n_total     = 160; // positions of the stream
        constexpr double   nmse_eps    = 1e-4;
        constexpr double   nmse_repeat = 1e-6; // a repeat of a request: a restored memory gives the logits of the live one up to rounding, not bit for bit

        // the tokens --ignore-eos biases away: 1 and 2
        constexpr int32_t n_bias = 2;
        if (n_vocab < 16) {
            return case_result::skip("the vocabulary is too small");
        }

        const auto is_biased = [](llama_token t) {
            return t == 1 || t == 2;
        };

        ctx_opts opts_spec;
        opts_spec.n_ctx         = 256;
        opts_spec.n_batch       = 256;
        opts_spec.n_ubatch      = n_ubatch;
        opts_spec.n_seq_max     = 1;
        opts_spec.n_rs_seq      = n_draft_max;
        opts_spec.n_graph_slots = 8;
        opts_spec.n_outputs_max         = 2*(n_draft_max + 1);
        opts_spec.n_outputs_max_per_seq = n_draft_max + 1;

        llama_context_ptr ctx_spec = mr.ctx(opts_spec);
        llama_context_ptr ctx_bs   = mr.ctx(opts_spec);
        if (!ctx_spec || !ctx_bs) {
            return case_result::fail();
        }
        if (llama_n_rs_seq(ctx_spec.get()) < n_draft_max) {
            return case_result::skip("n_rs_seq is disabled");
        }

        enum chain_kind { CHAIN_GREEDY, CHAIN_PRESELECT };
        enum start_kind { START_FRESH, START_RESTORE };

        // common_sampler_init: bare greedy on the backend, or logit biases then greedy with the backend pre-selecting 1 + n_bias candidates
        const auto make_chain = [&](chain_kind kind) {
            llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if (kind == CHAIN_PRESELECT) {
                const llama_logit_bias bias[n_bias] = { { 1, -INFINITY }, { 2, -INFINITY } };
                llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, n_bias, bias));
            }
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            if (kind == CHAIN_PRESELECT) {
                llama_sampler_chain_set_preselect_k(chain, 1 + n_bias);
            }
            return chain;
        };

        ctx_opts opts_ref;
        opts_ref.n_ctx     = 256;
        opts_ref.n_batch   = 256;
        opts_ref.n_ubatch  = 128;
        opts_ref.n_seq_max = 1;

        llama_context_ptr ctx_ref = mr.ctx(opts_ref);
        if (!ctx_ref) {
            return case_result::fail();
        }

        const auto tok = [&](llama_pos pos) {
            return gen_token(n_vocab, 0, pos);
        };

        {
            common_batch batch(ctx_ref.get());
            for (uint32_t i = 0; i < n_total; ++i) {
                batch.add(tok((llama_pos) i), (llama_pos) i, 0, true);
            }
            CTX_REQUIRE(llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "the reference decode failed");
        }

        std::vector<float> logits_ref((size_t) n_total*n_vocab);
        for (uint32_t i = 0; i < n_total; ++i) {
            const float * l = llama_get_logits_ith(ctx_ref.get(), (int) i);
            CTX_REQUIRE(l != nullptr, "missing reference logits at position " + std::to_string(i));
            std::copy(l, l + n_vocab, logits_ref.begin() + (size_t) i*n_vocab);
        }

        // the row of a position without the biased tokens: what the pre-selection chain picks from
        const auto best_unbiased = [&](llama_pos pos) {
            const float * l = logits_ref.data() + (size_t) pos*n_vocab;
            float best = -INFINITY;
            for (int t = 0; t < n_vocab; ++t) {
                if (!is_biased(t)) {
                    best = std::max(best, l[t]);
                }
            }
            return best;
        };

        struct row_t {
            llama_token tok;
            llama_pos   pos;
            bool        output;
        };

        // what a request leaves behind: the plain logits of every verified position and the decision of the backend sampler there
        struct trace_t {
            std::vector<llama_pos>          pos;
            std::vector<std::vector<float>> logits;
            std::vector<llama_token>        decision;
            std::vector<std::string>        where; // the round of each verified row
        };

        std::vector<uint8_t> ckpt_spec;
        std::vector<uint8_t> ckpt_bs;

        llama_sampler_ptr smpl;

        // one request, the error text when it does not hold
        const auto request = [&](chain_kind kind, uint32_t n_prompt, start_kind start, trace_t & tr) -> std::string {
            const auto ctxs = { ctx_spec.get(), ctx_bs.get() };

            // slot.smpl.reset(common_sampler_init(...)): the old chain is gone while the context still has it, then the context is told
            smpl.reset(make_chain(kind));
            if (!llama_set_sampler(ctx_bs.get(), 0, smpl.get())) {
                return "the backend refused the sampler chain";
            }

            // the server cuts the prompt 4 + n_ubatch and 4 tokens before its end and takes a checkpoint before each part
            std::vector<uint32_t> bounds = { 0 };
            if (n_prompt > 4 + n_ubatch) {
                bounds.push_back(n_prompt - (4 + n_ubatch));
            }
            if (n_prompt > 4) {
                bounds.push_back(n_prompt - 4);
            }
            bounds.push_back(n_prompt);

            size_t c0 = 0;
            if (start == START_RESTORE) {
                if (ckpt_spec.empty() || bounds.size() < 3) {
                    return "no checkpoint to restore";
                }
                c0 = bounds.size() - 2; // the last part, the 4 new tokens
                if (!set_seq_state(ctx_spec.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, ckpt_spec) ||
                    !set_seq_state(ctx_bs.get(),   0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, ckpt_bs)) {
                    return "the checkpoint did not restore";
                }
            }

            // slot.mem.seq_rm(slot.id, p0, -1) with the end of the cached prompt as p0
            for (llama_context * ctx : ctxs) {
                if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) bounds[c0], -1)) {
                    return "the memory was not emptied from position " + std::to_string(bounds[c0]);
                }
            }

            const auto decode_rows = [&](const std::vector<row_t> & rows) {
                for (llama_context * ctx : ctxs) {
                    common_batch batch(ctx);
                    for (const auto & r : rows) {
                        batch.add(r.tok, r.pos, 0, r.output);
                    }
                    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                        return false;
                    }
                }
                return true;
            };

            std::string where = "prompt";

            // the verified row i_row of the batch at position pos: the plain logits against the reference, the decision against the plain row
            const auto verify = [&](int i_row, llama_pos pos) -> std::string {
                const float * l = llama_get_logits_ith(ctx_spec.get(), i_row);
                if (l == nullptr) {
                    return "missing logits at position " + std::to_string(pos);
                }
                const double nmse = compare_logits(l, logits_ref.data() + (size_t) pos*n_vocab, n_vocab).nmse();
                if (!(nmse <= nmse_eps)) {
                    return "logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(nmse);
                }

                llama_token decision = LLAMA_TOKEN_NULL;
                if (kind == CHAIN_GREEDY) {
                    decision = llama_get_sampled_token_ith(ctx_bs.get(), i_row);
                    if (decision < 0 || decision >= n_vocab) {
                        return "no sampled token at position " + std::to_string(pos);
                    }
                    const float * r = logits_ref.data() + (size_t) pos*n_vocab;
                    if (!(r[decision] >= *std::max_element(r, r + n_vocab) - 1e-3f)) {
                        return "the sampled token at position " + std::to_string(pos) + " is not a maximum";
                    }
                } else {
                    // the chain on the candidates: the best one that no bias lowered
                    const uint32_t n_cand = llama_get_sampled_candidates_count_ith(ctx_bs.get(), i_row);
                    const llama_token * cand = llama_get_sampled_candidates_ith(ctx_bs.get(), i_row);
                    const float * cand_logit = llama_get_sampled_logits_ith(ctx_bs.get(), i_row);
                    if (cand == nullptr || cand_logit == nullptr || n_cand != 1 + n_bias) {
                        return "the pre-selection gave " + std::to_string(n_cand) + " candidates at position " + std::to_string(pos);
                    }
                    float best = -INFINITY;
                    for (uint32_t k = 0; k < n_cand; ++k) {
                        if (cand[k] < 0 || cand[k] >= n_vocab || !(std::fabs(cand_logit[k] - logits_ref[(size_t) pos*n_vocab + cand[k]]) <= 1e-3f)) {
                            return "candidate " + std::to_string(k) + " is not a token of the row at position " + std::to_string(pos);
                        }
                        if (!is_biased(cand[k]) && cand_logit[k] > best) {
                            best  = cand_logit[k];
                            decision = cand[k];
                        }
                    }
                    if (!(best >= best_unbiased(pos) - 1e-3f)) {
                        return "the pre-selected candidates miss the maximum at position " + std::to_string(pos);
                    }
                }

                tr.pos.push_back(pos);
                tr.logits.emplace_back(l, l + n_vocab);
                tr.decision.push_back(decision);
                tr.where.push_back(where);
                return {};
            };

            for (size_t c = c0; c + 1 < bounds.size(); ++c) {
                // create_checkpoint: before the part that starts 4 tokens before the end
                if (start == START_FRESH && n_prompt > 4 + n_ubatch && c + 2 == bounds.size()) {
                    if (!get_seq_state(ctx_spec.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, ckpt_spec) ||
                        !get_seq_state(ctx_bs.get(),   0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, ckpt_bs)) {
                        return "the checkpoint was not taken";
                    }
                }

                std::vector<row_t> rows;
                for (uint32_t i = bounds[c]; i < bounds[c + 1]; ++i) {
                    rows.push_back({ tok((llama_pos) i), (llama_pos) i, i + 1 == n_prompt });
                }
                if (!decode_rows(rows)) {
                    return "prompt part " + std::to_string(c) + " did not decode";
                }
            }

            std::string err = verify(-1, (llama_pos) n_prompt - 1); // the last output row
            if (!err.empty()) {
                return "prompt: " + err;
            }

            llama_pos pos = (llama_pos) n_prompt;

            for (uint32_t r = 0; pos + (llama_pos) n_draft_max < (llama_pos) n_total; ++r) {
                // a plain decode now and then, and a window padded to the full width (launch-ahead of the server, with a draft
                // shorter than n_draft_max): the padding rows leave with the rejected ones in one removal
                const uint32_t pad = r % 4 == 1 ? 1 : 0;
                const uint32_t k   = n_draft_max - (r % 4 == 3 ? 3 : 0) - pad;
                const uint32_t a   = (r*5 + n_prompt) % (k + 1); // accepted drafts

                std::vector<row_t> rows = { { tok(pos), pos, true } };
                for (uint32_t i = 1; i <= k; ++i) {
                    const llama_token t = i <= a ? tok(pos + (llama_pos) i) : (llama_token) ((tok(pos + (llama_pos) i) + i) % n_vocab);
                    rows.push_back({ t, pos + (llama_pos) i, true });
                }
                for (uint32_t i = k + 1; i <= k + pad; ++i) {
                    rows.push_back({ rows.back().tok, pos + (llama_pos) i, true });
                }
                if (!decode_rows(rows)) {
                    return "round " + std::to_string(r) + " did not decode";
                }

                where = "round " + std::to_string(r) + " (" + std::to_string(a) + "/" + std::to_string(k) + " accepted)";
                for (uint32_t i = 0; i <= a; ++i) {
                    err = verify((int) i, pos + (llama_pos) i);
                    if (!err.empty()) {
                        return "round " + std::to_string(r) + ": " + err;
                    }
                }

                if (a < k + pad) { // the padding rows leave with the rejected ones (as in case_spec_rounds)
                    for (llama_context * ctx : ctxs) {
                        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, pos + (llama_pos) a + 1, -1)) {
                            return "the rollback of " + std::to_string(k + pad - a) + " tokens was refused in round " + std::to_string(r);
                        }
                    }
                }

                pos += (llama_pos) a + 1;
            }

            return {};
        };

        // what the bench and the serve check of the 3-node DeepSeek run did, in its order
        struct req_t {
            const char * what;
            chain_kind   chain;
            uint32_t     n_prompt;
            start_kind   start;
            int          same_as; // the request whose rows this one repeats, -1: none
        };

        const std::vector<req_t> reqs = {
            { "sanity",                CHAIN_GREEDY,    5,   START_FRESH,   -1 },
            { "sanity",                CHAIN_GREEDY,    6,   START_FRESH,   -1 },
            { "warmup",                CHAIN_GREEDY,    20,  START_FRESH,   -1 },
            { "bench run 1",           CHAIN_PRESELECT, 100, START_FRESH,   -1 },
            { "bench run 2",           CHAIN_PRESELECT, 100, START_FRESH,    3 }, // cache_prompt: false, the memory emptied
            { "cache hit",             CHAIN_PRESELECT, 100, START_RESTORE,  3 },
            { "cache hit, bare",       CHAIN_GREEDY,    100, START_RESTORE,  3 },
            { "cache hit again",       CHAIN_GREEDY,    100, START_RESTORE,  3 },
            { "bench run 3",           CHAIN_PRESELECT, 100, START_FRESH,    3 },
            { "short",                 CHAIN_GREEDY,    33,  START_FRESH,   -1 },
        };

        std::vector<trace_t> traces(reqs.size());

        const auto sci = [](double v) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.3g", v);
            return std::string(buf);
        };

        for (size_t i = 0; i < reqs.size(); ++i) {
            const auto & rq = reqs[i];

            const std::string err = request(rq.chain, rq.n_prompt, rq.start, traces[i]);
            CTX_REQUIRE(err.empty(), std::string("request ") + std::to_string(i) + " (" + rq.what + "): " + err);

            if (rq.same_as >= 0) {
                const trace_t & a = traces[rq.same_as];
                const trace_t & b = traces[i];
                CTX_REQUIRE(a.pos == b.pos, std::string("request ") + std::to_string(i) + " (" + rq.what + ") verified other positions than request " + std::to_string(rq.same_as));
                for (size_t k = 0; k < a.pos.size(); ++k) {
                    const logits_diff d = compare_logits(b.logits[k].data(), a.logits[k].data(), n_vocab);
                    CTX_REQUIRE(d.nmse() <= nmse_repeat, std::string("request ") + std::to_string(i) + " (" + rq.what + ") differs from request " + std::to_string(rq.same_as) +
                            " at position " + std::to_string(a.pos[k]) + " (" + a.where[k] + " against " + b.where[k] + "), max diff " + std::to_string(d.diff_max) + ", nmse " + sci(d.nmse()));
                    if (reqs[rq.same_as].chain == rq.chain) {
                        CTX_REQUIRE(a.decision[k] == b.decision[k], std::string("request ") + std::to_string(i) + " (" + rq.what + ") decided another token than request " +
                                std::to_string(rq.same_as) + " at position " + std::to_string(a.pos[k]));
                    }
                }
            }
        }

        llama_set_sampler(ctx_bs.get(), 0, nullptr);

        TLOG_INF("%s: n_draft_max %u: %zu requests, the repeats of a prompt gave the rows of the first\n", __func__, n_draft_max, reqs.size());

        return case_result::pass();
    }
};

std::vector<ctx_test_ptr> make_rs_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_rollback>());
    cases.push_back(std::make_unique<case_multi_seq_split_replay>());
    cases.push_back(std::make_unique<case_shared_seq_reserve>());
    cases.push_back(std::make_unique<case_spec_rounds>());
    cases.push_back(std::make_unique<case_spec_requests>());

    return cases;
}
