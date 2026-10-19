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
                const uint32_t k = n_draft_max - (r % 4 == 3 ? 3 : 0); // a plain decode now and then
                const uint32_t a = (r*5 + n_prompt) % (k + 1);         // accepted drafts

                std::vector<row_t> rows = { { tok(pos), pos, true } };
                for (uint32_t i = 1; i <= k; ++i) {
                    const llama_token t = i <= a ? tok(pos + (llama_pos) i) : (llama_token) ((tok(pos + (llama_pos) i) + i) % n_vocab);
                    rows.push_back({ t, pos + (llama_pos) i, true });
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

                if (a < k) {
                    for (llama_context * ctx : { ctx_spec.get(), ctx_bs.get() }) {
                        CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx), 0, pos + (llama_pos) a + 1, -1),
                                "the rollback of " + std::to_string(k - a) + " tokens was refused");
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

std::vector<ctx_test_ptr> make_rs_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_rollback>());
    cases.push_back(std::make_unique<case_multi_seq_split_replay>());
    cases.push_back(std::make_unique<case_shared_seq_reserve>());
    cases.push_back(std::make_unique<case_spec_rounds>());

    return cases;
}
