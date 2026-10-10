#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model (if any) - this also creates the MTP draft context when MTP speculation is enabled
    common_speculative_init_result_ptr spec_init;

    {
        common_params params_dft = common_base_params_to_speculative(params);

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }

    llama_context * ctx_dft = params.speculative.draft.ctx_dft;

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);

    if (spec == nullptr) {
        LOG_ERR("%s", "failed to initialize speculative decoding\n");
        return 1;
    }

    // eval the prompt on the target and feed it to the speculative implementation(s)
    {
        common_batch batch_prompt(ctx_tgt);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            batch_prompt.add(inp[i], i, seq_id, false);
        }

        llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get());

        if (!common_speculative_process(spec, batch_prompt)) {
            LOG_ERR("%s", "failed to process speculative prompt\n");
            return 1;
        }
    }

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    common_batch batch_tgt(ctx_tgt);

    llama_tokens draft;

    common_prompt_checkpoint ckpt;

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    int64_t t_tgt_us = 0, t_smpl_us = 0, n_tgt = 0, n_tgt_tok = 0;
    // temporary: per-round trace (LLAMA_SPEC_TIMING=2): start, target window and accepted tokens of every round
    const bool spec_trace = getenv("LLAMA_SPEC_TIMING") != nullptr && atoi(getenv("LLAMA_SPEC_TIMING")) >= 2;
    std::vector<int64_t> trace_t0, trace_tgt;
    std::vector<int>     trace_acc;

    // launch-ahead (LLAMA_SPEC_AHEAD=1): the target's decode is launched from inside the drafter, when its first
    // step computes, with placeholder ids for the draft rows (llama_decode_prepare); the ids follow when the draft
    // is done (llama_decode_commit). The draft is padded to the prepared rows: the padding rows are rejected like
    // any draft token. Off by default: the launch already overlapped the window's start, so the round is the same
    // (1K: 26.7 ms either way); it is the base for a draft computed in one window.
    // LLAMA_SPEC_AHEAD=2 prepares after the draft, with the real ids (a test of the mechanism alone).
    const int  spec_ahead_mode = getenv("LLAMA_SPEC_AHEAD") == nullptr ? 0 : atoi(getenv("LLAMA_SPEC_AHEAD"));
    const bool spec_ahead = spec_ahead_mode != 0;
    const bool spec_ahead_late = spec_ahead_mode == 2;
    int  n_prepared = 0; // rows of the prepared decode, 0 = none
    int64_t t_prepared = 0;

    while (true) {
        if (spec_trace) {
            trace_t0.push_back(ggml_time_us());
        }
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // determine the max draft that fits the remaining context and generation budget
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            if (params.n_predict >= 0) {
                n_draft_max = std::min(n_draft_max, params.n_predict - n_predict - 1);
            }
            n_draft_max = std::max(n_draft_max, 0);

            // generate a new draft
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .pos0       = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
            };
            n_prepared = 0;
            // n_draft_max 0 is no limit for the drafter (common_speculative_draft), not zero tokens
            const int n_rows = 1 + (n_draft_max > 0 ? std::min(n_draft_max, params_spec.draft.n_max) : params_spec.draft.n_max);
            auto prepare = [&, n_rows]() {
                batch_tgt.clear();
                for (int i = 0; i < n_rows; ++i) {
                    // the ids of the draft rows come with the commit (the real ones in the late test mode)
                    const llama_token id = spec_ahead_late && i > 0 && i - 1 < (int) draft.size() ? draft[i - 1] : id_last;
                    batch_tgt.add(id, n_past + i, seq_id, true);
                }
                const int64_t t0 = ggml_time_us();
                const int32_t rc = llama_decode_prepare(ctx_tgt, batch_tgt.get());
                if (rc == 0) {
                    n_prepared = n_rows;
                    t_prepared = t0;
                } else if (rc == 3) {
                    LOG_DBG("%s", "no gated window for this decode yet, decoding it after the draft\n");
                } else {
                    LOG_ERR("llama_decode_prepare failed, rc = %d\n", rc);
                }
            };
            if (spec_ahead && !spec_ahead_late) {
                common_speculative_set_launch_hook(spec, prepare);
            }
            common_speculative_draft(spec);
            common_speculative_set_launch_hook(spec, nullptr);
            if (spec_ahead_late) {
                prepare();
            }

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            // reset the draft context to the checkpoint before verification
            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, seq_id, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                batch_tgt.add(draft[i], n_past + i, seq_id, true);
            }

            if (n_prepared > 0) {
                // the prepared decode has n_prepared rows: the draft padded to them (the padding is rejected)
                GGML_ASSERT((int) draft.size() + 1 <= n_prepared);
                std::vector<llama_token> ids;
                ids.push_back(id_last);
                ids.insert(ids.end(), draft.begin(), draft.end());
                while ((int) ids.size() < n_prepared) {
                    batch_tgt.add(id_last, n_past + (int) ids.size() - 1, seq_id, true);
                    ids.push_back(id_last);
                }
                if (llama_decode_commit(ctx_tgt, ids.data(), ids.size()) != 0) {
                    LOG_ERR("%s", "llama_decode_commit failed\n");
                    break;
                }
                if (getenv("LLAMA_SPEC_TIMING") != nullptr) {
                    llama_synchronize(ctx_tgt);
                }
                t_tgt_us += ggml_time_us() - t_prepared;
                if (spec_trace) {
                    trace_tgt.push_back(ggml_time_us() - t_prepared);
                }
                n_prepared = 0;
            } else {
                const int64_t t0 = ggml_time_us();
                llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
                if (getenv("LLAMA_SPEC_TIMING") != nullptr) {
                    llama_synchronize(ctx_tgt);
                }
                t_tgt_us += ggml_time_us() - t0;
                if (spec_trace) {
                    trace_tgt.push_back(ggml_time_us() - t0);
                }
            }
            n_tgt++;
            n_tgt_tok += batch_tgt.size();
        }

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("%s", "failed to process speculative batch\n");
            break;
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // save the size of the draft being verified
        const size_t n_draft = draft.size();

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        const int64_t t_smpl0 = ggml_time_us();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        t_smpl_us += ggml_time_us() - t_smpl0;

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, n_draft);

            draft = std::move(ids);

            {
                ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            if (ctx_dft) {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        if (spec_trace) {
            trace_acc.push_back((int) ids.size());
        }
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);

            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            }
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));
    if (getenv("LLAMA_SPEC_TIMING") != nullptr && n_tgt > 0) {
        LOG_INF("spec-timing: %d rounds, %.2f ms/round; target %.2f ms/window (%.2f tokens), sample+accept %.2f ms/round\n",
                (int) n_tgt, (t_dec_end - t_dec_start)/1e3/n_tgt, t_tgt_us/1e3/n_tgt, (double) n_tgt_tok/n_tgt, t_smpl_us/1e3/n_tgt);
    }

    if (spec_trace) {
        // round i: its length (to the start of the next round), its target window, tokens it produced; in 0.01 ms
        fprintf(stderr, "spec-trace:");
        for (size_t i = 0; i + 1 < trace_t0.size() && i < trace_tgt.size() && i < trace_acc.size(); ++i) {
            fprintf(stderr, " %lld/%lld/%d", (long long) (trace_t0[i + 1] - trace_t0[i])/10, (long long) trace_tgt[i]/10, trace_acc[i]);
        }
        fprintf(stderr, "\n");
    }

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    common_speculative_print_stats(spec);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());


    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
