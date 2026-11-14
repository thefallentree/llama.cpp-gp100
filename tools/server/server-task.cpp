#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"
#include "../../src/llama-ext.h" // staging API: llama_state_seq_*_range

#include <cstring>
#include <mutex>
#include <thread>
#include <sstream>

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_format)},
            {"reasoning_format",          common_reasoning_format_name(reasoning_format)},
            {"reasoning_in_content",      stream && reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY},
            {"generation_prompt",         sampling.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_format)},
        {"reasoning_format",          common_reasoning_format_name(reasoning_format)},
        {"reasoning_in_content",      stream && reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY},
        {"generation_prompt",         sampling.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(common_chat_session session)
    : chat_session(std::move(session))
    , chat_msg(chat_session.msg())
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
}

common_chat_msg task_result_state::update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls,
        bool truncated) {
    auto msg_prv_copy = chat_msg;
    auto new_msg = is_partial ? chat_session.feed(added) : chat_session.finish(added, truncated);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content.text},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;
    size_t output_index = 0;

    if (oaicompat_msg.reasoning_content != "") {
        const size_t item_output_index = output_index++;
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        const size_t item_output_index = output_index++;
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"output_index", item_output_index},
                {"content_index", 0},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"output_index", item_output_index},
                {"content_index", 0},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const size_t item_output_index = output_index++;
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content.text},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"output_index", 0},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"output_index", 0},
                    {"content_index", 0},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"output_index", anthropic_has_reasoning ? 1 : 0},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"output_index", anthropic_has_reasoning ? 1 : 0},
                        {"content_index", 0},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"output_index", anthropic_has_reasoning ? 1 : 0},
                    {"content_index", 0},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"output_index", (anthropic_has_reasoning ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                    {"output_index", (anthropic_has_reasoning ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content.text},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_tts
//
json server_task_result_realtime::to_json() {
    return json {
        {"text",          content.text},
        {"n_samples",     audio.size()},
        {"response_done", response_done},
        {"cancelled",     cancelled},
        {"n_decoded",     n_decoded},
    };
}

json server_task_result_tts::to_json() {
    return json {
        {"sample_rate", sample_rate},
        {"n_bytes",     audio.size()},
        {"final",       final},
    };
}

//
// server_task_result_decision
//
json server_task_result_decision::to_json() {
    return json {
        {"index",            index},
        {"scores",           scores},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//

static bool srv_cache_timing() {
    static const bool v = getenv("LLAMA_SRV_TIMING") != nullptr && atoi(getenv("LLAMA_SRV_TIMING")) != 0;
    return v;
}

server_state_pool::~server_state_pool() {
    if (th.joinable()) {
        th.join();
    }
    for (auto & s : slabs) {
        if (s.buf) {
            ggml_backend_buffer_free(s.buf);
        }
    }
    for (auto & s : pending) {
        if (s.buf) {
            ggml_backend_buffer_free(s.buf);
        }
    }
}

size_t server_state_pool::capacity() const {
    size_t res = 0;
    for (const auto & s : slabs) {
        res += s.cap;
    }
    return res;
}

bool server_state_pool::new_slab(size_t cap, slab & s) {
    ggml_backend_buffer_t buf = nullptr;
    if (buft != nullptr) {
        buf = ggml_backend_buft_alloc_buffer(buft, cap);
    }
    if (buf == nullptr) {
        buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), cap);
    }
    if (buf == nullptr) {
        return false;
    }
    s.buf  = buf;
    s.base = (uint8_t *) ggml_backend_buffer_get_base(buf);
    s.cap  = cap;
    s.free_blocks.clear();
    s.free_blocks[0] = cap;
    return true;
}

void server_state_pool::prealloc(size_t bytes) {
    const int64_t t0 = ggml_time_us();
    size_t n = 0;
    while (capacity() < bytes) {
        slab s;
        if (!new_slab(slab_size, s)) {
            break;
        }
        slabs.push_back(std::move(s));
        n++;
    }
    SRV_INF("prompt cache: %zu state slabs of %.0f MiB allocated up front in %.0f ms\n", n, slab_size / (1024.0*1024.0), (ggml_time_us() - t0) / 1e3);
}

// a slab is pinned on a thread while the pool still has room, so a save seldom waits for cudaMallocHost (0.4 s per 512 MiB)
void server_state_pool::prefetch() {
    static const bool off = getenv("LLAMA_CACHE_SLAB_PREFETCH") != nullptr && atoi(getenv("LLAMA_CACHE_SLAB_PREFETCH")) == 0;
    const size_t cmax = max_capacity();
    if (off || (limit > 0 && capacity() >= limit) || (cmax > 0 && capacity() + slab_size > cmax)) {
        return;
    }
    size_t largest = 0;
    for (const auto & s : slabs) {
        for (const auto & [_, sz] : s.free_blocks) {
            largest = std::max(largest, sz);
        }
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        if (prefetching || !pending.empty() || largest >= slab_size / 2) {
            return;
        }
        prefetching = true;
    }
    if (th.joinable()) {
        th.join();
    }
    th = std::thread([this]() {
        const int64_t t0 = ggml_time_us();
        slab s;
        const bool ok = new_slab(slab_size, s);
        std::lock_guard<std::mutex> lk(mtx);
        if (ok) {
            pending.push_back(std::move(s));
        }
        prefetching = false;
        SRV_DBG("prompt cache: a %.0f MiB state slab pinned ahead in %.0f ms (%s)\n", slab_size / (1024.0*1024.0), (ggml_time_us() - t0) / 1e3, ok ? "ok" : "failed");
    });
}

// joins the slabs pinned ahead (pending) to the pool; mtx must NOT be held by the caller
void server_state_pool::adopt_pending() {
    std::lock_guard<std::mutex> lk(mtx);
    for (auto & s : pending) {
        slabs.push_back(std::move(s));
    }
    pending.clear();
}

uint8_t * server_state_pool::fit(size_t n, int32_t & slab_out) {
    for (size_t si = 0; si < slabs.size(); ++si) {
        auto & s = slabs[si];
        for (auto it = s.free_blocks.begin(); it != s.free_blocks.end(); ++it) {
            if (it->second >= n) {
                const size_t off  = it->first;
                const size_t rest = it->second - n;
                s.free_blocks.erase(it);
                if (rest > 0) {
                    s.free_blocks[off + n] = rest;
                }
                n_used  += n;
                slab_out = (int32_t) si;
                return s.base + off;
            }
        }
    }
    return nullptr;
}

size_t server_state_pool::max_capacity() const {
    if (cap_max > 0) {
        return cap_max;
    }
    // the cache's limit plus two slabs of fragmentation slack (0 = no limit, no cap)
    return limit > 0 ? limit + 2*slab_size : 0;
}

// HC3: no path loops or recurses. The pre-HC3 code re-entered alloc() while holding mtx after joining a prefetch whose
// slab was still pending; the inner call locked mtx again and the main thread hung for good (gpu-50, 2026-11-12: a save
// after a 32K-token answer needed a block bigger than any free one while the prefetch of the previous allocation ran)
uint8_t * server_state_pool::alloc(size_t n, int32_t & slab_out) {
    n = round(n);
    slab_out = -1;
    cap_refused = false;
    if (n == 0) {
        return nullptr;
    }

    // slabs pinned ahead join the pool
    adopt_pending();

    // first fit over the slabs
    if (uint8_t * q = fit(n, slab_out)) {
        prefetch();
        return q;
    }

    // a prefetch in flight is waited for (its slab may fit), then first fit once more
    if (th.joinable()) {
        th.join();
        adopt_pending();
        if (uint8_t * q = fit(n, slab_out)) {
            prefetch();
            return q;
        }
    }

    // a new slab (pinned through the device's host buffer type, plain when there is none or it fails), unless the pool
    // would grow past its cap: free blocks too small for n (fragmentation) must not grow the pinned memory without bound
    const size_t cap = std::max(slab_size, n);
    const size_t cmax = max_capacity();
    if (cmax > 0 && capacity() + cap > cmax) {
        cap_refused = true;
        SRV_WRN("prompt cache: no free block of %.1f MiB and a new %.1f MiB slab would grow the pool past %.1f MiB (%zu slabs, %.1f MiB, %.1f MiB in use)\n",
                n / (1024.0*1024.0), cap / (1024.0*1024.0), cmax / (1024.0*1024.0), slabs.size(), capacity() / (1024.0*1024.0), n_used / (1024.0*1024.0));
        return nullptr;
    }

    const int64_t t0 = ggml_time_us();
    slab s;
    if (!new_slab(cap, s)) {
        return nullptr;
    }
    s.free_blocks.clear();
    if (cap > n) {
        s.free_blocks[n] = cap - n;
    }
    slabs.push_back(std::move(s));
    n_used  += n;
    slab_out = (int32_t) slabs.size() - 1;

    SRV_WRN("prompt cache: allocated a %.1f MiB state slab in %.0f ms (%zu slabs, %.1f MiB, %.1f MiB in use)\n",
            cap / (1024.0*1024.0), (ggml_time_us() - t0) / 1e3, slabs.size(), capacity() / (1024.0*1024.0), n_used / (1024.0*1024.0));

    prefetch();
    return slabs.back().base;
}

void server_state_pool::free(int32_t slab, uint8_t * p, size_t n) {
    n = round(n);
    if (slab < 0 || p == nullptr || n == 0) {
        return;
    }
    GGML_ASSERT((size_t) slab < slabs.size());
    auto & s = slabs[slab];
    GGML_ASSERT(p >= s.base && p + n <= s.base + s.cap);

    size_t off = p - s.base;
    n_used -= n;

    // coalesce with the neighbours
    auto next = s.free_blocks.lower_bound(off);
    if (next != s.free_blocks.end() && next->first == off + n) {
        n += next->second;
        s.free_blocks.erase(next);
    }
    if (!s.free_blocks.empty()) {
        auto prev = s.free_blocks.lower_bound(off);
        if (prev != s.free_blocks.begin()) {
            --prev;
            if (prev->first + prev->second == off) {
                off = prev->first;
                n  += prev->second;
                s.free_blocks.erase(prev);
            }
        }
    }
    s.free_blocks[off] = n;
}

bool server_state_buf::alloc(server_state_pool & pl, size_t c) {
    if (c <= cap && pool == &pl) {
        n = c;
        return true;
    }
    release();
    if (c == 0) {
        return true;
    }
    int32_t si = -1;
    uint8_t * q = pl.alloc(c, si);
    if (q == nullptr) {
        return false;
    }
    pool = &pl;
    slab = si;
    p    = q;
    cap  = c;
    n    = c;
    return true;
}

void server_state_buf::release() {
    if (pool != nullptr && p != nullptr) {
        pool->free(slab, p, cap);
    }
    pool = nullptr;
    slab = -1;
    p    = nullptr;
    n    = cap = 0;
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::evictable() {
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (it->owner < 0) {
            return it;
        }
    }
    return states.end();
}

void server_prompt_cache::release(server_prompt_cache_state * e) {
    if (e != nullptr) {
        e->owner = -1;
    }
}

// HC1: LRU order (LLAMA_CACHE_LRU=0: entries keep the position of their first save, the pre-HC1 FIFO eviction)
static bool cache_lru_on() {
    static const bool on = getenv("LLAMA_CACHE_LRU") == nullptr || atoi(getenv("LLAMA_CACHE_LRU")) != 0;
    return on;
}

void server_prompt_cache::touch(server_prompt_cache_state * e) {
    if (e == nullptr || !cache_lru_on()) {
        return;
    }
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (&*it == e) {
            // splice keeps every pointer to the entry valid (slots hold them as their origin)
            states.splice(states.end(), states, it);
            return;
        }
    }
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::best_entry(const server_prompt & prompt, const server_tokens & tokens_new, int32_t id_slot, int * lcp_out) {
    int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        // an entry another slot holds is in use there
        if (it->owner >= 0 && it->owner != id_slot) {
            continue;
        }

        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
            lcp_best    = lcp_cur;

            it_best = it;
        }
    }

    if (lcp_out != nullptr) {
        *lcp_out = lcp_best;
    }
    return it_best;
}

const server_prompt_cache_state * server_prompt_cache::find_better(const server_prompt & prompt, const server_tokens & tokens_new, int32_t id_slot, int min_gain) {
    int lcp_best = 0;
    auto it = best_entry(prompt, tokens_new, id_slot, &lcp_best);
    if (it == states.end()) {
        return nullptr;
    }
    const int lcp_slot = prompt.tokens.get_common_prefix(tokens_new);
    return lcp_best - lcp_slot >= min_gain ? &*it : nullptr;
}

size_t server_prompt_cache::trim_checkpoints_one() {
    static const int keep = getenv("LLAMA_CACHE_TRIM_CKPTS") != nullptr ? atoi(getenv("LLAMA_CACHE_TRIM_CKPTS")) : 3;
    if (keep < 0) {
        return 0;
    }
    for (auto & st : states) {
        if (st.owner >= 0 || st.prompt.checkpoints.size() <= (size_t) keep) {
            continue;
        }
        size_t freed = 0;
        const size_t n0 = st.prompt.checkpoints.size();
        // the newest checkpoints (largest n_tokens) are the restore points of the conversation's next turn (a
        // re-rendered last answer); the older ones only serve an edit far back in the conversation
        while (st.prompt.checkpoints.size() > (size_t) keep) {
            auto victim = st.prompt.checkpoints.begin();
            for (auto it = st.prompt.checkpoints.begin(); it != st.prompt.checkpoints.end(); ++it) {
                if (it->n_tokens < victim->n_tokens) {
                    victim = it;
                }
            }
            freed += victim->size();
            st.prompt.checkpoints.erase(victim);
        }
        SRV_WRN(" - prompt cache: entry of %d tokens trimmed from %zu to %zu checkpoints (%.3f MiB freed)\n",
                st.prompt.n_tokens(), n0, st.prompt.checkpoints.size(), freed / (1024.0 * 1024.0));
        if (freed > 0) {
            return freed;
        }
    }
    return 0;
}

void server_prompt_cache::make_room(size_t need, const char * why) {
    if (limit_size == 0) {
        return;
    }
    while (size() + need > limit_size) {
        if (trim_checkpoints_one() > 0) {
            continue;
        }
        auto it = evictable();
        if (it == states.end()) {
            break;
        }
        SRV_WRN(" - %s, removing oldest entry (size = %.3f MiB, %d tokens)\n", why, it->size() / (1024.0 * 1024.0), it->prompt.n_tokens());
        states.erase(it);
    }
}

// LLAMA_CACHE_DELTA=0: every save is a new full entry (measurement switch)
static bool cache_delta_on() {
    static const bool on = getenv("LLAMA_CACHE_DELTA") == nullptr || atoi(getenv("LLAMA_CACHE_DELTA")) != 0;
    return on;
}

server_prompt_cache_state * server_prompt_cache::save(server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, int32_t id_slot, server_prompt_cache_state ** origin, bool move_checkpoints) {
    const int64_t t0 = ggml_time_us();
    const int32_t n = (int32_t) prompt.tokens.size();

    // a segment holds at most LLAMA_CACHE_SEG cells; a tail shorter than LLAMA_CACHE_SEG_MIN is re-saved with the next
    // delta instead of starting a new segment (a conversation of a hundred small turns stays a few dozen segments)
    static const int32_t seg_max = getenv("LLAMA_CACHE_SEG")     != nullptr ? std::max(1024, atoi(getenv("LLAMA_CACHE_SEG")))    : 32768;
    static const int32_t seg_min = getenv("LLAMA_CACHE_SEG_MIN") != nullptr ? std::max(0,    atoi(getenv("LLAMA_CACHE_SEG_MIN"))) : 4096;

    server_prompt_cache_state * e = (*origin != nullptr && (*origin)->owner == id_slot) ? *origin : nullptr;
    if (!cache_delta_on() && e != nullptr) {
        // the old behaviour: the previous entry of the conversation goes, a full one is made
        for (auto it = states.begin(); it != states.end(); ++it) {
            if (&*it == e) {
                states.erase(it);
                break;
            }
        }
        e = nullptr;
        *origin = nullptr;
    }
    if (e == nullptr) {
        *origin = nullptr;
    }

    int lcp = e != nullptr ? e->prompt.tokens.get_common_prefix(prompt.tokens) : 0;

    // CL1: the slot's origin entry is the conversation the slot came from; when the slot has since moved on to ANOTHER
    // conversation (prefix reuse without a cache update), the prompt shares only a prefix with the entry and the entry's
    // tail is that other conversation. Saving over it would destroy its cached state (the next request of that
    // conversation re-prefills everything). A stale tail longer than LLAMA_CACHE_FORK_TOL tokens (default 256, more than
    // a re-tokenized answer tail) therefore means a different conversation: the entry stays, this prompt gets its own.
    static const int fork_tol = getenv("LLAMA_CACHE_FORK_TOL") != nullptr ? std::max(0, atoi(getenv("LLAMA_CACHE_FORK_TOL"))) : 256;
    if (e != nullptr && e->prompt.n_tokens() - lcp > fork_tol) {
        SRV_DBG("prompt cache: slot %d holds another conversation than its entry (%d of %d tokens shared): the entry stays, a new one is made\n",
                id_slot, lcp, e->prompt.n_tokens());
        e->owner = -1;
        e = nullptr;
        *origin = nullptr;
        lcp = 0;
    }

    // nothing new since the entry was saved: the checkpoints go back in, that is all
    if (e != nullptr && lcp == n && e->prompt.n_tokens() == n) {
        if (move_checkpoints) {
            e->prompt.checkpoints = std::move(prompt.checkpoints);
            prompt.checkpoints.clear();
        } else {
            e->prompt.checkpoints = prompt.checkpoints;
        }
        SRV_DBG("prompt cache: entry of slot %d unchanged (%d tokens), checkpoints back\n", id_slot, n);
        touch(e);
        return e;
    }

    // first check if the current state is contained fully in another entry of the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (&*it == e) {
            continue;
        }
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == n) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            if (e != nullptr) {
                e->owner = -1;
                *origin = nullptr;
            }
            return nullptr;
        }
    }

    // VS1: segments are ranges of POSITIONS (the state API's unit); n and lcp count tokens. They are the same for text,
    // but an M-RoPE image of n_tokens cells takes only max(nx, ny) positions: the boundaries below are positions
    const llama_pos n_pos   = prompt.tokens.pos_next();
    const llama_pos lcp_pos = prompt.tokens.pos_next(lcp);

    // the entry's segments that still hold valid cells: [p0, min(p1, lcp)); a segment with too many cells past the
    // common prefix, or a short tail, is re-saved with the new cells
    size_t n_keep = 0;
    // a segment's cells end where the next segment starts (the last one: at the entry's prompt end); its p1 is open-ended
    // (INT32_MAX) for every prompt shorter than a chunk, so the rules below compare against this effective end, never p1
    // (CL1: comparing the drafter's end d1 against an open-ended p1 dropped every later segment and left their cells in no
    // segment: the restored entry then had a hole, which leaked another conversation's cells into the prompt)
    auto seg_end = [&](size_t i) -> llama_pos {
        const auto & segs = e->data.segs;
        const llama_pos n_prev = e->prompt.tokens.pos_next();
        return std::min<llama_pos>(segs[i].p1, i + 1 < segs.size() ? segs[i + 1].p0 : n_prev);
    };
    if (e != nullptr) {
        for (const auto & seg : e->data.segs) {
            if (seg.p0 >= lcp_pos) {
                break;
            }
            n_keep++;
        }
        if (n_keep > 0) {
            const auto & last = e->data.segs[n_keep - 1];
            const llama_pos end = seg_end(n_keep - 1);
            const int64_t n_seg   = (int64_t) end - last.p0;
            const int64_t n_stale = (int64_t) end - lcp_pos;
            if (n_stale > std::max<int64_t>(seg_min, n_seg / 4)) {
                n_keep--;
            } else if (end <= lcp_pos && n_seg < seg_min && n_keep > 1) {
                n_keep--;
            }
        }
        // nothing of the entry survives (the slot kept its cells, the new prompt shares only a short prefix: the
        // other conversation of a --parallel 1 node): the entry stays as it is, this prompt gets one of its own
        if (n_keep == 0) {
            e->owner = -1;
            e = nullptr;
            *origin = nullptr;
        }
    }
    // M3: the drafter's cells of the last kept segment end where its drafter part ended (d1, it lags the target): a kept
    // segment whose drafter part would not reach its own start goes too (a drafter that lagged a whole segment)
    if (e != nullptr && ctx_dft != nullptr) {
        // an earlier kept segment whose drafter part has a hole becomes the last kept one (the delta re-saves from there).
        // HC2: a hole is a drafter cell in NO segment: this segment's drafter part ends before the next one's starts.
        // A delta save starts the next drafter part where this one ended (d0 = the kept d1, the MTP drafter lags the
        // target by a token or more), so d1 < seg_end(i) alone is no hole: comparing against seg_end re-saved every
        // segment after the first on every switch (O(conversation) bytes per save: 80-100 MiB at 12K, measured)
        static const bool hole_old = getenv("LLAMA_CACHE_HOLE_OLD") != nullptr && atoi(getenv("LLAMA_CACHE_HOLE_OLD")) != 0;
        for (size_t i = 0; i + 1 < n_keep; ++i) {
            const auto & seg = e->data.segs[i];
            if (seg.d1 < (hole_old ? seg_end(i) : e->data.segs[i + 1].d0)) {
                n_keep = i + 1;
                break;
            }
        }
        while (n_keep > 0) {
            const auto & last = e->data.segs[n_keep - 1];
            const llama_pos d_end = std::min<llama_pos>({seg_end(n_keep - 1), lcp_pos, last.d1});
            if (d_end > last.d0) {
                break;
            }
            n_keep--;
        }
        if (n_keep == 0) {
            e->owner = -1;
            e = nullptr;
            *origin = nullptr;
        }
    }
    // the kept cells end at the last kept segment's effective end: the new segments start there, never past it
    const llama_pos kept_end = (e != nullptr && n_keep > 0) ? seg_end(n_keep - 1) : 0;
    std::vector<server_state_seg> kept;
    if (e != nullptr) {
        for (size_t i = 0; i < n_keep; ++i) {
            kept.push_back(std::move(e->data.segs[i]));
        }
        e->data.segs.clear();
    }
    const llama_pos keep_end = kept.empty() ? 0 : std::min<llama_pos>(kept_end, lcp_pos);
    // where the new drafter part starts: the kept drafter cells reach only d1 of the last kept segment
    const llama_pos keep_end_dft = kept.empty() ? 0 : std::min<llama_pos>(keep_end, kept.back().d1);

    // the new segments: [keep_end, n) in chunks of seg_max, the last one open-ended (cells past n go with it)
    struct chunk { llama_pos a, b, ad; size_t sz_tgt, sz_dft; };
    std::vector<chunk> chunks;
    for (llama_pos a = keep_end; a < n_pos; a += seg_max) {
        const llama_pos b = (a + seg_max >= n_pos) ? INT32_MAX : a + seg_max;
        chunks.push_back({a, b, a, 0, 0});
    }
    if (chunks.empty()) {
        // n_pos == keep_end: the tail was trimmed only; one open-ended chunk for whatever cells sit past it
        chunks.push_back({keep_end, INT32_MAX, keep_end, 0, 0});
    }
    chunks.front().ad = keep_end_dft;

    size_t state_size_new = 0;
    for (auto & c : chunks) {
        c.sz_tgt = llama_state_seq_get_size_range(ctx_tgt, id_slot, c.a, c.b);
        c.sz_dft = ctx_dft ? llama_state_seq_get_size_range(ctx_dft, id_slot, c.ad, c.b) : 0;
        state_size_new += c.sz_tgt + c.sz_dft;
    }
    // the drafter's cells now end here (its last position + 1)
    // (a drafter without cells has nothing to lose: no limit then)
    const llama_pos dft_pm  = ctx_dft ? llama_memory_seq_pos_max(llama_get_memory(ctx_dft), id_slot) : -1;
    const size_t sz_recr = llama_state_seq_get_size_ext(ctx_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    state_size_new += sz_recr;

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }
    state_size_new += checkpoints_size;

    size_t size_kept = 0;
    for (const auto & seg : kept) {
        size_kept += seg.size();
    }

    const int64_t t1 = ggml_time_us();

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new + size_kept > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                (state_size_new + size_kept) / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        if (e != nullptr) {
            e->data.segs = std::move(kept);
            e->owner = -1;
            *origin = nullptr;
        }
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (&*it != e && it->owner < 0 && len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    // make room before allocating the new buffers to avoid breaching the limit (HC1: older checkpoints of unowned
    // entries go first, then whole entries, least recently used first)
    // (HC3: the kept segments are out of the entry while it is rebuilt: size() misses them, so they count as needed)
    make_room(state_size_new + size_kept, "making room for prompt cache entry");

    if (e == nullptr) {
        states.push_back({});
        e = &states.back();
    }
    e->owner = id_slot;
    touch(e);
    e->data.segs = std::move(kept);

    auto fail = [&]() {
        for (auto it = states.begin(); it != states.end(); ++it) {
            if (&*it == e) {
                states.erase(it);
                break;
            }
        }
        *origin = nullptr;

        // HC3: the pool refused to grow past its cap (no free block fits, nothing left to evict): this entry is skipped,
        // the cache and its limit stay as they are and serving goes on
        if (pool.cap_refused) {
            SRV_WRN("prompt cache: the state of slot %d (%d tokens, %.1f MiB) does not fit the pool, entry skipped\n",
                    id_slot, n, (state_size_new + size_kept) / (1024.0 * 1024.0));
            pool.cap_refused = false;
            return (server_prompt_cache_state *) nullptr;
        }

        SRV_ERR("%s", "failed to allocate memory for prompt cache state\n");

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return (server_prompt_cache_state *) nullptr;
    };

    // a buffer from the pool; when the pool refuses to grow (fragmentation past its cap) the least recently used unowned
    // entry goes and the allocation is retried: every round removes an entry, so this ends (HC3)
    auto alloc_buf = [&](server_state_buf & b, size_t sz) -> bool {
        while (!b.alloc(pool, sz)) {
            if (!pool.cap_refused) {
                return false;
            }
            auto it = evictable();
            if (it == states.end()) {
                return false;
            }
            SRV_WRN(" - prompt cache: pool full or fragmented, removing oldest entry (size = %.3f MiB, %d tokens)\n",
                    it->size() / (1024.0 * 1024.0), it->prompt.n_tokens());
            states.erase(it);
        }
        return true;
    };

    const int64_t t2 = ggml_time_us();
    size_t n_saved = 0;
    for (const auto & c : chunks) {
        server_state_seg seg;
        seg.p0 = c.a;
        seg.p1 = c.b;
        seg.d0 = c.ad;
        // the drafter's cells of this chunk end at its first hole (a deferred catch-up can leave one mid-sequence that a
        // later turn fills: the next delta must re-save the drafter from there)
        if (dft_pm < 0) {
            seg.d1 = INT32_MAX;
        } else {
            const llama_pos gap = server_tokens_pos_gap(ctx_dft, id_slot, c.ad, prompt.tokens);
            seg.d1 = gap < 0 ? c.b : std::min<llama_pos>(c.b, gap);
        }
        if (!alloc_buf(seg.main, c.sz_tgt) || !alloc_buf(seg.drft, c.sz_dft)) {
            return fail();
        }
        if (c.sz_tgt > 0 && llama_state_seq_get_data_range(ctx_tgt, seg.main.data(), c.sz_tgt, id_slot, c.a, c.b) != c.sz_tgt) {
            SRV_ERR("%s", "failed to save the target state\n");
            return fail();
        }
        if (c.sz_dft > 0 && llama_state_seq_get_data_range(ctx_dft, seg.drft.data(), c.sz_dft, id_slot, c.ad, c.b) != c.sz_dft) {
            SRV_ERR("%s", "failed to save the draft state\n");
            return fail();
        }
        n_saved += seg.size();
        e->data.segs.push_back(std::move(seg));
    }
    const int64_t t3 = ggml_time_us();

    if (!alloc_buf(e->data.recr, sz_recr)) {
        return fail();
    }
    if (sz_recr > 0 && llama_state_seq_get_data_ext(ctx_tgt, e->data.recr.data(), sz_recr, id_slot, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != sz_recr) {
        SRV_ERR("%s", "failed to save the recurrent state\n");
        return fail();
    }
    const int64_t t4 = ggml_time_us();

    // the drafter's deferred boundary, if it keeps one (restored together with the contexts)
    e->data.spec.clear();
    if (spec) {
        common_speculative_get_state(spec, id_slot, e->data.spec);
    }

    e->prompt.tokens = prompt.tokens.clone();
    if (move_checkpoints) {
        e->prompt.checkpoints = std::move(prompt.checkpoints);
        prompt.checkpoints.clear();
    } else {
        e->prompt.checkpoints = prompt.checkpoints;
    }
    *origin = e;

    if (srv_cache_timing()) {
        SRV_WRN("timing cache save: slot %d n_tokens %d lcp %d kept %zu segs (to %d) new %zu segs: sizes %.1f alloc %.1f segs %.1f (%.1f MiB) recr %.1f (%.1f MiB) rest %.1f ms (cache %zu prompts, %.1f MiB, pool %.1f MiB)\n",
                id_slot, n, lcp, e->data.segs.size() - chunks.size(), keep_end, chunks.size(), (t1 - t0) / 1e3, (t2 - t1) / 1e3,
                (t3 - t2) / 1e3, n_saved / (1024.0 * 1024.0), (t4 - t3) / 1e3, sz_recr / (1024.0 * 1024.0), (ggml_time_us() - t4) / 1e3,
                states.size(), size() / (1024.0 * 1024.0), pool.capacity() / (1024.0 * 1024.0));
    }

    return e;
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, int32_t id_slot, server_prompt_cache_state * saved, server_prompt_cache_state ** origin) {
    const int64_t t_load = ggml_time_us();
    int lcp_best = 0;
    auto it_best = best_entry(prompt, tokens_new, id_slot, &lcp_best);

    if (it_best == states.end()) {
        if (saved != nullptr) {
            // the slot keeps its prompt and cells: the checkpoints save() moved into the entry come back (the prompt
            // reuse erases the ones past the common prefix; a recurrent model needs the earlier ones); the entry stays
            // the slot's, so the next save is a delta again
            prompt.checkpoints = std::move(saved->prompt.checkpoints);
            saved->prompt.checkpoints.clear();
            saved->owner = id_slot;
            *origin = saved;
            touch(saved);
            SRV_DBG("prompt cache: the slot keeps its prompt (%d of %d tokens shared), %zu checkpoints back\n",
                    lcp_best, (int) prompt.tokens.size(), prompt.checkpoints.size());
        }
        return true;
    }

    auto & e = *it_best;
    SRV_TRC(" - found better prompt with lcp = %d\n", lcp_best);

    SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_RESTORE__\n");

    // the whole cached prompt is a prefix of the new one
    if (lcp_best == (int) it_best->prompt.tokens.size()) {
        SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_FULL_RESTORE__\n");
    }

    auto fail = [&](const char * what) {
        SRV_ERR("failed to restore the cached prompt: %s\n", what);
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
        }
        if (saved != nullptr && saved != &e) {
            saved->owner = -1;
        }

        states.erase(it_best);
        *origin = nullptr;
        return false;
    };

    const int64_t t_set = ggml_time_us();

    // the slot's cells go (the ones of a just-saved prompt stay as ghost cells for its next restore), the segments
    // come in order: a segment's cells past the next one's start are the stale tail of an earlier turn
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
    if (ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
    }

    const int32_t n_e = e.prompt.tokens.pos_next(); // positions (VS1: an image's tokens take fewer positions)
    for (size_t i = 0; i < e.data.segs.size(); ++i) {
        auto & seg = e.data.segs[i];

        if (!seg.main.empty() && llama_state_seq_set_data_range(ctx_tgt, seg.main.data(), seg.main.size(), id_slot) != seg.main.size()) {
            return fail("target segment");
        }
        if (!seg.drft.empty()) {
            GGML_ASSERT(ctx_dft);
            if (llama_state_seq_set_data_range(ctx_dft, seg.drft.data(), seg.drft.size(), id_slot) != seg.drft.size()) {
                return fail("draft segment");
            }
        }

        const bool has_next = i + 1 < e.data.segs.size();
        const llama_pos next_p0 = has_next ? e.data.segs[i + 1].p0 : n_e;
        const llama_pos next_d0 = has_next ? e.data.segs[i + 1].d0 : n_e;
        if (seg.p1 > next_p0) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, next_p0, -1);
        }
        // the drafter's part of the next segment may start earlier (d0 <= p0: the cells this one lacked)
        if (ctx_dft && seg.p1 > next_d0) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, next_d0, -1);
        }
    }
    // CL1: the restored segments must cover the whole prompt; an entry that lost earlier segments must not be reused
    {
        const llama_pos gap = server_tokens_pos_gap(ctx_tgt, id_slot, 0, e.prompt.tokens);
        if (gap >= 0 && gap < n_e) {
            SRV_WRN("prompt cache: entry of %d tokens restores only up to pos %d (%zu segments): dropped\n", n_e, gap, e.data.segs.size());
            return fail("segments do not cover the prompt");
        }
    }
    const int64_t t_recr = ggml_time_us();

    if (!e.data.recr.empty() && llama_state_seq_set_data_ext(ctx_tgt, e.data.recr.data(), e.data.recr.size(), id_slot, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != e.data.recr.size()) {
        return fail("recurrent state");
    }

    // the drafter's state belongs to the restored prompt too; an empty state clears what the previous prompt left
    if (spec) {
        common_speculative_set_state(spec, id_slot, e.data.spec);
    }

    prompt.tokens      = e.prompt.tokens.clone();
    prompt.checkpoints = std::move(e.prompt.checkpoints);
    e.prompt.checkpoints.clear();

    if (saved != nullptr && saved != &e) {
        saved->owner = -1;
    }
    e.owner = id_slot;
    *origin = &e;
    touch(&e);

    if (srv_cache_timing()) {
        SRV_WRN("timing cache load: slot %d n_tokens %d: lcp search %.1f segs %zu %.1f recr %.1f ms (%.1f MiB)\n", id_slot, n_e,
                (t_set - t_load) / 1e3, e.data.segs.size(), (t_recr - t_set) / 1e3, (ggml_time_us() - t_recr) / 1e3, e.size() / (1024.0 * 1024.0));
    }

    return true;
}

void server_prompt_cache::update() {
    make_room(0, "cache size limit reached");

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (n_tokens() > limit_tokens_cur) {
            auto it = evictable();
            if (it == states.end()) {
                break;
            }
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, it->size() / (1024.0 * 1024.0));

            states.erase(it);
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}
