#include "llama-gguf-compat.h"

#include "llama-hparams.h"
#include "llama-impl.h"

#include "gguf.h"

#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

static size_t kv_scalar_size(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default:                return 0;
    }
}

// copies a pair to another key, the source stays
static void kv_copy(gguf_context * meta, const std::string & from, const std::string & to) {
    const int64_t id = gguf_find_key(meta, from.c_str());
    if (id < 0) {
        throw std::runtime_error(format("key not found in model: %s", from.c_str()));
    }

    const gguf_type type = gguf_get_kv_type(meta, id);
    switch (type) {
        case GGUF_TYPE_UINT8:   gguf_set_val_u8  (meta, to.c_str(), gguf_get_val_u8  (meta, id)); break;
        case GGUF_TYPE_INT8:    gguf_set_val_i8  (meta, to.c_str(), gguf_get_val_i8  (meta, id)); break;
        case GGUF_TYPE_UINT16:  gguf_set_val_u16 (meta, to.c_str(), gguf_get_val_u16 (meta, id)); break;
        case GGUF_TYPE_INT16:   gguf_set_val_i16 (meta, to.c_str(), gguf_get_val_i16 (meta, id)); break;
        case GGUF_TYPE_UINT32:  gguf_set_val_u32 (meta, to.c_str(), gguf_get_val_u32 (meta, id)); break;
        case GGUF_TYPE_INT32:   gguf_set_val_i32 (meta, to.c_str(), gguf_get_val_i32 (meta, id)); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (meta, to.c_str(), gguf_get_val_f32 (meta, id)); break;
        case GGUF_TYPE_UINT64:  gguf_set_val_u64 (meta, to.c_str(), gguf_get_val_u64 (meta, id)); break;
        case GGUF_TYPE_INT64:   gguf_set_val_i64 (meta, to.c_str(), gguf_get_val_i64 (meta, id)); break;
        case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (meta, to.c_str(), gguf_get_val_f64 (meta, id)); break;
        case GGUF_TYPE_BOOL:    gguf_set_val_bool(meta, to.c_str(), gguf_get_val_bool(meta, id)); break;
        case GGUF_TYPE_STRING:  {
            const std::string val = gguf_get_val_str(meta, id);
            gguf_set_val_str(meta, to.c_str(), val.c_str());
        } break;
        case GGUF_TYPE_ARRAY: {
            const gguf_type type_arr = gguf_get_arr_type(meta, id);
            const size_t    n        = gguf_get_arr_n(meta, id);
            if (type_arr == GGUF_TYPE_STRING) {
                std::vector<std::string> vals(n);
                std::vector<const char *> ptrs(n);
                for (size_t i = 0; i < n; ++i) {
                    vals[i] = gguf_get_arr_str(meta, id, i);
                    ptrs[i] = vals[i].c_str();
                }
                gguf_set_arr_str(meta, to.c_str(), ptrs.data(), n);
            } else {
                const size_t size = kv_scalar_size(type_arr);
                if (size == 0) {
                    throw std::runtime_error(format("key %s: unsupported array type", from.c_str()));
                }
                const char * data = (const char *) gguf_get_arr_data(meta, id);
                const std::vector<char> vals(data, data + n*size);
                gguf_set_arr_data(meta, to.c_str(), type_arr, vals.data(), n);
            }
        } break;
        default:
            throw std::runtime_error(format("key %s: unsupported type", from.c_str()));
    }
}

static bool kv_get_int(const gguf_context * meta, const std::string & key, int64_t & val) {
    const int64_t id = gguf_find_key(meta, key.c_str());
    if (id < 0) {
        return false;
    }

    switch (gguf_get_kv_type(meta, id)) {
        case GGUF_TYPE_UINT8:  val = gguf_get_val_u8  (meta, id); return true;
        case GGUF_TYPE_INT8:   val = gguf_get_val_i8  (meta, id); return true;
        case GGUF_TYPE_UINT16: val = gguf_get_val_u16 (meta, id); return true;
        case GGUF_TYPE_INT16:  val = gguf_get_val_i16 (meta, id); return true;
        case GGUF_TYPE_UINT32: val = gguf_get_val_u32 (meta, id); return true;
        case GGUF_TYPE_INT32:  val = gguf_get_val_i32 (meta, id); return true;
        case GGUF_TYPE_UINT64: val = (int64_t) gguf_get_val_u64(meta, id); return true;
        case GGUF_TYPE_INT64:  val = gguf_get_val_i64 (meta, id); return true;
        case GGUF_TYPE_BOOL:   val = gguf_get_val_bool(meta, id); return true;
        default:               return false;
    }
}

// Lucebox DSpark drafts (huggingface.co/Lucebox/DeepSeek-V4.1-Flash-DSpark-GGUF)
static void apply_lucebox_dflash(gguf_context * meta, llama_gguf_compat & compat) {
    const std::string src = "deepseek41-dflash-draft.";
    const std::string dst = "dflash.";

    std::set<std::string> keys;
    for (int64_t i = 0; i < gguf_get_n_kv(meta); ++i) {
        keys.insert(gguf_get_key(meta, i));
    }

    // only the V4.1 flavour (no output hyper-connection head) maps onto the loader
    int64_t val = 0;
    if (kv_get_int(meta, src + "dflash.head_hc_enabled", val) && val != 0) {
        throw std::runtime_error("draft has an output hyper-connection head (head_hc_enabled), its tensor names are not known");
    }

    // <arch>.<key> -> dflash.<key>, the file nests the DFlash keys as <arch>.dflash.<key>
    std::set<std::string> targets;
    for (const std::string & key : keys) {
        if (key.compare(0, src.size(), src) != 0) {
            continue;
        }

        std::string rest = key.substr(src.size());
        if (rest.compare(0, dst.size(), dst) == 0) {
            rest.erase(0, dst.size());
        }
        const std::string to = dst + (rest == "target_layer_ids" ? "target_layers" : rest);
        if (keys.count(to) > 0 || !targets.insert(to).second) {
            throw std::runtime_error(format("key %s collides with %s", key.c_str(), to.c_str()));
        }

        kv_copy(meta, key, to);
        gguf_remove_key(meta, key.c_str());
    }
    gguf_set_val_str(meta, "general.architecture", "dflash");

    // the file names the MoE scoring function, the loader reads the enum
    {
        const int64_t id = gguf_find_key(meta, "dflash.expert_gating_func");
        if (id >= 0 && gguf_get_kv_type(meta, id) == GGUF_TYPE_STRING) {
            const std::string name = gguf_get_val_str(meta, id);
            uint32_t func = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;
            if      (name == "softmax")      { func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;       }
            else if (name == "sigmoid")      { func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;       }
            else if (name == "sqrtsoftplus") { func = LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS; }
            else {
                throw std::runtime_error(format("unknown MoE scoring function: %s", name.c_str()));
            }
            gguf_set_val_u32(meta, "dflash.expert_gating_func", func);
        }
    }

    if (kv_get_int(meta, "dflash.dspark.confidence.enabled", val)) {
        gguf_set_val_bool(meta, "dflash.has_confidence_head", val != 0);
    }

    int64_t n_layer = 0;
    int64_t n_vocab = 0;
    if (!kv_get_int(meta, "dflash.block_count", n_layer) || n_layer <= 0 || n_layer > LLAMA_MAX_LAYERS ||
        !kv_get_int(meta, "dflash.vocab_size",  n_vocab) || n_vocab <= 0) {
        throw std::runtime_error("draft has no valid block_count or vocab_size");
    }

    // the file has no vocab, the draft shares the target's (the noise token stays in dflash.mask_token_id)
    gguf_set_val_str(meta, "tokenizer.ggml.model", "no_vocab");

    // keys the loader requires and the file leaves out
    std::string assumed;
    if (gguf_find_key(meta, "dflash.context_length") < 0) {
        gguf_set_val_u32(meta, "dflash.context_length", 1048576); // max_position_embeddings of V4.1
        assumed += " context_length=1048576";
    }
    if (gguf_find_key(meta, "dflash.expert_weights_norm") < 0) {
        gguf_set_val_bool(meta, "dflash.expert_weights_norm", true); // norm_topk_prob of V4.1, the draft gate is the same
        assumed += " expert_weights_norm=true";
    }
    if (gguf_find_key(meta, "dflash.attention.compress_ratios") < 0) {
        const std::vector<int32_t> ratios((size_t) n_layer, 0); // every draft layer is sliding-window only
        gguf_set_arr_data(meta, "dflash.attention.compress_ratios", GGUF_TYPE_INT32, ratios.data(), ratios.size());
        assumed += " compress_ratios=0";
    }

    // tensors of the draft proper are called dflash.*, ours have no prefix
    static const char * const renames[][2] = {
        { "dflash.fc.weight",                "fc.weight"             },
        { "dflash.hidden_norm.weight",       "enc.output_norm.weight" },
        { "dflash.dspark.markov.w1",         "markov_w1.weight"      },
        { "dflash.dspark.markov.w2",         "markov_w2.weight"      },
        { "dflash.dspark.confidence.weight", "conf_proj.weight"      },
        { "dflash.dspark.confidence.bias",   "conf_proj.bias"        },
    };
    for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
        const std::string name = gguf_get_tensor_name(meta, i);
        if (name.compare(0, dst.size(), dst) != 0) {
            continue;
        }

        const char * to = nullptr;
        for (const auto & r : renames) {
            if (name == r[0]) {
                to = r[1];
            }
        }
        if (to == nullptr) {
            throw std::runtime_error(format("tensor %s has no counterpart in the loader", name.c_str()));
        }
        if (gguf_find_tensor(meta, to) >= 0) {
            throw std::runtime_error(format("tensor %s collides with %s", name.c_str(), to));
        }
        compat.tensor_rename[name] = to;
    }

    LLAMA_LOG_WARN("%s: Lucebox DFlash draft read as arch dflash: %d keys, %d tensors renamed, assumed:%s\n", __func__,
            (int) targets.size(), (int) compat.tensor_rename.size(), assumed.empty() ? " nothing" : assumed.c_str());
}

bool llama_gguf_compat_apply(gguf_context * meta, llama_gguf_compat & compat) {
    const int64_t id = gguf_find_key(meta, "general.architecture");
    if (id < 0 || gguf_get_kv_type(meta, id) != GGUF_TYPE_STRING) {
        return false;
    }

    const std::string arch = gguf_get_val_str(meta, id);
    if (arch == "deepseek41-dflash-draft") {
        apply_lucebox_dflash(meta, compat);
        compat.dialect = arch;
        return true;
    }

    return false;
}
