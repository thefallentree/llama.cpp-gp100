#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    uint32_t n_graph_slots;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    // pipelined prefill over two nodes (llama_set_layer_range): the graph runs layers [layer_begin, layer_end) only
    // (layer_end < 0: to the last); from layer_begin > 0 the residual comes in through llama_set_hc_input, and up to
    // layer_end < n_layer the residual after layer_end - 1 goes out as the nextn embeddings (no head, no logits)
    int32_t  layer_begin = 0;
    int32_t  layer_end   = -1;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool fused_top_k_mask;   // use fused top-k sparse attention mask
    bool auto_ftkm;
    bool fused_kv_rows;      // unified cache: attend to the cells of each sequence only (ggml_flash_attn_ext_rows)
    bool auto_fkvr;
    bool no_perf;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;
    bool training;           // set by llama_opt_init()

    size_t moe_cache_size;
    float  moe_cache_cpu_share;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer
    bool embeddings_layer_inp_dev; // write enabled layer inputs to a device buffer instead of host
    bool embeddings_nextn_dev;     // write the unmasked h_nextn rows to a device buffer instead of host, see llama_set_embeddings_nextn_dev()

    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;

    // context whose h_nextn device buffer holds the state rows of a batch, see llama_set_ctx_state_src()
    llama_context * ctx_state_src;
};
