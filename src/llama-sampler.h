#pragma once

#include "llama.h"

#include <string>
#include <vector>

// the part of a sampler that shapes the backend graph: two samplers with the same non-empty key build the same
// sampling graph and set the same inputs, so a graph built with one can run with the other (a server's request
// after request with the same settings). "" when the graph depends on the sampler object itself.
std::string llama_sampler_graph_key(const struct llama_sampler * smpl);

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    // llama_sampler_chain_set_preselect_k: on a backend that cannot run the chain (a sharded vocabulary), the
    // backend keeps the top preselect_k logits of every shard as the candidates the whole chain then samples from
    // on the CPU; 0 = the chain runs on the backend as far as it can
    int32_t preselect_k = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
void llama_sampler_backend_begin(llama_sampler * sampler);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
