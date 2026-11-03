#pragma once

// Pipelined prefill over two nodes (qwen4exp).
//
// A long cold prompt is split by layers between this server (the head: layers [0, split)) and a peer llama-server
// started with the same model and arguments plus LLAMA_PREFILL_PEER_LISTEN=port (layers [split, n_layer) and the MTP
// drafter's catch-up). The head sends the residual after layer split - 1 of every ubatch (n_embd_out floats per token)
// as soon as it is computed and goes on with the next ubatch, so the two nodes work on consecutive ubatches at the same
// time. At the end the peer sends its layers' part of the sequence state (KV, indexer, recurrent rows), the drafter's
// state and the drafter's pending row; the head writes them into its own memories and decodes the rest of the prompt
// as usual.
//
// head:  LLAMA_PREFILL_PEER=host:port, LLAMA_PREFILL_PEER_SPLIT=layer (default n_layer/2),
//        LLAMA_PREFILL_PEER_MIN=tokens (default 8192: shorter prompts stay local); several peers form a chain
//        (LLAMA_PREFILL_PEER=a:p,b:p LLAMA_PREFILL_PEER_SPLIT=16,32): each forwards its residual to the next
// peer:  LLAMA_PREFILL_PEER_LISTEN=port (the server runs only this loop, no tasks)

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct common_speculative;

struct prefill_peer_stats {
    double t_total_ms   = 0.0; // the whole split part on the head
    double t_local_ms   = 0.0; // the head's own layers (compute + h_nextn copy)
    double t_wait_ms    = 0.0; // waiting for the sender and for the peer's result at the end
    double t_restore_ms = 0.0; // writing the peer's state into the memories
    double peer_ms      = 0.0; // the peer's compute time as it reports it
    size_t bytes_state  = 0;
};

// one peer of the chain: it runs the layers from layer_begin to the next peer's layer_begin (the last one: to the end)
struct prefill_peer_stage {
    std::string host;
    int         port        = 0;
    int         layer_begin = 0;
};

// the head's configuration from the environment; enabled() is false without LLAMA_PREFILL_PEER
// LLAMA_PREFILL_PEER=host:port[,host:port...]: a chain of peers, the residual goes from each to the next;
// LLAMA_PREFILL_PEER_SPLIT=l1[,l2...]: the first layer of each peer (default: the layers split evenly)
struct prefill_peer_config {
    std::vector<prefill_peer_stage> stages;
    int min_tokens = 8192;

    static prefill_peer_config from_env(const llama_model * model);

    bool enabled() const { return !stages.empty(); }
    int  split()   const { return stages.empty() ? -1 : stages[0].layer_begin; }
};

// The head: decodes tokens [0, n_split) of seq_id with the peer. The memories of seq_id must be empty on entry and the
// tokens plain text. On success the target and drafter memories hold the sequence up to n_split as if it had been
// decoded here; on failure seq_id is removed from both memories again and the caller decodes the prompt locally.
bool prefill_peer_run(const prefill_peer_config & cfg,
                      llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec,
                      llama_seq_id seq_id, const llama_token * tokens, int32_t n_split, int32_t n_ubatch,
                      prefill_peer_stats & stats);

// The peer: serves heads one connection at a time, forever (returns only on a socket setup error).
void prefill_peer_serve(int port, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec);
