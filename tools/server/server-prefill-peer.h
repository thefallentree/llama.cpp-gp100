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
// Protocol 4: the tail of the prompt goes through the pipeline too (all but the last 4 tokens); the hello carries the
// position n_ckpt of the context checkpoint the local path makes before its last n_ubatch tokens, every stage snapshots
// its layers' recurrent rows (the last one also the drafter) before the ubatch that starts there and returns the
// snapshot with its result, and the head assembles the checkpoint from them.
//
// Protocol 3: the hello carries the head's result endpoint (an ephemeral listener and a token); every stage returns its
// part of the state straight to the head as soon as it is done (the middle stages while the last one still computes),
// the chain passes only an OK / error status back. The residual rows are sent from the context's double-buffered nextn
// output (llama_set_embeddings_nextn_slot) and received into pinned slots that the graph input reads without a copy.
//
// head:  LLAMA_PREFILL_PEER=host:port, LLAMA_PREFILL_PEER_SPLIT=layer (default n_layer/2),
//        LLAMA_PREFILL_PEER_MIN=tokens (default 8192: shorter prompts stay local); several peers form a chain
//        (LLAMA_PREFILL_PEER=a:p,b:p LLAMA_PREFILL_PEER_SPLIT=16,32): each forwards its residual to the next
// peer:  LLAMA_PREFILL_PEER_LISTEN=port (the server runs only this loop, no tasks)

#include "ggml-backend.h"
#include "llama.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct common_speculative;
struct server_prompt_cache;

// Pinned host memory buffer (GPU host buffer type / cudaMallocHost with CPU fallback); no zero-fill on resize,
// keeps allocated capacity across clear() for fault-free reuse.
struct prefill_peer_hbuf {
    uint8_t * p   = nullptr;
    size_t    n   = 0;
    size_t    cap = 0;
    ggml_backend_buffer_t buf = nullptr;

    prefill_peer_hbuf() = default;
    prefill_peer_hbuf(const prefill_peer_hbuf &) = delete;
    prefill_peer_hbuf & operator=(const prefill_peer_hbuf &) = delete;
    prefill_peer_hbuf(prefill_peer_hbuf && o) noexcept { *this = std::move(o); }
    prefill_peer_hbuf & operator=(prefill_peer_hbuf && o) noexcept {
        if (this != &o) {
            release();
            p = o.p; n = o.n; cap = o.cap; buf = o.buf;
            o.p = nullptr; o.n = o.cap = 0; o.buf = nullptr;
        }
        return *this;
    }
    ~prefill_peer_hbuf() { release(); }

    uint8_t *       data()           { return p; }
    const uint8_t * data()     const { return p; }
    size_t          size()     const { return n; }
    size_t          capacity() const { return cap; }
    bool            empty()    const { return n == 0; }
    void            clear()          { n = 0; }

    void reserve(size_t c);
    void resize(size_t m) {
        reserve(m);
        n = m;
    }
    void release();

    std::vector<uint8_t> to_vector() const {
        return std::vector<uint8_t>(p, p + n);
    }
};

// S4: prompt-cache segment and checkpoint payloads pulled over RoCE on session migration
struct prefill_peer_cache_seg {
    llama_pos p0 = 0;
    llama_pos p1 = 0;
    llama_pos d0 = 0;
    llama_pos d1 = 0x7fffffff;
    prefill_peer_hbuf main;
    prefill_peer_hbuf drft;
};

struct prefill_peer_cache_ckpt {
    int64_t   n_tokens = 0;
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;
    std::vector<uint8_t> data_spec;
};

struct prefill_peer_cache_res {
    int32_t lcp = 0;
    std::vector<llama_token> tokens;
    std::vector<prefill_peer_cache_seg> segs;
    prefill_peer_hbuf recr;
    std::vector<uint8_t> spec;
    std::vector<prefill_peer_cache_ckpt> ckpts;
    double ms          = 0.0;
    size_t total_bytes = 0;
    std::vector<prefill_peer_cache_seg> seg_pool; // retained between pulls to preserve pinned capacities

    void   clear_keep_cap();
    size_t pinned_bytes() const;
    void   trim_pinned(size_t cap_bytes);
    void   release();
};

// the context checkpoint at n_tokens the head assembles from the stages' snapshots (empty data_tgt: none)
struct prefill_peer_ckpt {
    int32_t   n_tokens = 0;
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;
    std::vector<uint8_t> data_spec;
};

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
// LLAMA_PREFILL_PEER=host:port[,host:port...]: the peers, used as one chain (the residual goes from each to the next);
// a peer that is busy refuses, then the head tries each peer alone, then decodes the prompt itself.
// LLAMA_PREFILL_PEER_SPLIT=l1[,l2...]: the first layer of each peer of the whole chain (default: even), and
// LLAMA_PREFILL_PEER_SPLIT1=l: the first layer of a single peer (default n_layer/2)
struct prefill_peer_config {
    std::vector<prefill_peer_stage> peers;    // layer_begin unused here
    std::vector<int>                split_all; // the first layer of each peer when all of them take part
    int                             split_one = -1;
    int min_tokens = 8192;

    static prefill_peer_config from_env(const llama_model * model);

    bool enabled() const { return !peers.empty(); }

    // the chains to try, best first
    std::vector<std::vector<prefill_peer_stage>> chains() const;
};

// The head: decodes tokens [0, n_split) of seq_id with the peer. The memories of seq_id must be empty on entry and the
// tokens plain text. On success the target and drafter memories hold the sequence up to n_split as if it had been
// decoded here; on failure seq_id is removed from both memories again and the caller decodes the prompt locally.
// n_ckpt > 0: the ubatches are cut at n_ckpt and ckpt receives the state at that position (partial: the recurrent rows,
// the drafter's partial state and the drafter's pending rows), as a checkpoint made there before the local decode would.
bool prefill_peer_run(const prefill_peer_config & cfg,
                      llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec,
                      llama_seq_id seq_id, const llama_token * tokens, int32_t n_split, int32_t n_ubatch,
                      int32_t n_ckpt, prefill_peer_ckpt * ckpt, prefill_peer_stats & stats, int32_t align_at = -1);

// align_at > 0: a message ends there (the first row of DeepSeek-V4.1's layer-skip tail), so that no ubatch mixes skipped
// rows with tail rows: a ubatch the tail starts in runs every layer for all its rows

// The peer: serves heads one connection at a time, forever (returns only on a socket setup error).
void prefill_peer_serve(int port, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec);

// At load, before any prompt (opt-in: LLAMA_PREFILL_PEER_WARM=1, or lo:hi,... for explicit ranges): one n_ubatch-token
// ubatch through each layer range this server can serve as a peer or send from as a head, and the pinned receive
// slots, so the first prompt after a start pays no graph build, buffer growth or pinned allocation (~1.2 s once).
// cfg may be null (a peer with no LLAMA_PREFILL_PEER of its own: default ranges).
void prefill_peer_warmup(llama_context * ctx_tgt, const prefill_peer_config * cfg);
// Pre-fault the drafter's receive and own state buffers once ctx_dft is initialized.
void prefill_peer_warmup_dft(llama_context * ctx_dft, const prefill_peer_config * cfg);

// A serving server that is also a peer (LLAMA_PREFILL_PEER_SERVE=port): a thread accepts the heads' connections
// (prefill_peer_listen, prefill_peer_accept) and hands each to the task loop, which serves it when the server is
// idle (prefill_peer_serve_conn: its memories are taken over and left empty) or refuses it (prefill_peer_refuse).
int  prefill_peer_listen(int port);
int  prefill_peer_accept(int lfd);
void prefill_peer_serve_conn(int fd, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec);
void prefill_peer_refuse(int fd);
// true while this server is a head (prefill_peer_run): the listener refuses hellos itself then (M3: two heads deadlocked)
bool prefill_peer_head_active();

// S4: compute model + context compatibility signature for RoCE cache pulls
uint64_t prefill_peer_cache_model_sig(const llama_model * model, const llama_context * ctx_tgt, const llama_context * ctx_dft);

// S4: OS-agnostic socket close / shutdown helpers for listener & pull workers
void prefill_peer_shutdown(int fd);
void prefill_peer_close(int fd);

// S4: peek accepted :9555 socket (bounded wait) to distinguish cache-pull requests from prefill hellos
bool prefill_peer_is_cache_pull(int fd);

// S4: serve a single peer cache pull directly from server_prompt_cache in RAM (never touches ctx_tgt/slots)
void prefill_peer_cache_serve(int fd, server_prompt_cache * cache, uint64_t model_sig, const std::atomic<bool> * stop = nullptr);

// S4: pull matching cached prompt state from peer_addr ("host:port")
bool prefill_peer_cache_pull(const std::string & peer_addr,
                             const std::vector<llama_token> & tokens,
                             int min_lcp,
                             uint64_t model_sig,
                             int timeout_ms,
                             prefill_peer_cache_res & out);

// S4: clear active pull payload while preserving pinned buffer capacities up to LLAMA_PEER_CACHE_PIN_CAP_MB
void prefill_peer_cache_trim(prefill_peer_cache_res & res);
