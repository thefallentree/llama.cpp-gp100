#include "server-prefill-peer.h"

#include "common.h"
#include "log.h"
#include "speculative.h"
#include "../../src/llama-ext.h" // staging API: layer range, hc input, layer-filtered state

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace {

constexpr uint32_t MSG_HELLO  = 0x31504650; // "PFP1"
constexpr uint32_t MSG_OK     = 0x4b4f4650; // "PFOK"
constexpr uint32_t MSG_UBATCH = 0x42554650; // "PFUB"
constexpr uint32_t MSG_END    = 0x4e454650; // "PFEN"
constexpr uint32_t MSG_RESULT = 0x53524650; // "PFRS"
constexpr uint32_t MSG_ERROR  = 0x52454650; // "PFER"

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool send_all(int fd, const void * data, size_t n) {
    const char * p = (const char *) data;
    while (n > 0) {
        const ssize_t k = ::send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) {
            return false;
        }
        p += k;
        n -= (size_t) k;
    }
    return true;
}

bool recv_all(int fd, void * data, size_t n) {
    char * p = (char *) data;
    while (n > 0) {
        const ssize_t k = ::recv(fd, p, n, 0);
        if (k <= 0) {
            return false;
        }
        p += k;
        n -= (size_t) k;
    }
    return true;
}

template <typename T> bool send_val(int fd, const T & v) { return send_all(fd, &v, sizeof(T)); }
template <typename T> bool recv_val(int fd, T & v)       { return recv_all(fd, &v, sizeof(T)); }

bool send_blob(int fd, const std::vector<uint8_t> & b) {
    const uint64_t n = b.size();
    return send_val(fd, n) && (n == 0 || send_all(fd, b.data(), n));
}

bool recv_blob(int fd, std::vector<uint8_t> & b) {
    uint64_t n = 0;
    if (!recv_val(fd, n) || n > (64ull << 30)) {
        return false;
    }
    b.resize(n);
    return n == 0 || recv_all(fd, b.data(), n);
}

void tune_socket(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    int buf = 32 << 20;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
}

int connect_to(const std::string & host, int port) {
    addrinfo hints = {};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || res == nullptr) {
        return -1;
    }
    int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd >= 0) {
        tune_socket(fd);
        if (::connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
            ::close(fd);
            fd = -1;
        }
    }
    freeaddrinfo(res);
    return fd;
}

// the residual rows of one ubatch on their way to the peer
struct ubatch_msg {
    int32_t            pos0 = 0;
    int32_t            n    = 0;
    std::vector<float> h;
};

// sends the queued ubatches in order on its own thread, so the head computes the next ubatch meanwhile
struct sender {
    int fd = -1;
    std::mutex m;
    std::condition_variable cv;
    std::deque<ubatch_msg> q;
    bool stop   = false;
    bool failed = false;
    size_t max_q = 2;
    std::thread th;

    void start(int fd_) {
        fd = fd_;
        th = std::thread([this]() { run(); });
    }

    void run() {
        while (true) {
            ubatch_msg msg;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [this]() { return stop || !q.empty(); });
                if (q.empty()) {
                    return;
                }
                msg = std::move(q.front());
                q.pop_front();
            }
            cv.notify_all();
            const bool ok = send_val(fd, MSG_UBATCH) && send_val(fd, msg.pos0) && send_val(fd, msg.n) &&
                            send_all(fd, msg.h.data(), msg.h.size()*sizeof(float));
            if (!ok) {
                std::lock_guard<std::mutex> lk(m);
                failed = true;
                q.clear();
                cv.notify_all();
                return;
            }
        }
    }

    // false when the connection failed
    bool push(ubatch_msg && msg) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [this]() { return failed || q.size() < max_q; });
        if (failed) {
            return false;
        }
        q.push_back(std::move(msg));
        cv.notify_all();
        return true;
    }

    // waits until everything is sent; false when the connection failed
    bool finish() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv.notify_all();
        if (th.joinable()) {
            th.join();
        }
        return !failed;
    }
};

} // namespace

prefill_peer_config prefill_peer_config::from_env(const llama_model * model) {
    prefill_peer_config cfg;
    const char * peer = getenv("LLAMA_PREFILL_PEER");
    if (peer == nullptr || *peer == '\0') {
        return cfg;
    }
    std::string s = peer;
    const auto colon = s.rfind(':');
    if (colon == std::string::npos) {
        LOG_WRN("prefill-peer: LLAMA_PREFILL_PEER=%s is not host:port, ignored\n", peer);
        return cfg;
    }
    cfg.host = s.substr(0, colon);
    cfg.port = atoi(s.c_str() + colon + 1);
    const int n_layer = llama_model_n_layer(model);
    cfg.split = getenv("LLAMA_PREFILL_PEER_SPLIT") ? atoi(getenv("LLAMA_PREFILL_PEER_SPLIT")) : n_layer/2;
    if (cfg.split <= 0 || cfg.split >= n_layer) {
        LOG_WRN("prefill-peer: split layer %d outside (0, %d), disabled\n", cfg.split, n_layer);
        cfg.port = 0;
        return cfg;
    }
    if (getenv("LLAMA_PREFILL_PEER_MIN")) {
        cfg.min_tokens = atoi(getenv("LLAMA_PREFILL_PEER_MIN"));
    }
    LOG_WRN("prefill-peer: head of %s:%d, layers [0, %d) here, prompts from %d tokens\n", cfg.host.c_str(), cfg.port, cfg.split, cfg.min_tokens);
    return cfg;
}

bool prefill_peer_run(const prefill_peer_config & cfg,
                      llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec,
                      llama_seq_id seq_id, const llama_token * tokens, int32_t n_split, int32_t n_ubatch,
                      prefill_peer_stats & stats) {
    const double t0 = now_ms();
    stats = {};

    const int64_t width = llama_model_n_embd_out(llama_get_model(ctx_tgt));

    auto fail = [&](const char * what) {
        LOG_WRN("prefill-peer: %s, the prompt is decoded here\n", what);
        llama_set_layer_range(ctx_tgt, 0, -1);
        llama_synchronize(ctx_tgt);
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, -1, -1);
        }
        return false;
    };

    const int fd = connect_to(cfg.host, cfg.port);
    if (fd < 0) {
        LOG_WRN("prefill-peer: cannot connect to %s:%d\n", cfg.host.c_str(), cfg.port);
        return false;
    }

    // the drafter's pending row and deferred rows as they are here, so the peer's catch-up starts from the same state
    std::vector<uint8_t> spec_in;
    if (spec) {
        common_speculative_get_state(spec, seq_id, spec_in);
    }

    {
        const uint32_t version = 1;
        const int32_t  split   = cfg.split;
        bool ok = send_val(fd, MSG_HELLO) && send_val(fd, version) && send_val(fd, split) && send_val(fd, n_split) &&
                  send_val(fd, n_ubatch) && send_all(fd, tokens, (size_t) n_split*sizeof(llama_token)) && send_blob(fd, spec_in);
        uint32_t msg = 0;
        int64_t  width_peer = 0;
        ok = ok && recv_val(fd, msg) && msg == MSG_OK && recv_val(fd, width_peer) && width_peer == width;
        if (!ok) {
            ::close(fd);
            LOG_WRN("prefill-peer: the peer at %s:%d did not accept the prompt\n", cfg.host.c_str(), cfg.port);
            return false;
        }
    }

    // the residual after the last local layer of every token comes out as the nextn embeddings
    llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
    llama_set_layer_range(ctx_tgt, 0, cfg.split);

    sender snd;
    snd.start(fd);

    // whole batches per call, as the server decodes a prompt: the next ubatch's host side overlaps the previous one
    const int32_t n_chunk = std::max<int32_t>(n_ubatch, ((int32_t) llama_n_batch(ctx_tgt) / n_ubatch) * n_ubatch);

    common_batch batch(ctx_tgt);
    bool ok = true;
    for (int32_t off = 0; off < n_split && ok; off += n_chunk) {
        const int32_t n = std::min(n_chunk, n_split - off);
        batch.clear();
        for (int32_t i = 0; i < n; ++i) {
            batch.add(tokens[off + i], off + i, seq_id, false);
        }
        const double tl = now_ms();
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            ok = false;
            break;
        }
        const float * h = llama_get_embeddings_nextn(ctx_tgt); // waits for the ubatch
        if (h == nullptr) {
            ok = false;
            break;
        }
        ubatch_msg msg;
        msg.pos0 = off;
        msg.n    = n;
        msg.h.assign(h, h + (size_t) n*width);
        stats.t_local_ms += now_ms() - tl;
        const double tw = now_ms();
        ok = snd.push(std::move(msg));
        stats.t_wait_ms += now_ms() - tw;
    }
    llama_set_layer_range(ctx_tgt, 0, -1);

    const double tw = now_ms();
    ok = snd.finish() && ok;
    if (!ok) {
        ::close(fd);
        return fail("a local decode or the connection failed");
    }

    std::vector<uint8_t> st_tgt, st_dft, st_spec;
    {
        uint32_t msg = 0;
        bool r = send_val(fd, MSG_END) && recv_val(fd, msg);
        if (r && msg == MSG_ERROR) {
            std::vector<uint8_t> err;
            recv_blob(fd, err);
            err.push_back(0);
            LOG_WRN("prefill-peer: the peer failed: %s\n", (const char *) err.data());
            r = false;
        }
        r = r && msg == MSG_RESULT && recv_blob(fd, st_tgt) && recv_blob(fd, st_dft) && recv_blob(fd, st_spec) && recv_val(fd, stats.peer_ms);
        ::close(fd);
        if (!r) {
            return fail("no state from the peer");
        }
    }
    stats.t_wait_ms += now_ms() - tw;
    stats.bytes_state = st_tgt.size() + st_dft.size() + st_spec.size();

    const double tr = now_ms();
    if (llama_state_seq_set_data_layers(ctx_tgt, st_tgt.data(), st_tgt.size(), seq_id, cfg.split, -1) != st_tgt.size()) {
        return fail("the peer's layers did not fit the local sequence");
    }
    if (ctx_dft) {
        if (llama_state_seq_set_data_ext(ctx_dft, st_dft.data(), st_dft.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) != st_dft.size()) {
            return fail("the peer's drafter state did not load");
        }
    }
    if (spec) {
        common_speculative_set_state(spec, seq_id, st_spec);
    }
    stats.t_restore_ms = now_ms() - tr;
    stats.t_total_ms   = now_ms() - t0;

    LOG_WRN("prefill-peer: %d tokens in %.0f ms (%.0f tok/s): local %.0f ms, waits %.0f ms, peer %.0f ms, state %.1f MiB restored in %.0f ms\n",
            n_split, stats.t_total_ms, n_split*1000.0/stats.t_total_ms, stats.t_local_ms, stats.t_wait_ms, stats.peer_ms,
            stats.bytes_state/1048576.0, stats.t_restore_ms);
    return true;
}

namespace {

// one head's prompt on the peer; false = the connection is done (error or not)
void serve_one(int fd, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec) {
    const int64_t width = llama_model_n_embd_out(llama_get_model(ctx_tgt));
    const llama_seq_id seq_id = 0;

    uint32_t msg = 0, version = 0;
    int32_t  split = 0, n_split = 0, n_ubatch = 0;
    if (!recv_val(fd, msg) || msg != MSG_HELLO || !recv_val(fd, version) || version != 1 ||
        !recv_val(fd, split) || !recv_val(fd, n_split) || !recv_val(fd, n_ubatch) || n_split <= 0) {
        LOG_WRN("prefill-peer: bad hello\n");
        return;
    }
    std::vector<llama_token> tokens(n_split);
    std::vector<uint8_t> spec_in;
    if (!recv_all(fd, tokens.data(), tokens.size()*sizeof(llama_token)) || !recv_blob(fd, spec_in)) {
        return;
    }

    // an empty memory, as on the head
    llama_memory_clear(llama_get_memory(ctx_tgt), true);
    if (ctx_dft) {
        llama_memory_clear(llama_get_memory(ctx_dft), true);
    }
    if (spec) {
        common_speculative_set_state(spec, seq_id, spec_in);
    }

    if (!send_val(fd, MSG_OK) || !send_val(fd, width)) {
        return;
    }

    LOG_WRN("prefill-peer: %d tokens, layers [%d, end) here\n", n_split, split);

    llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
    llama_set_layer_range(ctx_tgt, split, -1);

    const double t0 = now_ms();
    double t_compute = 0.0;
    common_batch batch(ctx_tgt);
    std::vector<float> h;
    std::string err;

    // the next message is read while the previous ubatch computes: llama_process returns before the GPU is done,
    // the catch-up of the drafter reads the outputs (and waits for them) after the next receive
    bool ok = true;
    while (ok) {
        if (!recv_val(fd, msg)) {
            ok = false;
            break;
        }
        if (msg == MSG_END) {
            break;
        }
        int32_t pos0 = 0, n = 0;
        if (msg != MSG_UBATCH || !recv_val(fd, pos0) || !recv_val(fd, n) || n <= 0 || pos0 < 0 || pos0 + n > n_split) {
            ok = false;
            err = "bad ubatch";
            break;
        }
        h.resize((size_t) n*width);
        if (!recv_all(fd, h.data(), h.size()*sizeof(float))) {
            ok = false;
            break;
        }
        const double tc = now_ms();
        llama_set_hc_input(ctx_tgt, h.data(), pos0, n);
        batch.clear();
        for (int32_t i = 0; i < n; ++i) {
            batch.add(tokens[pos0 + i], pos0 + i, seq_id, false);
        }
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            ok = false;
            err = "decode failed";
            break;
        }
        if (spec && !common_speculative_process(spec, batch)) {
            ok = false;
            err = "drafter catch-up failed";
            break;
        }
        t_compute += now_ms() - tc;
    }
    llama_set_layer_range(ctx_tgt, 0, -1);
    llama_synchronize(ctx_tgt);
    if (ctx_dft) {
        llama_synchronize(ctx_dft);
    }

    if (!ok) {
        if (!err.empty()) {
            std::vector<uint8_t> e(err.begin(), err.end());
            send_val(fd, MSG_ERROR);
            send_blob(fd, e);
        }
        LOG_WRN("prefill-peer: prompt abandoned (%s)\n", err.empty() ? "connection" : err.c_str());
        return;
    }

    std::vector<uint8_t> st_tgt, st_dft, st_spec;
    st_tgt.resize(llama_state_seq_get_size_layers(ctx_tgt, seq_id, split, -1));
    st_tgt.resize(llama_state_seq_get_data_layers(ctx_tgt, st_tgt.data(), st_tgt.size(), seq_id, split, -1));
    if (ctx_dft) {
        st_dft.resize(llama_state_seq_get_size_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
        st_dft.resize(llama_state_seq_get_data_ext(ctx_dft, st_dft.data(), st_dft.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
    }
    if (spec) {
        common_speculative_get_state(spec, seq_id, st_spec);
    }
    const double peer_ms = now_ms() - t0;
    send_val(fd, MSG_RESULT) && send_blob(fd, st_tgt) && send_blob(fd, st_dft) && send_blob(fd, st_spec) && send_val(fd, peer_ms);

    LOG_WRN("prefill-peer: %d tokens done in %.0f ms (compute %.0f ms), state %.1f MiB sent\n", n_split, peer_ms, t_compute,
            (st_tgt.size() + st_dft.size() + st_spec.size())/1048576.0);
}

} // namespace

void prefill_peer_serve(int port, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec) {
    const int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        LOG_ERR("prefill-peer: socket failed\n");
        return;
    }
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons((uint16_t) port);
    if (::bind(lfd, (sockaddr *) &addr, sizeof(addr)) != 0 || ::listen(lfd, 4) != 0) {
        LOG_ERR("prefill-peer: cannot listen on port %d\n", port);
        ::close(lfd);
        return;
    }
    LOG_WRN("prefill-peer: serving heads on port %d\n", port);
    while (true) {
        const int fd = ::accept(lfd, nullptr, nullptr);
        if (fd < 0) {
            continue;
        }
        tune_socket(fd);
        serve_one(fd, ctx_tgt, ctx_dft, spec);
        ::close(fd);
    }
}
