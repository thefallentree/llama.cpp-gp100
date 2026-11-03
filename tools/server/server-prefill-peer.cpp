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
    const int n_layer = llama_model_n_layer(model);
    for (const auto & item : string_split<std::string>(peer, ',')) {
        const auto colon = item.rfind(':');
        if (colon == std::string::npos) {
            LOG_WRN("prefill-peer: '%s' of LLAMA_PREFILL_PEER is not host:port, disabled\n", item.c_str());
            return {};
        }
        prefill_peer_stage st;
        st.host = item.substr(0, colon);
        st.port = atoi(item.c_str() + colon + 1);
        cfg.stages.push_back(st);
    }
    const int n_parts = (int) cfg.stages.size() + 1;
    std::vector<int> splits;
    if (getenv("LLAMA_PREFILL_PEER_SPLIT")) {
        for (const auto & x : string_split<std::string>(getenv("LLAMA_PREFILL_PEER_SPLIT"), ',')) {
            splits.push_back(atoi(x.c_str()));
        }
    } else {
        for (int k = 1; k < n_parts; ++k) {
            splits.push_back((n_layer*k + n_parts/2)/n_parts);
        }
    }
    if ((int) splits.size() != n_parts - 1) {
        LOG_WRN("prefill-peer: %d peers need %d split layers, disabled\n", n_parts - 1, n_parts - 1);
        return {};
    }
    for (int k = 0; k < n_parts - 1; ++k) {
        const int lo = k == 0 ? 0 : splits[k - 1];
        if (splits[k] <= lo || splits[k] >= n_layer) {
            LOG_WRN("prefill-peer: split layer %d out of order or range, disabled\n", splits[k]);
            return {};
        }
        cfg.stages[k].layer_begin = splits[k];
    }
    if (getenv("LLAMA_PREFILL_PEER_MIN")) {
        cfg.min_tokens = atoi(getenv("LLAMA_PREFILL_PEER_MIN"));
    }
    for (size_t k = 0; k < cfg.stages.size(); ++k) {
        LOG_WRN("prefill-peer: peer %zu %s:%d from layer %d\n", k, cfg.stages[k].host.c_str(), cfg.stages[k].port, cfg.stages[k].layer_begin);
    }
    LOG_WRN("prefill-peer: head, layers [0, %d) here, prompts from %d tokens\n", cfg.split(), cfg.min_tokens);
    return cfg;
}

namespace {

// the state parts of the chain: [il_lo, il_hi) of the target, as the stages wrote them
struct state_part {
    int32_t il_lo = 0;
    int32_t il_hi = -1;
    std::vector<uint8_t> data;
};

struct chain_result {
    std::vector<state_part> parts;
    std::vector<uint8_t>    st_dft;
    std::vector<uint8_t>    st_spec;
    double                  peer_ms = 0.0;
};

// HELLO: the prompt and the stages from the receiver on (the receiver is stages[0])
bool send_hello(int fd, const std::vector<prefill_peer_stage> & stages, const llama_token * tokens, int32_t n_split,
                int32_t n_ubatch, const std::vector<uint8_t> & spec_in, int64_t width) {
    const uint32_t version = 2;
    const uint32_t n_st    = stages.size();
    bool ok = send_val(fd, MSG_HELLO) && send_val(fd, version) && send_val(fd, n_split) && send_val(fd, n_ubatch) &&
              send_all(fd, tokens, (size_t) n_split*sizeof(llama_token)) && send_blob(fd, spec_in) && send_val(fd, n_st);
    for (const auto & st : stages) {
        std::vector<uint8_t> h(st.host.begin(), st.host.end());
        const int32_t port = st.port, lb = st.layer_begin;
        ok = ok && send_blob(fd, h) && send_val(fd, port) && send_val(fd, lb);
    }
    uint32_t msg = 0;
    int64_t  width_peer = 0;
    return ok && recv_val(fd, msg) && msg == MSG_OK && recv_val(fd, width_peer) && width_peer == width;
}

// after END: the result of the stage behind fd (and of the ones behind it)
bool recv_result(int fd, chain_result & res) {
    uint32_t msg = 0;
    if (!recv_val(fd, msg)) {
        return false;
    }
    if (msg == MSG_ERROR) {
        std::vector<uint8_t> err;
        recv_blob(fd, err);
        err.push_back(0);
        LOG_WRN("prefill-peer: the next stage failed: %s\n", (const char *) err.data());
        return false;
    }
    uint32_t n_parts = 0;
    if (msg != MSG_RESULT || !recv_val(fd, n_parts) || n_parts > 64) {
        return false;
    }
    for (uint32_t i = 0; i < n_parts; ++i) {
        state_part p;
        if (!recv_val(fd, p.il_lo) || !recv_val(fd, p.il_hi) || !recv_blob(fd, p.data)) {
            return false;
        }
        res.parts.push_back(std::move(p));
    }
    return recv_blob(fd, res.st_dft) && recv_blob(fd, res.st_spec) && recv_val(fd, res.peer_ms);
}

bool send_result(int fd, const chain_result & res) {
    const uint32_t n_parts = res.parts.size();
    bool ok = send_val(fd, MSG_RESULT) && send_val(fd, n_parts);
    for (const auto & p : res.parts) {
        ok = ok && send_val(fd, p.il_lo) && send_val(fd, p.il_hi) && send_blob(fd, p.data);
    }
    return ok && send_blob(fd, res.st_dft) && send_blob(fd, res.st_spec) && send_val(fd, res.peer_ms);
}

} // namespace

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

    const auto & first = cfg.stages[0];
    const int fd = connect_to(first.host, first.port);
    if (fd < 0) {
        LOG_WRN("prefill-peer: cannot connect to %s:%d\n", first.host.c_str(), first.port);
        return false;
    }

    // the drafter's pending row and deferred rows as they are here, so the last stage's catch-up starts from them
    std::vector<uint8_t> spec_in;
    if (spec) {
        common_speculative_get_state(spec, seq_id, spec_in);
    }

    if (!send_hello(fd, cfg.stages, tokens, n_split, n_ubatch, spec_in, width)) {
        ::close(fd);
        LOG_WRN("prefill-peer: the chain at %s:%d did not accept the prompt\n", first.host.c_str(), first.port);
        return false;
    }

    // the residual after the last local layer of every token comes out as the nextn embeddings
    llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
    llama_set_layer_range(ctx_tgt, 0, cfg.split());

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
        const float * h = llama_get_embeddings_nextn(ctx_tgt); // waits for the batch
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

    chain_result res;
    {
        const bool r = send_val(fd, MSG_END) && recv_result(fd, res);
        ::close(fd);
        if (!r) {
            return fail("no state from the peers");
        }
    }
    stats.t_wait_ms += now_ms() - tw;
    stats.peer_ms    = res.peer_ms;

    const double tr = now_ms();
    for (const auto & p : res.parts) {
        stats.bytes_state += p.data.size();
        if (llama_state_seq_set_data_layers(ctx_tgt, p.data.data(), p.data.size(), seq_id, p.il_lo, p.il_hi) != p.data.size()) {
            return fail("a peer's layers did not fit the local sequence");
        }
    }
    stats.bytes_state += res.st_dft.size() + res.st_spec.size();
    if (ctx_dft) {
        if (llama_state_seq_set_data_ext(ctx_dft, res.st_dft.data(), res.st_dft.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) != res.st_dft.size()) {
            return fail("the drafter state did not load");
        }
    }
    if (spec) {
        common_speculative_set_state(spec, seq_id, res.st_spec);
    }
    stats.t_restore_ms = now_ms() - tr;
    stats.t_total_ms   = now_ms() - t0;

    LOG_WRN("prefill-peer: %d tokens in %.0f ms (%.0f tok/s): local %.0f ms, waits %.0f ms, peers %.0f ms, state %.1f MiB restored in %.0f ms\n",
            n_split, stats.t_total_ms, n_split*1000.0/stats.t_total_ms, stats.t_local_ms, stats.t_wait_ms, stats.peer_ms,
            stats.bytes_state/1048576.0, stats.t_restore_ms);
    return true;
}

namespace {

// one head's prompt on a peer: its layers, then the next stage's (forwarded) or the drafter's catch-up (last stage)
void serve_one(int fd, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec) {
    const int64_t width = llama_model_n_embd_out(llama_get_model(ctx_tgt));
    const llama_seq_id seq_id = 0;

    uint32_t msg = 0, version = 0, n_st = 0;
    int32_t  n_split = 0, n_ubatch = 0;
    if (!recv_val(fd, msg) || msg != MSG_HELLO || !recv_val(fd, version) || version != 2 ||
        !recv_val(fd, n_split) || !recv_val(fd, n_ubatch) || n_split <= 0) {
        LOG_WRN("prefill-peer: bad hello\n");
        return;
    }
    std::vector<llama_token> tokens(n_split);
    std::vector<uint8_t> spec_in;
    if (!recv_all(fd, tokens.data(), tokens.size()*sizeof(llama_token)) || !recv_blob(fd, spec_in) ||
        !recv_val(fd, n_st) || n_st == 0 || n_st > 16) {
        return;
    }
    std::vector<prefill_peer_stage> stages(n_st);
    for (auto & st : stages) {
        std::vector<uint8_t> h;
        int32_t port = 0, lb = 0;
        if (!recv_blob(fd, h) || !recv_val(fd, port) || !recv_val(fd, lb)) {
            return;
        }
        st.host.assign(h.begin(), h.end());
        st.port        = port;
        st.layer_begin = lb;
    }
    const int32_t il_lo = stages[0].layer_begin;
    const int32_t il_hi = n_st > 1 ? stages[1].layer_begin : -1;
    const bool    last  = n_st == 1;

    // an empty memory, as on the head
    llama_memory_clear(llama_get_memory(ctx_tgt), true);
    if (ctx_dft) {
        llama_memory_clear(llama_get_memory(ctx_dft), true);
    }
    if (spec && last) {
        common_speculative_set_state(spec, seq_id, spec_in);
    }

    // the next stage first, so the whole chain is up before the head starts
    int fd_next = -1;
    if (!last) {
        fd_next = connect_to(stages[1].host, stages[1].port);
        std::vector<prefill_peer_stage> rest(stages.begin() + 1, stages.end());
        if (fd_next < 0 || !send_hello(fd_next, rest, tokens.data(), n_split, n_ubatch, spec_in, width)) {
            LOG_WRN("prefill-peer: the next stage %s:%d did not accept the prompt\n", stages[1].host.c_str(), stages[1].port);
            if (fd_next >= 0) {
                ::close(fd_next);
            }
            return;
        }
    }

    if (!send_val(fd, MSG_OK) || !send_val(fd, width)) {
        if (fd_next >= 0) {
            ::close(fd_next);
        }
        return;
    }

    LOG_WRN("prefill-peer: %d tokens, layers [%d, %d) here%s\n", n_split, il_lo, il_hi, last ? " + the drafter" : "");

    llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
    llama_set_layer_range(ctx_tgt, il_lo, il_hi);

    sender snd;
    if (!last) {
        snd.start(fd_next);
    }

    const double t0 = now_ms();
    double t_compute = 0.0;
    common_batch batch(ctx_tgt);
    std::string err;

    // a receiver thread reads the next batch while this one computes (a 3072-token batch is ~50 ms on the 25GbE mesh)
    struct rx_state {
        std::mutex m;
        std::condition_variable cv;
        std::deque<ubatch_msg> q;
        bool end = false, failed = false, stop = false;
    } rx;
    std::thread rx_th([&]() {
        while (true) {
            uint32_t m = 0;
            ubatch_msg u;
            bool good = recv_val(fd, m);
            if (good && m == MSG_END) {
                std::lock_guard<std::mutex> lk(rx.m);
                rx.end = true;
                rx.cv.notify_all();
                return;
            }
            good = good && m == MSG_UBATCH && recv_val(fd, u.pos0) && recv_val(fd, u.n) && u.n > 0 && u.pos0 >= 0 && u.pos0 + u.n <= n_split;
            if (good) {
                u.h.resize((size_t) u.n*width);
                good = recv_all(fd, u.h.data(), u.h.size()*sizeof(float));
            }
            std::unique_lock<std::mutex> lk(rx.m);
            if (!good) {
                rx.failed = true;
                rx.cv.notify_all();
                return;
            }
            rx.cv.wait(lk, [&]() { return rx.stop || rx.q.size() < 2; });
            if (rx.stop) {
                return;
            }
            rx.q.push_back(std::move(u));
            rx.cv.notify_all();
        }
    });

    bool ok = true;
    while (ok) {
        ubatch_msg u;
        {
            std::unique_lock<std::mutex> lk(rx.m);
            rx.cv.wait(lk, [&]() { return !rx.q.empty() || rx.end || rx.failed; });
            if (rx.q.empty()) {
                ok = !rx.failed;
                break;
            }
            u = std::move(rx.q.front());
            rx.q.pop_front();
        }
        rx.cv.notify_all();
        const int32_t pos0 = u.pos0, n = u.n;
        const double tc = now_ms();
        llama_set_hc_input(ctx_tgt, u.h.data(), pos0, n);
        batch.clear();
        for (int32_t i = 0; i < n; ++i) {
            batch.add(tokens[pos0 + i], pos0 + i, seq_id, false);
        }
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            ok = false;
            err = "decode failed";
            break;
        }
        if (!last) {
            const float * h = llama_get_embeddings_nextn(ctx_tgt);
            if (h == nullptr) {
                ok = false;
                err = "no residual";
                break;
            }
            ubatch_msg out;
            out.pos0 = pos0;
            out.n    = n;
            out.h.assign(h, h + (size_t) n*width);
            t_compute += now_ms() - tc;
            if (!snd.push(std::move(out))) {
                ok = false;
                err = "the next stage is gone";
                break;
            }
            continue;
        }
        if (spec && !common_speculative_process(spec, batch)) {
            ok = false;
            err = "drafter catch-up failed";
            break;
        }
        t_compute += now_ms() - tc;
    }
    {
        std::lock_guard<std::mutex> lk(rx.m);
        rx.stop = true;
    }
    rx.cv.notify_all();
    if (!ok) {
        ::shutdown(fd, SHUT_RD); // the receiver may sit in recv
    }
    rx_th.join();
    llama_set_layer_range(ctx_tgt, 0, -1);
    llama_synchronize(ctx_tgt);
    if (ctx_dft) {
        llama_synchronize(ctx_dft);
    }

    chain_result res;
    if (ok && !last) {
        ok = snd.finish() && send_val(fd_next, MSG_END) && recv_result(fd_next, res);
        if (!ok) {
            err = "no result from the next stage";
        }
    } else if (!last) {
        snd.finish();
    }
    if (fd_next >= 0) {
        ::close(fd_next);
    }

    if (!ok) {
        if (err.empty()) {
            err = "connection";
        }
        std::vector<uint8_t> e(err.begin(), err.end());
        send_val(fd, MSG_ERROR);
        send_blob(fd, e);
        LOG_WRN("prefill-peer: prompt abandoned (%s)\n", err.c_str());
        return;
    }

    state_part own;
    own.il_lo = il_lo;
    own.il_hi = il_hi;
    own.data.resize(llama_state_seq_get_size_layers(ctx_tgt, seq_id, il_lo, il_hi));
    own.data.resize(llama_state_seq_get_data_layers(ctx_tgt, own.data.data(), own.data.size(), seq_id, il_lo, il_hi));
    const size_t own_bytes = own.data.size();
    res.parts.insert(res.parts.begin(), std::move(own));
    if (last) {
        if (ctx_dft) {
            res.st_dft.resize(llama_state_seq_get_size_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
            res.st_dft.resize(llama_state_seq_get_data_ext(ctx_dft, res.st_dft.data(), res.st_dft.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
        }
        if (spec) {
            common_speculative_get_state(spec, seq_id, res.st_spec);
        }
    }
    const double peer_ms = now_ms() - t0;
    res.peer_ms = std::max(res.peer_ms, peer_ms);
    send_result(fd, res);

    LOG_WRN("prefill-peer: %d tokens done in %.0f ms (compute %.0f ms), own state %.1f MiB\n", n_split, peer_ms, t_compute, own_bytes/1048576.0);
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
