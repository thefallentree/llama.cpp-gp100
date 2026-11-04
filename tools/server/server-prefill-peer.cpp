#include "server-prefill-peer.h"

#include "common.h"
#include "ggml-backend.h"
#include "log.h"
#include "speculative.h"
#include "../../src/llama-ext.h" // staging API: layer range, hc input, layer-filtered state

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <thread>

namespace {

constexpr uint32_t MSG_HELLO  = 0x31504650; // "PFP1"
constexpr uint32_t MSG_OK     = 0x4b4f4650; // "PFOK"
constexpr uint32_t MSG_UBATCH = 0x42554650; // "PFUB"
constexpr uint32_t MSG_END    = 0x4e454650; // "PFEN"
constexpr uint32_t MSG_RESULT = 0x53524650; // "PFRS"
constexpr uint32_t MSG_ERROR  = 0x52454650; // "PFER"

// protocol 3: the hello carries the head's result endpoint and a token; every stage returns its state directly to the
// head on its own connection (the chain only passes an OK / error status back), the residual rows are sent from the
// context's nextn buffer and received into pinned memory
// protocol 4: the hello carries n_ckpt; the stages snapshot their recurrent rows (and the drafter) before the ubatch
// that starts there and return the snapshot with the result
constexpr uint32_t PROTO_VERSION = 4;

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// wall-clock ms since the epoch: the nodes are NTP-synced, so the fill / handoff timestamps of the three logs align
int64_t wall_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
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

// into a buffer that keeps its capacity: a fresh 500 MB vector costs ~0.5 s of page faults (measured 0.7 GB/s into
// a fresh buffer vs 2.2 GB/s into a reused one on the mesh), so the state buffers persist across prompts
bool recv_blob(int fd, std::vector<uint8_t> & b) {
    uint64_t n = 0;
    if (!recv_val(fd, n) || n > (64ull << 30)) {
        return false;
    }
    b.resize(n);
    return n == 0 || recv_all(fd, b.data(), n);
}

// a state buffer in pinned host memory (the GPU's host buffer type, cudaMallocHost; plain memory without a GPU): the
// state get/set of the CUDA backend issues one cudaMemcpyAsync per layer chunk, which is a DMA from pinned memory and a
// synchronous staged copy from pageable memory (1.3-1.7 GB/s measured for the state parts); no zero-fill on resize
struct hbuf {
    uint8_t * p   = nullptr;
    size_t    n   = 0;
    size_t    cap = 0;
    ggml_backend_buffer_t buf = nullptr;

    hbuf() = default;
    hbuf(const hbuf &) = delete;
    hbuf & operator=(const hbuf &) = delete;
    hbuf(hbuf && o) noexcept { *this = std::move(o); }
    hbuf & operator=(hbuf && o) noexcept {
        if (this != &o) {
            release();
            p = o.p; n = o.n; cap = o.cap; buf = o.buf;
            o.p = nullptr; o.n = o.cap = 0; o.buf = nullptr;
        }
        return *this;
    }
    ~hbuf() { release(); }

    uint8_t *       data()           { return p; }
    const uint8_t * data()     const { return p; }
    size_t          size()     const { return n; }
    size_t          capacity() const { return cap; }
    bool            empty()    const { return n == 0; }
    void            clear()          { n = 0; }

    void reserve(size_t c) {
        if (c <= cap) {
            return;
        }
        ggml_backend_buffer_t nb = nullptr;
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (buft) {
            nb = ggml_backend_buft_alloc_buffer(buft, c);
        }
        if (nb == nullptr) {
            nb = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), c);
        }
        GGML_ASSERT(nb != nullptr && "prefill-peer: no host memory for a state buffer");
        uint8_t * np = (uint8_t *) ggml_backend_buffer_get_base(nb);
        if (n > 0) {
            memcpy(np, p, n);
        }
        release();
        buf = nb;
        p   = np;
        cap = c;
    }

    void resize(size_t m) {
        reserve(m);
        n = m;
    }

    void release() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        buf = nullptr;
        p   = nullptr;
        n   = 0;
        cap = 0;
    }

    std::vector<uint8_t> to_vector() const {
        return std::vector<uint8_t>(p, p + n);
    }
};

bool send_blob(int fd, const hbuf & b) {
    const uint64_t n = b.size();
    return send_val(fd, n) && (n == 0 || send_all(fd, b.data(), n));
}

bool recv_blob(int fd, hbuf & b) {
    uint64_t n = 0;
    if (!recv_val(fd, n) || n > (64ull << 30)) {
        return false;
    }
    b.resize(n);
    return n == 0 || recv_all(fd, b.data(), n);
}

// LLAMA_PREFILL_PEER_RXBUF=0 leaves the head's receive buffers to grow with the prompts (page faults on every new size)
bool rxbuf_on() {
    static const bool on = getenv("LLAMA_PREFILL_PEER_RXBUF") == nullptr || atoi(getenv("LLAMA_PREFILL_PEER_RXBUF")) != 0;
    return on;
}

// grow a persistent buffer to n bytes and touch its pages once, so a later recv into it pays no faults; clear() keeps
// the capacity
void prefault(hbuf & b, size_t n) {
    b.reserve(n); // pinned pages are mapped at the allocation
}

// a buffer whose size is only known after a prompt: grown once to the estimate for the longest prompt with 10% headroom,
// so the estimate's drift between prompts never grows it again
void grow_rx(hbuf & b, size_t n_target) {
    if (b.capacity() < n_target) {
        prefault(b, (size_t) (n_target*1.1));
    }
}

// the state of the longest prompt of ctx from the state of n tokens: the per-token part scaled to n_ctx, the constant
// part (n_const bytes: the recurrent rows and the like) once
size_t state_estimate(const llama_context * ctx, size_t n_bytes, size_t n_const, int32_t n) {
    return n_bytes <= n_const ? n_bytes : (size_t) ((double) (n_bytes - n_const)/n*llama_n_ctx(ctx)) + n_const;
}

bool send_string(int fd, const std::string & s) {
    std::vector<uint8_t> b(s.begin(), s.end());
    return send_blob(fd, b);
}

bool recv_string(int fd, std::string & s) {
    std::vector<uint8_t> b;
    if (!recv_blob(fd, b) || b.size() > 4096) {
        return false;
    }
    s.assign(b.begin(), b.end());
    return true;
}

// TCP_NODELAY for the small control messages; the buffers stay autotuned (a fixed SO_SNDBUF/SO_RCVBUF is capped at
// net.core.*mem_max = 4 MiB in the containers and turns receive autotuning off: 1.5 vs 2.2 GB/s measured);
// LLAMA_PREFILL_PEER_SOCKBUF=bytes sets them anyway
void tune_socket(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    static const int buf = getenv("LLAMA_PREFILL_PEER_SOCKBUF") ? atoi(getenv("LLAMA_PREFILL_PEER_SOCKBUF")) : 0;
    if (buf > 0) {
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
    }
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

int listen_on(int port) {
    const int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        return -1;
    }
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons((uint16_t) port);
    if (::bind(lfd, (sockaddr *) &addr, sizeof(addr)) != 0 || ::listen(lfd, 8) != 0) {
        ::close(lfd);
        return -1;
    }
    return lfd;
}

int local_port(int fd) {
    sockaddr_in addr = {};
    socklen_t len = sizeof(addr);
    return getsockname(fd, (sockaddr *) &addr, &len) == 0 ? ntohs(addr.sin_port) : 0;
}

std::string local_ip(int fd) {
    sockaddr_in addr = {};
    socklen_t len = sizeof(addr);
    char buf[INET_ADDRSTRLEN] = {};
    if (getsockname(fd, (sockaddr *) &addr, &len) != 0 || inet_ntop(AF_INET, &addr.sin_addr, buf, sizeof(buf)) == nullptr) {
        return "";
    }
    return buf;
}

// pinned host memory (the device's host buffer type: cudaMallocHost on CUDA), so the residual rows go H2D as a DMA
// instead of a staged pageable copy (39.6 ms -> ~6 ms per 63 MB ubatch on the last stage)
struct pinned_buf {
    ggml_backend_buffer_t buf  = nullptr;
    float *               data = nullptr;
    size_t                n    = 0; // floats

    void alloc(const llama_model * model, size_t n_floats) {
        free();
        ggml_backend_dev_t dev = llama_model_get_device(model, 0);
        ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (buft == nullptr) {
            buft = ggml_backend_cpu_buffer_type();
        }
        buf = ggml_backend_buft_alloc_buffer(buft, n_floats*sizeof(float));
        if (buf == nullptr) {
            buft = ggml_backend_cpu_buffer_type();
            buf  = ggml_backend_buft_alloc_buffer(buft, n_floats*sizeof(float));
        }
        GGML_ASSERT(buf != nullptr && "prefill-peer: no host memory for the residual rows");
        data = (float *) ggml_backend_buffer_get_base(buf);
        n    = n_floats;
    }

    void free() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        buf  = nullptr;
        data = nullptr;
        n    = 0;
    }
};

// the receiver's ring of pinned slots for the incoming residual rows; persists across prompts (one head at a time)
struct rx_ring {
    std::vector<pinned_buf> slots;
    std::vector<bool>       busy;
    size_t                  rows_cap = 0; // rows per slot
    const llama_model *     model    = nullptr;
    int64_t                 width    = 0;

    static int n_slots() {
        static const int n = std::max(2, getenv("LLAMA_PREFILL_PEER_RX_SLOTS") ? atoi(getenv("LLAMA_PREFILL_PEER_RX_SLOTS")) : 4);
        return n;
    }

    void prepare(const llama_model * m, int64_t w, size_t rows) {
        model = m;
        width = w;
        if (slots.empty()) {
            slots.resize(n_slots());
            busy.assign(slots.size(), false);
        }
        if (rows > rows_cap) {
            const double t0 = now_ms();
            for (auto & s : slots) {
                s.alloc(model, rows*width);
                memset(s.data, 0, s.n*sizeof(float));
            }
            rows_cap = rows;
            LOG_WRN("prefill-peer: %zu pinned slots of %zu rows (%.1f MiB) in %.0f ms\n", slots.size(), rows, slots.size()*rows*width*4/1048576.0, now_ms() - t0);
        }
    }

    // a free slot, or -1; a message bigger than the slots (LLAMA_PREFILL_PEER_CHUNK) grows them, which waits until no
    // slot is in use (the caller holds the lock of the queue that tracks busy and retries)
    int take(size_t rows) {
        if (rows > rows_cap) {
            for (size_t i = 0; i < slots.size(); ++i) {
                if (busy[i]) {
                    return -1;
                }
            }
            prepare(model, width, rows);
        }
        for (size_t i = 0; i < slots.size(); ++i) {
            if (!busy[i]) {
                busy[i] = true;
                return (int) i;
            }
        }
        return -1;
    }
};

rx_ring & the_rx_ring() {
    static rx_ring r;
    return r;
}

// the residual rows of one ubatch on their way to the next stage: a view of the context's nextn buffer (the sender) or of
// a pinned slot (the receiver)
struct ubatch_msg {
    int32_t       pos0 = 0;
    int32_t       n    = 0;
    float *       h    = nullptr; // the context's nextn buffer (sender) or a pinned slot (receiver)
    int           slot = -1;
};

// sends the queued ubatches in order on its own thread straight from the nextn buffer (no copy on the main thread); the
// main thread waits for n_sent before it reuses a nextn slot
struct sender {
    int fd = -1;
    int64_t width = 0;
    std::mutex m;
    std::condition_variable cv;
    std::deque<ubatch_msg> q;
    bool stop   = false;
    bool failed = false;
    int64_t n_pushed = 0;
    int64_t n_sent   = 0;
    std::thread th;

    void start(int fd_, int64_t width_) {
        fd    = fd_;
        width = width_;
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
                msg = q.front();
                q.pop_front();
            }
            const bool ok = send_val(fd, MSG_UBATCH) && send_val(fd, msg.pos0) && send_val(fd, msg.n) &&
                            send_all(fd, msg.h, (size_t) msg.n*width*sizeof(float));
            std::lock_guard<std::mutex> lk(m);
            if (!ok) {
                failed = true;
                q.clear();
                cv.notify_all();
                return;
            }
            n_sent++;
            cv.notify_all();
        }
    }

    // false when the connection failed
    bool push(const ubatch_msg & msg) {
        std::lock_guard<std::mutex> lk(m);
        if (failed) {
            return false;
        }
        q.push_back(msg);
        n_pushed++;
        cv.notify_all();
        return true;
    }

    // waits until the first k pushed messages are sent; false when the connection failed
    bool wait_sent(int64_t k) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&]() { return failed || n_sent >= k; });
        return !failed;
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

// the number of nextn output slots libllama reserved (the same rule as llama_context::output_reserve)
int nextn_slots() {
    static const int n = std::max(1, getenv("LLAMA_PREFILL_PEER_NEXTN_SLOTS") ? atoi(getenv("LLAMA_PREFILL_PEER_NEXTN_SLOTS")) : 2);
    return n;
}

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
        cfg.peers.push_back(st);
    }
    const int n_parts = (int) cfg.peers.size() + 1;
    if (getenv("LLAMA_PREFILL_PEER_SPLIT")) {
        for (const auto & x : string_split<std::string>(getenv("LLAMA_PREFILL_PEER_SPLIT"), ',')) {
            cfg.split_all.push_back(atoi(x.c_str()));
        }
    } else {
        for (int k = 1; k < n_parts; ++k) {
            cfg.split_all.push_back((n_layer*k + n_parts/2)/n_parts);
        }
    }
    cfg.split_one = getenv("LLAMA_PREFILL_PEER_SPLIT1") ? atoi(getenv("LLAMA_PREFILL_PEER_SPLIT1")) : n_layer/2;
    bool ok = (int) cfg.split_all.size() == n_parts - 1 && cfg.split_one > 0 && cfg.split_one < n_layer;
    for (int k = 0; ok && k < n_parts - 1; ++k) {
        const int lo = k == 0 ? 0 : cfg.split_all[k - 1];
        ok = cfg.split_all[k] > lo && cfg.split_all[k] < n_layer;
    }
    if (!ok) {
        LOG_WRN("prefill-peer: bad LLAMA_PREFILL_PEER_SPLIT / SPLIT1 for %d peers and %d layers, disabled\n", n_parts - 1, n_layer);
        return {};
    }
    if (getenv("LLAMA_PREFILL_PEER_MIN")) {
        cfg.min_tokens = atoi(getenv("LLAMA_PREFILL_PEER_MIN"));
    }
    for (size_t k = 0; k < cfg.peers.size(); ++k) {
        LOG_WRN("prefill-peer: peer %zu %s:%d, from layer %d in the whole chain\n", k, cfg.peers[k].host.c_str(), cfg.peers[k].port, cfg.split_all[k]);
    }
    LOG_WRN("prefill-peer: head, prompts from %d tokens; a single peer from layer %d\n", cfg.min_tokens, cfg.split_one);
    return cfg;
}

std::vector<std::vector<prefill_peer_stage>> prefill_peer_config::chains() const {
    std::vector<std::vector<prefill_peer_stage>> res;
    std::vector<prefill_peer_stage> all = peers;
    for (size_t k = 0; k < all.size(); ++k) {
        all[k].layer_begin = split_all[k];
    }
    res.push_back(all);
    if (peers.size() > 1) {
        for (const auto & p : peers) {
            prefill_peer_stage st = p;
            st.layer_begin = split_one;
            res.push_back({ st });
        }
    }
    return res;
}

namespace {

// where the stages return their state: the head's result listener
struct result_endpoint {
    std::string host;
    int32_t     port  = 0;
    uint64_t    token = 0;
};

// HELLO: the prompt, the stages from the receiver on (the receiver is stages[0]) and the result endpoint
bool send_hello(int fd, const std::vector<prefill_peer_stage> & stages, const llama_token * tokens, int32_t n_split,
                int32_t n_ubatch, int32_t n_ckpt, const std::vector<uint8_t> & spec_in, int64_t width, const result_endpoint & ep) {
    const uint32_t version = PROTO_VERSION;
    const uint32_t n_st    = stages.size();
    bool ok = send_val(fd, MSG_HELLO) && send_val(fd, version) && send_val(fd, n_split) && send_val(fd, n_ubatch) && send_val(fd, n_ckpt) &&
              send_all(fd, tokens, (size_t) n_split*sizeof(llama_token)) && send_blob(fd, spec_in) && send_val(fd, n_st);
    for (const auto & st : stages) {
        const int32_t port = st.port, lb = st.layer_begin;
        ok = ok && send_string(fd, st.host) && send_val(fd, port) && send_val(fd, lb);
    }
    ok = ok && send_string(fd, ep.host) && send_val(fd, ep.port) && send_val(fd, ep.token);
    uint32_t msg = 0;
    int64_t  width_peer = 0;
    return ok && recv_val(fd, msg) && msg == MSG_OK && recv_val(fd, width_peer) && width_peer == width;
}

// one stage's state as it returns it to the head: its layers' part, and from the last stage the drafter's state
struct stage_result {
    int32_t il_lo = 0;
    int32_t il_hi = -1;
    hbuf data;
    hbuf st_dft;
    std::vector<uint8_t> st_spec;
    // the snapshot before the ubatch at n_ckpt: the layers' recurrent rows, from the last stage the drafter's partial
    // state and pending rows (empty when the stage saw no ubatch starting at n_ckpt)
    hbuf ck_data;
    hbuf ck_dft;
    std::vector<uint8_t> ck_spec;
    double  peer_ms = 0.0;
};

bool send_result(int fd, uint64_t token, const stage_result & r) {
    return send_val(fd, token) && send_val(fd, MSG_RESULT) && send_val(fd, r.il_lo) && send_val(fd, r.il_hi) &&
           send_blob(fd, r.data) && send_blob(fd, r.st_dft) && send_blob(fd, r.st_spec) &&
           send_blob(fd, r.ck_data) && send_blob(fd, r.ck_dft) && send_blob(fd, r.ck_spec) && send_val(fd, r.peer_ms);
}

// after the token: one stage's result (the buffers of r keep their capacity across prompts)
bool recv_result(int fd, stage_result & r) {
    uint32_t msg = 0;
    if (!recv_val(fd, msg) || msg != MSG_RESULT) {
        return false;
    }
    return recv_val(fd, r.il_lo) && recv_val(fd, r.il_hi) && recv_blob(fd, r.data) && recv_blob(fd, r.st_dft) &&
           recv_blob(fd, r.st_spec) && recv_blob(fd, r.ck_data) && recv_blob(fd, r.ck_dft) && recv_blob(fd, r.ck_spec) &&
           recv_val(fd, r.peer_ms);
}

// the status a stage passes back up the chain after its own result went to the head: OK, or an error text
bool send_status(int fd, bool ok, const std::string & err) {
    if (ok) {
        return send_val(fd, MSG_OK);
    }
    return send_val(fd, MSG_ERROR) && send_string(fd, err);
}

bool recv_status(int fd, std::string & err) {
    uint32_t msg = 0;
    if (!recv_val(fd, msg)) {
        err = "connection";
        return false;
    }
    if (msg == MSG_OK) {
        return true;
    }
    if (msg != MSG_ERROR || !recv_string(fd, err)) {
        err = "bad status";
    }
    return false;
}

// the head's receive buffers persist (pre-faulted) across prompts, one per stage in arrival order
std::vector<stage_result> & head_results() {
    static std::vector<stage_result> r;
    return r;
}

// a peer's result buffers persist (pre-faulted) across prompts
stage_result & own_result() {
    static stage_result r;
    return r;
}

// the recurrent rows of layers [il_lo, il_hi) of seq_id, into a buffer that keeps its capacity
void snapshot_partial(llama_context * ctx, llama_seq_id seq_id, int32_t il_lo, int32_t il_hi, hbuf & out) {
    out.resize(llama_state_seq_get_size_layers_ext(ctx, seq_id, il_lo, il_hi, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    out.resize(llama_state_seq_get_data_layers_ext(ctx, out.data(), out.size(), seq_id, il_lo, il_hi, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
}

} // namespace

bool prefill_peer_run(const prefill_peer_config & cfg,
                      llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec,
                      llama_seq_id seq_id, const llama_token * tokens, int32_t n_split, int32_t n_ubatch,
                      int32_t n_ckpt, prefill_peer_ckpt * ckpt, prefill_peer_stats & stats) {
    const double t0 = now_ms();
    stats = {};
    if (ckpt) {
        *ckpt = {};
    }
    if (n_ckpt <= 0 || n_ckpt >= n_split || ckpt == nullptr) {
        n_ckpt = -1;
    }

    const int64_t width = llama_model_n_embd_out(llama_get_model(ctx_tgt));

    auto fail = [&](const char * what) {
        LOG_WRN("prefill-peer: %s, the prompt is decoded here\n", what);
        llama_set_layer_range(ctx_tgt, 0, -1);
        llama_set_embeddings_nextn_slot(ctx_tgt, 0);
        llama_synchronize(ctx_tgt);
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, -1, -1);
        }
        return false;
    };

    // the drafter's pending row and deferred rows as they are here, so the last stage's catch-up starts from them
    std::vector<uint8_t> spec_in;
    if (spec) {
        common_speculative_get_state(spec, seq_id, spec_in);
    }

    // the stages return their state here, each on its own connection, as soon as it is done
    const int lfd = listen_on(0);
    if (lfd < 0) {
        LOG_WRN("prefill-peer: no result listener, the prompt is decoded here\n");
        return false;
    }
    result_endpoint ep;
    ep.port = local_port(lfd);
    {
        std::random_device rd;
        ep.token = ((uint64_t) rd() << 32) ^ (uint64_t) rd() ^ (uint64_t) now_ms();
    }

    // the whole chain, else one peer at a time (a busy peer refuses at once)
    int fd = -1;
    int split = -1;
    size_t n_stages = 0;
    for (const auto & chain : cfg.chains()) {
        const int fdc = connect_to(chain[0].host, chain[0].port);
        if (fdc < 0) {
            LOG_WRN("prefill-peer: cannot connect to %s:%d\n", chain[0].host.c_str(), chain[0].port);
            continue;
        }
        ep.host = local_ip(fdc); // this host as the peers reach it
        if (send_hello(fdc, chain, tokens, n_split, n_ubatch, n_ckpt, spec_in, width, ep)) {
            fd       = fdc;
            split    = chain[0].layer_begin;
            n_stages = chain.size();
            LOG_WRN("prefill-peer: %zu peer(s), layers [0, %d) here, %d tokens, checkpoint at %d; results to %s:%d (t=%" PRId64 ")\n", chain.size(), split, n_split, n_ckpt, ep.host.c_str(), ep.port, wall_ms());
            break;
        }
        ::close(fdc);
        LOG_WRN("prefill-peer: a chain of %zu from %s:%d declined\n", chain.size(), chain[0].host.c_str(), chain[0].port);
    }
    if (fd < 0) {
        ::close(lfd);
        return false;
    }

    // the residual after the last local layer of every token comes out as the nextn embeddings
    llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
    llama_set_layer_range(ctx_tgt, 0, split);

    sender snd;
    snd.start(fd, width);

    // one ubatch per message: the chain fills and drains in one ubatch per stage (the ubatches are the ones the local
    // path decodes; LLAMA_PREFILL_PEER_CHUNK=n sends n ubatches per message)
    const int32_t n_chunk = n_ubatch * std::max(1, getenv("LLAMA_PREFILL_PEER_CHUNK") ? atoi(getenv("LLAMA_PREFILL_PEER_CHUNK")) : 1);
    const int n_slots = nextn_slots();

    common_batch batch(ctx_tgt);
    bool ok = true;
    int64_t i_msg = 0;
    static hbuf head_ck; // this node's layers' recurrent rows before the ubatch at n_ckpt
    head_ck.clear();
    double t_ck_ms = 0.0;
    for (int32_t off = 0; off < n_split && ok; ++i_msg) {
        // the messages are cut at n_ckpt, so the stages see one ubatch that starts there (the local path's last one)
        const int32_t n = std::min(n_chunk, (n_ckpt > off ? n_ckpt : n_split) - off);
        const int32_t pos0 = off;
        off += n;
        batch.clear();
        for (int32_t i = 0; i < n; ++i) {
            batch.add(tokens[pos0 + i], pos0 + i, seq_id, false);
        }
        // the slot this ubatch writes was read by the sender n_slots messages ago: wait for that send
        const double tw0 = now_ms();
        if (!snd.wait_sent(i_msg - (n_slots - 1))) {
            ok = false;
            break;
        }
        stats.t_wait_ms += now_ms() - tw0;
        if (pos0 == n_ckpt) {
            const double tc0 = now_ms();
            snapshot_partial(ctx_tgt, seq_id, 0, split, head_ck);
            t_ck_ms += now_ms() - tc0;
        }
        llama_set_embeddings_nextn_slot(ctx_tgt, (int32_t) (i_msg % n_slots));
        const double tl = now_ms();
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            ok = false;
            break;
        }
        float * h = llama_get_embeddings_nextn(ctx_tgt); // waits for the batch
        if (h == nullptr) {
            ok = false;
            break;
        }
        stats.t_local_ms += now_ms() - tl;
        if (i_msg == 0) {
            LOG_WRN("prefill-peer: first ubatch out after %.0f ms (t=%" PRId64 ")\n", now_ms() - t0, wall_ms());
        }
        ubatch_msg msg;
        msg.pos0 = pos0;
        msg.n    = n;
        msg.h    = h;
        ok = snd.push(msg);
    }
    llama_set_layer_range(ctx_tgt, 0, -1);
    llama_set_embeddings_nextn_slot(ctx_tgt, 0);

    const double tw = now_ms();
    ok = snd.finish() && ok;
    if (!ok) {
        ::close(fd);
        ::close(lfd);
        return fail("a local decode or the connection failed");
    }
    const double t_split_done = now_ms();
    LOG_WRN("prefill-peer: split part sent, END (t=%" PRId64 ")\n", wall_ms());

    // the results: each stage connects here when it is done (the middle stages while the last one still computes), and
    // its part is restored as it lands; the chain's status (OK / error) comes back on fd when every stage is done
    auto & results = head_results();
    if (results.size() < n_stages) {
        results.resize(n_stages);
    }
    bool   chain_status_read = false;
    size_t n_got = 0;
    double t_recv_ms = 0.0;
    std::vector<size_t> got_idx;
    ok = send_val(fd, MSG_END);
    const double t_restore_sum0 = now_ms();
    double t_restore = 0.0;
    while (ok && n_got < n_stages) {
        pollfd pfd[2] = { { lfd, POLLIN, 0 }, { fd, POLLIN, 0 } };
        const int n_poll = chain_status_read ? 1 : 2;
        const int r = ::poll(pfd, n_poll, 600000);
        if (r <= 0) {
            LOG_WRN("prefill-peer: timed out waiting for the stages' results\n");
            ok = false;
            break;
        }
        if (!chain_status_read && (pfd[1].revents & (POLLIN | POLLHUP | POLLERR))) {
            std::string err;
            if (!recv_status(fd, err)) {
                LOG_WRN("prefill-peer: the chain reports: %s\n", err.c_str());
                ok = false;
                break;
            }
            chain_status_read = true;
        }
        if (pfd[0].revents & POLLIN) {
            const int rfd = ::accept(lfd, nullptr, nullptr);
            if (rfd < 0) {
                continue;
            }
            tune_socket(rfd);
            uint64_t token = 0;
            if (!recv_val(rfd, token) || token != ep.token) {
                LOG_WRN("prefill-peer: a result connection with a wrong token, ignored\n");
                ::close(rfd);
                continue;
            }
            const double tr0 = now_ms();
            stage_result & res = results[n_got];
            const bool r_ok = recv_result(rfd, res);
            ::close(rfd);
            if (!r_ok) {
                LOG_WRN("prefill-peer: a stage's result did not arrive whole\n");
                ok = false;
                break;
            }
            const double tr1 = now_ms();
            t_recv_ms += tr1 - tr0;
            stats.bytes_state += res.data.size() + res.st_dft.size() + res.st_spec.size();
            stats.peer_ms = std::max(stats.peer_ms, res.peer_ms);
            if (llama_state_seq_set_data_layers(ctx_tgt, res.data.data(), res.data.size(), seq_id, res.il_lo, res.il_hi) != res.data.size()) {
                ok = false;
                LOG_WRN("prefill-peer: the layers [%d, %d) did not fit the local sequence\n", res.il_lo, res.il_hi);
                break;
            }
            if (!res.st_dft.empty() && ctx_dft) {
                if (llama_state_seq_set_data_ext(ctx_dft, res.st_dft.data(), res.st_dft.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) != res.st_dft.size()) {
                    ok = false;
                    LOG_WRN("prefill-peer: the drafter state did not load\n");
                    break;
                }
            }
            if (!res.st_spec.empty() && spec) {
                common_speculative_set_state(spec, seq_id, res.st_spec);
            }
            t_restore += now_ms() - tr1;
            LOG_WRN("prefill-peer: layers [%d, %d) landed: %.1f MiB received in %.0f ms (%.2f GB/s), restored in %.0f ms (t=%" PRId64 ")\n",
                    res.il_lo, res.il_hi, res.data.size()/1048576.0, tr1 - tr0, (res.data.size() + res.st_dft.size())/1e6/std::max(1.0, tr1 - tr0), now_ms() - tr1, wall_ms());
            n_got++;
        }
    }
    if (ok && !chain_status_read) {
        // every result landed: the chain's status is only a formality now, but read it so the stages finish cleanly
        pollfd pfd = { fd, POLLIN, 0 };
        if (::poll(&pfd, 1, 5000) > 0) {
            std::string err;
            recv_status(fd, err);
        }
    }
    ::close(fd);
    ::close(lfd);
    if (!ok) {
        return fail("no state from the peers");
    }

    // the checkpoint at n_ckpt: the stages' snapshots are written over the recurrent rows (their positions adopted),
    // the whole partial state read back as the checkpoint, then the final recurrent rows restored
    if (n_ckpt > 0) {
        const double tc0 = now_ms();
        static hbuf final_rs;
        bool have = !head_ck.empty();
        const stage_result * last = nullptr;
        for (size_t k = 0; k < n_stages; ++k) {
            have = have && !results[k].ck_data.empty();
            if (!results[k].st_dft.empty() || !results[k].st_spec.empty()) {
                last = &results[k];
            }
        }
        if (!have) {
            LOG_WRN("prefill-peer: no checkpoint at %d: a stage did not snapshot it\n", n_ckpt);
        } else {
            final_rs.resize(llama_state_seq_get_size_ext(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            final_rs.resize(llama_state_seq_get_data_ext(ctx_tgt, final_rs.data(), final_rs.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ADOPT_POS;
            bool ck_ok = llama_state_seq_set_data_layers_ext(ctx_tgt, head_ck.data(), head_ck.size(), seq_id, 0, split, fl) == head_ck.size();
            for (size_t k = 0; ck_ok && k < n_stages; ++k) {
                const auto & r = results[k];
                ck_ok = llama_state_seq_set_data_layers_ext(ctx_tgt, r.ck_data.data(), r.ck_data.size(), seq_id, r.il_lo, r.il_hi, fl) == r.ck_data.size();
            }
            if (ck_ok) {
                ckpt->n_tokens = n_ckpt;
                ckpt->pos_min  = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id);
                ckpt->pos_max  = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id);
                ckpt->data_tgt.resize(llama_state_seq_get_size_ext(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                ckpt->data_tgt.resize(llama_state_seq_get_data_ext(ctx_tgt, ckpt->data_tgt.data(), ckpt->data_tgt.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                if (last) {
                    ckpt->data_dft  = last->ck_dft.to_vector();
                    ckpt->data_spec = last->ck_spec;
                }
            } else {
                LOG_WRN("prefill-peer: no checkpoint at %d: a snapshot did not fit the local sequence\n", n_ckpt);
                ckpt->data_tgt.clear();
            }
            // the final rows again, whatever happened above: without them the prompt is wrong
            if (llama_state_seq_set_data_ext(ctx_tgt, final_rs.data(), final_rs.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != final_rs.size()) {
                return fail("the final recurrent state did not restore after the checkpoint");
            }
        }
        t_ck_ms += now_ms() - tc0;
        LOG_WRN("prefill-peer: checkpoint at %d (pos %d..%d): %.1f MiB target, %.1f MiB drafter, assembled in %.0f ms (t=%" PRId64 ")\n",
                n_ckpt, ckpt->pos_min, ckpt->pos_max, ckpt->data_tgt.size()/1048576.0, ckpt->data_dft.size()/1048576.0, t_ck_ms, wall_ms());
    }

    stats.t_wait_ms    += t_split_done - tw + (now_ms() - t_restore_sum0) - t_restore;
    stats.t_restore_ms  = t_restore;
    stats.t_total_ms    = now_ms() - t0;

    // the receive buffers grow once to what the longest prompt of this context would need (its pages touched now,
    // between prompts, not by the receive of the next one): the drafter's part here, the layers' part at the warm-up
    // (the estimate from a short prompt overshoots it: the state has a constant part beyond the recurrent rows)
    if (rxbuf_on()) {
        for (size_t k = 0; k < n_stages; ++k) {
            auto & r = results[k];
            const double tg0 = now_ms();
            const size_t c0 = r.data.capacity() + r.st_dft.capacity();
            if (r.data.capacity() == 0) {
                grow_rx(r.data, state_estimate(ctx_tgt, r.data.size(), r.ck_data.size(), n_split));
            }
            grow_rx(r.st_dft, state_estimate(ctx_tgt, r.st_dft.size(), r.ck_dft.size(), n_split));
            if (r.data.capacity() + r.st_dft.capacity() != c0) {
                LOG_WRN("prefill-peer: receive buffers of stage %zu grown to %.0f + %.0f MiB in %.0f ms\n", k, r.data.capacity()/1048576.0, r.st_dft.capacity()/1048576.0, now_ms() - tg0);
            }
        }
    }

    LOG_WRN("prefill-peer: %d tokens in %.0f ms (%.0f tok/s): local %.0f ms, waits %.0f ms, peers %.0f ms, state %.1f MiB received in %.0f ms, restored in %.0f ms; handoff (split done -> here) %.0f ms (t=%" PRId64 ")\n",
            n_split, stats.t_total_ms, n_split*1000.0/stats.t_total_ms, stats.t_local_ms, stats.t_wait_ms, stats.peer_ms,
            stats.bytes_state/1048576.0, t_recv_ms, stats.t_restore_ms, now_ms() - t_split_done, wall_ms());
    return true;
}

namespace {

// one head's prompt on a peer: its layers, then the next stage's (forwarded) or the drafter's catch-up (last stage);
// the state goes straight to the head, the chain passes only a status back
void serve_one(int fd, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec) {
    const llama_model * model = llama_get_model(ctx_tgt);
    const int64_t width = llama_model_n_embd_out(model);
    const llama_seq_id seq_id = 0;

    uint32_t msg = 0, version = 0, n_st = 0;
    int32_t  n_split = 0, n_ubatch = 0, n_ckpt = -1;
    if (!recv_val(fd, msg) || msg != MSG_HELLO || !recv_val(fd, version) || version != PROTO_VERSION ||
        !recv_val(fd, n_split) || !recv_val(fd, n_ubatch) || !recv_val(fd, n_ckpt) || n_split <= 0 || n_ubatch <= 0) {
        LOG_WRN("prefill-peer: bad hello (version %u, this build speaks %u)\n", version, PROTO_VERSION);
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
        int32_t port = 0, lb = 0;
        if (!recv_string(fd, st.host) || !recv_val(fd, port) || !recv_val(fd, lb)) {
            return;
        }
        st.port        = port;
        st.layer_begin = lb;
    }
    result_endpoint ep;
    if (!recv_string(fd, ep.host) || !recv_val(fd, ep.port) || !recv_val(fd, ep.token)) {
        return;
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
        if (fd_next < 0 || !send_hello(fd_next, rest, tokens.data(), n_split, n_ubatch, n_ckpt, spec_in, width, ep)) {
            LOG_WRN("prefill-peer: the next stage %s:%d did not accept the prompt\n", stages[1].host.c_str(), stages[1].port);
            if (fd_next >= 0) {
                ::close(fd_next);
            }
            return;
        }
    }

    // the pinned slots the residual rows land in (allocated once, at the warm-up or the first prompt)
    rx_ring & ring = the_rx_ring();
    ring.prepare(model, width, (size_t) n_ubatch);

    if (!send_val(fd, MSG_OK) || !send_val(fd, width)) {
        if (fd_next >= 0) {
            ::close(fd_next);
        }
        return;
    }

    LOG_WRN("prefill-peer: %d tokens, layers [%d, %d) here%s, checkpoint at %d (t=%" PRId64 ")\n", n_split, il_lo, il_hi, last ? " + the drafter" : "", n_ckpt, wall_ms());

    llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
    llama_set_layer_range(ctx_tgt, il_lo, il_hi);

    sender snd;
    if (!last) {
        snd.start(fd_next, width);
    }
    const int n_slots = nextn_slots();
    // WP10: the last stage's drafter catch-up overlaps the next ubatch when the rows are double-buffered
    const bool catchup_async = last && spec != nullptr && common_speculative_catchup_async(spec) &&
                               llama_get_embeddings_nextn_slots(ctx_tgt) >= 2 && n_slots >= 2;

    const double t0 = now_ms();
    double t_compute = 0.0;
    common_batch batch(ctx_tgt);
    std::string err;

    // a receiver thread reads the next ubatch into a pinned slot while this one computes
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
                std::unique_lock<std::mutex> lk(rx.m);
                rx.cv.wait(lk, [&]() { return rx.stop || (u.slot = ring.take((size_t) u.n)) >= 0; });
                if (rx.stop) {
                    return;
                }
                u.h = ring.slots[u.slot].data;
            }
            good = good && recv_all(fd, u.h, (size_t) u.n*width*sizeof(float));
            std::unique_lock<std::mutex> lk(rx.m);
            if (!good) {
                if (u.slot >= 0) {
                    ring.busy[u.slot] = false;
                }
                rx.failed = true;
                rx.cv.notify_all();
                return;
            }
            rx.q.push_back(u);
            rx.cv.notify_all();
        }
    });

    // this stage's result; its buffers keep their capacity across prompts (no page faults on the get)
    stage_result & own = own_result();
    own.ck_data.clear();
    own.ck_dft.clear();
    own.ck_spec.clear();
    double t_ck = 0.0;

    bool ok = true;
    int64_t i_msg = 0;
    while (ok) {
        ubatch_msg u;
        const double tq0 = now_ms();
        {
            std::unique_lock<std::mutex> lk(rx.m);
            rx.cv.wait(lk, [&]() { return !rx.q.empty() || rx.end || rx.failed; });
            if (rx.q.empty()) {
                ok = !rx.failed;
                break;
            }
            u = rx.q.front();
            rx.q.pop_front();
        }
        const double t_qwait = now_ms() - tq0;
        const int32_t pos0 = u.pos0, n = u.n;
        // the nextn slot this ubatch writes was read by the sender n_slots messages ago
        if (!last && !snd.wait_sent(i_msg - (n_slots - 1))) {
            ok = false;
            err = "the next stage is gone";
            break;
        }
        // the last stage: the drafter's catch-up of message i_msg - 1 runs on its thread and reads the other slot; the
        // one that read this slot (i_msg - n_slots) must be done (WP10, common_speculative_process_async); the
        // checkpoint snapshot below needs the whole draft context, so every pending job joins first
        if (last && catchup_async && !common_speculative_catchup_wait(spec, pos0 == n_ckpt && n_ckpt > 0 ? i_msg - 1 : i_msg - n_slots)) {
            ok = false;
            err = "drafter catch-up failed";
            break;
        }
        if (pos0 == n_ckpt && n_ckpt > 0) {
            // the state before this ubatch: what the local path's checkpoint here holds of these layers
            const double tk = now_ms();
            snapshot_partial(ctx_tgt, seq_id, il_lo, il_hi, own.ck_data);
            if (last) {
                if (ctx_dft) {
                    own.ck_dft.resize(llama_state_seq_get_size_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                    own.ck_dft.resize(llama_state_seq_get_data_ext(ctx_dft, own.ck_dft.data(), own.ck_dft.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }
                if (spec) {
                    common_speculative_get_state(spec, seq_id, own.ck_spec);
                }
            }
            t_ck = now_ms() - tk;
        }
        const double tc = now_ms();
        llama_set_hc_input(ctx_tgt, u.h, pos0, n);
        llama_set_embeddings_nextn_slot(ctx_tgt, (int32_t) (i_msg % n_slots));
        batch.clear();
        for (int32_t i = 0; i < n; ++i) {
            batch.add(tokens[pos0 + i], pos0 + i, seq_id, false);
        }
        const int rc = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        // the rows were copied to the device inside llama_process: the slot takes the next message
        {
            std::lock_guard<std::mutex> lk(rx.m);
            ring.busy[u.slot] = false;
        }
        rx.cv.notify_all();
        if (rc != 0) {
            ok = false;
            err = "decode failed";
            break;
        }
        if (!last) {
            float * h = llama_get_embeddings_nextn(ctx_tgt);
            if (h == nullptr) {
                ok = false;
                err = "no residual";
                break;
            }
            t_compute += now_ms() - tc;
            if (i_msg < 2) {
                LOG_WRN("prefill-peer: ubatch %" PRId64 " (pos %d): waited %.0f ms, layers in %.0f ms (t=%" PRId64 ")\n", i_msg, pos0, t_qwait, now_ms() - tc, wall_ms());
            }
            ubatch_msg out;
            out.pos0 = pos0;
            out.n    = n;
            out.h    = h;
            if (!snd.push(out)) {
                ok = false;
                err = "the next stage is gone";
                break;
            }
            i_msg++;
            continue;
        }
        const double td = now_ms();
        if (spec && catchup_async) {
            const float * h = llama_get_embeddings_nextn(ctx_tgt); // waits for the ubatch: its rows are on the host
            if (common_speculative_process_async(spec, batch, h) < 0) {
                ok = false;
                err = "drafter catch-up failed";
                break;
            }
        } else if (spec && !common_speculative_process(spec, batch)) {
            ok = false;
            err = "drafter catch-up failed";
            break;
        }
        t_compute += now_ms() - tc;
        if (i_msg < 2) {
            LOG_WRN("prefill-peer: ubatch %" PRId64 " (pos %d): waited %.0f ms, layers in %.0f ms, drafter %.0f ms (t=%" PRId64 ")\n", i_msg, pos0, t_qwait, td - tc, now_ms() - td, wall_ms());
        }
        i_msg++;
    }
    if (spec && catchup_async && !common_speculative_catchup_join(spec)) {
        ok = false;
        err = "drafter catch-up failed";
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
    {
        std::lock_guard<std::mutex> lk(rx.m);
        ring.busy.assign(ring.busy.size(), false);
    }
    llama_set_layer_range(ctx_tgt, 0, -1);
    llama_set_embeddings_nextn_slot(ctx_tgt, 0);
    llama_synchronize(ctx_tgt);
    if (ctx_dft) {
        llama_synchronize(ctx_dft);
    }
    const double t_last = now_ms();

    if (!last) {
        if (ok) {
            ok = snd.finish() && send_val(fd_next, MSG_END);
            if (!ok) {
                err = "the next stage is gone";
            }
        } else {
            snd.finish();
        }
    }

    // this stage's part of the state goes to the head now, on its own connection: the stages behind still compute
    double t_get = 0.0, t_send = 0.0;
    size_t own_bytes = 0;
    if (ok) {
        const double tg = now_ms();
        own.il_lo = il_lo;
        own.il_hi = il_hi;
        own.data.resize(llama_state_seq_get_size_layers(ctx_tgt, seq_id, il_lo, il_hi));
        own.data.resize(llama_state_seq_get_data_layers(ctx_tgt, own.data.data(), own.data.size(), seq_id, il_lo, il_hi));
        own.st_dft.clear();
        own.st_spec.clear();
        if (last) {
            if (ctx_dft) {
                own.st_dft.resize(llama_state_seq_get_size_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
                own.st_dft.resize(llama_state_seq_get_data_ext(ctx_dft, own.st_dft.data(), own.st_dft.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
            }
            if (spec) {
                common_speculative_get_state(spec, seq_id, own.st_spec);
            }
        }
        own.peer_ms = now_ms() - t0;
        own_bytes   = own.data.size() + own.st_dft.size() + own.st_spec.size() + own.ck_data.size() + own.ck_dft.size() + own.ck_spec.size();
        t_get = now_ms() - tg;
        const double ts = now_ms();
        const int rfd = connect_to(ep.host, ep.port);
        if (rfd < 0 || !send_result(rfd, ep.token, own)) {
            ok  = false;
            err = "the head did not take the result";
        }
        if (rfd >= 0) {
            ::shutdown(rfd, SHUT_WR);
            uint8_t b = 0;
            ::recv(rfd, &b, 1, 0); // the head closes after it read everything
            ::close(rfd);
        }
        t_send = now_ms() - ts;
    }

    // the status of the stages behind, then this stage's status up the chain
    if (!last && fd_next >= 0) {
        if (ok) {
            std::string e;
            if (!recv_status(fd_next, e)) {
                ok  = false;
                err = "the next stage: " + e;
            }
        }
        ::close(fd_next);
    }
    if (!ok && err.empty()) {
        err = "connection";
    }
    send_status(fd, ok, err);

    if (!ok) {
        LOG_WRN("prefill-peer: prompt abandoned (%s)\n", err.c_str());
        return;
    }
    LOG_WRN("prefill-peer: %d tokens done in %.0f ms (compute %.0f ms, checkpoint snapshot %.1f MiB in %.0f ms), own state %.1f MiB: get %.0f ms, sent to the head in %.0f ms (%.2f GB/s), last kernel -> sent %.0f ms (t=%" PRId64 ")\n",
            n_split, own.peer_ms, t_compute, (own.ck_data.size() + own.ck_dft.size())/1048576.0, t_ck, own_bytes/1048576.0, t_get, t_send, own_bytes/1e6/std::max(1.0, t_send), now_ms() - t_last, wall_ms());
    // the drafter's buffer grows once to the longest prompt's (the layers' at the warm-up)
    if (rxbuf_on()) {
        const double tg0 = now_ms();
        const size_t c0 = own.data.capacity() + own.st_dft.capacity();
        if (own.data.capacity() == 0) {
            grow_rx(own.data, state_estimate(ctx_tgt, own.data.size(), own.ck_data.size(), n_split));
        }
        grow_rx(own.st_dft, state_estimate(ctx_tgt, own.st_dft.size(), own.ck_dft.size(), n_split));
        if (own.data.capacity() + own.st_dft.capacity() != c0) {
            LOG_WRN("prefill-peer: own state buffers grown to %.0f + %.0f MiB in %.0f ms\n", own.data.capacity()/1048576.0, own.st_dft.capacity()/1048576.0, now_ms() - tg0);
        }
    }
}

} // namespace

void prefill_peer_warmup(llama_context * ctx_tgt, const prefill_peer_config * cfg) {
    const llama_model * model = llama_get_model(ctx_tgt);
    const int n_layer  = llama_model_n_layer(model);
    const int n_ubatch = llama_n_ubatch(ctx_tgt);
    const int64_t width = llama_model_n_embd_out(model);

    // LLAMA_PREFILL_PEER_WARM=lo:hi,... or 1 = every layer range this server can serve or send from
    std::vector<std::pair<int, int>> ranges;
    const char * w = getenv("LLAMA_PREFILL_PEER_WARM");
    if (w != nullptr && strchr(w, ':') != nullptr) {
        for (const auto & item : string_split<std::string>(w, ',')) {
            const auto colon = item.find(':');
            if (colon != std::string::npos) {
                ranges.push_back({ atoi(item.c_str()), atoi(item.c_str() + colon + 1) });
            }
        }
    } else if (cfg && cfg->enabled()) {
        const auto & s = cfg->split_all;
        for (size_t k = 0; k < s.size(); ++k) {
            ranges.push_back({ s[k], k + 1 < s.size() ? s[k + 1] : n_layer }); // a peer of the whole chain
        }
        ranges.push_back({ cfg->split_one, n_layer });                       // a single peer
        ranges.push_back({ 0, s[0] });                                        // the head of the chain
        if (cfg->split_one != s[0]) {
            ranges.push_back({ 0, cfg->split_one });                          // the head with a single peer
        }
    } else {
        ranges = { { n_layer/3, 2*n_layer/3 }, { 2*n_layer/3, n_layer }, { n_layer/2, n_layer } };
    }
    if (ranges.empty()) {
        return;
    }

    const double t0 = now_ms();
    rx_ring & ring = the_rx_ring();
    ring.prepare(model, width, (size_t) n_ubatch); // pinned slots, zero-filled: the residual input of the warm-up

    const llama_token tok = llama_vocab_bos(llama_model_get_vocab(model)) >= 0 ? llama_vocab_bos(llama_model_get_vocab(model)) : 0;
    common_batch batch(ctx_tgt);
    size_t est_rx = 0, est_own = 0;
    for (const auto & range : ranges) {
        const auto [lo, hi] = range;
        if (lo < 0 || lo >= n_layer || (hi >= 0 && hi <= lo)) {
            continue;
        }
        const double t1 = now_ms();
        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_layer_range(ctx_tgt, lo, hi >= n_layer ? -1 : hi);
        if (lo > 0) {
            llama_set_hc_input(ctx_tgt, ring.slots[0].data, 0, n_ubatch);
        }
        batch.clear();
        for (int i = 0; i < n_ubatch; ++i) {
            batch.add(tok, i, 0, false);
        }
        const int rc = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        llama_synchronize(ctx_tgt);
        // the state of the longest prompt for this range (the per-token part scaled to n_ctx): what this node receives
        // from a stage of its chain as a head (est_rx), or gets and sends as a peer (est_own)
        if (rc == 0 && rxbuf_on()) {
            const int32_t hi_l = hi >= n_layer ? -1 : hi;
            const size_t s_all = llama_state_seq_get_size_layers(ctx_tgt, 0, lo, hi_l);
            const size_t s_rs  = llama_state_seq_get_size_layers_ext(ctx_tgt, 0, lo, hi_l, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            const size_t est   = (size_t) (state_estimate(ctx_tgt, s_all, s_rs, n_ubatch)*1.05);
            const size_t i_stage = &range - ranges.data();
            if (cfg && cfg->enabled() && i_stage < cfg->split_all.size()) {
                est_rx = std::max(est_rx, est);
            }
            est_own = std::max(est_own, est);
        }
        llama_memory_clear(llama_get_memory(ctx_tgt), true);
        LOG_WRN("prefill-peer: warm-up of layers [%d, %d) with %d tokens: %.0f ms%s\n", lo, hi, n_ubatch, now_ms() - t1, rc == 0 ? "" : " FAILED");
    }
    llama_set_layer_range(ctx_tgt, 0, -1);
    llama_set_embeddings_nextn_slot(ctx_tgt, 0);
    llama_synchronize(ctx_tgt);
    if (est_rx > 0 || est_own > 0) {
        const double t1 = now_ms();
        auto & results = head_results();
        if (est_rx > 0) {
            if (results.size() < cfg->split_all.size()) {
                results.resize(cfg->split_all.size());
            }
            for (auto & r : results) {
                prefault(r.data, est_rx);
            }
        }
        prefault(own_result().data, est_own);
        LOG_WRN("prefill-peer: state buffers pre-faulted: %zu receive buffers of %.0f MiB, the own one %.0f MiB, in %.0f ms\n", results.size(), est_rx/1048576.0, est_own/1048576.0, now_ms() - t1);
    }
    LOG_WRN("prefill-peer: warm-up done in %.0f ms\n", now_ms() - t0);
}

void prefill_peer_serve_conn(int fd, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec) {
    tune_socket(fd);
    serve_one(fd, ctx_tgt, ctx_dft, spec);
    // nothing of the head's prompt stays behind
    llama_memory_clear(llama_get_memory(ctx_tgt), true);
    if (ctx_dft) {
        llama_memory_clear(llama_get_memory(ctx_dft), true);
    }
    ::close(fd);
}

void prefill_peer_refuse(int fd) {
    send_val(fd, MSG_ERROR) && send_string(fd, "busy");
    ::shutdown(fd, SHUT_WR);
    ::close(fd);
}

int prefill_peer_listen(int port) {
    return listen_on(port);
}

int prefill_peer_accept(int lfd) {
    return ::accept(lfd, nullptr, nullptr);
}

void prefill_peer_serve(int port, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec) {
    const int lfd = listen_on(port);
    if (lfd < 0) {
        LOG_ERR("prefill-peer: cannot listen on port %d\n", port);
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
