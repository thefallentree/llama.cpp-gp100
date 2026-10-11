#include "expert-cache.cuh"
#include "mmid-f16-sm60.cuh"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

#include "fn-prof.h"
FN_PROF_DECL("ec");

#define EC_MAX_LAYERS  128
#define EC_MAX_EXPERTS 1024
#define EC_MAX_SWAPS   8192  // per update
#define EC_SWAP_GRID   16    // blocks of the exchange kernel: PCIe is saturated from about 16 SMs
#define EC_SWAP_TPB    256
#define EC_SCAN_TPB    512
#define EC_RES_CAND    7     // exchanges of a layer that come back with its counts: a decode window has fewer

// one slice exchange: n16 16-byte words. The incoming slice goes from host[h] to the spare VRAM slot dev_to, the
// evicted one from dev to the spare host slot host_to (spare slots: nothing reads them meanwhile, so the copies
// need no wait from the graphs); without spares (dev_to == dev) the two are swapped in place.
struct ec_job {
    uint4 *  dev;
    uint4 *  host;
    uint4 *  dev_to;
    uint4 *  host_to;
    uint32_t n16;
    uint32_t pad[3];
};

// the new positions of the two experts of an exchange (indexes into the table of all layers)
struct ec_patch {
    int32_t idx_in;
    int32_t pos_in;
    int32_t idx_out;
    int32_t pos_out;
};

struct ec_row {
    int32_t n_hot;
    int32_t n_all; // 0: no layer
};

// a cold expert that was routed to, and the idle expert whose VRAM slot it can take
struct ec_cand {
    float   gain;
    int32_t e_in;
    int32_t e_out;
    int32_t pad;
};

// what an update needs of a layer
struct ec_res {
    uint32_t total;  // routed pairs since the last update
    uint32_t cold;   // those to cold experts
    uint32_t n_cand;
    uint32_t pad;
    ec_cand  cand[EC_RES_CAND]; // by gain; the ones after these are in the layer's row of ec_device::more_dev
};

struct ec_layer {
    char * hot[3]  = {};          // up, gate, down: the slices of the experts in VRAM
    char * cold[3] = {};          // their slices in pinned host memory
    size_t nb2[3]  = {};          // bytes per expert
    int    n_hot   = 0;           // VRAM slots (the experts there plus the spares)
    int    n_cold  = 0;           // host slots
    int    n_expert = 0;          // the experts: n_hot + n_cold - 2*n_spare
    bool   dead    = false;       // its buffer was freed
    bool   perm_dirty = false;    // the row on the device is not this perm (registered during a capture)
    std::vector<int32_t> perm;    // expert -> position: a VRAM slot below n_hot, else n_hot + host slot
    // spare slots (LLAMA_EXPERT_SPARE): free now, and freed by the last exchange (free once it is done)
    std::vector<int32_t> free_v, free_h, pend_v, pend_h;
};

struct ec_device {
    std::mutex            mtx;
    std::vector<ec_layer> layers;
    std::atomic<int>      n_layers { 0 };
    std::atomic<bool>     perm_dirty { false }; // a layer's perm registered during a capture is not on the device yet
    std::atomic<bool>     ready { false }; // the tables exist
    int32_t *    perm_dev    = nullptr;  // [EC_MAX_LAYERS][EC_MAX_EXPERTS]
    uint32_t *   counts_dev  = nullptr;  // likewise: how often each expert was routed to
    uint32_t *   seen_dev    = nullptr;  // likewise: the counts at the last update
    float *      freq_dev    = nullptr;  // likewise: decayed use count
    ec_cand *    more_dev    = nullptr;  // likewise: the exchanges of a layer, by gain
    ec_row *     rows_dev    = nullptr;  // [EC_MAX_LAYERS]
    ec_res *     res_dev     = nullptr;  // [EC_MAX_LAYERS]
    ec_job *     jobs_dev    = nullptr;  // the jobs of an update, then its patches
    ec_res *     res_host    = nullptr;  // pinned copies
    ec_job *     jobs_host   = nullptr;
    size_t       n_rows_dev  = 0;        // layers whose sizes and first use counts are on the device
    bool         rows_dirty  = false;    // a layer died since
    const void * owner       = nullptr;  // the context that updates the layers
    bool         frozen      = false;    // another context uses them too
    uint64_t     n_updates   = 0;
    uint64_t     n_swaps     = 0;
    int          n_used      = 0; // experts per token, from the routing
    int          last_prompt_tokens = 0; // tokens of the last update if it was a prompt ubatch
    uint64_t     n_pairs     = 0;
    uint64_t     n_pairs_cold = 0;
    std::vector<uint64_t> row_pairs, row_cold; // per layer (GGML_CUDA_EXPERT_CACHE_STATS)
};

static ec_device g_ec[GGML_CUDA_MAX_DEVICES];

static bool ec_enabled() {
    static const bool enabled = getenv("GGML_CUDA_EXPERT_CACHE") == nullptr || atoi(getenv("GGML_CUDA_EXPERT_CACHE")) != 0;
    return enabled;
}

// ids -> positions, and the routing counts
static __global__ void ec_remap(int32_t * ids, const int si1, const int n_used, const int n,
                                const int32_t * __restrict__ perm, uint32_t * counts) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int32_t * p = ids + (i / n_used)*si1 + i % n_used;
    const int e = *p;
    atomicAdd(counts + e, 1u);
    *p = perm[e];
}

// exchanges the slices of the jobs between VRAM and the host and writes the new positions; block b takes the jobs
// b, b + grid, ...
static __global__ void ec_swap(const ec_job * __restrict__ jobs, const int n_jobs, const ec_patch * __restrict__ patches,
                               const int n_patches, int32_t * __restrict__ perm) {
    for (int j = blockIdx.x; j < n_jobs; j += gridDim.x) {
        const ec_job job = jobs[j];
        for (uint32_t i = threadIdx.x; i < job.n16; i += blockDim.x) {
            const uint4 h = job.host[i];
            const uint4 d = job.dev[i];
            job.dev_to[i]  = h;
            job.host_to[i] = d;
        }
    }
    if (threadIdx.x == 0) {
        for (int j = blockIdx.x; j < n_patches; j += gridDim.x) {
            const ec_patch p = patches[j];
            perm[p.idx_in]  = p.pos_in;
            perm[p.idx_out] = p.pos_out;
        }
    }
    // the host threads read the slices in the next graph
    __threadfence_system();
}

// One block per layer: the use counts of its experts since the last update, and which cold experts that were routed to
// could take the VRAM slot of which idle one. The host only reads the result: its own copies of these tables were
// 400 KB per device that had left its caches by the end of every window.
// The frequency of an expert is its count per update, decayed per update. After a prompt ubatch the counts are those
// of a thousand tokens and a decode window adds one to three: the decode spent a hundred windows at 12-19% cold pairs
// before its own counts mattered (16K prompt). So the frequencies are scaled down by the ratio of the window sizes
// when a decode follows a prompt (the shares of the prompt's tokens stay, in the decode's units).
static __global__ void ec_scan(const ec_row * __restrict__ rows, const uint32_t * __restrict__ counts, uint32_t * __restrict__ seen,
                               float * __restrict__ freq, const int32_t * __restrict__ perm, const float decay, const float min_gain,
                               const float rescale, ec_res * __restrict__ res, ec_cand * __restrict__ more) {
    __shared__ float    s_freq[EC_MAX_EXPERTS];
    __shared__ uint32_t s_delta[EC_MAX_EXPERTS];
    __shared__ int32_t  s_in[EC_MAX_EXPERTS];   // rank -> expert
    __shared__ int32_t  s_out[EC_MAX_EXPERTS];
    __shared__ uint8_t  s_kind[EC_MAX_EXPERTS]; // 1: cold and routed to, 2: in VRAM and idle
    __shared__ uint32_t s_sum[5];               // total, cold, the experts of kind 1, of kind 2, exchanges

    const int    row  = blockIdx.x;
    const int    tid  = threadIdx.x;
    const ec_row r    = rows[row];
    const size_t base = (size_t) row*EC_MAX_EXPERTS;

    if (tid < 5) {
        s_sum[tid] = 0;
    }
    __syncthreads();
    uint32_t total = 0;
    for (int e = tid; e < r.n_all; e += blockDim.x) {
        const uint32_t d = counts[base + e] - seen[base + e];
        s_delta[e] = d;
        total += d;
    }
    if (total != 0) {
        atomicAdd(&s_sum[0], total);
    }
    __syncthreads();
    total = s_sum[0];
    if (total == 0) {
        // a layer that was not computed keeps its counts as they are
        if (tid == 0) {
            res[row].total  = 0;
            res[row].cold   = 0;
            res[row].n_cand = 0;
        }
        return;
    }
    uint32_t cold = 0, n_in = 0, n_out = 0;
    for (int e = tid; e < r.n_all; e += blockDim.x) {
        const uint32_t d = s_delta[e];
        const float    f = freq[base + e]*decay*rescale + (float) d;
        freq[base + e]  = f;
        seen[base + e] += d;
        s_freq[e] = f;
        const bool hot  = perm[base + e] < r.n_hot;
        const int  kind = !hot && d != 0 ? 1 : hot && d == 0 ? 2 : 0;
        s_kind[e] = (uint8_t) kind;
        cold  += hot ? 0 : d;
        n_in  += kind == 1;
        n_out += kind == 2;
    }
    atomicAdd(&s_sum[1], cold);
    atomicAdd(&s_sum[2], n_in);
    atomicAdd(&s_sum[3], n_out);
    __syncthreads();
    const uint32_t n_pair = min(s_sum[2], s_sum[3]);
    // the most used cold expert against the least used idle one, and so on: the rank of an expert in its list
    for (int e = tid; e < r.n_all && n_pair > 0; e += blockDim.x) {
        const int kind = s_kind[e];
        if (kind == 0) {
            continue;
        }
        const float f = s_freq[e];
        uint32_t rank = 0;
        if (kind == 1) {
            for (int j = 0; j < r.n_all; ++j) {
                rank += s_kind[j] == 1 && (s_freq[j] > f || (s_freq[j] == f && j < e));
            }
            s_in[rank] = e;
        } else {
            for (int j = 0; j < r.n_all; ++j) {
                rank += s_kind[j] == 2 && (s_freq[j] < f || (s_freq[j] == f && j < e));
            }
            if (rank < n_pair) {
                s_out[rank] = e;
            }
        }
    }
    __syncthreads();
    // the gains do not grow with the rank: the exchanges worth their copy are the first ones
    uint32_t n_cand = 0;
    for (uint32_t i = tid; i < n_pair; i += blockDim.x) {
        const ec_cand c = { s_freq[s_in[i]] - s_freq[s_out[i]], s_in[i], s_out[i], 0 };
        if (c.gain > min_gain) {
            if (i < EC_RES_CAND) {
                res[row].cand[i] = c;
            } else {
                more[base + i] = c;
            }
            n_cand++;
        }
    }
    if (n_cand != 0) {
        atomicAdd(&s_sum[4], n_cand);
    }
    __syncthreads();
    if (tid == 0) {
        res[row].total  = total;
        res[row].cold   = s_sum[1];
        res[row].n_cand = s_sum[4];
    }
}

static bool ec_alloc(ec_device & d, const int device) {
    if (d.perm_dev != nullptr) {
        return true;
    }
    ggml_cuda_set_device(device);
    const size_t n_tab     = (size_t) EC_MAX_LAYERS*EC_MAX_EXPERTS;
    const size_t jobs_size = (size_t) EC_MAX_SWAPS*(3*sizeof(ec_job) + sizeof(ec_patch));
    void * hp = nullptr;
    if (cudaMallocHost(&hp, EC_MAX_LAYERS*sizeof(ec_res) + jobs_size) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    d.res_host  = (ec_res *) hp;
    d.jobs_host = (ec_job *) (d.res_host + EC_MAX_LAYERS);
    CUDA_CHECK(cudaMalloc((void **) &d.perm_dev,   n_tab*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc((void **) &d.counts_dev, n_tab*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **) &d.seen_dev,   n_tab*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **) &d.freq_dev,   n_tab*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d.more_dev,   n_tab*sizeof(ec_cand)));
    CUDA_CHECK(cudaMalloc((void **) &d.rows_dev,   EC_MAX_LAYERS*sizeof(ec_row)));
    CUDA_CHECK(cudaMalloc((void **) &d.res_dev,    EC_MAX_LAYERS*sizeof(ec_res)));
    CUDA_CHECK(cudaMalloc((void **) &d.jobs_dev,   jobs_size));
    CUDA_CHECK(cudaMemset(d.counts_dev, 0, n_tab*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d.seen_dev,   0, n_tab*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d.rows_dev,   0, EC_MAX_LAYERS*sizeof(ec_row)));
    // every row starts as the identity: a registration then writes nothing to the device and may happen while a
    // CUDA graph is being captured
    std::vector<int32_t> perm(n_tab);
    for (size_t i = 0; i < n_tab; ++i) {
        perm[i] = (int32_t) (i % EC_MAX_EXPERTS);
    }
    CUDA_CHECK(cudaMemcpy(d.perm_dev, perm.data(), n_tab*sizeof(int32_t), cudaMemcpyHostToDevice));
    return true;
}

// do the layers have spare slots? (then the exchanges never hold a graph)
static bool n_spare_any(const ec_device & d) {
    for (const ec_layer & l : d.layers) {
        if (!l.free_v.empty() || !l.pend_v.empty()) {
            return true;
        }
    }
    return false;
}

// the sizes of the layers that registered since the last update and their first use counts; no layer where one died
static void ec_rows_sync(ec_device & d) {
    const size_t n_rows = d.layers.size();
    if (d.n_rows_dev == n_rows && !d.rows_dirty) {
        return;
    }
    std::vector<ec_row> rows(n_rows);
    std::vector<float>  prior(EC_MAX_EXPERTS);
    for (size_t row = 0; row < n_rows; ++row) {
        const ec_layer & l = d.layers[row];
        const int n_all = l.n_expert;
        rows[row] = { l.n_hot, l.dead ? 0 : n_all };
        if (row >= d.n_rows_dev) {
            // the file's order is by use in a calibration text: a weak prior
            for (int e = 0; e < n_all; ++e) {
                prior[e] = 0.5f*(float) (n_all - e)/(float) n_all;
            }
            CUDA_CHECK(cudaMemcpy(d.freq_dev + row*EC_MAX_EXPERTS, prior.data(), n_all*sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaMemcpy(d.rows_dev, rows.data(), n_rows*sizeof(ec_row), cudaMemcpyHostToDevice));
    d.n_rows_dev = n_rows;
    d.rows_dirty = false;
}

void ggml_cuda_expert_cache_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    ec_device & d = g_ec[ctx.device];
    if (!ec_enabled() || d.ready.load(std::memory_order_acquire)) {
        return;
    }
    bool cold = false;
    for (int i = 0; i < cgraph->n_nodes && !cold; ++i) {
        cold = ggml_cuda_mmid_cold(cgraph->nodes[i]);
    }
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &status));
    if (!cold || status != cudaStreamCaptureStatusNone) {
        return;
    }
    ggml_cuda_expert_cache_join(ctx);
    std::lock_guard<std::mutex> lock(d.mtx);
    if (ec_alloc(d, ctx.device)) {
        d.ready.store(true, std::memory_order_release);
    }
}

bool ggml_cuda_expert_cache_known(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    ec_device & d = g_ec[ctx.device];
    if (d.n_layers.load(std::memory_order_acquire) == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(d.mtx);
    for (const ec_layer & l : d.layers) {
        if (!l.dead && (l.hot[0] == w->data || l.hot[1] == w->data || l.hot[2] == w->data)) {
            // the context that computes with the layer
            if (d.owner == nullptr) {
                d.owner = &ctx;
            } else if (d.owner != &ctx && !d.frozen) {
                d.frozen = true;
                GGML_LOG_WARN("%s: a second context uses the MoE layers of device %d: their hot set stays as it is\n", __func__, ctx.device);
            }
            return true;
        }
    }
    return false;
}

void ggml_cuda_expert_cache_register(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
                                     const ggml_tensor * down, const char * c_up, const char * c_gate, const char * c_down,
                                     const int n_cold) {
    ec_device & d = g_ec[ctx.device];
    if (!d.ready.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(d.mtx);
    const ggml_tensor * w[3]    = { up->src[0], gate->src[0], down->src[0] };
    const char *        cold[3] = { c_up, c_gate, c_down };
    const int n_hot = (int) down->src[0]->ne[2];
    // the spare slots at the end of both tensors (llama-model: LLAMA_EXPERT_SPARE pads them past the file's data)
    static const int n_spare_env = getenv("LLAMA_EXPERT_SPARE") != nullptr ? std::max(0, atoi(getenv("LLAMA_EXPERT_SPARE"))) : 4;
    const int n_spare = std::min(n_spare_env, std::min(n_hot, n_cold) - 1);
    bool ok = n_hot + n_cold <= EC_MAX_EXPERTS && n_cold > 0 && d.layers.size() < EC_MAX_LAYERS && n_spare >= 0;
    for (int k = 0; k < 3 && ok; ++k) {
        ok = w[k]->ne[2] == n_hot && w[k]->nb[2] % 16 == 0 && ((uintptr_t) w[k]->data & 0xF) == 0 &&
             ((uintptr_t) cold[k] & 0xF) == 0 && cold[k] != nullptr;
    }
    for (const ec_layer & l : d.layers) {
        ok = ok && (l.dead || l.hot[2] != w[2]->data);
    }
    if (!ok) {
        return;
    }
    ec_layer l;
    for (int k = 0; k < 3; ++k) {
        l.hot[k]  = (char *) w[k]->data;
        l.cold[k] = (char *) cold[k];
        l.nb2[k]  = w[k]->nb[2];
    }
    l.n_hot    = n_hot;
    l.n_cold   = n_cold;
    l.n_expert = n_hot + n_cold - 2*n_spare;
    l.perm.resize(l.n_expert);
    const int n_hot_map = n_hot - n_spare; // the file's experts in VRAM: the first ones, then the host slots
    for (int e = 0; e < l.n_expert; ++e) {
        l.perm[e] = e < n_hot_map ? e : n_hot + (e - n_hot_map);
    }
    for (int i = 0; i < n_spare; ++i) {
        l.free_v.push_back(n_hot_map + i);
        l.free_h.push_back(n_cold - n_spare + i);
    }
    // the row of the layer on the device is the identity (ec_alloc): with spares the perm differs from it and goes
    // to the device now, or before the next graph when a capture is in progress (ggml_cuda_expert_cache_wait)
    if (n_spare > 0) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &cap));
        if (cap == cudaStreamCaptureStatusNone) {
            CUDA_CHECK(cudaMemcpy(d.perm_dev + d.layers.size()*EC_MAX_EXPERTS, l.perm.data(), l.perm.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
        } else {
            l.perm_dirty = true;
            d.perm_dirty = true;
        }
    }
    d.layers.push_back(std::move(l));
    d.n_layers.store((int) d.layers.size(), std::memory_order_release);
    if (d.owner == nullptr) {
        d.owner = &ctx;
    } else if (d.owner != &ctx && !d.frozen) {
        d.frozen = true;
        GGML_LOG_WARN("%s: a second context uses the MoE layers of device %d: their hot set stays as it is\n", __func__, ctx.device);
    }
}

// the row of the layer that the MUL_MAT_ID node computes with, or -1
static int ec_row_of(ec_device & d, const ggml_tensor * node) {
    if (d.n_layers.load(std::memory_order_acquire) == 0) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(d.mtx);
    const char * w = (const char *) node->src[0]->data;
    for (size_t i = 0; i < d.layers.size(); ++i) {
        const ec_layer & l = d.layers[i];
        if (!l.dead && (l.hot[0] == w || l.hot[1] == w || l.hot[2] == w)) {
            return (int) i;
        }
    }
    return -1;
}

bool ggml_cuda_expert_cache_take(ggml_backend_cuda_context & ctx, const ggml_tensor * node, const int32_t ** perm, uint32_t ** counts) {
    const ggml_tensor * ids = node->src[2];
    ec_device & d = g_ec[ctx.device];
    const int row = ec_row_of(d, node);
    if (row < 0 || ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    ctx.ec_last_ids = ids;
    d.n_used = (int) ids->ne[0];
    *perm   = d.perm_dev + (size_t) row*EC_MAX_EXPERTS;
    *counts = d.counts_dev + (size_t) row*EC_MAX_EXPERTS;
    return true;
}

void ggml_cuda_expert_cache_remap(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    const ggml_tensor * ids = node->src[2];
    ec_device & d = g_ec[ctx.device];
    if (ctx.ec_last_ids == ids) {
        return;
    }
    const int row = ec_row_of(d, node);
    if (row < 0) {
        return;
    }
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    ctx.ec_last_ids = ids;
    const int n_used = (int) ids->ne[0];
    const int n      = (int) (ids->ne[0]*ids->ne[1]);
    d.n_used = n_used;
    ec_remap<<<(n + 255)/256, 256, 0, ctx.stream()>>>((int32_t *) ids->data, (int) (ids->nb[1]/sizeof(int32_t)), n_used, n,
        d.perm_dev + (size_t) row*EC_MAX_EXPERTS, d.counts_dev + (size_t) row*EC_MAX_EXPERTS);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_expert_cache_update(ggml_backend_cuda_context & ctx) {
    if (!ctx.ec_pending) {
        return;
    }
    ctx.ec_pending = false;
    ec_device & d = g_ec[ctx.device];
    if (d.n_layers.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(d.mtx);
    if (d.owner != &ctx || d.frozen) {
        return;
    }
    // the cap per update and the gain an exchange needs (below): measured on the 1K server bench (2026-10-10), a
    // cap of 48 with gain 0.25 exchanged ~45 experts every round at a steady 4-6% cold share (the marginal experts
    // going in and out: 130 MB of PCIe traffic and an ec_swap kernel beside every window) and took ~8 rounds to
    // follow a shift of the routing (a 15-25% cold share, windows of 35-45 ms on the host tier); 96 / 2.0 exchange
    // 10-20 per round in the steady state and follow a shift twice as fast: the mean round 29.3 -> 28.3 ms
    static const int   n_swaps_max = getenv("GGML_CUDA_EXPERT_CACHE_SWAPS") != nullptr ? atoi(getenv("GGML_CUDA_EXPERT_CACHE_SWAPS")) : 96;
    static const float decay       = getenv("GGML_CUDA_EXPERT_CACHE_DECAY") != nullptr ? (float) atof(getenv("GGML_CUDA_EXPERT_CACHE_DECAY")) : 0.98f;

    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();
    const size_t n_rows = d.layers.size();
    FN_PROF_T(t_ec0);
    ec_rows_sync(d);
    static const float min_gain = getenv("GGML_CUDA_EXPERT_CACHE_GAIN") != nullptr ? (float) atof(getenv("GGML_CUDA_EXPERT_CACHE_GAIN")) : 2.0f;
    // the update after a prompt ubatch: its counts in the units of a decode window (3 tokens)
    static const bool rescale_on = getenv("GGML_CUDA_EXPERT_CACHE_RESCALE") == nullptr || atoi(getenv("GGML_CUDA_EXPERT_CACHE_RESCALE")) != 0;
    const float rescale = rescale_on && d.last_prompt_tokens > 3 ? 3.0f/(float) d.last_prompt_tokens : 1.0f;
    ec_scan<<<(unsigned int) n_rows, EC_SCAN_TPB, 0, stream>>>(d.rows_dev, d.counts_dev, d.seen_dev, d.freq_dev, d.perm_dev, decay, min_gain,
        rescale, d.res_dev, d.more_dev);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(d.res_host, d.res_dev, n_rows*sizeof(ec_res), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    FN_PROF_ADD("ec.scan", t_ec0);

    struct swap {
        float gain;
        int   row;
        int   e_in;
        int   e_out;
    };
    std::vector<swap>    swaps;
    std::vector<ec_cand> more;
    bool     prompt    = false;
    uint32_t total_max = 0;
    if (d.row_pairs.size() < n_rows) {
        d.row_pairs.resize(n_rows, 0);
        d.row_cold.resize(n_rows, 0);
    }
    for (size_t row = 0; row < n_rows; ++row) {
        const ec_res & r = d.res_host[row];
        d.n_pairs      += r.total;
        d.n_pairs_cold += r.cold;
        if (r.total <= 100) { // decode windows only: a prompt ubatch routes to most experts
            d.row_pairs[row] += r.total;
            d.row_cold[row]  += r.cold;
        }
        total_max = std::max(total_max, r.total);
        prompt = prompt || r.total > 100; // more than a decode window routes
        if (r.n_cand > EC_RES_CAND) {
            more.resize(r.n_cand);
            CUDA_CHECK(cudaMemcpy(more.data() + EC_RES_CAND, d.more_dev + row*EC_MAX_EXPERTS + EC_RES_CAND,
                (r.n_cand - EC_RES_CAND)*sizeof(ec_cand), cudaMemcpyDeviceToHost));
        }
        for (uint32_t i = 0; i < r.n_cand; ++i) {
            const ec_cand & c = i < EC_RES_CAND ? r.cand[i] : more[i];
            swaps.push_back({ c.gain, (int) row, c.e_in, c.e_out });
        }
    }
    d.n_updates++;
    d.last_prompt_tokens = prompt ? (int) (total_max/std::max(1, d.n_used)) : 0;
    {
        // temporary: GGML_CUDA_EXPERT_CACHE_STATS=<n> prints the cold share of every n updates
        static const int every = getenv("GGML_CUDA_EXPERT_CACHE_STATS") != nullptr ? atoi(getenv("GGML_CUDA_EXPERT_CACHE_STATS")) : 0;
        static uint64_t  pairs_last = 0, cold_last = 0, swaps_last = 0;
        if (every > 0 && d.n_updates % every == 0 && ctx.device == 0) {
            fprintf(stderr, "ec-stats dev%d: updates %llu-%llu: %.2f%% of %llu routed pairs cold, %llu exchanges\n", ctx.device,
                    (unsigned long long) (d.n_updates - every + 1), (unsigned long long) d.n_updates,
                    100.0*(double) (d.n_pairs_cold - cold_last)/(double) std::max<uint64_t>(1, d.n_pairs - pairs_last),
                    (unsigned long long) (d.n_pairs - pairs_last), (unsigned long long) (d.n_swaps - swaps_last));
            pairs_last = d.n_pairs; cold_last = d.n_pairs_cold; swaps_last = d.n_swaps;
        }
    }
    // after a prompt every cold expert it routed to comes in; in a decode the best of the last window
    const size_t budget = prompt ? (size_t) EC_MAX_SWAPS : (size_t) std::max(0, n_swaps_max);
    if (swaps.size() > budget) {
        std::partial_sort(swaps.begin(), swaps.begin() + budget, swaps.end(), [](const swap & a, const swap & b) { return a.gain > b.gain; });
        swaps.resize(budget);
    }
    if (swaps.empty()) {
        return;
    }
    FN_PROF_T(t_ec2);
    const int  n_swaps = (int) swaps.size();
    ec_patch * patches = (ec_patch *) (d.jobs_host + 3*n_swaps);
    // the slots the last exchange freed are free once it is done: it had the round (the join is usually immediate)
    if (ctx.ec_event != nullptr && ctx.ec_pend) {
        CUDA_CHECK(cudaEventSynchronize(ctx.ec_event));
        ctx.ec_pend = false;
        for (ec_layer & l : d.layers) {
            l.free_v.insert(l.free_v.end(), l.pend_v.begin(), l.pend_v.end());
            l.free_h.insert(l.free_h.end(), l.pend_h.begin(), l.pend_h.end());
            l.pend_v.clear();
            l.pend_h.clear();
        }
    }
    int n_taken = 0;
    for (int i = 0; i < n_swaps; ++i) {
        const swap & s = swaps[i];
        ec_layer &   l = d.layers[s.row];
        if (l.perm[s.e_out] >= l.n_hot || l.perm[s.e_in] < l.n_hot) {
            continue; // the scan saw the table before the last patch: not an exchange any more
        }
        const int v = l.perm[s.e_out];           // VRAM slot
        const int h = l.perm[s.e_in] - l.n_hot;  // host slot
        int v_to = v, h_to = h;
        if (!l.free_v.empty() && !l.free_h.empty()) {
            // into spare slots: the graphs keep reading v and h until the patch, nothing reads the spares
            v_to = l.free_v.back(); l.free_v.pop_back();
            h_to = l.free_h.back(); l.free_h.pop_back();
            l.pend_v.push_back(v);
            l.pend_h.push_back(h);
        } else if (!l.free_v.empty() || !l.free_h.empty() || n_spare_any(d)) {
            continue; // this layer's spares are taken this round: the exchange waits for the next update
        }
        for (int k = 0; k < 3; ++k) {
            d.jobs_host[3*n_taken + k] = { (uint4 *) (l.hot[k] + (size_t) v*l.nb2[k]), (uint4 *) (l.cold[k] + (size_t) h*l.nb2[k]),
                                           (uint4 *) (l.hot[k] + (size_t) v_to*l.nb2[k]), (uint4 *) (l.cold[k] + (size_t) h_to*l.nb2[k]),
                                           (uint32_t) (l.nb2[k]/16), { 0, 0, 0 } };
        }
        l.perm[s.e_in]  = v_to;
        l.perm[s.e_out] = l.n_hot + h_to;
        patches[n_taken] = { s.row*EC_MAX_EXPERTS + s.e_in, v_to, s.row*EC_MAX_EXPERTS + s.e_out, l.n_hot + h_to };
        n_taken++;
    }
    if (n_taken == 0) {
        return;
    }
    d.n_swaps += n_taken;
    // The slices and the positions, on a stream of their own: what the caller copies from the device next (the
    // results of the window) does not wait for megabytes over PCIe. The next graph does (ggml_cuda_expert_cache_wait).
    cudaStream_t xs = ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - 1);
    const int n_jobs = 3*n_taken;
    // the patches follow the jobs of the swaps taken; the table was laid out for every candidate, pack it
    if (n_taken < n_swaps) {
        memmove(d.jobs_host + n_jobs, patches, n_taken*sizeof(ec_patch));
        patches = (ec_patch *) (d.jobs_host + n_jobs);
    }
    CUDA_CHECK(cudaMemcpyAsync(d.jobs_dev, d.jobs_host, n_jobs*sizeof(ec_job) + n_taken*sizeof(ec_patch), cudaMemcpyHostToDevice, xs));
    ec_swap<<<std::min(n_jobs, EC_SWAP_GRID), EC_SWAP_TPB, 0, xs>>>(d.jobs_dev, n_jobs, (const ec_patch *) (d.jobs_dev + n_jobs), n_taken,
        d.perm_dev);
    CUDA_CHECK(cudaGetLastError());
    if (ctx.ec_event == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.ec_event, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(ctx.ec_event, xs));
    // with spares the graphs need not wait for the exchange (the slots they read are untouched, the patch is
    // consistent at any point); without, the next graph waits for it
    ctx.ec_wait = !n_spare_any(d);
    ctx.ec_pend = true;
    FN_PROF_ADD("ec.enqueue", t_ec2);
}

void ggml_cuda_expert_cache_join(ggml_backend_cuda_context & ctx) {
    if (ctx.ec_future.valid()) {
        ctx.ec_future.get();
    }
}

void ggml_cuda_expert_cache_update_async(ggml_backend_cuda_context & ctx) {
    // off by default: on the 14-core box the thread displaces a host-tier worker and the window pays more than
    // the 0.15 ms per device it saves the host (1K round 32.3 -> 32.9 ms)
    static const bool threaded = getenv("GGML_CUDA_EXPERT_CACHE_THREAD") != nullptr && atoi(getenv("GGML_CUDA_EXPERT_CACHE_THREAD")) != 0;
    ggml_cuda_expert_cache_join(ctx);
    if (!ctx.ec_pending) {
        return;
    }
    // only the context that updates the layers has work for a thread
    ec_device & d = g_ec[ctx.device];
    if (!threaded || d.n_layers.load(std::memory_order_acquire) == 0 || d.owner != &ctx || d.frozen) {
        ggml_cuda_expert_cache_update(ctx);
        return;
    }
    ctx.ec_future = std::async(std::launch::async, [&ctx] {
        ggml_cuda_set_device(ctx.device);
        ggml_cuda_expert_cache_update(ctx);
    });
}

void ggml_cuda_expert_cache_wait(ggml_backend_cuda_context & ctx) {
    ggml_cuda_expert_cache_join(ctx);
    ec_device & d = g_ec[ctx.device];
    if (d.perm_dirty.load(std::memory_order_acquire)) {
        // a layer registered during a capture: its perm goes to the device before the graph that reads it
        std::lock_guard<std::mutex> lock(d.mtx);
        ggml_cuda_set_device(ctx.device);
        for (size_t row = 0; row < d.layers.size(); ++row) {
            ec_layer & l = d.layers[row];
            if (l.perm_dirty) {
                CUDA_CHECK(cudaMemcpy(d.perm_dev + row*EC_MAX_EXPERTS, l.perm.data(), l.perm.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
                l.perm_dirty = false;
            }
        }
        d.perm_dirty.store(false, std::memory_order_release);
    }
    if (!ctx.ec_wait) {
        return;
    }
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), ctx.ec_event));
    ctx.ec_wait = false;
}

void ggml_cuda_expert_cache_context_free(ggml_backend_cuda_context & ctx) {
    ggml_cuda_expert_cache_join(ctx);
    ec_device & d = g_ec[ctx.device];
    std::lock_guard<std::mutex> lock(d.mtx);
    if (d.owner != &ctx) {
        return;
    }
    if (d.n_pairs > 0) {
        // temporary: on stderr until the numbers are settled
        fprintf(stderr, "%s: device %d: %llu updates, %llu expert exchanges, %.2f%% of %llu routed pairs were cold\n", __func__, ctx.device,
                (unsigned long long) d.n_updates, (unsigned long long) d.n_swaps, 100.0*(double) d.n_pairs_cold/(double) d.n_pairs,
                (unsigned long long) d.n_pairs);
        if (getenv("GGML_CUDA_EXPERT_CACHE_STATS") != nullptr && ctx.device == 0) {
            // the cold share by layer, for a hot budget per layer
            fprintf(stderr, "ec-layers dev0, decode windows (hot/all: cold%%):");
            for (size_t row = 0; row < d.row_pairs.size(); ++row) {
                const ec_layer & l = d.layers[row];
                fprintf(stderr, " %zu:%d/%d:%.1f", row, l.n_hot, l.n_hot + l.n_cold,
                        100.0*(double) d.row_cold[row]/(double) std::max<uint64_t>(1, d.row_pairs[row]));
            }
            fprintf(stderr, "\n");
        }
    }
    // the exchanges it queued must be done before another context computes with the layers
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    if (ctx.ec_event != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - 1)));
        CUDA_CHECK(cudaEventDestroy(ctx.ec_event));
        ctx.ec_event = nullptr;
        ctx.ec_wait  = false;
    }
    d.owner = nullptr;
}

void ggml_cuda_expert_cache_release(const void * base, const size_t size) {
    for (ec_device & d : g_ec) {
        if (d.n_layers.load(std::memory_order_acquire) == 0) {
            continue;
        }
        std::lock_guard<std::mutex> lock(d.mtx);
        for (ec_layer & l : d.layers) {
            if (l.hot[2] >= (const char *) base && l.hot[2] < (const char *) base + size) {
                l.dead       = true;
                d.rows_dirty = true;
            }
        }
    }
}
