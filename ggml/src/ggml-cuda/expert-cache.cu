#include "expert-cache.cuh"
#include "mmid-f16-sm60.cuh"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

#define EC_MAX_LAYERS  128
#define EC_MAX_EXPERTS 1024
#define EC_MAX_SWAPS   8192  // per update
#define EC_SWAP_GRID   16    // blocks of the exchange kernel: PCIe is saturated from about 16 SMs
#define EC_SWAP_TPB    256

// one slice exchange: n16 16-byte words
struct ec_job {
    uint4 *  dev;
    uint4 *  host;
    uint32_t n16;
    uint32_t pad;
};

struct ec_layer {
    char * hot[3]  = {};          // up, gate, down: the slices of the experts in VRAM
    char * cold[3] = {};          // their slices in pinned host memory
    size_t nb2[3]  = {};          // bytes per expert
    int    n_hot   = 0;
    int    n_cold  = 0;
    bool   dead    = false;       // its buffer was freed
    std::vector<int32_t>  perm;   // expert -> position: a VRAM slot below n_hot, else n_hot + host slot
    std::vector<int32_t>  owner;  // position -> expert
    std::vector<float>    freq;   // decayed use count
    std::vector<uint32_t> seen;   // the device's counts at the last update
};

struct ec_device {
    std::mutex            mtx;
    std::vector<ec_layer> layers;
    std::atomic<int>      n_layers { 0 };
    std::atomic<bool>     ready { false }; // the tables exist
    int32_t *    perm_dev    = nullptr;  // [EC_MAX_LAYERS][EC_MAX_EXPERTS]
    uint32_t *   counts_dev  = nullptr;  // likewise: how often each expert was routed to
    ec_job *     jobs_dev    = nullptr;  // [3*EC_MAX_SWAPS]
    int32_t *    perm_host   = nullptr;  // pinned copies
    uint32_t *   counts_host = nullptr;
    ec_job *     jobs_host   = nullptr;
    const void * owner       = nullptr;  // the context that updates the layers
    bool         frozen      = false;    // another context uses them too
    uint64_t     n_updates   = 0;
    uint64_t     n_swaps     = 0;
    uint64_t     n_pairs     = 0;
    uint64_t     n_pairs_cold = 0;
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

// exchanges the slices of the jobs between VRAM and the host; block b takes the jobs b, b + grid, ...
static __global__ void ec_swap(const ec_job * __restrict__ jobs, const int n_jobs) {
    for (int j = blockIdx.x; j < n_jobs; j += gridDim.x) {
        const ec_job job = jobs[j];
        for (uint32_t i = threadIdx.x; i < job.n16; i += blockDim.x) {
            const uint4 h = job.host[i];
            const uint4 d = job.dev[i];
            job.dev[i]  = h;
            job.host[i] = d;
        }
    }
    // the host threads read the slices in the next graph
    __threadfence_system();
}

static bool ec_alloc(ec_device & d, const int device) {
    if (d.perm_dev != nullptr) {
        return true;
    }
    ggml_cuda_set_device(device);
    const size_t n_tab = (size_t) EC_MAX_LAYERS*EC_MAX_EXPERTS;
    void * hp = nullptr;
    const size_t host_size = n_tab*(sizeof(int32_t) + sizeof(uint32_t)) + (size_t) 3*EC_MAX_SWAPS*sizeof(ec_job);
    if (cudaMallocHost(&hp, host_size) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    d.perm_host   = (int32_t *) hp;
    d.counts_host = (uint32_t *) (d.perm_host + n_tab);
    d.jobs_host   = (ec_job *) (d.counts_host + n_tab);
    CUDA_CHECK(cudaMalloc((void **) &d.perm_dev,   n_tab*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc((void **) &d.counts_dev, n_tab*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **) &d.jobs_dev,   (size_t) 3*EC_MAX_SWAPS*sizeof(ec_job)));
    CUDA_CHECK(cudaMemset(d.counts_dev, 0, n_tab*sizeof(uint32_t)));
    // every row starts as the identity: a registration then writes nothing to the device and may happen while a
    // CUDA graph is being captured
    for (size_t i = 0; i < n_tab; ++i) {
        d.perm_host[i] = (int32_t) (i % EC_MAX_EXPERTS);
    }
    CUDA_CHECK(cudaMemcpy(d.perm_dev, d.perm_host, n_tab*sizeof(int32_t), cudaMemcpyHostToDevice));
    return true;
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
    bool ok = n_hot + n_cold <= EC_MAX_EXPERTS && n_cold > 0 && d.layers.size() < EC_MAX_LAYERS;
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
    l.n_hot  = n_hot;
    l.n_cold = n_cold;
    const int n_all = n_hot + n_cold;
    l.perm.resize(n_all);
    l.owner.resize(n_all);
    l.freq.resize(n_all);
    l.seen.assign(n_all, 0);
    for (int e = 0; e < n_all; ++e) {
        l.perm[e]  = e;
        l.owner[e] = e;
        // the file's order is by use in a calibration text: a weak prior
        l.freq[e]  = 0.5f*(float) (n_all - e)/(float) n_all;
    }
    // the row of the layer on the device is the identity already (ec_alloc)
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
    static const int   n_swaps_max = getenv("GGML_CUDA_EXPERT_CACHE_SWAPS") != nullptr ? atoi(getenv("GGML_CUDA_EXPERT_CACHE_SWAPS")) : 16;
    static const float decay       = getenv("GGML_CUDA_EXPERT_CACHE_DECAY") != nullptr ? (float) atof(getenv("GGML_CUDA_EXPERT_CACHE_DECAY")) : 0.98f;

    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();
    const size_t n_rows = d.layers.size();
    CUDA_CHECK(cudaMemcpyAsync(d.counts_host, d.counts_dev, n_rows*EC_MAX_EXPERTS*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    struct swap {
        float gain;
        int   row;
        int   e_in;
        int   e_out;
    };
    std::vector<swap> swaps;
    std::vector<int>  in, out;
    bool prompt = false;
    for (size_t row = 0; row < n_rows; ++row) {
        ec_layer & l = d.layers[row];
        if (l.dead) {
            continue;
        }
        const uint32_t * counts = d.counts_host + row*EC_MAX_EXPERTS;
        const int n_all = l.n_hot + l.n_cold;
        uint32_t total = 0;
        in.clear();
        out.clear();
        for (int e = 0; e < n_all; ++e) {
            total += counts[e] - l.seen[e];
        }
        if (total == 0) {
            continue;
        }
        for (int e = 0; e < n_all; ++e) {
            const uint32_t delta = counts[e] - l.seen[e];
            l.seen[e] = counts[e];
            l.freq[e] = l.freq[e]*decay + (float) delta;
            if (l.perm[e] >= l.n_hot) {
                d.n_pairs_cold += delta;
                if (delta > 0) {
                    in.push_back(e);
                }
            } else if (delta == 0) {
                out.push_back(e);
            }
        }
        d.n_pairs += total;
        prompt = prompt || total > 100; // more than a decode window routes
        std::sort(in.begin(),  in.end(),  [&](const int a, const int b) { return l.freq[a] > l.freq[b]; });
        std::sort(out.begin(), out.end(), [&](const int a, const int b) { return l.freq[a] < l.freq[b]; });
        for (size_t i = 0; i < std::min(in.size(), out.size()); ++i) {
            const float gain = l.freq[in[i]] - l.freq[out[i]];
            if (gain <= 0.25f) {
                break;
            }
            swaps.push_back({ gain, (int) row, in[i], out[i] });
        }
    }
    d.n_updates++;
    // after a prompt every cold expert it routed to comes in; in a decode the best of the last window
    const size_t budget = prompt ? (size_t) EC_MAX_SWAPS : (size_t) std::max(0, n_swaps_max);
    if (swaps.size() > budget) {
        std::partial_sort(swaps.begin(), swaps.begin() + budget, swaps.end(), [](const swap & a, const swap & b) { return a.gain > b.gain; });
        swaps.resize(budget);
    }
    if (swaps.empty()) {
        return;
    }
    int n_jobs = 0;
    for (const swap & s : swaps) {
        ec_layer & l = d.layers[s.row];
        const int v = l.perm[s.e_out];           // VRAM slot
        const int h = l.perm[s.e_in] - l.n_hot;  // host slot
        for (int k = 0; k < 3; ++k) {
            d.jobs_host[n_jobs++] = { (uint4 *) (l.hot[k] + (size_t) v*l.nb2[k]), (uint4 *) (l.cold[k] + (size_t) h*l.nb2[k]),
                                      (uint32_t) (l.nb2[k]/16), 0 };
        }
        l.perm[s.e_in]        = v;
        l.perm[s.e_out]       = l.n_hot + h;
        l.owner[v]            = s.e_in;
        l.owner[l.n_hot + h]  = s.e_out;
        std::copy(l.perm.begin(), l.perm.end(), d.perm_host + (size_t) s.row*EC_MAX_EXPERTS);
    }
    d.n_swaps += swaps.size();
    // on the stream, so before any graph that is launched after this: the slices, then the positions
    CUDA_CHECK(cudaMemcpyAsync(d.jobs_dev, d.jobs_host, n_jobs*sizeof(ec_job), cudaMemcpyHostToDevice, stream));
    ec_swap<<<std::min(n_jobs, EC_SWAP_GRID), EC_SWAP_TPB, 0, stream>>>(d.jobs_dev, n_jobs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(d.perm_dev, d.perm_host, n_rows*EC_MAX_EXPERTS*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
}

void ggml_cuda_expert_cache_context_free(ggml_backend_cuda_context & ctx) {
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
    }
    // the exchanges it queued must be done before another context computes with the layers
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
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
                l.dead = true;
            }
        }
    }
}
