// Staging of the cold experts of Q2_K / Q3_K hot/cold layers in VRAM for prompt batches, see cold-kq.cuh.

#include "cold-kq.cuh"
#include "mmid-f16-sm60.cuh"
#include "mmvq.cuh"

#include <algorithm>
#include <cinttypes>

bool ggml_cuda_cold_kq_enabled() {
    static const bool enabled = getenv("GGML_CUDA_COLD_PREFETCH_KQ") == nullptr || atoi(getenv("GGML_CUDA_COLD_PREFETCH_KQ")) != 0;
    return enabled;
}

bool ggml_cuda_cold_kq_node(const ggml_tensor * node) {
    return node->op == GGML_OP_MUL_MAT_ID && ggml_cuda_mmid_cold(node) && node->ne[2] > MMVQ_MAX_BATCH_SIZE &&
           (node->src[0]->type == GGML_TYPE_Q2_K || node->src[0]->type == GGML_TYPE_Q3_K);
}

static size_t cold_kq_budget() {
    static const size_t bytes = [] {
        const char * e = getenv("GGML_CUDA_COLD_PREFETCH_MB");
        return (size_t) std::max(1, e != nullptr ? atoi(e) : 1024)*1048576;
    }();
    return bytes;
}

// VRAM that stays free next to the staging buffer (and next to the caller's reserve)
static size_t cold_kq_keep() {
    static const size_t bytes = [] {
        const char * e = getenv("GGML_CUDA_COLD_PREFETCH_KEEP_MB");
        return (size_t) std::max(0, e != nullptr ? atoi(e) : 512)*1048576;
    }();
    return bytes;
}

static bool cold_kq_sync_mode() {
    static const bool on = getenv("GGML_CUDA_COLD_PREFETCH_SYNC") != nullptr && atoi(getenv("GGML_CUDA_COLD_PREFETCH_SYNC")) != 0;
    return on;
}

// per device: the least free VRAM seen at the start of an op, and the warnings that are given once
struct cold_kq_dbg {
    size_t min_free   = SIZE_MAX;
    bool   warned_cut = false;
    bool   warned_off = false;
};
static cold_kq_dbg g_dbg[GGML_CUDA_MAX_DEVICES];

// GGML_CUDA_COLD_PREFETCH_SYNC=1: wait for the work enqueued so far, so that a fault is reported here and names the op
static void cold_kq_check(ggml_backend_cuda_context & ctx, const char * where, const size_t n_experts) {
    if (!cold_kq_sync_mode()) {
        return;
    }
    ggml_cuda_cold_kq & st = ctx.cold_kq;
    ggml_cuda_set_device(ctx.device);
    cudaError_t err = cudaStreamSynchronize(ctx.stream());
    if (err == cudaSuccess && st.copy != nullptr) {
        err = cudaStreamSynchronize(st.copy);
    }
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: dev%d: cold op %" PRId64 " (%zu cold experts), at its %s: %s\n", __func__, ctx.device, st.n_ops, n_experts, where, cudaGetErrorString(err));
        CUDA_CHECK(err);
    }
}

// the staging buffer goes back to VRAM (the streams are idle first)
static void cold_kq_release(ggml_backend_cuda_context & ctx) {
    ggml_cuda_cold_kq & st = ctx.cold_kq;
    if (st.buf == nullptr) {
        return;
    }
    if (st.copy != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(st.copy));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    CUDA_CHECK(cudaFree(st.buf));
    st.buf = nullptr;
    st.cap = 0;
}

// the stream work of cold_kq_ring on the CUDA streams
struct cold_kq_dev {
    ggml_cuda_cold_kq & st;
    cudaStream_t        stream;

    void copy(const int slot, const int32_t expert, const int n) const {
        GGML_ASSERT(slot >= 0 && n > 0 && (size_t) (slot + n)*st.nb02 <= st.cap && "cold-kq: a copy beyond the staging buffer");
        CUDA_CHECK(cudaMemcpyAsync(st.buf + (size_t) slot*st.nb02, st.host + (size_t) expert*st.nb02, (size_t) n*st.nb02, cudaMemcpyDefault, st.copy));
    }
    void copy_wait_freed(const int b) const {
        CUDA_CHECK(cudaStreamWaitEvent(st.copy, st.freed[b], 0));
    }
    void copy_record_ready(const int b) const {
        CUDA_CHECK(cudaEventRecord(st.ready[b], st.copy));
    }
    void wait_ready(const int b) const {
        CUDA_CHECK(cudaStreamWaitEvent(stream, st.ready[b], 0));
    }
    void record_freed(const int b) const {
        CUDA_CHECK(cudaEventRecord(st.freed[b], stream));
    }
};

bool ggml_cuda_cold_kq_begin(ggml_backend_cuda_context & ctx, const char * host, const std::vector<int32_t> & used, const size_t nb02, const size_t reserve,
                             const int64_t n_cold) {
    ggml_cuda_cold_kq & st = ctx.cold_kq;
    if (st.failed) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
    GGML_ASSERT(cap == cudaStreamCaptureStatusNone && "cold-kq: the Q2_K/Q3_K cold prompt path reads the ids on the host, no CUDA graph capture");

    const size_t n = used.size();
    GGML_ASSERT(std::is_sorted(used.begin(), used.end()) && "cold-kq: the cold experts of an op must be ascending");
    GGML_ASSERT((n == 0 || (used.front() >= 0 && (n_cold < 0 || used.back() < n_cold))) && "cold-kq: a cold expert beyond the cold tensor");
    cold_kq_check(ctx, "start", n);

    st.host      = host;
    st.nb02      = nb02;
    st.ring.list = used;
    if (n == 0) {
        st.ring.plan(1);
        return true;
    }

    ggml_cuda_set_device(ctx.device);
    if (st.copy == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&st.copy, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&st.fork, cudaEventDisableTiming));
    }

    // what the card has: the buffer counts as free (it is reused or given back), the margin stays; no number: no limit
    size_t free_b = 0, total_b = 0;
    const bool have_mem = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
    if (!have_mem) {
        (void) cudaGetLastError();
    }
    cold_kq_dbg & dbg = g_dbg[ctx.device];
    if (have_mem) {
        dbg.min_free = std::min(dbg.min_free, free_b);
    }
    // slots: experts of this op the buffer can hold, at least two (one is read while the other is copied)
    const size_t budget = cold_kq_budget();
    const cold_kq_vram_out fit = cold_kq_vram({ n, nb02, budget, reserve, cold_kq_keep(), have_mem, free_b, st.cap });
    if (fit.release) {
        cold_kq_release(ctx);   // refused, or the card is short and the buffer holds more than the op may take: sized again below
    }
    if (fit.refuse) {
        // not even two slots fit next to the margin: the caller reads the cold experts in place
        if (!dbg.warned_off) {
            dbg.warned_off = true;
            GGML_LOG_WARN("%s: dev%d: %.0f MiB of VRAM free, %.0f MiB to keep (GGML_CUDA_COLD_PREFETCH_KEEP_MB): no room for staging, the cold experts are read in place\n",
                    __func__, ctx.device, free_b/1048576.0, (cold_kq_keep() + reserve)/1048576.0);
        }
        return false;
    }
    if (fit.cut && !dbg.warned_cut) {
        dbg.warned_cut = true;
        GGML_LOG_WARN("%s: dev%d: %.0f MiB of VRAM free, %.0f MiB to keep (GGML_CUDA_COLD_PREFETCH_KEEP_MB): the staging holds %zu experts (%.0f MiB) of this op's %zu\n",
                __func__, ctx.device, free_b/1048576.0, (cold_kq_keep() + reserve)/1048576.0, fit.fit_slots, fit.fit_slots*nb02/1048576.0, n);
    }
    const size_t fit_slots = fit.fit_slots;
    size_t slots = std::min(n, fit_slots);

    if (slots*nb02 > st.cap && st.cap < st.limit) {
        // grow: nothing of the buffer is in use, the streams are idle after the ids were read back; a quarter more
        // than this op needs, so that the next ones rarely grow it again (never beyond what fits)
        CUDA_CHECK(cudaStreamSynchronize(st.copy));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (st.buf != nullptr) {
            CUDA_CHECK(cudaFree(st.buf));
            st.buf = nullptr;
            st.cap = 0;
        }
        const size_t first = std::min(slots + slots/4, fit_slots);
        size_t       want  = first;
        while (want > 0 && cudaMalloc((void **) &st.buf, want*nb02) != cudaSuccess) {
            (void) cudaGetLastError();
            st.buf = nullptr;
            want   = want > slots ? slots : want/2;
        }
        if (want == 0) {
            st.failed = true;
            GGML_LOG_WARN("%s: no VRAM for staging %zu cold experts of %.2f MiB (budget GGML_CUDA_COLD_PREFETCH_MB %zu), the cold experts are read in place\n",
                    __func__, slots, nb02/1048576.0, budget/1048576);
            return false;
        }
        st.cap = want*nb02;
        if (want != first) {
            st.limit = st.cap;   // no retry at every op
            GGML_LOG_WARN("%s: dev%d: only %.1f MiB of the %.1f MiB staging buffer fit in VRAM\n", __func__, ctx.device, st.cap/1048576.0, first*nb02/1048576.0);
        }
    }
    slots = std::min(slots, st.cap/nb02);
    if (slots == 0) {
        return false;   // a limited buffer (see above) smaller than one expert of this op
    }
    st.ring.plan(slots);
    GGML_ASSERT(st.ring.n_slots()*nb02 <= st.cap && "cold-kq: the ring is larger than the staging buffer");

    while ((int) st.ready.size() < st.ring.depth) {
        cudaEvent_t r = nullptr, f = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&r, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&f, cudaEventDisableTiming));
        st.ready.push_back(r);
        st.freed.push_back(f);
    }

    if (st.n_ops < 3) {
        GGML_LOG_WARN("%s: dev%d op %" PRId64 ": %zu cold experts of %.2f MiB (%.1f MiB) in %d chunks of %d, ring of %d, buffer %.1f MiB, VRAM free %.0f MiB\n", __func__,
                ctx.device, st.n_ops, n, nb02/1048576.0, n*nb02/1048576.0, st.ring.nchunks, st.ring.chunk, st.ring.depth, st.cap/1048576.0, free_b/1048576.0);
    }

    // an exchange of the expert cache that still runs on its own stream writes the hot and cold slots that this op reads
    // (the graph does not wait for it when the layers have spare slots): the reads of the op come after it
    if (ctx.ec_event != nullptr && ctx.ec_pend) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, ctx.ec_event, 0));
        CUDA_CHECK(cudaStreamWaitEvent(st.copy, ctx.ec_event, 0));
    }
    // the copy stream follows the work enqueued so far: the buffer is free
    CUDA_CHECK(cudaEventRecord(st.fork, stream));
    CUDA_CHECK(cudaStreamWaitEvent(st.copy, st.fork, 0));
    cold_kq_dev dev { st, stream };
    st.ring.start(dev);
    return true;
}

char * ggml_cuda_cold_kq_next(ggml_backend_cuda_context & ctx) {
    ggml_cuda_cold_kq & st = ctx.cold_kq;
    GGML_ASSERT(st.ring.next < (int) st.ring.list.size() && "cold-kq: more slices than staged experts");
    cold_kq_dev dev { st, ctx.stream() };
    const int slot = st.ring.take(dev);
    GGML_ASSERT(slot >= 0 && (size_t) (slot + 1)*st.nb02 <= st.cap && "cold-kq: a slot beyond the staging buffer");
    return st.buf + (size_t) slot*st.nb02;
}

void ggml_cuda_cold_kq_end(ggml_backend_cuda_context & ctx) {
    static const int64_t every = getenv("GGML_CUDA_COLD_PREFETCH_STATS") != nullptr ? atoi(getenv("GGML_CUDA_COLD_PREFETCH_STATS")) : 0;
    ggml_cuda_cold_kq & st = ctx.cold_kq;
    GGML_ASSERT(st.ring.next == (int) st.ring.list.size() && "cold-kq: fewer slices than staged experts");
    cold_kq_check(ctx, "end", st.ring.list.size());
    st.n_ops++;
    st.n_experts += (int64_t) st.ring.list.size();
    st.n_bytes   += (int64_t) (st.ring.list.size()*st.nb02);
    if (every > 0 && st.n_ops % every == 0) {
        const size_t min_free = g_dbg[ctx.device].min_free;
        GGML_LOG_WARN("%s: dev%d %" PRId64 " ops, %" PRId64 " cold experts (%.1f experts per op), %.2f GiB copied, least free VRAM %.0f MiB\n", __func__,
                ctx.device, st.n_ops, st.n_experts, (double) st.n_experts/st.n_ops, st.n_bytes/1073741824.0,
                min_free == SIZE_MAX ? -1.0 : min_free/1048576.0);
    }
    st.ring.list.clear();
}

void ggml_cuda_cold_kq_free(ggml_backend_cuda_context & ctx) {
    ggml_cuda_cold_kq & st = ctx.cold_kq;
    if (st.copy != nullptr) {
        cudaStreamSynchronize(st.copy);
    }
    if (st.buf != nullptr) {
        cudaFree(st.buf);
    }
    for (cudaEvent_t e : st.ready) {
        cudaEventDestroy(e);
    }
    for (cudaEvent_t e : st.freed) {
        cudaEventDestroy(e);
    }
    if (st.fork != nullptr) {
        cudaEventDestroy(st.fork);
    }
    if (st.copy != nullptr) {
        cudaStreamDestroy(st.copy);
    }
    st = {};
    g_dbg[ctx.device] = {};
}
