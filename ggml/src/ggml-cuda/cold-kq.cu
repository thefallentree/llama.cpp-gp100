// Staging of the cold experts of Q2_K / Q3_K hot/cold layers in VRAM for prompt batches, see cold-kq.cuh.

#include "cold-kq.cuh"
#include "mmid-f16-sm60.cuh"
#include "mmvq.cuh"

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

// the stream work of cold_kq_ring on the CUDA streams
struct cold_kq_dev {
    ggml_cuda_cold_kq & st;
    cudaStream_t        stream;

    void copy(const int slot, const int32_t expert, const int n) const {
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

bool ggml_cuda_cold_kq_begin(ggml_backend_cuda_context & ctx, const char * host, const std::vector<int32_t> & used, const size_t nb02, const size_t reserve) {
    ggml_cuda_cold_kq & st = ctx.cold_kq;
    if (st.failed) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
    GGML_ASSERT(cap == cudaStreamCaptureStatusNone && "cold-kq: the Q2_K/Q3_K cold prompt path reads the ids on the host, no CUDA graph capture");

    const size_t n = used.size();
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

    // slots: experts of this op the buffer can hold, at least two (one is read while the other is copied)
    const size_t budget       = cold_kq_budget();
    const size_t budget_slots = std::max<size_t>(2, budget > reserve ? (budget - reserve)/nb02 : 0);
    size_t slots = std::min(n, budget_slots);
    if (slots*nb02 > st.cap && st.cap < st.limit) {
        // grow: nothing of the buffer is in use, the streams are idle after the ids were read back; a quarter more
        // than this op needs, so that the next ones rarely grow it again (never beyond the budget)
        CUDA_CHECK(cudaStreamSynchronize(st.copy));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (st.buf != nullptr) {
            CUDA_CHECK(cudaFree(st.buf));
            st.buf = nullptr;
            st.cap = 0;
        }
        const size_t first = std::min(slots + slots/4, budget_slots);
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

    while ((int) st.ready.size() < st.ring.depth) {
        cudaEvent_t r = nullptr, f = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&r, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&f, cudaEventDisableTiming));
        st.ready.push_back(r);
        st.freed.push_back(f);
    }

    if (st.n_ops < 3) {
        GGML_LOG_WARN("%s: dev%d op %" PRId64 ": %zu cold experts of %.2f MiB (%.1f MiB) in %d chunks of %d, ring of %d, buffer %.1f MiB\n", __func__,
                ctx.device, st.n_ops, n, nb02/1048576.0, n*nb02/1048576.0, st.ring.nchunks, st.ring.chunk, st.ring.depth, st.cap/1048576.0);
    }

    // the copy stream follows the work enqueued so far: the buffer is free and the exchange of the expert cache is done
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
    return st.buf + (size_t) slot*st.nb02;
}

void ggml_cuda_cold_kq_end(ggml_backend_cuda_context & ctx) {
    static const int64_t every = getenv("GGML_CUDA_COLD_PREFETCH_STATS") != nullptr ? atoi(getenv("GGML_CUDA_COLD_PREFETCH_STATS")) : 0;
    ggml_cuda_cold_kq & st = ctx.cold_kq;
    GGML_ASSERT(st.ring.next == (int) st.ring.list.size() && "cold-kq: fewer slices than staged experts");
    st.n_ops++;
    st.n_experts += (int64_t) st.ring.list.size();
    st.n_bytes   += (int64_t) (st.ring.list.size()*st.nb02);
    if (every > 0 && st.n_ops % every == 0) {
        GGML_LOG_WARN("%s: dev%d %" PRId64 " ops, %" PRId64 " cold experts (%.1f experts per op), %.2f GiB copied\n", __func__,
                ctx.device, st.n_ops, st.n_experts, (double) st.n_experts/st.n_ops, st.n_bytes/1073741824.0);
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
}
