#include "moe-host.cuh"
#include "moe-host-cpu.h"
#include "mmid-f16-sm60.cuh"

#include <atomic>
#include <cstring>

#define MH_DSTATE_SEQ  0 // last request number
#define MH_DSTATE_NEED 1 // cold pairs of the last request
#define MH_DSTATE_PAIR 2 // their t*n_used + k, MH_MAX_PAIRS entries

#define MH_PUBLISH_THREADS 256
#define MH_COLLECT_BLOCKS  8

static_assert(MH_MAX_PAIRS <= MH_PUBLISH_THREADS, "one publish thread per (token, slot) pair");

static bool ggml_cuda_moe_host_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_MOE_HOST");
        return (e == nullptr || atoi(e) != 0) && mh_cpu_supported();
    }();
    return enabled;
}

// One thread per (token, slot) pair: the cold ones are compacted with a ballot. Then the input rows are copied
// and the doorbell is rung. Nothing is published when the step has no cold pair.
static __global__ void mh_publish(
        const int32_t * __restrict__ ids, const int si1, const int n_used, const int n_tokens, const int n_hot,
        const float * __restrict__ x, const int64_t sx_tok, const int n_embd,
        mh_mailbox * mb, uint32_t * dstate, const int slot) {
    __shared__ int wcount[MH_PUBLISH_THREADS/WARP_SIZE];

    const int i     = threadIdx.x;
    const int lane  = i % WARP_SIZE;
    const int w     = i / WARP_SIZE;
    const int n_all = n_tokens*n_used;

    int  e    = 0;
    bool cold = false;
    if (i < n_all) {
        e    = ids[(i / n_used)*si1 + i % n_used];
        cold = e >= n_hot;
    }
    const unsigned int m = __ballot_sync(0xffffffff, cold);
    if (lane == 0) {
        wcount[w] = __popc(m);
    }
    __syncthreads();
    int base = 0;
    int np   = 0;
    for (int j = 0; j < MH_PUBLISH_THREADS/WARP_SIZE; ++j) {
        base += j < w ? wcount[j] : 0;
        np   += wcount[j];
    }
    if (cold) {
        const int pos = base + __popc(m & ((1u << lane) - 1));
        mb->pair_idx[pos] = i;
        mb->pair_exp[pos] = e - n_hot;
        dstate[MH_DSTATE_PAIR + pos] = i;
    }
    if (i == 0) {
        dstate[MH_DSTATE_NEED] = np;
    }
    if (np == 0) {
        return;
    }

    const int n4 = n_embd/4;
    for (int k = i; k < n_tokens*n4; k += blockDim.x) {
        const int t = k / n4;
        const int c = k % n4;
        ((float4 *) mb->x)[(int64_t) t*n4 + c] = ((const float4 *) (x + t*sx_tok))[c];
    }
    __threadfence_system();
    __syncthreads();
    if (i == 0) {
        mb->slot     = slot;
        mb->n_tokens = n_tokens;
        mb->n_used   = n_used;
        mb->n_pairs  = np;
        const uint32_t s = dstate[MH_DSTATE_SEQ] + 1;
        dstate[MH_DSTATE_SEQ] = s;
        __threadfence_system();
        mb->req = s;
        __threadfence_system();
    }
}

static __global__ void mh_collect(
        const mh_mailbox * mb, const uint32_t * __restrict__ dstate, float * __restrict__ dst,
        const int64_t sd_slot, const int64_t sd_tok, const int n_used, const int n_embd) {
    const int np = dstate[MH_DSTATE_NEED];
    if (np == 0 || (int) blockIdx.x >= np) {
        return;
    }
    if (threadIdx.x == 0) {
        const uint32_t  s  = dstate[MH_DSTATE_SEQ];
        const long long t0 = clock64();
        while (*((volatile const uint32_t *) &mb->done) != s) {
            if (clock64() - t0 > 20000000000LL) {
                printf("moe-host: no answer from the host threads for request %u\n", s);
                __trap();
            }
        }
    }
    __syncthreads();
    const int n4 = n_embd/4;
    for (int p = blockIdx.x; p < np; p += gridDim.x) {
        const int idx = dstate[MH_DSTATE_PAIR + p];
        const int t   = idx / n_used;
        const int k   = idx % n_used;
        float4 *       out = (float4 *) (dst + t*sd_tok + k*sd_slot);
        const float4 * src = (const float4 *) (mb->y + (int64_t) p*n_embd);
        for (int c = threadIdx.x; c < n4; c += blockDim.x) {
            out[c] = __ldcv(src + c);
        }
    }
}

static std::atomic<int> g_mh_mailboxes { 0 };

static bool ggml_cuda_moe_host_init(ggml_backend_cuda_context & ctx) {
    if (ctx.moe_host_mb != nullptr) {
        return true;
    }
    // the request counter must not be reset by a captured memset, so set up before any capture
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &status));
    if (status != cudaStreamCaptureStatusNone) {
        return false;
    }
    ggml_cuda_set_device(ctx.device);
    mh_mailbox * mb = nullptr;
    if (cudaHostAlloc((void **) &mb, sizeof(mh_mailbox), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: no pinned mailbox, cold experts stay on the GPU\n", __func__);
        return false;
    }
    memset((void *) mb, 0, sizeof(mh_mailbox));
    CUDA_CHECK(cudaMalloc((void **) &ctx.moe_host_dstate, (MH_DSTATE_PAIR + MH_MAX_PAIRS)*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(ctx.moe_host_dstate, 0, (MH_DSTATE_PAIR + MH_MAX_PAIRS)*sizeof(uint32_t)));
    ctx.moe_host_mb = mb;
    mh_pool_attach(g_mh_mailboxes.fetch_add(1), mb);
    return true;
}

void ggml_cuda_moe_host_free(ggml_backend_cuda_context & ctx) {
    // the host threads may still look at the mailbox: it is left to the process exit
    if (ctx.moe_host_dstate != nullptr) {
        CUDA_CHECK(cudaFree(ctx.moe_host_dstate));
        ctx.moe_host_dstate = nullptr;
    }
}

bool ggml_cuda_moe_host_active(const ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    const ggml_cuda_moe_host_triple & t = ctx.moe_host;
    return t.ids != nullptr && node->src[2] == t.ids && (node == t.up || node == t.gate || node == t.down);
}

static bool mh_is_cold_q2(const ggml_tensor * n) {
    return n->op == GGML_OP_MUL_MAT_ID && ggml_cuda_mmid_cold(n) && n->src[0]->type == GGML_TYPE_Q2_0 &&
           n->src[1]->type == GGML_TYPE_F32 && n->type == GGML_TYPE_F32 &&
           ggml_cuda_mmid_vec_f16_sm60_supported(n->src[0], n->src[1], n->src[2], n);
}

bool ggml_cuda_moe_host_begin(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_MUL_MAT_ID || !ggml_cuda_mmid_cold(node)) {
        return false;
    }
    if (ggml_cuda_moe_host_active(ctx, node)) {
        return true;
    }
    if (!ggml_cuda_moe_host_enabled()) {
        return false;
    }
    GGML_ASSERT(ctx.moe_host.ids == nullptr && "moe-host: the previous triple was not collected");

    const ggml_tensor * ids = node->src[2];
    const int n_used   = (int) ids->ne[0];
    const int n_tokens = (int) ids->ne[1];
    if (n_tokens > MH_MAX_TOK || n_used*n_tokens > MH_MAX_PAIRS || ids->type != GGML_TYPE_I32 || !mh_is_cold_q2(node)) {
        return false;
    }

    // the other two MUL_MAT_IDs on the same ids and the swiglu between gate/up and down
    const ggml_tensor * mm[3] = { node, nullptr, nullptr };
    int n_mm = 1;
    const ggml_tensor * glu = nullptr;
    for (int j = i + 1; j < std::min(cgraph->n_nodes, i + 32) && (n_mm < 3 || glu == nullptr); ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_MUL_MAT_ID && n->src[2] == ids) {
            if (n_mm == 3 || !mh_is_cold_q2(n)) {
                return false;
            }
            mm[n_mm++] = n;
        } else if (n->op == GGML_OP_GLU && glu == nullptr) {
            glu = n;
        }
    }
    if (n_mm != 3 || glu == nullptr || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    const ggml_tensor * down = nullptr;
    bool has_gate = false, has_up = false;
    for (const ggml_tensor * m : mm) {
        has_gate |= m == gate;
        has_up   |= m == up;
        if (m != gate && m != up) {
            down = m;
        }
    }
    if (!has_gate || !has_up || gate == up || down == nullptr || down->src[1] != glu) {
        return false;
    }
    const ggml_tensor * x = up->src[1];
    const int64_t n_embd = up->src[0]->ne[0];
    const int64_t n_ff   = up->src[0]->ne[1];
    const int64_t n_hot  = up->src[0]->ne[2];
    if (gate->src[1] != x || x->ne[0] != n_embd || x->ne[1] != 1 || x->ne[2] != n_tokens || x->nb[0] != sizeof(float) ||
        x->nb[2] % 16 != 0 || !ggml_are_same_shape(gate->src[0], up->src[0]) ||
        down->src[0]->ne[0] != n_ff || down->src[0]->ne[1] != n_embd || down->src[0]->ne[2] != n_hot ||
        !ggml_is_contiguous(down) || n_embd % 64 != 0 || n_ff % 64 != 0 || n_embd > MH_MAX_EMBD || n_ff > MH_MAX_FF) {
        return false;
    }
    const char * c_up = nullptr, * c_gate = nullptr, * c_down = nullptr;
    int n_up = 0, n_gate = 0, n_down = 0;
    ggml_cuda_mmid_cold(up,   &c_up,   &n_up);
    ggml_cuda_mmid_cold(gate, &c_gate, &n_gate);
    ggml_cuda_mmid_cold(down, &c_down, &n_down);
    if (n_up != n_gate || n_up != n_down || n_up <= 0) {
        return false;
    }
    if (!ggml_cuda_moe_host_init(ctx)) {
        return false;
    }

    mh_slot_desc d;
    d.up       = (const uint8_t *) c_up;
    d.gate     = (const uint8_t *) c_gate;
    d.down     = (const uint8_t *) c_down;
    d.up_nb1   = up->src[0]->nb[1];
    d.up_nb2   = up->src[0]->nb[2];
    d.gate_nb1 = gate->src[0]->nb[1];
    d.gate_nb2 = gate->src[0]->nb[2];
    d.down_nb1 = down->src[0]->nb[1];
    d.down_nb2 = down->src[0]->nb[2];
    d.n_embd   = (int) n_embd;
    d.n_ff     = (int) n_ff;
    const int slot = mh_pool_register(d);

    mh_publish<<<1, MH_PUBLISH_THREADS, 0, ctx.stream()>>>(
        (const int32_t *) ids->data, (int) (ids->nb[1]/sizeof(int32_t)), n_used, n_tokens, (int) n_hot,
        (const float *) x->data, (int64_t) (x->nb[2]/sizeof(float)), (int) n_embd,
        ctx.moe_host_mb, ctx.moe_host_dstate, slot);
    CUDA_CHECK(cudaGetLastError());

    ctx.moe_host.ids  = ids;
    ctx.moe_host.up   = up;
    ctx.moe_host.gate = gate;
    ctx.moe_host.down = down;
    return true;
}

void ggml_cuda_moe_host_end(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    if (ctx.moe_host.down == nullptr || node != ctx.moe_host.down) {
        return;
    }
    const int n_used = (int) node->src[2]->ne[0];
    mh_collect<<<MH_COLLECT_BLOCKS, 256, 0, ctx.stream()>>>(
        ctx.moe_host_mb, ctx.moe_host_dstate, (float *) node->data,
        (int64_t) (node->nb[1]/sizeof(float)), (int64_t) (node->nb[2]/sizeof(float)), n_used, (int) node->ne[0]);
    CUDA_CHECK(cudaGetLastError());
    ctx.moe_host = {};
}
