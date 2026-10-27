#include "moe-host.cuh"
#include "moe-host-cpu.h"
#include "expert-cache.cuh"
#include "fn-engine.cuh"
#include "mmid-f16-sm60.cuh"
#include "mmvq.cuh"

#include <atomic>
#include <climits>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#define MH_DSTATE_SEQ  0 // last request number
#define MH_DSTATE_NEED 1 // cold pairs of the last request
#define MH_DSTATE_RSEQ 2 // routing kernels launched (mh_route: the quantize blocks' counter is compared against it)
#define MH_DSTATE_QCNT 3 // quantize blocks done, monotonic
#define MH_DSTATE_PAIR 4 // their t*n_used + k, cap_pairs entries

#define MH_ROUTE_QBLOCKS 8 // blocks of mh_route that quantize the rows while block 0 routes

#define MH_PUBLISH_THREADS 256 // decode batches: one fused publish block, one thread per (token, slot) pair
#define MH_RING_THREADS    1024
#define MH_SMALL_TOK       8   // up to this many tokens the fused publish is used

static __device__ __forceinline__ unsigned long long mh_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static bool ggml_cuda_moe_host_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_MOE_HOST");
        return (e == nullptr || atoi(e) != 0) && mh_cpu_supported();
    }();
    return enabled;
}

// One warp quantizes 64 values into the host kernels' activation block: staged in shared memory, copied out with
// 16-byte stores (mapped memory is written over PCIe).
static __device__ __forceinline__ void mh_quantize_block(const float * __restrict__ x, mh_act_block & ob, mh_act_block * out, const int lane) {
    const float2 v = *(const float2 *) (x + 2*lane); // elements 2*lane, 2*lane + 1
    float amax = fmaxf(fabsf(v.x), fabsf(v.y));
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o)); // within each half of the warp
    }
    const float d  = amax/127.0f;
    const float id = d != 0.0f ? 1.0f/d : 0.0f;
    const int   q0 = __float2int_rn(v.x*id);
    const int   q1 = __float2int_rn(v.y*id);
    int sum = q0 + q1;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    const int e0 = 2*lane;
    const int e1 = 2*lane + 1;
    ob.q[16*(e0 % 4) + e0/4] = (int8_t) q0;
    ob.q[16*(e1 % 4) + e1/4] = (int8_t) q1;
    const float d0 = __shfl_sync(0xffffffff, d,   0);
    const float d1 = __shfl_sync(0xffffffff, d,   16);
    const int   s1 = __shfl_sync(0xffffffff, sum, 16);
    if (lane < 16) {
        ob.s[lane] = (lane & 2) ? d1 : d0;
    }
    if (lane == 0) {
        ob.corr = d*(float) sum + d1*(float) s1;
    }
    __syncwarp();
    if (lane < (int) (sizeof(mh_act_block)/16)) {
        ((int4 *) out)[lane] = ((const int4 *) &ob)[lane];
    }
    __syncwarp();
}

// Prompt-sized batches, 1/2: quantize all rows, one warp per block of 64.
static __global__ void mh_quantize_rows(const float * __restrict__ x, const int64_t sx_tok, const int n_embd, const int n_tokens,
                                        mh_mailbox * mb) {
    __shared__ mh_act_block s_blk[MH_PUBLISH_THREADS/WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int w    = threadIdx.x / WARP_SIZE;
    const int nb_t = n_embd/64;
    const int bi   = blockIdx.x*(MH_PUBLISH_THREADS/WARP_SIZE) + w;
    if (bi < n_tokens*nb_t) {
        mh_quantize_block(x + (bi / nb_t)*sx_tok + 64*(bi % nb_t), s_blk[w], mh_xq(mb) + bi, lane);
    }
}

// Prompt-sized batches, 2/2: compact the cold pairs (one block, a running offset over chunks of the ids) and ring.
static __global__ void mh_ring(const int32_t * __restrict__ ids, const int si1, const int n_used, const int n_tokens,
                               const int n_hot, mh_mailbox * mb, uint32_t * dstate, const int slot) {
    __shared__ int wcount[MH_RING_THREADS/WARP_SIZE];
    __shared__ int s_base;
    const int lane  = threadIdx.x % WARP_SIZE;
    const int w     = threadIdx.x / WARP_SIZE;
    const int n_all = n_tokens*n_used;
    if (threadIdx.x == 0) {
        s_base = 0;
    }
    __syncthreads();
    for (int c0 = 0; c0 < n_all; c0 += MH_RING_THREADS) {
        const int i = c0 + threadIdx.x;
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
        int base = s_base;
        int nc   = 0;
        for (int j = 0; j < MH_RING_THREADS/WARP_SIZE; ++j) {
            base += j < w ? wcount[j] : 0;
            nc   += wcount[j];
        }
        if (cold) {
            const int pos = base + __popc(m & ((1u << lane) - 1));
            mh_pair_idx(mb)[pos] = i;
            mh_pair_exp(mb)[pos] = e - n_hot;
            dstate[MH_DSTATE_PAIR + pos] = i;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            s_base += nc;
        }
        __syncthreads();
    }
    const int np = s_base;
    if (threadIdx.x == 0) {
        dstate[MH_DSTATE_NEED] = np;
    }
    if (np == 0) {
        return;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
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

// Decode batches: one thread per (token, slot) pair, the cold ones compacted with a ballot; then the input rows are
// quantized and the doorbell is rung. Nothing is published when the step has no cold pair.
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
        mh_pair_idx(mb)[pos] = i;
        mh_pair_exp(mb)[pos] = e - n_hot;
        dstate[MH_DSTATE_PAIR + pos] = i;
    }
    if (i == 0) {
        dstate[MH_DSTATE_NEED] = np;
    }
    if (np == 0) {
        return;
    }

    // quantize the rows into the host kernels' activation blocks
    __shared__ mh_act_block s_blk[MH_PUBLISH_THREADS/WARP_SIZE];
    const int nb_t = n_embd/64;
    for (int b0 = 0; b0 < n_tokens*nb_t; b0 += MH_PUBLISH_THREADS/WARP_SIZE) {
        const int bi = b0 + w;
        if (bi < n_tokens*nb_t) {
            mh_quantize_block(x + (bi / nb_t)*sx_tok + 64*(bi % nb_t), s_blk[w], mh_xq(mb) + bi, lane);
        }
    }
    if (i == 0) {
        mb->slot     = slot;
        mb->n_tokens = n_tokens;
        mb->n_used   = n_used;
        mb->n_pairs  = np;
    }
    // One fence per thread, then the barrier: everything above is in host memory before the doorbell is written.
    // A fence takes microseconds on GP100; the doorbell needs none after it.
    __threadfence_system();
    __syncthreads();
    if (i == 0) {
        const uint32_t s = dstate[MH_DSTATE_SEQ] + 1;
        dstate[MH_DSTATE_SEQ] = s;
        mb->req = s;
    }
}

// The hot pairs of a window grouped by expert, two pairs (tokens) per group, for fn_moe_up_g, which reads such an expert
// once for both: groups = [count][group: first pair | (second pair + 1) << 16, 0 when there is none], in pair order.
// Windows of up to 32 pairs (mh_route's caller passes null otherwise); called by warp 0 of mh_route with lane i = pair i:
// an occurrence of an expert of even rank starts a group, the next occurrence is its second pair.
static __device__ __forceinline__ void mh_groups(uint32_t * groups, const int i, const bool hot, const int e) {
    const int ek = hot ? e : -1 - i; // cold pairs and unused lanes match nothing
    int rank = 0;
    int nxt  = -1;
#pragma unroll
    for (int j = 0; j < WARP_SIZE; ++j) {
        const int ej = __shfl_sync(0xffffffff, ek, j);
        rank += j < i && ej == ek;
        nxt   = j > i && ej == ek && nxt < 0 ? j : nxt;
    }
    const bool         start = hot && (rank & 1) == 0;
    const unsigned int sb    = __ballot_sync(0xffffffff, start);
    if (start) {
        groups[1 + __popc(sb & ((1u << i) - 1))] = (uint32_t) i | ((uint32_t) (nxt + 1) << 16);
    }
    if (i == 0) {
        groups[0] = (uint32_t) __popc(sb);
    }
}

#define MH_ROUTE_EXPERTS 512
#define MH_ROUTE_EPT     (MH_ROUTE_EXPERTS/WARP_SIZE) // experts per thread

// Decode batches, with the routing: warp w of block 0 takes the top n_used of token w's logits (softmax, the weights
// divided by their clamped sum, as topk_moe_cuda does), a thread per pair then writes the expert's position
// (expert-cache.cuh) and its weight, and the block goes on as mh_publish. The quantized rows the host kernels read
// are written by blocks 1..MH_ROUTE_QBLOCKS meanwhile (the quantize on one block took 13 us after the routing's 16;
// the host does not read them before the doorbell, which block 0 rings once their counter says they are done).
// logits, weights and ids may share memory.
static __global__ void mh_route(
        const float * logits, const float * __restrict__ logits2, float * weights, int32_t * ids, const int si1, const int n_used, const int n_tokens,
        const float clamp_val, const int32_t * __restrict__ perm, uint32_t * counts, const int n_hot,
        const float * __restrict__ x, const int64_t sx_tok, const int n_embd,
        mh_mailbox * mb, uint32_t * dstate, const int slot, unsigned long long * stats, uint32_t * groups, const bool early_fence) {
    __shared__ int   s_e[MH_PUBLISH_THREADS];
    __shared__ float s_wt[MH_PUBLISH_THREADS];
    __shared__ float s_norm[MH_PUBLISH_THREADS/WARP_SIZE];
    __shared__ int   wcount[MH_PUBLISH_THREADS/WARP_SIZE];
    __shared__ mh_act_block s_blk[MH_PUBLISH_THREADS/WARP_SIZE];

    const int i     = threadIdx.x;
    const int lane  = i % WARP_SIZE;
    const int w     = i / WARP_SIZE;
    const int n_all = n_tokens*n_used;
    const int nb_t  = n_embd/64;

    if (blockIdx.x > 0) {
        // a quantize block: its share of the rows' blocks into the mailbox, fenced, then counted
        constexpr int nw = MH_PUBLISH_THREADS/WARP_SIZE;
        for (int b0 = (blockIdx.x - 1)*nw; b0 < n_tokens*nb_t; b0 += MH_ROUTE_QBLOCKS*nw) {
            const int bi = b0 + w;
            if (bi < n_tokens*nb_t) {
                mh_quantize_block(x + (bi / nb_t)*sx_tok + 64*(bi % nb_t), s_blk[w], mh_xq(mb) + bi, lane);
            }
        }
        __threadfence_system();
        __syncthreads();
        if (i == 0) {
            atomicAdd(dstate + MH_DSTATE_QCNT, 1u);
        }
        return;
    }
    uint32_t rseq = 0;
    if (i == 0) {
        rseq = dstate[MH_DSTATE_RSEQ] + 1;
        dstate[MH_DSTATE_RSEQ] = rseq;
    }
    // the expert cache's positions, fetched while the warps route (a dependent read of ~1.5 us from a cold L2 after it)
    __shared__ int s_perm[MH_ROUTE_EXPERTS];
    if (perm != nullptr) {
        for (int q = i; q < MH_ROUTE_EXPERTS; q += MH_PUBLISH_THREADS) {
            s_perm[q] = perm[q];
        }
    }

    if (w < n_tokens) {
        float wt[MH_ROUTE_EPT];
        const float * lg = logits + (int64_t) w*MH_ROUTE_EXPERTS;
#pragma unroll
        for (int q = 0; q < MH_ROUTE_EPT; ++q) {
            wt[q] = lg[lane + q*WARP_SIZE];
        }
        if (logits2 != nullptr) {
#pragma unroll
            for (int q = 0; q < MH_ROUTE_EPT; ++q) {
                wt[q] += logits2[(int64_t) w*MH_ROUTE_EXPERTS + lane + q*WARP_SIZE];
            }
        }
        float mx = wt[0];
#pragma unroll
        for (int q = 1; q < MH_ROUTE_EPT; ++q) {
            mx = fmaxf(mx, wt[q]);
        }
        mx = warp_reduce_max(mx);
        float sum = 0.0f;
#pragma unroll
        for (int q = 0; q < MH_ROUTE_EPT; ++q) {
            wt[q] = expf(wt[q] - mx);
            sum  += wt[q];
        }
        sum = warp_reduce_sum(sum);
        const float inv = 1.0f/sum;
#pragma unroll
        for (int q = 0; q < MH_ROUTE_EPT; ++q) {
            wt[q] *= inv;
            if (__isnanf(wt[q])) {
                wt[q] = -FLT_MAX; // the rounds below then still find distinct experts
            }
        }
        float wsum = 0.0f;
        for (int k = 0; k < n_used; ++k) {
            float mv = wt[0];
            int   me = lane;
#pragma unroll
            for (int q = 1; q < MH_ROUTE_EPT; ++q) {
                if (wt[q] > mv) {
                    mv = wt[q];
                    me = lane + q*WARP_SIZE;
                }
            }
#pragma unroll
            for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
                const float v = __shfl_xor_sync(0xffffffff, mv, o);
                const int   e = __shfl_xor_sync(0xffffffff, me, o);
                if (v > mv || (v == mv && e < me)) {
                    mv = v;
                    me = e;
                }
            }
            if ((me & (WARP_SIZE - 1)) == lane) {
                wt[me / WARP_SIZE] = -INFINITY;
            }
            wsum += mv;
            if (lane == 0) {
                s_e[w*n_used + k]  = me;
                s_wt[w*n_used + k] = mv;
            }
        }
        if (lane == 0) {
            s_norm[w] = 1.0f/fmaxf(wsum, clamp_val);
        }
    }
    __syncthreads();

    int  e    = 0;
    bool cold = false;
    if (i < n_all) {
        const int t  = i / n_used;
        const int k  = i - t*n_used;
        const int ex = s_e[i];
        e = ex;
        if (perm != nullptr) {
            e = s_perm[ex];
            atomicAdd(counts + ex, 1u);
        }
        ids[t*si1 + k]         = e;
        weights[t*n_used + k]  = s_wt[i]*s_norm[t];
        cold = e >= n_hot;
    }
    // temporary (GGML_CUDA_FN_WAITS): pairs whose expert an earlier pair of the window already has
    if (stats != nullptr && n_tokens == 3 && i < n_all) {
        bool first = true;
        for (int q = 0; q < i; ++q) {
            first = first && s_e[q] != s_e[i];
        }
        atomicAdd(stats + 267, 1ull);
        if (first) {
            atomicAdd(stats + 268, 1ull);
        }
        if (!cold) {
            atomicAdd(stats + 265, 1ull);
            if (first) {
                atomicAdd(stats + 266, 1ull);
            }
        }
        if (i == 0) {
            atomicAdd(stats + 264, 1ull);
        }
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
        mh_pair_idx(mb)[pos] = i;
        mh_pair_exp(mb)[pos] = e - n_hot;
        dstate[MH_DSTATE_PAIR + pos] = i;
    }
    if (i == 0) {
        dstate[MH_DSTATE_NEED] = np;
    }
    if (np == 0) {
        if (groups != nullptr && w == 0) {
            mh_groups(groups, i, i < n_all && !cold, e);
        }
        return;
    }
    if (i == 0) {
        mb->slot     = slot;
        mb->n_tokens = n_tokens;
        mb->n_used   = n_used;
        mb->n_pairs  = np;
    }
    // This block's mailbox writes are made visible while the quantize blocks finish theirs (each fences before it
    // counts itself): the fence (~6 us on GP100) used to start only after the counter said they were done. The doorbell
    // store follows the counter's spin, so the host cannot see it before any of the request's data.
    // GGML_CUDA_MOE_HOST_EARLY_FENCE=0: fence after the spin, as before.
    if (stats != nullptr && i == 0) {
        mb->t_req = mh_globaltimer(); // FN_WAITS: the transport latency of the request (fences, doorbell, host's poll)
    }
    if (early_fence) {
        __threadfence_system();
        __syncthreads();
    }
    if (i == 0) {
        // the quantize blocks of this launch are done (the counter counts every launch's)
        const long long t0 = clock64();
        while ((uint32_t) (atomicAdd(dstate + MH_DSTATE_QCNT, 0u) - rseq*MH_ROUTE_QBLOCKS) & 0x80000000u) {
            if (clock64() - t0 > 2000000000LL) {
                __trap(); // the quantize blocks of this launch did not finish (no printf: it costs the kernel a stack)
            }
        }
    }
    if (!early_fence) {
        __threadfence_system();
        __syncthreads();
    }
    if (i == 0) {
        const uint32_t s = dstate[MH_DSTATE_SEQ] + 1;
        dstate[MH_DSTATE_SEQ] = s;
        mb->req = s;
    }
    // after the doorbell: the host threads need nothing of this
    if (groups != nullptr && w == 0) {
        mh_groups(groups, i, i < n_all && !cold, e);
    }
}

static __global__ void mh_collect(
        const mh_mailbox * mb, const uint32_t * __restrict__ dstate, float * __restrict__ dst,
        const int64_t sd_slot, const int64_t sd_tok, const int n_used, const int n_embd, unsigned long long * dbg) {
    const int np = dstate[MH_DSTATE_NEED];
    if (np == 0 || (int) blockIdx.x >= np) {
        return;
    }
    if (threadIdx.x == 0) {
        const uint32_t  s  = dstate[MH_DSTATE_SEQ];
        const long long t0 = clock64();
        const unsigned long long g0 = dbg != nullptr ? mh_globaltimer() : 0;
        while (*((volatile const uint32_t *) &mb->done) != s) {
            if (clock64() - t0 > 20000000000LL) {
                printf("moe-host: no answer from the host threads for request %u\n", s);
                __trap();
            }
        }
        if (dbg != nullptr && blockIdx.x == 0) {
            atomicAdd(dbg + 0, (unsigned long long) (clock64() - t0));
            atomicAdd(dbg + 1, 1ull);
            // ns: the answer's flight to the GPU (in the host's estimate of the GPU time base), and the GPU's work from the
            // request to the collect
            const unsigned long long g2 = mh_globaltimer();
            const unsigned long long td = mb->t_done;
            const unsigned long long tr = mb->t_req;
            if (td != 0 && tr != 0) {
                atomicAdd(dbg + 269, (unsigned long long) ((long long) g2 - (long long) td));
                atomicAdd(dbg + 270, 1ull);
                atomicAdd(dbg + 4, (unsigned long long) ((long long) g0 - (long long) tr));
                atomicAdd(dbg + 5, 1ull);
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
        const float4 * src = (const float4 *) (mh_y((mh_mailbox *) mb) + (int64_t) p*n_embd);
        for (int c = threadIdx.x; c < n4; c += blockDim.x) {
            out[c] = __ldcv(src + c);
        }
    }
}

// The same for a down projection whose output is the sum of the token's pairs times their weights (fn-engine.cuh):
// one block per token adds the token's cold rows to it.
static __global__ void mh_collect_weighted(
        const mh_mailbox * mb, const uint32_t * __restrict__ dstate, const ggml_cuda_fn_moe_route * __restrict__ route,
        float * __restrict__ y, const int64_t sy, const int n_used, const int n_embd, unsigned long long * dbg) {
    const int np = dstate[MH_DSTATE_NEED];
    if (np == 0) {
        return;
    }
    if (threadIdx.x == 0) {
        const uint32_t  s  = dstate[MH_DSTATE_SEQ];
        const long long t0 = clock64();
        const unsigned long long g0 = dbg != nullptr ? mh_globaltimer() : 0;
        while (*((volatile const uint32_t *) &mb->done) != s) {
            if (clock64() - t0 > 20000000000LL) {
                printf("moe-host: no answer from the host threads for request %u\n", s);
                __trap();
            }
        }
        if (dbg != nullptr && blockIdx.x == 0) {
            atomicAdd(dbg + 0, (unsigned long long) (clock64() - t0));
            atomicAdd(dbg + 1, 1ull);
            // ns: the answer's flight to the GPU (in the host's estimate of the GPU time base), and the GPU's work from the
            // request to the collect
            const unsigned long long g2 = mh_globaltimer();
            const unsigned long long td = mb->t_done;
            const unsigned long long tr = mb->t_req;
            if (td != 0 && tr != 0) {
                atomicAdd(dbg + 269, (unsigned long long) ((long long) g2 - (long long) td));
                atomicAdd(dbg + 270, 1ull);
                atomicAdd(dbg + 4, (unsigned long long) ((long long) g0 - (long long) tr));
                atomicAdd(dbg + 5, 1ull);
            }
        }
    }
    __syncthreads();
    const int n4 = n_embd/4;
    float4 *  out = (float4 *) (y + blockIdx.x*sy);
    for (int p = 0; p < np; ++p) {
        const int idx = dstate[MH_DSTATE_PAIR + p];
        if (idx / n_used != (int) blockIdx.x) {
            continue;
        }
        const float    w   = route[idx].w;
        const float4 * src = (const float4 *) (mh_y((mh_mailbox *) mb) + (int64_t) p*n_embd);
        for (int c = threadIdx.x; c < n4; c += blockDim.x) {
            const float4 v = __ldcv(src + c);
            const float4 o = out[c];
            out[c] = make_float4(o.x + w*v.x, o.y + w*v.y, o.z + w*v.z, o.w + w*v.w);
        }
    }
}

static std::atomic<int> g_mh_mailboxes { 0 };

// The mailbox is sized once, at the first host triple: GGML_CUDA_MOE_HOST_MAX_TOK tokens (default 512, the usual
// prompt ubatch) of that triple's n_embd. Wider batches keep the GPU path. It is never reallocated: graphs queued
// earlier may still be using it.
static bool ggml_cuda_moe_host_init(ggml_backend_cuda_context & ctx, const int n_embd, const int n_used) {
    if (ctx.moe_host_mb != nullptr) {
        return true;
    }
    // the request counter must not be reset by a captured memset, so set up before any capture
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &status));
    if (status != cudaStreamCaptureStatusNone) {
        return false;
    }
    const char * env = getenv("GGML_CUDA_MOE_HOST_MAX_TOK");
    const int cap_tok   = std::max(MH_SMALL_TOK, env ? atoi(env) : 512);
    const int cap_pairs = cap_tok*n_used;
    const size_t size   = mh_mailbox_layout(nullptr, cap_tok, cap_pairs, n_embd);

    ggml_cuda_set_device(ctx.device);
    mh_mailbox * mb = nullptr;
    const int numa_node = ggml_cuda_host_numa_node_of_device(ctx.device);
    if (numa_node >= 0) {
        ggml_cuda_host_numa_bind_node(true, numa_node);
    }
    const cudaError_t mb_err = cudaHostAlloc((void **) &mb, size, cudaHostAllocMapped | cudaHostAllocPortable);
    if (numa_node >= 0) {
        ggml_cuda_host_numa_bind_node(false, numa_node);
    }
    if (mb_err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: no pinned mailbox (%.1f MiB), cold experts stay on the GPU\n", __func__, size/1048576.0);
        return false;
    }
    memset((void *) mb, 0, sizeof(mh_mailbox));
    mh_mailbox_layout(mb, cap_tok, cap_pairs, n_embd);
    // [state][pairs: cap_pairs][groups: 1 + cap_pairs] (mh_route)
    CUDA_CHECK(cudaMalloc((void **) &ctx.moe_host_dstate, (MH_DSTATE_PAIR + 1 + 2*(size_t) cap_pairs)*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(ctx.moe_host_dstate, 0, (MH_DSTATE_PAIR + 1 + 2*(size_t) cap_pairs)*sizeof(uint32_t)));
    ctx.moe_host_mb = mb;
    mh_pool_attach(g_mh_mailboxes.fetch_add(1), mb, ggml_cuda_info().device_count, numa_node >= 0 ? ctx.device : -1);
    return true;
}

static void mh_mirror_check(int device);

void ggml_cuda_moe_host_free(ggml_backend_cuda_context & ctx) {
    mh_mirror_check(ctx.device);
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
    if (n->op != GGML_OP_MUL_MAT_ID || !ggml_cuda_mmid_cold(n) ||
        (n->src[0]->type != GGML_TYPE_Q2_0 && n->src[0]->type != GGML_TYPE_Q2_K && n->src[0]->type != GGML_TYPE_Q3_K &&
         n->src[0]->type != GGML_TYPE_Q4_0) ||
        n->src[1]->type != GGML_TYPE_F32 || n->type != GGML_TYPE_F32) {
        return false;
    }
    if (n->src[0]->type == GGML_TYPE_Q2_K || n->src[0]->type == GGML_TYPE_Q3_K || n->src[0]->type == GGML_TYPE_Q4_0) {
        // glm5-next (Q4_0: its NextN block): the hot pairs run the generic mul_mat_id (mmvq Q2_K), the cold ones the host's Q2_K kernels
        // (GGML_CUDA_MMID_ALLGPU=1: the whole layer on the GPU instead, a correctness bisect)
        static const bool allgpu = getenv("GGML_CUDA_MMID_ALLGPU") != nullptr && atoi(getenv("GGML_CUDA_MMID_ALLGPU")) != 0;
        return !allgpu;
    }
    // the hot pairs go to the sm_60 kernels: the mat-vec for decode batches, the GEMM for prompt batches
    return n->ne[2] <= MMVQ_MAX_BATCH_SIZE ? ggml_cuda_mmid_vec_f16_sm60_supported(n->src[0], n->src[1], n->src[2], n)
                                           : ggml_cuda_mmid_f16_sm60_supported(n->src[0], n->src[1], n->src[2], n);
}

bool ggml_cuda_moe_host_takes_prompts() {
    // prompt batches: the GPU with prefetched cold experts is faster than the host threads here (many tokens per
    // expert make the host kernels compute-bound); GGML_CUDA_MOE_HOST_PROMPT=1 sends them to the host anyway
    static const bool prompt = getenv("GGML_CUDA_MOE_HOST_PROMPT") != nullptr && atoi(getenv("GGML_CUDA_MOE_HOST_PROMPT")) != 0;
    return prompt && ggml_cuda_moe_host_enabled();
}

// GGML_CUDA_MOE_HOST_PROMPT_MAX_TOK=n: with GGML_CUDA_MOE_HOST_PROMPT=1 the host tier takes prompt batches of up to n tokens only,
// the larger ones are the GPU's (Q2_K / Q3_K: the cold-expert staging, cold-kq.cuh). Default: no limit.
static int ggml_cuda_moe_host_prompt_max() {
    static const int n = getenv("GGML_CUDA_MOE_HOST_PROMPT_MAX_TOK") != nullptr ? atoi(getenv("GGML_CUDA_MOE_HOST_PROMPT_MAX_TOK")) : INT_MAX;
    return n;
}

// The gate/up/swiglu/down triple of a hot/cold Q2_0 layer that the MUL_MAT_ID node i opens: the other two
// MUL_MAT_IDs on the same ids and the swiglu between gate/up and down.
static bool mh_triple(const ggml_cgraph * cgraph, const int i, const ggml_tensor *& up, const ggml_tensor *& gate,
                      const ggml_tensor *& down, const ggml_tensor ** glu_out = nullptr) {
    const ggml_tensor * node = cgraph->nodes[i];
    const ggml_tensor * ids  = node->src[2];
    if (ids->type != GGML_TYPE_I32 || !mh_is_cold_q2(node)) {
        return false;
    }
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
    if (n_mm != 3 || glu == nullptr || ggml_get_op_params_i32(glu, 1) != 0 ||
        (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU && ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU_CLAMP)) {
        return false;
    }
    if (glu_out != nullptr) {
        *glu_out = glu;
    }
    gate = glu->src[0];
    up   = glu->src[1];
    down = nullptr;
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
    const int n_tokens = (int) ids->ne[1];
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
    int n_up = 0, n_gate = 0, n_down = 0;
    ggml_cuda_mmid_cold(up,   nullptr, &n_up);
    ggml_cuda_mmid_cold(gate, nullptr, &n_gate);
    ggml_cuda_mmid_cold(down, nullptr, &n_down);
    return n_up == n_gate && n_up == n_down && n_up > 0;
}

// GGML_CUDA_MOE_HOST_MIRROR=1 (with GGML_CUDA_MOE_HOST_NUMA_POOLS=1): the cold experts stay pinned next to the GPUs, where
// the prompt path copies them to VRAM (cold prefetch), and the host tier pool of a device on the other socket reads a copy
// of them on its own node. Q262 z20: the per-socket pools take 0.7-1.0 ms off a decode step, but with device 0's cold
// experts on node 0 every prompt ubatch fetched them across the sockets (+4-6% cold prefill). The copy is made when the
// layer is first seen (before the expert cache registers it), is pinned and mapped so that the expert cache's exchanges
// write evicted slices into both (ec_swap), and costs the cold set of that device once more in host memory.
struct mh_mirror_tab {
    std::mutex                                     mtx;
    std::unordered_map<const char *, const char *> map;   // primary -> copy (itself when no copy is needed)
    std::unordered_map<const char *, size_t>       size;  // primary -> bytes of a copy
    size_t                                         bytes = 0;
    int                                            n     = 0;
};
static mh_mirror_tab g_mh_mirror;

#if defined(__linux__)
static int mh_page_node(const void * p) {
    void * page   = (void *) ((uintptr_t) p & ~(uintptr_t) 4095);
    int    status = -1;
    if (syscall(SYS_move_pages, 0, 1, &page, nullptr, &status, 0) != 0) {
        return -1;
    }
    return status;
}

static const char * mh_mirror_make(const char * c, const size_t bytes, const int node, const int device) {
    // GGML_CUDA_MOE_HOST_MIRROR_HUGE=1: the copy on 2 MB pages (p6e: the same VRAM at load, 15119 MiB, and no faster jobs; off)
    static const bool huge = getenv("GGML_CUDA_MOE_HOST_MIRROR_HUGE") != nullptr && atoi(getenv("GGML_CUDA_MOE_HOST_MIRROR_HUGE")) != 0;
    const size_t align = huge ? (size_t) 2 << 20 : 4096;
    const size_t len   = (bytes + align - 1)/align*align;
    char * base = (char *) mmap(nullptr, len + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        return nullptr;
    }
    char * p = (char *) (((uintptr_t) base + align - 1) & ~(uintptr_t) (align - 1));
    if (p != base) {
        munmap(base, p - base);
    }
    if (p + len != base + len + align) {
        munmap(p + len, base + len + align - (p + len));
    }
    if (huge) {
        madvise(p, len, MADV_HUGEPAGE);
    }
    const unsigned long mask = 1ul << node;
    // MPOL_PREFERRED: a full node spills over instead of an OOM kill
    if (syscall(SYS_mbind, p, len, 1 /* MPOL_PREFERRED */, &mask, 64, 0) != 0) {
        GGML_LOG_WARN("%s: mbind to NUMA node %d failed\n", __func__, node);
    }
    {
        const size_t n_threads = 8;
        std::vector<std::thread> workers;
        for (size_t t = 0; t < n_threads; ++t) {
            workers.emplace_back([=] {
                const size_t a = std::min(bytes, (bytes*t/n_threads + 4095)/4096*4096);
                const size_t b = t + 1 == n_threads ? bytes : std::min(bytes, (bytes*(t + 1)/n_threads + 4095)/4096*4096);
                if (b > a) {
                    memcpy(p + a, c + a, b - a);
                }
            });
        }
        for (auto & w : workers) {
            w.join();
        }
    }
    // mapped into the pool's device only (its exchange kernel writes the copy): a portable registration maps it into every
    // device and the page tables of 8.75 GiB cost each one ~18 MiB of VRAM (g5: card 1 16195 vs 16175 MiB)
    ggml_cuda_set_device(device);
    cudaError_t err = cudaHostRegister(p, len, cudaHostRegisterMapped);
    void * dp = nullptr;
    if (err == cudaSuccess) {
        err = cudaHostGetDevicePointer(&dp, p, 0);
        if (err == cudaSuccess && dp != (void *) p) {
            // the expert cache writes the copy through the host address (UVA)
            cudaHostUnregister(p);
            err = cudaErrorInvalidValue;
        }
    }
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: cudaHostRegister of the %.1f MiB copy failed (%s): the pool reads the primary\n", __func__,
                      len/1048576.0, cudaGetErrorString(err));
        munmap(p, len);
        return nullptr;
    }
    return p;
}
#endif

// the copy of the cold tensor data at c (bytes long) that the device's pool reads, or c itself; create: make it if there
// is none yet (outside a capture only; mh_open only looks up)
static const char * mh_mirror_of(const int device, const char * c, const size_t bytes, const bool create) {
    if (c == nullptr || !ggml_cuda_moe_host_mirror()) {
        return c;
    }
    const int node = ggml_cuda_host_numa_node_of_device(device);
    if (node < 0 || node == 7) {
        return c;
    }
    std::lock_guard<std::mutex> lock(g_mh_mirror.mtx);
    auto it = g_mh_mirror.map.find(c);
    if (it != g_mh_mirror.map.end()) {
        return it->second;
    }
    if (!create) {
        return c;
    }
    const char * m = c;
#if defined(__linux__)
    const int pn = mh_page_node(c);
    if (pn >= 0 && pn != node) {
        const char * p = mh_mirror_make(c, bytes, node, device);
        if (p != nullptr) {
            m = p;
            g_mh_mirror.bytes += bytes;
            g_mh_mirror.n++;
            g_mh_mirror.size[c] = bytes;
            if (g_mh_mirror.n <= 3 || g_mh_mirror.n % 48 == 0) {
                GGML_LOG_WARN("moe-host: dev%d cold copy %d on NUMA node %d (primary on node %d): %.1f MiB, %.2f GiB in all\n", device,
                              g_mh_mirror.n, mh_page_node(p), pn, bytes/1048576.0, g_mh_mirror.bytes/1073741824.0);
            }
        }
    }
#endif
    g_mh_mirror.map[c] = m;
    return m;
}

// GGML_CUDA_MOE_HOST_MIRROR_CHECK=1: when a context is freed, every copy is compared with its primary (they must be equal
// whenever no exchange is in flight: the exchanges write an evicted slice into both)
static void mh_mirror_check(const int device) {
    static const bool check = getenv("GGML_CUDA_MOE_HOST_MIRROR_CHECK") != nullptr && atoi(getenv("GGML_CUDA_MOE_HOST_MIRROR_CHECK")) != 0;
    if (!check || !ggml_cuda_moe_host_mirror()) {
        return;
    }
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::lock_guard<std::mutex> lock(g_mh_mirror.mtx);
    size_t n = 0, n_bad = 0, bytes = 0;
    for (const auto & kv : g_mh_mirror.size) {
        const char * c = kv.first;
        const char * m = g_mh_mirror.map[c];
        if (m == c) {
            continue;
        }
        n++;
        bytes += kv.second;
        size_t bad = 0;
        for (size_t o = 0; o < kv.second; o += 4096) {
            const size_t len = std::min<size_t>(4096, kv.second - o);
            bad += memcmp(c + o, m + o, len) != 0;
        }
        if (bad > 0) {
            n_bad++;
            GGML_LOG_WARN("moe-host: mirror check: copy of %p differs from its primary in %zu of %zu pages\n", (const void *) c, bad, (kv.second + 4095)/4096);
        }
    }
    GGML_LOG_WARN("moe-host: mirror check (device %d freed): %zu copies, %.2f GiB, %zu differ from their primary\n", device, n, bytes/1073741824.0, n_bad);
}

void ggml_cuda_moe_host_register(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (!ggml_cuda_mmid_cold(node) || !ggml_cuda_moe_host_enabled() || ggml_cuda_expert_cache_known(ctx, node->src[0])) {
        return;
    }
    const ggml_tensor * up = nullptr, * gate = nullptr, * down = nullptr;
    if (!mh_triple(cgraph, i, up, gate, down)) {
        return;
    }
    const char * c_up = nullptr, * c_gate = nullptr, * c_down = nullptr;
    int n_cold = 0;
    ggml_cuda_mmid_cold(up,   &c_up,   &n_cold);
    ggml_cuda_mmid_cold(gate, &c_gate, &n_cold);
    ggml_cuda_mmid_cold(down, &c_down, &n_cold);
    const char * m[3] = { nullptr, nullptr, nullptr };
    if (ggml_cuda_moe_host_mirror()) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &status));
        if (status != cudaStreamCaptureStatusNone) {
            return; // the copies and the expert cache's registration wait for a call outside a capture
        }
        const ggml_tensor * w[3] = { up->src[0], gate->src[0], down->src[0] };
        const char *        c[3] = { c_up, c_gate, c_down };
        for (int k = 0; k < 3; ++k) {
            m[k] = mh_mirror_of(ctx.device, c[k], (size_t) n_cold*w[k]->nb[2], true);
            m[k] = m[k] != c[k] ? m[k] : nullptr;
        }
    }
    ggml_cuda_expert_cache_register(ctx, up, gate, down, c_up, c_gate, c_down, n_cold, m[0], m[1], m[2]);
}

// what the publish of a triple needs
struct mh_open_args {
    const ggml_tensor * ids;
    const ggml_tensor * x;
    int                 n_used, n_tokens, n_hot, n_embd, slot;
};

// Opens the triple of the MUL_MAT_ID node i (its publish is then due); false: the node keeps the GPU path.
static bool mh_open(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const int i, mh_open_args & a) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (!ggml_cuda_moe_host_enabled()) {
        return false;
    }
    GGML_ASSERT(ctx.moe_host.ids == nullptr && "moe-host: the previous triple was not collected");

    const ggml_tensor * ids = node->src[2];
    const int n_used   = (int) ids->ne[0];
    const int n_tokens = (int) ids->ne[1];
    if (n_tokens > MH_SMALL_TOK && (!ggml_cuda_moe_host_takes_prompts() || n_tokens > ggml_cuda_moe_host_prompt_max())) {
        return false;
    }
    if (ctx.moe_host_mb != nullptr && (n_tokens > ctx.moe_host_mb->cap_tok || n_tokens*n_used > ctx.moe_host_mb->cap_pairs)) {
        return false;
    }
    const ggml_tensor * up = nullptr, * gate = nullptr, * down = nullptr, * glu = nullptr;
    if (!mh_triple(cgraph, i, up, gate, down, &glu)) {
        return false;
    }
    const ggml_tensor * x = up->src[1];
    const int64_t n_embd = up->src[0]->ne[0];
    const int64_t n_ff   = up->src[0]->ne[1];
    const int64_t n_hot  = up->src[0]->ne[2];
    if (ctx.moe_host_mb != nullptr && n_embd > ctx.moe_host_mb->cap_embd) {
        return false;
    }
    const char * c_up = nullptr, * c_gate = nullptr, * c_down = nullptr;
    ggml_cuda_mmid_cold(up,   &c_up);
    ggml_cuda_mmid_cold(gate, &c_gate);
    ggml_cuda_mmid_cold(down, &c_down);
    {
        static int n_logged = 0;
        if (n_logged < 4) {
            n_logged++;
            const ggml_tensor * cs = up->src[GGML_CUDA_EXPS_COLD_SRC];
            GGML_LOG_WARN("moe-host: route %s: cold up %p gate %p down %p (src[3] %s%s, buffer %s)\n", up->name,
                          (const void *) c_up, (const void *) c_gate, (const void *) c_down, cs ? cs->name : "none",
                          cs ? "" : ": op_params pointer", cs && cs->buffer ? ggml_backend_buffer_name(cs->buffer) : "-");
        }
    }
    if (!ggml_cuda_moe_host_init(ctx, (int) n_embd, n_used) ||
        n_tokens > ctx.moe_host_mb->cap_tok || n_tokens*n_used > ctx.moe_host_mb->cap_pairs) {
        return false;
    }

    mh_slot_desc d;
    d.n_cold   = 0;
    ggml_cuda_mmid_cold(up, nullptr, &d.n_cold);
    // the pool's copy of the cold experts (GGML_CUDA_MOE_HOST_MIRROR), when there is one
    d.up       = (const uint8_t *) mh_mirror_of(ctx.device, c_up,   (size_t) d.n_cold*up->src[0]->nb[2],   false);
    d.gate     = (const uint8_t *) mh_mirror_of(ctx.device, c_gate, (size_t) d.n_cold*gate->src[0]->nb[2], false);
    d.down     = (const uint8_t *) mh_mirror_of(ctx.device, c_down, (size_t) d.n_cold*down->src[0]->nb[2], false);
    d.up_nb1   = up->src[0]->nb[1];
    d.up_nb2   = up->src[0]->nb[2];
    d.gate_nb1 = gate->src[0]->nb[1];
    d.gate_nb2 = gate->src[0]->nb[2];
    d.down_nb1 = down->src[0]->nb[1];
    d.down_nb2 = down->src[0]->nb[2];
    d.n_embd   = (int) n_embd;
    d.n_ff     = (int) n_ff;
    d.wtype    = up->src[0]->type == GGML_TYPE_Q2_K ? 1 : up->src[0]->type == GGML_TYPE_Q4_0 ? 3 : 0;
    d.wtype_down = down->src[0]->type == GGML_TYPE_Q3_K ? 2 : down->src[0]->type == GGML_TYPE_Q4_0 ? 3 : d.wtype;
    GGML_ASSERT((d.wtype == 1 || d.wtype_down != 2) && "a Q3_K down needs Q2_K gate/up (natural-order activations)");
    GGML_ASSERT((d.wtype == 3) == (d.wtype_down == 3) && "Q4_0 experts need Q4_0 gate/up/down");
    GGML_ASSERT(gate->src[0]->type == up->src[0]->type && "gate and up of a layer share one type");
    d.clamp    = ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_CLAMP ? ggml_get_op_params_f32(glu, 3) : 0.0f;

    a.ids      = ids;
    a.x        = x;
    a.n_used   = n_used;
    a.n_tokens = n_tokens;
    a.n_hot    = (int) n_hot;
    a.n_embd   = (int) n_embd;
    a.slot     = mh_pool_register(d);

    ctx.moe_host.ids  = ids;
    ctx.moe_host.up   = up;
    ctx.moe_host.gate = gate;
    ctx.moe_host.down = down;
    return true;
}

bool ggml_cuda_moe_host_begin(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_MUL_MAT_ID || !ggml_cuda_mmid_cold(node)) {
        return false;
    }
    if (ggml_cuda_moe_host_active(ctx, node)) {
        return true;
    }
    mh_open_args a;
    if (!mh_open(ctx, cgraph, i, a)) {
        return false;
    }
    const int     si1    = (int) (a.ids->nb[1]/sizeof(int32_t));
    const int64_t sx_tok = (int64_t) (a.x->nb[2]/sizeof(float));
    if (a.n_tokens <= MH_SMALL_TOK && a.n_tokens*a.n_used <= MH_PUBLISH_THREADS) {
        mh_publish<<<1, MH_PUBLISH_THREADS, 0, ctx.stream()>>>(
            (const int32_t *) a.ids->data, si1, a.n_used, a.n_tokens, a.n_hot, (const float *) a.x->data, sx_tok, a.n_embd,
            ctx.moe_host_mb, ctx.moe_host_dstate, a.slot);
    } else {
        const int nblk = a.n_tokens*(a.n_embd/64);
        const int wpb  = MH_PUBLISH_THREADS/WARP_SIZE;
        mh_quantize_rows<<<(nblk + wpb - 1)/wpb, MH_PUBLISH_THREADS, 0, ctx.stream()>>>(
            (const float *) a.x->data, sx_tok, a.n_embd, a.n_tokens, ctx.moe_host_mb);
        mh_ring<<<1, MH_RING_THREADS, 0, ctx.stream()>>>(
            (const int32_t *) a.ids->data, si1, a.n_used, a.n_tokens, a.n_hot, ctx.moe_host_mb, ctx.moe_host_dstate, a.slot);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_moe_host_route(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const int i_first, const ggml_tensor * logits,
                              ggml_tensor * weights, ggml_tensor * ids, const float clamp_val) {
    static const bool enabled = getenv("GGML_CUDA_MOE_HOST_ROUTE") == nullptr || atoi(getenv("GGML_CUDA_MOE_HOST_ROUTE")) != 0;
    const int n_used   = (int) weights->ne[1];
    const int n_tokens = (int) logits->ne[1];
    if (!enabled || !ggml_cuda_moe_host_enabled() || ctx.moe_host.ids != nullptr || logits->ne[0] != MH_ROUTE_EXPERTS ||
        logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) || weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(weights) ||
        ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->nb[1] != MH_ROUTE_EXPERTS*sizeof(int32_t) ||
        ids->ne[0] != n_used || ids->ne[1] != n_tokens || ggml_nelements(weights) != (int64_t) n_used*n_tokens ||
        n_tokens > MH_SMALL_TOK || n_tokens*n_used > MH_PUBLISH_THREADS || n_used > WARP_SIZE) {
        return false;
    }
    // the triple that these ids route
    int j = -1;
    for (int q = i_first; q < std::min(cgraph->n_nodes, i_first + 8) && j < 0; ++q) {
        const ggml_tensor * n = cgraph->nodes[q];
        if (n->op == GGML_OP_MUL_MAT_ID && n->src[2] == ids) {
            j = q;
        }
    }
    if (j < 0 || !ggml_cuda_mmid_cold(cgraph->nodes[j])) {
        return false;
    }
    const ggml_tensor * node = cgraph->nodes[j];
    ggml_cuda_moe_host_register(ctx, cgraph, j);
    mh_open_args a;
    if (!mh_open(ctx, cgraph, j, a)) {
        return false;
    }
    GGML_ASSERT(a.ids == ids && a.n_used == n_used && a.n_tokens == n_tokens);
    const int32_t * perm   = nullptr;
    uint32_t *      counts = nullptr;
    ggml_cuda_expert_cache_take(ctx, node, &perm, &counts);
    // GGML_CUDA_FN_MOE_GROUPS=1: fn_moe_up_g reads an expert once for two tokens (opt-in: -4.5 us of the hot experts per
    // layer, but the layer waits for the host tier anyway, and the attention segments of the window measured 4-7 us
    // longer with it on, consistently - the larger kernel's code is the suspect)
    static const bool use_groups = getenv("GGML_CUDA_FN_MOE_GROUPS") != nullptr && atoi(getenv("GGML_CUDA_FN_MOE_GROUPS")) != 0;
    uint32_t * groups = use_groups && n_tokens*n_used <= WARP_SIZE ? ctx.moe_host_dstate + MH_DSTATE_PAIR + ctx.moe_host_mb->cap_pairs : nullptr;
    ctx.fn_moe_groups = groups;
    static const bool early_fence = getenv("GGML_CUDA_MOE_HOST_EARLY_FENCE") == nullptr || atoi(getenv("GGML_CUDA_MOE_HOST_EARLY_FENCE")) != 0;
    mh_route<<<1 + MH_ROUTE_QBLOCKS, MH_PUBLISH_THREADS, 0, ctx.stream()>>>(
        (const float *) logits->data, ggml_cuda_fn_router_rest(ctx, logits), (float *) weights->data, (int32_t *) ids->data,
        MH_ROUTE_EXPERTS, n_used, n_tokens, clamp_val,
        perm, counts, a.n_hot, (const float *) a.x->data, (int64_t) (a.x->nb[2]/sizeof(float)), a.n_embd,
        ctx.moe_host_mb, ctx.moe_host_dstate, a.slot, ctx.fn_dbg_get(), groups, early_fence);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_moe_host_end(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    if (ctx.moe_host.down == nullptr || node != ctx.moe_host.down) {
        return;
    }
    const int n_used   = (int) node->src[2]->ne[0];
    const int n_tokens = (int) node->src[2]->ne[1];
    mh_collect<<<n_tokens <= MH_SMALL_TOK ? 8 : 64, 256, 0, ctx.stream()>>>(
        ctx.moe_host_mb, ctx.moe_host_dstate, (float *) node->data,
        (int64_t) (node->nb[1]/sizeof(float)), (int64_t) (node->nb[2]/sizeof(float)), n_used, (int) node->ne[0], ctx.fn_dbg_get());
    CUDA_CHECK(cudaGetLastError());
    ctx.moe_host = {};
    ctx.fn_moe_groups = nullptr;
}

void ggml_cuda_moe_host_end_weighted(ggml_backend_cuda_context & ctx, const ggml_tensor * node, ggml_tensor * dst) {
    GGML_ASSERT(node == ctx.moe_host.down);
    const int n_used   = (int) node->src[2]->ne[0];
    const int n_tokens = (int) node->src[2]->ne[1];
    GGML_ASSERT(((uintptr_t) dst->data & 0xF) == 0 && dst->nb[1] % 16 == 0);
    mh_collect_weighted<<<n_tokens, 256, 0, ctx.stream()>>>(
        ctx.moe_host_mb, ctx.moe_host_dstate, ggml_cuda_fn_moe_routes(ctx), (float *) dst->data,
        (int64_t) (dst->nb[1]/sizeof(float)), n_used, (int) dst->ne[0], ctx.fn_dbg_get());
    CUDA_CHECK(cudaGetLastError());
    ctx.moe_host = {};
    ctx.fn_moe_groups = nullptr;
}

static const ggml_tensor * mh_root(const ggml_tensor * t) {
    while (t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

void ggml_cuda_moe_host_reorder(ggml_cgraph * cgraph) {
    static const bool enabled = ggml_cuda_moe_host_enabled() &&
        (getenv("GGML_CUDA_MOE_HOST_REORDER") == nullptr || atoi(getenv("GGML_CUDA_MOE_HOST_REORDER")) != 0);
    if (!enabled) {
        return;
    }
    const int n_nodes = cgraph->n_nodes;
    // a prompt graph keeps its order even where it is decode-sized (the last layer, cut down to the output rows):
    // the order would depend on the number of outputs, unlike the worst-case graph its allocation was reserved
    // with, and a mismatch re-allocates, which synchronizes every backend at every ubatch
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op == GGML_OP_MUL_MAT_ID && n->src[2]->ne[1] > MH_SMALL_TOK) {
            return;
        }
    }
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * first = cgraph->nodes[i];
        if (first->op != GGML_OP_MUL_MAT_ID || !ggml_cuda_mmid_cold(first) || first->src[2]->ne[1] > MH_SMALL_TOK) {
            continue;
        }
        const ggml_tensor * ids = first->src[2];
        int d = -1;
        for (int j = i + 1; j < std::min(n_nodes, i + 32); ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (n->op == GGML_OP_MUL_MAT_ID && n->src[2] == ids && n->src[1]->op == GGML_OP_GLU) {
                d = j;
                break;
            }
        }
        if (d < 0) {
            continue;
        }
        // nodes after the down projection computed from the triple's input and weights alone
        const ggml_tensor * x = mh_root(first->src[1]);
        std::vector<int> mv;
        std::unordered_set<const ggml_tensor *> mvset;
        for (int j = d + 1; j < std::min(n_nodes, d + 128); ++j) {
            ggml_tensor * m = cgraph->nodes[j];
            if (m->op == GGML_OP_MUL_MAT_ID) {
                break; // the next layer's MoE
            }
            switch (m->op) {
                case GGML_OP_MUL_MAT: case GGML_OP_GLU: case GGML_OP_UNARY: case GGML_OP_MUL:
                case GGML_OP_SCALE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_CONT:
                    break;
                default:
                    continue;
            }
            bool ok    = true;
            bool reads = false;
            for (int k = 0; k < GGML_MAX_SRC && ok; ++k) {
                const ggml_tensor * s = m->src[k];
                if (s == nullptr) {
                    continue;
                }
                if (mh_root(s) == x || mvset.count(s) || mvset.count(mh_root(s))) {
                    reads = true;
                } else if (s->buffer == nullptr || ggml_backend_buffer_get_usage(s->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    ok = false;
                }
            }
            if (ok && reads && !(m->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                mv.push_back(j);
                mvset.insert(m);
            }
        }
        if (mv.empty()) {
            continue;
        }
        // nodes[d .. mv.back()] -> the moved nodes, then the rest in their order
        std::vector<ggml_tensor *> seg;
        seg.reserve(mv.back() - d + 1);
        for (int j : mv) {
            seg.push_back(cgraph->nodes[j]);
        }
        for (int j = d; j <= mv.back(); ++j) {
            if (!mvset.count(cgraph->nodes[j])) {
                seg.push_back(cgraph->nodes[j]);
            }
        }
        std::copy(seg.begin(), seg.end(), cgraph->nodes + d);
        i = d + (int) mv.size();
    }
}
