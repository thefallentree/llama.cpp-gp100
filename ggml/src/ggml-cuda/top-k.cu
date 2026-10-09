#include "argsort.cuh"
#include "top-k.cuh"

#include <climits>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

// the grid-over-rows radix select (upstream's HIP path), for the indexer's few long rows, see ggml_cuda_op_top_k
static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

// Partial selection for small k.  Where cub::DeviceTopK (CCCL >= 3.2) is unavailable, the
// caller below falls back to fully sorting the row and taking the first k -- but obtaining k
// elements does not require a sort.
//
// Comparisons use the same monotonic uint32 mapping cub uses, not float, and break ties toward
// the smaller index (cub's radix sort is stable, so the original order -- ascending index -- is
// preserved).  That reproduces the radix sort's ordering down to signed zeros and NaNs, so the
// replacement is bit-identical.
#define CUDA_TOP_K_BLOCK      256
#define CUDA_TOP_K_WARPS      (CUDA_TOP_K_BLOCK/32)
#define CUDA_TOP_K_MAX_K      16
#define CUDA_TOP_K_MAX_BLOCKS 128
#define CUDA_TOP_K_MIN_NCOLS  4096
#define CUDA_TOP_K_MAX_ELEMS  16
// Candidates held by one lane during the in-block merge (= ceil(WARPS*MAX_K / 32))
#define CUDA_TOP_K_MERGE      ((CUDA_TOP_K_WARPS*CUDA_TOP_K_MAX_K + 31)/32)

// Map a float to an order-preserving unsigned integer.  This is the mapping cub's radix sort uses
// internally, so signed zeros and NaNs order the same way.
static __device__ __forceinline__ uint32_t top_k_key(const float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Pack the key into the high 32 bits and ~idx into the low 32.  A plain unsigned comparison then
// yields "key descending, index ascending" -- the same order as cub's stable radix sort -- and
// every comparison branch disappears.  For idx >= 0 the top bit of ~idx is always set, so a
// packed value of 0 can only mean "no candidate".
// Measured 2x faster than shuffling two 32-bit values and comparing with branches
// (stage 1: 22.5 -> 11.3 us).
static __device__ __forceinline__ uint64_t top_k_pack(const uint32_t k32, const int i32) {
    return ((uint64_t) k32 << 32) | (uint32_t) ~(uint32_t) i32;
}

// Broadcast the warp maximum to every lane.  Avoiding __syncthreads is the point: doing the same
// thing with a block-wide tree reduction makes the k rounds of barriers the bottleneck
// (70.9 us against 20.6 us at k = 10).
static __device__ __forceinline__ void top_k_warp_max(uint64_t & p) {
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        const uint64_t o = __shfl_xor_sync(0xffffffffu, p, s);
        p = o > p ? o : p;
    }
}

// Stage 1: each warp produces the top k of its own share, and warp 0 merges them within the block.
// The share stays in registers, so the k rounds never re-read memory.
template <int ELEMS>
static __global__ void __launch_bounds__(CUDA_TOP_K_BLOCK)
    k_top_k_stage1(const float * __restrict__ x, uint64_t * __restrict__ out,
                   const int ncols, const int k) {
    __shared__ uint64_t sp[CUDA_TOP_K_WARPS*CUDA_TOP_K_MAX_K];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const float * row   = x + (int64_t) blockIdx.y*ncols;
    const int     wbase = blockIdx.x*(CUDA_TOP_K_BLOCK*ELEMS) + warp*(32*ELEMS);

    uint64_t p[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        const int i = wbase + e*32 + lane;
        p[e] = i < ncols ? top_k_pack(top_k_key(row[i]), i) : 0ull;
    }

    for (int r = 0; r < k; ++r) {
        uint64_t b = 0;
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) { b = p[e] > b ? p[e] : b; }
        top_k_warp_max(b);
        if (lane == 0) {
            sp[warp*k + r] = b;
        }
        // Remove the selected element from this lane's share (indices are unique within a row, so
        // at most one matches)
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) { if (p[e] == b) { p[e] = 0ull; } }
    }
    __syncthreads();

    if (warp != 0) {
        return;
    }
    const int M = CUDA_TOP_K_WARPS*k;
    uint64_t c[CUDA_TOP_K_MERGE];
#pragma unroll
    for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) {
        const int i = e*32 + lane;
        c[e] = i < M ? sp[i] : 0ull;
    }
    uint64_t * o = out + ((int64_t) blockIdx.y*gridDim.x + blockIdx.x)*k;
    for (int r = 0; r < k; ++r) {
        uint64_t b = 0;
#pragma unroll
        for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) { b = c[e] > b ? c[e] : b; }
        top_k_warp_max(b);
        if (lane == 0) {
            o[r] = b;
        }
#pragma unroll
        for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) { if (c[e] == b) { c[e] = 0ull; } }
    }
}

// Stage 2: reduce stage 1's candidates (blocks x k of them) to the final top k.
// The candidates have distinct indices within a row, since stage 1's blocks partition the row.
template <int ELEMS>
static __global__ void __launch_bounds__(CUDA_TOP_K_BLOCK)
    k_top_k_stage2(const uint64_t * __restrict__ in, int * __restrict__ dst,
                   const int ncand, const int k) {
    __shared__ uint64_t sp[CUDA_TOP_K_WARPS*CUDA_TOP_K_MAX_K];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const uint64_t * cp = in + (int64_t) blockIdx.y*ncand;

    uint64_t p[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        const int i = warp*(32*ELEMS) + e*32 + lane;
        p[e] = i < ncand ? cp[i] : 0ull;
    }

    for (int r = 0; r < k; ++r) {
        uint64_t b = 0;
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) { b = p[e] > b ? p[e] : b; }
        top_k_warp_max(b);
        if (lane == 0) {
            sp[warp*k + r] = b;
        }
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) { if (p[e] == b) { p[e] = 0ull; } }
    }
    __syncthreads();

    if (warp != 0) {
        return;
    }
    const int M = CUDA_TOP_K_WARPS*k;
    uint64_t c[CUDA_TOP_K_MERGE];
#pragma unroll
    for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) {
        const int i = e*32 + lane;
        c[e] = i < M ? sp[i] : 0ull;
    }
    int * out = dst + (int64_t) blockIdx.y*k;
    for (int r = 0; r < k; ++r) {
        uint64_t b = 0;
#pragma unroll
        for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) { b = c[e] > b ? c[e] : b; }
        top_k_warp_max(b);
        if (lane == 0) {
            out[r] = (int) ~(uint32_t) b;
        }
#pragma unroll
        for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) { if (c[e] == b) { c[e] = 0ull; } }
    }
}

static bool top_k_partial_cuda(ggml_cuda_pool & pool, const float * src, int * dst,
                               const int ncols, const int nrows, const int k, cudaStream_t stream) {
    if (k < 1 || k > CUDA_TOP_K_MAX_K || ncols < CUDA_TOP_K_MIN_NCOLS || nrows < 1) {
        return false;
    }

    // Give each thread a larger share to keep the block count down: candidates = blocks x k, so
    // too many blocks makes stage 2 expensive
    int elems = 1;
    while (elems < CUDA_TOP_K_MAX_ELEMS &&
           (int64_t) ncols > (int64_t) CUDA_TOP_K_BLOCK*elems*CUDA_TOP_K_MAX_BLOCKS) {
        elems *= 2;
    }
    if ((int64_t) ncols > (int64_t) CUDA_TOP_K_BLOCK*elems*CUDA_TOP_K_MAX_BLOCKS) {
        return false;   // leave anything larger to the existing sort
    }

    const int stripe = CUDA_TOP_K_BLOCK*elems;
    const int nb     = (ncols + stripe - 1) / stripe;
    const int ncand  = nb*k;

    int elems2 = 1;
    while (elems2 < CUDA_TOP_K_MAX_ELEMS && ncand > CUDA_TOP_K_BLOCK*elems2) {
        elems2 *= 2;
    }
    if (ncand > CUDA_TOP_K_BLOCK*elems2) {
        return false;
    }

    ggml_cuda_pool_alloc<uint64_t> cand_alloc(pool, (size_t) ncand*nrows);

    const dim3 g1(nb, nrows, 1);
    const dim3 g2(1,  nrows, 1);

#define CUDA_TOP_K_LAUNCH1(E)                                                                    \
    case E: k_top_k_stage1<E><<<g1, CUDA_TOP_K_BLOCK, 0, stream>>>(                              \
                src, cand_alloc.get(), ncols, k); break
    switch (elems) {
        CUDA_TOP_K_LAUNCH1(1);
        CUDA_TOP_K_LAUNCH1(2);
        CUDA_TOP_K_LAUNCH1(4);
        CUDA_TOP_K_LAUNCH1(8);
        CUDA_TOP_K_LAUNCH1(16);
        default: return false;
    }
#undef CUDA_TOP_K_LAUNCH1

#define CUDA_TOP_K_LAUNCH2(E)                                                                    \
    case E: k_top_k_stage2<E><<<g2, CUDA_TOP_K_BLOCK, 0, stream>>>(                              \
                cand_alloc.get(), dst, ncand, k); break
    switch (elems2) {
        CUDA_TOP_K_LAUNCH2(1);
        CUDA_TOP_K_LAUNCH2(2);
        CUDA_TOP_K_LAUNCH2(4);
        CUDA_TOP_K_LAUNCH2(8);
        CUDA_TOP_K_LAUNCH2(16);
        default: return false;
    }
#undef CUDA_TOP_K_LAUNCH2

    return true;
}

// The k largest of each row, in column order, for a k too large for top_k_partial_cuda, where the fallback sorts the whole row.
// One block per row. A thread owns PART columns in groups of 4: group j of thread t is the columns 4*(j*blockDim + t)
// .. + 3, so the loads of a warp are consecutive; the first REG keys of a thread stay in registers, the rest are read
// again from the row (L2) at every pass.
// A radix select over the 32 key bits, a byte per pass, finds the k-th largest key: each pass histograms the next
// byte of the keys that share the prefix found so far, one histogram per warp, and takes the byte at which the
// count from the top reaches the rank. (A bit-by-bit search, 32 counts over the row, took 1070 us at 64K columns.)
// Of the columns equal to the k-th key the first ones are taken, as a stable sort does: the output walks the groups
// in column order with a prefix count per group.
#define CUDA_TOP_K_SEL_WARPS 32

template <int PART>
static __global__ void __launch_bounds__(CUDA_TOP_K_SEL_WARPS*WARP_SIZE)
k_top_k_select(const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k) {
    __shared__ int      s_hist[CUDA_TOP_K_SEL_WARPS][256];
    __shared__ int      s_sum[256];
    __shared__ int      s_gt[CUDA_TOP_K_SEL_WARPS];
    __shared__ int      s_eq[CUDA_TOP_K_SEL_WARPS];
    __shared__ int      s_base[2]; // selected columns, and columns equal to the threshold, before the group
    __shared__ uint32_t s_thr;
    __shared__ int      s_above;
    constexpr int NQ  = PART/4;                 // groups of 4 columns per thread
    constexpr int REG = PART < 32 ? PART : 32;  // keys in registers
    constexpr int NQR = REG/4;                  // their groups

    const float * row  = src + (size_t) blockIdx.x*ncols;
    int *         out  = dst + (size_t) blockIdx.x*k;
    const int     lane = threadIdx.x % WARP_SIZE;
    const int     warp = threadIdx.x / WARP_SIZE;
    const int     nw   = blockDim.x / WARP_SIZE;
    const int     nt   = blockDim.x;

    // the keys of group j; a column past the row gets key 0, no candidate is that low
    auto load = [&](const int j, uint32_t (&kk)[4]) {
        const int c = 4*(j*nt + (int) threadIdx.x);
        if (c + 3 < ncols) {
            kk[0] = top_k_key(row[c]);
            kk[1] = top_k_key(row[c + 1]);
            kk[2] = top_k_key(row[c + 2]);
            kk[3] = top_k_key(row[c + 3]);
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                kk[e] = c + e < ncols ? top_k_key(row[c + e]) : 0u;
            }
        }
    };
    uint32_t key[REG];
#pragma unroll
    for (int j = 0; j < NQR; ++j) {
        uint32_t kk[4];
        load(j, kk);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            key[4*j + e] = kk[e];
        }
    }

    uint32_t thr     = 0; // the prefix found so far, in the high bits
    int      k_above = 0; // keys above every key of the prefix
    for (int pass = 0; pass < 4; ++pass) {
        const int      shift = 24 - 8*pass;
        const uint32_t pmask = pass == 0 ? 0u : ~0u << (shift + 8);
        for (int b = lane; b < 256; b += WARP_SIZE) {
            s_hist[warp][b] = 0;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < REG; ++i) {
            if ((key[i] & pmask) == thr) {
                atomicAdd(&s_hist[warp][(key[i] >> shift) & 0xFF], 1);
            }
        }
#pragma unroll 4
        for (int j = NQR; j < NQ; ++j) {
            uint32_t kk[4];
            load(j, kk);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                if ((kk[e] & pmask) == thr) {
                    atomicAdd(&s_hist[warp][(kk[e] >> shift) & 0xFF], 1);
                }
            }
        }
        __syncthreads();
        // the bins over the warps, then the count from the top: the first bin at which it reaches the rank
        for (int b = threadIdx.x; b < 256; b += blockDim.x) {
            int n = 0;
            for (int w = 0; w < nw; ++w) {
                n += s_hist[w][b];
            }
            s_sum[b] = n;
        }
        __syncthreads();
        if (warp == 0) {
            const int rank  = k - k_above;
            int       above = 0; // keys in the bins above the chunk
            int       found = -1;
            // a descending prefix over the 256 bins: 8 chunks of 32 from the top, lane 0 at the highest bin
            for (int ch = 7; ch >= 0 && found < 0; --ch) {
                const int b    = 32*ch + (WARP_SIZE - 1 - lane);
                const int n    = s_sum[b];
                const int incl = warp_prefix_inclusive_sum(n);   // keys in this bin and the bins above it within the chunk
                const int tot  = __shfl_sync(0xffffffff, incl, WARP_SIZE - 1);
                const bool hit = above + incl >= rank && above + incl - n < rank;
                const unsigned m = __ballot_sync(0xffffffff, hit);
                if (m != 0) {
                    const int l = __ffs(m) - 1;
                    found = 32*ch + (WARP_SIZE - 1 - l);
                    const int before = __shfl_sync(0xffffffff, above + incl - n, l);
                    if (lane == 0) {
                        s_thr   = thr | ((uint32_t) found << shift);
                        s_above = k_above + before;
                    }
                }
                above += tot;
            }
        }
        __syncthreads();
        thr     = s_thr;
        k_above = s_above;
    }
    // thr is the k-th largest key; k_above keys are larger, k - k_above of the keys equal to it are taken, the first
    // ones in column order: the groups in column order, a prefix count of the selected and the equal keys per group
    const int need = k - k_above;
    if (threadIdx.x == 0) {
        s_base[0] = 0;
        s_base[1] = 0;
    }
    __syncthreads();
    // the register groups unrolled (a runtime index into key[] would put it in local memory), then the rest
#pragma unroll
    for (int j = 0; j < NQ; ++j) {
        uint32_t kk[4];
        if (j < NQR) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                kk[e] = key[(j < NQR ? j : 0)*4 + e];
            }
        } else {
            load(j, kk);
        }
        const int c = 4*(j*nt + (int) threadIdx.x);
        int n_gt = 0;
        int n_eq = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            n_gt += kk[e] > thr;
            n_eq += kk[e] == thr && c + e < ncols;
        }
        const int gi = warp_prefix_inclusive_sum(n_gt);
        const int ei = warp_prefix_inclusive_sum(n_eq);
        if (lane == WARP_SIZE - 1) {
            s_gt[warp] = gi;
            s_eq[warp] = ei;
        }
        __syncthreads();
        // warp totals -> what the warps before hold, and the group's totals
        int gt0 = s_base[0];
        int eq0 = s_base[1];
        for (int w = 0; w < warp; ++w) {
            gt0 += s_gt[w];
            eq0 += s_eq[w];
        }
        int eq_before = eq0 + ei - n_eq;
        int pos       = gt0 + gi - n_gt + min(eq_before, need);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            if (kk[e] > thr) {
                out[pos++] = c + e;
            } else if (kk[e] == thr && c + e < ncols) {
                if (eq_before < need) {
                    out[pos++] = c + e;
                }
                eq_before++;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int g = 0, e = 0;
            for (int w = 0; w < nw; ++w) {
                g += s_gt[w];
                e += s_eq[w];
            }
            s_base[0] += g;
            s_base[1] += e;
        }
        __syncthreads();
    }
}

static bool top_k_select_cuda(const float * src, int * dst, const int ncols, const int nrows, const int k, cudaStream_t stream) {
    static const bool enabled = getenv("GGML_CUDA_TOP_K_SELECT") == nullptr || atoi(getenv("GGML_CUDA_TOP_K_SELECT")) != 0;
    const int max_threads = CUDA_TOP_K_SEL_WARPS*WARP_SIZE;
    // the keys of a thread stay in registers, a block of 1024 threads has 63 per thread; past 32 per thread (rows of
    // more than 32K columns, a 128K context) the rest are read from the row at every pass
    // k == ncols stays with the sort: its order is the order the consumer sums in, and nothing is saved there
    if (!enabled || k <= CUDA_TOP_K_MAX_K || k >= ncols || nrows < 1 || ncols > 64*max_threads) {
        return false;
    }
    // short rows over more threads: the passes have a fixed cost in zeroing, summing and scanning the bins
    const int part    = ncols <= 4*max_threads ? 4 : ncols <= 8*max_threads ? 8 : ncols <= 16*max_threads ? 16 : ncols <= 32*max_threads ? 32 : 64;
    const int threads = GGML_PAD((ncols + part - 1)/part, WARP_SIZE);
    if (part == 4) {
        k_top_k_select<4><<<nrows, threads, 0, stream>>>(src, dst, ncols, k);
    } else if (part == 8) {
        k_top_k_select<8><<<nrows, threads, 0, stream>>>(src, dst, ncols, k);
    } else if (part == 16) {
        k_top_k_select<16><<<nrows, threads, 0, stream>>>(src, dst, ncols, k);
    } else if (part == 32) {
        k_top_k_select<32><<<nrows, threads, 0, stream>>>(src, dst, ncols, k);
    } else {
        k_top_k_select<64><<<nrows, threads, 0, stream>>>(src, dst, ncols, k);
    }
    return true;
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifndef CUB_TOP_K_AVAILABLE
    // without cub::DeviceTopK, selecting a few elements out of a wide row is much cheaper than sorting it:
    // 3.7 ms -> 0.09 ms for 4 rows x 248k on a P100 (k = 16)
    if (ncols <= INT_MAX && nrows <= INT_MAX && k <= INT_MAX &&
        top_k_partial_cuda(pool, src0_d, dst_d, (int) ncols, (int) nrows, (int) k, stream)) {
        return;
    }
    // a large k over long rows: the grid-over-rows radix select (a dozen launches, up to 64 blocks per row) from
    // GGML_CUDA_TOP_K_GRID columns (32K; 0 never), the one-block-per-row select below it: measured on a P100 for 3
    // rows and k = 512, 64 / 77 us (grid) against 49 / 173 us (one block) at 16K / 64K columns
    static const int64_t grid_min = getenv("GGML_CUDA_TOP_K_GRID") != nullptr ? atoll(getenv("GGML_CUDA_TOP_K_GRID")) : 32768;
    if (grid_min > 0 && ncols >= grid_min && ncols <= INT_MAX && nrows <= INT_MAX && k <= INT_MAX && k > CUDA_TOP_K_MAX_K && k < ncols) {
        top_k_radix_cuda(pool, src0_d, dst_d, (int) ncols, (int) nrows, (int) k, stream);
        return;
    }
    if (ncols <= INT_MAX && nrows <= INT_MAX && k <= INT_MAX &&
        top_k_select_cuda(src0_d, dst_d, (int) ncols, (int) nrows, (int) k, stream)) {
        return;
    }
#endif // CUB_TOP_K_AVAILABLE
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
