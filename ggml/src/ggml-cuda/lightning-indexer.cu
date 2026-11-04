#include "common.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

#include <type_traits>

// the mask of key i_kv for a token: the F16 row M (added), or the I32 range R = {lo, hi} (0 inside, -inf outside)
static __device__ __forceinline__ float lightning_indexer_mask(const half * M, const int32_t * R, const int i_kv) {
    if (R != nullptr) {
        return i_kv >= R[0] && i_kv < R[1] ? 0.0f : -INFINITY;
    }
    return __half2float(M[i_kv]);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(TURING_MMA_AVAILABLE)

typedef union {
    int2 i2;
    half2 h2[2];
} half4;

// TODO add support for AMD cards via rocWMMA
#include <mma.h>
namespace wmma = nvcuda::wmma;

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, const int32_t * R, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEADS_PER_INNER_LOOP = 8;
    constexpr int K_EMBD_PER_INNER_LOOP = 16;
    constexpr int N_EMBD_PADDED = N_EMBD + 8;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each block processes K_VECS_PER_BLOCK K vectors
    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load weights and first Q tile to shared memory

    __shared__ float w_shared[N_HEAD];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][N_EMBD_PADDED / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }

    // total number of half4 elements in HEADS_PER_INNER_LOOP x N_EMBD Q tile
    constexpr int N_Q_TILE = HEADS_PER_INNER_LOOP * (N_EMBD / 4);
    // number of registers needed in each thread to store Q tile in thread block
    constexpr int N_Q_NEXT = (N_Q_TILE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma unroll
    for (int i_q = tid; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (N_EMBD / 4);
        const int i_embd = i_q % (N_EMBD / 4);
        const float4 q = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        half4 q_packed;
        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
        q_shared_h[i_head][i_embd] = q_packed.i2;
    }

    // phase 2 - load (and dequantize if needed) K to shared mem

    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][N_EMBD_PADDED / 4][2];

    constexpr int n_k = K_VECS_PER_BLOCK * (N_EMBD / 4);

    if constexpr (TYPE_K == GGML_TYPE_F16) {
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    }

    __syncthreads();

    // phase 3 - calculate lightning indexer scores

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    // load K fragment
    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        // we don't use accumulator for anything, fill it with zeros
        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
        wmma::fill_fragment(frag_acc, 0.0f);

        // load Q fragment
        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

        // preload next Q tile to registers during matrix multiplication
        float4 q_next[N_Q_NEXT];

        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (N_EMBD / 4);
                const int i_embd =               i_q % (N_EMBD / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
            }
        }

        // perform matrix multiplication
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        // make sure all threads finished using q_shared_h so we can store next tile
        __syncthreads();

        // write preloaded Q tile to shared memory
        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_q / (N_EMBD / 4);
                const int i_embd = i_q % (N_EMBD / 4);
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
                q_shared_h[i_head][i_embd] = q_packed.i2;
                ++i_q_next;
            }
        }

        // accumulate QK multiplication results from all block warps
        // (there are 256 threads in block and 256 matmul outputs)
        // TODO it will break if WARP_SIZE is not 32
        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += qk_shared[w][h][k];
        }

        // ReLU, weight
        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        // wait until qk_shared[0] is no longer used
        __syncthreads();

        // reuse qk_shared[0] for storing partial results
        qk_shared[0][h][k] = sum;

        // wait until all threads write their results
        __syncthreads();

        // accumulate result over heads
        if (tid < K_VECS_PER_BLOCK) {
#pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        // make sure all threads finished using qk_shared
        __syncthreads();
    }

    // phase 4 - store output to VRAM

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = start_kv + tid;
        if (i_kv < n_kv) {
            const half *    m_base = (const half *)    ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            const int32_t * r_base = R == nullptr ? nullptr : (const int32_t *) ((const char *) R + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k + lightning_indexer_mask(m_base, r_base, i_kv);
        }
    }
}

#else // defined(TURING_MMA_AVAILABLE)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, const int32_t * R, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {
    GGML_UNUSED_VARS(Q, K, W, M, dst,
        n_stream, n_batch, n_kv,
        nb1, nb2, nb3,
        nbq1, nbq2, nbq3,
        nbk1, nbk2, nbk3,
        nbw1, nbw2, nbw3,
        nem3);
    NO_DEVICE_CODE;
}

#endif // defined(TURING_MMA_AVAILABLE)
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// tokens scored per block by the tile kernel
#define LIGHTNING_INDEXER_TILE_TOKENS 8

// heads whose queries the tile kernel stages per pass, MUSA arch 21 caps static shared memory
// at 28 KB and the queries of four heads do not fit there next to the key tile
#if defined(GGML_USE_MUSA) && defined(__MUSA_ARCH__) && __MUSA_ARCH__ < 220
#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 2
#else
#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 4
#endif

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (N_EMBD)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_vec(
        const float * Q, const char * K, const float * W, const half * M, const int32_t * R, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load (and dequantize if needed) K to registers

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
        // direct copy of float4
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const float4 * k_base = (const float4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else {
        // dequantize remaining types to float
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    // load weights and Q only for N_HEAD_INNER heads at once to reduce shared memory usage
    constexpr int N_HEAD_INNER = N_HEAD / 4;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // phase 2 - load weights and Q to shared memory

        __shared__ float  w_shared[N_HEAD_INNER];
        __shared__ float4 q_shared_f[N_HEAD_INNER][N_EMBD / 4];

        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = N_HEAD_INNER * (N_EMBD / 4);
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_head_inner][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        }

        __syncthreads();

        // phase 3 - calculate lightning indexer scores

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            // dot product of floats
            const float4 q_vec = q_shared_f[i_head_inner][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                // ReLU, weight
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const half *    m_base = (const half *)    ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            const int32_t * r_base = R == nullptr ? nullptr : (const int32_t *) ((const char *) R + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid] + lightning_indexer_mask(m_base, r_base, i_kv);
        }
    }
}

// A decode window (NT tokens of up to 8) against f16 keys: a thread per key walks its key row once for every
// (token, head) with the queries as shared memory broadcasts, so no key is read twice and no dot product crosses
// threads. 4 heads of 128.
#define LIGHTNING_INDEXER_LANES_KEYS 256

template <int NT>
static __global__ void __launch_bounds__(LIGHTNING_INDEXER_LANES_KEYS, 2)
lightning_indexer_kernel_lanes(
        const float * Q, const char * K, const float * W, const half * M, const int32_t * R, float * dst,
        int64_t n_kv,
        size_t nb1, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3,
        size_t nbm1, size_t nbm3,
        int64_t nem3) {
    constexpr int NH = 4;
    constexpr int D  = 128;
    __shared__ float4 s_q[NT*NH][D/4];
    __shared__ float  s_w[NT*NH];
    const int i_stream = blockIdx.y;
    const int tid      = threadIdx.x;
    for (int e = tid; e < NT*NH*(D/4); e += blockDim.x) {
        const int th = e/(D/4);
        const int t  = th/NH;
        const int h  = th - t*NH;
        s_q[th][e - th*(D/4)] = *(const float4 *) ((const char *) Q + t*nbq2 + h*nbq1 + i_stream*nbq3 + (e - th*(D/4))*sizeof(float4));
    }
    if (tid < NT*NH) {
        const int t = tid/NH;
        s_w[tid] = *(const float *) ((const char *) W + t*nbw1 + i_stream*nbw3 + (tid - t*NH)*sizeof(float));
    }
    __syncthreads();
    const int64_t i_kv = (int64_t) blockIdx.x*blockDim.x + tid;
    if (i_kv >= n_kv) {
        return;
    }
    const uint4 * kr = (const uint4 *) (K + i_kv*nbk2 + i_stream*nbk3);
    float acc[NT*NH];
#pragma unroll
    for (int th = 0; th < NT*NH; ++th) {
        acc[th] = 0.0f;
    }
#pragma unroll 4
    for (int i = 0; i < D/8; ++i) {
        const uint4  u  = kr[i];
        const float2 k0 = __half22float2(*(const half2 *) &u.x);
        const float2 k1 = __half22float2(*(const half2 *) &u.y);
        const float2 k2 = __half22float2(*(const half2 *) &u.z);
        const float2 k3 = __half22float2(*(const half2 *) &u.w);
#pragma unroll
        for (int th = 0; th < NT*NH; ++th) {
            const float4 q0 = s_q[th][2*i];
            const float4 q1 = s_q[th][2*i + 1];
            acc[th] += k0.x*q0.x + k0.y*q0.y + k1.x*q0.z + k1.y*q0.w + k2.x*q1.x + k2.y*q1.y + k3.x*q1.z + k3.y*q1.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float score = 0.0f;
#pragma unroll
        for (int h = 0; h < NH; ++h) {
            const float v = acc[t*NH + h];
            score += (v > 0.0f ? v : 0.0f)*s_w[t*NH + h];
        }
        const half *    m_base = (const half *)    ((const char *) M + t*nbm1 + (i_stream % nem3)*nbm3);
        const int32_t * r_base = R == nullptr ? nullptr : (const int32_t *) ((const char *) R + t*nbm1 + (i_stream % nem3)*nbm3);
        *(float *) ((char *) dst + t*nb1 + i_stream*nb3 + i_kv*sizeof(float)) = score + lightning_indexer_mask(m_base, r_base, i_kv);
    }
}

// one block scores a tile of K_VECS_PER_BLOCK keys against TOKENS_PER_BLOCK tokens: the keys are
// staged in half precision and the queries of every head in float, each thread owns KEYS_PER_THREAD
// keys for one token, a warp shares its token so the query reads are broadcasts, and every key
// element is widened once for all heads, so no dot product needs a cross thread reduction
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_tile(
        const float * Q, const char * K, const float * W, const half * M, const int32_t * R, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int TOKENS_PER_BLOCK  = LIGHTNING_INDEXER_TILE_TOKENS;
    constexpr int KEY_LANES         = THREADS_PER_BLOCK / TOKENS_PER_BLOCK;
    constexpr int KEYS_PER_THREAD   = K_VECS_PER_BLOCK / KEY_LANES;
    constexpr int N_EMBD_H2         = N_EMBD / 2;
    constexpr int HEADS_PER_PASS    = N_HEAD < LIGHTNING_INDEXER_TILE_HEADS_PER_PASS ? N_HEAD : LIGHTNING_INDEXER_TILE_HEADS_PER_PASS;

    static_assert(THREADS_PER_BLOCK % TOKENS_PER_BLOCK == 0, "threads must cover the token tile");
    static_assert(K_VECS_PER_BLOCK % KEY_LANES == 0, "key lanes must cover the key tile");
    static_assert(N_HEAD % HEADS_PER_PASS == 0, "head passes must cover the heads");

    const int tid         = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int start_kv    = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_batch = blockIdx.y * TOKENS_PER_BLOCK;
    const int i_stream    = blockIdx.z;

    // the row padding keeps the keys of consecutive threads in distinct banks
    __shared__ half2 k_shared[K_VECS_PER_BLOCK][N_EMBD_H2 + 1];
    __shared__ float2 q_shared[HEADS_PER_PASS][TOKENS_PER_BLOCK][N_EMBD_H2];
    __shared__ float w_shared[N_HEAD][TOKENS_PER_BLOCK];

    // phase 1 - stage the key tile four elements at a time, rows past n_kv are zero

#pragma unroll
    for (int i = tid; i < K_VECS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
        const int r  = i / (N_EMBD / 4);
        const int c4 = i % (N_EMBD / 4);

        half2 lo = make_half2(0.0f, 0.0f);
        half2 hi = lo;
        if (start_kv + r < n_kv) {
            const char * k_row = K + (start_kv + r)*nbk2 + i_stream*nbk3;
            if constexpr (TYPE_K == GGML_TYPE_F16) {
                lo = ((const half2 *) k_row)[2*c4 + 0];
                hi = ((const half2 *) k_row)[2*c4 + 1];
            } else {
                float4 v;
                if constexpr (TYPE_K == GGML_TYPE_F32) {
                    v = ((const float4 *) k_row)[c4];
                } else {
                    constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
                    dequantize_k(k_row, &v, c4 * 4);
                }
                lo = make_half2(v.x, v.y);
                hi = make_half2(v.z, v.w);
            }
        }

        k_shared[r][2*c4 + 0] = lo;
        k_shared[r][2*c4 + 1] = hi;
    }

    // phase 2 - stage the weights of every head, tokens past n_batch are zero

    if (tid < N_HEAD * TOKENS_PER_BLOCK) {
        const int h = tid / TOKENS_PER_BLOCK;
        const int r = tid % TOKENS_PER_BLOCK;
        w_shared[h][r] = start_batch + r < n_batch ?
            ((const float *) ((const char *) W + (start_batch + r)*nbw1 + i_stream*nbw3))[h] : 0.0f;
    }

    const int kl = tid % KEY_LANES;
    const int tl = tid / KEY_LANES;

    float qk[N_HEAD][KEYS_PER_THREAD] = { { 0.0f } };

#pragma unroll
    for (int h0 = 0; h0 < N_HEAD; h0 += HEADS_PER_PASS) {
        // the previous pass is fully consumed before its queries are replaced
        if (h0 > 0) {
            __syncthreads();
        }

        // phase 3 - stage the queries of the heads of this pass, tokens past n_batch are zero

#pragma unroll
        for (int i = tid; i < HEADS_PER_PASS * TOKENS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
            const int h  = i / (TOKENS_PER_BLOCK * (N_EMBD / 4));
            const int r  = i / (N_EMBD / 4) % TOKENS_PER_BLOCK;
            const int c4 = i % (N_EMBD / 4);

            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (start_batch + r < n_batch) {
                v = *(const float4 *) ((const char *) Q + (h0 + h)*nbq1 + (start_batch + r)*nbq2 + i_stream*nbq3 + c4*sizeof(float4));
            }

            q_shared[h][r][2*c4 + 0] = make_float2(v.x, v.y);
            q_shared[h][r][2*c4 + 1] = make_float2(v.z, v.w);
        }

        __syncthreads();

        // phase 4 - float products of the widened keys for the heads of this pass

#pragma unroll 8
        for (int c = 0; c < N_EMBD_H2; ++c) {
            float2 k_val[KEYS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);
            }
#pragma unroll
            for (int h = 0; h < HEADS_PER_PASS; ++h) {
                const float2 q_val = q_shared[h][tl][c];
#pragma unroll
                for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                    qk[h0 + h][j] = fmaf(k_val[j].x, q_val.x, qk[h0 + h][j]);
                    qk[h0 + h][j] = fmaf(k_val[j].y, q_val.y, qk[h0 + h][j]);
                }
            }
        }
    }

    // phase 5 - ReLU, weight, add the mask and write, consecutive threads write consecutive keys

    float score[KEYS_PER_THREAD] = { 0.0f };

#pragma unroll
    for (int h = 0; h < N_HEAD; ++h) {
#pragma unroll
        for (int j = 0; j < KEYS_PER_THREAD; ++j) {
            score[j] += fmaxf(qk[h][j], 0.0f) * w_shared[h][tl];
        }
    }

    const int i_batch = start_batch + tl;
    if (i_batch >= n_batch) {
        return;
    }

    const half *    m_base = (const half *)    ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
    const int32_t * r_base = R == nullptr ? nullptr : (const int32_t *) ((const char *) R + i_batch*nbm1 + (i_stream%nem3)*nbm3);
    float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);

#pragma unroll
    for (int j = 0; j < KEYS_PER_THREAD; ++j) {
        const int i_kv = start_kv + kl + j*KEY_LANES;
        if (i_kv < n_kv) {
            dst_base[i_kv] = score[j] + lightning_indexer_mask(m_base, r_base, i_kv);
        }
    }
}

// The prompt-batch path of the 4-head indexer over a long context: the per-head scores of a chunk of tokens are a cuBLAS
// GEMM [n_kv x chunk] = K^T Q_h into a temp (the keys widened to F32 once per call, or an F16 GEMM: see the mode), then
// one epilogue kernel does ReLU x weight x head sum + mask. The FP32 products are the tile kernel's, summed in another
// order; the F16 modes round the queries (and in mode 2 the sums) to half.
//   GGML_CUDA_LI_GEMM        0 = off (the tile kernel), 1 = SGEMM on F32 keys (default), 2 = HGEMM (F16 in/out/accumulate),
//                            3 = F16 inputs, F32 accumulate and output
//   GGML_CUDA_LI_GEMM_CHUNK  tokens per GEMM chunk (default 256)
//   GGML_CUDA_LI_GEMM_TEMP_MB  the pool temp that bounds the keys per GEMM block (default 32: 4 x chunk x kv_block x 4 B
//                            <= 32 MiB, i.e. 8192 keys per block at 256 tokens; the widened F16 keys of a block add
//                            kv_block x 512 B). One temp of [n_kv x chunk] per head (256 MiB at 65536 pooled keys) ran the
//                            255K state out of VRAM in cudaGraphInstantiate (PP5_PROGRESS.md).
//   GGML_CUDA_LI_GEMM_MIN_KV / _MIN_BATCH  the smallest n_kv / n_batch that take this path (default 8192 / 64)
struct lightning_indexer_gemm_cfg {
    int mode;
    int chunk;
    int temp_mb;
    int min_kv;
    int min_batch;
};

static const lightning_indexer_gemm_cfg & lightning_indexer_gemm_config() {
    static const lightning_indexer_gemm_cfg cfg = [] {
        auto geti = [](const char * name, int def) {
            const char * e = getenv(name);
            return e != nullptr ? atoi(e) : def;
        };
        lightning_indexer_gemm_cfg c;
        c.mode      = geti("GGML_CUDA_LI_GEMM", 1);
        c.chunk     = std::max(8, geti("GGML_CUDA_LI_GEMM_CHUNK", 256));
        c.temp_mb   = std::max(1, geti("GGML_CUDA_LI_GEMM_TEMP_MB", 32));
        c.min_kv    = geti("GGML_CUDA_LI_GEMM_MIN_KV", 8192);
        c.min_batch = std::max(1, geti("GGML_CUDA_LI_GEMM_MIN_BATCH", 64));
        return c;
    }();
    return cfg;
}

// dst[t][s] = sum_h w[t][h] * max(0, tmp[h][t][s]) + mask(t, s) for the chunk's tokens; a thread owns 4 consecutive keys
template <typename T>
static __global__ void __launch_bounds__(256)
lightning_indexer_gemm_epilogue(
        const T * __restrict__ tmp, const float * W, const half * M, const int32_t * R, float * dst,
        const int n_kv, const int s_off, const int n_blk, const int ldt, const int64_t strideh, const int t0,
        const size_t nb1, const size_t nbw1, const size_t nbm1) {
    constexpr int NH = 4;
    const int tc = blockIdx.y;
    const int t  = t0 + tc;
    const int sb = 4*((int) blockIdx.x*blockDim.x + threadIdx.x); // key within the block
    if (sb >= n_blk) {
        return;
    }
    const int s0 = s_off + sb;                                     // absolute key
    const float * w  = (const float *) ((const char *) W + (size_t) t*nbw1);
    float wh[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        wh[h] = w[h];
    }
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        const T * row = tmp + h*strideh + (int64_t) tc*ldt + sb;
        float v[4];
        if constexpr (std::is_same<T, float>::value) {
            const float4 f = *(const float4 *) row;
            v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w;
        } else {
            const uint2  u  = *(const uint2 *) row;
            const float2 a  = __half22float2(*(const half2 *) &u.x);
            const float2 b  = __half22float2(*(const half2 *) &u.y);
            v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[j] = fmaf(fmaxf(v[j], 0.0f), wh[h], acc[j]);
        }
    }
    float * d = (float *) ((char *) dst + (size_t) t*nb1);
    if (R != nullptr) {
        const int32_t * r = (const int32_t *) ((const char *) R + (size_t) t*nbm1);
        const int lo = r[0], hi = r[1];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int s = s0 + j;
            acc[j] += s >= lo && s < hi ? 0.0f : -INFINITY;
        }
    } else {
        const half * m = (const half *) ((const char *) M + (size_t) t*nbm1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (s0 + j < n_kv) {
                acc[j] += __half2float(m[s0 + j]);
            }
        }
    }
    if (s0 + 3 < n_kv && ((uintptr_t) (d + s0) & 15) == 0) {
        *(float4 *) (d + s0) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (s0 + j < n_kv) {
                d[s0 + j] = acc[j];
            }
        }
    }
}

// the GEMM path applies: 4 heads of 128, F16 or F32 keys with contiguous rows, contiguous queries per head, a big enough batch
static bool lightning_indexer_gemm_applies(const ggml_tensor * q, const ggml_tensor * k, int n_batch, int n_kv) {
    const lightning_indexer_gemm_cfg & cfg = lightning_indexer_gemm_config();
    if (cfg.mode <= 0 || cfg.mode > 3) {
        return false;
    }
    if (q->ne[0] != 128 || q->ne[1] != 4) {
        return false;
    }
    if (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) {
        return false;
    }
    if (n_batch < cfg.min_batch || n_kv < cfg.min_kv) {
        return false;
    }
    if (k->nb[2] != 128*ggml_type_size(k->type)) {
        return false;
    }
    if (q->nb[1] != 128*sizeof(float) || q->nb[2] % sizeof(float) != 0) {
        return false;
    }
    if (cfg.mode != 1 && (q->nb[2] != 4*q->nb[1] || k->type != GGML_TYPE_F16)) {
        // the F16 modes take F16 keys and convert the queries as one contiguous [128 x 4 x n_batch] block
        return false;
    }
    return true;
}

static void ggml_cuda_lightning_indexer_gemm(ggml_backend_cuda_context & ctx,
        const float * Q, const char * K, const float * W, const half * M, const int32_t * R, float * dst,
        const ggml_type type_K, const int n_stream, const int n_batch, const int n_kv,
        const size_t nb1, const size_t nb3,
        const size_t nbq1, const size_t nbq2, const size_t nbq3,
        const size_t nbk3,
        const size_t nbw1, const size_t nbw3,
        const size_t nbm1, const size_t nbm3,
        const int64_t nem3) {
    const lightning_indexer_gemm_cfg & cfg = lightning_indexer_gemm_config();
    cudaStream_t stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    constexpr int NH = 4;
    constexpr int D  = 128;

    const int chunk = std::min(cfg.chunk, n_batch);
    // keys per GEMM block: the temp of 4 heads x chunk tokens x block keys stays within GGML_CUDA_LI_GEMM_TEMP_MB
    const size_t elt = cfg.mode == 2 ? sizeof(half) : sizeof(float);
    int kv_block = (int) std::min<int64_t>(n_kv, ((int64_t) cfg.temp_mb*1048576)/((int64_t) NH*chunk*elt));
    kv_block = std::max(256, kv_block & ~255);
    kv_block = std::min(kv_block, (n_kv + 3) & ~3);
    const int ldt   = (kv_block + 3) & ~3; // 16-byte aligned temp rows
    const int64_t strideh = (int64_t) chunk*ldt;

    // the temp: F32 for modes 1 and 3, F16 for mode 2
    ggml_cuda_pool_alloc<float> tmp_f32(ctx.pool());
    ggml_cuda_pool_alloc<half>  tmp_f16(ctx.pool());
    ggml_cuda_pool_alloc<float> k_f32(ctx.pool());
    ggml_cuda_pool_alloc<half>  q_f16(ctx.pool());
    if (cfg.mode == 2) {
        tmp_f16.alloc(NH*strideh);
    } else {
        tmp_f32.alloc(NH*strideh);
    }
    if (cfg.mode == 1 && type_K == GGML_TYPE_F16) {
        k_f32.alloc((size_t) kv_block*D);
    }
    if (cfg.mode != 1) {
        q_f16.alloc((size_t) n_batch*NH*D);
    }

    const float one_f = 1.0f, zero_f = 0.0f;
    const half  one_h = __float2half(1.0f), zero_h = __float2half(0.0f);

    for (int i_stream = 0; i_stream < n_stream; ++i_stream) {
        const char  * k_s = K + (size_t) i_stream*nbk3;
        const float * q_s = (const float *) ((const char *) Q + (size_t) i_stream*nbq3);
        const float * w_s = (const float *) ((const char *) W + (size_t) i_stream*nbw3);
        const half    * m_s = M == nullptr ? nullptr : (const half    *) ((const char *) M + (size_t) (i_stream % nem3)*nbm3);
        const int32_t * r_s = R == nullptr ? nullptr : (const int32_t *) ((const char *) R + (size_t) (i_stream % nem3)*nbm3);
        float * dst_s = (float *) ((char *) dst + (size_t) i_stream*nb3);

        if (cfg.mode != 1) {
            ggml_get_to_fp16_cuda(GGML_TYPE_F32)(q_s, q_f16.get(), (int64_t) n_batch*NH*D, stream);
        }

        for (int s_off = 0; s_off < n_kv; s_off += kv_block) {
        const int n_blk = std::min(kv_block, n_kv - s_off);
        // the A operand: [D x n_blk] column-major keys of the block, op(A) = A^T
        const void * a_ptr = k_s + (size_t) s_off*D*ggml_type_size(type_K);
        if (cfg.mode == 1 && type_K == GGML_TYPE_F16) {
            ggml_get_to_fp32_cuda(GGML_TYPE_F16)(a_ptr, k_f32.get(), (int64_t) n_blk*D, stream);
            a_ptr = k_f32.get();
        }

        for (int t0 = 0; t0 < n_batch; t0 += chunk) {
            const int nt = std::min(chunk, n_batch - t0);
            for (int h = 0; h < NH; ++h) {
                if (cfg.mode == 1) {
                    // B = Q_h: [D x nt] column-major with the token stride, C = [n_blk x nt]
                    const float * b_ptr = (const float *) ((const char *) q_s + (size_t) t0*nbq2 + (size_t) h*nbq1);
                    CUBLAS_CHECK(cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                            n_blk, nt, D,
                            &one_f, (const float *) a_ptr, D,
                                    b_ptr, (int) (nbq2/sizeof(float)),
                            &zero_f, tmp_f32.get() + h*strideh, ldt));
                } else {
                    const half * b_ptr = q_f16.get() + (size_t) t0*NH*D + (size_t) h*D;
                    if (cfg.mode == 2) {
                        CUBLAS_CHECK(cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                                n_blk, nt, D,
                                &one_h, a_ptr, CUDA_R_16F, D,
                                        b_ptr, CUDA_R_16F, NH*D,
                                &zero_h, tmp_f16.get() + h*strideh, CUDA_R_16F, ldt,
                                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
                    } else {
                        CUBLAS_CHECK(cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                                n_blk, nt, D,
                                &one_f, a_ptr, CUDA_R_16F, D,
                                        b_ptr, CUDA_R_16F, NH*D,
                                &zero_f, tmp_f32.get() + h*strideh, CUDA_R_32F, ldt,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    }
                }
            }
            const dim3 block(256, 1, 1);
            const dim3 grid((unsigned) ((n_blk + 4*256 - 1)/(4*256)), (unsigned) nt, 1);
            if (cfg.mode == 2) {
                lightning_indexer_gemm_epilogue<half><<<grid, block, 0, stream>>>(
                    tmp_f16.get(), w_s, m_s, r_s, dst_s, n_kv, s_off, n_blk, ldt, strideh, t0, nb1, nbw1, nbm1);
            } else {
                lightning_indexer_gemm_epilogue<float><<<grid, block, 0, stream>>>(
                    tmp_f32.get(), w_s, m_s, r_s, dst_s, n_kv, s_off, n_blk, ldt, strideh, t0, nb1, nbw1, nbm1);
            }
        }
        }
    }
}

#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)         \
    if (K->type == (type_K)) {                                                              \
        lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K> \
            <<<grid, block, 0, ctx.stream()>>>(                                             \
            q_d, k_d, w_d, m_d, r_d, dst_d,                                                 \
            n_stream, n_batch, n_kv,                                                        \
            nb1, nb2, nb3,                                                                  \
            nbq1, nbq2, nbq3,                                                               \
            nbk1, nbk2, nbk3,                                                               \
            nbw1, nbw2, nbw3,                                                               \
            nbm1, nbm2, nbm3,                                                               \
            nem3                                                                            \
        );                                                                                  \
    } else

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16 || m->type == GGML_TYPE_I32);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];

    const float *   q_d = (const float *)   q->data;
    const char  *   k_d = (const char  *)   k->data;
    const float *   w_d = (const float *)   w->data;
    // F16 rows, or I32 {lo, hi} ranges per token (see ggml_lightning_indexer)
    const half    *   m_d = m->type == GGML_TYPE_F16 ? (const half    *) m->data : nullptr;
    const int32_t *   r_d = m->type == GGML_TYPE_I32 ? (const int32_t *) m->data : nullptr;
    float         * dst_d = (float *) dst->data;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;

    if (n_embd == 128 && n_head == 64) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 32) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 4 && n_batch <= 8 && k->type == GGML_TYPE_F16 && nbk2 % 16 == 0 && nbq1 % 16 == 0 && ((uintptr_t) k_d & 15) == 0 && ((uintptr_t) q_d & 15) == 0) {
        // a decode window: a thread per key, the keys read once for every token
        const dim3 block(LIGHTNING_INDEXER_LANES_KEYS, 1, 1);
        const dim3 grid((unsigned) ((n_kv + LIGHTNING_INDEXER_LANES_KEYS - 1)/LIGHTNING_INDEXER_LANES_KEYS), (unsigned) n_stream, 1);
#define LIGHTNING_INDEXER_LANES_CASE(nt)                                                        \
        case nt:                                                                                 \
            lightning_indexer_kernel_lanes<nt><<<grid, block, 0, ctx.stream()>>>(                \
                q_d, k_d, w_d, m_d, r_d, dst_d, n_kv, nb1, nb3, nbq1, nbq2, nbq3, nbk2, nbk3,    \
                nbw1, nbw3, nbm1, nbm3, nem3);                                                   \
            break;
        switch (n_batch) {
            LIGHTNING_INDEXER_LANES_CASE(1)
            LIGHTNING_INDEXER_LANES_CASE(2)
            LIGHTNING_INDEXER_LANES_CASE(3)
            LIGHTNING_INDEXER_LANES_CASE(4)
            LIGHTNING_INDEXER_LANES_CASE(5)
            LIGHTNING_INDEXER_LANES_CASE(6)
            LIGHTNING_INDEXER_LANES_CASE(7)
            LIGHTNING_INDEXER_LANES_CASE(8)
            default: GGML_ABORT("fatal error");
        }
#undef LIGHTNING_INDEXER_LANES_CASE
    } else if (n_embd == 128 && n_head == 4 && lightning_indexer_gemm_applies(q, k, n_batch, n_kv)) {
        // a prompt batch over a long context: cuBLAS GEMM per head + fused epilogue
        ggml_cuda_lightning_indexer_gemm(ctx, q_d, k_d, w_d, m_d, r_d, dst_d, k->type, n_stream, n_batch, n_kv,
            nb1, nb3, nbq1, nbq2, nbq3, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
    } else if (n_embd == 128 && n_head == 4 && n_batch >= LIGHTNING_INDEXER_TILE_TOKENS) {
        // too few heads for a wmma tile, the tile kernel shares the keys across the tokens
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = 64;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        int num_batch_blocks = (n_batch + LIGHTNING_INDEXER_TILE_TOKENS - 1) / LIGHTNING_INDEXER_TILE_TOKENS;
        dim3 grid(num_kv_blocks, num_batch_blocks, n_stream);

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    } else if (n_embd == 128 && n_head == 4) {
        // a batch smaller than a token tile, use vector kernel
        constexpr int K_VECS_PER_WARP = 8;
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        dim3 grid(num_kv_blocks, n_batch, n_stream);

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    } else {
        GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    if (neq0 != 128) {
        return false;
    }

    if (neq1 != 64 && neq1 != 32 && neq1 != 4) {
        return false;
    }

    // alignment checks
    for (const ggml_tensor * t : {q, k}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch(k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}
