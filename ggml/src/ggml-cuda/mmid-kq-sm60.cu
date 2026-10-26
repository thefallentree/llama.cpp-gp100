// Q2_K / Q3_K MUL_MAT_ID mat-vec for decode batches on sm_60 (GP100).
//
// GP100 has no DP4A, so the generic mmvq kernels emulate every int8 dot product. Here the weights stay in their 2-bit fields:
// a mask turns a field into a half subnormal, HFMA2 multiplies it with the activation (converted once per op to half pairs,
// in the order of the decode) and the sub-block scales are applied in half2 once per 16 weights. The math of one lane is in
// mmid-kq-sm60-math.cuh, which a CPU test also compiles.
//
// A warp handles 8 rows, 4 lanes per row: lane l reads the 16-byte part (l & 3) of every super-block of its row, so the
// weight bytes are read once. One blockIdx.y per (token, expert) slot, as in the Q2_0 mat-vec (mmid-f16-sm60.cu).

#include "mmid-kq-sm60.cuh"
#include "mmid-f16-sm60.cuh"
#include "mmvq.cuh"

#include "mmid-kq-sm60-math.cuh"

#include <atomic>

// one src1 row (i11, i12) per block, in src1 order; a warp per super-block (up to 32 warps)
static __global__ void __launch_bounds__(1024) mmid_kq_convert_rows(
        const float * __restrict__ src1, char * __restrict__ dst,
        const int64_t s11, const int64_t s12, const int ne11, const int nsb) {
#if defined(FP16_AVAILABLE)
    const int     r    = blockIdx.x;
    const float * x    = src1 + (int64_t) (r / ne11)*s12 + (int64_t) (r % ne11)*s11;
    const int     lane = threadIdx.x % WARP_SIZE;
    char *        rec  = dst + (int64_t) r*nsb*KQ_REC_BYTES;
    for (int sb = threadIdx.x/WARP_SIZE; sb < nsb; sb += blockDim.x/WARP_SIZE) {
        const float4 v0 = *((const float4 *) (x + sb*256 + 4*lane));
        const float4 v1 = *((const float4 *) (x + sb*256 + 128 + 4*lane));
        float amax = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        }
        // a block of (near) zeros keeps the scale 1: 1/amax would overflow, and what it holds cannot show in the dot product
        const float s   = amax > 1e-30f ? amax : 1.0f;
        const float inv = 1.0f/s;
        ((uint2 *) rec)[kq_conv_xh_index(sb, 0, lane)] = kq_conv_pairs(v0, inv, lane);
        ((uint2 *) rec)[kq_conv_xh_index(sb, 1, lane)] = kq_conv_pairs(v1, inv, lane);
        // the sum of the 16 values of a sub-block: its 4 lanes are consecutive
        float p0 = (v0.x + v0.y + v0.z + v0.w)*inv;
        float p1 = (v1.x + v1.y + v1.z + v1.w)*inv;
        p0 += __shfl_xor_sync(0xffffffff, p0, 1);
        p1 += __shfl_xor_sync(0xffffffff, p1, 1);
        p0 += __shfl_xor_sync(0xffffffff, p0, 2);
        p1 += __shfl_xor_sync(0xffffffff, p1, 2);
        if ((lane & 3) == 0) {
            ((__half *) rec)[kq_conv_xx_index(sb, 0, lane)] = __float2half_rn(p0);
            ((__half *) rec)[kq_conv_xx_index(sb, 1, lane)] = __float2half_rn(p1);
        }
        if (lane < 4) {
            ((float *) rec)[kq_conv_xs_index(sb, lane)] = s;
        }
    }
#else
    GGML_UNUSED_VARS(src1, dst, s11, s12, ne11, nsb);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// A pair c = (token t, used expert s) per blockIdx.y. A pair whose expert is not in w (e >= n_hot) is skipped.
template <ggml_type type>
static __global__ void __launch_bounds__(WARP_SIZE*KQ_NWARPS, 8)
mmid_kq_vec(
        const char * __restrict__ w, const char * __restrict__ x,
        const int32_t * __restrict__ ids, float * __restrict__ dst, const int nsb, const int ne01, const int64_t nb01, const int64_t nb02,
        const int ne11, const int n_used, const int si1, const int n_hot, const int64_t dst_nb1, const int64_t dst_nb2) {
#if defined(FP16_AVAILABLE)
    float sum;
    int t, s, row0;
    if (!kq_vec_lane<type == GGML_TYPE_Q3_K>(w, x, ids, blockIdx.x, blockIdx.y, threadIdx.y, threadIdx.x, nsb, ne01, nb01, nb02,
            ne11, n_used, si1, n_hot, sum, t, s, row0)) {
        return;
    }
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    if ((threadIdx.x & 3) == 0 && row0 < ne01) {
        *((float *) ((char *) dst + (int64_t) t*dst_nb2 + (int64_t) s*dst_nb1) + row0) = sum;
    }
#else
    GGML_UNUSED_VARS(w, x, ids, dst, nsb, ne01, nb01, nb02, ne11, n_used, si1, n_hot, dst_nb1, dst_nb2);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

bool ggml_cuda_mmid_vec_kq_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_CUDA_MMID_KQ_SM60") == nullptr || atoi(getenv("GGML_CUDA_MMID_KQ_SM60")) != 0;
    if (!enabled || ids == nullptr) {
        return false;
    }
    if ((src0->type != GGML_TYPE_Q2_K && src0->type != GGML_TYPE_Q3_K) || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_PASCAL || cc >= GGML_CUDA_CC_DP4A) {
        return false;
    }
    // decode batches only; every (token, expert) pair is one blockIdx.y
    if (dst->ne[2] > MMVQ_MAX_BATCH_SIZE || ids->ne[0]*ids->ne[1] > 65535 || src0->ne[3] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] % 256 != 0 || src1->ne[0] != src0->ne[0] || dst->ne[0] != src0->ne[1] || src1->ne[2] != ids->ne[1]) {
        return false;
    }
    // the weights are read in 4-byte words (Q2_K, block 84 bytes) or 2-byte aligned and realigned (Q3_K, block 110 bytes)
    const size_t align = src0->type == GGML_TYPE_Q2_K ? 4 : 2;
    if (src0->nb[0] != ggml_type_size(src0->type) || src0->nb[1] % align != 0 || src0->nb[2] % align != 0 ||
            (uintptr_t) src0->data % align != 0) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || src1->nb[2] % 16 != 0 || (uintptr_t) src1->data % 16 != 0) {
        return false;
    }
    if (ids->nb[0] != sizeof(int32_t) || ids->nb[1] % sizeof(int32_t) != 0 ||
            dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 || dst->nb[2] % sizeof(float) != 0) {
        return false;
    }
    return true;
}

// whether the last dispatch of this thread found its converted rows in the cache (for the log of GGML_CUDA_MMID_CHECK)
static thread_local bool g_last_hit = false;

bool ggml_cuda_mmid_vec_kq_sm60_last_hit() {
    return g_last_hit;
}

// GGML_CUDA_MMID_KQ_SM60_STATS=n: log the first n dispatches (the node, its shapes and strides, the cache) and then one line per
// 1000 dispatches with the totals, in every process
static void mmid_kq_log(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst,
        const int nsb, const bool hit) {
    static const int n_log = getenv("GGML_CUDA_MMID_KQ_SM60_STATS") != nullptr ? atoi(getenv("GGML_CUDA_MMID_KQ_SM60_STATS")) : 0;
    if (n_log <= 0) {
        return;
    }
    static std::atomic<int64_t> n_calls { 0 }, n_hits { 0 }, n_pairs { 0 };
    const int64_t i = n_calls.fetch_add(1);
    n_hits  += hit ? 1 : 0;
    n_pairs += ids->ne[0]*ids->ne[1];
    if (i < n_log) {
        GGML_LOG_WARN("mmid-kq-sm60: %-26s %s %s tokens %d used %d n_hot %d ne01 %d nsb %d ne11 %d ne12 %d | ids nb1 %zu | src1 nb1 %zu nb2 %zu | "
                      "dst nb1 %zu nb2 %zu | %s\n", dst->name, ggml_type_name(src0->type), ggml_cuda_mmid_cold(dst) ? "cold-tagged" : "plain",
                      (int) ids->ne[1], (int) ids->ne[0], (int) src0->ne[2], (int) src0->ne[1], nsb, (int) src1->ne[1], (int) src1->ne[2],
                      ids->nb[1], src1->nb[1], src1->nb[2], dst->nb[1], dst->nb[2], hit ? "cache hit" : "converted");
    } else if (i % 1000 == 0) {
        GGML_LOG_WARN("mmid-kq-sm60: %lld dispatches, %lld cache hits, %lld pairs\n", (long long) i, (long long) n_hits.load(), (long long) n_pairs.load());
    }
}

const char * ggml_cuda_mmid_kq_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const int nsb, bool & hit) {
    cudaStream_t stream = ctx.stream();

    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t s11  = src1->nb[1]/sizeof(float);
    const int64_t s12  = src1->nb[2]/sizeof(float);

    // gate and up read the same src1 back to back: the converted rows are kept for the second op (see mmid16_cache in common.cuh)
    const int64_t nrows1  = ne11*ne12;
    const size_t  total   = nrows1*nsb*KQ_REC_BYTES;
    const int64_t key     = ((nrows1 << 20) ^ ((int64_t) nsb << 8) ^ (s11 << 32) ^ s12) ^ (0x4b51LL << 48);

    // GGML_CUDA_MMID_KQ_SM60_NOCACHE=1: every op converts its own rows (a bisect against the sharing)
    static const bool nocache = getenv("GGML_CUDA_MMID_KQ_SM60_NOCACHE") != nullptr && atoi(getenv("GGML_CUDA_MMID_KQ_SM60_NOCACHE")) != 0;
    hit = !nocache && ctx.mmid16_cache_mem != nullptr && ctx.mmid16_cache_src1 == src1 &&
          ctx.mmid16_cache_data == src1->data && ctx.mmid16_cache_key == key;
    if (!hit) {
        if (total > ctx.mmid16_cache_cap) {
            ctx.retire_mem(ctx.mmid16_cache_mem, ctx.mmid16_cache_cap);
            ctx.mmid16_cache_clear();
            ggml_cuda_set_device(ctx.device);
            CUDA_CHECK(cudaMalloc((void **) &ctx.mmid16_cache_mem, total));
            ctx.mmid16_cache_cap = total;
        }
        mmid_kq_convert_rows<<<nrows1, WARP_SIZE*std::min(nsb, 32), 0, stream>>>(
            (const float *) src1->data, ctx.mmid16_cache_mem, s11, s12, (int) ne11, nsb);
        CUDA_CHECK(cudaGetLastError());
        ctx.mmid16_cache_src1 = src1;
        ctx.mmid16_cache_data = src1->data;
        ctx.mmid16_cache_key  = key;
    }
    return ctx.mmid16_cache_mem;
}

void ggml_cuda_mmid_vec_kq_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int64_t ne01   = src0->ne[1];
    const int64_t ne11   = src1->ne[1];
    const int64_t ne12   = src1->ne[2];
    const int     n_used = ids->ne[0];
    const int     nsb    = src0->ne[0]/256;
    const int     si1    = ids->nb[1]/sizeof(int32_t);

    bool hit = false;
    const char * x = ggml_cuda_mmid_kq_rows(ctx, src1, nsb, hit);
    g_last_hit = hit;
    mmid_kq_log(src0, src1, ids, dst, nsb, hit);

    const dim3 grid((ne01 + KQ_ROWS - 1)/KQ_ROWS, ne12*n_used, 1);
    const dim3 block(WARP_SIZE, KQ_NWARPS, 1);
    if (src0->type == GGML_TYPE_Q2_K) {
        mmid_kq_vec<GGML_TYPE_Q2_K><<<grid, block, 0, stream>>>(
            (const char *) src0->data, x, (const int32_t *) ids->data, (float *) dst->data, nsb, (int) ne01,
            src0->nb[1], src0->nb[2], (int) ne11, n_used, si1, (int) src0->ne[2], dst->nb[1], dst->nb[2]);
    } else {
        mmid_kq_vec<GGML_TYPE_Q3_K><<<grid, block, 0, stream>>>(
            (const char *) src0->data, x, (const int32_t *) ids->data, (float *) dst->data, nsb, (int) ne01,
            src0->nb[1], src0->nb[2], (int) ne11, n_used, si1, (int) src0->ne[2], dst->nb[1], dst->nb[2]);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------
// MUL_MAT: the same lanes and converter without the ids, grid (ne11*r2 columns, row tiles, ne02 matrices)

template <ggml_type type>
static __global__ void __launch_bounds__(WARP_SIZE*KQ_NWARPS, 8)
mm_kq_vec(
        const char * __restrict__ w, const char * __restrict__ x, float * __restrict__ dst, const int nsb, const int ne01,
        const int64_t nb01, const int64_t nb02, const int ne11, const int r2, const int64_t dst_nb1, const int64_t dst_nb2) {
#if defined(FP16_AVAILABLE)
    int i1, i2, row0;
    float sum = kq_mm_lane_t<type == GGML_TYPE_Q3_K ? 1 : type == GGML_TYPE_Q6_K ? 2 : 0>(w, x, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.y,
            threadIdx.x, nsb, ne01, nb01, nb02, ne11, r2, i1, i2, row0);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    if ((threadIdx.x & 3) == 0 && row0 < ne01) {
        *((float *) ((char *) dst + (int64_t) i2*dst_nb2 + (int64_t) i1*dst_nb1) + row0) = sum;
    }
#else
    GGML_UNUSED_VARS(w, x, dst, nsb, ne01, nb01, nb02, ne11, r2, dst_nb1, dst_nb2);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

bool ggml_cuda_mm_vec_kq_sm60_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_CUDA_MM_KQ_SM60") == nullptr || atoi(getenv("GGML_CUDA_MM_KQ_SM60")) != 0;
    // the widest batch that takes the kernel, for the crossover with the generic path (one block per column redoes the weight decode)
    static const int max_n = getenv("GGML_CUDA_MM_KQ_SM60_MAXN") != nullptr ?
        std::max(0, std::min(atoi(getenv("GGML_CUDA_MM_KQ_SM60_MAXN")), (int) MMVQ_MAX_BATCH_SIZE)) : (int) MMVQ_MAX_BATCH_SIZE;
    if (!enabled) {
        return false;
    }
    if ((src0->type != GGML_TYPE_Q2_K && src0->type != GGML_TYPE_Q3_K && src0->type != GGML_TYPE_Q6_K) || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_PASCAL || cc >= GGML_CUDA_CC_DP4A) {
        return false;
    }
    if (ggml_is_empty(src0) || ggml_is_empty(src1) || ggml_is_empty(dst)) {
        return false;
    }
    // decode batches of a matrix, or of its groups (src0 [k, m, g], src1 [k, n, g*r2]); no 4th dim
    if (src1->ne[1] > max_n || src0->ne[3] != 1 || src1->ne[3] != 1 || src1->ne[2] % src0->ne[2] != 0 ||
            src1->ne[2] > 65535 || (src0->ne[1] + KQ_ROWS - 1)/KQ_ROWS > 65535) {
        return false;
    }
    if (src0->ne[0] % 256 != 0 || src1->ne[0] != src0->ne[0] || dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2]) {
        return false;
    }
    // the weights are read in 4-byte words (Q2_K, block 84 bytes) or 2-byte aligned and realigned (Q3_K 110 bytes, Q6_K 210 bytes)
    const size_t align = src0->type == GGML_TYPE_Q2_K ? 4 : 2;
    if (src0->nb[0] != ggml_type_size(src0->type) || src0->nb[1] % align != 0 || src0->nb[2] % align != 0 ||
            (uintptr_t) src0->data % align != 0) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || src1->nb[2] % 16 != 0 || (uintptr_t) src1->data % 16 != 0) {
        return false;
    }
    if (dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 || dst->nb[2] % sizeof(float) != 0) {
        return false;
    }
    return true;
}

// GGML_CUDA_MM_KQ_SM60_STATS=n: log the first n dispatches of the plain MUL_MAT path (the node, its type and shape) and then one
// line per 1000 with the totals, in every process - the proof that the attention / dense matvecs take this path in a model run
static void mm_kq_log(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const int n_log = getenv("GGML_CUDA_MM_KQ_SM60_STATS") != nullptr ? atoi(getenv("GGML_CUDA_MM_KQ_SM60_STATS")) : 0;
    if (n_log <= 0) {
        return;
    }
    static std::atomic<int64_t> n_calls { 0 }, n_bytes { 0 };
    const int64_t i = n_calls.fetch_add(1);
    n_bytes += ggml_nbytes(src0);
    if (i < n_log) {
        GGML_LOG_WARN("mm-kq-sm60: %-26s %s src0 %d x %d x %d (%.1f MiB) src1 n %d x %d | src1 nb1 %zu | dst nb1 %zu\n", dst->name,
                      ggml_type_name(src0->type), (int) src0->ne[0], (int) src0->ne[1], (int) src0->ne[2], ggml_nbytes(src0)/1048576.0,
                      (int) src1->ne[1], (int) src1->ne[2], src1->nb[1], dst->nb[1]);
    } else if (i % 1000 == 0) {
        GGML_LOG_WARN("mm-kq-sm60: %lld dispatches, %.1f GiB of weights\n", (long long) i, n_bytes.load()/1073741824.0);
    }
}

void ggml_cuda_mm_vec_kq_sm60(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int     nsb  = src0->ne[0]/256;
    const int     r2   = ne12/ne02;

    mm_kq_log(src0, src1, dst);

    // converted per op into pool scratch: no cache, so the Q2_0 / MUL_MAT_ID gate-up cache (mmid16_cache) is never touched
    // and a temporary src1 (the slices of the generic MUL_MAT_ID path) cannot hit a stale entry
    const int64_t nrows1 = ne11*ne12;
    ggml_cuda_pool_alloc<char> x(ctx.pool(), nrows1*nsb*KQ_REC_BYTES);
    mmid_kq_convert_rows<<<nrows1, WARP_SIZE*std::min(nsb, 32), 0, stream>>>(
        (const float *) src1->data, x.get(), src1->nb[1]/sizeof(float), src1->nb[2]/sizeof(float), (int) ne11, nsb);
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid(ne11*r2, (ne01 + KQ_ROWS - 1)/KQ_ROWS, ne02);
    const dim3 block(WARP_SIZE, KQ_NWARPS, 1);
    // Q6_K: the warp-per-row kernel (mm-q6k-sm60.cu) for up to MMVQ_MAX_BATCH_SIZE columns (GGML_CUDA_MM_Q6K_WARP=0: the lane kernel)
    static const bool q6_warp = getenv("GGML_CUDA_MM_Q6K_WARP") == nullptr || atoi(getenv("GGML_CUDA_MM_Q6K_WARP")) != 0;
    if (src0->type == GGML_TYPE_Q6_K && q6_warp && ne11*r2 <= MMVQ_MAX_BATCH_SIZE) {
        ggml_cuda_mm_q6k_vec(ctx, src0, src1, dst, x.get());
        return;
    }
    if (src0->type == GGML_TYPE_Q2_K) {
        mm_kq_vec<GGML_TYPE_Q2_K><<<grid, block, 0, stream>>>(
            (const char *) src0->data, x.get(), (float *) dst->data, nsb, (int) ne01, src0->nb[1], src0->nb[2], (int) ne11, r2,
            dst->nb[1], dst->nb[2]);
    } else if (src0->type == GGML_TYPE_Q6_K) {
        mm_kq_vec<GGML_TYPE_Q6_K><<<grid, block, 0, stream>>>(
            (const char *) src0->data, x.get(), (float *) dst->data, nsb, (int) ne01, src0->nb[1], src0->nb[2], (int) ne11, r2,
            dst->nb[1], dst->nb[2]);
    } else {
        mm_kq_vec<GGML_TYPE_Q3_K><<<grid, block, 0, stream>>>(
            (const char *) src0->data, x.get(), (float *) dst->data, nsb, (int) ne01, src0->nb[1], src0->nb[2], (int) ne11, r2,
            dst->nb[1], dst->nb[2]);
    }
    CUDA_CHECK(cudaGetLastError());
}
