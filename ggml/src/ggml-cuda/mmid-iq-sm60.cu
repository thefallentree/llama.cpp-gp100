// IQ2_S / IQ3_XXS / IQ3_S MUL_MAT_ID mat-vec for decode batches on sm_60 (GP100): the hot pairs of Project Maya's files.
//
// GP100 has no DP4A, so the generic mmvq kernels emulate every int8 dot product and every __vcmpne4 / __vsub4 of the sign
// decode. Here a grid word's bytes stay where they are: masked into half subnormals, OR'ed with the half2 sign masks of a
// shared table (by the group's sign byte), multiplied with HFMA2 against the activation pairs of the Q2_K kernel's converted
// rows (mmid-kq-sm60.cu, one conversion per src1 shared by gate and up). The math of one lane is in mmid-iq-sm60-math.cuh,
// which a CPU test also compiles (runs/fn-m2/mayam/iq_harness.cpp).
//
// A warp handles 8 rows, 4 lanes per row: lane l reads the 64 weights 64l..64l+63 of every super-block of its row, so the
// block bytes are read once. One blockIdx.y per (token, expert) slot, as in the Q2_K mat-vec.

#include "mmid-iq-sm60.cuh"
#include "mmid-kq-sm60.cuh"
#include "mmid-f16-sm60.cuh"
#include "mmvq.cuh"

#include "mmid-iq-sm60-math.cuh"

#include <atomic>

// IQ_T: 0 = IQ2_S, 1 = IQ3_XXS, 2 = IQ3_S. A pair c = (token t, used expert s) per blockIdx.y; a pair whose expert is not in w
// (e >= n_hot) is skipped by the whole block.
template <int IQ_T>
static __global__ void __launch_bounds__(WARP_SIZE*KQ_NWARPS, 8)
mmid_iq_vec(
        const char * __restrict__ w, const char * __restrict__ x,
        const int32_t * __restrict__ ids, float * __restrict__ dst, const int nsb, const int ne01, const int64_t nb01, const int64_t nb02,
        const int ne11, const int n_used, const int si1, const int n_hot, const int64_t dst_nb1, const int64_t dst_nb2) {
#if defined(FP16_AVAILABLE)
    // the half2 sign masks by sign byte (IQ2_S, IQ3_S) or by the 7-bit sign index (IQ3_XXS)
    __shared__ uint4 sgn[256];
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    for (int i = tid; i < (IQ_T == 1 ? 128 : 256); i += WARP_SIZE*KQ_NWARPS) {
        sgn[i] = iq_sign_masks(IQ_T == 1 ? (uint32_t) ksigns_iq2xs[i] : (uint32_t) i);
    }
    __syncthreads();
    const void * grid = IQ_T == 0 ? (const void *) iq2s_grid : IQ_T == 1 ? (const void *) iq3xxs_grid : (const void *) iq3s_grid;
    float sum;
    int t, s, row0;
    if (!iq_vec_lane<IQ_T>(w, x, ids, blockIdx.x, blockIdx.y, threadIdx.y, threadIdx.x, nsb, ne01, nb01, nb02,
            ne11, n_used, si1, n_hot, sgn, grid, sum, t, s, row0)) {
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

static int mmid_iq_type(const ggml_type t) {
    return t == GGML_TYPE_IQ2_S ? 0 : t == GGML_TYPE_IQ3_XXS ? 1 : t == GGML_TYPE_IQ3_S ? 2 : -1;
}

bool ggml_cuda_mmid_vec_iq_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_CUDA_MMID_IQ_SM60") == nullptr || atoi(getenv("GGML_CUDA_MMID_IQ_SM60")) != 0;
    if (!enabled || ids == nullptr) {
        return false;
    }
    if (mmid_iq_type(src0->type) < 0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
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
    // the blocks (82 / 98 / 110 bytes) are read 2-byte aligned and realigned into words
    if (src0->nb[0] != ggml_type_size(src0->type) || src0->nb[1] % 2 != 0 || src0->nb[2] % 2 != 0 || (uintptr_t) src0->data % 2 != 0) {
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

static thread_local bool g_last_hit = false;

bool ggml_cuda_mmid_vec_iq_sm60_last_hit() {
    return g_last_hit;
}

// GGML_CUDA_MMID_IQ_SM60_STATS=n: log the first n dispatches, then one line per 1000 with the totals
static void mmid_iq_log(const ggml_tensor * src0, const ggml_tensor * ids, const ggml_tensor * dst, const int nsb, const bool hit) {
    static const int n_log = getenv("GGML_CUDA_MMID_IQ_SM60_STATS") != nullptr ? atoi(getenv("GGML_CUDA_MMID_IQ_SM60_STATS")) : 0;
    if (n_log <= 0) {
        return;
    }
    static std::atomic<int64_t> n_calls { 0 }, n_hits { 0 }, n_pairs { 0 };
    const int64_t i = n_calls.fetch_add(1);
    n_hits  += hit ? 1 : 0;
    n_pairs += ids->ne[0]*ids->ne[1];
    if (i < n_log) {
        GGML_LOG_WARN("mmid-iq-sm60: %-26s %s %s tokens %d used %d n_hot %d ne01 %d nsb %d | %s\n", dst->name, ggml_type_name(src0->type),
                      ggml_cuda_mmid_cold(dst) ? "cold-tagged" : "plain", (int) ids->ne[1], (int) ids->ne[0], (int) src0->ne[2], (int) src0->ne[1],
                      nsb, hit ? "cache hit" : "converted");
    } else if (i % 1000 == 0) {
        GGML_LOG_WARN("mmid-iq-sm60: %lld dispatches, %lld cache hits, %lld pairs\n", (long long) i, (long long) n_hits.load(), (long long) n_pairs.load());
    }
}

void ggml_cuda_mmid_vec_iq_sm60(
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
    mmid_iq_log(src0, ids, dst, nsb, hit);

    const dim3 grid((ne01 + KQ_ROWS - 1)/KQ_ROWS, ne12*n_used, 1);
    const dim3 block(WARP_SIZE, KQ_NWARPS, 1);
    switch (mmid_iq_type(src0->type)) {
        case 0:
            mmid_iq_vec<0><<<grid, block, 0, stream>>>(
                (const char *) src0->data, x, (const int32_t *) ids->data, (float *) dst->data, nsb, (int) ne01,
                src0->nb[1], src0->nb[2], (int) ne11, n_used, si1, (int) src0->ne[2], dst->nb[1], dst->nb[2]);
            break;
        case 1:
            mmid_iq_vec<1><<<grid, block, 0, stream>>>(
                (const char *) src0->data, x, (const int32_t *) ids->data, (float *) dst->data, nsb, (int) ne01,
                src0->nb[1], src0->nb[2], (int) ne11, n_used, si1, (int) src0->ne[2], dst->nb[1], dst->nb[2]);
            break;
        default:
            mmid_iq_vec<2><<<grid, block, 0, stream>>>(
                (const char *) src0->data, x, (const int32_t *) ids->data, (float *) dst->data, nsb, (int) ne01,
                src0->nb[1], src0->nb[2], (int) ne11, n_used, si1, (int) src0->ne[2], dst->nb[1], dst->nb[2]);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}
