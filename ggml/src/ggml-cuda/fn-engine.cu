#include "fn-engine.cuh"
#include "convert.cuh"

#include <atomic>
#include <mutex>
#include <unordered_map>

#define FN_QK 32 // weights per Q8_0 block

// Geometry by row width (TPR = columns/16): rows of up to 2560 columns in blocks of 160 threads, several rows per
// block when they are narrower; 6144 and 10240 columns in blocks of 384 and 640 threads.
// R rows per tile: 8, or 4 where 8 rows of T tokens exceed the registers, the 48 KB of shared memory of a block or
// the R*T/2 gathering threads a row has. BPS: blocks per SM; minBlocksPerMultiprocessor caps the registers so that
// nsm*BPS blocks are resident together, and is chosen so that no instantiation spills under that cap
// (cuobjdump --dump-resource-usage).
static bool fn_dense_geometry(const int tpr) {
    return tpr == 20 || tpr == 40 || tpr == 160 || tpr == 384 || tpr == 640;
}

static int fn_dense_tile_rows(const int tpr) { // the largest tile, rows must be a multiple of it
    return tpr == 20 ? 64 : tpr == 40 ? 32 : 8;
}


#ifndef FN_STANDALONE

// ---------------------------------------------------------------------------------------------------------------
// planar weights: the registry

struct fn_plane_entry {
    ggml_cuda_fn_plane plane;
    int                device;
};

static std::mutex                                        g_fn_mutex;
static std::unordered_map<const void *, fn_plane_entry> g_fn_planes;
static std::atomic<int>                                  g_fn_n_planes { 0 };

bool ggml_cuda_fn_enabled() {
    static const bool enabled = getenv("GGML_CUDA_FN") == nullptr || atoi(getenv("GGML_CUDA_FN")) != 0;
    return enabled;
}

bool ggml_cuda_fn_planar(const ggml_tensor * t, ggml_cuda_fn_plane * plane) {
    if (g_fn_n_planes.load(std::memory_order_relaxed) == 0 || t->type != GGML_TYPE_Q8_0 || t->data == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_fn_mutex);
    const auto it = g_fn_planes.find(t->data);
    if (it == g_fn_planes.end()) {
        return false;
    }
    if (plane != nullptr) {
        *plane = it->second.plane;
    }
    return true;
}

void ggml_cuda_fn_planes_release(const void * base, const size_t size) {
    if (g_fn_n_planes.load(std::memory_order_relaxed) == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_fn_mutex);
    for (auto it = g_fn_planes.begin(); it != g_fn_planes.end();) {
        const char * p = (const char *) it->first;
        if (p >= (const char *) base && p < (const char *) base + size) {
            ggml_cuda_set_device(it->second.device);
            cudaFree(it->second.plane.rowscale);
            it = g_fn_planes.erase(it);
        } else {
            ++it;
        }
    }
    g_fn_n_planes.store((int) g_fn_planes.size(), std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------------------------------------------
// planar weights: the repack

// the largest block scale of each row; one warp per row
static __global__ void fn_p8_rowmax(const char * __restrict__ src, const int nb, float * __restrict__ rowscale) {
    const char * row = src + (size_t) blockIdx.x*nb*(FN_QK + 2);
    float m = 0.0f;
    for (int b = threadIdx.x; b < nb; b += WARP_SIZE) {
        m = fmaxf(m, fabsf(__half2float(*(const half *) (row + (size_t) b*(FN_QK + 2)))));
    }
#pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    }
    if (threadIdx.x == 0) {
        rowscale[blockIdx.x] = m;
    }
}

// one thread per Q8_0 block of `src` (a copy of the tensor): its quants go to the code plane of `dst` (the tensor's
// own memory), its scale, relative to the row's, to the scale plane behind it
static __global__ void fn_p8_repack(const char * __restrict__ src, const int nb, const int64_t nrows,
                                    const float * __restrict__ rowscale, char * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nrows*nb) {
        return;
    }
    const char *  blk = src + i*(FN_QK + 2);
    const float   d   = __half2float(*(const half *) blk);
    const float   rs  = rowscale[i / nb];
    const uint16_t * q  = (const uint16_t *) (blk + 2); // the blocks are 2-byte aligned
    uint32_t *       out = (uint32_t *) (dst + i*FN_QK);
#pragma unroll
    for (int k = 0; k < FN_QK/4; ++k) {
        out[k] = ((uint32_t) q[2*k] | ((uint32_t) q[2*k + 1] << 16)) ^ 0x80808080u;
    }
    ((half *) (dst + nrows*nb*FN_QK))[i] = __float2half(rs > 0.0f ? d/rs : 0.0f);
}

static bool fn_planar_eligible(const ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = node->src[0];
    if (w->type != GGML_TYPE_Q8_0 || w->data == nullptr || w->buffer == nullptr || w->view_src != nullptr ||
        ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        ggml_backend_buffer_is_host(w->buffer) || !ggml_is_contiguous(w) || w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    const int64_t cols = w->ne[0];
    const int64_t rows = w->ne[1];
    // the row widths and tile heights the mat-vec is instantiated for
    if (cols % FN_QK != 0 || !fn_dense_geometry((int) (cols/16)) || rows % fn_dense_tile_rows((int) (cols/16)) != 0 ||
        rows*cols >= ((int64_t) 1 << 31)) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_DP4A;
}

void ggml_cuda_fn_planes_optimize(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    if (!ggml_cuda_fn_enabled()) {
        return;
    }
    bool         synced = false;
    cudaStream_t stream = ctx.stream();
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!fn_planar_eligible(ctx, node)) {
            continue;
        }
        ggml_tensor * w = node->src[0];
        {
            std::lock_guard<std::mutex> lock(g_fn_mutex);
            if (g_fn_planes.count(w->data) != 0) {
                continue;
            }
        }
        if (!synced) {
            ggml_cuda_set_device(ctx.device);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            synced = true;
        }
        const int64_t rows   = w->ne[1];
        const int     nb     = (int) (w->ne[0]/FN_QK);
        const size_t  nbytes = ggml_nbytes(w);
        GGML_ASSERT(nbytes == (size_t) rows*nb*(FN_QK + 2));

        fn_plane_entry e;
        e.device = ctx.device;
        char * tmp = nullptr;
        CUDA_CHECK(cudaMalloc((void **) &e.plane.rowscale, rows*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **) &tmp, nbytes));
        CUDA_CHECK(cudaMemcpyAsync(tmp, w->data, nbytes, cudaMemcpyDeviceToDevice, stream));
        fn_p8_rowmax<<<(unsigned) rows, WARP_SIZE, 0, stream>>>(tmp, nb, e.plane.rowscale);
        const int64_t nblk = rows*nb;
        fn_p8_repack<<<(unsigned) ((nblk + 255)/256), 256, 0, stream>>>(tmp, nb, rows, e.plane.rowscale, (char *) w->data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(tmp));

        std::lock_guard<std::mutex> lock(g_fn_mutex);
        g_fn_planes[w->data] = e;
        g_fn_n_planes.store((int) g_fn_planes.size(), std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// planar weights: dequantize (prompt batches)

template <typename dst_t>
static __global__ void fn_p8_dequantize(const uint4 * __restrict__ W, const half * __restrict__ D,
                                        const float * __restrict__ rowscale, const int tpr, const int64_t n16,
                                        dst_t * __restrict__ y) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n16) {
        return;
    }
    const int64_t r = i / tpr;
    const int     k = (int) (i - r*tpr);
    const uint4   v = W[i];
    const float   s = __half2float(D[r*(tpr/2) + (k >> 1)])*rowscale[r];
    const uint32_t w[4] = { v.x, v.y, v.z, v.w };
    dst_t * yo = y + i*16;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int q = (int) ((w[j / 4] >> (8*(j % 4))) & 0xff) - 128;
        yo[j] = ggml_cuda_cast<dst_t>(s*(float) q);
    }
}

void ggml_cuda_fn_dequantize(const ggml_tensor * src0, void * dst, const ggml_type dst_type, cudaStream_t stream) {
    ggml_cuda_fn_plane plane;
    GGML_ASSERT(ggml_cuda_fn_planar(src0, &plane));
    const int64_t rows = src0->ne[1];
    const int64_t cols = src0->ne[0];
    const int     tpr  = (int) (cols/16);
    const int64_t n16  = rows*tpr;
    const uint4 * W = (const uint4 *) src0->data;
    const half *  D = (const half *) ((const char *) src0->data + rows*cols);
    const unsigned grid = (unsigned) ((n16 + 255)/256);
    switch (dst_type) {
        case GGML_TYPE_F16:
            fn_p8_dequantize<half><<<grid, 256, 0, stream>>>(W, D, plane.rowscale, tpr, n16, (half *) dst);
            break;
        case GGML_TYPE_BF16:
            fn_p8_dequantize<nv_bfloat16><<<grid, 256, 0, stream>>>(W, D, plane.rowscale, tpr, n16, (nv_bfloat16 *) dst);
            break;
        case GGML_TYPE_F32:
            fn_p8_dequantize<float><<<grid, 256, 0, stream>>>(W, D, plane.rowscale, tpr, n16, (float *) dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // FN_STANDALONE

// ---------------------------------------------------------------------------------------------------------------
// activations: fp32 -> fp16, one scale per token
//
// X = x*gain/amax, with gain = 400/n: a row sum of n products of weights up to 127 then stays below 50.8K, inside
// fp16, whatever the values. One block per token.
#define FN_ACT_TPB 256

static __global__ void fn_act_h16(const float * __restrict__ x, const int64_t stride_t, const int n, const float gain,
                                  half * __restrict__ X, float * __restrict__ xscale) {
    __shared__ float s_red[FN_ACT_TPB/WARP_SIZE];
    const float * xt = x + blockIdx.x*stride_t;
    float m = 0.0f;
    for (int i = threadIdx.x; i < n; i += FN_ACT_TPB) {
        m = fmaxf(m, fabsf(xt[i]));
    }
#pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    }
    if (threadIdx.x % WARP_SIZE == 0) {
        s_red[threadIdx.x / WARP_SIZE] = m;
    }
    __syncthreads();
    m = 0.0f;
#pragma unroll
    for (int w = 0; w < FN_ACT_TPB/WARP_SIZE; ++w) {
        m = fmaxf(m, s_red[w]);
    }
    const float s = m > 0.0f ? gain/m : 0.0f;
    half * Xt = X + (int64_t) blockIdx.x*n;
    for (int i = threadIdx.x; i < n; i += FN_ACT_TPB) {
        Xt[i] = __float2half_rn(xt[i]*s);
    }
    if (threadIdx.x == 0) {
        xscale[blockIdx.x] = m > 0.0f ? m/gain : 0.0f;
    }
}

static float fn_gain(const int64_t n) {
    return 400.0f/(float) n;
}

// ---------------------------------------------------------------------------------------------------------------
// the dense mat-vec

static __device__ __forceinline__ half2 fn_h2(const int v) { return *(const half2 *) &v; }
static __device__ __forceinline__ int   fn_i2(const half2 v) { return *(const int *) &v; }

// 16 biased quants -> 8 half2 of exact signed weights: a byte under the exponent byte 0x64 is the half 1024 + byte
static __device__ __forceinline__ void fn_p8_decode(const uint4 v, const half2 magic, half2 * w) {
    w[0] = __hsub2(fn_h2(__byte_perm(v.x, 0x64646464, 0x4140)), magic);
    w[1] = __hsub2(fn_h2(__byte_perm(v.x, 0x64646464, 0x4342)), magic);
    w[2] = __hsub2(fn_h2(__byte_perm(v.y, 0x64646464, 0x4140)), magic);
    w[3] = __hsub2(fn_h2(__byte_perm(v.y, 0x64646464, 0x4342)), magic);
    w[4] = __hsub2(fn_h2(__byte_perm(v.z, 0x64646464, 0x4140)), magic);
    w[5] = __hsub2(fn_h2(__byte_perm(v.z, 0x64646464, 0x4342)), magic);
    w[6] = __hsub2(fn_h2(__byte_perm(v.w, 0x64646464, 0x4140)), magic);
    w[7] = __hsub2(fn_h2(__byte_perm(v.w, 0x64646464, 0x4342)), magic);
}

// the 16 columns of one thread for T tokens, as 8 half2 each
template <int T>
static __device__ __forceinline__ void fn_load_act(const half2 * __restrict__ X, const int tpr, const int k, half2 a[T][8]) {
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const int4 * ap = (const int4 *) (X + ((int64_t) t*tpr + k)*8);
        const int4 a0 = ap[0];
        const int4 a1 = ap[1];
        a[t][0] = fn_h2(a0.x); a[t][1] = fn_h2(a0.y); a[t][2] = fn_h2(a0.z); a[t][3] = fn_h2(a0.w);
        a[t][4] = fn_h2(a1.x); a[t][5] = fn_h2(a1.y); a[t][6] = fn_h2(a1.z); a[t][7] = fn_h2(a1.w);
    }
}

// the scaled partial dot product of one thread's 16 columns of one row, per token
template <int T>
static __device__ __forceinline__ void fn_dot16(const uint4 v, const half d, const half2 magic, const half2 a[T][8], half * s) {
    half2 w[8];
    fn_p8_decode(v, magic, w);
#pragma unroll
    for (int t = 0; t < T; ++t) {
        half2 acc = __hmul2(w[0], a[t][0]);
#pragma unroll
        for (int c = 1; c < 8; ++c) {
            acc = __hfma2(w[c], a[t][c], acc);
        }
        s[t] = __hmul(__hadd(__low2half(acc), __high2half(acc)), d);
    }
}

// the sum of the TPR partial pairs of one row
template <int TPR>
static __device__ __forceinline__ float2 fn_gather(const int * __restrict__ pp) {
    half2 c[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        c[j] = fn_h2(pp[j]);
    }
#pragma unroll
    for (int j = 4; j < TPR; ++j) {
        c[j % 4] = __hadd2(c[j % 4], fn_h2(pp[j]));
    }
    return __half22float2(__hadd2(__hadd2(c[0], c[1]), __hadd2(c[2], c[3])));
}

#define FN_EPI_NONE 0
#define FN_EPI_SILU 1 // silu(es*y + eb)

static __device__ __forceinline__ float fn_epilogue(const float y, const int epi, const float es, const float eb) {
    if (epi == FN_EPI_SILU) {
        const float z = es*y + eb;
        return z/(1.0f + expf(-z));
    }
    return y;
}

// y[t][row] = sum_c W[row][c] x[t][c], for T tokens.
//   block: G rows x TPR threads; thread (g, k) owns columns [16k, 16k + 16) of the rows g, g + G, ... of a tile
//   tile:  G*R rows; the block loops over the tiles blockIdx, blockIdx + grid, ...
// A thread's R*T partial sums go to shared memory two per word; thread (g, p) then sums pair p over row g's threads.
// The geometry is compile-time: with it as kernel arguments the index arithmetic (XMAD chains for every load and
// store) doubled the instruction count and the kernel ran at half the speed.
template <int T, int R, int TPR, int G, int BPS>
static __global__ void __launch_bounds__(G*TPR, BPS)
fn_dense_p8(const uint4 * __restrict__ W, const half * __restrict__ D, const float * __restrict__ rowscale,
            const half2 * __restrict__ X, const float * __restrict__ xscale,
            float * __restrict__ dst, const int dst_stride, const int nrows,
            const int epi, const float es, const float eb) {
#if defined(FP16_AVAILABLE)
    constexpr int NT  = G*TPR;
    constexpr int NPG = R*T/2;
    __shared__ int s_part[NPG][NT];
    const int tid = threadIdx.x;
    const int g   = tid/TPR;
    const int k   = tid - g*TPR;
    const half2 magic = __float2half2_rn(1152.0f);

    half2 a[T][8];
    fn_load_act<T>(X, TPR, k, a);

    // the threads that gather: pair p of row group gg, i.e. the values (2p, 2p + 1) = (row step, token)
    const bool gather = tid < G*NPG;
    const int  gg     = tid/NPG;
    const int  p      = tid - gg*NPG;
    const int  vr0    = ((2*p)/T)*G + gg;      // row within the tile, token of the pair's two values
    const int  vr1    = ((2*p + 1)/T)*G + gg;
    const int  vt0    = (2*p) % T;
    const int  vt1    = (2*p + 1) % T;
    float xs0 = 0.0f, xs1 = 0.0f;
    if (gather) {
        xs0 = xscale[vt0];
        xs1 = xscale[vt1];
    }

    const int ntiles = nrows/(G*R);
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        // row (r, g) of the tile is tile*G*R + r*G + g: consecutive rows for consecutive g
        const uint4 * wr = W + (size_t) tile*(R*NT) + tid;
        const half *  dr = D + (size_t) tile*(R*NT/2) + g*(TPR/2) + (k >> 1);
        uint4 v[R];
        half  d[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            v[r] = __ldg(wr + r*NT);
            d[r] = dr[r*(NT/2)];
        }
        // the row scales are loaded with the weights: a load issued by the gathering threads after the barrier
        // would make the whole block wait for it
        float rs0 = 0.0f, rs1 = 0.0f;
        if (gather) {
            rs0 = rowscale[tile*(G*R) + vr0];
            rs1 = rowscale[tile*(G*R) + vr1];
        }
        half s[R*T];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            fn_dot16<T>(v[r], d[r], magic, a, s + r*T);
        }
#pragma unroll
        for (int p = 0; p < NPG; ++p) {
            s_part[p][tid] = fn_i2(__halves2half2(s[2*p], s[2*p + 1]));
        }
        __syncthreads();
        if (gather) {
            const float2 f = fn_gather<TPR>(&s_part[p][gg*TPR]);
            dst[vt0*dst_stride + tile*(G*R) + vr0] = fn_epilogue(f.x*rs0*xs0, epi, es, eb);
            dst[vt1*dst_stride + tile*(G*R) + vr1] = fn_epilogue(f.y*rs1*xs1, epi, es, eb);
        }
        __syncthreads();
    }
#else
    GGML_UNUSED_VARS(W, D, rowscale, X, xscale, dst, dst_stride, nrows, epi, es, eb);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int T, int R, int TPR, int G, int BPS>
static void fn_dense_launch_t(const uint4 * W, const half * D, const float * rowscale, const half2 * X, const float * xscale,
                              float * dst, const int dst_stride, const int nrows, const int nsm,
                              const int epi, const float es, const float eb, cudaStream_t stream) {
    const int ntiles = nrows/(G*R);
    fn_dense_p8<T, R, TPR, G, BPS><<<std::min(ntiles, nsm*BPS), G*TPR, 0, stream>>>(
        W, D, rowscale, X, xscale, dst, dst_stride, nrows, epi, es, eb);
}

static void fn_dense_launch(const int nt, const uint4 * W, const half * D, const float * rowscale, const half2 * X,
                            const float * xscale, float * dst, const int dst_stride, const int nrows, const int tpr,
                            const int nsm, const int epi, const float es, const float eb, cudaStream_t stream) {
    GGML_ASSERT(fn_dense_geometry(tpr) && nrows % fn_dense_tile_rows(tpr) == 0);
#define FN_ARGS W, D, rowscale, X, xscale, dst, dst_stride, nrows, nsm, epi, es, eb, stream
#define FN_CASE(N, R20, B20, R40, B40, R160, B160, R384, B384, R640)                       \
        case N:                                                                              \
            switch (tpr) {                                                                   \
                case  20: fn_dense_launch_t<N, R20,   20, 8, B20 >(FN_ARGS); break;          \
                case  40: fn_dense_launch_t<N, R40,   40, 4, B40 >(FN_ARGS); break;          \
                case 160: fn_dense_launch_t<N, R160, 160, 1, B160>(FN_ARGS); break;          \
                case 384: fn_dense_launch_t<N, R384, 384, 1, B384>(FN_ARGS); break;          \
                default:  fn_dense_launch_t<N, R640, 640, 1, 1   >(FN_ARGS); break;          \
            }                                                                                \
            break;
    switch (nt) {
        FN_CASE(1, 8, 8, 8, 8, 8, 8, 8, 3, 8)
        FN_CASE(2, 8, 6, 8, 6, 8, 6, 8, 2, 8)
        FN_CASE(3, 8, 6, 8, 6, 8, 6, 8, 2, 8)
        FN_CASE(4, 8, 4, 8, 4, 8, 4, 8, 2, 8)
        FN_CASE(5, 8, 4, 8, 4, 8, 4, 8, 2, 4)
        FN_CASE(6, 4, 4, 8, 3, 8, 3, 4, 1, 4)
        FN_CASE(7, 4, 3, 4, 3, 4, 3, 4, 1, 4)
        FN_CASE(8, 4, 3, 4, 3, 4, 3, 4, 1, 4)
        default:
            GGML_ABORT("fatal error");
    }
#undef FN_CASE
#undef FN_ARGS
    CUDA_CHECK(cudaGetLastError());
}

#ifndef FN_STANDALONE

// ---------------------------------------------------------------------------------------------------------------
// MUL_MAT

bool ggml_cuda_fn_mul_mat_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return ggml_cuda_fn_planar(src0) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= 4*FN_MAX_T && src1->ne[2] == 1 && src1->ne[3] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1 &&
           src1->nb[0] == sizeof(float) && src1->ne[0] == src0->ne[0] && ggml_is_contiguous(dst);
}

// the fp16 activations of src1 (two slots: consecutive mat-vecs mostly read the same vector)
static const ggml_backend_cuda_context::fn_act_slot & fn_act_get(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
                                                                  const float * x, const int64_t stride_t, const int n,
                                                                  const int nt) {
    for (auto & s : ctx.fn_act) {
        if (s.mem != nullptr && s.src1 == src1 && s.data == x && s.n == n && s.nt == nt && s.stride == stride_t) {
            return s;
        }
    }
    auto & s = ctx.fn_act[ctx.fn_act_next];
    ctx.fn_act_next ^= 1;
    const size_t need = (size_t) FN_MAX_T*n*sizeof(half) + FN_MAX_T*sizeof(float);
    if (need > s.cap) {
        ctx.retire_mem(s.mem, s.cap);
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc((void **) &s.mem, need));
        s.cap = need;
    }
    s.src1   = src1;
    s.data   = x;
    s.n      = n;
    s.nt     = nt;
    s.stride = stride_t;
    fn_act_h16<<<nt, FN_ACT_TPB, 0, ctx.stream()>>>(x, stride_t, n, fn_gain(n), (half *) s.mem,
                                                    (float *) (s.mem + (size_t) FN_MAX_T*n*sizeof(half)));
    CUDA_CHECK(cudaGetLastError());
    return s;
}

void ggml_cuda_fn_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_fn_plane plane;
    GGML_ASSERT(ggml_cuda_fn_planar(src0, &plane));
    const int     cols = (int) src0->ne[0];
    const int     rows = (int) src0->ne[1];
    const int     tpr  = cols/16;
    const int     nsm  = ggml_cuda_info().devices[ctx.device].nsm;
    const int64_t s1   = src1->nb[1]/sizeof(float);
    const int64_t sd   = dst->nb[1]/sizeof(float);
    const uint4 * W = (const uint4 *) src0->data;
    const half *  D = (const half *) ((const char *) src0->data + (int64_t) rows*cols);

    for (int64_t c0 = 0; c0 < src1->ne[1]; c0 += FN_MAX_T) {
        const int     nt = (int) std::min<int64_t>(FN_MAX_T, src1->ne[1] - c0);
        const float * x  = (const float *) src1->data + c0*s1;
        const auto &  s  = fn_act_get(ctx, src1, x, s1, cols, nt);
        fn_dense_launch(nt, W, D, plane.rowscale, (const half2 *) s.mem,
                        (const float *) (s.mem + (size_t) FN_MAX_T*cols*sizeof(half)),
                        (float *) dst->data + c0*sd, sd, rows, tpr, nsm, FN_EPI_NONE, 0.0f, 0.0f, ctx.stream());
    }
}

#endif // FN_STANDALONE

// ---------------------------------------------------------------------------------------------------------------
// the hyper-connection read (see hc-mix.cuh for the math), planar weights

#define FN_HC      4   // streams
#define FN_HC_TPB  512

// One block per token: xn = rms_norm(R)*w_norm per stream (written for the inject mat-vec and the up kernel), and
// xn as fp16 activations of the down mat-vec.
static __global__ void fn_hc_norm(const float * __restrict__ R, const int64_t sr_c, const int64_t sr_t,
                                  const float * __restrict__ wn, const int n_embd, const float eps,
                                  float * __restrict__ xn, const int64_t sxn_c, const int64_t sxn_t,
                                  half * __restrict__ X, float * __restrict__ xscale, const float gain) {
    __shared__ float s_red[FN_HC][FN_HC_TPB/WARP_SIZE];
    __shared__ float s_max[FN_HC_TPB/WARP_SIZE];
    const int t    = blockIdx.x;
    const int warp = threadIdx.x / WARP_SIZE;
    const float * Rt = R + t*sr_t;

    float ss[FN_HC];
#pragma unroll
    for (int c = 0; c < FN_HC; ++c) {
        ss[c] = 0.0f;
        for (int i = 4*threadIdx.x; i < n_embd; i += 4*FN_HC_TPB) {
            const float4 r = *(const float4 *) (Rt + c*sr_c + i);
            ss[c] += r.x*r.x + r.y*r.y + r.z*r.z + r.w*r.w;
        }
        ss[c] = warp_reduce_sum(ss[c]);
        if (threadIdx.x % WARP_SIZE == 0) {
            s_red[c][warp] = ss[c];
        }
    }
    __syncthreads();
    float rs[FN_HC];
#pragma unroll
    for (int c = 0; c < FN_HC; ++c) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < FN_HC_TPB/WARP_SIZE; ++w) {
            sum += s_red[c][w];
        }
        rs[c] = rsqrtf(sum/n_embd + eps);
    }
    float m = 0.0f;
#pragma unroll
    for (int c = 0; c < FN_HC; ++c) {
        for (int i = 4*threadIdx.x; i < n_embd; i += 4*FN_HC_TPB) {
            const float4 r = *(const float4 *) (Rt + c*sr_c + i);
            const float4 g = *(const float4 *) (wn + (int64_t) c*n_embd + i);
            const float4 v = make_float4(r.x*g.x*rs[c], r.y*g.y*rs[c], r.z*g.z*rs[c], r.w*g.w*rs[c]);
            *(float4 *) (xn + t*sxn_t + c*sxn_c + i) = v;
            m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    }
#pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    }
    if (threadIdx.x % WARP_SIZE == 0) {
        s_max[warp] = m;
    }
    __syncthreads();
    m = 0.0f;
#pragma unroll
    for (int w = 0; w < FN_HC_TPB/WARP_SIZE; ++w) {
        m = fmaxf(m, s_max[w]);
    }
    const float s = m > 0.0f ? gain/m : 0.0f;
    half * Xt = X + (int64_t) t*FN_HC*n_embd;
#pragma unroll
    for (int c = 0; c < FN_HC; ++c) {
        for (int i = 4*threadIdx.x; i < n_embd; i += 4*FN_HC_TPB) {
            const float4 r = *(const float4 *) (Rt + c*sr_c + i);
            const float4 g = *(const float4 *) (wn + (int64_t) c*n_embd + i);
            const float  f = rs[c]*s;
            *(half2 *) (Xt + (int64_t) c*n_embd + i)     = __floats2half2_rn(r.x*g.x*f, r.y*g.y*f);
            *(half2 *) (Xt + (int64_t) c*n_embd + i + 2) = __floats2half2_rn(r.z*g.z*f, r.w*g.w*f);
        }
    }
    if (threadIdx.x == 0) {
        xscale[t] = m > 0.0f ? m/gain : 0.0f;
    }
}

// up: mixed[t][d] = scale * sum_c xn[t][c][d] * sigmoid(w_up[c*n_embd + d] . lo[t]).
// The dense kernel with the rows of a tile taken from all four streams: row (g, r) of a tile is stream g % 4 of
// column d = 8*tile + 2*r + g/4, so the block that computes a column's four gates also sums them.
// tpr = hc_lr/16 threads per row, 8 rows per step, R = 4 steps per tile.
#define FN_HC_UP_G   8
#define FN_HC_UP_R   4
#define FN_HC_UP_TPR 20 // hc_lr = 320

template <int T, int TPR, int BPS>
static __global__ void __launch_bounds__(FN_HC_UP_G*TPR, BPS)
fn_hc_up(const uint4 * __restrict__ W, const half * __restrict__ D, const float * __restrict__ rowscale,
         const half2 * __restrict__ X, const float * __restrict__ xscale, const int n_embd,
         const float scale, const float * __restrict__ xn, const int sxn_c, const int sxn_t,
         float * __restrict__ mixed, const int smx_t) {
#if defined(FP16_AVAILABLE)
    constexpr int R   = FN_HC_UP_R;
    constexpr int G   = FN_HC_UP_G;
    constexpr int NT  = G*TPR;
    constexpr int NPG = R*T/2;
    __shared__ int   s_part[NPG][NT];
    __shared__ float s_term[G][R*T];
    const int tid = threadIdx.x;
    const int g   = tid/TPR;
    const int k   = tid - g*TPR;
    const int c   = g & 3;
    const int dd  = g >> 2;
    const half2 magic = __float2half2_rn(1152.0f);

    half2 a[T][8];
    fn_load_act<T>(X, TPR, k, a);

    // the gathering threads: pair p of row group gg (stream gg % 4 of the column step gg / 4)
    const bool gather = tid < G*NPG;
    const int  gg     = tid/NPG;
    const int  p      = tid - gg*NPG;
    const int  gc     = gg & 3;
    const int  vd0    = 2*((2*p)/T) + (gg >> 2);      // column within the tile, token of the pair's two values
    const int  vd1    = 2*((2*p + 1)/T) + (gg >> 2);
    const int  vt0    = (2*p) % T;
    const int  vt1    = (2*p + 1) % T;
    float xs0 = 0.0f, xs1 = 0.0f;
    if (gather) {
        xs0 = xscale[vt0];
        xs1 = xscale[vt1];
    }

    const int ntiles = n_embd/(2*R);
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int     row0 = c*n_embd + tile*(2*R) + dd;
        const uint4 * wr   = W + (size_t) row0*TPR + k;
        const half *  dr   = D + (size_t) row0*(TPR/2) + (k >> 1);
        uint4 v[R];
        half  d[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            v[r] = __ldg(wr + r*(2*TPR));
            d[r] = dr[r*TPR];
        }
        // what the gathering threads need after the barrier is loaded with the weights
        float rs0 = 0.0f, rs1 = 0.0f, x0 = 0.0f, x1 = 0.0f;
        if (gather) {
            rs0 = rowscale[gc*n_embd + tile*(2*R) + vd0];
            rs1 = rowscale[gc*n_embd + tile*(2*R) + vd1];
            x0  = xn[vt0*sxn_t + gc*sxn_c + tile*(2*R) + vd0];
            x1  = xn[vt1*sxn_t + gc*sxn_c + tile*(2*R) + vd1];
        }
        half s[R*T];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            fn_dot16<T>(v[r], d[r], magic, a, s + r*T);
        }
#pragma unroll
        for (int p = 0; p < NPG; ++p) {
            s_part[p][tid] = fn_i2(__halves2half2(s[2*p], s[2*p + 1]));
        }
        __syncthreads();
        if (gather) {
            const float2 f = fn_gather<TPR>(&s_part[p][gg*TPR]);
            s_term[gg][2*p]     = x0/(1.0f + expf(-f.x*rs0*xs0));
            s_term[gg][2*p + 1] = x1/(1.0f + expf(-f.y*rs1*xs1));
        }
        __syncthreads();
        if (tid < 2*R*T) {
            const int hi = tid/(R*T);  // which of the two columns of a step
            const int vi = tid - hi*(R*T);
            float m = 0.0f;
#pragma unroll
            for (int cc = 0; cc < FN_HC; ++cc) {
                m += s_term[hi*4 + cc][vi];
            }
            mixed[(vi % T)*smx_t + tile*(2*R) + 2*(vi/T) + hi] = scale*m;
        }
        __syncthreads();
    }
#else
    GGML_UNUSED_VARS(W, D, rowscale, X, xscale, n_embd, scale, xn, sxn_c, sxn_t, mixed, smx_t);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

#ifndef FN_STANDALONE

bool ggml_cuda_fn_hc_mix_supported(const ggml_cuda_hc_mix_args & a) {
    const ggml_tensor * R  = a.rms->src[0];
    const ggml_tensor * wd = a.mm_down->src[0];
    const ggml_tensor * wu = a.mm_up->src[0];
    const int64_t n_embd = R->ne[0];
    const int64_t hc_lr  = wd->ne[1];
    return ggml_cuda_fn_planar(wd) && ggml_cuda_fn_planar(wu) && R->ne[1] == FN_HC && R->ne[2] <= FN_MAX_T &&
           n_embd % (2*FN_HC_UP_R) == 0 && n_embd % 4 == 0 && fn_dense_geometry((int) (FN_HC*n_embd/16)) &&
           hc_lr == 16*FN_HC_UP_TPR && hc_lr % 8 == 0;
}

void ggml_cuda_fn_hc_mix(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & a) {
    const ggml_tensor * R  = a.rms->src[0];
    const ggml_tensor * wn = a.mul->src[1];
    const ggml_tensor * wd = a.mm_down->src[0];
    const ggml_tensor * wu = a.mm_up->src[0];
    const int     n_embd = (int) R->ne[0];
    const int     nt     = (int) R->ne[2];
    const int     hc_lr  = (int) wd->ne[1];
    const int     hd     = FN_HC*n_embd;
    const int64_t sr_c   = R->nb[1]/sizeof(float);
    const int64_t sr_t   = R->nb[2]/sizeof(float);
    const int64_t sxn_c  = a.mul->nb[1]/sizeof(float);
    const int64_t sxn_t  = a.mul->nb[2]/sizeof(float);
    const int64_t smx_t  = a.pre->nb[1]/sizeof(float);
    const int     nsm    = ggml_cuda_info().devices[ctx.device].nsm;
    cudaStream_t  stream = ctx.stream();

    ggml_cuda_fn_plane pd, pu;
    GGML_ASSERT(ggml_cuda_fn_planar(wd, &pd) && ggml_cuda_fn_planar(wu, &pu));

    ggml_cuda_pool_alloc<half>  X  (ctx.pool(), (size_t) nt*hd);
    ggml_cuda_pool_alloc<float> xs (ctx.pool(), nt);
    ggml_cuda_pool_alloc<float> lo (ctx.pool(), (size_t) nt*hc_lr);
    ggml_cuda_pool_alloc<half>  Xl (ctx.pool(), (size_t) nt*hc_lr);
    ggml_cuda_pool_alloc<float> xsl(ctx.pool(), nt);

    fn_hc_norm<<<nt, FN_HC_TPB, 0, stream>>>((const float *) R->data, sr_c, sr_t, (const float *) wn->data, n_embd,
        ggml_get_op_params_f32(a.rms, 0), (float *) a.mul->data, sxn_c, sxn_t, X.get(), xs.get(), fn_gain(hd));

    fn_dense_launch(nt, (const uint4 *) wd->data, (const half *) ((const char *) wd->data + (int64_t) hc_lr*hd),
        pd.rowscale, (const half2 *) X.get(), xs.get(), lo.get(), hc_lr, hc_lr, hd/16, nsm, FN_EPI_SILU,
        ggml_get_op_params_f32(a.scale, 0), ggml_get_op_params_f32(a.scale, 1), stream);

    fn_act_h16<<<nt, FN_ACT_TPB, 0, stream>>>(lo.get(), hc_lr, hc_lr, fn_gain(hc_lr), Xl.get(), xsl.get());

    const uint4 * Wu    = (const uint4 *) wu->data;
    const half *  Du    = (const half *) ((const char *) wu->data + (int64_t) hd*hc_lr);
    const float   scale = ggml_get_op_params_f32(a.pre, 0);
    const int     ntiles = n_embd/(2*FN_HC_UP_R);
    switch (nt) {
#define FN_CASE(N, BPS)                                                                                               \
        case N:                                                                                                       \
            fn_hc_up<N, FN_HC_UP_TPR, BPS><<<std::min(ntiles, nsm*BPS), FN_HC_UP_G*FN_HC_UP_TPR, 0, stream>>>(        \
                Wu, Du, pu.rowscale, (const half2 *) Xl.get(), xsl.get(), n_embd, scale,                              \
                (const float *) a.mul->data, (int) sxn_c, (int) sxn_t, (float *) a.pre->data, (int) smx_t);           \
            break;
        FN_CASE(1, 8)
        FN_CASE(2, 8)
        FN_CASE(3, 5)
        FN_CASE(4, 5)
        FN_CASE(5, 4)
        FN_CASE(6, 4)
        FN_CASE(7, 3)
        FN_CASE(8, 3)
#undef FN_CASE
        default:
            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // FN_STANDALONE
