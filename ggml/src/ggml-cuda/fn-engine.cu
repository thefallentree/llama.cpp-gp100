#include "fn-engine.cuh"
#include "convert.cuh"

#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>

#define FN_QK 32 // weights per Q8_0 block

// Geometry by row width (TPR = columns/16): rows of up to 2560 columns in blocks of 160 threads, several rows per
// block when they are narrower; 3072 and 6144 columns in blocks of 192 and 384 threads. Rows of 10240 and 5120
// columns are read as four or two parts of 2560 (S = 4 / 2, see fn_dense_p8).
// R rows per tile: 8, or 4 where 8 rows of T tokens exceed the registers, the 48 KB of shared memory of a block or
// the R*T/2 gathering threads a row has. BPS: blocks per SM; minBlocksPerMultiprocessor caps the registers so that
// nsm*BPS blocks are resident together, and is chosen so that no instantiation spills under that cap
// (cuobjdump --dump-resource-usage). A tight cap without a spill still costs: T = 3 in 64 registers (BPS 6) was
// 11-19% slower than in 96 (BPS 4).
//                     T  R20 B20 R40 B40 R160 B160 R192 B192 R384 B384
#define FN_DENSE_TABLE(X)                                   \
                    X(1,  8,  7,  8,  7,  8,   8,   8,   6,   8,   3)  \
                    X(2,  8,  6,  8,  6,  8,   6,   8,   5,   8,   2)  \
                    X(3,  8,  5,  8,  5,  8,   4,   8,   4,   8,   2)  \
                    X(4,  8,  4,  8,  4,  8,   4,   8,   3,   8,   2)  \
                    X(5,  8,  4,  8,  4,  8,   4,   8,   3,   8,   1)  \
                    X(6,  4,  4,  8,  3,  8,   3,   8,   2,   4,   1)  \
                    X(7,  4,  3,  4,  3,  4,   3,   4,   2,   4,   1)  \
                    X(8,  4,  3,  4,  3,  4,   3,   4,   2,   4,   1)

struct fn_dense_geom {
    int tpr; // threads per row (of a part)
    int s;   // parts of a row
    int g;   // row groups of a block
    int r;   // rows per group and tile
    int bps;
};

// the row widths the mat-vec is instantiated for
static bool fn_dense_geometry(const int64_t cols, fn_dense_geom * gm = nullptr, const int nt = 1) {
    const int s   = cols/16 == 640 ? 4 : cols/16 == 320 ? 2 : 1; // rows of 10240 or 5120 columns: parts of 2560
    const int tpr = (int) (cols/(16*s));
    if (cols % (16*s) != 0 || (tpr != 20 && tpr != 40 && tpr != 160 && tpr != 192 && tpr != 384)) {
        return false;
    }
    if (gm != nullptr) {
        gm->tpr = tpr;
        gm->s   = s;
        gm->g   = tpr == 20 ? 8 : tpr == 40 ? 4 : 1;
        switch (nt) {
#define FN_ROW(N, R20, B20, R40, B40, R160, B160, R192, B192, R384, B384)                                                  \
            case N:                                                                                                   \
                gm->r   = tpr == 20 ? R20 : tpr == 40 ? R40 : tpr == 160 ? R160 : tpr == 192 ? R192 : R384;           \
                gm->bps = tpr == 20 ? B20 : tpr == 40 ? B40 : tpr == 160 ? B160 : tpr == 192 ? B192 : B384;           \
                break;
            FN_DENSE_TABLE(FN_ROW)
#undef FN_ROW
            default:
                return false;
        }
    }
    return true;
}

// the largest tile of a row width: the rows of a matrix must be a multiple of it
static int fn_dense_tile_rows(const int64_t cols) {
    return cols == 320 ? 64 : cols == 640 ? 32 : 8;
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

// A small F32 matrix (the router of a MoE layer) gets planar Q8 copies of its own, by the data of the tensor: codes,
// then the scale plane. Its mat-vec then is two segments of the launch that reads the same vector: the copy and the
// copy of what the first one left out (with 8 bits alone the routing changed enough to cost 0.0009 of KLD).
struct fn_copy_entry {
    char *  w         = nullptr;
    float * rowscale  = nullptr;
    char *  w2        = nullptr;
    float * rowscale2 = nullptr;
    int     device    = 0;
};
static std::unordered_map<const void *, fn_copy_entry> g_fn_copies;
static std::atomic<int>                                 g_fn_n_copies { 0 };

// The copies of a device are carved from a few large allocations: a device allocation of their own size (1.4 MB)
// took 2 MB each. They are freed when the device has no copy left.
#define FN_COPY_CHUNK ((size_t) 32 << 20)
struct fn_copy_arena {
    std::vector<char *> chunks;
    size_t              used = FN_COPY_CHUNK; // of the last chunk
};
static fn_copy_arena g_fn_copy_arena[GGML_CUDA_MAX_DEVICES];

// under g_fn_mutex; null if the device has no memory left
static char * fn_copy_alloc(const int device, size_t size) {
    fn_copy_arena & a = g_fn_copy_arena[device];
    size = (size + 255) & ~(size_t) 255;
    GGML_ASSERT(size <= FN_COPY_CHUNK);
    if (a.used + size > FN_COPY_CHUNK) {
        char * chunk = nullptr;
        if (cudaMalloc((void **) &chunk, FN_COPY_CHUNK) != cudaSuccess) {
            (void) cudaGetLastError();
            return nullptr;
        }
        a.chunks.push_back(chunk);
        a.used = 0;
    }
    char * p = a.chunks.back() + a.used;
    a.used += size;
    return p;
}

static bool fn_router_enabled() {
    static const bool enabled = getenv("GGML_CUDA_FN_ROUTER") == nullptr || atoi(getenv("GGML_CUDA_FN_ROUTER")) != 0;
    return enabled;
}

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

// the planar weights that a MUL_MAT with these weights reads: the tensor itself (Q8_0, repacked) or its copy (F32)
static bool fn_plane_of(const ggml_tensor * t, const char ** w, ggml_cuda_fn_plane * plane) {
    if (ggml_cuda_fn_planar(t, plane)) {
        *w = (const char *) t->data;
        return true;
    }
    if (t->type != GGML_TYPE_F32 || t->data == nullptr || g_fn_n_copies.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_fn_mutex);
    const auto it = g_fn_copies.find(t->data);
    if (it == g_fn_copies.end()) {
        return false;
    }
    *w              = it->second.w;
    plane->rowscale = it->second.rowscale;
    return true;
}

// the second copy of a router: its output is added to the first one's
static bool fn_plane_rest(const ggml_tensor * t, const char ** w, ggml_cuda_fn_plane * plane) {
    if (t->type != GGML_TYPE_F32 || t->data == nullptr || g_fn_n_copies.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_fn_mutex);
    const auto it = g_fn_copies.find(t->data);
    if (it == g_fn_copies.end() || it->second.w2 == nullptr) {
        return false;
    }
    *w              = it->second.w2;
    plane->rowscale = it->second.rowscale2;
    return true;
}

void ggml_cuda_fn_planes_release(const void * base, const size_t size) {
    if (g_fn_n_planes.load(std::memory_order_relaxed) == 0 && g_fn_n_copies.load(std::memory_order_relaxed) == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_fn_mutex);
    bool erased = false;
    for (auto it = g_fn_copies.begin(); it != g_fn_copies.end();) {
        const char * p = (const char *) it->first;
        if (p >= (const char *) base && p < (const char *) base + size) {
            it     = g_fn_copies.erase(it);
            erased = true;
        } else {
            ++it;
        }
    }
    g_fn_n_copies.store((int) g_fn_copies.size(), std::memory_order_relaxed);
    for (int dev = 0; dev < GGML_CUDA_MAX_DEVICES && erased; ++dev) {
        bool live = false;
        for (const auto & kv : g_fn_copies) {
            live = live || kv.second.device == dev;
        }
        fn_copy_arena & a = g_fn_copy_arena[dev];
        if (!live && !a.chunks.empty()) {
            ggml_cuda_set_device(dev);
            for (char * chunk : a.chunks) {
                cudaFree(chunk);
            }
            a.chunks.clear();
            a.used = FN_COPY_CHUNK;
        }
    }
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

// The repack works in the tensor's own memory with a small temporary: the tensors of a model that fills the device
// leave no room for a copy of the largest of them.

// the block scales of `src` (Q8_0 blocks) to `d`, one thread per block
static __global__ void fn_p8_scales(const char * __restrict__ src, const int64_t nblk, half * __restrict__ d) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nblk) {
        d[i] = *(const half *) (src + i*(FN_QK + 2));
    }
}

// the largest block scale of each row; one warp per row
static __global__ void fn_p8_rowmax(const half * __restrict__ d, const int nb, float * __restrict__ rowscale) {
    const half * row = d + (size_t) blockIdx.x*nb;
    float m = 0.0f;
    for (int b = threadIdx.x; b < nb; b += WARP_SIZE) {
        m = fmaxf(m, fabsf(__half2float(row[b])));
    }
#pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    }
    if (threadIdx.x == 0) {
        rowscale[blockIdx.x] = m;
    }
}

// one thread per Q8_0 block of `src` (a copy of n blocks of the tensor): its quants go to the code plane
static __global__ void fn_p8_repack(const char * __restrict__ src, const int64_t n, char * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const uint16_t * q   = (const uint16_t *) (src + i*(FN_QK + 2) + 2); // the blocks are 2-byte aligned
    uint32_t *       out = (uint32_t *) (dst + i*FN_QK);
#pragma unroll
    for (int k = 0; k < FN_QK/4; ++k) {
        out[k] = ((uint32_t) q[2*k] | ((uint32_t) q[2*k + 1] << 16)) ^ 0x80808080u;
    }
}

// the scale plane: the block scales relative to their row's
static __global__ void fn_p8_repack_scales(const half * __restrict__ d, const int nb, const int64_t nblk,
                                           const float * __restrict__ rowscale, half * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nblk) {
        return;
    }
    const float rs = rowscale[i / nb];
    dst[i] = __float2half(rs > 0.0f ? __half2float(d[i])/rs : 0.0f);
}

// F32 rows to the planar layout, one thread per block of FN_QK weights: its scale as Q8_0 has it (max/127 as fp16)
// and the codes round(w/scale) + 128. With `prev` (the planar copy of the same rows, nb blocks per row) the rows are
// what that copy left out.
static __global__ void fn_f32_p8(const float * __restrict__ w, const int64_t nblk, half * __restrict__ d, char * __restrict__ codes,
                                 const char * __restrict__ prev, const float * __restrict__ prev_rowscale, const int nb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nblk) {
        return;
    }
    float x[FN_QK];
    for (int k = 0; k < FN_QK; ++k) {
        x[k] = w[i*FN_QK + k];
    }
    if (prev != nullptr) {
        const float s = __half2float(((const half *) (prev + nblk*FN_QK))[i])*prev_rowscale[i / nb];
        for (int k = 0; k < FN_QK; ++k) {
            x[k] -= s*(float) ((int) (unsigned char) prev[i*FN_QK + k] - 128);
        }
    }
    float amax = 0.0f;
    for (int k = 0; k < FN_QK; ++k) {
        amax = fmaxf(amax, fabsf(x[k]));
    }
    const half  dh = __float2half(amax/127.0f);
    const float df = __half2float(dh);
    const float id = df > 0.0f ? 1.0f/df : 0.0f;
    d[i] = dh;
    uint32_t * out = (uint32_t *) (codes + i*FN_QK);
    for (int k = 0; k < FN_QK/4; ++k) {
        uint32_t v = 0;
        for (int j = 0; j < 4; ++j) {
            const int q = max(-127, min(127, (int) roundf(x[4*k + j]*id)));
            v |= (uint32_t) ((q + 128) & 0xff) << (8*j);
        }
        out[k] = v;
    }
}

// Node i is the MUL_MAT of a MoE router: a small F32 matrix whose logits go to the softmax of the routing. Its
// neighbours may be the MUL_MATs that ggml_cuda_fn_reorder moved behind it.
static bool fn_router_like(const ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (!fn_router_enabled() || node->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];
    fn_dense_geom gm;
    if (w->type != GGML_TYPE_F32 || w->ne[2] != 1 || w->ne[3] != 1 || w->ne[1] > 1024 || w->ne[0] % FN_QK != 0 ||
        !fn_dense_geometry(w->ne[0], &gm) || gm.s != 1 || w->ne[1] % fn_dense_tile_rows(w->ne[0]) != 0 ||
        x->type != GGML_TYPE_F32 || x->ne[1] > FN_MAX_T || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    // behind it at most the other segments of its launch
    for (int j = i + 1; j < std::min(cgraph->n_nodes, i + 5); ++j) {
        if (cgraph->nodes[j]->op == GGML_OP_SOFT_MAX && cgraph->nodes[j]->src[0] == node) {
            return true;
        }
    }
    return false;
}

// the planar copies of the routers that the graph reads on this device
static void fn_copies_optimize(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    bool         synced = false;
    cudaStream_t stream = ctx.stream();
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (!fn_router_like(cgraph, i)) {
            continue;
        }
        const ggml_tensor * w = cgraph->nodes[i]->src[0];
        if (w->data == nullptr || w->buffer == nullptr || w->view_src != nullptr || !ggml_is_contiguous(w) ||
            ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            ggml_backend_buffer_get_type(w->buffer) != ggml_backend_cuda_buffer_type(ctx.device)) {
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(g_fn_mutex);
            if (g_fn_copies.count(w->data) != 0) {
                continue;
            }
        }
        if (!synced) {
            ggml_cuda_set_device(ctx.device);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            synced = true;
        }
        const int64_t rows = w->ne[1];
        const int     nb   = (int) (w->ne[0]/FN_QK);
        const int64_t nblk = rows*nb;
        fn_copy_entry e;
        e.device = ctx.device;
        half *       dtmp  = nullptr;
        const size_t nbyte = nblk*FN_QK + nblk*sizeof(half);
        {
            std::lock_guard<std::mutex> lock(g_fn_mutex);
            char * mem = fn_copy_alloc(ctx.device, 2*nbyte + 2*rows*sizeof(float));
            if (mem == nullptr) {
                return; // no memory left: the routers stay F32
            }
            e.w         = mem;
            e.w2        = mem + nbyte;
            e.rowscale  = (float *) (mem + 2*nbyte);
            e.rowscale2 = e.rowscale + rows;
        }
        CUDA_CHECK(cudaMalloc((void **) &dtmp, nblk*sizeof(half)));
        const unsigned grid = (unsigned) ((nblk + 255)/256);
        fn_f32_p8<<<grid, 256, 0, stream>>>((const float *) w->data, nblk, dtmp, e.w, nullptr, nullptr, nb);
        fn_p8_rowmax<<<(unsigned) rows, WARP_SIZE, 0, stream>>>(dtmp, nb, e.rowscale);
        fn_p8_repack_scales<<<grid, 256, 0, stream>>>(dtmp, nb, nblk, e.rowscale, (half *) (e.w + nblk*FN_QK));
        fn_f32_p8<<<grid, 256, 0, stream>>>((const float *) w->data, nblk, dtmp, e.w2, e.w, e.rowscale, nb);
        fn_p8_rowmax<<<(unsigned) rows, WARP_SIZE, 0, stream>>>(dtmp, nb, e.rowscale2);
        fn_p8_repack_scales<<<grid, 256, 0, stream>>>(dtmp, nb, nblk, e.rowscale2, (half *) (e.w2 + nblk*FN_QK));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(dtmp));

        std::lock_guard<std::mutex> lock(g_fn_mutex);
        g_fn_copies[w->data] = e;
        g_fn_n_copies.store((int) g_fn_copies.size(), std::memory_order_relaxed);
    }
}

static bool fn_planar_eligible(const ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = node->src[0];
    // only weights in this device's own memory (not host memory, not the tensors of a tensor-split meta device)
    if (w->type != GGML_TYPE_Q8_0 || w->data == nullptr || w->buffer == nullptr || w->view_src != nullptr ||
        ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        ggml_backend_buffer_get_type(w->buffer) != ggml_backend_cuda_buffer_type(ctx.device) ||
        !ggml_is_contiguous(w) || w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    const int64_t cols = w->ne[0];
    const int64_t rows = w->ne[1];
    // the row widths and tile heights the mat-vec is instantiated for
    if (cols % FN_QK != 0 || !fn_dense_geometry(cols) || rows % fn_dense_tile_rows(cols) != 0 || rows*cols >= ((int64_t) 1 << 31)) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_DP4A;
}

void ggml_cuda_fn_planes_optimize(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    if (!ggml_cuda_fn_enabled()) {
        return;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_DP4A) {
        fn_copies_optimize(ctx, cgraph);
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
        const int64_t nblk  = rows*nb;
        const int64_t chunk = std::min<int64_t>(nblk, 1 << 19); // blocks per pass
        char * tmp  = nullptr;
        half * dtmp = nullptr;
        CUDA_CHECK(cudaMalloc((void **) &e.plane.rowscale, rows*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **) &dtmp, nblk*sizeof(half)));
        CUDA_CHECK(cudaMalloc((void **) &tmp, chunk*(FN_QK + 2)));
        fn_p8_scales<<<(unsigned) ((nblk + 255)/256), 256, 0, stream>>>((const char *) w->data, nblk, dtmp);
        fn_p8_rowmax<<<(unsigned) rows, WARP_SIZE, 0, stream>>>(dtmp, nb, e.plane.rowscale);
        // the codes of a pass land on blocks that earlier passes consumed or that are in the copy
        for (int64_t b0 = 0; b0 < nblk; b0 += chunk) {
            const int64_t n = std::min(chunk, nblk - b0);
            CUDA_CHECK(cudaMemcpyAsync(tmp, (const char *) w->data + b0*(FN_QK + 2), n*(FN_QK + 2), cudaMemcpyDeviceToDevice, stream));
            fn_p8_repack<<<(unsigned) ((n + 255)/256), 256, 0, stream>>>(tmp, n, (char *) w->data + b0*FN_QK);
        }
        fn_p8_repack_scales<<<(unsigned) ((nblk + 255)/256), 256, 0, stream>>>(dtmp, nb, nblk, e.plane.rowscale,
                                                                               (half *) ((char *) w->data + nblk*FN_QK));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(tmp));
        CUDA_CHECK(cudaFree(dtmp));

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
// fp16, whatever the values. amax is the largest magnitude of the token's vector, or any bound of it: the values
// are floating point, a loose bound costs no precision.
// A vector is converted once, by the kernel that produces it where that kernel sees the whole vector (the norm of
// the hyper-connection read) or knows a bound (its gated sum), otherwise by fn_act_h16. Converting in the mat-vec
// itself costs more than it saves: each of its blocks would convert the whole vector again.
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

#ifndef FN_STANDALONE

// The fp16 activations of the vectors that mat-vecs read, keyed by the tensor (ggml_backend_cuda_context::fn_act).
// A slot is [FN_MAX_T*n halves][FN_MAX_T floats]. Slots are an optimization only: a mat-vec that finds none converts
// its input with fn_act_h16.
typedef ggml_backend_cuda_context::fn_act_slot fn_act_slot;

static half * fn_act_X(const fn_act_slot & s) {
    return (half *) s.mem;
}

static float * fn_act_xs(const fn_act_slot & s) {
    return (float *) (s.mem + (size_t) FN_MAX_T*s.n*sizeof(half));
}

static fn_act_slot * fn_act_find(ggml_backend_cuda_context & ctx, const ggml_tensor * key, const int n, const int nt) {
    for (auto & s : ctx.fn_act) {
        if (s.mem != nullptr && s.key == key && s.n == n && s.nt == nt) {
            s.last_use = ++ctx.fn_act_clock;
            return &s;
        }
    }
    return nullptr;
}

// the least recently used slot, for the producer of key's vector to fill; full: the activations will be there
// when the producer has run, otherwise only the scales (the activations follow, see fn_hc_up)
static fn_act_slot & fn_act_put(ggml_backend_cuda_context & ctx, const ggml_tensor * key, const int n, const int nt, const bool full) {
    fn_act_slot * s = &ctx.fn_act[0];
    for (auto & c : ctx.fn_act) {
        if (c.key == key) {
            s = &c;
            break;
        }
        if (c.last_use < s->last_use) {
            s = &c;
        }
    }
    const size_t need = (size_t) FN_MAX_T*n*sizeof(half) + FN_MAX_T*sizeof(float);
    if (need > s->cap) {
        ctx.retire_mem(s->mem, s->cap);
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc((void **) &s->mem, need));
        s->cap = need;
    }
    s->key      = key;
    s->n        = n;
    s->nt       = nt;
    s->full     = full;
    s->last_use = ++ctx.fn_act_clock;
    return *s;
}

// the activations of src1 = nt vectors of n floats at x, stride_t apart
static const fn_act_slot & fn_act_get(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const float * x,
                                      const int64_t stride_t, const int n, const int nt) {
    fn_act_slot * f = fn_act_find(ctx, src1, n, nt);
    if (f != nullptr && f->full) {
        return *f;
    }
    fn_act_slot & s = fn_act_put(ctx, src1, n, nt, true);
    fn_act_h16<<<nt, FN_ACT_TPB, 0, ctx.stream()>>>(x, stride_t, n, fn_gain(n), fn_act_X(s), fn_act_xs(s));
    CUDA_CHECK(cudaGetLastError());
    return s;
}

#endif // FN_STANDALONE

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

// The 16 columns of one thread for T tokens, as 8 half2 each; the tokens are xstride half2 apart. They are loaded
// as words: built from floats in registers, the compiler keeps the two halves of a pair apart and packs them again
// at every use, which takes twice the registers.
template <int T>
static __device__ __forceinline__ void fn_load_act(const half2 * __restrict__ X, const int xstride, const int k, half2 a[T][8]) {
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const int4 * ap = (const int4 *) (X + (int64_t) t*xstride + k*8);
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

// The sum of the n partial pairs of one row (n a multiple of 4). A loop: a few threads of the block run this while
// the others wait, and unrolled it is kilobytes of code that an SM fetches on every launch of the kernel (the fetch
// of code that only a warp or two execute is not hidden behind other warps: about 0.3 us per KB).
static __device__ __forceinline__ float2 fn_gather(const int * __restrict__ pp, const int n) {
    half2 c[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        c[j] = fn_h2(pp[j]);
    }
#pragma unroll 1
    for (int j = 4; j < n; j += 4) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            c[q] = __hadd2(c[q], fn_h2(pp[j + q]));
        }
    }
    return __half22float2(__hadd2(__hadd2(c[0], c[1]), __hadd2(c[2], c[3])));
}

// the sum of a few partial pairs
static __device__ __forceinline__ float2 fn_gather_few(const int * __restrict__ pp, const int n) {
    half2 c = fn_h2(pp[0]);
#pragma unroll
    for (int j = 1; j < n; ++j) {
        c = __hadd2(c, fn_h2(pp[j]));
    }
    return __half22float2(c);
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

// A launch computes up to FN_MAX_SEG mat-vecs of one geometry (segments), e.g. the projections that read the same
// vector: a launch has a fixed cost of several microseconds (its code and the activations reach every SM again),
// more than the mat-vec of a small matrix takes. Each block serves one segment.
#define FN_MAX_SEG 4

struct fn_dense_seg {
    const uint4 * W;          // the codes of the segment's part of the first row
    const half *  D;          // the block scales, likewise
    const float * rowscale;
    const half2 * X;          // the activations of the tokens, xstride half2 apart
    const float * xscale;     // [T]
    float *       dst;        // [T][dst_stride]
    int           xstride;
    int           dst_stride;
    int           ntiles;
    int           block_end;  // the segment's blocks: from the block_end of the segment before to this one
};

struct fn_dense_job {
    fn_dense_seg seg[FN_MAX_SEG];
};

// y[t][row] = sum_c W[row][c] x[t][c], for T tokens.
//   block: G rows x TPR threads; thread (g, k) owns columns [16k, 16k + 16) of the rows g, g + G, ... of a tile
//   tile:  G*R rows; block b of a segment's n blocks loops over its tiles b, b + n, ...
//   S:     the rows of W are S parts of 16*TPR columns wide and the segment is one of the parts (S = 1: the row)
// A thread's R*T partial sums go to shared memory two per word; thread (g, p) then sums pair p over row g's threads.
// The geometry is compile-time: with it as kernel arguments the index arithmetic (XMAD chains for every load and
// store) doubled the instruction count and the kernel ran at half the speed.
template <int T, int R, int TPR, int G, int BPS, int S>
static __global__ void __launch_bounds__(G*TPR, BPS)
fn_dense_p8(const fn_dense_job job) {
#if defined(FP16_AVAILABLE)
    constexpr int  NT   = G*TPR;
    constexpr int  NPG  = R*T/2;
    // A row group of whole warps: the warps reduce their partial sums with shuffles and thread (0, p) sums pair p
    // over the warps. Otherwise the threads store their partial sums and thread (g, p) sums pair p over the threads
    // of row group g.
    constexpr bool WRED = G == 1 && TPR % WARP_SIZE == 0;
    constexpr int  NSP  = WRED ? TPR/WARP_SIZE : NT;
    // The rows are an odd number of words long: shared memory has 32 four-byte banks, and the gathering threads of
    // a warp, which read the same position of consecutive rows, would otherwise all hit one bank.
    __shared__ int s_part[NPG][NSP | 1];
    const int tid = threadIdx.x;
    const int g   = tid/TPR;
    const int k   = tid - g*TPR;
    const half2 magic = __float2half2_rn(1152.0f);

    int si = 0, b0 = 0;
#pragma unroll
    for (int j = 0; j + 1 < FN_MAX_SEG; ++j) {
        if ((int) blockIdx.x >= job.seg[j].block_end) {
            si = j + 1;
            b0 = job.seg[j].block_end;
        }
    }
    const int nb     = job.seg[si].block_end - b0;
    const int ntiles = job.seg[si].ntiles;
    // The segment's pointers are read from the kernel arguments where they are used, like the arguments of a kernel
    // without segments. Its index is read from memory for that: the pointers of a segment the compiler knows are
    // loop invariants, which it keeps in registers that the tile loop does not have.
    volatile int si_mem = si;

    half2 a[T][8];
    fn_load_act<T>(job.seg[si].X, job.seg[si].xstride, k, a);

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
        xs0 = job.seg[si].xscale[vt0];
        xs1 = job.seg[si].xscale[vt1];
    }

    for (int tile = blockIdx.x - b0; tile < ntiles; tile += nb) {
        const fn_dense_seg & sg = job.seg[si_mem];
        // row (r, g) of the tile is tile*G*R + r*G + g: consecutive rows for consecutive g
        const uint4 * wr = sg.W + (size_t) tile*(R*NT*S) + g*(TPR*S) + k;
        const half *  dr = sg.D + (size_t) tile*(R*NT*S/2) + g*(TPR*S/2) + (k >> 1);
        uint4 v[R];
        half  d[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            v[r] = __ldg(wr + r*(NT*S));
            d[r] = dr[r*(NT*S/2)];
        }
        // the row scales are loaded with the weights: a load issued by the gathering threads after the barrier
        // would make the whole block wait for it
        float rs0 = 0.0f, rs1 = 0.0f;
        if (gather) {
            rs0 = sg.rowscale[tile*(G*R) + vr0];
            rs1 = sg.rowscale[tile*(G*R) + vr1];
        }
        // in two halves of the rows: the partial sums of all rows next to the words of the rows still to come would
        // not fit the registers
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            half s[(R/2)*T];
#pragma unroll
            for (int r = 0; r < R/2; ++r) {
                fn_dot16<T>(v[h*(R/2) + r], d[h*(R/2) + r], magic, a, s + r*T);
            }
            int pr[NPG/2];
#pragma unroll
            for (int q = 0; q < NPG/2; ++q) {
                pr[q] = fn_i2(__halves2half2(s[2*q], s[2*q + 1]));
            }
            if (WRED) {
                // A serial sum over the row's threads would be several hundred instructions of one warp, which has
                // to compete for them with the warps of the other blocks of the SM while the rest of its block waits.
#pragma unroll
                for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
#pragma unroll
                    for (int q = 0; q < NPG/2; ++q) {
                        pr[q] = fn_i2(__hadd2(fn_h2(pr[q]), fn_h2(__shfl_xor_sync(0xffffffff, pr[q], off))));
                    }
                }
                if (tid % WARP_SIZE == 0) {
#pragma unroll
                    for (int q = 0; q < NPG/2; ++q) {
                        s_part[h*(NPG/2) + q][tid/WARP_SIZE] = pr[q];
                    }
                }
            } else {
#pragma unroll
                for (int q = 0; q < NPG/2; ++q) {
                    s_part[h*(NPG/2) + q][tid] = pr[q];
                }
            }
        }
        __syncthreads();
        if (gather) {
            const fn_dense_seg & so = job.seg[si_mem];
            const float2 f = WRED ? fn_gather_few(&s_part[p][0], NSP) : fn_gather(&s_part[p][gg*TPR], TPR);
            so.dst[vt0*so.dst_stride + tile*(G*R) + vr0] = f.x*rs0*xs0;
            so.dst[vt1*so.dst_stride + tile*(G*R) + vr1] = f.y*rs1*xs1;
        }
        __syncthreads();
    }
#else
    GGML_UNUSED_VARS(job);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

static void fn_dense_launch(const int nt, const fn_dense_geom & gm, const fn_dense_job & job, const int grid, cudaStream_t stream) {
    GGML_ASSERT(gm.s == 1 || gm.tpr == 160);
    switch (nt) {
#define FN_ROW(N, R20, B20, R40, B40, R160, B160, R192, B192, R384, B384)                                                  \
        case N:                                                                                                       \
            switch (gm.tpr) {                                                                                         \
                case  20: fn_dense_p8<N, R20,   20, 8, B20,  1><<<grid, 8*20, 0, stream>>>(job); break;               \
                case  40: fn_dense_p8<N, R40,   40, 4, B40,  1><<<grid, 4*40, 0, stream>>>(job); break;               \
                case 160:                                                                                             \
                    if (gm.s == 1) {                                                                                  \
                        fn_dense_p8<N, R160, 160, 1, B160, 1><<<grid, 160, 0, stream>>>(job);                         \
                    } else if (gm.s == 2) {                                                                           \
                        fn_dense_p8<N, R160, 160, 1, B160, 2><<<grid, 160, 0, stream>>>(job);                         \
                    } else {                                                                                          \
                        fn_dense_p8<N, R160, 160, 1, B160, 4><<<grid, 160, 0, stream>>>(job);                         \
                    }                                                                                                 \
                    break;                                                                                            \
                case 192: fn_dense_p8<N, R192, 192, 1, B192, 1><<<grid, 192, 0, stream>>>(job); break;                \
                default:  fn_dense_p8<N, R384, 384, 1, B384, 1><<<grid, 384, 0, stream>>>(job); break;                \
            }                                                                                                         \
            break;
        FN_DENSE_TABLE(FN_ROW)
#undef FN_ROW
        default:
            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

// one mat-vec of a launch
struct fn_dense_part {
    const char *  w;          // planar weights [rows x cols]
    int64_t       rows, cols;
    const float * rowscale;
    int           part;       // which of the row's parts (S > 1)
    const half *  X;          // activations: [T][xstride halves]
    int           xstride;
    const float * xscale;
    float *       dst;
    int           dst_stride;
};

// n mat-vecs of the same row width in one launch
static void fn_dense(const int nt, const fn_dense_part * parts, const int n, const int nsm, cudaStream_t stream) {
    GGML_ASSERT(n >= 1 && n <= FN_MAX_SEG);
    fn_dense_geom gm;
    GGML_ASSERT(fn_dense_geometry(parts[0].cols, &gm, nt));
    const int tile_rows = gm.g*gm.r;
    int ntiles[FN_MAX_SEG];
    int total = 0;
    for (int i = 0; i < n; ++i) {
        GGML_ASSERT(parts[i].cols == parts[0].cols && parts[i].rows % tile_rows == 0 && parts[i].part < gm.s);
        ntiles[i] = (int) (parts[i].rows/tile_rows);
        total    += ntiles[i];
    }
    // the blocks of a segment by its share of the tiles, at least one
#ifdef FN_STANDALONE
    extern int g_fn_grid_max;
    const int grid = std::max(n, std::min(total, g_fn_grid_max > 0 ? g_fn_grid_max : nsm*gm.bps));
#else
    const int grid = std::max(n, std::min(total, nsm*gm.bps));
#endif
    int nb[FN_MAX_SEG];
    int used = 0;
    for (int i = 0; i < n; ++i) {
        nb[i] = std::min(ntiles[i], std::max(1, (int) ((int64_t) grid*ntiles[i]/total)));
        used += nb[i];
    }
    while (used != grid) {
        // one block more where a block has the most tiles, one less where it has the fewest
        int best = -1;
        for (int i = 0; i < n; ++i) {
            if (used < grid ? nb[i] >= ntiles[i] : nb[i] <= 1) {
                continue;
            }
            if (best < 0 || (used < grid ? (int64_t) ntiles[i]*nb[best] > (int64_t) ntiles[best]*nb[i]
                                         : (int64_t) ntiles[i]*nb[best] < (int64_t) ntiles[best]*nb[i])) {
                best = i;
            }
        }
        GGML_ASSERT(best >= 0);
        nb[best] += used < grid ? 1 : -1;
        used     += used < grid ? 1 : -1;
    }
    fn_dense_job job;
    int end = 0;
    for (int i = 0; i < FN_MAX_SEG; ++i) {
        fn_dense_seg & sg = job.seg[i];
        if (i >= n) {
            sg = job.seg[n - 1];
            sg.ntiles    = 0;
            sg.block_end = grid;
            continue;
        }
        const fn_dense_part & pt = parts[i];
        end += nb[i];
        sg.W          = (const uint4 *) pt.w + (size_t) pt.part*gm.tpr;
        sg.D          = (const half *) (pt.w + pt.rows*pt.cols) + (size_t) pt.part*(gm.tpr/2);
        sg.rowscale   = pt.rowscale;
        sg.X          = (const half2 *) pt.X;
        sg.xscale     = pt.xscale;
        sg.dst        = pt.dst;
        sg.xstride    = pt.xstride/2;
        sg.dst_stride = pt.dst_stride;
        sg.ntiles     = ntiles[i];
        sg.block_end  = end;
    }
    fn_dense_launch(nt, gm, job, grid, stream);
}

// out[t][i] = epilogue(sum over the np parts of parts[p][t][i]), one block per token: as floats, and as fp16
// activations with their scale if X is not null.
static __global__ void fn_sum_parts(const float * __restrict__ parts, const int np, const int n, const int epi, const float es,
                                    const float eb, float * __restrict__ dst, const int dst_stride, const float gain,
                                    half * __restrict__ X, float * __restrict__ xscale) {
    __shared__ float s_red[FN_ACT_TPB/WARP_SIZE];
    const int t  = blockIdx.x;
    const int nt = gridDim.x;
    float m = 0.0f;
    for (int i = threadIdx.x; i < n; i += FN_ACT_TPB) {
        float v = 0.0f;
        for (int q = 0; q < np; ++q) {
            v += parts[((int64_t) q*nt + t)*n + i];
        }
        v = fn_epilogue(v, epi, es, eb);
        dst[t*dst_stride + i] = v;
        m = fmaxf(m, fabsf(v));
    }
    if (X == nullptr) {
        return;
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
    const float sc = m > 0.0f ? gain/m : 0.0f;
    for (int i = threadIdx.x; i < n; i += FN_ACT_TPB) {
        X[(int64_t) t*n + i] = __float2half_rn(dst[t*dst_stride + i]*sc);
    }
    if (threadIdx.x == 0) {
        xscale[t] = m > 0.0f ? m/gain : 0.0f;
    }
}

#ifndef FN_STANDALONE

// ---------------------------------------------------------------------------------------------------------------
// MUL_MAT

// codes: the planar weights where they are a copy (fn_plane_of), the tensor's own data otherwise
static fn_dense_part fn_dense_part_of(const ggml_tensor * w, const ggml_cuda_fn_plane & plane, const int part,
                                      const fn_act_slot & x, float * dst, const int64_t dst_stride, const char * codes = nullptr) {
    fn_dense_part pt;
    pt.w          = codes != nullptr ? codes : (const char *) w->data;
    pt.rows       = w->ne[1];
    pt.cols       = w->ne[0];
    pt.rowscale   = plane.rowscale;
    pt.part       = part;
    pt.X          = fn_act_X(x);
    pt.xstride    = x.n;
    pt.xscale     = fn_act_xs(x);
    pt.dst        = dst;
    pt.dst_stride = (int) dst_stride;
    return pt;
}

// The mat-vec of a matrix whose rows are several parts wide: one segment per part, then the sum of their outputs.
// x: activations of the nt vectors of all columns. Xo, xso (may be null): the output as activations.
static void fn_dense_wide(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const fn_act_slot & x, const int nt,
                          const int epi, const float es, const float eb, float * dst, const int64_t dst_stride,
                          half * Xo, float * xso) {
    ggml_cuda_fn_plane plane;
    fn_dense_geom      gm;
    GGML_ASSERT(ggml_cuda_fn_planar(w, &plane) && fn_dense_geometry(w->ne[0], &gm, nt) && gm.s <= FN_MAX_SEG);
    const int rows = (int) w->ne[1];
    GGML_ASSERT(x.n == w->ne[0]);
    ggml_cuda_pool_alloc<float> parts(ctx.pool(), (size_t) gm.s*nt*rows);
    fn_dense_part pt[FN_MAX_SEG];
    for (int c = 0; c < gm.s; ++c) {
        pt[c]   = fn_dense_part_of(w, plane, c, x, parts.get() + (size_t) c*nt*rows, rows);
        pt[c].X = fn_act_X(x) + (size_t) c*(x.n/gm.s);
    }
    fn_dense(nt, pt, gm.s, ggml_cuda_info().devices[ctx.device].nsm, ctx.stream());
    fn_sum_parts<<<nt, FN_ACT_TPB, 0, ctx.stream()>>>(parts.get(), gm.s, rows, epi, es, eb, dst, (int) dst_stride,
                                                     fn_gain(rows), Xo, xso);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_fn_mul_mat_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    fn_dense_geom      gm;
    ggml_cuda_fn_plane plane;
    const char *       codes = nullptr;
    if (!fn_plane_of(src0, &codes, &plane)) {
        return false;
    }
    // a copy is for decode windows only: a wider batch reads the F32 weights
    if (src0->type == GGML_TYPE_F32 && src1->ne[1] > FN_MAX_T) {
        return false;
    }
    // a batch of vectors beyond ne[1] (the hyper-connection streams of a token) is one 2D batch when both sides are
    // contiguous: the weights are not batched, so every vector multiplies the same matrix
    const int64_t n_vec = src1->ne[1]*src1->ne[2]*src1->ne[3];
    const bool    flat  = src1->ne[2] == 1 && src1->ne[3] == 1 ? true : ggml_is_contiguous(src1) && src0->ne[2] == 1 && src0->ne[3] == 1;
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           n_vec <= 4*FN_MAX_T && flat && dst->ne[2] == src1->ne[2] && dst->ne[3] == src1->ne[3] &&
           src1->nb[0] == sizeof(float) && src1->ne[0] == src0->ne[0] && ggml_is_contiguous(dst) &&
           fn_dense_geometry(src0->ne[0], &gm);
}

// The segment of a router's second copy, if node is a router's MUL_MAT. Its output goes to a buffer of the context
// that the routing kernel adds to the node's (ggml_cuda_fn_router_rest); returns the number of segments (0 or 1).
static int fn_router_rest(ggml_backend_cuda_context & ctx, const ggml_tensor * node, const fn_act_slot & x, fn_dense_part * pt) {
    ggml_cuda_fn_plane plane;
    const char *       codes = nullptr;
    if (!fn_plane_rest(node->src[0], &codes, &plane)) {
        return 0;
    }
    const int64_t rows = node->src[0]->ne[1];
    if (ctx.fn_router_mem == nullptr || ctx.fn_router_rows < rows) {
        ggml_cuda_set_device(ctx.device);
        if (ctx.fn_router_mem != nullptr) {
            ctx.retired_mem.push_back(ctx.fn_router_mem); // captured graphs may still write it
        }
        CUDA_CHECK(cudaMalloc((void **) &ctx.fn_router_mem, (size_t) FN_MAX_T*rows*sizeof(float)));
        ctx.fn_router_rows = rows;
    }
    *pt = fn_dense_part_of(node->src[0], plane, 0, x, ctx.fn_router_mem, rows, codes);
    ctx.fn_router_node = node;
    return 1;
}

const float * ggml_cuda_fn_router_rest(ggml_backend_cuda_context & ctx, const ggml_tensor * logits) {
    return ctx.fn_router_node == logits ? ctx.fn_router_mem : nullptr;
}

void ggml_cuda_fn_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_fn_plane plane;
    fn_dense_geom      gm;
    const char *       codes = nullptr;
    GGML_ASSERT(fn_plane_of(src0, &codes, &plane) && fn_dense_geometry(src0->ne[0], &gm));
    const int     cols  = (int) src0->ne[0];
    const int     nsm   = ggml_cuda_info().devices[ctx.device].nsm;
    const int64_t s1    = src1->nb[1]/sizeof(float);
    const int64_t sd    = dst->nb[1]/sizeof(float);
    const int64_t n_vec = src1->ne[1]*src1->ne[2]*src1->ne[3]; // contiguous beyond ne[1], see ggml_cuda_fn_mul_mat_supported

    if (n_vec <= FN_MAX_T) {
        const int    nt = (int) n_vec;
        const auto & x  = fn_act_get(ctx, src1, (const float *) src1->data, s1, cols, nt);
        if (gm.s > 1) {
            fn_dense_wide(ctx, src0, x, nt, FN_EPI_NONE, 0.0f, 0.0f, (float *) dst->data, sd, nullptr, nullptr);
            return;
        }
        fn_dense_part pt[2] = { fn_dense_part_of(src0, plane, 0, x, (float *) dst->data, sd, codes) };
        fn_dense(nt, pt, 1 + fn_router_rest(ctx, dst, x, pt + 1), nsm, ctx.stream());
        return;
    }
    // wider batches in passes; their activations are not kept
    fn_act_slot x;
    ggml_cuda_pool_alloc<char> xm(ctx.pool(), (size_t) FN_MAX_T*cols*sizeof(half) + FN_MAX_T*sizeof(float));
    x.mem = xm.get();
    x.n   = cols;
    for (int64_t c0 = 0; c0 < n_vec; c0 += FN_MAX_T) {
        const int nt = (int) std::min<int64_t>(FN_MAX_T, n_vec - c0);
        fn_act_h16<<<nt, FN_ACT_TPB, 0, ctx.stream()>>>((const float *) src1->data + c0*s1, s1, cols, fn_gain(cols), fn_act_X(x), fn_act_xs(x));
        if (gm.s > 1) {
            fn_dense_wide(ctx, src0, x, nt, FN_EPI_NONE, 0.0f, 0.0f, (float *) dst->data + c0*sd, sd, nullptr, nullptr);
            continue;
        }
        const fn_dense_part pt = fn_dense_part_of(src0, plane, 0, x, (float *) dst->data + c0*sd, sd);
        fn_dense(nt, &pt, 1, nsm, ctx.stream());
    }
}

// Consecutive MUL_MAT nodes from node i that read the same vector with planar weights of the same row width: they
// are one launch. Returns their number, 0 if there are less than two.
int ggml_cuda_fn_mul_mat_run_match(const ggml_cgraph * cgraph, const int i) {
    if (!ggml_cuda_fn_enabled()) {
        return 0;
    }
    const ggml_tensor * first = cgraph->nodes[i];
    int n     = 0;
    int n_seg = 0; // a router is two
    while (i + n < cgraph->n_nodes) {
        const ggml_tensor * node = cgraph->nodes[i + n];
        fn_dense_geom      gm;
        ggml_cuda_fn_plane plane;
        const char *       codes = nullptr;
        if (node->op != GGML_OP_MUL_MAT || node->src[1] != first->src[1] || node->src[0]->ne[0] != first->src[0]->ne[0] ||
            node->src[1]->ne[1] > FN_MAX_T || node->flags != first->flags || ggml_is_empty(node) ||
            !ggml_cuda_fn_mul_mat_supported(node->src[0], node->src[1], node) ||
            !fn_dense_geometry(node->src[0]->ne[0], &gm) || gm.s != 1) {
            break;
        }
        const int segs = fn_plane_rest(node->src[0], &codes, &plane) ? 2 : 1;
        if (n_seg + segs > FN_MAX_SEG) {
            break;
        }
        n_seg += segs;
        ++n;
    }
    return n >= 2 ? n : 0;
}

void ggml_cuda_fn_mul_mat_run(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, const int n) {
    const ggml_tensor * src1 = nodes[0]->src[1];
    const int    nt = (int) src1->ne[1];
    const auto & x  = fn_act_get(ctx, src1, (const float *) src1->data, src1->nb[1]/sizeof(float), (int) src1->ne[0], nt);
    fn_dense_part pt[FN_MAX_SEG + 1];
    int n_seg = 0;
    for (int j = 0; j < n; ++j) {
        ggml_cuda_fn_plane plane;
        const char *       codes = nullptr;
        GGML_ASSERT(fn_plane_of(nodes[j]->src[0], &codes, &plane) && n_seg < FN_MAX_SEG);
        pt[n_seg++] = fn_dense_part_of(nodes[j]->src[0], plane, 0, x, (float *) nodes[j]->data, nodes[j]->nb[1]/sizeof(float), codes);
        n_seg += fn_router_rest(ctx, nodes[j], x, pt + n_seg);
    }
    GGML_ASSERT(n_seg <= FN_MAX_SEG);
    fn_dense(nt, pt, n_seg, ggml_cuda_info().devices[ctx.device].nsm, ctx.stream());
}

// graph_optimize: the MUL_MAT nodes that read the same vector become neighbours (a later one moves up behind the
// first: all it needs is that vector), so that ggml_cuda_fn_mul_mat_run_match finds them.
void ggml_cuda_fn_reorder(ggml_cgraph * cgraph) {
    if (!ggml_cuda_fn_enabled()) {
        return;
    }
    const int max_dist = 96;
    auto candidate = [](const ggml_tensor * node) {
        const ggml_tensor * w = node->src[0];
        const ggml_tensor * x = node->src[1];
        return node->op == GGML_OP_MUL_MAT && w->type == GGML_TYPE_Q8_0 && w->ne[2] == 1 && w->ne[3] == 1 &&
               x->type == GGML_TYPE_F32 && x->ne[1] <= FN_MAX_T && x->ne[2] == 1 && x->ne[3] == 1 &&
               fn_dense_geometry(w->ne[0]) && w->ne[0]/16 != 640;
    };
    // A router takes the shared expert's gate and up into its launch. Not in a prompt graph, where only the last
    // layer is decode-sized: its order would differ from the graph the allocation was reserved with.
    bool routers = true;
    for (int i = 0; i < cgraph->n_nodes && routers; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        routers = node->op != GGML_OP_MUL_MAT_ID || node->src[2]->ne[1] <= FN_MAX_T;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (!candidate(cgraph->nodes[i]) && !(routers && fn_router_like(cgraph, i))) {
            continue;
        }
        const ggml_tensor * first = cgraph->nodes[i];
        int n = 1;
        while (i + n < cgraph->n_nodes && n < FN_MAX_SEG && candidate(cgraph->nodes[i + n]) &&
               cgraph->nodes[i + n]->src[1] == first->src[1] && cgraph->nodes[i + n]->src[0]->ne[0] == first->src[0]->ne[0]) {
            ++n;
        }
        for (int j = i + n; j < cgraph->n_nodes && j <= i + max_dist && n < FN_MAX_SEG; ++j) {
            ggml_tensor * node = cgraph->nodes[j];
            if (!candidate(node) || node->src[1] != first->src[1] || node->src[0]->ne[0] != first->src[0]->ne[0]) {
                continue;
            }
            for (int q = j; q > i + n; --q) {
                cgraph->nodes[q] = cgraph->nodes[q - 1];
            }
            cgraph->nodes[i + n] = node;
            ++n;
        }
        i += n - 1;
    }
}

#endif // FN_STANDALONE

// ---------------------------------------------------------------------------------------------------------------
// the hyper-connection read (see hc-mix.cuh for the math), planar weights
//   norm: one block per token; thread i owns the values [4i, 4i + 4) of the four streams, in registers. It also
//         writes the fp16 activations of xn for the down mat-vec and the scale of the read's output for the up kernel
//   down: the dense mat-vec, a segment per stream (a block then loads the activations of one stream, not of all
//         four: they were as many bytes as the weights), and the sum of the four with the SiLU, as activations
//   up:   the dense mat-vec with the rows of a tile taken from all four streams, so that the block that computes a
//         column's four gates also sums them; it writes the output as fp32 and as fp16 activations

#define FN_HC         4    // streams
#define FN_HCN_MAX_NW 32   // warps of a norm block: n_embd <= 4096

// what the norm writes besides xn (any of them may be null)
struct fn_hc_norm_out {
    half *  X;          // fp16 activations of xn: [nt][hc*n_embd]
    float * xs;         // their scales: [nt]
    float * xs_mixed;   // the scales of the read's output: [nt]
    float   mix_scale;  // |scale of the read| / gain of its output
};

// Block-wide reductions of N values per thread, for blocks of up to FN_HCN_MAX_NW warps: the warps reduce, the
// first warp reduces their results, one thread applies `fin` to each total and every thread reads the results.
//   - a thread that summed the warp results itself would do one shared-memory load per warp and value: with 20
//     warps that kept the load/store units of the SM busy for most of the kernel;
//   - what follows a reduction is mostly divisions, square roots and exponentials of the totals, which the SM
//     computes on 16 units: done by every thread they cost more than the reduction.
// s_w: [N][FN_HCN_MAX_NW], s_tot: [N]; two barriers.
template <int N, bool is_max, typename F>
static __device__ __forceinline__ void fn_block_reduce(float v[N], float (* s_w)[FN_HCN_MAX_NW], float * s_tot, const F fin) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int nw   = blockDim.x / WARP_SIZE;
#pragma unroll
    for (int c = 0; c < N; ++c) {
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            const float o = __shfl_xor_sync(0xffffffff, v[c], off);
            v[c] = is_max ? fmaxf(v[c], o) : v[c] + o;
        }
        if (lane == 0) {
            s_w[c][warp] = v[c];
        }
    }
    __syncthreads();
    if (warp == 0) {
        float x[N];
#pragma unroll
        for (int c = 0; c < N; ++c) {
            x[c] = lane < nw ? s_w[c][lane] : 0.0f; // the maxima are of magnitudes
#pragma unroll
            for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                const float o = __shfl_xor_sync(0xffffffff, x[c], off);
                x[c] = is_max ? fmaxf(x[c], o) : x[c] + o;
            }
        }
        if (lane == 0) {
            fin(x);
#pragma unroll
            for (int c = 0; c < N; ++c) {
                s_tot[c] = x[c];
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int c = 0; c < N; ++c) {
        v[c] = s_tot[c];
    }
}

// r[c]: this thread's four values of stream c, g[c]: their norm weights (the caller loads them with its other
// inputs: a load issued after the barriers below would add its latency to the kernel).
// xn = r*w_norm*rsqrt(mean(r^2) + eps) over each stream of the token.
static __device__ __forceinline__ void fn_hc_norm_part(const float4 r[FN_HC], const float4 g[FN_HC], const int n_embd,
                                                       const float eps, float * xn_t, const int64_t sxn_c,
                                                       const fn_hc_norm_out o, const int t,
                                                       float (* s_w)[FN_HCN_MAX_NW], float * s_tot) {
    const int i = threadIdx.x;
    float ss[FN_HC];
#pragma unroll
    for (int c = 0; c < FN_HC; ++c) {
        ss[c] = r[c].x*r[c].x + r[c].y*r[c].y + r[c].z*r[c].z + r[c].w*r[c].w;
    }
    // the totals become the factors rsqrt(mean + eps)
    fn_block_reduce<FN_HC, false>(ss, s_w, s_tot, [=](float * x) {
#pragma unroll
        for (int c = 0; c < FN_HC; ++c) {
            x[c] = rsqrtf(x[c]/n_embd + eps);
        }
    });
    float4 v[FN_HC];
    float  mx[2] = { 0.0f, 0.0f };                       // max |xn|, max over d of sum over the streams of |xn|
    float4 b     = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int c = 0; c < FN_HC; ++c) {
        v[c] = make_float4(r[c].x*g[c].x*ss[c], r[c].y*g[c].y*ss[c], r[c].z*g[c].z*ss[c], r[c].w*g[c].w*ss[c]);
        *(float4 *) (xn_t + c*sxn_c + 4*i) = v[c];
        b = make_float4(b.x + fabsf(v[c].x), b.y + fabsf(v[c].y), b.z + fabsf(v[c].z), b.w + fabsf(v[c].w));
        mx[0] = fmaxf(mx[0], fmaxf(fmaxf(fabsf(v[c].x), fabsf(v[c].y)), fmaxf(fabsf(v[c].z), fabsf(v[c].w))));
    }
    if (o.xs == nullptr) {
        return;
    }
    mx[1] = fmaxf(fmaxf(b.x, b.y), fmaxf(b.z, b.w));
    // The down mat-vec reads the four streams of a token as one vector of hc*n_embd values. The maximum becomes the
    // factor of its activations; the scales are written by the one thread that has the totals.
    const float gain = 400.0f/(FN_HC*n_embd);
    fn_block_reduce<2, true>(mx, s_w, s_tot, [=](float * x) {
        o.xs[t] = x[0] > 0.0f ? x[0]/gain : 0.0f;
        if (o.xs_mixed != nullptr) {
            // |mixed[d]| <= |scale| * sum_c |xn[c][d]|: the gates are sigmoids
            o.xs_mixed[t] = x[1]*o.mix_scale;
        }
        x[0] = x[0] > 0.0f ? gain/x[0] : 0.0f;
    });
    const float sc = mx[0];
#pragma unroll
    for (int c = 0; c < FN_HC; ++c) {
        // 8-byte stores
        const half2 h0 = __floats2half2_rn(v[c].x*sc, v[c].y*sc);
        const half2 h1 = __floats2half2_rn(v[c].z*sc, v[c].w*sc);
        *(uint2 *) (o.X + ((int64_t) t*FN_HC + c)*n_embd + 4*i) = make_uint2((unsigned int) fn_i2(h0), (unsigned int) fn_i2(h1));
    }
}

static __global__ void fn_hc_norm(const float * __restrict__ R, const int64_t sr_c, const int64_t sr_t,
                                  const float * __restrict__ wn, const int n_embd, const float eps,
                                  float * __restrict__ xn, const int64_t sxn_c, const int64_t sxn_t, const fn_hc_norm_out o) {
    __shared__ float s_w[FN_HC][FN_HCN_MAX_NW];
    __shared__ float s_tot[FN_HC];
    const int t = blockIdx.x;
    float4 r[FN_HC], g[FN_HC];
#pragma unroll
    for (int c = 0; c < FN_HC; ++c) {
        r[c] = *(const float4 *) (R + t*sr_t + c*sr_c + 4*threadIdx.x);
        g[c] = *(const float4 *) (wn + (int64_t) c*n_embd + 4*threadIdx.x);
    }
    fn_hc_norm_part(r, g, n_embd, eps, xn + t*sxn_t, sxn_c, o, t, s_w, s_tot);
}

// ---------------------------------------------------------------------------------------------------------------
// The output of a recurrent (gated delta net) layer before its projection:
//   y = rms_norm(x)*w * gate(z), over the 128 values of each head; the gate is a SiLU or a sigmoid
// as floats and as the activations of the projection. One block per token, one warp per head: a lane owns four
// values, the warp sums the squares with shuffles.
#define FN_GDN_HEAD 128

static __device__ __forceinline__ float fn_gate(const float z, const bool silu) {
    const float s = 1.0f/(1.0f + expf(-z));
    return silu ? z*s : s;
}

static __global__ void fn_gdn_out(const float * __restrict__ x, const int64_t sx_t, const float * __restrict__ z, const int64_t sz_t,
                                  const float * __restrict__ w, const float eps, const bool silu, const float gain,
                                  float * __restrict__ y, const int64_t sy_t, half * __restrict__ X, float * __restrict__ xscale) {
    __shared__ float s_w[1][FN_HCN_MAX_NW];
    __shared__ float s_tot[1];
    const int t    = blockIdx.x;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n    = blockDim.x*4;
    const float4 x4 = *(const float4 *) (x + t*sx_t + 4*threadIdx.x);
    const float4 z4 = *(const float4 *) (z + t*sz_t + 4*threadIdx.x);
    const float4 w4 = *(const float4 *) (w + 4*lane);
    float ss = x4.x*x4.x + x4.y*x4.y + x4.z*x4.z + x4.w*x4.w;
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        ss += __shfl_xor_sync(0xffffffff, ss, off);
    }
    const float rs = rsqrtf(ss/FN_GDN_HEAD + eps);
    const float4 v = make_float4(x4.x*rs*w4.x*fn_gate(z4.x, silu), x4.y*rs*w4.y*fn_gate(z4.y, silu),
                                 x4.z*rs*w4.z*fn_gate(z4.z, silu), x4.w*rs*w4.w*fn_gate(z4.w, silu));
    *(float4 *) (y + t*sy_t + 4*threadIdx.x) = v;
    float m[1] = { fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))) };
    fn_block_reduce<1, true>(m, s_w, s_tot, [=](float * a) {
        xscale[t] = a[0] > 0.0f ? a[0]/gain : 0.0f;
        a[0] = a[0] > 0.0f ? gain/a[0] : 0.0f;
    });
    const float sc = m[0];
    const half2 h0 = __floats2half2_rn(v.x*sc, v.y*sc);
    const half2 h1 = __floats2half2_rn(v.z*sc, v.w*sc);
    *(uint2 *) (X + (int64_t) t*n + 4*threadIdx.x) = make_uint2((unsigned int) fn_i2(h0), (unsigned int) fn_i2(h1));
}

// The input side of a recurrent (gated delta net) layer, from its projections to the inputs of the delta rule:
//   conv:  y[t][c] = silu(sum_j in[t + j][c]*w[c][j]); in: the K - 1 columns of the layer's conv state, then x
//   q, k:  y*rsqrt(mean of the head's y^2 + eps)*norm_scale (the l2 norm over each head)
//   state: the last K - 1 columns of in, once per rollback slot (slot s ends s tokens earlier)
//   gate:  softplus(alpha + dt)*a
// One warp per head; a lane owns four channels, whose window of K - 1 columns it keeps in registers while it steps
// through the tokens. A lane reads its channels of the state before it stores them: the row of a slot may be the
// row that is read. The last block computes the gates.
#define FN_GDN_PRE_SLOTS 4

struct fn_gdn_pre_args {
    const float *   cache;    // the conv state: row rows[0] (crow floats per row), or the state itself if rows is null
    const int32_t * rows;
    int64_t         crow;
    float *         slot[FN_GDN_PRE_SLOTS]; // [C][K - 1] each
    const float *   x;        // [T][C], sx_t apart
    const float *   w;        // [C][K]
    float *         y;        // [T][C], sy_t apart
    float *         q;        // [T][n_head_k][128], sq_t apart
    float *         k;
    const float *   alpha;    // [T][n_gate], sa_t apart
    const float *   dt;       // [n_gate]
    const float *   a;        // [n_gate]
    float *         gate;     // [T][n_gate], sg_t apart
    int             sx_t, sy_t, sq_t, sa_t, sg_t;
    int             nt, n_slots, n_head_k, n_heads, n_gate;
    float           eps, norm_scale;
};

static __device__ __forceinline__ float4 fn_mul4(const float4 a, const float4 b) {
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

static __device__ __forceinline__ float4 fn_fma4(const float4 a, const float4 b, const float4 c) {
    return make_float4(a.x*b.x + c.x, a.y*b.y + c.y, a.z*b.z + c.z, a.w*b.w + c.w);
}

// the window of a lane's four channels as it is stored: three columns per channel
static __device__ __forceinline__ void fn_gdn_pre_store(float * row, const float4 c0, const float4 c1, const float4 c2) {
    *(float4 *) (row + 0) = make_float4(c0.x, c1.x, c2.x, c0.y);
    *(float4 *) (row + 4) = make_float4(c1.y, c2.y, c0.z, c1.z);
    *(float4 *) (row + 8) = make_float4(c2.z, c0.w, c1.w, c2.w);
}

static __global__ void fn_gdn_pre(const fn_gdn_pre_args a) {
    const int lane = threadIdx.x;
    const int h    = blockIdx.x;
    if (h == a.n_heads) {
        for (int e = lane; e < a.nt*a.n_gate; e += WARP_SIZE) {
            const int   t = e/a.n_gate;
            const int   g = e - t*a.n_gate;
            const float v = a.alpha[t*a.sa_t + g] + a.dt[g];
            a.gate[t*a.sg_t + g] = (v > 20.0f ? v : logf(1.0f + expf(v)))*a.a[g];
        }
        return;
    }
    const int     c0    = h*FN_GDN_HEAD + 4*lane;
    const float * state = a.rows != nullptr ? a.cache + (int64_t) a.rows[0]*a.crow : a.cache;
    float4 col0, col1, col2;
    {
        const float4 s0 = *(const float4 *) (state + 3*c0);
        const float4 s1 = *(const float4 *) (state + 3*c0 + 4);
        const float4 s2 = *(const float4 *) (state + 3*c0 + 8);
        col0 = make_float4(s0.x, s0.w, s1.z, s2.y);
        col1 = make_float4(s0.y, s1.x, s1.w, s2.z);
        col2 = make_float4(s0.z, s1.y, s2.x, s2.w);
    }
    float4 tap0, tap1, tap2, tap3;
    {
        const float4 w0 = *(const float4 *) (a.w + 4*c0);
        const float4 w1 = *(const float4 *) (a.w + 4*c0 + 4);
        const float4 w2 = *(const float4 *) (a.w + 4*c0 + 8);
        const float4 w3 = *(const float4 *) (a.w + 4*c0 + 12);
        tap0 = make_float4(w0.x, w1.x, w2.x, w3.x);
        tap1 = make_float4(w0.y, w1.y, w2.y, w3.y);
        tap2 = make_float4(w0.z, w1.z, w2.z, w3.z);
        tap3 = make_float4(w0.w, w1.w, w2.w, w3.w);
    }
    // the slots that end before the first token of this window keep the state that was read
    for (int sl = a.nt; sl < a.n_slots; ++sl) {
        fn_gdn_pre_store(a.slot[sl] + 3*c0, col0, col1, col2);
    }
    const bool is_qk = h < 2*a.n_head_k;
    float * qk = nullptr;
    if (is_qk) {
        qk = (h < a.n_head_k ? a.q + h*FN_GDN_HEAD : a.k + (h - a.n_head_k)*FN_GDN_HEAD) + 4*lane;
    }
    for (int t = 0; t < a.nt; ++t) {
        const float4 xt = *(const float4 *) (a.x + t*a.sx_t + c0);
        float4 v = fn_fma4(xt, tap3, fn_fma4(col2, tap2, fn_fma4(col1, tap1, fn_mul4(col0, tap0))));
        v = make_float4(v.x/(1.0f + expf(-v.x)), v.y/(1.0f + expf(-v.y)), v.z/(1.0f + expf(-v.z)), v.w/(1.0f + expf(-v.w)));
        col0 = col1;
        col1 = col2;
        col2 = xt;
        const int sl = a.nt - 1 - t;
        if (sl < a.n_slots) {
            fn_gdn_pre_store(a.slot[sl] + 3*c0, col0, col1, col2);
        }
        *(float4 *) (a.y + t*a.sy_t + c0) = v;
        if (is_qk) {
            float ss = v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
#pragma unroll
            for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
                ss += __shfl_xor_sync(0xffffffff, ss, o);
            }
            const float sc = rsqrtf(ss/FN_GDN_HEAD + a.eps)*a.norm_scale;
            *(float4 *) (qk + t*a.sq_t) = make_float4(v.x*sc, v.y*sc, v.z*sc, v.w*sc);
        }
    }
}

// The output of an attention layer before its projection: y = x*sigmoid(g), as floats and as the activations of the
// projection. g is a view: d values per head, the heads sg_h apart. One block per token, a thread owns four values.
static __global__ void fn_gate_out(const float * __restrict__ x, const int64_t sx_t, const float * __restrict__ g, const int64_t sg_h,
                                   const int64_t sg_t, const int d4, const float gain,
                                   float * __restrict__ y, const int64_t sy_t, half * __restrict__ X, float * __restrict__ xscale) {
    __shared__ float s_w[1][FN_HCN_MAX_NW];
    __shared__ float s_tot[1];
    const int t    = blockIdx.x;
    const int head = threadIdx.x/d4;
    const int n    = blockDim.x*4;
    const float4 x4 = *(const float4 *) (x + t*sx_t + 4*threadIdx.x);
    const float4 g4 = *(const float4 *) (g + t*sg_t + head*sg_h + 4*(threadIdx.x - head*d4));
    const float4 v  = make_float4(x4.x*fn_gate(g4.x, false), x4.y*fn_gate(g4.y, false), x4.z*fn_gate(g4.z, false), x4.w*fn_gate(g4.w, false));
    *(float4 *) (y + t*sy_t + 4*threadIdx.x) = v;
    float m[1] = { fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))) };
    fn_block_reduce<1, true>(m, s_w, s_tot, [=](float * a) {
        xscale[t] = a[0] > 0.0f ? a[0]/gain : 0.0f;
        a[0] = a[0] > 0.0f ? gain/a[0] : 0.0f;
    });
    const float sc = m[0];
    const half2 h0 = __floats2half2_rn(v.x*sc, v.y*sc);
    const half2 h1 = __floats2half2_rn(v.z*sc, v.w*sc);
    *(uint2 *) (X + (int64_t) t*n + 4*threadIdx.x) = make_uint2((unsigned int) fn_i2(h0), (unsigned int) fn_i2(h1));
}

// The selection mask of a QSA attention layer (the top blocks of the indexer plus the tail):
//   out[cell] = kq_mask[cell] if the cell is selected, -inf otherwise
// slot s of a token selects the cell sel_idx[s] if it is live: a slot of a top block if the block's score is not
// -inf, a tail slot if its cell is not the sentinel n_kv. A dead slot goes to its own row n_kv + s, beyond the mask.
// One block for all the tokens: the cells of the slots are staged in shared memory before the first store, because
// the output usually shares memory with the scores and the top blocks (their buffers are free by then).
#define FN_QSA_SEL_THREADS 1024
#define FN_QSA_SEL_DEAD    0xFFFFu

static __global__ void fn_qsa_sel(const int32_t * sel_idx, const int64_t s_sel, const int n_sel, const int n_top,
                                  const int kpool, const float * score, const int64_t s_score,
                                  const int32_t * top_k, const int64_t s_topk, const int n_kv, const int nt,
                                  const half * __restrict__ kq_mask, const int64_t s_kq, half * out, const int64_t s_out) {
    extern __shared__ unsigned short s_cell[];
    for (int e = threadIdx.x; e < nt*n_sel; e += blockDim.x) {
        const int t = e/n_sel;
        const int s = e - t*n_sel;
        const float idx = (float) sel_idx[t*s_sel + s];
        float live;
        if (s < n_top) {
            live = fminf(fmaxf(score[t*s_score + top_k[t*s_topk + s/kpool]] + 1.0f, 0.0f), 1.0f);
        } else {
            live = fminf(fmaxf((float) n_kv - idx, 0.0f), 1.0f);
        }
        const float dump = (float) (n_kv + s);
        const int   cell = (int) ((idx - dump)*live + dump);
        s_cell[e] = cell < n_kv ? (unsigned short) cell : (unsigned short) FN_QSA_SEL_DEAD;
    }
    __syncthreads();
    const half ninf = __ushort_as_half((unsigned short) 0xFC00);
    for (int e = threadIdx.x; e < nt*n_kv; e += blockDim.x) {
        const int t = e/n_kv;
        out[t*s_out + (e - t*n_kv)] = ninf;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < nt*n_sel; e += blockDim.x) {
        const int t = e/n_sel;
        const unsigned int cell = s_cell[e];
        if (cell != FN_QSA_SEL_DEAD) {
            out[t*s_out + cell] = kq_mask[t*s_kq + cell];
        }
    }
}

// block-wide maximum or sum of one value per thread, for blocks of up to 32 whole warps; two barriers
static __device__ __forceinline__ float fn_block_fold(float v, const bool is_max, float * s_w, float * s_tot) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int nw   = blockDim.x / WARP_SIZE;
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        const float o = __shfl_xor_sync(0xffffffff, v, off);
        v = is_max ? fmaxf(v, o) : v + o;
    }
    if (lane == 0) {
        s_w[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        float x = lane < nw ? s_w[lane] : (is_max ? -INFINITY : 0.0f);
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            const float o = __shfl_xor_sync(0xffffffff, x, off);
            x = is_max ? fmaxf(x, o) : x + o;
        }
        if (lane == 0) {
            s_tot[0] = x;
        }
    }
    __syncthreads();
    return s_tot[0];
}

// QSA attention over the selected cells only: the cells of fn_qsa_sel, then for a head's query q
//   s[j] = scale*(q . K[cell_j]) + kq_mask[cell_j],   out = sum over j of softmax(s)[j]*V[cell_j]
// The heads share the key and value rows (one KV head), so a block takes all NH heads of a token over a chunk of the
// slots and reads every row once. Scores: a lane per slot walks its key row against the queries in shared memory (no
// reductions). Values: a warp per slot and half the heads, a lane eight values. With several chunks a block leaves the
// softmax state and the unnormalized sum to part, and fn_qsa_attn_join adds the chunks up. The cost does not depend
// on the size of the cache. (A block per (head, token) read every row NH times: 137 us at 2051 slots.)
#define FN_QSA_ATTN_D         256
#define FN_QSA_ATTN_NW        8
#define FN_QSA_ATTN_CHUNK     64   // slots per block at most
#define FN_QSA_ATTN_MAX_CHUNK 40
#define FN_QSA_ATTN_MAX_HEAD  32   // heads per device of the partial state
#define FN_QSA_ATTN_MAX_SEL   (FN_QSA_ATTN_CHUNK*FN_QSA_ATTN_MAX_CHUNK) // slots: the cells of the indexer's top blocks (2048) and the tail

struct fn_qsa_attn_args {
    const int32_t * cells;    // [nt][n_sel] staged by fn_qsa_cells (-1: none)
    const int32_t * sel_idx;
    const float *   score;
    const int32_t * top_k;
    const half *    kq_mask;
    const float *   q;        // [nt][n_head][D], sq_t and sq_h apart
    const half *    K;        // rows of D values, sk apart
    const half *    V;
    float *         out;      // [nt][n_head][D], so_t and so_h apart
    float *         part;     // [nt][n_head][n_chunk][D + 4]: o, m, l
    int             s_sel, s_score, s_topk, s_kq, sq_t, sq_h, sk, sv, so_t, so_h;
    int             n_sel, n_top, kpool, n_kv, n_head, n_chunk, c_len;
    float           scale;
};

// the cell of slot j of token t, as fn_qsa_sel computes it; -1 if the slot is dead
static __device__ __forceinline__ int fn_qsa_cell(const int32_t * sel_idx, const int s_sel, const int n_top, const int kpool,
                                                  const float * score, const int s_score, const int32_t * top_k, const int s_topk,
                                                  const int n_kv, const int t, const int j) {
    const float idx = (float) sel_idx[t*s_sel + j];
    float live;
    if (j < n_top) {
        live = fminf(fmaxf(score[t*s_score + top_k[t*s_topk + j/kpool]] + 1.0f, 0.0f), 1.0f);
    } else {
        live = fminf(fmaxf((float) n_kv - idx, 0.0f), 1.0f);
    }
    const float dump = (float) (n_kv + j);
    const int   c    = (int) ((idx - dump)*live + dump);
    return c < n_kv ? c : -1;
}

// The cells of all slots: three dependent reads per slot, spread over the device. The attention's blocks could not read
// the selection anyway while other blocks write their output (they can share memory).
static __global__ void fn_qsa_cells(const fn_qsa_attn_args a, const int nt, int32_t * cells) {
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e < nt*a.n_sel) {
        const int t = e/a.n_sel;
        const int j = e - t*a.n_sel;
        cells[e] = fn_qsa_cell(a.sel_idx, a.s_sel, a.n_top, a.kpool, a.score, a.s_score, a.top_k, a.s_topk, a.n_kv, t, j);
    }
}

// block (token, chunk): heads h0 .. h0 + NH - 1 over the slots [chunk*c_len, +c_len)
#ifdef FN_QSA_MARKS
__device__ long long fn_qsa_marks[8];
#define FN_QSA_MARK(i) if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) { fn_qsa_marks[i] = clock64(); }
#else
#define FN_QSA_MARK(i)
#endif

template <int NH>
static __global__ void __launch_bounds__(FN_QSA_ATTN_NW*WARP_SIZE, 3)
fn_qsa_attn(const fn_qsa_attn_args a, const int h0) {
    FN_QSA_MARK(0)
    constexpr int D  = FN_QSA_ATTN_D;
    constexpr int NW = FN_QSA_ATTN_NW;
    constexpr int C  = FN_QSA_ATTN_CHUNK;
    constexpr int HW = NH/2;                 // heads per warp in the sum over the values
    __shared__ float s_q[NH][D];             // then the output
    __shared__ float s_part[NH][NW*WARP_SIZE]; // the warps' parts of 32 scores
    __shared__ float s_s[NH][C];             // scores, then the weights
    __shared__ int   s_cell[C];
    __shared__ float s_mk[C];                // the mask of each slot
    __shared__ float s_m[NH];
    __shared__ float s_l[NH];
    float (* const s_o)[D] = s_q;
    const int t     = blockIdx.x;
    const int chunk = blockIdx.y;
    const int lane  = threadIdx.x % WARP_SIZE;
    const int warp  = threadIdx.x / WARP_SIZE;
    const int c_lo  = chunk*a.c_len;
    const int n_c   = min(a.c_len, a.n_sel - c_lo);  // slots of this chunk
    const int32_t * cells = a.cells + (int64_t) t*a.n_sel + c_lo;
    const half *    mask  = a.kq_mask + (int64_t) t*a.s_kq;

    for (int e = threadIdx.x; e < NH*D/4; e += NW*WARP_SIZE) {
        const int h = e/(D/4);
        const int d = 4*(e - h*(D/4));
        *(float4 *) (s_q[h] + d) = *(const float4 *) (a.q + (int64_t) t*a.sq_t + (int64_t) (h0 + h)*a.sq_h + d);
    }
    __syncthreads();
    FN_QSA_MARK(1)
    // scores: 32 slots at a time, a lane per slot and a warp per 32 values of the key row, the queries read as
    // broadcasts; the warps' parts are added through shared memory
    for (int cg = 0; cg < a.c_len; cg += WARP_SIZE) {
        const int     c    = cg + lane;
        const int     cell = c < n_c ? cells[c] : -1;
        const uint4 * kr   = (const uint4 *) (a.K + (int64_t) (cell >= 0 ? cell : 0)*a.sk) + 4*warp;
        uint4 u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = kr[i];
        }
        if (warp == 0 && c < C) {
            s_cell[c] = cell >= 0 ? cell : 0; // a dead slot reads row 0 with the weight 0
            s_mk[c]   = cell >= 0 ? __half2float(mask[cell]) : -INFINITY;
        }
        float acc[NH];
#pragma unroll
        for (int h = 0; h < NH; ++h) {
            acc[h] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 k0 = __half22float2(fn_h2((int) u[i].x));
            const float2 k1 = __half22float2(fn_h2((int) u[i].y));
            const float2 k2 = __half22float2(fn_h2((int) u[i].z));
            const float2 k3 = __half22float2(fn_h2((int) u[i].w));
#pragma unroll
            for (int h = 0; h < NH; ++h) {
                const float4 q0 = *(const float4 *) (s_q[h] + 32*warp + 8*i);
                const float4 q1 = *(const float4 *) (s_q[h] + 32*warp + 8*i + 4);
                acc[h] += k0.x*q0.x + k0.y*q0.y + k1.x*q0.z + k1.y*q0.w + k2.x*q1.x + k2.y*q1.y + k3.x*q1.z + k3.y*q1.w;
            }
        }
#pragma unroll
        for (int h = 0; h < NH; ++h) {
            s_part[h][32*warp + lane] = acc[h];
        }
        __syncthreads();
        for (int e = threadIdx.x; e < NH*WARP_SIZE; e += NW*WARP_SIZE) {
            const int h  = e/WARP_SIZE;
            const int sl = e - h*WARP_SIZE;
            const int cs = cg + sl;
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < NW; ++w) {
                sum += s_part[h][32*w + sl];
            }
            if (cs < C) {
                const float m = cs < n_c ? s_mk[cs] : -INFINITY;
                s_s[h][cs] = m > -INFINITY ? sum*a.scale + m : -INFINITY;
            }
        }
        __syncthreads();
    }
    __syncthreads();
    FN_QSA_MARK(2)
    // softmax state per head: a warp per head
    for (int h = warp; h < NH; h += NW) {
        float mx = -INFINITY;
        for (int c = lane; c < n_c; c += WARP_SIZE) {
            mx = fmaxf(mx, s_s[h][c]);
        }
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
        }
        float l = 0.0f;
        for (int c = lane; c < n_c; c += WARP_SIZE) {
            const float e = s_s[h][c] > -INFINITY ? expf(s_s[h][c] - mx) : 0.0f;
            s_s[h][c] = e;
            l += e;
        }
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            l += __shfl_xor_sync(0xffffffff, l, off);
        }
        if (lane == 0) {
            s_m[h] = mx;
            s_l[h] = l;
        }
    }
    __syncthreads();
    FN_QSA_MARK(3)
    // the values: a warp per slot and half the heads
    const int hg = warp % 2;
    float acc[HW][8];
#pragma unroll
    for (int h = 0; h < HW; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[h][i] = 0.0f;
        }
    }
    for (int c0 = warp/2; c0 < n_c; c0 += 4*(NW/2)) {
        uint4 u[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int c = c0 + b*(NW/2);
            u[b] = *(const uint4 *) (a.V + (int64_t) s_cell[c < n_c ? c : 0]*a.sv + 8*lane);
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int c = c0 + b*(NW/2);
            if (c < n_c) {
                const float2 v0 = __half22float2(fn_h2((int) u[b].x));
                const float2 v1 = __half22float2(fn_h2((int) u[b].y));
                const float2 v2 = __half22float2(fn_h2((int) u[b].z));
                const float2 v3 = __half22float2(fn_h2((int) u[b].w));
#pragma unroll
                for (int h = 0; h < HW; ++h) {
                    const float p = s_s[hg*HW + h][c];
                    acc[h][0] += p*v0.x; acc[h][1] += p*v0.y; acc[h][2] += p*v1.x; acc[h][3] += p*v1.y;
                    acc[h][4] += p*v2.x; acc[h][5] += p*v2.y; acc[h][6] += p*v3.x; acc[h][7] += p*v3.y;
                }
            }
        }
    }
    FN_QSA_MARK(4)
    // the warps of a head group add up in turn (the queries are no longer needed)
    for (int w = 0; w < NW/2; ++w) {
        if (w == warp/2) {
#pragma unroll
            for (int h = 0; h < HW; ++h) {
                float4 * o0 = (float4 *) (s_o[hg*HW + h] + 8*lane);
                float4 * o1 = (float4 *) (s_o[hg*HW + h] + 8*lane + 4);
                if (w == 0) {
                    *o0 = make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
                    *o1 = make_float4(acc[h][4], acc[h][5], acc[h][6], acc[h][7]);
                } else {
                    const float4 p0 = *o0;
                    const float4 p1 = *o1;
                    *o0 = make_float4(p0.x + acc[h][0], p0.y + acc[h][1], p0.z + acc[h][2], p0.w + acc[h][3]);
                    *o1 = make_float4(p1.x + acc[h][4], p1.y + acc[h][5], p1.z + acc[h][6], p1.w + acc[h][7]);
                }
            }
        }
        __syncthreads();
    }
    FN_QSA_MARK(5)
    // out, or the partial state of the chunk
    for (int e = threadIdx.x; e < NH*D/4; e += NW*WARP_SIZE) {
        const int    h = e/(D/4);
        const int    d = 4*(e - h*(D/4));
        const float4 o = *(const float4 *) (s_o[h] + d);
        if (a.n_chunk == 1) {
            const float inv = s_l[h] > 0.0f ? 1.0f/s_l[h] : 0.0f;
            *(float4 *) (a.out + (int64_t) t*a.so_t + (int64_t) (h0 + h)*a.so_h + d) = make_float4(o.x*inv, o.y*inv, o.z*inv, o.w*inv);
        } else {
            float * p = a.part + ((int64_t) (t*a.n_head + h0 + h)*a.n_chunk + chunk)*(D + 4);
            *(float4 *) (p + d) = o;
            if (d == 0) {
                p[D]     = s_m[h];
                p[D + 1] = s_l[h];
            }
        }
    }
    FN_QSA_MARK(6)
}

// the chunks of a (head, token): out = sum over chunks of exp(m_c - m)*o_c / sum over chunks of exp(m_c - m)*l_c
static __global__ void fn_qsa_attn_join(const fn_qsa_attn_args a) {
    constexpr int D = FN_QSA_ATTN_D;
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const float * p0 = a.part + ((int64_t) (t*a.n_head + h)*a.n_chunk)*(D + 4);
    float m = -INFINITY;
    for (int c = 0; c < a.n_chunk; ++c) {
        m = fmaxf(m, p0[c*(D + 4) + D]);
    }
    float o = 0.0f, l = 0.0f;
    for (int c = 0; c < a.n_chunk; ++c) {
        const float * p = p0 + c*(D + 4);
        const float   w = p[D] > -INFINITY ? expf(p[D] - m) : 0.0f;
        o += w*p[threadIdx.x];
        l += w*p[D + 1];
    }
    a.out[(int64_t) t*a.so_t + (int64_t) h*a.so_h + threadIdx.x] = l > 0.0f ? o/l : 0.0f;
}

// all heads of a token over its chunks; a.c_len and a.n_chunk are set here
static void fn_qsa_attn_launch(fn_qsa_attn_args & a, const int nt, cudaStream_t stream) {
    // small selections in small chunks, so that the device is used
    a.c_len   = a.n_sel <= 512 ? 32 : FN_QSA_ATTN_CHUNK;
    a.n_chunk = (a.n_sel + a.c_len - 1)/a.c_len;
    const dim3 grid((unsigned) nt, (unsigned) a.n_chunk, 1);
    for (int h0 = 0; h0 < a.n_head; ) {
        const int nh = a.n_head - h0;
        if (nh >= 16) {
            fn_qsa_attn<16><<<grid, FN_QSA_ATTN_NW*WARP_SIZE, 0, stream>>>(a, h0);
            h0 += 16;
        } else if (nh >= 12) {
            fn_qsa_attn<12><<<grid, FN_QSA_ATTN_NW*WARP_SIZE, 0, stream>>>(a, h0);
            h0 += 12;
        } else if (nh >= 8) {
            fn_qsa_attn<8><<<grid, FN_QSA_ATTN_NW*WARP_SIZE, 0, stream>>>(a, h0);
            h0 += 8;
        } else {
            fn_qsa_attn<4><<<grid, FN_QSA_ATTN_NW*WARP_SIZE, 0, stream>>>(a, h0);
            h0 += 4;
        }
    }
    if (a.n_chunk > 1) {
        fn_qsa_attn_join<<<dim3((unsigned) a.n_head, (unsigned) nt, 1), FN_QSA_ATTN_D, 0, stream>>>(a);
    }
}

// The pooled indexer keys of a QSA attention layer (qwen4exp build_qsa_sel), one block:
//   the raw keys of the new tokens go to the first half of their cache rows (the second half is zeroed), then the
//   mean over the kpool member rows of each block to re-pool is taken from the cache.
static __global__ void fn_qsa_pool(const float * __restrict__ k_raw, const int64_t sk_t, const int d, const int nt,
                                   const int64_t * __restrict__ k_idxs, half * cache, const int64_t s_cell,
                                   const int32_t * __restrict__ pool_idx, const int kpool, const int n_new, const float scale,
                                   float * __restrict__ out, const int64_t so) {
    for (int e = threadIdx.x; e < nt*2*d; e += blockDim.x) {
        const int t = e/(2*d);
        const int c = e - t*(2*d);
        cache[k_idxs[t]*s_cell + c] = c < d ? __float2half(k_raw[t*sk_t + c]) : __float2half(0.0f);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_new*d; e += blockDim.x) {
        const int j = e/d;
        const int c = e - j*d;
        float acc = 0.0f;
        for (int m = 0; m < kpool; ++m) {
            acc += __half2float(cache[(int64_t) pool_idx[j*kpool + m]*s_cell + c]);
        }
        out[j*so + c] = acc*scale;
    }
}

// up: mixed[t][d] = scale * sum_c xn[t][c][d] * sigmoid(w_up[c*n_embd + d] . lo[t]).
// Row (g, r) of a tile is stream g % 4 of column d = 8*tile + 2*r + g/4.
// tpr = hc_lr/16 threads per row, 8 rows per step, R = 4 steps per tile.
// Xm (may be null): mixed as fp16 activations, with the scales xs_mixed that the norm derived from xn.
#define FN_HC_UP_G   8
#define FN_HC_UP_R   4
#define FN_HC_UP_TPR 20 // hc_lr = 320

template <int T, int TPR, int BPS>
static __global__ void __launch_bounds__(FN_HC_UP_G*TPR, BPS)
fn_hc_up(const uint4 * __restrict__ W, const half * __restrict__ D, const float * __restrict__ rowscale,
         const half2 * __restrict__ X, const float * __restrict__ xscale, const int n_embd,
         const float scale, const float * __restrict__ xn, const int sxn_c, const int sxn_t,
         float * __restrict__ mixed, const int smx_t, half * __restrict__ Xm, const float * __restrict__ xs_mixed) {
#if defined(FP16_AVAILABLE)
    constexpr int R   = FN_HC_UP_R;
    constexpr int G   = FN_HC_UP_G;
    constexpr int NT  = G*TPR;
    constexpr int NPG = R*T/2;
    __shared__ int   s_part[NPG][NT + 1]; // as in fn_dense_p8
    __shared__ float s_term[G][R*T];
    const int tid = threadIdx.x;
    const int g   = tid/TPR;
    const int k   = tid - g*TPR;
    const int c   = g & 3;
    const int dd  = g >> 2;
    const half2 magic = __float2half2_rn(1152.0f);

    half2 a[T][8];
    fn_load_act<T>(X, TPR*8, k, a);

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
    // the threads that sum a column's four terms: value vi = (column step, token) of column half hi
    const bool sums = tid < 2*R*T;
    const int  hi   = tid/(R*T);
    const int  vi   = tid - hi*(R*T);
    float mx = 0.0f;
    if (sums && Xm != nullptr) {
        const float xm = xs_mixed[vi % T];
        mx = xm > 0.0f ? 1.0f/xm : 0.0f;
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
            const float2 f = fn_gather(&s_part[p][gg*TPR], TPR);
            s_term[gg][2*p]     = x0/(1.0f + expf(-f.x*rs0*xs0));
            s_term[gg][2*p + 1] = x1/(1.0f + expf(-f.y*rs1*xs1));
        }
        __syncthreads();
        if (sums) {
            float m = 0.0f;
#pragma unroll
            for (int cc = 0; cc < FN_HC; ++cc) {
                m += s_term[hi*4 + cc][vi];
            }
            m *= scale;
            const int col = tile*(2*R) + 2*(vi/T) + hi;
            mixed[(vi % T)*smx_t + col] = m;
            if (Xm != nullptr) {
                Xm[(int64_t) (vi % T)*n_embd + col] = __float2half_rn(m*mx);
            }
        }
        __syncthreads();
    }
#else
    GGML_UNUSED_VARS(W, D, rowscale, X, xscale, n_embd, scale, xn, sxn_c, sxn_t, mixed, smx_t, Xm, xs_mixed);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// ---------------------------------------------------------------------------------------------------------------
// AllReduce of a projection's partial sums + hyper-connection write + the norm of the next read, one block per token
// (tensor-split windows, allreduce.cuh):
//   out      = y + y_peer [+ z]                (z: a term that every device has, e.g. the shared expert)
//   R'[c][d] = R[c][d] + out[d]*gate[c],   gate[c] = a.z*sigmoid(a.x*(w_inject[c] . xn_prev) + a.y) + a.w
//   xn       = rms_norm(R')*w_norm
// Thread i owns the values [4i, 4i + 4): it publishes its part of y to the peer first, computes its part of the
// inject dot products while the peer catches up, and keeps the 16 values of R' in registers for the norm.
// Every input is read before the first barrier and every output written after it, so the outputs may share memory
// with the inputs; R' and xn have to be distinct.
//
// The exchange: a thread's four values travel as one 16-byte store, 24 bits each plus the token of the reduction,
// and the peer's thread polls that unit until it carries the token. A 16-byte store arrives whole (4.4e9 units
// exchanged without a torn one), so a unit with the token is complete: no token ring and no
// __threadfence_system() between the data and a separate signal (that fence takes microseconds on GP100, three of
// them were most of this kernel). Both devices add the same two rounded values, so their results are identical.

// a float as its upper 24 bits, rounded to nearest
static __device__ __forceinline__ unsigned int fn_f24(const float x) {
    return (__float_as_uint(x) + 0x80u) >> 8;
}

static __device__ __forceinline__ float fn_f24_value(const unsigned int t) {
    return __uint_as_float(t << 8);
}

static __device__ __forceinline__ uint4 fn_ar_pack(const unsigned int t0, const unsigned int t1, const unsigned int t2,
                                                   const unsigned int t3, const int token) {
    return make_uint4(t0 | (t1 << 24), (t1 >> 8) | (t2 << 16), (t2 >> 16) | (t3 << 8), (unsigned int) token);
}

static __device__ __forceinline__ float4 fn_ar_unpack(const uint4 u) {
    return make_float4(fn_f24_value(u.x & 0xffffffu), fn_f24_value((u.x >> 24) | ((u.y & 0xffffu) << 8)),
                       fn_f24_value((u.y >> 16) | ((u.z & 0xffu) << 16)), fn_f24_value(u.z >> 8));
}

// four consecutive weights with one load
static __device__ __forceinline__ float4 fn_load4(const float * p) {
    return *(const float4 *) p;
}

static __device__ __forceinline__ float4 fn_load4(const nv_bfloat16 * p) {
    const uint2 v = *(const uint2 *) p; // a bfloat16 is the upper half of a float
    return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                       __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
}

static __device__ __forceinline__ float4 fn_load4(const half * p) {
    const uint2 v = *(const uint2 *) p;
    const float2 a = __half22float2(fn_h2((int) v.x));
    const float2 b = __half22float2(fn_h2((int) v.y));
    return make_float4(a.x, a.y, b.x, b.y);
}

template <typename T_inj>
static __global__ void fn_ar_hc(
        const float * y, const int64_t sy_t, const float * z, const int64_t sz_t,
        uint4 * wire_mine, const uint4 * wire_other, const unsigned int * epoch, const int site,
        const float * xn_prev, const int64_t sxp_c, const int64_t sxp_t,
        const T_inj * __restrict__ w_inject, const float4 act,
        const float * R, const int64_t sr_c, const int64_t sr_t,
        float * Rn, const int64_t sd_c, const int64_t sd_t,
        const float * __restrict__ wn, const int n_embd, const float eps,
        float * xn, const int64_t sxn_c, const int64_t sxn_t, const fn_hc_norm_out o, unsigned long long * dbg) {
    __shared__ float s_w[FN_HC][FN_HCN_MAX_NW];
    __shared__ float s_tot[FN_HC];
    const int t  = blockIdx.x;
    const int i  = threadIdx.x;
    const int n4 = n_embd/4;
    const int token = ggml_cuda_ar_window_token(epoch, site);

    float4 y4;
    {
        const float4 v = *(const float4 *) (y + t*sy_t + 4*i);
        const unsigned int t0 = fn_f24(v.x), t1 = fn_f24(v.y), t2 = fn_f24(v.z), t3 = fn_f24(v.w);
        wire_mine[t*n4 + i] = fn_ar_pack(t0, t1, t2, t3, token);
        y4 = make_float4(fn_f24_value(t0), fn_f24_value(t1), fn_f24_value(t2), fn_f24_value(t3));
    }
    // every input is loaded here, before the first barrier: one memory latency for the kernel instead of one per
    // phase, and the outputs may then share memory with any of them (see above)
    float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (z != nullptr) {
        z4 = *(const float4 *) (z + t*sz_t + 4*i);
    }
    float4 r[FN_HC], gn[FN_HC];
#pragma unroll
    for (int c = 0; c < FN_HC; ++c) {
        r[c]  = *(const float4 *) (R + t*sr_t + c*sr_c + 4*i);
        gn[c] = *(const float4 *) (wn + (int64_t) c*n_embd + 4*i);
    }

    // the inject gates of the four streams
    float gate[FN_HC];
    {
        float4 xp[FN_HC];
#pragma unroll
        for (int c = 0; c < FN_HC; ++c) {
            xp[c] = *(const float4 *) (xn_prev + t*sxp_t + c*sxp_c + 4*i);
        }
#pragma unroll
        for (int g = 0; g < FN_HC; ++g) {
            gate[g] = 0.0f;
#pragma unroll
            for (int c = 0; c < FN_HC; ++c) {
                const float4 w = fn_load4(w_inject + (int64_t) g*FN_HC*n_embd + (int64_t) c*n_embd + 4*i);
                gate[g] += w.x*xp[c].x + w.y*xp[c].y + w.z*xp[c].z + w.w*xp[c].w;
            }
        }
        fn_block_reduce<FN_HC, false>(gate, s_w, s_tot, [=](float * x) {
#pragma unroll
            for (int g = 0; g < FN_HC; ++g) {
                x[g] = act.z/(1.0f + expf(-(act.x*x[g] + act.y))) + act.w;
            }
        });
    }
    // The peer's unit for the same four values. The accesses are volatile: __ldcv() and inline PTX loads are
    // loop-invariant for the compiler, which then drops the loop.
    const volatile unsigned int * q = (const volatile unsigned int *) (wire_other + t*n4 + i);
    const long long t_poll = dbg != nullptr ? clock64() : 0;
    while ((int) q[3] != token) {
    }
    const uint4 u = make_uint4(q[0], q[1], q[2], 0);
    if (dbg != nullptr && i == 0 && t == 0) {
        const long long now = clock64();
        atomicAdd(dbg + 2, (unsigned long long) (now - t_poll));
        atomicAdd(dbg + 3, 1ull);
        // the time since the site before this one: a segment of the window (temporary)
        const unsigned long long prev = dbg[4];
        dbg[4] = (unsigned long long) now;
        // not the windows that the host paces (first executions, captures): they are ms per segment
        if (site > 0 && site < 126 && prev != 0 && (unsigned long long) now - prev < 4000000ull) {
            dbg[8 + 2*site] += (unsigned long long) now - prev;
            dbg[9 + 2*site] += 1;
        } else if (site == 0 && dbg[5] != 0 && (unsigned long long) now - dbg[5] < 4000000ull) {
            dbg[8] += (unsigned long long) now - dbg[5]; // since the window's first kernel
            dbg[9] += 1;
        }
    }

    const float4 o4  = fn_ar_unpack(u);
    const float4 out = make_float4(y4.x + o4.x + z4.x, y4.y + o4.y + z4.y, y4.z + o4.z + z4.z, y4.w + o4.w + z4.w);
#pragma unroll
    for (int c = 0; c < FN_HC; ++c) {
        r[c] = make_float4(r[c].x + out.x*gate[c], r[c].y + out.y*gate[c], r[c].z + out.z*gate[c], r[c].w + out.w*gate[c]);
        *(float4 *) (Rn + t*sd_t + c*sd_c + 4*i) = r[c];
    }
    fn_hc_norm_part(r, gn, n_embd, eps, xn + t*sxn_t, sxn_c, o, t, s_w, s_tot);
}

// ---------------------------------------------------------------------------------------------------------------
// MoE experts with Q2_0 weights: gate, up, SwiGLU and down of the routed (token, expert) pairs of a decode window
//
// A Q2_0 block is { half d; 16 bytes of 2-bit codes }: 64 weights (code - 1)*d, 18 bytes. A thread owns one block
// of every row it reads: 64 columns, whose activations it keeps in registers as 32 half2, in the order that the
// decode produces (pair j of a 16-code word holds its elements j and j + 8).
// The blocks are read where they are, with aligned 4-byte loads: an even block starts on a word, its codes two
// bytes into it; the codes of an odd block are words themselves and its scale ends the word before them. So every
// block is five words from an aligned address, and __byte_perm puts the codes together.
//   fn_moe_up:   block (row tile, pair): h = silu(w_gate . x)*(w_up . x) for 16 rows of the pair's expert
//   fn_moe_down: block (row tile, token): y = sum over the token's pairs of weight * (w_down . h), in registers
// Pairs whose expert is not in VRAM (position >= n_hot) are left to the host threads (moe-host.cuh).

#define FN_Q2_TPR 40   // threads per row of gate/up: n_embd = 2560

struct fn_q2_raw {
    uint32_t w[5];
};

static __device__ __forceinline__ void fn_q2_load(const char * __restrict__ row, const int kb, fn_q2_raw & r) {
    const uint32_t * p = (const uint32_t *) (row + kb*18 - 2*(kb & 1));
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        r.w[j] = p[j];
    }
}

// 16 codes (one word) -> 8 half2 holding code - 1, pair j = elements (j, j + 8) for j < 4, (j, j + 8) + 4 above
static __device__ __forceinline__ void fn_q2_decode16(const uint32_t q, half2 * h) {
    const uint32_t qs = q >> 8;
    uint32_t t[8];
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[0]) : "r"(q),  "n"(0x00030003), "n"(0x64006400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[1]) : "r"(q),  "n"(0x000c000c), "n"(0x5c005c00));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[2]) : "r"(q),  "n"(0x00300030), "n"(0x54005400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[3]) : "r"(q),  "n"(0x00c000c0), "n"(0x4c004c00));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[4]) : "r"(qs), "n"(0x00030003), "n"(0x64006400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[5]) : "r"(qs), "n"(0x000c000c), "n"(0x5c005c00));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[6]) : "r"(qs), "n"(0x00300030), "n"(0x54005400));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t[7]) : "r"(qs), "n"(0x00c000c0), "n"(0x4c004c00));
    // a code sits on a base (1024, 256, 64, 16 by its bit position): base + 1 maps it to code - 1
    const half2 m0 = __float2half2_rn(1025.0f);
    const half2 m1 = __float2half2_rn( 257.0f);
    const half2 m2 = __float2half2_rn(  65.0f);
    const half2 m3 = __float2half2_rn(  17.0f);
    h[0] = __hsub2(fn_h2((int) t[0]), m0);
    h[1] = __hsub2(fn_h2((int) t[1]), m1);
    h[2] = __hsub2(fn_h2((int) t[2]), m2);
    h[3] = __hsub2(fn_h2((int) t[3]), m3);
    h[4] = __hsub2(fn_h2((int) t[4]), m0);
    h[5] = __hsub2(fn_h2((int) t[5]), m1);
    h[6] = __hsub2(fn_h2((int) t[6]), m2);
    h[7] = __hsub2(fn_h2((int) t[7]), m3);
}

// the 64 activations of a thread (16 words of natural pairs) in the decode order
static __device__ __forceinline__ void fn_q2_load_act(const half2 * __restrict__ X, half2 a[32]) {
    const int4 * ap = (const int4 *) X;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int4 lo = ap[2*q];      // elements 16q .. 16q + 7
        const int4 hi = ap[2*q + 1];  // elements 16q + 8 .. 16q + 15
        a[8*q + 0] = fn_h2(__byte_perm(lo.x, hi.x, 0x5410));
        a[8*q + 1] = fn_h2(__byte_perm(lo.x, hi.x, 0x7632));
        a[8*q + 2] = fn_h2(__byte_perm(lo.y, hi.y, 0x5410));
        a[8*q + 3] = fn_h2(__byte_perm(lo.y, hi.y, 0x7632));
        a[8*q + 4] = fn_h2(__byte_perm(lo.z, hi.z, 0x5410));
        a[8*q + 5] = fn_h2(__byte_perm(lo.z, hi.z, 0x7632));
        a[8*q + 6] = fn_h2(__byte_perm(lo.w, hi.w, 0x5410));
        a[8*q + 7] = fn_h2(__byte_perm(lo.w, hi.w, 0x7632));
    }
}

// the dot product of one block with the thread's activations, times the block's scale
static __device__ __forceinline__ float fn_q2_dot(const fn_q2_raw & r, const int kb, const half2 a[32]) {
    const int sel = (kb & 1) ? 0x7654 : 0x5432;
    half2 acc = make_half2(0.0f, 0.0f);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        half2 h[8];
        fn_q2_decode16((uint32_t) __byte_perm(r.w[q], r.w[q + 1], sel), h);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc = __hfma2(h[j], a[8*q + j], acc);
        }
    }
    // the scale and the sum of the two halves of acc, converted as one pair
    const uint32_t db  = (r.w[0] >> ((kb & 1)*16)) & 0xffffu;
    const uint32_t sum = (uint32_t) __half_as_ushort(__hadd(__low2half(acc), __high2half(acc)));
    const float2   f   = __half22float2(fn_h2((int) (db | (sum << 16))));
    return f.x*f.y;
}

typedef ggml_cuda_fn_moe_route fn_moe_route;

#define FN_MOE_UP_G 4
#define FN_MOE_UP_R 4   // 16 rows of gate and of up per block, in two passes of 16 blocks

// grid: (n_ff/16, pairs). h: [pairs][n_ff].
// route (may be null): the pairs' positions and weights, copied for fn_moe_down, whose output may share memory with
// the ids and the weights.
static __global__ void __launch_bounds__(FN_MOE_UP_G*FN_Q2_TPR, 5)
fn_moe_up(const char * __restrict__ wg, const char * __restrict__ wu, const int64_t nb1, const int64_t nb2,
          const half2 * __restrict__ X, const int xstride, const float * __restrict__ xscale,
          const int32_t * __restrict__ ids, const int si1, const int n_used, const int n_hot,
          const float * __restrict__ weights, const int sw1,
          float * __restrict__ h, const int n_ff, fn_moe_route * __restrict__ route) {
#if defined(FP16_AVAILABLE)
    constexpr int G  = FN_MOE_UP_G;
    constexpr int R  = FN_MOE_UP_R;
    constexpr int NT = G*FN_Q2_TPR;
    __shared__ float s_part[2*R][NT + 1];
    const int tid = threadIdx.x;
    const int g   = tid/FN_Q2_TPR;
    const int k   = tid - g*FN_Q2_TPR;
    const int c   = blockIdx.y;
    const int t   = c/n_used;
    const int e   = ids[t*si1 + c % n_used];
    if (route != nullptr && blockIdx.x == 0 && tid == 0) {
        route[c].e = e;
        route[c].w = weights[t*sw1 + c % n_used];
    }
    if (e >= n_hot) {
        return;
    }
    half2 a[32];
    fn_q2_load_act(X + (int64_t) t*xstride + k*32, a);

    // row (r, g) of the tile is tile*G*R + r*G + g
    const int64_t row0 = (int64_t) blockIdx.x*(G*R) + g;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const char * w = (m == 0 ? wg : wu) + (int64_t) e*nb2 + row0*nb1;
        fn_q2_raw raw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            fn_q2_load(w + (int64_t) r*G*nb1, k, raw[r]);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            s_part[m*R + r][tid] = fn_q2_dot(raw[r], k, a);
        }
    }
    __syncthreads();
    // thread (gg, r) of the first G*R: row r of group gg
    if (tid < G*R) {
        const int gg = tid/R;
        const int r  = tid - gg*R;
        float sg = 0.0f, su = 0.0f;
#pragma unroll 4
        for (int j = 0; j < FN_Q2_TPR; ++j) {
            sg += s_part[r][gg*FN_Q2_TPR + j];
            su += s_part[R + r][gg*FN_Q2_TPR + j];
        }
        const float xs = xscale[t];
        sg *= xs;
        su *= xs;
        h[(int64_t) c*n_ff + blockIdx.x*(G*R) + r*G + gg] = sg/(1.0f + expf(-sg))*su;
    }
#else
    GGML_UNUSED_VARS(wg, wu, nb1, nb2, X, xstride, xscale, ids, si1, n_used, n_hot, weights, sw1, h, n_ff, route);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

#define FN_MOE_NU     10  // pairs of a token that a block of fn_moe_down has threads for
#define FN_MOE_DOWN_R 8   // rows per row group of a tile, in two passes

// grid: (n_embd/(G*R), tokens). Block: G row groups x FN_MOE_NU pairs x TPR = n_ff/64 threads (4, 6, 8 or 12): thread (g, s, k) owns
// block k of the rows g, g + G, ... of the tile in the expert of the token's pair s, so the pairs of a token are
// computed side by side (one after the other, a block waits for memory once per pair) and summed like the threads
// of a row. Xh: the activations of h [pairs][n_ff halves] with the scales xsh [pairs]. y: [tokens][sy].
// row groups of a block by TPR: blocks of 120 or 160 threads
static constexpr __host__ __device__ int fn_moe_down_groups(const int tpr) {
    return tpr == 4 ? 4 : tpr <= 8 ? 2 : 1;
}

template <int TPR>
static __global__ void __launch_bounds__(fn_moe_down_groups(TPR)*FN_MOE_NU*TPR, 5)
fn_moe_down(const char * __restrict__ wd, const int64_t nb1, const int64_t nb2,
            const half2 * __restrict__ Xh, const float * __restrict__ xsh, const fn_moe_route * __restrict__ route,
            const int n_used, const int n_hot, float * __restrict__ y, const int64_t sy) {
#if defined(FP16_AVAILABLE)
    constexpr int R   = FN_MOE_DOWN_R;
    constexpr int TPG = FN_MOE_NU*TPR;  // threads per row
    constexpr int G   = fn_moe_down_groups(TPR);
    constexpr int NT  = G*TPG;
    __shared__ float s_part[R][NT + 1];
    const int tid = threadIdx.x;
    const int g   = tid/TPG;
    const int j   = tid - g*TPG;
    const int s   = j/TPR;
    const int k   = j - s*TPR;
    const int t   = blockIdx.y;
    const int c   = t*n_used + s;

    fn_moe_route rt = { n_hot, 0.0f };
    if (s < n_used) {
        rt = route[c];
    }
    if (rt.e < n_hot) {
        half2 a[32];
        fn_q2_load_act(Xh + (int64_t) c*(TPR*32) + k*32, a);
        const float  ws = rt.w*xsh[c];
        const char * w  = wd + (int64_t) rt.e*nb2 + ((int64_t) blockIdx.x*(G*R) + g)*nb1;
#pragma unroll
        for (int b = 0; b < R; b += 4) {
            fn_q2_raw raw[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                fn_q2_load(w + (int64_t) (b + r)*G*nb1, k, raw[r]);
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                s_part[b + r][tid] = ws*fn_q2_dot(raw[r], k, a);
            }
        }
    } else {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            s_part[r][tid] = 0.0f;
        }
    }
    __syncthreads();
    // thread (gg, r) of the first G*R: row r of group gg
    if (tid < G*R) {
        const int gg = tid/R;
        const int r  = tid - gg*R;
        float v = 0.0f;
#pragma unroll 4
        for (int q = 0; q < TPG; ++q) {
            v += s_part[r][gg*TPG + q];
        }
        y[t*sy + blockIdx.x*(G*R) + r*G + gg] = v;
    }
#else
    GGML_UNUSED_VARS(wd, nb1, nb2, Xh, xsh, route, n_used, n_hot, y, sy);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

#ifndef FN_STANDALONE

// Are the n nodes from node i the ops, computed, and used by nothing but each other, except for the last one?
// ggml_can_fuse_subgraph asks more than these patterns have: it refuses a node that is a view of a tensor outside
// (a RESHAPE of an input, which nothing elides) and a cast (a CPY whose destination is the node itself and counts
// as a use).
// GGML_CUDA_FN_OFF: a bit mask of the fused patterns below that are left to the generic ops
enum fn_pattern_bit { FN_PAT_GDN_OUT, FN_PAT_GATE_OUT, FN_PAT_QSA_SEL, FN_PAT_QSA_POOL, FN_PAT_NORM_ROPE, FN_PAT_GDN_PRE, FN_PAT_QSA_ATTN };

static bool fn_pattern_on(const fn_pattern_bit bit) {
    static const int off = getenv("GGML_CUDA_FN_OFF") != nullptr ? atoi(getenv("GGML_CUDA_FN_OFF")) : 0;
    return ggml_cuda_fn_enabled() && (off & (1 << bit)) == 0;
}

bool ggml_cuda_fn_pattern_closed(const ggml_cgraph * cgraph, const int i, const ggml_op * ops, const int n, const uint64_t open) {
    if (i + n > cgraph->n_nodes) {
        return false;
    }
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * node = cgraph->nodes[i + k];
        if (node->op != ops[k] || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        if (k == n - 1 || ((open >> k) & 1)) {
            continue;
        }
        if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        int uses = 0;
        for (int j = 0; j < n; ++j) {
            const ggml_tensor * other = cgraph->nodes[i + j];
            for (int q = 0; q < GGML_MAX_SRC; ++q) {
                uses += other->src[q] == node;
            }
        }
        if (uses != ggml_node_get_use_count(cgraph, i + k)) {
            return false;
        }
    }
    return true;
}

static bool fn_pattern_closed(const ggml_cgraph * cgraph, const int i, const ggml_op * ops, const int n) {
    return ggml_cuda_fn_pattern_closed(cgraph, i, ops, n, 0);
}

// temporary: why a fused pattern was declined (GGML_CUDA_FN_DEBUG)
static bool fn_debug() {
    static const bool on = getenv("GGML_CUDA_FN_DEBUG") != nullptr;
    return on;
}

static void fn_pattern_why(const char * name, const ggml_cgraph * cgraph, const int i, const ggml_op * ops, const int n) {
    static std::atomic<int> n_msg { 0 };
    if (!fn_debug() || i + 3 > cgraph->n_nodes || cgraph->nodes[i + 1]->op != ops[1] || cgraph->nodes[i + 2]->op != ops[2] ||
        n_msg.fetch_add(1) >= 60) {
        return;
    }
    for (int k = 0; k < n; ++k) {
        if (i + k >= cgraph->n_nodes) {
            fprintf(stderr, "fn-decline: %s at %d: graph ends at +%d\n", name, i, k);
            return;
        }
        const ggml_tensor * node = cgraph->nodes[i + k];
        if (node->op != ops[k]) {
            fprintf(stderr, "fn-decline: %s at %d: +%d is %s, expected %s\n", name, i, k, ggml_op_name(node->op), ggml_op_name(ops[k]));
            return;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            fprintf(stderr, "fn-decline: %s at %d: +%d (%s) is not computed\n", name, i, k, ggml_op_name(node->op));
            return;
        }
        if (k == n - 1) {
            break;
        }
        int uses = 0;
        for (int j = 0; j < n && i + j < cgraph->n_nodes; ++j) {
            for (int q = 0; q < GGML_MAX_SRC; ++q) {
                uses += cgraph->nodes[i + j]->src[q] == node;
            }
        }
        if ((node->flags & GGML_TENSOR_FLAG_OUTPUT) || uses != ggml_node_get_use_count(cgraph, i + k)) {
            fprintf(stderr, "fn-decline: %s at %d: +%d (%s %s) output %d, uses inside %d of %d\n", name, i, k, ggml_op_name(node->op),
                    node->name, (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, uses, ggml_node_get_use_count(cgraph, i + k));
            return;
        }
    }
}

static bool fn_decline(const char * what, const int i) {
    static std::atomic<int> n_msg { 0 };
    if (fn_debug() && n_msg.fetch_add(1) < 60) {
        fprintf(stderr, "fn-decline: %s at %d\n", what, i);
    }
    return false;
}

static bool fn_hc_aligned(const ggml_tensor * t) {
    return ((uintptr_t) t->data & 0xF) == 0;
}

static bool fn_overlap(const ggml_tensor * x, const ggml_tensor * y) {
    const char * x0 = (const char *) x->data;
    const char * y0 = (const char *) y->data;
    return x0 < y0 + ggml_nbytes(y) && y0 < x0 + ggml_nbytes(x);
}

// the norm of a read: R [n_embd, hc, nt], w_norm [n_embd, hc]
static bool fn_hc_norm_supported(const ggml_tensor * rms, const ggml_tensor * mul) {
    const ggml_tensor * R  = rms->src[0];
    const ggml_tensor * wn = mul->src[1];
    const int64_t n_embd = R->ne[0];
    return mul->src[0] == rms && R->type == GGML_TYPE_F32 && wn->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
           R->ne[1] == FN_HC && R->ne[2] <= FN_MAX_T && R->ne[3] == 1 && ggml_is_contiguous(R) && ggml_is_contiguous(mul) &&
           ggml_are_same_shape(R, mul) && ggml_is_contiguous(wn) && wn->ne[0] == n_embd && wn->ne[1] == FN_HC && ggml_nrows(wn) == FN_HC &&
           n_embd % (4*WARP_SIZE) == 0 && n_embd/4 <= 1024 && n_embd/4/WARP_SIZE <= FN_HCN_MAX_NW &&
           fn_hc_aligned(R) && fn_hc_aligned(wn) && fn_hc_aligned(mul);
}

// down, SiLU, up and the gated sum of a read whose xn is in a.mul
static bool fn_hc_tail_supported(const ggml_cuda_hc_mix_args & a) {
    const ggml_tensor * wd = a.mm_down->src[0];
    const ggml_tensor * wu = a.mm_up->src[0];
    const int64_t n_embd = a.mul->ne[0];
    const int64_t hc_lr  = wd->ne[1];
    fn_dense_geom gm;
    return ggml_cuda_fn_planar(wd) && ggml_cuda_fn_planar(wu) && a.mul->ne[1] == FN_HC && a.mul->ne[2] <= FN_MAX_T &&
           a.mul->type == GGML_TYPE_F32 && ggml_is_contiguous(a.mul) &&
           wd->ne[0] == FN_HC*n_embd && wu->ne[0] == hc_lr && wu->ne[1] == FN_HC*n_embd &&
           n_embd % (2*FN_HC_UP_R) == 0 && n_embd % 4 == 0 && fn_dense_geometry(FN_HC*n_embd, &gm) && gm.s == FN_HC &&
           hc_lr == 16*FN_HC_UP_TPR && hc_lr % fn_dense_tile_rows(FN_HC*n_embd) == 0 && hc_lr <= 2*FN_ACT_TPB &&
           a.pre->type == GGML_TYPE_F32 && ggml_is_contiguous(a.pre) && !fn_overlap(a.mul, a.pre);
}

bool ggml_cuda_fn_hc_mix_supported(const ggml_cuda_hc_mix_args & a) {
    return fn_hc_norm_supported(a.rms, a.mul) && fn_hc_tail_supported(a) &&
           !fn_overlap(a.mul, a.rms->src[0]) && !fn_overlap(a.pre, a.rms->src[0]);
}

// What the norm of a read writes for its down mat-vec (reading xd, the view of xn as [hc*n_embd, nt]) and for the
// consumers of its output pre (either may be null)
static fn_hc_norm_out fn_hc_norm_outputs(ggml_backend_cuda_context & ctx, const ggml_tensor * mul, const ggml_tensor * xd, const ggml_tensor * pre) {
    const int n_embd = (int) mul->ne[0];
    const int nt     = (int) mul->ne[2];
    fn_hc_norm_out o = { nullptr, nullptr, nullptr, 0.0f };
    if (xd == nullptr) {
        return o;
    }
    fn_act_slot & sd = fn_act_put(ctx, xd, FN_HC*n_embd, nt, true);
    o.X  = fn_act_X(sd);
    o.xs = fn_act_xs(sd);
    if (pre != nullptr) {
        fn_act_slot & sp = fn_act_put(ctx, pre, n_embd, nt, false);
        // the put above may not have taken the slot of xd
        GGML_ASSERT(&sp != &sd);
        o.xs_mixed  = fn_act_xs(sp);
        o.mix_scale = fabsf(ggml_get_op_params_f32(pre, 0))/fn_gain(n_embd);
    }
    return o;
}

static void fn_hc_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & a) {
    const ggml_tensor * wd = a.mm_down->src[0];
    const ggml_tensor * wu = a.mm_up->src[0];
    const ggml_tensor * xd = a.mm_down->src[1];
    const int     n_embd = (int) a.mul->ne[0];
    const int     nt     = (int) a.mul->ne[2];
    const int     hc_lr  = (int) wd->ne[1];
    const int     hd     = FN_HC*n_embd;
    const int64_t sxn_c  = a.mul->nb[1]/sizeof(float);
    const int64_t sxn_t  = a.mul->nb[2]/sizeof(float);
    const int64_t smx_t  = a.pre->nb[1]/sizeof(float);
    const int     nsm    = ggml_cuda_info().devices[ctx.device].nsm;
    cudaStream_t  stream = ctx.stream();

    ggml_cuda_fn_plane pu;
    GGML_ASSERT(ggml_cuda_fn_planar(wu, &pu));
    GGML_ASSERT(sxn_c == n_embd); // the four streams of a token are one vector of hd values

    ggml_cuda_pool_alloc<float> lo (ctx.pool(), (size_t) nt*hc_lr);
    ggml_cuda_pool_alloc<half>  Xl (ctx.pool(), (size_t) nt*hc_lr);
    ggml_cuda_pool_alloc<float> xsl(ctx.pool(), nt);

    // the activations of xn: from the norm, or converted here
    const fn_act_slot & sx = fn_act_get(ctx, xd, (const float *) a.mul->data, sxn_t, hd, nt);
    fn_dense_wide(ctx, wd, sx, nt, FN_EPI_SILU, ggml_get_op_params_f32(a.scale, 0), ggml_get_op_params_f32(a.scale, 1),
                  lo.get(), hc_lr, Xl.get(), xsl.get());

    // the output as activations, if the norm left its scales
    fn_act_slot * sp = fn_act_find(ctx, a.pre, n_embd, nt);
    half *        Xm = nullptr;
    const float * xm = nullptr;
    if (sp != nullptr && !sp->full) {
        Xm = fn_act_X(*sp);
        xm = fn_act_xs(*sp);
        sp->full = true;
    }

    const uint4 * Wu    = (const uint4 *) wu->data;
    const half *  Du    = (const half *) ((const char *) wu->data + (int64_t) hd*hc_lr);
    const float   scale = ggml_get_op_params_f32(a.pre, 0);
    const int     ntiles = n_embd/(2*FN_HC_UP_R);
    switch (nt) {
#define FN_CASE(N, BPS)                                                                                               \
        case N:                                                                                                       \
            fn_hc_up<N, FN_HC_UP_TPR, BPS><<<std::min(ntiles, nsm*BPS), FN_HC_UP_G*FN_HC_UP_TPR, 0, stream>>>(        \
                Wu, Du, pu.rowscale, (const half2 *) Xl.get(), xsl.get(), n_embd, scale,                              \
                (const float *) a.mul->data, (int) sxn_c, (int) sxn_t, (float *) a.pre->data, (int) smx_t, Xm, xm);   \
            break;
        FN_CASE(1, 8)
        FN_CASE(2, 6)
        FN_CASE(3, 5)
        FN_CASE(4, 4)
        FN_CASE(5, 4)
        FN_CASE(6, 3)
        FN_CASE(7, 3)
        FN_CASE(8, 3)
#undef FN_CASE
        default:
            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_fn_hc_mix(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & a) {
    const ggml_tensor * R  = a.rms->src[0];
    const ggml_tensor * wn = a.mul->src[1];
    const int n_embd = (int) R->ne[0];
    const int nt     = (int) R->ne[2];

    const fn_hc_norm_out o = fn_hc_norm_outputs(ctx, a.mul, a.mm_down->src[1], a.pre);
    fn_hc_norm<<<nt, n_embd/4, 0, ctx.stream()>>>(
        (const float *) R->data, R->nb[1]/sizeof(float), R->nb[2]/sizeof(float), (const float *) wn->data, n_embd,
        ggml_get_op_params_f32(a.rms, 0), (float *) a.mul->data, a.mul->nb[1]/sizeof(float), a.mul->nb[2]/sizeof(float), o);
    CUDA_CHECK(cudaGetLastError());

    fn_hc_tail(ctx, a);
}

// The six nodes of a read after its norm, from the first one: MUL_MAT (down), SCALE, SILU, MUL_MAT (up), RESHAPE,
// DSV4_HC_PRE, on the xn of a MUL node.
static bool fn_hc_tail_nodes(ggml_tensor * const * n, ggml_cuda_hc_mix_args & a) {
    a.mm_down = n[0];
    a.scale   = n[1];
    a.silu    = n[2];
    a.mm_up   = n[3];
    a.pre     = n[5];
    if (a.mm_down->op != GGML_OP_MUL_MAT || a.scale->op != GGML_OP_SCALE || a.silu->op != GGML_OP_UNARY ||
        a.mm_up->op != GGML_OP_MUL_MAT || n[4]->op != GGML_OP_RESHAPE || a.pre->op != GGML_OP_DSV4_HC_PRE) {
        return false;
    }
    const ggml_tensor * xd = a.mm_down->src[1]; // view of xn as [hc*n_embd, nt]
    const ggml_tensor * xp = a.pre->src[0];     // view of xn as [n_embd, hc, nt]
    if (xd->view_src == nullptr || xd->view_src->op != GGML_OP_MUL || xp->view_src != xd->view_src ||
        xd->data != xd->view_src->data || xp->data != xd->view_src->data) {
        return false;
    }
    a.mul = xd->view_src;
    a.rms = nullptr;
    if (ggml_get_unary_op(a.silu) != GGML_UNARY_OP_SILU || ggml_get_op_params_i32(a.pre, 1) == 0 ||
        a.scale->src[0] != a.mm_down || a.silu->src[0] != a.scale || a.mm_up->src[1] != a.silu ||
        a.pre->src[1] != n[4] || n[4]->view_src != a.mm_up || a.mul->ne[3] != 1) {
        return false;
    }
    return fn_hc_tail_supported(a);
}

// the read without its norm (the xn of a.mul was computed with an AllReduce, ggml_cuda_fn_ar_epilogue)
bool ggml_cuda_fn_hc_tail_match(const ggml_cgraph * cgraph, const int i, ggml_cuda_hc_mix_args & a) {
    if (!ggml_cuda_fn_enabled() || i + 5 >= cgraph->n_nodes || cgraph->nodes[i]->op != GGML_OP_MUL_MAT) {
        return false;
    }
    static const ggml_op ops[] = { GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE };
    const int outputs[] = { i + 5 };
    return ggml_can_fuse_subgraph(cgraph, i, 6, ops, outputs, 1) && fn_hc_tail_nodes(cgraph->nodes + i, a);
}

void ggml_cuda_fn_hc_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & a) {
    fn_hc_tail(ctx, a);
}

// ---------------------------------------------------------------------------------------------------------------
// the shared expert (shexp-fuse.cuh) on planar weights:
//   gate and up as two segments of one launch, the SwiGLU of their outputs as activations, down.
// The gate of the expert's output, sigmoid(w_gate_inp . x), is one value per token: it goes into the scale of the
// activations that down reads.

bool ggml_cuda_fn_shexp_supported(const ggml_cuda_shexp_args & a) {
    const ggml_tensor * x   = a.gate->src[1];
    const ggml_tensor * wg  = a.gate->src[0];
    const ggml_tensor * wu  = a.up->src[0];
    const ggml_tensor * wd  = a.down->src[0];
    const ggml_tensor * wgi = a.ginp->src[0];
    const int64_t n_ff = wg->ne[1];
    fn_dense_geom gu, gd;
    return ggml_cuda_fn_enabled() && ggml_cuda_fn_planar(wg) && ggml_cuda_fn_planar(wu) && ggml_cuda_fn_planar(wd) &&
           x->ne[1] <= FN_MAX_T && fn_dense_geometry(wg->ne[0], &gu) && gu.s == 1 && fn_dense_geometry(wd->ne[0], &gd) && gd.s == 1 &&
           n_ff <= 4*FN_ACT_TPB && (wgi->type == GGML_TYPE_F32 || wgi->type == GGML_TYPE_BF16) && ggml_is_contiguous(x);
}

// h = silu(g)*u of a token as activations; their scale times sigmoid(w_gi . x). One block per token; n <= 4*FN_ACT_TPB.
template <typename T_gi>
static __global__ void fn_swiglu_h16(const float * __restrict__ g, const float * __restrict__ u, const int n,
                                     const float * __restrict__ x, const int64_t sx, const int n_embd,
                                     const T_gi * __restrict__ w_gi, const float gain,
                                     half * __restrict__ X, float * __restrict__ xscale) {
    __shared__ float s_red[2][FN_ACT_TPB/WARP_SIZE];
    const int t = blockIdx.x;
    float h[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float m    = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = threadIdx.x + j*FN_ACT_TPB;
        if (i < n) {
            const float gv = g[(int64_t) t*n + i];
            h[j] = gv/(1.0f + expf(-gv))*u[(int64_t) t*n + i];
            m    = fmaxf(m, fabsf(h[j]));
        }
    }
    float dot = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += FN_ACT_TPB) {
        dot += ggml_cuda_cast<float>(w_gi[i])*x[t*sx + i];
    }
#pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
        m    = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        dot += __shfl_xor_sync(0xffffffff, dot, o);
    }
    if (threadIdx.x % WARP_SIZE == 0) {
        s_red[0][threadIdx.x / WARP_SIZE] = m;
        s_red[1][threadIdx.x / WARP_SIZE] = dot;
    }
    __syncthreads();
    m   = 0.0f;
    dot = 0.0f;
#pragma unroll
    for (int w = 0; w < FN_ACT_TPB/WARP_SIZE; ++w) {
        m    = fmaxf(m, s_red[0][w]);
        dot += s_red[1][w];
    }
    const float sc = m > 0.0f ? gain/m : 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = threadIdx.x + j*FN_ACT_TPB;
        if (i < n) {
            X[(int64_t) t*n + i] = __float2half_rn(h[j]*sc);
        }
    }
    if (threadIdx.x == 0) {
        xscale[t] = (m > 0.0f ? m/gain : 0.0f)/(1.0f + expf(-dot));
    }
}

// g, u: gate and up of the nt tokens, n_ff apart
static void fn_shexp_down(ggml_backend_cuda_context & ctx, const ggml_cuda_shexp_args & a, const float * g, const float * u) {
    const ggml_tensor * x   = a.gate->src[1];
    const ggml_tensor * wd  = a.down->src[0];
    const ggml_tensor * wgi = a.ginp->src[0];
    const int     n_embd = (int) wd->ne[1];
    const int     n_ff   = (int) wd->ne[0];
    const int     nt     = (int) x->ne[1];
    const int64_t sx     = x->nb[1]/sizeof(float);
    cudaStream_t  stream = ctx.stream();

    ggml_cuda_fn_plane pd;
    GGML_ASSERT(ggml_cuda_fn_planar(wd, &pd));

    ggml_cuda_pool_alloc<char> hm(ctx.pool(), (size_t) FN_MAX_T*n_ff*sizeof(half) + FN_MAX_T*sizeof(float));
    fn_act_slot h;
    h.mem = hm.get();
    h.n   = n_ff;

    if (wgi->type == GGML_TYPE_F32) {
        fn_swiglu_h16<float><<<nt, FN_ACT_TPB, 0, stream>>>(g, u, n_ff, (const float *) x->data, sx,
            n_embd, (const float *) wgi->data, fn_gain(n_ff), fn_act_X(h), fn_act_xs(h));
    } else {
        fn_swiglu_h16<nv_bfloat16><<<nt, FN_ACT_TPB, 0, stream>>>(g, u, n_ff, (const float *) x->data, sx,
            n_embd, (const nv_bfloat16 *) wgi->data, fn_gain(n_ff), fn_act_X(h), fn_act_xs(h));
    }
    CUDA_CHECK(cudaGetLastError());

    const fn_dense_part pdn = fn_dense_part_of(wd, pd, 0, h, (float *) a.mul->data, a.mul->nb[1]/sizeof(float));
    fn_dense(nt, &pdn, 1, ggml_cuda_info().devices[ctx.device].nsm, stream);
}

void ggml_cuda_fn_shexp(ggml_backend_cuda_context & ctx, const ggml_cuda_shexp_args & a) {
    const ggml_tensor * x   = a.gate->src[1];
    const ggml_tensor * wg  = a.gate->src[0];
    const ggml_tensor * wu  = a.up->src[0];
    const int     n_embd = (int) wg->ne[0];
    const int     n_ff   = (int) wg->ne[1];
    const int     nt     = (int) x->ne[1];
    const int64_t sx     = x->nb[1]/sizeof(float);

    ggml_cuda_fn_plane pg, pu;
    GGML_ASSERT(ggml_cuda_fn_planar(wg, &pg) && ggml_cuda_fn_planar(wu, &pu));

    ggml_cuda_pool_alloc<float> gu(ctx.pool(), (size_t) 2*nt*n_ff);
    const fn_act_slot & sxa = fn_act_get(ctx, x, (const float *) x->data, sx, n_embd, nt);
    const fn_dense_part pt[2] = {
        fn_dense_part_of(wg, pg, 0, sxa, gu.get(), n_ff),
        fn_dense_part_of(wu, pu, 0, sxa, gu.get() + (size_t) nt*n_ff, n_ff),
    };
    fn_dense(nt, pt, 2, ggml_cuda_info().devices[ctx.device].nsm, ctx.stream());
    fn_shexp_down(ctx, a, gu.get(), gu.get() + (size_t) nt*n_ff);
}

bool ggml_cuda_fn_shexp_tail_match(const ggml_cgraph * cgraph, const int i, ggml_cuda_shexp_args & a) {
    static const ggml_op ops[] = { GGML_OP_GLU, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL };
    if (!ggml_cuda_fn_enabled() || i + 4 >= cgraph->n_nodes || cgraph->nodes[i]->op != GGML_OP_GLU) {
        return false;
    }
    const int outputs[] = { i + 4 };
    if (!ggml_can_fuse_subgraph(cgraph, i, 5, ops, outputs, 1)) {
        return false;
    }
    ggml_tensor * const * n = cgraph->nodes + i;
    a.glu  = n[0];
    a.gate = a.glu->src[0];
    a.up   = a.glu->src[1];
    a.down = n[1];
    a.ginp = n[2];
    a.sig  = n[3];
    a.mul  = n[4];
    if (a.gate->op != GGML_OP_MUL_MAT || a.up->op != GGML_OP_MUL_MAT ||
        ggml_get_glu_op(a.glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(a.glu, 1) != 0 ||
        ggml_get_unary_op(a.sig) != GGML_UNARY_OP_SIGMOID || a.sig->src[0] != a.ginp ||
        a.down->src[1] != a.glu || a.mul->src[0] != a.down || a.mul->src[1] != a.sig) {
        return false;
    }
    const ggml_tensor * x   = a.gate->src[1];
    const ggml_tensor * wg  = a.gate->src[0];
    const ggml_tensor * wu  = a.up->src[0];
    const ggml_tensor * wd  = a.down->src[0];
    const ggml_tensor * wgi = a.ginp->src[0];
    const int64_t n_embd = wg->ne[0];
    const int64_t n_ff   = wg->ne[1];
    const int64_t nt     = x->ne[1];
    // gate and up as the launch of their MUL_MATs left them
    for (const ggml_tensor * t : { a.gate, a.up }) {
        if (t->type != GGML_TYPE_F32 || t->ne[0] != n_ff || t->ne[1] != nt || !ggml_is_contiguous(t)) {
            return false;
        }
    }
    if (a.up->src[1] != x || a.ginp->src[1] != x || x->type != GGML_TYPE_F32 || x->ne[2] != 1 || x->ne[3] != 1 ||
        x->nb[0] != sizeof(float) || x->nb[1] % 16 != 0 || !ggml_are_same_shape(wg, wu) || wd->ne[0] != n_ff || wd->ne[1] != n_embd ||
        wd->ne[2] != 1 || wgi->ne[0] != n_embd || ggml_nrows(wgi) != 1 || !ggml_is_contiguous(wgi) ||
        a.mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(a.mul) || a.mul->ne[0] != n_embd || a.mul->ne[1] != nt) {
        return false;
    }
    return ggml_cuda_fn_shexp_supported(a);
}

void ggml_cuda_fn_shexp_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_shexp_args & a) {
    fn_shexp_down(ctx, a, (const float *) a.gate->data, (const float *) a.up->data);
}

// ---------------------------------------------------------------------------------------------------------------
// RMS_NORM, MUL (by the norm weights), RESHAPE (of z), UNARY (SiLU), MUL, RESHAPE from node i: fn_gdn_out

int ggml_cuda_fn_gdn_out(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if (!fn_pattern_on(FN_PAT_GDN_OUT) || i + 5 >= cgraph->n_nodes || cgraph->nodes[i]->op != GGML_OP_RMS_NORM) {
        return 0;
    }
    static const ggml_op ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_RESHAPE };
    if (!fn_pattern_closed(cgraph, i, ops, 6)) {
        return 0;
    }
    ggml_tensor * const * n = cgraph->nodes + i;
    const ggml_tensor * rms  = n[0];
    const ggml_tensor * mulw = n[1];
    const ggml_tensor * zr   = n[2];
    const ggml_tensor * silu = n[3];
    ggml_tensor *       out  = n[4];
    const ggml_tensor * key  = n[5];
    const ggml_tensor * x    = rms->src[0];
    const ggml_tensor * w    = mulw->src[0] == rms ? mulw->src[1] : mulw->src[0];
    const int64_t heads = x->ne[1];
    const int64_t nt    = x->ne[2];
    const bool is_silu = ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU;
    if ((mulw->src[0] != rms && mulw->src[1] != rms) || (!is_silu && ggml_get_unary_op(silu) != GGML_UNARY_OP_SIGMOID) || silu->src[0] != zr ||
        !((out->src[0] == mulw && out->src[1] == silu) || (out->src[1] == mulw && out->src[0] == silu)) || key->src[0] != out ||
        x->type != GGML_TYPE_F32 || x->ne[0] != FN_GDN_HEAD || heads < 1 || heads > FN_HCN_MAX_NW || nt < 1 || nt > FN_MAX_T ||
        x->ne[3] != 1 || !ggml_is_contiguous(x) || !ggml_are_same_shape(x, zr) || zr->type != GGML_TYPE_F32 || !ggml_is_contiguous(zr) ||
        !ggml_are_same_shape(x, out) || out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) ||
        w->type != GGML_TYPE_F32 || ggml_nelements(w) != FN_GDN_HEAD || !ggml_is_contiguous(w) ||
        !fn_hc_aligned(x) || !fn_hc_aligned(zr) || !fn_hc_aligned(out) || !fn_hc_aligned(w) ||
        (out->data != x->data && fn_overlap(out, x)) || (out->data != zr->data && fn_overlap(out, zr)) ||
        key->ne[0] != heads*FN_GDN_HEAD || key->ne[1] != nt || key->data != out->data) {
        return 0;
    }
    const int nvals = (int) (heads*FN_GDN_HEAD);
    fn_act_slot & slot = fn_act_put(ctx, key, nvals, (int) nt, true);
    fn_gdn_out<<<(unsigned) nt, (unsigned) (heads*WARP_SIZE), 0, ctx.stream()>>>(
        (const float *) x->data, x->nb[2]/sizeof(float), (const float *) zr->data, zr->nb[2]/sizeof(float),
        (const float *) w->data, ggml_get_op_params_f32(rms, 0), is_silu, fn_gain(nvals),
        (float *) out->data, out->nb[2]/sizeof(float), fn_act_X(slot), fn_act_xs(slot));
    CUDA_CHECK(cudaGetLastError());
    return 5;
}

// CONT (of the gate, a view of the q projection's output), UNARY (sigmoid), MUL from node i: fn_gate_out
int ggml_cuda_fn_gate_out(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if (!fn_pattern_on(FN_PAT_GATE_OUT) || i + 2 >= cgraph->n_nodes || cgraph->nodes[i]->op != GGML_OP_CONT) {
        return 0;
    }
    static const ggml_op ops[] = { GGML_OP_CONT, GGML_OP_UNARY, GGML_OP_MUL };
    if (!fn_pattern_closed(cgraph, i, ops, 3)) {
        return 0;
    }
    ggml_tensor * const * n = cgraph->nodes + i;
    const ggml_tensor * cont = n[0];
    const ggml_tensor * sig  = n[1];
    ggml_tensor *       out  = n[2];
    const ggml_tensor * g    = cont->src[0];
    const ggml_tensor * x    = out->src[0] == sig ? out->src[1] : out->src[0];
    const int64_t d  = g->ne[0];
    const int64_t nv = cont->ne[0];
    const int64_t nt = cont->ne[1];
    if (ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || sig->src[0] != cont || (out->src[0] != sig && out->src[1] != sig) ||
        x == sig || g->type != GGML_TYPE_F32 || g->nb[0] != sizeof(float) || d % 4 != 0 || g->ne[0]*g->ne[1] != nv || g->ne[2] != nt ||
        g->ne[3] != 1 || cont->ne[2] != 1 || nv % (4*WARP_SIZE) != 0 || nv/4 > 1024 || nt < 1 || nt > FN_MAX_T ||
        x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] != nv || x->ne[1] != nt || !ggml_are_same_shape(x, out) ||
        out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) ||
        !fn_hc_aligned(g) || !fn_hc_aligned(x) || !fn_hc_aligned(out) || g->nb[1] % 16 != 0 || g->nb[2] % 16 != 0 ||
        (out->data != x->data && fn_overlap(out, x)) || fn_overlap(out, g->view_src != nullptr ? g->view_src : g)) {
        return 0;
    }
    fn_act_slot & slot = fn_act_put(ctx, out, (int) nv, (int) nt, true);
    fn_gate_out<<<(unsigned) nt, (unsigned) (nv/4), 0, ctx.stream()>>>(
        (const float *) x->data, x->nb[1]/sizeof(float), (const float *) g->data, g->nb[1]/sizeof(float), g->nb[2]/sizeof(float),
        (int) (d/4), fn_gain((int) nv), (float *) out->data, out->nb[1]/sizeof(float), fn_act_X(slot), fn_act_xs(slot));
    CUDA_CHECK(cudaGetLastError());
    return 2;
}

// The input side of a recurrent layer (fn_gdn_pre). Its nodes are in two runs:
//   at the CONCAT of the conv input: CONCAT (state, x transposed), then per rollback slot VIEW, CONT, VIEW, CPY
//   at the SSM_CONV: SSM_CONV, UNARY (SiLU), VIEW, RMS_NORM, SCALE (q), VIEW, RMS_NORM, SCALE (k), VIEW (v),
//                    RESHAPE, ADD, UNARY (softplus), MUL, RESHAPE (the gate)
// The first run is matched with the second and skipped; the kernel is launched at the SSM_CONV, when the other
// projection that it reads (alpha) is computed as well.
static bool fn_gdn_pre_match(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int ic, fn_gdn_pre_args & a, int & i_conv,
                             int & n_slots) {
    ggml_tensor * const * n = cgraph->nodes + ic;
    const ggml_tensor * concat = n[0];
    if (concat->op != GGML_OP_CONCAT || ((const int32_t *) concat->op_params)[0] != 0 || concat->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * st = concat->src[0];
    const ggml_tensor * tr = concat->src[1];
    if (tr->op != GGML_OP_TRANSPOSE || st->type != GGML_TYPE_F32 || !ggml_is_contiguous(st)) {
        return false;
    }
    const ggml_tensor * x = tr->src[0];
    const int64_t C  = x->ne[0];
    const int64_t nt = x->ne[1];
    const int64_t K1 = st->ne[0];
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1 || nt < 1 || nt > FN_MAX_T ||
        K1 != 3 || st->ne[1] != C || st->ne[2] != 1 || concat->ne[0] != K1 + nt || concat->ne[1] != C || concat->ne[2] != 1 ||
        C % FN_GDN_HEAD != 0 || tr->data != x->data || !fn_hc_aligned(x)) {
        return false;
    }
    // the rollback slots
    n_slots = 0;
    while (n_slots < FN_GDN_PRE_SLOTS && ic + 4*n_slots + 4 < cgraph->n_nodes) {
        const ggml_tensor * tail = n[1 + 4*n_slots];
        const ggml_tensor * cont = n[2 + 4*n_slots];
        const ggml_tensor * dst  = n[3 + 4*n_slots];
        const ggml_tensor * cpy  = n[4 + 4*n_slots];
        if (tail->op != GGML_OP_VIEW || cont->op != GGML_OP_CONT || dst->op != GGML_OP_VIEW || cpy->op != GGML_OP_CPY) {
            break;
        }
        const int64_t s_idx = std::max<int64_t>(0, nt - n_slots);
        if (tail->view_src != concat || tail->data != (const char *) concat->data + s_idx*sizeof(float) || tail->ne[0] != K1 ||
            tail->ne[1] != C || tail->nb[1] != concat->nb[1] || cont->src[0] != tail || cpy->src[0] != cont || cpy->src[1] != dst ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ggml_nelements(dst) != K1*C || !fn_hc_aligned(dst) ||
            ggml_node_get_use_count(cgraph, ic + 1 + 4*n_slots) != 1 || ggml_node_get_use_count(cgraph, ic + 2 + 4*n_slots) != 1) {
            return false;
        }
        a.slot[n_slots] = (float *) dst->data;
        n_slots++;
    }
    if (n_slots < 1 || ggml_node_get_use_count(cgraph, ic) != n_slots + 1) {
        return false;
    }
    for (int sl = n_slots; sl < FN_GDN_PRE_SLOTS; ++sl) {
        a.slot[sl] = nullptr;
    }
    // the convolution of this input
    i_conv = -1;
    for (int j = ic + 4*n_slots + 1; j < std::min(cgraph->n_nodes, ic + 4*n_slots + 48); ++j) {
        if (cgraph->nodes[j]->op == GGML_OP_SSM_CONV && cgraph->nodes[j]->src[0] == concat) {
            i_conv = j;
            break;
        }
    }
    static const ggml_op ops[] = { GGML_OP_SSM_CONV, GGML_OP_UNARY, GGML_OP_VIEW, GGML_OP_RMS_NORM, GGML_OP_SCALE, GGML_OP_VIEW,
                                   GGML_OP_RMS_NORM, GGML_OP_SCALE, GGML_OP_VIEW, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY,
                                   GGML_OP_MUL, GGML_OP_RESHAPE };
    // the views of y, the scaled norms and the gate are read by the delta rule
    if (i_conv < 0 || !ggml_cuda_fn_pattern_closed(cgraph, i_conv, ops, 14, (1u << 4) | (1u << 7) | (1u << 8) | (1u << 12))) {
        return false;
    }
    ggml_tensor * const * m = cgraph->nodes + i_conv;
    const ggml_tensor * w     = m[0]->src[1];
    const ggml_tensor * y     = m[1];
    const ggml_tensor * qv    = m[2];
    const ggml_tensor * q     = m[4];
    const ggml_tensor * kv    = m[5];
    const ggml_tensor * k     = m[7];
    const ggml_tensor * vv    = m[8];
    const ggml_tensor * alpha = m[9];
    const ggml_tensor * add   = m[10];
    const ggml_tensor * gate  = m[12];
    const ggml_tensor * dt    = add->src[0] == alpha ? add->src[1] : add->src[0];
    const ggml_tensor * ga    = gate->src[0] == m[11] ? gate->src[1] : gate->src[0];
    const int64_t n_head_k = qv->ne[1];
    const int64_t n_gate   = alpha->ne[0];
    if (ggml_get_unary_op(m[1]) != GGML_UNARY_OP_SILU || m[1]->src[0] != m[0] || ggml_get_unary_op(m[11]) != GGML_UNARY_OP_SOFTPLUS ||
        w->type != GGML_TYPE_F32 || w->ne[0] != K1 + 1 || w->ne[1] != C || !ggml_is_contiguous(w) || !fn_hc_aligned(w) ||
        y->type != GGML_TYPE_F32 || y->ne[0] != C || y->ne[1] != nt || y->ne[2] != 1 || !ggml_is_contiguous(y) || !fn_hc_aligned(y) ||
        qv->view_src != y || qv->data != y->data || qv->ne[0] != FN_GDN_HEAD || qv->ne[2] != nt || qv->nb[2] != y->nb[1] ||
        kv->view_src != y || kv->data != (const char *) y->data + n_head_k*FN_GDN_HEAD*sizeof(float) || kv->ne[0] != FN_GDN_HEAD ||
        kv->ne[1] != n_head_k || kv->nb[2] != y->nb[1] ||
        vv->view_src != y || vv->data != (const char *) y->data + 2*n_head_k*FN_GDN_HEAD*sizeof(float) || 2*n_head_k*FN_GDN_HEAD > C ||
        m[3]->src[0] != qv || q->src[0] != m[3] || m[6]->src[0] != kv || k->src[0] != m[6] ||
        ggml_get_op_params_f32(m[3], 0) != ggml_get_op_params_f32(m[6], 0) ||
        ggml_get_op_params_f32(q, 0) != ggml_get_op_params_f32(k, 0) || ggml_get_op_params_f32(q, 1) != 0.0f ||
        ggml_get_op_params_f32(k, 1) != 0.0f ||
        q->type != GGML_TYPE_F32 || !ggml_is_contiguous(q) || !fn_hc_aligned(q) || !ggml_are_same_shape(q, qv) ||
        k->type != GGML_TYPE_F32 || !ggml_is_contiguous(k) || !fn_hc_aligned(k) || !ggml_are_same_shape(k, kv) ||
        (add->src[0] != alpha && add->src[1] != alpha) || m[11]->src[0] != add || (gate->src[0] != m[11] && gate->src[1] != m[11]) ||
        alpha->type != GGML_TYPE_F32 || alpha->ne[1] != nt || alpha->ne[2] != 1 || !ggml_is_contiguous(alpha) ||
        dt->type != GGML_TYPE_F32 || ggml_nelements(dt) != n_gate || !ggml_is_contiguous(dt) ||
        ga->type != GGML_TYPE_F32 || ggml_nelements(ga) != n_gate || !ggml_is_contiguous(ga) ||
        gate->type != GGML_TYPE_F32 || !ggml_are_same_shape(gate, alpha) || !ggml_is_contiguous(gate) ||
        (y->data != x->data && fn_overlap(y, x)) || fn_overlap(q, x) || fn_overlap(k, x) || fn_overlap(q, y) || fn_overlap(k, y) ||
        fn_overlap(q, k) || (gate->data != alpha->data && fn_overlap(gate, alpha)) || fn_overlap(gate, x) || fn_overlap(gate, y) ||
        fn_overlap(gate, q) || fn_overlap(gate, k)) {
        return false;
    }
    // x and the state are read at the convolution, later than their nodes read them: nothing computed between the
    // two runs may reuse their memory (a gather may be deferred further, it is checked as if it ran here)
    auto is_view = [](const ggml_tensor * t) {
        return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
               t->op == GGML_OP_TRANSPOSE || ggml_is_empty(t);
    };
    for (int j = ic + 4*n_slots + 1; j < i_conv; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (is_view(t)) {
            continue;
        }
        if (t->op != GGML_OP_GET_ROWS || fn_overlap(t, x) || fn_overlap(t, st) || fn_overlap(t, alpha)) {
            return false;
        }
    }
    // the state: the row of the cache that a deferred gather names, or the gathered copy
    if (ctx.gdn_gather_owner == concat && ctx.gdn_gather_node != nullptr) {
        const ggml_tensor * gr = ctx.gdn_gather_node;
        if (gr->src[0]->type != GGML_TYPE_F32 || gr->src[0]->nb[1] % (4*sizeof(float)) != 0 || !fn_hc_aligned(gr->src[0]) ||
            ggml_nelements(gr->src[1]) != 1 || gr->ne[0] != K1*C) {
            return false;
        }
        // the gathered copy is not written: only this CONCAT may read it
        auto root = [](const ggml_tensor * t) {
            while (t->view_src != nullptr) {
                t = t->view_src;
            }
            return t;
        };
        for (int j = ic + 1; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * t = cgraph->nodes[j];
            if (is_view(t)) {
                continue;
            }
            for (int q0 = 0; q0 < GGML_MAX_SRC && t->src[q0] != nullptr; ++q0) {
                if (root(t->src[q0]) == gr) {
                    return false;
                }
            }
        }
        a.cache = (const float *) gr->src[0]->data;
        a.rows  = ctx.gdn_rows_scratch;
        a.crow  = gr->src[0]->nb[1]/sizeof(float);
    } else {
        if (!fn_hc_aligned(st) || fn_overlap(st, y) || fn_overlap(st, q) || fn_overlap(st, k) || fn_overlap(st, gate)) {
            return false;
        }
        a.cache = (const float *) st->data;
        a.rows  = nullptr;
        a.crow  = 0;
    }
    a.x          = (const float *) x->data;
    a.w          = (const float *) w->data;
    a.y          = (float *) y->data;
    a.q          = (float *) q->data;
    a.k          = (float *) k->data;
    a.alpha      = (const float *) alpha->data;
    a.dt         = (const float *) dt->data;
    a.a          = (const float *) ga->data;
    a.gate       = (float *) gate->data;
    a.sx_t       = (int) (x->nb[1]/sizeof(float));
    a.sy_t       = (int) (y->nb[1]/sizeof(float));
    a.sq_t       = (int) (q->nb[2]/sizeof(float));
    a.sa_t       = (int) (alpha->nb[1]/sizeof(float));
    a.sg_t       = (int) (gate->nb[1]/sizeof(float));
    a.nt         = (int) nt;
    a.n_slots    = n_slots;
    a.n_head_k   = (int) n_head_k;
    a.n_heads    = (int) (C/FN_GDN_HEAD);
    a.n_gate     = (int) n_gate;
    a.eps        = ggml_get_op_params_f32(m[3], 0);
    a.norm_scale = ggml_get_op_params_f32(q, 0);
    return true;
}

int ggml_cuda_fn_gdn_pre_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if (!fn_pattern_on(FN_PAT_GDN_PRE) || cgraph->nodes[i]->op != GGML_OP_CONCAT) {
        return 0;
    }
    fn_gdn_pre_args a;
    int i_conv  = -1;
    int n_slots = 0;
    if (!fn_gdn_pre_match(ctx, cgraph, i, a, i_conv, n_slots)) {
        return 0;
    }
    static_assert(sizeof(a) <= sizeof(ctx.fn_gdn_pre_args), "fn_gdn_pre_args");
    memcpy(ctx.fn_gdn_pre_args, &a, sizeof(a));
    ctx.fn_gdn_pre_conv = cgraph->nodes[i_conv];
    if (ctx.gdn_gather_owner == cgraph->nodes[i]) {
        ctx.gdn_gather_clear();
    }
    return 4*n_slots;
}

int ggml_cuda_fn_gdn_pre(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if (ctx.fn_gdn_pre_conv == nullptr || ctx.fn_gdn_pre_conv != cgraph->nodes[i]) {
        return 0;
    }
    fn_gdn_pre_args a;
    memcpy(&a, ctx.fn_gdn_pre_args, sizeof(a));
    ctx.fn_gdn_pre_conv = nullptr;
    fn_gdn_pre<<<a.n_heads + 1, WARP_SIZE, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
    return 13;
}

// The pooled keys of a QSA layer, from the FILL of its first node: FILL, CONCAT, RESHAPE, VIEW, SET_ROWS (the raw
// keys into the cache), VIEW, GET_ROWS, RESHAPE (the members of the blocks to re-pool), VIEW, CONT, kpool - 1 times
// VIEW, ADD and the SCALE of their mean -> fn_qsa_pool
int ggml_cuda_fn_qsa_pool(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if (!fn_pattern_on(FN_PAT_QSA_POOL) || i + 11 >= cgraph->n_nodes || cgraph->nodes[i]->op != GGML_OP_FILL) {
        return 0;
    }
    ggml_tensor * const * n = cgraph->nodes + i;
    if (n[7]->op != GGML_OP_RESHAPE || n[7]->ne[1] < 2 || n[7]->ne[1] > 8) {
        if (fn_debug() && n[1]->op == GGML_OP_CONCAT && n[2]->op == GGML_OP_RESHAPE) {
            fprintf(stderr, "fn-decline: qsa_pool at %d:", i);
            for (int k = 0; k < 24 && i + k < cgraph->n_nodes; ++k) {
                fprintf(stderr, " %s[%lld,%lld,%lld]", ggml_op_name(n[k]->op), (long long) n[k]->ne[0], (long long) n[k]->ne[1], (long long) n[k]->ne[2]);
            }
            fprintf(stderr, "\n");
        }
        return 0;
    }
    const int64_t kpool = n[7]->ne[1];
    ggml_op ops[32] = { GGML_OP_FILL, GGML_OP_CONCAT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_SET_ROWS, GGML_OP_VIEW,
                        GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_CONT };
    int n_ops = 10;
    for (int m = 1; m < kpool; ++m) {
        ops[n_ops++] = GGML_OP_VIEW;
        ops[n_ops++] = GGML_OP_ADD;
    }
    ops[n_ops++] = GGML_OP_SCALE;
    if (!fn_pattern_closed(cgraph, i, ops, n_ops)) {
        fn_pattern_why("qsa_pool", cgraph, i, ops, n_ops);
        return 0;
    }
    const ggml_tensor * concat   = n[1];
    const ggml_tensor * k_raw    = concat->src[0];
    const ggml_tensor * set_rows = n[4];
    const ggml_tensor * k_idxs   = set_rows->src[1];
    const ggml_tensor * cache    = set_rows->src[2];
    const ggml_tensor * keys     = n[5];              // the cells' raw keys: the first half of the rows
    const ggml_tensor * get_rows = n[6];
    const ggml_tensor * pool_idx = get_rows->src[1];
    ggml_tensor *       out      = n[n_ops - 1];
    const int64_t d     = k_raw->ne[0];
    const int64_t nt    = k_raw->ne[1];
    const int64_t n_new = n[7]->ne[2];
    // the wiring: [k_raw | 0] rows scattered at k_idxs, the member rows gathered from the same cache, summed in order
    bool ok = concat->src[1] == n[0] && ggml_get_op_params_f32(n[0], 0) == 0.0f && ((const int32_t *) concat->op_params)[0] == 0 &&
              n[2]->src[0] == concat && n[3]->view_src != nullptr && n[3]->data == concat->data && set_rows->src[0] == n[3] &&
              keys->view_src == cache && keys->data == cache->data && get_rows->src[0] == keys &&
              n[7]->src[0] == get_rows && n[9]->src[0] == n[8] && n[8]->data == get_rows->data &&
              out->src[0] == n[n_ops - 2] && ggml_get_op_params_f32(out, 1) == 0.0f;
    for (int m = 1; m < kpool && ok; ++m) {
        const ggml_tensor * view = n[8 + 2*m];
        const ggml_tensor * add  = n[9 + 2*m];
        ok = view->data == (const char *) get_rows->data + m*get_rows->nb[1] && add->src[0] == n[m == 1 ? 9 : 7 + 2*m] && add->src[1] == view;
    }
    if (!ok) {
        fn_decline("qsa_pool: wiring", i);
    }
    ok = ok && k_raw->type == GGML_TYPE_F32 && ggml_is_contiguous(k_raw) && d <= 256 && nt >= 1 && nt <= FN_MAX_T &&
         n[0]->ne[0] == d && n[3]->ne[0] == 2*d && n[3]->ne[1] == nt &&
         k_idxs->type == GGML_TYPE_I64 && ggml_nelements(k_idxs) == nt && cache->type == GGML_TYPE_F16 && cache->ne[0] == 2*d &&
         cache->nb[0] == sizeof(half) && keys->ne[0] == d && keys->nb[1] == cache->nb[1] &&
         pool_idx->type == GGML_TYPE_I32 && ggml_nelements(pool_idx) == kpool*n_new && ggml_is_contiguous(pool_idx) &&
         get_rows->type == GGML_TYPE_F32 && get_rows->ne[0] == d && get_rows->ne[1] == kpool*n_new &&
         out->type == GGML_TYPE_F32 && ggml_is_contiguous(out) && out->ne[0] == d && out->ne[1] == n_new && n_new <= 64;
    if (!ok) {
        if (fn_debug()) {
            fprintf(stderr, "fn-decline: qsa_pool at %d: k_raw %s [%lld,%lld] cont %d, idxs %s n %lld, cache %s [%lld,%lld] nb0 %zu, keys [%lld] nb1 %zu/%zu, "
                    "pool_idx %s n %lld, rows %s [%lld,%lld], out %s [%lld,%lld], kpool %lld n_new %lld\n", i,
                    ggml_type_name(k_raw->type), (long long) k_raw->ne[0], (long long) k_raw->ne[1], ggml_is_contiguous(k_raw),
                    ggml_type_name(k_idxs->type), (long long) ggml_nelements(k_idxs),
                    ggml_type_name(cache->type), (long long) cache->ne[0], (long long) cache->ne[1], cache->nb[0],
                    (long long) keys->ne[0], keys->nb[1], cache->nb[1],
                    ggml_type_name(pool_idx->type), (long long) ggml_nelements(pool_idx),
                    ggml_type_name(get_rows->type), (long long) get_rows->ne[0], (long long) get_rows->ne[1],
                    ggml_type_name(out->type), (long long) out->ne[0], (long long) out->ne[1], (long long) kpool, (long long) n_new);
        }
        return 0;
    }
    fn_qsa_pool<<<1, 256, 0, ctx.stream()>>>(
        (const float *) k_raw->data, k_raw->nb[1]/sizeof(float), (int) d, (int) nt, (const int64_t *) k_idxs->data,
        (half *) cache->data, cache->nb[1]/sizeof(half), (const int32_t *) pool_idx->data, (int) kpool, (int) n_new,
        ggml_get_op_params_f32(out, 0), (float *) out->data, out->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    return n_ops - 1;
}

// The 28 nodes of the selection mask of a QSA layer, from the FILL of its first node
static const ggml_op fn_qsa_sel_ops[] = {
    GGML_OP_FILL, GGML_OP_REPEAT, GGML_OP_RESHAPE,                                     // the zeros of the selection
    GGML_OP_CPY, GGML_OP_RESHAPE, GGML_OP_GET_ROWS, GGML_OP_SCALE, GGML_OP_CLAMP,      // slots as floats; live blocks
    GGML_OP_REPEAT, GGML_OP_RESHAPE, GGML_OP_CPY, GGML_OP_SCALE, GGML_OP_CLAMP,        // ... per slot; live tail
    GGML_OP_CONCAT, GGML_OP_FILL, GGML_OP_CUMSUM, GGML_OP_SCALE,                       // live; the dump rows
    GGML_OP_SUB, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_CPY, GGML_OP_RESHAPE,               // dump + live*(slot - dump)
    GGML_OP_FILL, GGML_OP_REPEAT, GGML_OP_RESHAPE, GGML_OP_SET_ROWS, GGML_OP_VIEW,     // the scatter into -inf
    GGML_OP_ADD,                                                                        // + kq_mask
};
#define FN_QSA_SEL_N_OPS ((int) (sizeof(fn_qsa_sel_ops)/sizeof(fn_qsa_sel_ops[0])))

struct fn_qsa_sel_nodes {
    const ggml_tensor * sel_idx;
    const ggml_tensor * score;
    const ggml_tensor * top_k;
    const ggml_tensor * kq_mask;
    ggml_tensor *       out;
    int64_t             n_sel, nt, kpool, n_kv;
};

// the wiring and the shapes of a selection whose ops matched
static bool fn_qsa_sel_wiring(const ggml_cgraph * cgraph, const int i, fn_qsa_sel_nodes & s) {
    ggml_tensor * const * n = cgraph->nodes + i;
    s.sel_idx = n[3]->src[0];
    s.score   = n[4]->src[0];
    s.top_k   = n[5]->src[1];
    s.out     = n[27];
    s.kq_mask = s.out->src[0] == n[26] ? s.out->src[1] : s.out->src[0];
    s.n_sel   = s.sel_idx->ne[0];
    s.nt      = s.sel_idx->ne[1];
    s.kpool   = n[8]->ne[0];
    s.n_kv    = s.out->ne[0];
    const ggml_tensor * sel_idx = s.sel_idx;
    const ggml_tensor * score   = s.score;
    const ggml_tensor * top_k   = s.top_k;
    const ggml_tensor * out     = s.out;
    const ggml_tensor * kq_mask = s.kq_mask;
    const int64_t n_sel = s.n_sel;
    const int64_t nt    = s.nt;
    const int64_t kpool = s.kpool;
    const int64_t n_kv  = s.n_kv;
    auto params = [](const ggml_tensor * t, const float a, const float b) {
        return ggml_get_op_params_f32(t, 0) == a && ggml_get_op_params_f32(t, 1) == b;
    };
    // the wiring that the values depend on; the rest follows from the op sequence
    if (n[5]->src[0] != n[4] || n[6]->src[0] != n[5] || n[7]->src[0] != n[6] || n[8]->src[0] != n[7] ||
        n[11]->src[0] != n[10] || n[12]->src[0] != n[11] || n[10]->src[0]->type != GGML_TYPE_I32 ||
        n[13]->src[0] != n[9] || n[13]->src[1] != n[12] || n[14]->src[0] != n[13] || n[15]->src[0] != n[14] || n[16]->src[0] != n[15] ||
        n[17]->src[0] != n[3] || n[17]->src[1] != n[16] || n[18]->src[0] != n[17] || n[18]->src[1] != n[13] ||
        n[19]->src[0] != n[18] || n[19]->src[1] != n[16] || n[20]->src[0] != n[19] || n[21]->src[0] != n[20] ||
        n[25]->src[0] != n[2] || n[25]->src[1] != n[21] || n[25]->src[2] != n[24] || n[26]->src[0] != n[25] ||
        (out->src[0] != n[26] && out->src[1] != n[26]) ||
        ggml_get_op_params_f32(n[0], 0) != 0.0f || ggml_get_op_params_f32(n[14], 0) != 1.0f || !std::isinf(ggml_get_op_params_f32(n[22], 0)) ||
        ggml_get_op_params_f32(n[22], 0) > 0.0f ||
        !params(n[6], 1.0f, 1.0f) || !params(n[7], 0.0f, 1.0f) || !params(n[11], -1.0f, (float) n_kv) || !params(n[12], 0.0f, 1.0f) ||
        !params(n[16], 1.0f, (float) (n_kv - 1))) {
        return fn_decline("qsa_sel: wiring", i);
    }
    if (sel_idx->type != GGML_TYPE_I32 || !ggml_is_contiguous(sel_idx) || score->type != GGML_TYPE_F32 || !ggml_is_contiguous(score) ||
        score->ne[1] != nt || top_k->type != GGML_TYPE_I32 || top_k->nb[0] != sizeof(int32_t) || top_k->ne[1] != nt ||
        n_sel != kpool*top_k->ne[0] + kpool - 1 || n[10]->src[0]->ne[0] != kpool - 1 ||
        out->type != GGML_TYPE_F16 || !ggml_is_contiguous(out) || out->ne[1]*out->ne[2]*out->ne[3] != nt ||
        kq_mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(kq_mask) || kq_mask->ne[0] != n_kv ||
        kq_mask->ne[1]*kq_mask->ne[2]*kq_mask->ne[3] != nt || n[25]->ne[1] != n_kv + n_sel) {
        return fn_decline("qsa_sel: shapes", i);
    }
    return true;
}

// The selection mask of a QSA layer -> fn_qsa_sel
int ggml_cuda_fn_qsa_sel(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    constexpr int n_ops = FN_QSA_SEL_N_OPS;
    if (!fn_pattern_on(FN_PAT_QSA_SEL) || cgraph->nodes[i]->op != GGML_OP_FILL) {
        return 0;
    }
    if (!fn_pattern_closed(cgraph, i, fn_qsa_sel_ops, n_ops)) {
        fn_pattern_why("qsa_sel", cgraph, i, fn_qsa_sel_ops, n_ops);
        return 0;
    }
    fn_qsa_sel_nodes s;
    if (!fn_qsa_sel_wiring(cgraph, i, s)) {
        return 0;
    }
    // the slots are staged as 16-bit cells in shared memory; the mask is a graph input
    const size_t smem = (size_t) s.nt*s.n_sel*sizeof(unsigned short);
    if (s.n_kv >= (int64_t) FN_QSA_SEL_DEAD || smem > 40000 || fn_overlap(s.out, s.kq_mask)) {
        fn_decline("qsa_sel: size", i);
        return 0;
    }
    fn_qsa_sel<<<1, FN_QSA_SEL_THREADS, smem, ctx.stream()>>>(
        (const int32_t *) s.sel_idx->data, s.sel_idx->nb[1]/sizeof(int32_t), (int) s.n_sel, (int) (s.kpool*s.top_k->ne[0]), (int) s.kpool,
        (const float *) s.score->data, s.score->nb[1]/sizeof(float), (const int32_t *) s.top_k->data, s.top_k->nb[1]/sizeof(int32_t),
        (int) s.n_kv, (int) s.nt, (const half *) s.kq_mask->data, s.n_kv, (half *) s.out->data, s.n_kv);
    CUDA_CHECK(cudaGetLastError());
    return n_ops - 1;
}

// The selection and the attention over it: the nodes of the selection, RESHAPE (the mask), FLASH_ATTN_EXT -> fn_qsa_attn
int ggml_cuda_fn_qsa_attn(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    constexpr int n_sel_ops = FN_QSA_SEL_N_OPS;
    constexpr int n_ops     = n_sel_ops + 2;
    if (!fn_pattern_on(FN_PAT_QSA_ATTN) || cgraph->nodes[i]->op != GGML_OP_FILL || i + n_ops > cgraph->n_nodes ||
        cgraph->nodes[i + n_ops - 1]->op != GGML_OP_FLASH_ATTN_EXT) {
        return 0;
    }
    ggml_op ops[n_ops];
    std::copy(fn_qsa_sel_ops, fn_qsa_sel_ops + n_sel_ops, ops);
    ops[n_sel_ops]     = GGML_OP_RESHAPE;
    ops[n_sel_ops + 1] = GGML_OP_FLASH_ATTN_EXT;
    if (!fn_pattern_closed(cgraph, i, ops, n_ops)) {
        fn_pattern_why("qsa_attn", cgraph, i, ops, n_ops);
        return 0;
    }
    fn_qsa_sel_nodes s;
    if (!fn_qsa_sel_wiring(cgraph, i, s)) {
        return 0;
    }
    ggml_tensor * const * n = cgraph->nodes + i;
    const ggml_tensor * mask = n[n_sel_ops];
    ggml_tensor *       out  = n[n_sel_ops + 1];
    const ggml_tensor * q    = out->src[0];
    const ggml_tensor * k    = out->src[1];
    const ggml_tensor * v    = out->src[2];
    const int64_t d      = FN_QSA_ATTN_D;
    const int64_t n_head = q->ne[2];
    auto row16 = [](const ggml_tensor * t, const size_t nb) {
        return ((uintptr_t) t->data & 0xF) == 0 && nb % 16 == 0;
    };
    if (mask->src[0] != s.out || out->src[3] != mask || ggml_get_op_params_f32(out, 1) != 0.0f || ggml_get_op_params_f32(out, 2) != 0.0f ||
        q->type != GGML_TYPE_F32 || q->ne[0] != d || q->ne[1] != s.nt || q->ne[3] != 1 || q->nb[0] != sizeof(float) ||
        !row16(q, q->nb[1]) || q->nb[2] % 16 != 0 || n_head < 1 || s.nt > FN_MAX_T ||
        k->type != GGML_TYPE_F16 || k->ne[0] != d || k->ne[1] != s.n_kv || k->ne[2] != 1 || k->nb[0] != sizeof(half) || !row16(k, k->nb[1]) ||
        v->type != GGML_TYPE_F16 || v->ne[0] != d || v->ne[1] != s.n_kv || v->ne[2] != 1 || v->nb[0] != sizeof(half) || !row16(v, v->nb[1]) ||
        out->type != GGML_TYPE_F32 || out->ne[0] != d || out->ne[1] != n_head || out->ne[2] != s.nt || !ggml_is_contiguous(out) ||
        !fn_hc_aligned(out) || s.n_sel > FN_QSA_ATTN_MAX_SEL || n_head > FN_QSA_ATTN_MAX_HEAD || n_head % 4 != 0 || s.n_kv < 1 || s.n_kv + s.n_sel >= ((int64_t) 1 << 24)) {
        if (fn_debug()) {
            fprintf(stderr, "fn-decline: qsa_attn at %d: shapes: q [%lld,%lld,%lld] k %s [%lld,%lld,%lld] nb1 %zu out [%lld,%lld,%lld] n_sel %lld n_kv %lld nt %lld\n", i,
                    (long long) q->ne[0], (long long) q->ne[1], (long long) q->ne[2], ggml_type_name(k->type), (long long) k->ne[0],
                    (long long) k->ne[1], (long long) k->ne[2], k->nb[1], (long long) out->ne[0], (long long) out->ne[1], (long long) out->ne[2],
                    (long long) s.n_sel, (long long) s.n_kv, (long long) s.nt);
        }
        return 0;
    }
    // a block writes its output while other blocks still read: the output may share memory with the selection
    // only (its buffers are free by then), which is then staged first
    for (const ggml_tensor * x : { q->view_src != nullptr ? q->view_src : q, s.kq_mask }) {
        if (fn_overlap(out, x)) {
            fn_decline("qsa_attn: overlap", i);
            return 0;
        }
    }
    fn_qsa_attn_args a;
    a.cells   = nullptr;
    a.sel_idx = (const int32_t *) s.sel_idx->data;
    a.score   = (const float *) s.score->data;
    a.top_k   = (const int32_t *) s.top_k->data;
    a.kq_mask = (const half *) s.kq_mask->data;
    a.q       = (const float *) q->data;
    a.K       = (const half *) k->data;
    a.V       = (const half *) v->data;
    a.out     = (float *) out->data;
    a.s_sel   = (int) (s.sel_idx->nb[1]/sizeof(int32_t));
    a.s_score = (int) (s.score->nb[1]/sizeof(float));
    a.s_topk  = (int) (s.top_k->nb[1]/sizeof(int32_t));
    a.s_kq    = (int) s.n_kv;
    a.sq_t    = (int) (q->nb[1]/sizeof(float));
    a.sq_h    = (int) (q->nb[2]/sizeof(float));
    a.sk      = (int) (k->nb[1]/sizeof(half));
    a.sv      = (int) (v->nb[1]/sizeof(half));
    a.so_t    = (int) (out->nb[2]/sizeof(float));
    a.so_h    = (int) (out->nb[1]/sizeof(float));
    a.n_sel   = (int) s.n_sel;
    a.n_top   = (int) (s.kpool*s.top_k->ne[0]);
    a.kpool   = (int) s.kpool;
    a.n_kv    = (int) s.n_kv;
    a.scale   = ggml_get_op_params_f32(out, 0);
    if (ctx.fn_qsa_cells == nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc((void **) &ctx.fn_qsa_cells, (size_t) FN_MAX_T*FN_QSA_ATTN_MAX_SEL*sizeof(int32_t)));
    }
    fn_qsa_cells<<<(unsigned) ((s.nt*s.n_sel + 255)/256), 256, 0, ctx.stream()>>>(a, (int) s.nt, ctx.fn_qsa_cells);
    a.cells   = ctx.fn_qsa_cells;
    a.n_head  = (int) n_head;
    if (ctx.fn_qsa_part == nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc((void **) &ctx.fn_qsa_part, (size_t) FN_MAX_T*FN_QSA_ATTN_MAX_HEAD*FN_QSA_ATTN_MAX_CHUNK*(FN_QSA_ATTN_D + 4)*sizeof(float)));
    }
    a.part = ctx.fn_qsa_part;
    fn_qsa_attn_launch(a, (int) s.nt, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    return n_ops - 1;
}

// ---------------------------------------------------------------------------------------------------------------
// MoE experts: the nodes

static bool fn_moe_q2(const ggml_tensor * w, const int64_t cols, const int64_t rows) {
    return w->type == GGML_TYPE_Q2_0 && w->ne[0] == cols && w->ne[1] == rows && w->ne[3] == 1 &&
           w->nb[1] == (size_t) cols/64*18 && w->nb[2] == w->nb[1]*rows && ((uintptr_t) w->data & 0x3) == 0;
}

bool ggml_cuda_fn_moe_supported(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
                                const ggml_tensor * weights, const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_CUDA_FN_MOE") == nullptr || atoi(getenv("GGML_CUDA_FN_MOE")) != 0;
    if (!enabled || !ggml_cuda_fn_enabled()) {
        return false;
    }
    const ggml_tensor * x   = gate->src[1];
    const ggml_tensor * ids = gate->src[2];
    const int64_t n_embd = gate->src[0]->ne[0];
    const int64_t n_ff   = gate->src[0]->ne[1];
    const int64_t n_used = ids->ne[0];
    const int64_t nt     = ids->ne[1];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_DP4A &&
           n_embd == 64*FN_Q2_TPR && (n_ff == 256 || n_ff == 384 || n_ff == 512 || n_ff == 768) &&
           n_embd % (4*FN_MOE_DOWN_R) == 0 && n_used <= FN_MOE_NU &&
           nt >= 1 && nt <= FN_MAX_T &&
           fn_moe_q2(gate->src[0], n_embd, n_ff) && fn_moe_q2(up->src[0], n_embd, n_ff) && fn_moe_q2(down->src[0], n_ff, n_embd) &&
           gate->src[0]->ne[2] == up->src[0]->ne[2] && gate->src[0]->ne[2] == down->src[0]->ne[2] &&
           up->src[1] == x && up->src[2] == ids && down->src[2] == ids &&
           x->type == GGML_TYPE_F32 && x->ne[0] == n_embd && x->ne[1] == 1 && x->ne[2] == nt && x->nb[0] == sizeof(float) &&
           ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) &&
           weights->type == GGML_TYPE_F32 && weights->ne[0] == 1 && weights->ne[1] == n_used && weights->ne[2] == nt &&
           weights->nb[1] == sizeof(float) &&
           dst->type == GGML_TYPE_F32 && dst->ne[0] == n_embd && dst->ne[1] == nt && ggml_is_contiguous(dst);
}

// the layout of the context's memory for a layer: [pairs*n_ff halves: the activations of h][pairs*n_ff: h]
// [pairs: scales of h][pairs: routes]; the activations are read as 16-byte words
static half * fn_moe_Xh(const ggml_backend_cuda_context & ctx) {
    return (half *) ctx.fn_moe_mem;
}

static float * fn_moe_h(const ggml_backend_cuda_context & ctx) {
    return (float *) (ctx.fn_moe_mem + (size_t) ctx.fn_moe_n_pairs*ctx.fn_moe_n_ff*sizeof(half));
}

static float * fn_moe_xsh(const ggml_backend_cuda_context & ctx) {
    return fn_moe_h(ctx) + (size_t) ctx.fn_moe_n_pairs*ctx.fn_moe_n_ff;
}

static fn_moe_route * fn_moe_route_ptr(const ggml_backend_cuda_context & ctx) {
    return (fn_moe_route *) (fn_moe_xsh(ctx) + ctx.fn_moe_n_pairs);
}

const ggml_cuda_fn_moe_route * ggml_cuda_fn_moe_routes(const ggml_backend_cuda_context & ctx) {
    return fn_moe_route_ptr(ctx);
}

void ggml_cuda_fn_moe_up(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * weights) {
    const ggml_tensor * x   = gate->src[1];
    const ggml_tensor * ids = gate->src[2];
    const int n_embd = (int) gate->src[0]->ne[0];
    const int n_ff   = (int) gate->src[0]->ne[1];
    const int n_used = (int) ids->ne[0];
    const int nt     = (int) ids->ne[1];
    const int np     = n_used*nt;

    const size_t need = (size_t) np*(sizeof(fn_moe_route) + sizeof(float)) + (size_t) np*n_ff*(sizeof(float) + sizeof(half));
    if (need > ctx.fn_moe_cap) {
        ctx.retire_mem(ctx.fn_moe_mem, ctx.fn_moe_cap);
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc((void **) &ctx.fn_moe_mem, need));
        ctx.fn_moe_cap = need;
    }
    ctx.fn_moe_n_ff    = n_ff;
    ctx.fn_moe_n_pairs = np;

    // the activations of the layer's input: the read that produced it left them under its own tensor
    const ggml_tensor * key = x->view_src != nullptr && x->view_offs == 0 ? x->view_src : x;
    const fn_act_slot & sx  = fn_act_get(ctx, key, (const float *) x->data, x->nb[2]/sizeof(float), n_embd, nt);

    fn_moe_up<<<dim3(n_ff/(FN_MOE_UP_G*FN_MOE_UP_R), np, 1), FN_MOE_UP_G*FN_Q2_TPR, 0, ctx.stream()>>>(
        (const char *) gate->src[0]->data, (const char *) up->src[0]->data, gate->src[0]->nb[1], gate->src[0]->nb[2],
        (const half2 *) fn_act_X(sx), n_embd/2, fn_act_xs(sx),
        (const int32_t *) ids->data, (int) (ids->nb[1]/sizeof(int32_t)), n_used, (int) gate->src[0]->ne[2],
        (const float *) weights->data, (int) (weights->nb[2]/sizeof(float)),
        fn_moe_h(ctx), n_ff, fn_moe_route_ptr(ctx));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_fn_moe_down(ggml_backend_cuda_context & ctx, const ggml_tensor * down, ggml_tensor * dst) {
    const ggml_tensor * ids = down->src[2];
    const int n_ff   = (int) down->src[0]->ne[0];
    const int n_embd = (int) down->src[0]->ne[1];
    const int n_used = (int) ids->ne[0];
    const int nt     = (int) ids->ne[1];
    const int np     = n_used*nt;
    GGML_ASSERT(ctx.fn_moe_n_ff == n_ff && ctx.fn_moe_n_pairs == np);
    cudaStream_t stream = ctx.stream();

    // |sum of n_ff products of weights up to 2| stays far inside fp16 with activations up to 16
    fn_act_h16<<<np, FN_ACT_TPB, 0, stream>>>(fn_moe_h(ctx), n_ff, n_ff, 16.0f, fn_moe_Xh(ctx), fn_moe_xsh(ctx));
    switch (n_ff/64) {
#define FN_CASE(TPR)                                                                                                  \
        case TPR:                                                                                                     \
            fn_moe_down<TPR><<<dim3(n_embd/(fn_moe_down_groups(TPR)*FN_MOE_DOWN_R), nt, 1),                           \
                               fn_moe_down_groups(TPR)*FN_MOE_NU*TPR, 0, stream>>>(                                   \
                (const char *) down->src[0]->data, down->src[0]->nb[1], down->src[0]->nb[2],                          \
                (const half2 *) fn_moe_Xh(ctx), fn_moe_xsh(ctx), fn_moe_route_ptr(ctx), n_used, (int) down->src[0]->ne[2], \
                (float *) dst->data, dst->nb[1]/sizeof(float));                                                       \
            break;
        FN_CASE(4)
        FN_CASE(6)
        FN_CASE(8)
        FN_CASE(12)
#undef FN_CASE
        default:
            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------
// AllReduce epilogue: the nodes after a reduced projection output y,
//   [RESHAPE] MUL_MAT (inject) SCALE SIGMOID SCALE DSV4_HC_POST RMS_NORM MUL
// computed with the reduction by fn_ar_hc.

struct fn_ar_hc_args {
    const ggml_tensor * y;       // the reduced tensor [n_embd, nt]
    const ggml_tensor * z;       // what an ADD adds to it before the write, or null
    const ggml_tensor * inject;  // MUL_MAT(w_inject [hc*n_embd -> hc], xn_prev)
    const ggml_tensor * scale0;  // SCALE before the sigmoid
    const ggml_tensor * scale1;  // SCALE after it
    const ggml_tensor * post;    // DSV4_HC_POST(y, R, gates): dst = R'
    const ggml_tensor * rms;     // RMS_NORM(R')
    const ggml_tensor * mul;     // MUL by w_norm: dst = xn
    int                 n_nodes;
    // the read that follows, if it is one the engine computes: its activations are written with the norm
    const ggml_tensor * xd;      // src[1] of the down MUL_MAT
    const ggml_tensor * pre;     // DSV4_HC_PRE
};

static bool fn_ar_hc_match(const ggml_tensor * y, ggml_tensor ** next, const int n_next, fn_ar_hc_args & a) {
    int i = 0;
    const ggml_tensor * yv = y; // what the write reads
    a.z = nullptr;
    if (n_next > 0 && next[0]->op == GGML_OP_RESHAPE && next[0]->src[0] == y) {
        yv = next[0];
        i  = 1;
    } else if (n_next > 0 && next[0]->op == GGML_OP_ADD && (next[0]->src[0] == y || next[0]->src[1] == y)) {
        yv  = next[0];
        a.z = next[0]->src[0] == y ? next[0]->src[1] : next[0]->src[0];
        i   = 1;
        if (a.z == y || !ggml_are_same_shape(a.z, y) || a.z->type != GGML_TYPE_F32 || !ggml_is_contiguous(a.z) ||
            ((uintptr_t) a.z->data & 0xF) != 0) {
            return false;
        }
    }
    if (i + 7 > n_next) {
        return false;
    }
    a.y      = y;
    a.inject = next[i];
    a.scale0 = next[i + 1];
    const ggml_tensor * sig = next[i + 2];
    a.scale1 = next[i + 3];
    a.post   = next[i + 4];
    a.rms    = next[i + 5];
    a.mul    = next[i + 6];
    a.n_nodes = i + 7;
    a.xd      = nullptr;
    a.pre     = nullptr;
    if (a.inject->op != GGML_OP_MUL_MAT || a.scale0->op != GGML_OP_SCALE || sig->op != GGML_OP_UNARY ||
        ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || a.scale1->op != GGML_OP_SCALE ||
        a.post->op != GGML_OP_DSV4_HC_POST || a.rms->op != GGML_OP_RMS_NORM || a.mul->op != GGML_OP_MUL) {
        return false;
    }
    if (a.scale0->src[0] != a.inject || sig->src[0] != a.scale0 || a.scale1->src[0] != sig ||
        a.post->src[0] != yv || a.post->src[2] != a.scale1 || a.post->src[3] != nullptr || a.rms->src[0] != a.post) {
        return false;
    }
    // nothing but the write reads the reduced values, the gates or the norm
    for (const ggml_tensor * t : { y, yv, a.inject, a.scale0, sig, a.scale1, a.rms }) {
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
    }
    const ggml_tensor * R   = a.post->src[1];
    const ggml_tensor * wi  = a.inject->src[0];
    const ggml_tensor * xpv = a.inject->src[1];                               // [hc*n_embd, nt]
    const ggml_tensor * xp  = xpv->view_src != nullptr ? xpv->view_src : xpv; // [n_embd, hc, nt]
    const int64_t n_embd = y->ne[0];
    const int64_t nt     = y->ne[1];
    if (!fn_hc_norm_supported(a.rms, a.mul) || a.post->type != GGML_TYPE_F32 || !ggml_is_contiguous(a.post) ||
        !ggml_are_same_shape(a.post, R) || R->type != GGML_TYPE_F32 || !ggml_is_contiguous(R) || R->ne[0] != n_embd ||
        R->ne[2] != nt || y->type != GGML_TYPE_F32 || !ggml_is_contiguous(y) || y->ne[2] != 1 || y->ne[3] != 1 ||
        nt > GGML_CUDA_AR_WINDOW_BLOCKS || y->nb[1] % 16 != 0 ||
        (wi->type != GGML_TYPE_F32 && wi->type != GGML_TYPE_F16 && wi->type != GGML_TYPE_BF16) || !ggml_is_contiguous(wi) ||
        wi->ne[0] != FN_HC*n_embd || wi->ne[1] != FN_HC || ggml_nrows(wi) != FN_HC ||
        xpv->type != GGML_TYPE_F32 || xpv->data != xp->data || xp->ne[0] != n_embd || xp->ne[1] != FN_HC || xp->ne[2] != nt ||
        !ggml_is_contiguous(xp) || xpv->ne[0] != FN_HC*n_embd || xpv->ne[1] != nt ||
        !fn_hc_aligned(y) || !fn_hc_aligned(R) || !fn_hc_aligned(a.post) || !fn_hc_aligned(xp)) {
        return false;
    }
    // see fn_ar_hc for what may share memory
    if (fn_overlap(a.post, a.mul)) {
        return false;
    }
    // RESHAPE, RESHAPE and the six nodes of the read
    ggml_cuda_hc_mix_args h;
    if (a.n_nodes + 8 <= n_next && next[a.n_nodes]->op == GGML_OP_RESHAPE && next[a.n_nodes + 1]->op == GGML_OP_RESHAPE &&
        fn_hc_tail_nodes(next + a.n_nodes + 2, h) && h.mul == a.mul) {
        a.xd  = h.mm_down->src[1];
        a.pre = h.pre;
    }
    return true;
}

int ggml_cuda_fn_ar_epilogue_match(const ggml_tensor * reduced, ggml_tensor ** next, const int n_next) {
    if (!ggml_cuda_fn_enabled()) {
        return 0;
    }
    static const bool enabled = getenv("GGML_CUDA_FN_AR") == nullptr || atoi(getenv("GGML_CUDA_FN_AR")) != 0;
    fn_ar_hc_args a;
    return enabled && fn_ar_hc_match(reduced, next, n_next, a) ? a.n_nodes : 0;
}

void ggml_cuda_fn_ar_epilogue(ggml_backend_cuda_context & ctx, const ggml_tensor * reduced, ggml_tensor ** next, const int n_next,
                              const int n_fused, const ggml_cuda_ar_window_io & io) {
    fn_ar_hc_args a;
    GGML_ASSERT(fn_ar_hc_match(reduced, next, n_next, a) && a.n_nodes == n_fused);
    GGML_ASSERT((reduced->flags & GGML_TENSOR_FLAG_COMPUTE) != 0);
    const ggml_tensor * y   = a.y;
    const ggml_tensor * R   = a.post->src[1];
    const ggml_tensor * wi  = a.inject->src[0];
    const ggml_tensor * xpv = a.inject->src[1];
    const ggml_tensor * xp  = xpv->view_src != nullptr ? xpv->view_src : xpv;
    const ggml_tensor * wn  = a.mul->src[1];
    const int n_embd = (int) y->ne[0];
    const int nt     = (int) y->ne[1];
    // the units go to the upper half of the slot: the lower half may hold the data of a plain reduction
    GGML_ASSERT((size_t) nt*n_embd*sizeof(float) <= io.wire_bytes/2);

    const float * p0 = (const float *) a.scale0->op_params;
    const float * p1 = (const float *) a.scale1->op_params;
    const float4 act = make_float4(p0[0], p0[1], p1[0], p1[1]);

    const fn_hc_norm_out o = fn_hc_norm_outputs(ctx, a.mul, a.xd, a.pre);
    {
        static int n_dbg = 0;
        if (getenv("GGML_CUDA_FN_DEBUG") != nullptr && n_dbg++ < 400 && (n_dbg % 97) == 0) {
            fprintf(stderr, "fn-debug: ar_hc n_next %d xd %d pre %d\n", n_next, a.xd != nullptr, a.pre != nullptr);
        }
    }

#define FN_AR_HC(T_inj)                                                                                               \
    fn_ar_hc<T_inj><<<nt, n_embd/4, 0, ctx.stream()>>>(                                                               \
        (const float *) y->data, y->nb[1]/sizeof(float),                                                              \
        a.z != nullptr ? (const float *) a.z->data : nullptr, a.z != nullptr ? a.z->nb[1]/sizeof(float) : 0,          \
        (uint4 *) (io.wire_mine + io.wire_bytes/2), (const uint4 *) (io.wire_other + io.wire_bytes/2), io.epoch, io.site, \
        (const float *) xp->data, xp->nb[1]/sizeof(float), xp->nb[2]/sizeof(float),                                   \
        (const T_inj *) wi->data, act,                                                                                \
        (const float *) R->data, R->nb[1]/sizeof(float), R->nb[2]/sizeof(float),                                      \
        (float *) a.post->data, a.post->nb[1]/sizeof(float), a.post->nb[2]/sizeof(float),                             \
        (const float *) wn->data, n_embd, ggml_get_op_params_f32(a.rms, 0),                                           \
        (float *) a.mul->data, a.mul->nb[1]/sizeof(float), a.mul->nb[2]/sizeof(float), o, ctx.fn_dbg_get())
    switch (wi->type) {
        case GGML_TYPE_F32:  FN_AR_HC(float);       break;
        case GGML_TYPE_F16:  FN_AR_HC(half);        break;
        case GGML_TYPE_BF16: FN_AR_HC(nv_bfloat16); break;
        default: GGML_ABORT("fatal error");
    }
#undef FN_AR_HC
    CUDA_CHECK(cudaGetLastError());
}

#endif // FN_STANDALONE
