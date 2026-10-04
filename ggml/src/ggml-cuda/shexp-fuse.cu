#include "shexp-fuse.cuh"
#include "convert.cuh"

#define SHEXP_QK        32
#define SHEXP_UP_RB     4   // rows of gate and up per block of the first kernel
#define SHEXP_DOWN_RB   8   // rows of down per block of the second kernel

// Q8_0 half-block -> 8 half2 in the decode order of the A16 activations (see mul_mat_vec_q8_0_a16 and hc-mix.cu).
// Loading and decoding are split so that a kernel can issue its weight loads before anything else: these
// matrices are small, so every kernel is a single round of loads and their latency is the cost.
struct shexp_raw {
    int  u[6];
    half d;
};

static __device__ __forceinline__ void shexp_q8_load(const char * __restrict__ row, const int kb, const int kh, shexp_raw & r) {
    const int boff  = kb*34;
    const int shift = (kb & 1) ? 0 : 2;
    const int * pA = (const int *) (row + boff + 2 + 8*kh - shift);
    const int * pB = pA + 4;
    r.d    = *(const half *) (row + boff);
    r.u[0] = pA[0];
    r.u[1] = pA[1];
    r.u[2] = shift ? pA[2] : 0;
    r.u[3] = pB[0];
    r.u[4] = pB[1];
    r.u[5] = shift ? pB[2] : 0;
}

static __device__ __forceinline__ void shexp_q8_decode(const shexp_raw & r, const int kb, half2 w[8], float & d) {
    const int sel = (kb & 1) ? 0x3210 : 0x5432;
    d = __half2float(r.d);
    const int A0 = __byte_perm(r.u[0], r.u[1], sel) ^ 0x80808080;
    const int A1 = __byte_perm(r.u[1], r.u[2], sel) ^ 0x80808080;
    const int B0 = __byte_perm(r.u[3], r.u[4], sel) ^ 0x80808080;
    const int B1 = __byte_perm(r.u[4], r.u[5], sel) ^ 0x80808080;
    const half2 magic = __float2half2_rn(1152.0f);
    int tt[8];
    tt[0] = __byte_perm(A0, 0x64646464, 0x5240);
    tt[1] = __byte_perm(B0, 0x64646464, 0x5240);
    tt[2] = __byte_perm(A0, 0x64646464, 0x5341);
    tt[3] = __byte_perm(B0, 0x64646464, 0x5341);
    tt[4] = __byte_perm(A1, 0x64646464, 0x5240);
    tt[5] = __byte_perm(B1, 0x64646464, 0x5240);
    tt[6] = __byte_perm(A1, 0x64646464, 0x5341);
    tt[7] = __byte_perm(B1, 0x64646464, 0x5341);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        w[i] = __hsub2(*((const half2 *) &tt[i]), magic);
    }
}

// x (nt rows of n f32, row stride sx) -> the A16 format in shared memory: per 32 values amax and values/amax as
// half2 in the decode order. Eight lanes per block of 32, all threads of the block take part.
static __device__ __forceinline__ void shexp_quantize_smem(const float * __restrict__ x, const int64_t sx, const int nt,
                                                           const int n, half2 * s_aq, float * s_yd) {
    const int nb   = n/SHEXP_QK;
    const int nthr = blockDim.x*blockDim.y;
    const int tid  = threadIdx.x + threadIdx.y*blockDim.x;
    for (int task = tid; task < nt*nb*8; task += nthr) {
        const int j  = task / (nb*8);
        const int ib = (task / 8) % nb;
        const int l  = task % 8;
        const unsigned int gmask = 0xffu << ((tid % WARP_SIZE) & ~7);
        const float4 v4 = *(const float4 *) (x + j*sx + ib*SHEXP_QK + 4*l);
        float v[4] = { v4.x, v4.y, v4.z, v4.w };
        float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
        for (int s = 4; s > 0; s >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(gmask, amax, s, 8));
        }
        const float id = amax > 0.0f ? 1.0f/amax : 0.0f;
        if (l == 0) {
            s_yd[j*nb + ib] = amax;
        }
        const int off = l >= 4 ? 1 : 0;
        const int k   = (4*l - off*16)/4;
        half2 * qs = s_aq + (j*nb + ib)*(SHEXP_QK/2);
        qs[4*k + 0 + off] = make_half2(__float2half(v[0]*id), __float2half(v[2]*id));
        qs[4*k + 2 + off] = make_half2(__float2half(v[1]*id), __float2half(v[3]*id));
    }
}

// 1. h[t][r] = silu(w_gate[r] . x[t]) * (w_up[r] . x[t]) for SHEXP_UP_RB rows per block; one half block of each row
//    per thread (the block has as many threads as a row has half blocks). Block 0 also computes the gate scalar
//    g[t] = sigmoid(w_gate_inp . x[t]) (f32 or bf16 weights).
template <int NT, typename gi_t>
static __global__ void shexp_up(const char * __restrict__ wg, const char * __restrict__ wu, const int64_t w_row,
                                const float * __restrict__ x, const int64_t sx, const int n_embd,
                                const gi_t * __restrict__ wgi, float * __restrict__ h, const int64_t sh,
                                float * __restrict__ gsc) {
#if defined(FP16_AVAILABLE)
    extern __shared__ char s_raw[];
    const int nb   = n_embd/SHEXP_QK;
    half2 *   s_aq = (half2 *) s_raw;                      // [NT][nb][16]
    float *   s_yd = (float *) (s_aq + NT*nb*(SHEXP_QK/2)); // [NT][nb]
    float *   s_red = s_yd + NT*nb;                        // [nwarps][NT][2*RB]

    const int tid  = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nw   = blockDim.y;
    const int row0 = blockIdx.x*SHEXP_UP_RB;

    // the block has one thread per half block of a row (checked by the matcher): issue the weight loads first
    const int kb = tid >> 1;
    const int kh = tid &  1;
    shexp_raw rg[SHEXP_UP_RB], ru[SHEXP_UP_RB];
#pragma unroll
    for (int i = 0; i < SHEXP_UP_RB; ++i) {
        shexp_q8_load(wg + (int64_t) (row0 + i)*w_row, kb, kh, rg[i]);
        shexp_q8_load(wu + (int64_t) (row0 + i)*w_row, kb, kh, ru[i]);
    }

    shexp_quantize_smem(x, sx, NT, n_embd, s_aq, s_yd);
    __syncthreads();

    float sg[NT][SHEXP_UP_RB];
    float su[NT][SHEXP_UP_RB];
#pragma unroll
    for (int i = 0; i < SHEXP_UP_RB; ++i) {
        half2 wgv[8], wuv[8];
        float dg, du;
        shexp_q8_decode(rg[i], kb, wgv, dg);
        shexp_q8_decode(ru[i], kb, wuv, du);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const half2 * a  = s_aq + (j*nb + kb)*(SHEXP_QK/2) + kh*8;
            const float   yd = s_yd[j*nb + kb];
            half2 ag = make_half2(0.0f, 0.0f);
            half2 au = make_half2(0.0f, 0.0f);
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                ag = __hfma2(wgv[q], a[q], ag);
                au = __hfma2(wuv[q], a[q], au);
            }
            sg[j][i] = yd*(dg*__half2float(__hadd(__low2half(ag), __high2half(ag))));
            su[j][i] = yd*(du*__half2float(__hadd(__low2half(au), __high2half(au))));
        }
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int i = 0; i < SHEXP_UP_RB; ++i) {
            const float vg = warp_reduce_sum(sg[j][i]);
            const float vu = warp_reduce_sum(su[j][i]);
            if (threadIdx.x == 0) {
                s_red[(threadIdx.y*NT + j)*2*SHEXP_UP_RB + i]               = vg;
                s_red[(threadIdx.y*NT + j)*2*SHEXP_UP_RB + SHEXP_UP_RB + i] = vu;
            }
        }
    }
    __syncthreads();
    if (tid < NT*SHEXP_UP_RB) {
        const int j = tid / SHEXP_UP_RB;
        const int i = tid % SHEXP_UP_RB;
        float g = 0.0f, u = 0.0f;
        for (int w = 0; w < nw; ++w) {
            g += s_red[(w*NT + j)*2*SHEXP_UP_RB + i];
            u += s_red[(w*NT + j)*2*SHEXP_UP_RB + SHEXP_UP_RB + i];
        }
        h[j*sh + row0 + i] = g/(1.0f + expf(-g))*u;
    }
    if (blockIdx.x == 0) {
        // the gate scalar: warp j of the block takes token j
        for (int j = threadIdx.y; j < NT; j += nw) {
            float acc = 0.0f;
            for (int k = threadIdx.x; k < n_embd; k += WARP_SIZE) {
                acc += ggml_cuda_cast<float>(wgi[k])*x[j*sx + k];
            }
            acc = warp_reduce_sum(acc);
            if (threadIdx.x == 0) {
                gsc[j] = 1.0f/(1.0f + expf(-acc));
            }
        }
    }
#else
    GGML_UNUSED_VARS(wg, wu, w_row, x, sx, n_embd, wgi, h, sh, gsc);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// 2. y[t][r] = (w_down[r] . h[t]) * g[t] for SHEXP_DOWN_RB rows per block (one warp).
template <int NT>
static __global__ void __launch_bounds__(WARP_SIZE)
shexp_down(const char * __restrict__ wd, const int64_t w_row, const float * __restrict__ h, const int64_t sh,
           const int n_ff, const float * __restrict__ gsc, float * __restrict__ y, const int64_t sy) {
#if defined(FP16_AVAILABLE)
    extern __shared__ char s_raw[];
    const int nb   = n_ff/SHEXP_QK;
    half2 *   s_aq = (half2 *) s_raw;
    float *   s_yd = (float *) (s_aq + NT*nb*(SHEXP_QK/2));

    const int lane = threadIdx.x;
    const int row0 = blockIdx.x*SHEXP_DOWN_RB;

    // all of the lane's weights first (2*nb <= 2*WARP_SIZE half blocks per row, checked by the matcher)
    shexp_raw raw[2][SHEXP_DOWN_RB];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int it = lane + p*WARP_SIZE;
        if (it < 2*nb) {
#pragma unroll
            for (int i = 0; i < SHEXP_DOWN_RB; ++i) {
                shexp_q8_load(wd + (int64_t) (row0 + i)*w_row, it >> 1, it & 1, raw[p][i]);
            }
        }
    }

    shexp_quantize_smem(h, sh, NT, n_ff, s_aq, s_yd);
    __syncwarp();

    float acc[NT][SHEXP_DOWN_RB];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int i = 0; i < SHEXP_DOWN_RB; ++i) {
            acc[j][i] = 0.0f;
        }
    }
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int it = lane + p*WARP_SIZE;
        if (it >= 2*nb) {
            continue;
        }
        const int kb = it >> 1;
        const int kh = it &  1;
#pragma unroll
        for (int i = 0; i < SHEXP_DOWN_RB; ++i) {
            half2 w[8];
            float d;
            shexp_q8_decode(raw[p][i], kb, w, d);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const half2 * a = s_aq + (j*nb + kb)*(SHEXP_QK/2) + kh*8;
                half2 sacc = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    sacc = __hfma2(w[q], a[q], sacc);
                }
                acc[j][i] += s_yd[j*nb + kb]*(d*__half2float(__hadd(__low2half(sacc), __high2half(sacc))));
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int i = 0; i < SHEXP_DOWN_RB; ++i) {
            const float v = warp_reduce_sum(acc[j][i]);
            if (lane == i) {
                y[j*sy + row0 + i] = v*gsc[j];
            }
        }
    }
#else
    GGML_UNUSED_VARS(wd, w_row, h, sh, n_ff, gsc, y, sy);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

bool ggml_cuda_shexp_match(const ggml_cgraph * cgraph, int i, ggml_cuda_shexp_args & a) {
    static const bool enabled = getenv("GGML_CUDA_SHEXP_FUSE") == nullptr || atoi(getenv("GGML_CUDA_SHEXP_FUSE")) != 0;
    if (!enabled || i + 6 >= cgraph->n_nodes || cgraph->nodes[i]->op != GGML_OP_MUL_MAT) {
        return false;
    }
    static const ggml_op ops[] = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT,
                                   GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL };
    const int outputs[] = { i + 6 };
    if (!ggml_can_fuse_subgraph(cgraph, i, 7, ops, outputs, 1)) {
        return false;
    }
    ggml_tensor * const * n = cgraph->nodes + i;
    a.glu  = n[2];
    a.gate = a.glu->src[0];
    a.up   = a.glu->src[1];
    a.down = n[3];
    a.ginp = n[4];
    a.sig  = n[5];
    a.mul  = n[6];
    if (!((a.gate == n[0] && a.up == n[1]) || (a.gate == n[1] && a.up == n[0])) ||
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
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_PASCAL) {
        return false;
    }
    if (a.up->src[1] != x || a.ginp->src[1] != x || x->type != GGML_TYPE_F32 || x->ne[2] != 1 || x->ne[3] != 1 ||
        nt > 2*SHEXP_FUSE_MAX_T || x->nb[0] != sizeof(float) || x->nb[1] % 16 != 0 ||
        wg->type != GGML_TYPE_Q8_0 || wu->type != GGML_TYPE_Q8_0 || wd->type != GGML_TYPE_Q8_0 ||
        !ggml_are_same_shape(wg, wu) || wg->nb[1] != wu->nb[1] || !ggml_is_contiguous(wg) || !ggml_is_contiguous(wu) ||
        !ggml_is_contiguous(wd) || wd->ne[0] != n_ff || wd->ne[1] != n_embd || wg->ne[2] != 1 || wd->ne[2] != 1 ||
        (wgi->type != GGML_TYPE_F32 && wgi->type != GGML_TYPE_BF16) || wgi->ne[0] != n_embd || ggml_nrows(wgi) != 1 ||
        !ggml_is_contiguous(wgi) ||
        n_embd % 64 != 0 || n_ff % 64 != 0 || n_ff % SHEXP_UP_RB != 0 || n_embd % SHEXP_DOWN_RB != 0 ||
        a.mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(a.mul) || a.mul->ne[0] != n_embd || a.mul->ne[1] != nt) {
        return false;
    }
    // one half block of each row per thread in the first kernel, at most two per lane in the second
    if ((2*n_embd/SHEXP_QK) % WARP_SIZE != 0 || 2*n_embd/SHEXP_QK > 1024 || 2*n_ff/SHEXP_QK > 2*WARP_SIZE) {
        return false;
    }
    // the output must not overlap the input (the first kernel reads x in every block)
    const char * x0 = (const char *) x->data;
    const char * y0 = (const char *) a.mul->data;
    if (y0 < x0 + ggml_nbytes(x) && x0 < y0 + ggml_nbytes(a.mul)) {
        return false;
    }
    return true;
}

void ggml_cuda_shexp(ggml_backend_cuda_context & ctx, const ggml_cuda_shexp_args & a) {
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * x   = a.gate->src[1];
    const ggml_tensor * wg  = a.gate->src[0];
    const ggml_tensor * wu  = a.up->src[0];
    const ggml_tensor * wd  = a.down->src[0];
    const ggml_tensor * wgi = a.ginp->src[0];
    const int     n_embd = (int) wg->ne[0];
    const int     n_ff   = (int) wg->ne[1];
    const int     nt     = (int) x->ne[1];
    const int64_t sx     = x->nb[1]/sizeof(float);
    const int64_t sy     = a.mul->nb[1]/sizeof(float);

    ggml_cuda_pool_alloc<float> h  (ctx.pool(), (size_t) nt*n_ff);
    ggml_cuda_pool_alloc<float> gsc(ctx.pool(), (size_t) nt);

    const int nw_up = (2*n_embd/SHEXP_QK)/WARP_SIZE;
    for (int t0 = 0; t0 < nt; t0 += SHEXP_FUSE_MAX_T) {
        const int ntg = std::min(SHEXP_FUSE_MAX_T, nt - t0);
        const size_t smem_up   = (size_t) ntg*(n_embd/SHEXP_QK)*(SHEXP_QK/2)*sizeof(half2) + (size_t) ntg*(n_embd/SHEXP_QK)*sizeof(float) +
                                 (size_t) nw_up*ntg*2*SHEXP_UP_RB*sizeof(float);
        const size_t smem_down = (size_t) ntg*(n_ff/SHEXP_QK)*(SHEXP_QK/2)*sizeof(half2) + (size_t) ntg*(n_ff/SHEXP_QK)*sizeof(float);
        const float * xg = (const float *) x->data + t0*sx;
        float *       hg = h.get() + (size_t) t0*n_ff;
        float *       gg = gsc.get() + t0;
        float *       yg = (float *) a.mul->data + t0*sy;
        const dim3 grid_up(n_ff/SHEXP_UP_RB, 1, 1);
        const dim3 block_up(WARP_SIZE, nw_up, 1);
        const dim3 grid_down(n_embd/SHEXP_DOWN_RB, 1, 1);
#define SHEXP_CASE(N)                                                                                                   \
        case N:                                                                                                       \
            if (wgi->type == GGML_TYPE_F32) {                                                                         \
                shexp_up<N, float><<<grid_up, block_up, smem_up, stream>>>((const char *) wg->data, (const char *) wu->data, \
                    wg->nb[1], xg, sx, n_embd, (const float *) wgi->data, hg, n_ff, gg);                              \
            } else {                                                                                                  \
                shexp_up<N, nv_bfloat16><<<grid_up, block_up, smem_up, stream>>>((const char *) wg->data, (const char *) wu->data, \
                    wg->nb[1], xg, sx, n_embd, (const nv_bfloat16 *) wgi->data, hg, n_ff, gg);                        \
            }                                                                                                         \
            shexp_down<N><<<grid_down, WARP_SIZE, smem_down, stream>>>((const char *) wd->data, wd->nb[1], hg, n_ff,  \
                n_ff, gg, yg, sy);                                                                                    \
            break;
        switch (ntg) {
            SHEXP_CASE(1)
            SHEXP_CASE(2)
            SHEXP_CASE(3)
            SHEXP_CASE(4)
            default:
                GGML_ABORT("fatal error");
        }
#undef SHEXP_CASE
        CUDA_CHECK(cudaGetLastError());
    }
}
