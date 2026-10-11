#include "hc-mix.cuh"

#define HC_MIX_HC        4  // streams
#define HC_MIX_QK        32 // Q8_0 block
#define HC_MIX_NORM_TPB  256
#define HC_MIX_DOWN_RB   2  // rows of w_down per block: 160 blocks for hc_lr = 320, enough to cover the load latency
#define HC_MIX_DOWN_NW   4
#ifndef HC_MIX_UP_COLS
#define HC_MIX_UP_COLS   8  // columns d per block of the up kernel: 4 x 8 rows of w_up, staged in shared memory
#endif

// ---------------------------------------------------------------------------------------------------------------
// 1. norm: one block per (stream, token). rs = rsqrt(mean(R^2) + eps), xn = R * w_norm * rs (written for the
//    inject mat-vec), and xn in the half2 activation format of the sm_60 Q8_0 mat-vec (mmvq-f16-sm60.cu): per 32
//    values the scale amax and the values / amax as half2 in the decode order (e, e+2), (16+e, 18+e), (e+1, e+3) ...
template <int dummy = 0>
static __global__ void __launch_bounds__(HC_MIX_NORM_TPB)
hc_mix_norm(const float * __restrict__ R, const int64_t sr_c, const int64_t sr_t,
            const float * __restrict__ wn, const int n_embd, const float eps,
            float * __restrict__ xn, const int64_t sxn_c, const int64_t sxn_t,
            half2 * __restrict__ aq, half2 * __restrict__ ads) {
    extern __shared__ float s_x[]; // n_embd
    __shared__ float s_red[HC_MIX_NORM_TPB/WARP_SIZE];

    const int c = blockIdx.x;
    const int t = blockIdx.y;
    const float * Rr  = R  + t*sr_t + c*sr_c;
    const float * wnc = wn + (int64_t) c*n_embd;

    float ss = 0.0f;
    for (int i = 4*threadIdx.x; i < n_embd; i += 4*HC_MIX_NORM_TPB) {
        const float4 r = *(const float4 *) (Rr + i);
        ss += r.x*r.x + r.y*r.y + r.z*r.z + r.w*r.w;
        *(float4 *) (s_x + i) = r;
    }
    ss = warp_reduce_sum(ss);
    if (threadIdx.x % WARP_SIZE == 0) {
        s_red[threadIdx.x / WARP_SIZE] = ss;
    }
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int w = 0; w < HC_MIX_NORM_TPB/WARP_SIZE; ++w) {
        ss += s_red[w];
    }
    const float rs = rsqrtf(ss/n_embd + eps);

    float * xr = xn + t*sxn_t + c*sxn_c;
    for (int i = 4*threadIdx.x; i < n_embd; i += 4*HC_MIX_NORM_TPB) {
        const float4 r = *(const float4 *) (s_x + i);
        const float4 g = *(const float4 *) (wnc + i);
        const float4 v = make_float4(r.x*g.x*rs, r.y*g.y*rs, r.z*g.z*rs, r.w*g.w*rs);
        *(float4 *) (xr + i)  = v;
        *(float4 *) (s_x + i) = v;
    }
    __syncthreads();

    // the A16 format, 8 lanes per block of 32 (same arithmetic as quantize_a16)
    const int nb     = n_embd/HC_MIX_QK;
    const int nb_tok = HC_MIX_HC*nb;
    for (int task = threadIdx.x; task < 8*nb; task += HC_MIX_NORM_TPB) {
        const int ib = task / 8;
        const int l  = task % 8;
        const unsigned int gmask = 0xffu << ((threadIdx.x % WARP_SIZE) & ~7);
        float v[4];
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] = s_x[ib*HC_MIX_QK + 4*l + i];
            amax = fmaxf(amax, fabsf(v[i]));
        }
#pragma unroll
        for (int s = 4; s > 0; s >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(gmask, amax, s, 8));
        }
        const float id = amax > 0.0f ? 1.0f/amax : 0.0f;
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] *= id;
            sum  += v[i];
        }
#pragma unroll
        for (int s = 4; s > 0; s >>= 1) {
            sum += __shfl_xor_sync(gmask, sum, s, 8);
        }
        const int kb = c*nb + ib;
        if (l == 0) {
            ads[t*nb_tok + kb] = make_half2(__float2half(amax), __float2half(sum));
        }
        const int off = l >= 4 ? 1 : 0;
        const int k   = (4*l - off*16)/4;
        half2 * qs = aq + ((int64_t) t*nb_tok + kb)*(HC_MIX_QK/2);
        qs[4*k + 0 + off] = make_half2(__float2half(v[0]), __float2half(v[2]));
        qs[4*k + 2 + off] = make_half2(__float2half(v[1]), __float2half(v[3]));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Q8_0 half-block -> 8 half2 in the decode order of the A16 activations (see mul_mat_vec_q8_0_a16): the 16 quants of
// half kh of block kb start (34 kb + 2 + 16 kh) bytes into the row, 2 bytes off a word for even kb; one PRMT per
// word realigns them, one PRMT per half2 puts two bytes under a 0x64 exponent and one HSUB2 makes them exact.
static __device__ __forceinline__ void hc_mix_q8_half2(const char * __restrict__ row, const int kb, const int kh,
                                                       half2 w[8], float & d) {
    const int boff  = kb*34;
    const int shift = (kb & 1) ? 0 : 2;
    const int sel   = shift ? 0x5432 : 0x3210;
    d = __half2float(*(const half *) (row + boff));
    const int * pA = (const int *) (row + boff + 2 + 8*kh - shift);
    const int * pB = pA + 4;
    const int uA0 = pA[0], uA1 = pA[1];
    const int uB0 = pB[0], uB1 = pB[1];
    const int uA2 = shift ? pA[2] : 0;
    const int uB2 = shift ? pB[2] : 0;
    const int A0 = __byte_perm(uA0, uA1, sel) ^ 0x80808080;
    const int A1 = __byte_perm(uA1, uA2, sel) ^ 0x80808080;
    const int B0 = __byte_perm(uB0, uB1, sel) ^ 0x80808080;
    const int B1 = __byte_perm(uB1, uB2, sel) ^ 0x80808080;
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

// 2. down: lo[t][k] = silu(s * (w_down[k] . xn[t]) + b), HC_MIX_DOWN_RB rows per block (the A16 mat-vec with an
//    epilogue).
template <int NT>
static __global__ void __launch_bounds__(WARP_SIZE*HC_MIX_DOWN_NW)
hc_mix_down(const char * __restrict__ wd, const int64_t wd_row, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
            const int nblocks, const float s, const float b, float * __restrict__ lo, const int64_t slo_t) {
#if defined(FP16_AVAILABLE)
    const int row0 = blockIdx.x*HC_MIX_DOWN_RB;
    const int tid  = threadIdx.x + threadIdx.y*WARP_SIZE;

    float sumf[NT][HC_MIX_DOWN_RB];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int i = 0; i < HC_MIX_DOWN_RB; ++i) {
            sumf[j][i] = 0.0f;
        }
    }

    for (int it = tid; it < 2*nblocks; it += WARP_SIZE*HC_MIX_DOWN_NW) {
        const int kb = it >> 1;
        const int kh = it &  1;
        half2 a[NT][8];
        float yd[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int4 * ap = (const int4 *) (aq + ((int64_t) j*nblocks + kb)*(HC_MIX_QK/2) + kh*8);
            const int4 a0 = ap[0];
            const int4 a1 = ap[1];
            a[j][0] = *(const half2 *) &a0.x; a[j][1] = *(const half2 *) &a0.y;
            a[j][2] = *(const half2 *) &a0.z; a[j][3] = *(const half2 *) &a0.w;
            a[j][4] = *(const half2 *) &a1.x; a[j][5] = *(const half2 *) &a1.y;
            a[j][6] = *(const half2 *) &a1.z; a[j][7] = *(const half2 *) &a1.w;
            yd[j] = __low2float(ads[(int64_t) j*nblocks + kb]);
        }
#pragma unroll
        for (int i = 0; i < HC_MIX_DOWN_RB; ++i) {
            half2 w[8];
            float d;
            hc_mix_q8_half2(wd + (int64_t) (row0 + i)*wd_row, kb, kh, w, d);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                half2 acc = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    acc = __hfma2(w[q], a[j][q], acc);
                }
                sumf[j][i] += yd[j]*(d*__half2float(__hadd(__low2half(acc), __high2half(acc))));
            }
        }
    }

    __shared__ float tmp[HC_MIX_DOWN_NW][NT][HC_MIX_DOWN_RB];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int i = 0; i < HC_MIX_DOWN_RB; ++i) {
            const float v = warp_reduce_sum(sumf[j][i]);
            if (threadIdx.x == 0) {
                tmp[threadIdx.y][j][i] = v;
            }
        }
    }
    __syncthreads();
    if (tid < NT*HC_MIX_DOWN_RB) {
        const int j = tid / HC_MIX_DOWN_RB;
        const int i = tid % HC_MIX_DOWN_RB;
        float y = 0.0f;
#pragma unroll
        for (int w = 0; w < HC_MIX_DOWN_NW; ++w) {
            y += tmp[w][j][i];
        }
        const float z = s*y + b;
        lo[j*slo_t + row0 + i] = z/(1.0f + expf(-z));
    }
#else
    GGML_UNUSED_VARS(wd, wd_row, aq, ads, nblocks, s, b, lo, slo_t);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// 3. up: mixed[t][d] = scale * sum_c xn[t][c][d] * sigmoid(w_up[c*n_embd + d] . lo[t]).
//    A block owns HC_MIX_UP_COLS columns, i.e. HC_MIX_HC contiguous runs of HC_MIX_UP_COLS rows of w_up. They are
//    staged in shared memory with coalesced 16-byte loads (a row is only 10 blocks long, so reading rows directly
//    from global memory scatters every load over as many rows as there are lanes); lo is converted to the half2
//    format in shared memory. Warp c computes stream c; two lanes split each row.
template <int NT>
static __global__ void __launch_bounds__(WARP_SIZE*HC_MIX_HC)
hc_mix_up(const char * __restrict__ wu, const int64_t wu_row, const float * __restrict__ lo, const int64_t slo_t,
          const int hc_lr, const int n_embd, const float scale,
          const float * __restrict__ xn, const int64_t sxn_c, const int64_t sxn_t,
          float * __restrict__ mixed, const int64_t smx_t) {
#if defined(FP16_AVAILABLE)
    extern __shared__ int4 s_mem[];
    const int     nb      = hc_lr/HC_MIX_QK;                      // blocks per row
    const int     seg16   = HC_MIX_UP_COLS*(int) wu_row/16;       // int4 per stream run
    int4 *        s_w     = s_mem;                                // [HC][seg16]
    half2 *       s_aq    = (half2 *) (s_w + HC_MIX_HC*seg16);    // [NT][nb][16]
    float *       s_yd    = (float *) (s_aq + NT*nb*(HC_MIX_QK/2)); // [NT][nb]
    float *       s_m     = s_yd + NT*nb;                         // [HC][NT][HC_MIX_UP_COLS]

    const int tid = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int d0  = blockIdx.x*HC_MIX_UP_COLS;

    // stage the weights (run c starts at row c*n_embd + d0): all loads of a thread first, then the stores, so that
    // they are in flight together
    {
        constexpr int MAXPER = 8;
        int4 tmp[MAXPER];
#pragma unroll
        for (int k = 0; k < MAXPER; ++k) {
            const int i = tid + k*WARP_SIZE*HC_MIX_HC;
            if (i < HC_MIX_HC*seg16) {
                tmp[k] = ((const int4 *) (wu + ((int64_t) (i / seg16)*n_embd + d0)*wu_row))[i % seg16];
            }
        }
#pragma unroll
        for (int k = 0; k < MAXPER; ++k) {
            const int i = tid + k*WARP_SIZE*HC_MIX_HC;
            if (i < HC_MIX_HC*seg16) {
                s_w[i] = tmp[k];
            }
        }
    }
    // lo -> half2 activations, 8 lanes per block of 32
    for (int task = tid; task < NT*nb*8; task += WARP_SIZE*HC_MIX_HC) {
        const int j  = task / (nb*8);
        const int ib = (task / 8) % nb;
        const int l  = task % 8;
        const unsigned int gmask = 0xffu << ((threadIdx.x % WARP_SIZE) & ~7);
        float v[4];
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] = lo[j*slo_t + ib*HC_MIX_QK + 4*l + i];
            amax = fmaxf(amax, fabsf(v[i]));
        }
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
        half2 * qs = s_aq + (j*nb + ib)*(HC_MIX_QK/2);
        qs[4*k + 0 + off] = make_half2(__float2half(v[0]*id), __float2half(v[2]*id));
        qs[4*k + 2 + off] = make_half2(__float2half(v[1]*id), __float2half(v[3]*id));
    }
    __syncthreads();

    constexpr int LPR = WARP_SIZE/HC_MIX_UP_COLS; // lanes per row
    const int c    = threadIdx.y;
    const int lane = threadIdx.x;
    const int r    = lane / LPR; // column within the block
    const int h    = lane % LPR; // which part of the row's half blocks
    const int nh   = 2*nb/LPR;   // half blocks per lane
    const char * row = (const char *) (s_w + c*seg16) + r*wu_row;

    float acc[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        acc[j] = 0.0f;
    }
    for (int it = h*nh; it < (h + 1)*nh; ++it) { // the lane's part of the 2*nb half blocks
        const int kb = it >> 1;
        const int kh = it &  1;
        half2 w[8];
        float d;
        hc_mix_q8_half2(row, kb, kh, w, d);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const half2 * a = s_aq + (j*nb + kb)*(HC_MIX_QK/2) + kh*8;
            half2 hacc = make_half2(0.0f, 0.0f);
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                hacc = __hfma2(w[q], a[q], hacc);
            }
            acc[j] += s_yd[j*nb + kb]*(d*__half2float(__hadd(__low2half(hacc), __high2half(hacc))));
        }
    }

    const int d = d0 + r;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        float g = acc[j];
#pragma unroll
        for (int o = LPR/2; o > 0; o >>= 1) {
            g += __shfl_xor_sync(0xffffffff, g, o);
        }
        if (h == 0) {
            s_m[(c*NT + j)*HC_MIX_UP_COLS + r] = xn[j*sxn_t + c*sxn_c + d]/(1.0f + expf(-g));
        }
    }
    __syncthreads();
    if (tid < NT*HC_MIX_UP_COLS) {
        const int j  = tid / HC_MIX_UP_COLS;
        const int rr = tid % HC_MIX_UP_COLS;
        float m = 0.0f;
#pragma unroll
        for (int cc = 0; cc < HC_MIX_HC; ++cc) {
            m += s_m[(cc*NT + j)*HC_MIX_UP_COLS + rr];
        }
        mixed[j*smx_t + d0 + rr] = scale*m;
    }
#else
    GGML_UNUSED_VARS(wu, wu_row, lo, slo_t, hc_lr, n_embd, scale, xn, sxn_c, sxn_t, mixed, smx_t);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

static size_t hc_mix_up_smem(const int nt, const int hc_lr, const int64_t wu_row) {
    const int nb = hc_lr/HC_MIX_QK;
    return (size_t) HC_MIX_HC*HC_MIX_UP_COLS*wu_row + (size_t) nt*nb*(HC_MIX_QK/2)*sizeof(half2) +
           (size_t) nt*nb*sizeof(float) + (size_t) HC_MIX_HC*nt*HC_MIX_UP_COLS*sizeof(float);
}

// The three launches for up to HC_MIX_MAX_T tokens. aq/ads: scratch for nt*hc*n_embd/32 half2 blocks.
static void hc_mix_launch(const int nt, const float * R, const int64_t sr_c, const int64_t sr_t, const float * wn,
                          const char * wd, const int64_t wd_row, const char * wu, const int64_t wu_row,
                          const int n_embd, const int hc_lr, const float eps, const float s, const float b, const float scale,
                          float * xn, const int64_t sxn_c, const int64_t sxn_t, float * lo, const int64_t slo_t,
                          float * mixed, const int64_t smx_t, half2 * aq, half2 * ads, cudaStream_t stream) {
    const int nblocks = HC_MIX_HC*n_embd/HC_MIX_QK;
    hc_mix_norm<<<dim3(HC_MIX_HC, nt, 1), HC_MIX_NORM_TPB, n_embd*sizeof(float), stream>>>(
        R, sr_c, sr_t, wn, n_embd, eps, xn, sxn_c, sxn_t, aq, ads);
    const dim3   grid_down(hc_lr/HC_MIX_DOWN_RB, 1, 1);
    const dim3   block_down(WARP_SIZE, HC_MIX_DOWN_NW, 1);
    const dim3   grid_up(n_embd/HC_MIX_UP_COLS, 1, 1);
    const dim3   block_up(WARP_SIZE, HC_MIX_HC, 1);
    const size_t smem_up = hc_mix_up_smem(nt, hc_lr, wu_row);
    switch (nt) {
#define HC_MIX_CASE(N)                                                                                               \
        case N:                                                                                                    \
            hc_mix_down<N><<<grid_down, block_down, 0, stream>>>(wd, wd_row, aq, ads, nblocks, s, b, lo, slo_t);    \
            hc_mix_up<N><<<grid_up, block_up, smem_up, stream>>>(wu, wu_row, lo, slo_t, hc_lr, n_embd, scale,       \
                xn, sxn_c, sxn_t, mixed, smx_t);                                                                   \
            break;
        HC_MIX_CASE(1)
        HC_MIX_CASE(2)
        HC_MIX_CASE(3)
        HC_MIX_CASE(4)
#undef HC_MIX_CASE
        default:
            GGML_ABORT("fatal error");
    }
}

#ifndef HC_MIX_STANDALONE

bool ggml_cuda_hc_mix_match(const ggml_cgraph * cgraph, int i, ggml_cuda_hc_mix_args & a) {
    static const bool enabled = getenv("GGML_CUDA_HC_MIX") == nullptr || atoi(getenv("GGML_CUDA_HC_MIX")) != 0;
    if (!enabled || i + 9 >= cgraph->n_nodes || cgraph->nodes[i]->op != GGML_OP_RMS_NORM) {
        return false;
    }
    static const ggml_op ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_MUL_MAT,
                                   GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE };
    const int outputs[] = { i + 2, i + 9 };
    if (!ggml_can_fuse_subgraph(cgraph, i, 10, ops, outputs, 2)) {
        return false;
    }
    ggml_tensor * const * n = cgraph->nodes + i;
    a.rms     = n[0];
    a.mul     = n[1];
    a.mm_down = n[4];
    a.scale   = n[5];
    a.silu    = n[6];
    a.mm_up   = n[7];
    a.pre     = n[9];

    const ggml_tensor * R  = a.rms->src[0];
    const ggml_tensor * wn = a.mul->src[1];
    const ggml_tensor * wd = a.mm_down->src[0];
    const ggml_tensor * wu = a.mm_up->src[0];
    const int64_t n_embd = R->ne[0];
    const int64_t hc     = R->ne[1];
    const int64_t nt     = R->ne[2];
    const int64_t hc_lr  = wd->ne[1];
    if (ggml_get_unary_op(a.silu) != GGML_UNARY_OP_SILU || ggml_get_op_params_i32(a.pre, 1) == 0 ||
        a.mul->src[0] != a.rms || a.mm_down->src[1] != n[2] || n[2]->view_src != a.mul || a.scale->src[0] != a.mm_down ||
        a.silu->src[0] != a.scale || a.mm_up->src[1] != a.silu || a.pre->src[0] != n[3] || n[3]->view_src != a.mul ||
        a.pre->src[1] != n[8] || n[8]->view_src != a.mm_up) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_PASCAL) {
        return false;
    }
    // xn and mixed are written and read back across blocks, so neither may share memory with the other or with R.
    // lo goes to scratch: the allocator often places mixed in the memory of the elided intermediates.
    auto overlap = [](const ggml_tensor * x, const ggml_tensor * y) {
        const char * x0 = (const char *) x->data;
        const char * y0 = (const char *) y->data;
        return x0 < y0 + ggml_nbytes(y) && y0 < x0 + ggml_nbytes(x);
    };
    if (overlap(a.mul, R) || overlap(a.pre, R) || overlap(a.mul, a.pre)) {
        return false;
    }
    if (hc != HC_MIX_HC || nt > 2*HC_MIX_MAX_T || R->ne[3] != 1 || n_embd % HC_MIX_UP_COLS != 0 || n_embd % 32 != 0 ||
        hc_lr % 64 != 0 || hc_lr % HC_MIX_DOWN_RB != 0 || (2*hc_lr/HC_MIX_QK) % (WARP_SIZE/HC_MIX_UP_COLS) != 0 ||
        R->type != GGML_TYPE_F32 || !ggml_is_contiguous(R) || wn->type != GGML_TYPE_F32 || !ggml_is_contiguous(wn) ||
        wn->ne[0] != n_embd || wn->ne[1] != hc || ggml_nrows(wn) != hc ||
        wd->type != GGML_TYPE_Q8_0 || wu->type != GGML_TYPE_Q8_0 || !ggml_is_contiguous(wd) || !ggml_is_contiguous(wu) ||
        wd->ne[0] != hc*n_embd || wu->ne[0] != hc_lr || wu->ne[1] != hc*n_embd ||
        !ggml_is_contiguous(a.mul) || !ggml_is_contiguous(a.pre) || a.pre->type != GGML_TYPE_F32 ||
        hc_mix_up_smem(HC_MIX_MAX_T, (int) hc_lr, wu->nb[1]) > 48*1024 ||
        HC_MIX_HC*HC_MIX_UP_COLS*(int64_t) wu->nb[1]/16 > 8*WARP_SIZE*HC_MIX_HC) {
        return false;
    }
    return true;
}

void ggml_cuda_hc_mix(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & a) {
    const ggml_tensor * R  = a.rms->src[0];
    const ggml_tensor * wn = a.mul->src[1];
    const ggml_tensor * wd = a.mm_down->src[0];
    const ggml_tensor * wu = a.mm_up->src[0];
    const int     n_embd = (int) R->ne[0];
    const int     nt     = (int) R->ne[2];
    const int     hc_lr  = (int) wd->ne[1];
    const int64_t sr_c   = R->nb[1]/sizeof(float);
    const int64_t sr_t   = R->nb[2]/sizeof(float);
    const int64_t sxn_c  = a.mul->nb[1]/sizeof(float);
    const int64_t sxn_t  = a.mul->nb[2]/sizeof(float);
    const int64_t smx_t  = a.pre->nb[1]/sizeof(float);

    const int nblocks = HC_MIX_HC*n_embd/HC_MIX_QK;
    ggml_cuda_pool_alloc<half2> aq (ctx.pool(), (size_t) HC_MIX_MAX_T*nblocks*(HC_MIX_QK/2));
    ggml_cuda_pool_alloc<half2> ads(ctx.pool(), (size_t) HC_MIX_MAX_T*nblocks);
    ggml_cuda_pool_alloc<float> lo (ctx.pool(), (size_t) nt*hc_lr);

    for (int t0 = 0; t0 < nt; t0 += HC_MIX_MAX_T) {
        const int ntg = std::min(HC_MIX_MAX_T, nt - t0);
        hc_mix_launch(ntg, (const float *) R->data + t0*sr_t, sr_c, sr_t, (const float *) wn->data,
            (const char *) wd->data, wd->nb[1], (const char *) wu->data, wu->nb[1], n_embd, hc_lr,
            ggml_get_op_params_f32(a.rms, 0), ggml_get_op_params_f32(a.scale, 0), ggml_get_op_params_f32(a.scale, 1),
            ggml_get_op_params_f32(a.pre, 0),
            (float *) a.mul->data + t0*sxn_t, sxn_c, sxn_t, lo.get() + t0*hc_lr, hc_lr,
            (float *) a.pre->data + t0*smx_t, smx_t, aq.get(), ads.get(), ctx.stream());
        CUDA_CHECK(cudaGetLastError());
    }
}

#endif // HC_MIX_STANDALONE
