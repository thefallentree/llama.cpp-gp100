#include "fattn-sel.cuh"

// Attention over the selected cells only: for a head's query q and the token's slots
//   s[j] = scale*(q . K[sel_j]),   out = sum over j of softmax(s)[j]*V[sel_j]
// The heads of one kv head share the key and value rows, so a block takes NH of them for a token over a group of
// chunks of the slots and reads every row once. Scores: a lane per slot walks its key row against the queries in
// shared memory (no reductions). Values: a warp per slot and half the heads, a lane eight values. The chunks of a
// block are folded with an online softmax; with several chunk blocks per token (few tokens: a decode window) a block
// leaves the softmax state and the unnormalized sum to part, and fattn_sel_join adds the blocks up. The cost does not
// depend on the size of the cache. (A block per (head, token) read every row NH times: 137 us at 2051 slots.)
#define FATTN_SEL_NW 8

static __device__ __forceinline__ half2 fattn_sel_h2(const int v) { return *(const half2 *) &v; }

// 32 values of a row as 8 float4 (as 4 uint4 of f16 pairs are read): f16, or one q8_0 block (34 bytes, 2-byte
// aligned: the scale, then 16 pairs of quants)
static __device__ __forceinline__ void fattn_sel_row32_f16(const char * row, const int w, float4 (&v)[8]) {
    const uint4 * p = (const uint4 *) row + 4*w;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4  u  = p[i];
        const float2 k0 = __half22float2(fattn_sel_h2((int) u.x));
        const float2 k1 = __half22float2(fattn_sel_h2((int) u.y));
        const float2 k2 = __half22float2(fattn_sel_h2((int) u.z));
        const float2 k3 = __half22float2(fattn_sel_h2((int) u.w));
        v[2*i]     = make_float4(k0.x, k0.y, k1.x, k1.y);
        v[2*i + 1] = make_float4(k2.x, k2.y, k3.x, k3.y);
    }
}

static __device__ __forceinline__ void fattn_sel_row32_q8(const char * row, const int w, float4 (&v)[8]) {
    const unsigned short * p = (const unsigned short *) (row + 34*w);
    const float d = __half2float(__ushort_as_half(p[0]));
    unsigned short q[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        q[i] = p[1 + i];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int a = (int) (signed char) (q[2*i] & 0xFF);
        const int b = (int) (signed char) (q[2*i] >> 8);
        const int c = (int) (signed char) (q[2*i + 1] & 0xFF);
        const int e = (int) (signed char) (q[2*i + 1] >> 8);
        v[i] = make_float4(d*(float) a, d*(float) b, d*(float) c, d*(float) e);
    }
}

// 8 values of a row at 8*lane (one uint4 of f16 pairs): f16, or a quarter of a q8_0 block
static __device__ __forceinline__ void fattn_sel_row8_f16(const char * row, const int lane, float4 & v0, float4 & v1) {
    const uint4  u  = *(const uint4 *) (row + 16*lane);
    const float2 a0 = __half22float2(fattn_sel_h2((int) u.x));
    const float2 a1 = __half22float2(fattn_sel_h2((int) u.y));
    const float2 a2 = __half22float2(fattn_sel_h2((int) u.z));
    const float2 a3 = __half22float2(fattn_sel_h2((int) u.w));
    v0 = make_float4(a0.x, a0.y, a1.x, a1.y);
    v1 = make_float4(a2.x, a2.y, a3.x, a3.y);
}

static __device__ __forceinline__ void fattn_sel_row8_q8(const char * row, const int lane, float4 & v0, float4 & v1) {
    const unsigned short * p = (const unsigned short *) (row + 34*(lane/4));
    const float d = __half2float(__ushort_as_half(p[0]));
    const unsigned short * q = p + 1 + 4*(lane % 4);
    const unsigned short q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
    v0 = make_float4(d*(float) (int) (signed char) (q0 & 0xFF), d*(float) (int) (signed char) (q0 >> 8),
                     d*(float) (int) (signed char) (q1 & 0xFF), d*(float) (int) (signed char) (q1 >> 8));
    v1 = make_float4(d*(float) (int) (signed char) (q2 & 0xFF), d*(float) (int) (signed char) (q2 >> 8),
                     d*(float) (int) (signed char) (q3 & 0xFF), d*(float) (int) (signed char) (q3 >> 8));
}

// block (token, chunk block): heads h0 .. h0 + NH - 1 of one kv head over the chunks [cb*cpb, (cb + 1)*cpb)
template <int NH, bool Q8>
static __global__ void __launch_bounds__(FATTN_SEL_NW*WARP_SIZE, 2)
fattn_sel(const ggml_cuda_fattn_sel_args a, const int h0) {
    constexpr int D  = FATTN_SEL_D;
    constexpr int NW = FATTN_SEL_NW;
    constexpr int C  = FATTN_SEL_CHUNK;
    constexpr int HW = NH/2;                   // heads per warp in the sum over the values
    __shared__ float s_q[NH][D];               // then the output
    __shared__ float s_part[NH][NW*WARP_SIZE]; // the warps' parts of 32 scores
    __shared__ float s_s[NH][C];               // scores, then the weights
    __shared__ int   s_cell[C];
    __shared__ float s_mk[C];                  // 0 for a live slot, -inf for a dead one
    __shared__ float s_m[NH];
    __shared__ float s_l[NH];
    __shared__ float s_alpha[NH];              // the rescale of the running sums by the chunk
    float (* const s_o)[D] = s_q;
    const int t     = blockIdx.x;
    const int cb    = blockIdx.y;
    const int lane  = threadIdx.x % WARP_SIZE;
    const int warp  = threadIdx.x / WARP_SIZE;
    const int32_t * sel = a.sel + (int64_t) t*a.s_sel;
    const char *    K   = a.K + (int64_t) (h0/a.gqa)*a.skh;
    const char *    V   = a.V + (int64_t) (h0/a.gqa)*a.svh;

    for (int e = threadIdx.x; e < NH*D/4; e += NW*WARP_SIZE) {
        const int h = e/(D/4);
        const int d = 4*(e - h*(D/4));
        *(float4 *) (s_q[h] + d) = *(const float4 *) (a.q + (int64_t) t*a.sq_t + (int64_t) (h0 + h)*a.sq_h + d);
    }
    if (threadIdx.x < NH) {
        s_m[threadIdx.x] = -INFINITY;
        s_l[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    const int hg = warp % 2;
    float acc[HW][8];
#pragma unroll
    for (int h = 0; h < HW; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[h][i] = 0.0f;
        }
    }

    for (int ch = 0; ch < a.cpb; ++ch) {
        const int c_lo = (cb*a.cpb + ch)*a.c_len;
        const int n_c  = min(a.c_len, a.n_sel - c_lo); // slots of this chunk
        if (n_c <= 0) {
            break;
        }
        // scores: 32 slots at a time, a lane per slot and a warp per 32 values of the key row, the queries read as
        // broadcasts; the warps' parts are added through shared memory
        for (int cg = 0; cg < a.c_len; cg += WARP_SIZE) {
            const int c    = cg + lane;
            int       cell = c < n_c ? sel[c_lo + c] : -1;
            cell = cell >= 0 && cell < a.n_kv ? cell : -1;
            const char * kr = K + (int64_t) (cell >= 0 ? cell : 0)*a.sk;
            float4 kv[8];
            if (Q8) {
                fattn_sel_row32_q8(kr, warp, kv);
            } else {
                fattn_sel_row32_f16(kr, warp, kv);
            }
            if (warp == 0 && c < C) {
                s_cell[c] = cell >= 0 ? cell : 0; // a dead slot reads row 0 with the weight 0
                s_mk[c]   = cell >= 0 ? 0.0f : -INFINITY;
            }
            float sc[NH];
#pragma unroll
            for (int h = 0; h < NH; ++h) {
                sc[h] = 0.0f;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 k = kv[i];
#pragma unroll
                for (int h = 0; h < NH; ++h) {
                    const float4 q = *(const float4 *) (s_q[h] + 32*warp + 4*i);
                    sc[h] += k.x*q.x + k.y*q.y + k.z*q.z + k.w*q.w;
                }
            }
#pragma unroll
            for (int h = 0; h < NH; ++h) {
                s_part[h][32*warp + lane] = sc[h];
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
        // the softmax state per head, folded with the chunks before: a warp per head
        for (int h = warp; h < NH; h += NW) {
            float mx = -INFINITY;
            for (int c = lane; c < n_c; c += WARP_SIZE) {
                mx = fmaxf(mx, s_s[h][c]);
            }
#pragma unroll
            for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            }
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, mx);
            if (m_new > -INFINITY) {
                float l = 0.0f;
                for (int c = lane; c < n_c; c += WARP_SIZE) {
                    const float e = s_s[h][c] > -INFINITY ? expf(s_s[h][c] - m_new) : 0.0f;
                    s_s[h][c] = e;
                    l += e;
                }
#pragma unroll
                for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                    l += __shfl_xor_sync(0xffffffff, l, off);
                }
                const float alpha = m_old > -INFINITY ? expf(m_old - m_new) : 0.0f;
                if (lane == 0) {
                    s_alpha[h] = alpha;
                    s_l[h]     = s_l[h]*alpha + l;
                    s_m[h]     = m_new;
                }
            } else {
                // every slot so far is dead
                for (int c = lane; c < n_c; c += WARP_SIZE) {
                    s_s[h][c] = 0.0f;
                }
                if (lane == 0) {
                    s_alpha[h] = 1.0f;
                }
            }
        }
        __syncthreads();
        // the values: a warp per slot and half the heads
#pragma unroll
        for (int h = 0; h < HW; ++h) {
            const float alpha = s_alpha[hg*HW + h];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[h][i] *= alpha;
            }
        }
        for (int c0 = warp/2; c0 < n_c; c0 += 4*(NW/2)) {
            float4 v0[4], v1[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int    c  = c0 + b*(NW/2);
                const char * vr = V + (int64_t) s_cell[c < n_c ? c : 0]*a.sv;
                if (Q8) {
                    fattn_sel_row8_q8(vr, lane, v0[b], v1[b]);
                } else {
                    fattn_sel_row8_f16(vr, lane, v0[b], v1[b]);
                }
            }
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int c = c0 + b*(NW/2);
                if (c < n_c) {
#pragma unroll
                    for (int h = 0; h < HW; ++h) {
                        const float p = s_s[hg*HW + h][c];
                        acc[h][0] += p*v0[b].x; acc[h][1] += p*v0[b].y; acc[h][2] += p*v0[b].z; acc[h][3] += p*v0[b].w;
                        acc[h][4] += p*v1[b].x; acc[h][5] += p*v1[b].y; acc[h][6] += p*v1[b].z; acc[h][7] += p*v1[b].w;
                    }
                }
            }
        }
        __syncthreads(); // the next chunk overwrites the slots and the weights
    }
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
    // out, or the partial state of the chunk block
    for (int e = threadIdx.x; e < NH*D/4; e += NW*WARP_SIZE) {
        const int    h = e/(D/4);
        const int    d = 4*(e - h*(D/4));
        const float4 o = *(const float4 *) (s_o[h] + d);
        if (a.n_cb == 1) {
            const float inv = s_l[h] > 0.0f ? 1.0f/s_l[h] : 0.0f;
            *(float4 *) (a.out + (int64_t) t*a.so_t + (int64_t) (h0 + h)*a.so_h + d) = make_float4(o.x*inv, o.y*inv, o.z*inv, o.w*inv);
        } else {
            float * p = a.part + ((int64_t) (t*a.n_head + h0 + h)*a.n_cb + cb)*(D + 4);
            *(float4 *) (p + d) = o;
            if (d == 0) {
                p[D]     = s_m[h];
                p[D + 1] = s_l[h];
            }
        }
    }
}

// the chunk blocks of a (head, token): out = sum over blocks of exp(m_c - m)*o_c / sum over blocks of exp(m_c - m)*l_c
static __global__ void fattn_sel_join(const ggml_cuda_fattn_sel_args a) {
    constexpr int D = FATTN_SEL_D;
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const float * p0 = a.part + ((int64_t) (t*a.n_head + h)*a.n_cb)*(D + 4);
    float m = -INFINITY;
    for (int c = 0; c < a.n_cb; ++c) {
        m = fmaxf(m, p0[c*(D + 4) + D]);
    }
    float o = 0.0f, l = 0.0f;
    for (int c = 0; c < a.n_cb; ++c) {
        const float * p = p0 + c*(D + 4);
        const float   w = p[D] > -INFINITY ? expf(p[D] - m) : 0.0f;
        o += w*p[threadIdx.x];
        l += w*p[D + 1];
    }
    a.out[(int64_t) t*a.so_t + (int64_t) h*a.so_h + threadIdx.x] = l > 0.0f ? o/l : 0.0f;
}

void ggml_cuda_fattn_sel_plan(ggml_cuda_fattn_sel_args & a, const int nt) {
    // small selections in small chunks, so that the device is used
    a.c_len = a.n_sel <= 512 ? 32 : FATTN_SEL_CHUNK;
    const int n_chunk = (a.n_sel + a.c_len - 1)/a.c_len;
    // chunk blocks: enough blocks for the device from a few tokens, one block per token and head group from many
    a.n_cb = std::min(n_chunk, std::max(1, (512 + nt - 1)/nt));
    a.cpb  = (n_chunk + a.n_cb - 1)/a.n_cb;
    a.n_cb = (n_chunk + a.cpb - 1)/a.cpb;
}

template <bool Q8>
static void fattn_sel_launch_t(const ggml_cuda_fattn_sel_args & a, const int nt, cudaStream_t stream) {
    const dim3 grid((unsigned) nt, (unsigned) a.n_cb, 1);
    // the head groups of a block share one kv head
    for (int hk0 = 0; hk0 < a.n_head; hk0 += a.gqa) {
        for (int h0 = hk0; h0 < hk0 + a.gqa; ) {
            const int nh = hk0 + a.gqa - h0;
            if (nh >= 16) {
                fattn_sel<16, Q8><<<grid, FATTN_SEL_NW*WARP_SIZE, 0, stream>>>(a, h0);
                h0 += 16;
            } else if (nh >= 12) {
                fattn_sel<12, Q8><<<grid, FATTN_SEL_NW*WARP_SIZE, 0, stream>>>(a, h0);
                h0 += 12;
            } else if (nh >= 8) {
                fattn_sel<8, Q8><<<grid, FATTN_SEL_NW*WARP_SIZE, 0, stream>>>(a, h0);
                h0 += 8;
            } else {
                fattn_sel<4, Q8><<<grid, FATTN_SEL_NW*WARP_SIZE, 0, stream>>>(a, h0);
                h0 += 4;
            }
        }
    }
}

void ggml_cuda_fattn_sel_launch(const ggml_cuda_fattn_sel_args & a, const int nt, cudaStream_t stream) {
    GGML_ASSERT(a.n_cb == 1 || a.part != nullptr);
    if (a.q8) {
        fattn_sel_launch_t<true>(a, nt, stream);
    } else {
        fattn_sel_launch_t<false>(a, nt, stream);
    }
    if (a.n_cb > 1) {
        fattn_sel_join<<<dim3((unsigned) a.n_head, (unsigned) nt, 1), FATTN_SEL_D, 0, stream>>>(a);
    }
}

bool ggml_cuda_flash_attn_sel_supported(const ggml_tensor * op) {
    const ggml_tensor * q   = op->src[0];
    const ggml_tensor * k   = op->src[1];
    const ggml_tensor * v   = op->src[2];
    const ggml_tensor * sel = op->src[3];
    const int64_t n_head    = q->ne[1];
    const int64_t n_head_kv = k->ne[2];
    const int64_t gqa       = n_head/n_head_kv;
    const bool    f16       = k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16;
    const bool    q8        = k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0;
    // f16 rows are read 16 bytes at a time, q8_0 rows 2 bytes at a time
    const size_t  al        = f16 ? 16 : 2;
    return q->type == GGML_TYPE_F32 && (f16 || q8) && sel->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
           q->ne[0] == FATTN_SEL_D && q->nb[0] == sizeof(float) && q->nb[1] % 16 == 0 && q->nb[2] % 16 == 0 &&
           k->nb[0] == ggml_type_size(k->type) && k->nb[1] % al == 0 && k->nb[2] % al == 0 &&
           v->nb[0] == ggml_type_size(v->type) && v->nb[1] % al == 0 && v->nb[2] % al == 0 &&
           sel->nb[0] == sizeof(int32_t) && sel->ne[0] <= FATTN_SEL_MAX_SEL &&
           n_head % 4 == 0 && gqa % 4 == 0 && k->ne[1] < INT32_MAX;
}

void ggml_cuda_flash_attn_sel(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * sel = dst->src[3];
    GGML_ASSERT(ggml_cuda_flash_attn_sel_supported(dst));
    const bool   q8 = k->type == GGML_TYPE_Q8_0;
    const size_t al = q8 ? 2 : 16;
    GGML_ASSERT(((uintptr_t) q->data & 0xF) == 0 && ((uintptr_t) k->data % al) == 0 && ((uintptr_t) v->data % al) == 0 &&
                ((uintptr_t) dst->data & 0xF) == 0);

    const int nt = (int) q->ne[2];

    ggml_cuda_fattn_sel_args a;
    a.sel    = (const int32_t *) sel->data;
    a.q      = (const float *) q->data;
    a.K      = (const char *) k->data;
    a.V      = (const char *) v->data;
    a.out    = (float *) dst->data;
    a.part   = nullptr;
    a.q8     = q8;
    a.s_sel  = (int) (sel->nb[1]/sizeof(int32_t));
    a.sq_t   = (int) (q->nb[2]/sizeof(float));
    a.sq_h   = (int) (q->nb[1]/sizeof(float));
    a.sk     = (int64_t) k->nb[1];
    a.skh    = (int64_t) k->nb[2];
    a.sv     = (int64_t) v->nb[1];
    a.svh    = (int64_t) v->nb[2];
    a.so_t   = (int) (dst->nb[2]/sizeof(float));
    a.so_h   = (int) (dst->nb[1]/sizeof(float));
    a.n_sel  = (int) sel->ne[0];
    a.n_kv   = (int) k->ne[1];
    a.n_head = (int) q->ne[1];
    a.gqa    = (int) (q->ne[1]/k->ne[2]);
    a.scale  = ggml_get_op_params_f32(dst, 0);
    ggml_cuda_fattn_sel_plan(a, nt);

    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (a.n_cb > 1) {
        a.part = part.alloc((size_t) nt*a.n_head*a.n_cb*(FATTN_SEL_D + 4));
    }
    ggml_cuda_fattn_sel_launch(a, nt, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
}
