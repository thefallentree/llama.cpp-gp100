// Q4_1 mat-vec specialised for sm_60 (GP100), built on HFMA2.
//
// GP100 has no DP4A, so the generic mmvq path emulates the 4x int8 dot product with a chain of
// PTX vmad.  Measured instruction throughput on this card:
//     VMAD  = 32   lane-op/clk/SM (half rate)
//     HFMA2 = 61.5 lane-op/clk/SM (full rate, two MACs per instruction)
//     LOP3  = full rate;  I2F / F2F = 1/4 rate
// Per issue slot that makes HFMA2 worth ~4x VMAD in MACs.
//
// So the 4-bit nibbles are expanded straight to half2 with LOP3 magic constants -- no
// integer-to-float conversion instruction anywhere -- and accumulated with HFMA2.  The
// activations are quantized to half rather than int8, so the kernel contains no conversions
// either.  Q4_1's 20-byte block keeps `qs` at offset 4, which is always 4-byte aligned, so the
// nibbles can be read as `int`; Q4_0's 18-byte block would be 4-byte aligned on only half its
// blocks, which is why this path is Q4_1-only.
//
// Measured on a Tesla P100 (llama-bench, Q4_1 weights): see the PR description.

#include "mmvq-f16-sm60.cuh"
#include "common.cuh"

#define A16_QK     32 // elements per block, matching block_q4_1
#define A16_NWARPS  4
#define A16_ROWS    4 // output rows per CUDA block
#define A16_QG      8 // lanes cooperating on one block in the quantize kernel
#define A16_QBLOCK 128 // ... and its block size

// Quantize the f32 activations.  A16_QG lanes cooperate on one block of A16_QK elements.
//
// Originally one thread owned a whole block.  This kernel launches as often as the GEMV itself
// (18,094 times in one trace window) yet cost 4.78 us per call, and TRIPLING its work
// (gridX 2 -> 6) did not change that time -- so it was limited neither by SM count nor by
// bandwidth, but by the dependency chain inside a thread: a 32-step sequential max followed by a
// 32-step sequential sum.  Spreading a block over G lanes cuts the chain by G and lands at
// 2.0-2.6 us/call, essentially the 1.9 us launch floor measured for small kernels on this card.
//
// G = 8 is optimal: elements 4k..4k+3 and 4k+16..4k+19 each fall inside a single lane, so the
// output permutation below stays lane-local and needs no shuffle.  Splitting across the whole warp
// (G = 32) needs a shuffle to collect the output and is slower.  Measured us/call
// (4096x5 / 12288x5): G=1 4.88/6.29, G=2 2.63/3.62, G=4 2.29/2.99, G=8 2.25/2.64, G=32 2.62/3.90.
//
// Keeping one thread per block and turning the sum into a tree instead is 20% slower (5.84 us at
// 4096x5): 32 partial sums have to live in registers, and registers bind before the chain does.
// Break the chain by splitting threads, not by restructuring within a thread.
template <bool vec4>
static __global__ void quantize_a16(
        const float * __restrict__ x, half2 * __restrict__ aq, half2 * __restrict__ ads,
        const int s01, const int nblocks) {

    constexpr int G = A16_QG;       // lanes per block
    constexpr int E = A16_QK/G;     // elements per lane

    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int ib  = tid/G;          // block index within the row
    const int l   = tid % G;        // lane within the block
    const int row = blockIdx.y;
    if (ib >= nblocks) {
        return;
    }

    // ib is shared by all G lanes, so the tail drops out G lanes at a time.  G divides 32 and the
    // block size is a multiple of 32, so a segment never straddles a warp and shuffle partners are
    // always live.
    const unsigned int gmask = ((1u << G) - 1) << ((threadIdx.x & (WARP_SIZE - 1)) & ~(G - 1));

    const float * xp = x + (size_t) row*s01 + ib*A16_QK + l*E;

    float v[E];
    float amax = 0.0f;
    if (vec4) {
#pragma unroll
        for (int i = 0; i < E/4; ++i) {
            const float4 t = ((const float4 *) xp)[i];
            v[4*i + 0] = t.x; v[4*i + 1] = t.y; v[4*i + 2] = t.z; v[4*i + 3] = t.w;
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
        }
    } else {
#pragma unroll
        for (int i = 0; i < E; ++i) {
            v[i] = xp[i];
            amax = fmaxf(amax, fabsf(v[i]));
        }
    }
#pragma unroll
    for (int s = G/2; s > 0; s >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(gmask, amax, s, G));
    }

    const float id = amax > 0.0f ? 1.0f/amax : 0.0f;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        v[i] *= id;
        sum  += v[i];
    }
#pragma unroll
    for (int s = G/2; s > 0; s >>= 1) {
        sum += __shfl_xor_sync(gmask, sum, s, G);
    }

    if (l == 0) {
        ads[row*nblocks + ib] = make_half2(__float2half(amax), __float2half(sum));
    }

    // Lanes holding elements 0..15 write outputs 4k+0 / 4k+2; those holding 16..31 write 4k+1 / 4k+3.
    const int off   = l >= G/2 ? 1 : 0;
    const int kbase = (E*l - off*16)/4;

    half2 * qs = aq + (size_t) row*nblocks*(A16_QK/2) + ib*(A16_QK/2);
#pragma unroll
    for (int e = 0; e < E/4; ++e) {
        const int k = kbase + e;
        qs[4*k + 0 + off] = make_half2(__float2half(v[4*e + 0]), __float2half(v[4*e + 2]));
        qs[4*k + 2 + off] = make_half2(__float2half(v[4*e + 1]), __float2half(v[4*e + 3]));
    }
}

// Inner loop structure and epilogue.
//
// Loop order is "activations into registers first, then one row at a time" -- the row loop sits
// outside the column loop.  The natural ordering instead keeps w[A16_ROWS][8] live across the
// column loop, so register pressure grows with the row count (153 registers = 3 blocks/SM at
// width 5 with 8 rows).  This ordering only keeps a[ncols][8], which does not depend on the row
// count, plus w[8] for a single row.  The natural ordering measured -6.4% at width 3 and -4.2% at
// width 4 against this one (widths 2 and 5 unchanged).
//
// The epilogue (applying the per-block scales) runs once per (row, column, block) and is therefore
// heavier than the dot product itself.  In GP100 issue slots (HFMA2 = 1), F2F counts 4 (quarter
// rate) and FP32 counts 2 (half rate), which puts the original epilogue at 18 against 8 for the
// dot product body.  A measurement-only build that collapsed the epilogue to a single conversion
// ran 18.3% faster at width 3 and 19.4% faster at width 5 -- that is what the epilogue costs.
// What is actually obtainable is summing the low and high halves in fp16 and converting once
// (two F2F -> one).
//
// The accumulator stays float: summing across blocks in fp16 loses significance.  That adds one
// rounding step, but acc has already been through 8 fp16 FMAs inside the block, so roundings per
// block go 8 -> 9.  Perplexity at ubatch 4 moved 7.2370 -> 7.2359, -0.015%.  That is 200x below
// the +/-2.9% spread of the throughput measurement, so it bounds the change rather than confirming
// it; the argument that it is harmless is the exactness of the dequantization above.  A version that
// also does the scale product in fp16 (yd / ys / dm are all half to begin with) measured the same
// and only adds overflow risk, so it is not used.
//
// Raising A16_ROWS to 8 or 16 was rejected.  Every block in the grid reads the whole activation
// vector, so doubling the rows halves that traffic -- but width 5 moves by only -3.2%, and the
// row-count-independent register form above does not rescue it either (ROWS = 16 spills, +48%).
// This kernel is not limited by activation L2 traffic.
//
// It is limited by instruction issue, not by bandwidth.  Comparing static SASS instruction counts
// against measured time, width 5 consumes 96-105% of the issue slots:
//
//     tensor                width  measured us/call  issue floor us  utilization
//     ffn_gate/up  gridX 3072  5        139.05           133.6           96%
//     ffn_down     gridX 1024  5         91.64            96.0          105%
//     lm_head     gridX 62080  5       2618.85          2699.0          103%
//
//   (issue floor = warps x instructions per warp / (56 SM x 1.92 warp-inst/clk x 1.328 GHz))
// In bandwidth terms width 5 reaches 226 GB/s, 37% of the 606 GB/s measured ceiling.  Instruction
// issue is therefore the dominant term at this width -- though note the model is only good to
// about +/-10%, as the 105% row shows, and later attempts to cash it in all measured neutral or
// negative.  Treat the issue count as a bound, not a lever.
//
// Hence nwarps as a template parameter.  The K loop body is a fixed total amount of work regardless
// of the warp count, but the prologue and the inter-warp reduction are per-warp overheads, so when
// K is large enough, fewer warps with more K each means fewer instructions overall.  Solving for
// the fixed cost C and loop body L from two measured shapes gives C = 2.09 L; at K = 4096 (two
// iterations per thread) half of a warp's work was overhead.  At nwarps = 1 the shared-memory
// inter-warp reduction (LDS/STS/SYNC) disappears entirely, because warp_reduce_sum leaves the total
// in every lane -- no shared memory and no __syncthreads.
//
// minBlocksPerMultiprocessor caps registers, so it has to be scaled up as the block shrinks;
// otherwise nwarps = 1 lets register use run free and resident blocks drop.  Multiplying by
// (A16_NWARPS/nwarps) keeps the implied floor on resident threads per SM constant.
// min_blocks_tpl == 0 keeps the historical occupancy target.  It is otherwise the
// minBlocksPerMultiprocessor asked of the compiler, which on GP100 is what fixes both the
// register cap (65536 / (min_blocks * threads)) and the resident-warp count.  The production
// width-4 two-warp shape resolves to 8, i.e. 128 registers and 512 threads per SM (25%
// occupancy); 10, 12, and 16 trade registers for residency.
template <int ncols_dst, int nwarps, int nrows_block = A16_ROWS, bool pair_halves = false,
          bool raw_activations = false, int min_blocks_tpl = 0>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps,
        min_blocks_tpl != 0 ? min_blocks_tpl : (ncols_dst >= 6 ? 2 : 4)*(A16_NWARPS/nwarps))
mul_mat_vec_q4_1_a16(
        const void * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int nblocks, const int stride_row_x,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
#if defined(FP16_AVAILABLE)
    const int row0     = blockIdx.x*nrows_block;
    const int tid      = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nthreads = WARP_SIZE*nwarps;

    const block_q4_1 * x = (const block_q4_1 *) vx;

    const half2 magic_lo = __float2half2_rn(1024.0f);
    const half2 magic_hi = __float2half2_rn(  64.0f);

    float sumf[ncols_dst][nrows_block] = {{0.0f}};

    // block_q4_1 is 20 bytes, so `&x[(row0+i)*stride_row_x + kb]` becomes a per-row 64-bit
    // multiply (a 32-bit index times 20, i.e. an XMAD chain).  Hoisting each row's base out and
    // adding a byte offset built once from kb removes the multiplies.
    const char * const xbase = (const char *) &x[row0*stride_row_x];
    int rowoff[nrows_block];
#pragma unroll
    for (int i = 0; i < nrows_block; ++i) {
        rowoff[i] = i*stride_row_x*(int) sizeof(block_q4_1);
    }

    for (int t = tid; t < 2*nblocks; t += nthreads) {
        const int kb = t >> 1;
        const int kh = t &  1;
        const int koff = kb*(int) sizeof(block_q4_1);

        half2 a[ncols_dst][8];
        float yd[ncols_dst];
        float ys[ncols_dst];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const half2 * ap = aq + (size_t) j*stride_col_aq + kb*(A16_QK/2) + kh*8;
#pragma unroll
            for (int c2 = 0; c2 < 2; ++c2) {
                const int4 av = *((const int4 *) (ap + 4*c2));
                a[j][4*c2 + 0] = *((const half2 *) &av.x);
                a[j][4*c2 + 1] = *((const half2 *) &av.y);
                a[j][4*c2 + 2] = *((const half2 *) &av.z);
                a[j][4*c2 + 3] = *((const half2 *) &av.w);
            }
            const half2 ds = ads[j*stride_col_ads + kb];
            if constexpr (!raw_activations) {
                yd[j] = __low2float(ds);
            }
            ys[j] = __high2float(ds)*0.5f;
        }

#pragma unroll
        for (int i = 0; i < nrows_block; ++i) {
            const block_q4_1 * b = (const block_q4_1 *) (xbase + (rowoff[i] + koff));
            const float2 dm = __half22float2(b->dm);
            const int * wq = (const int *) b->qs;

            half2 w[8];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int w32  = wq[2*kh + k];
                const int w32s = w32 >> 8;
                int t0, t1, t2, t3;
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t0) : "r"(w32),  "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t1) : "r"(w32),  "n"(0x00f000f0), "n"(0x54005400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t2) : "r"(w32s), "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t3) : "r"(w32s), "n"(0x00f000f0), "n"(0x54005400));
                w[4*k + 0] = __hsub2(*((const half2 *) &t0), magic_lo);
                w[4*k + 1] = __hsub2(*((const half2 *) &t1), magic_hi);
                w[4*k + 2] = __hsub2(*((const half2 *) &t2), magic_lo);
                w[4*k + 3] = __hsub2(*((const half2 *) &t3), magic_hi);
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                half2 s = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int c2 = 0; c2 < 2; ++c2) {
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        s = __hfma2(w[4*c2 + c], a[j][4*c2 + c], s);
                    }
                }
                const float af = __half2float(__hadd(__low2half(s), __high2half(s)));
                if constexpr (pair_halves) {
                    // Adjacent lanes own kh=0/1 of the same 32-value Q4_1 block.  Combine their
                    // dot products before applying d/m so only the even lane pays the expensive
                    // FP32 epilogue.  The launch path enables this only when every warp has a
                    // complete final iteration, which makes the full-warp shuffle mask valid.
                    const float af_peer = __shfl_xor_sync(0xffffffffu, af, 1);
                    if ((threadIdx.x & 1) == 0) {
                        if constexpr (raw_activations) {
                            sumf[j][i] += dm.x*(af + af_peer) + dm.y*(2.0f*ys[j]);
                        } else {
                            sumf[j][i] += yd[j]*(dm.x*(af + af_peer) + dm.y*(2.0f*ys[j]));
                        }
                    }
                } else {
                    if constexpr (raw_activations) {
                        sumf[j][i] += dm.x*af + dm.y*ys[j];
                    } else {
                        sumf[j][i] += yd[j]*(dm.x*af + dm.y*ys[j]);
                    }
                }
            }
        }
    }

    // warp_reduce_sum is a __shfl_xor_sync butterfly, so the total ends up in every lane.  With a
    // single warp the storing lane can be rotated per row, so no shared memory and no
    // __syncthreads are needed.
    if constexpr (nwarps == 1) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < nrows_block; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (threadIdx.x == i) {
                    dst[j*stride_col_dst + row0 + i] = v;
                }
            }
        }
    } else {
        __shared__ float tmp[nwarps][ncols_dst][nrows_block];

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < nrows_block; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (threadIdx.x == 0) {
                    tmp[threadIdx.y][j][i] = v;
                }
            }
        }

        __syncthreads();

        if (threadIdx.y != 0) {
            return;
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < nrows_block; ++i) {
                if (threadIdx.x == i) {
                    float s = 0.0f;
#pragma unroll
                    for (int w = 0; w < nwarps; ++w) {
                        s += tmp[w][j][i];
                    }
                    dst[j*stride_col_dst + row0 + i] = s;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// Warp count: the K loop runs 2*nblocks iterations shared across the block, so a wide grid
// wants fewer warps (less cross-warp reduction) and a narrow one wants more (to fill the SM).
// The thresholds below are measured on a P100; they only affect scheduling, never arithmetic.
static int a16_pick_nwarps(const int nblocks, const int nblocks_grid) {
    int nw = nblocks_grid < 64 ? A16_NWARPS : 2;
    while (nw > 1 && 2*nblocks < WARP_SIZE*nw) {
        nw /= 2;
    }
    return nw;
}

template <int ncols_dst>
static void launch_a16(
        const void * vx, const half2 * aq, const half2 * ads, float * dst, const int nblocks,
        const int nrows, const int stride_row_x, const int stride_col_aq, const int stride_col_ads,
        const int stride_col_dst, cudaStream_t stream) {
    // Tall tensors take the 8-row tile: it amortises the per-block scale load and the row
    // pointer arithmetic across twice as many rows.  One warp is measurably better there
    // because the 8-row reduction then needs no shared memory.
    if (nrows >= 1024 && nrows % 8 == 0) {
        const dim3 grid(nrows/8, 1, 1);
        switch (a16_pick_nwarps(nblocks, nrows/8)) {
            case 1:
                mul_mat_vec_q4_1_a16<ncols_dst, 1, 8><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                    vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
                return;
            case 2:
                mul_mat_vec_q4_1_a16<ncols_dst, 2, 8><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                    vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
                return;
            default:
                mul_mat_vec_q4_1_a16<ncols_dst, A16_NWARPS, 8><<<grid, dim3(WARP_SIZE, A16_NWARPS, 1), 0, stream>>>(
                    vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
                return;
        }
    }

    const dim3 grid(nrows/A16_ROWS, 1, 1);
    switch (a16_pick_nwarps(nblocks, nrows/A16_ROWS)) {
        case 1:
            mul_mat_vec_q4_1_a16<ncols_dst, 1><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
            return;
        case 2:
            mul_mat_vec_q4_1_a16<ncols_dst, 2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
            return;
        default:
            mul_mat_vec_q4_1_a16<ncols_dst, A16_NWARPS><<<grid, dim3(WARP_SIZE, A16_NWARPS, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst);
            return;
    }
}

bool ggml_cuda_mmvq_f16_sm60_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst) {
    if (ids) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_1 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // No DP4A but full-rate HFMA2 means GP100 only.  Everywhere else the generic kernel wins.
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_PASCAL || cc >= GGML_CUDA_CC_DP4A) {
        return false;
    }
    // Channel/sample dims are left to the generic path; Q4_1 weights do not use them.
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] % A16_QK != 0 || src0->ne[1] % A16_ROWS != 0) {
        return false;
    }
    // Width 1 is limited by the DRAM read of the weights, so making the arithmetic free moves it
    // by ~1%, while this path costs twice the activation bytes of int8 -- a measured net loss.
    // Above width 8 the generic batched path takes over.
    if (src1->ne[1] < 2 || src1->ne[1] > 8) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

void ggml_cuda_mmvq_f16_sm60(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];
    const int nblocks  = ne00/A16_QK;

    // Activations are quantized to half2 pairs plus a per-block (scale, sum) pair.  The sum lets
    // the Q4_1 `m` term be applied once per block in the epilogue instead of per weight.
    const size_t aq_size  = (size_t) ne11*nblocks*(A16_QK/2)*sizeof(half2);
    const size_t ads_size = (size_t) ne11*nblocks*sizeof(half2);

    const int64_t s11 = src1->nb[1]/ggml_type_size(src1->type);

    // Gate and up projections consume the same activation tensor back to back; cache the
    // quantized form so the second mat-vec skips the quantize launch (see common.cuh).
    // A second concurrent stream falls back to scoped pool storage.
    ggml_cuda_pool_alloc<half2> aq_scoped (ctx.pool());
    ggml_cuda_pool_alloc<half2> ads_scoped(ctx.pool());
    half2 * aq  = nullptr;
    half2 * ads = nullptr;
    const size_t total_size = aq_size + ads_size;

    const bool cacheable = ctx.a16_cache_stream == nullptr || ctx.a16_cache_stream == stream;
    const bool hit = cacheable &&
                     ctx.a16_cache_mem != nullptr &&
                     ctx.a16_cache_src1     == src1 &&
                     ctx.a16_cache_data     == src1->data &&
                     ctx.a16_cache_stream   == stream &&
                     ctx.a16_cache_aq_size  == aq_size &&
                     ctx.a16_cache_ads_size == ads_size &&
                     ctx.a16_cache_s[0]     == nblocks &&
                     ctx.a16_cache_s[1]     == ne11 &&
                     ctx.a16_cache_s[2]     == s11;

    if (hit) {
        aq  = (half2 *) ctx.a16_cache_mem;
        ads = (half2 *) (ctx.a16_cache_mem + aq_size);
    } else if (!cacheable) {
        aq  = aq_scoped.alloc(aq_size/sizeof(half2));
        ads = ads_scoped.alloc(ads_size/sizeof(half2));
    } else {
        if (total_size > ctx.a16_cache_cap) {
            ctx.a16_cache_free();
            ggml_cuda_set_device(ctx.device);
            CUDA_CHECK(cudaMalloc((void **) &ctx.a16_cache_mem, total_size));
            ctx.a16_cache_cap = total_size;
        }
        aq  = (half2 *) ctx.a16_cache_mem;
        ads = (half2 *) (ctx.a16_cache_mem + aq_size);
    }

    if (!hit) {
        const int nthreads = nblocks*A16_QG;
        const dim3 grid((nthreads + A16_QBLOCK - 1)/A16_QBLOCK, ne11, 1);
        // src1 need not be contiguous, so use float4 loads only when each row starts on a
        // 16-byte boundary; the scalar fallback costs ~0.5 us at these sizes.
        const bool vec4 = s11 % 4 == 0 && ((uintptr_t) src1->data) % 16 == 0;
        if (vec4) {
            quantize_a16<true> <<<grid, A16_QBLOCK, 0, stream>>>(
                (const float *) src1->data, aq, ads, s11, nblocks);
        } else {
            quantize_a16<false><<<grid, A16_QBLOCK, 0, stream>>>(
                (const float *) src1->data, aq, ads, s11, nblocks);
        }
        if (cacheable) {
            ctx.a16_cache_src1     = src1;
            ctx.a16_cache_data     = src1->data;
            ctx.a16_cache_stream   = stream;
            ctx.a16_cache_aq_size  = aq_size;
            ctx.a16_cache_ads_size = ads_size;
            ctx.a16_cache_s[0]     = nblocks;
            ctx.a16_cache_s[1]     = ne11;
            ctx.a16_cache_s[2]     = s11;
        }
    }

    const int stride_row_x   = ne00/ggml_blck_size(src0->type);
    const int stride_col_aq  = nblocks*(A16_QK/2);
    const int stride_col_ads = nblocks;
    const int stride_col_dst = dst->nb[1]/ggml_type_size(dst->type);

#define A16_CASE(N)                                                                            \
    case N:                                                                                    \
        launch_a16<N>(src0->data, aq, ads, (float *) dst->data, nblocks, ne01,     \
                      stride_row_x, stride_col_aq, stride_col_ads, stride_col_dst, stream);    \
        break;
    switch (ne11) {
        A16_CASE(2) A16_CASE(3) A16_CASE(4)
        A16_CASE(5) A16_CASE(6) A16_CASE(7) A16_CASE(8)
        default:
            GGML_ABORT("unsupported width for the sm_60 Q4_1 mat-vec");
    }
#undef A16_CASE
}
