#pragma once

// Interface between the CUDA side of the host cold-expert path (moe-host.cu) and its host worker pool
// (moe-host-cpu.cpp, plain C++ so it can use AVX-512 intrinsics). See moe-host.cuh for the design.

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define MH_HD __host__ __device__
#else
#define MH_HD
#endif

#define MH_MAX_EMBD  8192
#define MH_MAX_FF    4096

// A 64-element activation block quantized to int8 for the Q2_0 dot product, written by the GPU (mh_publish).
// q is in lane order: byte 16*L + j holds element 4*j + L, which is where the codes land when the 16 code bytes
// are broadcast to four 128-bit lanes and lane L is shifted right by 2*L. s holds the scale of each int32 lane of
// madd(maddubs(codes, q)): lanes 4L+0, 4L+1 sum elements 0..31, lanes 4L+2, 4L+3 elements 32..63.
// The codes are unsigned (value + 1), so the dot product subtracts corr = sum of the dequantized activations.
struct mh_act_block {
    int8_t q[64];
    float  s[16];
    float  corr;
    float  pad[15];
};
static_assert(sizeof(mh_act_block) == 192, "unexpected act block size");

// One mailbox per device context, in mapped pinned memory: this header, then the pair lists, the quantized input
// rows and the result rows at the offsets it records (sized at allocation for cap_tok tokens of cap_embd values and
// cap_pairs cold pairs). The GPU writes the request and then `req`; the host writes the results and then `done`.
// Layers run one after another on a device, so one mailbox per device is enough.
struct alignas(64) mh_mailbox {
    volatile uint32_t req;   uint32_t pad0[15];
    volatile uint32_t done;  uint32_t pad1[15];
    int32_t slot;
    int32_t n_tokens;
    int32_t n_used;
    int32_t n_pairs;
    int32_t cap_tok;
    int32_t cap_pairs;
    int32_t cap_embd;
    int32_t pad2;
    int64_t off_pair_idx;                   // int32_t[cap_pairs]: t*n_used + k
    int64_t off_pair_exp;                   // int32_t[cap_pairs]: cold expert index (id - n_hot)
    int64_t off_xq;                         // mh_act_block[cap_tok*cap_embd/64]: the input rows, quantized by the GPU
    int64_t off_y;                          // float[cap_pairs*cap_embd]: the cold pairs' down-projection outputs
};

MH_HD inline int32_t      * mh_pair_idx(mh_mailbox * m) { return (int32_t      *) ((char *) m + m->off_pair_idx); }
MH_HD inline int32_t      * mh_pair_exp(mh_mailbox * m) { return (int32_t      *) ((char *) m + m->off_pair_exp); }
MH_HD inline mh_act_block * mh_xq      (mh_mailbox * m) { return (mh_act_block *) ((char *) m + m->off_xq); }
MH_HD inline float        * mh_y       (mh_mailbox * m) { return (float        *) ((char *) m + m->off_y); }

// Lays out a mailbox of the given capacities: returns the bytes it needs and, when m is given, fills its header.
inline size_t mh_mailbox_layout(mh_mailbox * m, int cap_tok, int cap_pairs, int cap_embd) {
    auto up64 = [](size_t x) { return (x + 63) & ~(size_t) 63; };
    size_t off = up64(sizeof(mh_mailbox));
    const size_t o_idx = off; off = up64(off + (size_t) cap_pairs*sizeof(int32_t));
    const size_t o_exp = off; off = up64(off + (size_t) cap_pairs*sizeof(int32_t));
    const size_t o_xq  = off; off = up64(off + (size_t) cap_tok*(cap_embd/64)*sizeof(mh_act_block));
    const size_t o_y   = off; off = up64(off + (size_t) cap_pairs*cap_embd*sizeof(float));
    if (m) {
        m->cap_tok      = cap_tok;
        m->cap_pairs    = cap_pairs;
        m->cap_embd     = cap_embd;
        m->off_pair_idx = (int64_t) o_idx;
        m->off_pair_exp = (int64_t) o_exp;
        m->off_xq       = (int64_t) o_xq;
        m->off_y        = (int64_t) o_y;
    }
    return off;
}

// One hot/cold MoE layer: the cold Q2_0 experts in host memory.
struct mh_slot_desc {
    const uint8_t * up;
    const uint8_t * gate;
    const uint8_t * down;
    size_t up_nb1,   up_nb2;
    size_t gate_nb1, gate_nb2;
    size_t down_nb1, down_nb2;
    int    n_embd;
    int    n_ff;
};

bool mh_cpu_supported();
// Starts the workers on first call; later calls only attach the context's mailbox.
void mh_pool_attach(int index, mh_mailbox * mb);
// Returns the slot id of the layer (registered once, looked up by its down tensor afterwards).
int  mh_pool_register(const mh_slot_desc & d);
