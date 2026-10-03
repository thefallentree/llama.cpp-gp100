#pragma once

// Interface between the CUDA side of the host cold-expert path (moe-host.cu) and its host worker pool
// (moe-host-cpu.cpp, plain C++ so it can use AVX-512 intrinsics). See moe-host.cuh for the design.

#include <cstddef>
#include <cstdint>

#define MH_MAX_TOK   8                      // widest ubatch offloaded: the sm_60 Q2_0 mat-vec path
#define MH_MAX_PAIRS 80                     // (token, expert slot) pairs per op
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

// One mailbox per device, in mapped pinned memory. The GPU writes the request (x, the cold pairs) and then
// `req`; the host writes the results (y) and then `done`. Layers run one after another on a device, so one
// mailbox per device is enough.
struct alignas(64) mh_mailbox {
    volatile uint32_t req;   uint32_t pad0[15];
    volatile uint32_t done;  uint32_t pad1[15];
    int32_t slot;
    int32_t n_tokens;
    int32_t n_used;
    int32_t n_pairs;
    int32_t pad2[12];
    int32_t pair_idx[MH_MAX_PAIRS];         // t*n_used + k
    int32_t pair_exp[MH_MAX_PAIRS];         // cold expert index (id - n_hot)
    alignas(64) mh_act_block xq[MH_MAX_TOK*MH_MAX_EMBD/64]; // the tokens' input rows, quantized by the GPU
    alignas(64) float y[MH_MAX_PAIRS*MH_MAX_EMBD];  // the cold pairs' down-projection outputs
};

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
// Starts the workers on first call; later calls only attach the device's mailbox.
void mh_pool_attach(int device, mh_mailbox * mb);
// Returns the slot id of the layer (registered once, looked up by its down tensor afterwards).
int  mh_pool_register(const mh_slot_desc & d);
