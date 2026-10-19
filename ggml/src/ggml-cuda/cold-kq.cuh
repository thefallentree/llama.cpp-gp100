#pragma once

#include "common.cuh"

// Cold experts of a Q2_K / Q3_K hot/cold layer for prompt batches, on the GPU (the Q2_0 layers have the cold prefetch of
// mmid-f16-sm60.cuh). The host tier (moe-host.cuh) computes the cold pairs of decode batches; for a prompt batch many
// tokens share each cold expert, so the host kernels are compute bound while one copy over PCIe serves them all.
//
// A prompt batch (more than MMVQ_MAX_BATCH_SIZE tokens) that the host tier does not take goes through the generic
// MUL_MAT_ID's dequant fallback for all its pairs. The cold experts that the batch touches are copied from the pinned
// cold tensor to a VRAM staging buffer, in chunks that fit GGML_CUDA_COLD_PREFETCH_MB (default 1024; the F16 dequant
// of one expert counts against it), by DMA on a copy stream: the copy of the next chunks overlaps the GEMMs of the
// hot experts and of the chunks before. The buffer is allocated when a batch needs it and grows with the need; what does
// not fit in VRAM is cut (a warning), and when not even one expert fits the cold experts are read in place over UVA.
//
// GGML_CUDA_COLD_PREFETCH_KQ=0 turns it off: such a batch then aborts, as it did before.
// GGML_CUDA_MOE_HOST_PROMPT=1 has priority: the host tier keeps the prompt batches it opens a triple for, this path
// only takes the batches it does not (also the ones above GGML_CUDA_MOE_HOST_MAX_TOK tokens). Set GGML_CUDA_MOE_HOST_PROMPT=0
// to run all prompt batches here, or GGML_CUDA_MOE_HOST_PROMPT_MAX_TOK=n (with PROMPT=1) for the host tier up to n tokens.
// GGML_CUDA_COLD_PREFETCH_STATS=N logs the totals every N ops (the first ops are always logged).
// GGML_CUDA_MMID_ALLGPU=1 still reads the cold experts in place over UVA (a bisect, no staging).
bool ggml_cuda_cold_kq_enabled();

// A prompt-sized Q2_K / Q3_K hot/cold MUL_MAT_ID: whichever tier takes it, its generic path copies the ids to the host
bool ggml_cuda_cold_kq_node(const ggml_tensor * node);

// Starts the copy of the cold experts `used` (indices into the cold tensor at `host`, ascending, nb02 bytes each) of one op.
// reserve: bytes the op's own scratch takes next to the staging buffer. False: no VRAM for it, the caller reads host memory.
bool ggml_cuda_cold_kq_begin(ggml_backend_cuda_context & ctx, const char * host, const std::vector<int32_t> & used, size_t nb02, size_t reserve);

// The VRAM copy of the next expert of `used`; the caller enqueues the work that reads it on ctx.stream() before the next call
char * ggml_cuda_cold_kq_next(ggml_backend_cuda_context & ctx);

void ggml_cuda_cold_kq_end(ggml_backend_cuda_context & ctx);
void ggml_cuda_cold_kq_free(ggml_backend_cuda_context & ctx);
