#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Batched scalar copy.  A tensor-parallel recurrent model emits one small strided CPY per
// rollback snapshot per layer (48 layers x 8 snapshots on Qwen3.8), each a few microseconds of
// launch overhead for a few kilobytes of data.  ggml_cuda_cpy_batch_count() reports how many
// consecutive CPY nodes starting at `i` share a shape, strides and type and cannot alias each
// other, and ggml_cuda_cpy_batch() runs them in one launch.
#define GGML_CUDA_CPY_BATCH_MAX 16

// Fills idx[] with the graph indices of the batchable copies (views and other no-ops between them
// are skipped, as the compute loop skips them too) and returns how many there are; *span is the
// number of graph nodes they cover, i.e. what the caller must skip.
int  ggml_cuda_cpy_batch_plan(const ggml_cgraph * cgraph, int i, int max_n, int * idx, int * span);
void ggml_cuda_cpy_batch(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int * idx, int n);
