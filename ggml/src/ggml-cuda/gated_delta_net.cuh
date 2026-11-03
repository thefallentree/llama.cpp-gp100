#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Description of a folded-in SSM state GET_ROWS: read row rows[seq] straight out of base, and write
// the same values to gather_dst so the folded GET_ROWS node is preserved.
struct ggml_cuda_gated_delta_net_fused_gather {
    const float *   base       = nullptr; // the GET_ROWS input (the state cache)
    const int32_t * rows       = nullptr; // row indices (n_seqs of them)
    int64_t         row_stride = 0;       // floats per cache row
    float *         gather_dst = nullptr; // dst of the folded GET_ROWS node
};

// Variant with beta = sigmoid(x) folded into the prologue.  cache behaves as before (nullptr
// disables it).  With fuse_beta_sigmoid the kernel reads dst->src[4]->src[0], applies sigmoid,
// and also writes the result to dst->src[4]->data, so the folded node is preserved.
void ggml_cuda_op_gated_delta_net_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_cuda_gated_delta_net_fused_cache *  cache,
                                        bool                                           fuse_beta_sigmoid,
                                        const ggml_cuda_gated_delta_net_fused_gather * gather = nullptr);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);
