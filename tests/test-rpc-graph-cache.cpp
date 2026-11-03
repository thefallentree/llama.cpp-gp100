// The RPC client and server cache computed graphs by uid and both clear their cache when it holds GRAPH_CACHE_MAX
// (1024) graphs, so they must hold the same set of uids at every point of the message stream. A freed buffer made the
// server invalidate its graphs while keeping their entries, and the client forget its uids: the server's map then
// reached the limit (and cleared) before the client's set did, and the client asked to recompute a uid the server
// no longer had ("graph with uid N not found", the server closed the connection, the client aborted).
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

static const int GRAPH_CACHE_MAX = 1024; // ggml-rpc.cpp

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 2);
    ggml_backend_load_all();
    const char * endpoint = argv[1];

    ggml_backend_t backend = ggml_backend_rpc_init(endpoint, 0);
    GGML_ASSERT(backend != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ 8*ggml_tensor_overhead() + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * out = ggml_scale(ctx, a, 2.0f);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_context * ctx_scratch = ggml_init(params);
    ggml_new_tensor_1d(ctx_scratch, GGML_TYPE_F32, 4);
    ggml_backend_buffer_t scratch = ggml_backend_alloc_ctx_tensors(ctx_scratch, backend);
    GGML_ASSERT(scratch != nullptr);

    uint64_t next_uid = 1000; // any nonzero uid is cached
    const auto compute = [&](uint64_t uid) {
        graph->uid = uid;
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    };

    float x[4] = {1, 2, 3, 4};
    ggml_backend_tensor_set(a, x, 0, sizeof(x));

    // fill the caches part of the way, then free a buffer: both sides drop their graphs here
    const int n_before = GRAPH_CACHE_MAX/2 + 88;
    for (int i = 0; i < n_before; i++) {
        compute(next_uid++);
    }
    ggml_backend_buffer_free(scratch);

    // new graphs, more than the room the server had left when it kept the invalidated entries
    std::vector<uint64_t> uids;
    for (int i = 0; i < GRAPH_CACHE_MAX - n_before + 64; i++) {
        uids.push_back(next_uid++);
        compute(uids.back());
    }

    // the client still holds every uid since the free; recomputing the first one must find it on the server
    float y[4] = {5, 6, 7, 8};
    ggml_backend_tensor_set(a, y, 0, sizeof(y));
    compute(uids.front());
    float r[4];
    ggml_backend_tensor_get(out, r, 0, sizeof(r));
    for (int i = 0; i < 4; i++) {
        if (r[i] != 2.0f*y[i]) {
            fprintf(stderr, "wrong result at %d: %f != %f\n", i, r[i], 2.0f*y[i]);
            return 1;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx_scratch);
    ggml_free(ctx);
    ggml_backend_free(backend);
    printf("OK\n");
    return 0;
}
