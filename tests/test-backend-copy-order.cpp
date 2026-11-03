// An asynchronous copy between two device backends must not land in the destination buffer while work queued
// earlier on the destination backend still reads it (ggml_backend_tensor_copy_async: "after all the queued
// operations on both backends"). The scheduler relies on this when its input copies of consecutive splits share
// memory: with its events on it does not wait for the destination backend before the copy.
//
// dev0 computes out = W x A (long enough to overlap a copy); right after the launch, B (on dev1, other values) is
// copied into A from dev1. out must be what the original A gives. Needs two device backends, else it is skipped.

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    ggml_backend_load_all();

    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devs.push_back(dev);
        }
    }
    if (devs.size() < 2) {
        printf("test-backend-copy-order: fewer than 2 device backends, skipped\n");
        return 0;
    }

    ggml_backend_t b0 = ggml_backend_dev_init(devs[0], nullptr);
    ggml_backend_t b1 = ggml_backend_dev_init(devs[1], nullptr);
    if (b0 == nullptr || b1 == nullptr) {
        printf("test-backend-copy-order: backend init failed, skipped\n");
        return 0;
    }
    printf("test-backend-copy-order: %s (dst) and %s (src)\n", ggml_backend_name(b0), ggml_backend_name(b1));

    const int64_t K = 8192, M = 8192, N = 2048;

    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * out = ggml_mul_mat(ctx, W, A);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf0 = ggml_backend_alloc_ctx_tensors(ctx, b0);

    ggml_context * ctx1 = ggml_init(ip);
    ggml_tensor * B = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, K, N);
    ggml_backend_buffer_t buf1 = ggml_backend_alloc_ctx_tensors(ctx1, b1);

    if (buf0 == nullptr || buf1 == nullptr) {
        printf("test-backend-copy-order: allocation failed, skipped\n");
        return 0;
    }

    std::vector<float> w(K*M), a(K*N), b(K*N);
    for (int64_t i = 0; i < K*M; i++) {
        w[i] = (float) ((i*7919) % 13) - 6.0f;
    }
    for (int64_t i = 0; i < K*N; i++) {
        a[i] = (float) ((i*104729) % 11) - 5.0f;
        b[i] = 1000.0f;   // far from a: a copy that lands early changes out
    }
    ggml_backend_tensor_set(W, w.data(), 0, ggml_nbytes(W));
    ggml_backend_tensor_set(B, b.data(), 0, ggml_nbytes(B));

    // the reference: the product of the original A, computed without a copy in flight
    std::vector<float> ref(M*N), got(M*N);
    ggml_backend_tensor_set(A, a.data(), 0, ggml_nbytes(A));
    ggml_backend_graph_compute(b0, gf);
    ggml_backend_tensor_get(out, ref.data(), 0, ggml_nbytes(out));

    int failures = 0;
    const int n_iter = 5;
    for (int it = 0; it < n_iter; it++) {
        ggml_backend_tensor_set(A, a.data(), 0, ggml_nbytes(A));
        ggml_backend_synchronize(b0);
        ggml_backend_graph_compute_async(b0, gf);
        // B lands in A: after the product above, not during it
        ggml_backend_tensor_copy_async(b1, b0, B, A);
        ggml_backend_synchronize(b1);
        ggml_backend_synchronize(b0);
        ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
        int64_t n_diff = 0;
        for (int64_t i = 0; i < M*N; i++) {
            if (got[i] != ref[i]) {
                n_diff++;
            }
        }
        // A holds B now
        std::vector<float> a_now(K*N);
        ggml_backend_tensor_get(A, a_now.data(), 0, ggml_nbytes(A));
        const bool copied = memcmp(a_now.data(), b.data(), ggml_nbytes(A)) == 0;
        printf("  iteration %d: %lld of %lld outputs differ from the reference, copy landed: %s\n", it, (long long) n_diff, (long long) (M*N), copied ? "yes" : "NO");
        if (n_diff != 0 || !copied) {
            failures++;
        }
    }

    ggml_backend_buffer_free(buf0);
    ggml_backend_buffer_free(buf1);
    ggml_free(ctx);
    ggml_free(ctx1);
    ggml_backend_free(b0);
    ggml_backend_free(b1);

    if (failures > 0) {
        printf("test-backend-copy-order: FAIL (%d of %d iterations: the copy landed while the product read its destination)\n", failures, n_iter);
        return 1;
    }
    printf("test-backend-copy-order: OK\n");
    return 0;
}
