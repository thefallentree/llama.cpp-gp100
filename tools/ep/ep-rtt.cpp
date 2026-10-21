// EP gate: round trip of a 20 KB hidden state through ggml-rpc, to N servers in parallel.
// usage: ep-rtt [-n iters] [-w warmup] [-s sync] [-k n_ops] host:port [host:port ...]
// per iteration and server: set_tensor(x 5120 f32) + set_tensor(ids 6 i32 + w 6 f32) + graph_compute(tiny) + get_tensor(y 5120 f32)
// async mode: all servers are issued first, then all synchronized (what the scheduler does); sync mode: blocking calls, one by one.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml-impl.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct node {
    ggml_backend_t backend;
    ggml_context * ctx;
    ggml_backend_buffer_t buf;
    ggml_tensor * x, * ids, * w, * y;
    ggml_cgraph * gf;
    std::vector<float> xh, yh;
    std::vector<int32_t> idh;
    std::vector<float> wh;
};

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    int iters = 2000, warm = 200, sync_mode = 0, n_ops = 1;
    std::vector<std::string> eps;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-n") iters = atoi(argv[++i]);
        else if (a == "-w") warm = atoi(argv[++i]);
        else if (a == "-s") sync_mode = atoi(argv[++i]);
        else if (a == "-k") n_ops = atoi(argv[++i]);
        else eps.push_back(a);
    }
    if (eps.empty()) { fprintf(stderr, "usage: ep-rtt [-n iters] [-w warmup] [-s 0|1] [-k ops] host:port...\n"); return 1; }
    std::vector<node> nodes(eps.size());
    for (size_t i = 0; i < eps.size(); i++) {
        node & n = nodes[i];
        n.backend = ggml_backend_rpc_init(eps[i].c_str(), 0);
        if (!n.backend) { fprintf(stderr, "cannot connect %s\n", eps[i].c_str()); return 2; }
        ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        n.ctx = ggml_init(ip);
        n.x   = ggml_new_tensor_1d(n.ctx, GGML_TYPE_F32, 5120);
        n.ids = ggml_new_tensor_1d(n.ctx, GGML_TYPE_I32, 6);
        n.w   = ggml_new_tensor_1d(n.ctx, GGML_TYPE_F32, 6);
        ggml_tensor * t = n.x;
        for (int k = 0; k < n_ops; k++) t = ggml_scale(n.ctx, t, 1.0001f);
        n.y = t;
        n.gf = ggml_new_graph(n.ctx);
        ggml_build_forward_expand(n.gf, n.y);
        if (getenv("EP_UID")) n.gf->uid = 1000 + i; // graph cache: later computes send GRAPH_RECOMPUTE only
        n.buf = ggml_backend_alloc_ctx_tensors(n.ctx, n.backend);
        n.xh.assign(5120, 1.0f); n.yh.assign(5120, 0.0f); n.idh = {1,2,3,4,5,6}; n.wh.assign(6, 0.1f);
    }
    std::vector<double> t_all;
    for (int it = 0; it < warm + iters; it++) {
        double t0 = now_us();
        for (auto & n : nodes) {
            n.xh[0] = (float) it;
            if (sync_mode) {
                ggml_backend_tensor_set(n.x, n.xh.data(), 0, 5120 * 4);
                ggml_backend_tensor_set(n.ids, n.idh.data(), 0, 24);
                ggml_backend_tensor_set(n.w, n.wh.data(), 0, 24);
                ggml_backend_graph_compute(n.backend, n.gf);
                ggml_backend_synchronize(n.backend);
                ggml_backend_tensor_get(n.y, n.yh.data(), 0, 5120 * 4);
            } else {
                ggml_backend_tensor_set_async(n.backend, n.x, n.xh.data(), 0, 5120 * 4);
                ggml_backend_tensor_set_async(n.backend, n.ids, n.idh.data(), 0, 24);
                ggml_backend_tensor_set_async(n.backend, n.w, n.wh.data(), 0, 24);
                ggml_backend_graph_compute_async(n.backend, n.gf);
                ggml_backend_tensor_get_async(n.backend, n.y, n.yh.data(), 0, 5120 * 4);
            }
        }
        for (auto & n : nodes) ggml_backend_synchronize(n.backend);
        double t1 = now_us();
        if (it >= warm) t_all.push_back(t1 - t0);
        if (it == warm + 1) { // check result
            for (auto & n : nodes) if (n.yh[0] == 0.0f && n.yh[1] == 0.0f) fprintf(stderr, "WARNING: output zero\n");
        }
    }
    std::sort(t_all.begin(), t_all.end());
    auto q = [&](double p) { return t_all[(size_t) (p * (t_all.size() - 1))]; };
    double sum = 0; for (double v : t_all) sum += v;
    printf("servers=%zu mode=%s ops=%d iters=%zu: min %.0f  median %.0f  mean %.0f  p90 %.0f  p99 %.0f  max %.0f us  | x40 layers (median) %.2f ms\n",
           eps.size(), sync_mode ? "sync" : "async", n_ops, t_all.size(), q(0), q(0.5), sum / t_all.size(), q(0.9), q(0.99), q(1), q(0.5) * 40 / 1000);
    return 0;
}
