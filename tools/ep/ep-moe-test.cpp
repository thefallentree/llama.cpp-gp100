// EP correctness test: one MoE layer (router, top-k, gate/up/down experts as Q2_K/Q2_K/Q3_K, weighted sum) computed
//  (a) on one node, and (b) expert-parallel: the experts split over 3 shards (shard 0 local, shards 1-2 on ggml-rpc servers
//      started in this process), each shard computing only its own experts (the other pairs carry id -1 = skipped) and
//      returning its [n_embd, T] partial sum; the head adds the partials.
// Prints the NMSE of (b) against (a). usage: ep-moe-test [-T tokens] [-e experts] [-k used] [-p partition 0:mod 1:blocks 2:skewed]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <vector>

static const int N_EMBD = 256, N_FF = 256;

static ggml_tensor * sum_used(ggml_context * ctx, ggml_tensor * experts, int n_used) { // [n_embd, n_used, T] -> [n_embd, T]
    ggml_tensor * cur = nullptr;
    for (int i = 0; i < n_used; i++) {
        ggml_tensor * v = ggml_view_2d(ctx, experts, experts->ne[0], experts->ne[2], experts->nb[2], i * experts->nb[1]);
        cur = cur ? ggml_add(ctx, cur, v) : v;
    }
    return ggml_cont(ctx, cur);
}

// the experts' part: x [n_embd,T], ids [n_used,T], w [1,n_used,T] -> [n_embd,T]
static ggml_tensor * experts_fwd(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down,
                                 ggml_tensor * x, ggml_tensor * ids, ggml_tensor * w, int n_used) {
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, N_EMBD, 1, x->ne[1]);
    ggml_tensor * u = ggml_mul_mat_id(ctx, up, x3, ids);
    ggml_tensor * g = ggml_mul_mat_id(ctx, gate, x3, ids);
    ggml_tensor * h = ggml_swiglu_split(ctx, g, u);
    ggml_tensor * d = ggml_mul_mat_id(ctx, down, h, ids);
    d = ggml_mul(ctx, d, w);
    return sum_used(ctx, d, n_used);
}

static void quant_fill(ggml_tensor * t, std::mt19937 & rng, float scale) { // random f32 -> quantized, expert by expert
    std::normal_distribution<float> nd(0.f, scale);
    const int64_t row = t->ne[0], rows = t->ne[1] * t->ne[2];
    std::vector<float> f(row * rows);
    for (auto & v : f) v = nd(rng);
    ggml_quantize_init(t->type);
    ggml_quantize_chunk(t->type, f.data(), t->data, 0, rows, row, nullptr);
}

static double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    double e = 0, n = 0;
    for (size_t i = 0; i < a.size(); i++) { e += (a[i] - ref[i]) * (a[i] - ref[i]); n += (double) ref[i] * ref[i]; }
    return e / n;
}

int main(int argc, char ** argv) {
    int T = 6, n_exp = 24, n_used = 6, part = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-T") T = atoi(argv[++i]); else if (a == "-e") n_exp = atoi(argv[++i]);
        else if (a == "-k") n_used = atoi(argv[++i]); else if (a == "-p") part = atoi(argv[++i]);
    }
    const int N_SH = 3;
    std::mt19937 rng(1234);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);

    // owner of each expert and its slot in the owner's tensor
    std::vector<int> owner(n_exp), slot(n_exp), cnt(N_SH, 0);
    for (int e = 0; e < n_exp; e++) {
        owner[e] = part == 0 ? e % N_SH : part == 1 ? std::min(e * N_SH / n_exp, N_SH - 1) : (e < n_exp / 2 ? 0 : (e < 3 * n_exp / 4 ? 1 : 2));
        slot[e] = cnt[owner[e]]++;
    }

    // full model layer on the "single node"
    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(1024, false), nullptr, true };
    ggml_context * cw = ggml_init(ip);
    ggml_tensor * router = ggml_new_tensor_2d(cw, GGML_TYPE_F32, N_EMBD, n_exp);
    ggml_tensor * gate = ggml_new_tensor_3d(cw, GGML_TYPE_Q2_K, N_EMBD, N_FF, n_exp);
    ggml_tensor * up   = ggml_new_tensor_3d(cw, GGML_TYPE_Q2_K, N_EMBD, N_FF, n_exp);
    ggml_tensor * down = ggml_new_tensor_3d(cw, GGML_TYPE_Q3_K, N_FF, N_EMBD, n_exp);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, cpu);
    { std::normal_distribution<float> nd(0, 1); std::vector<float> r(N_EMBD * n_exp); for (auto & v : r) v = nd(rng); ggml_backend_tensor_set(router, r.data(), 0, ggml_nbytes(router)); }
    quant_fill(gate, rng, 0.05f); quant_fill(up, rng, 0.05f); quant_fill(down, rng, 0.05f);

    std::vector<float> xh(N_EMBD * T);
    { std::normal_distribution<float> nd(0, 1); for (auto & v : xh) v = nd(rng); }

    // (a) reference graph, with the routing
    std::vector<float> yref(N_EMBD * T);
    std::vector<int32_t> ids_h(n_used * T);
    std::vector<float> w_h(n_used * T);
    {
        ggml_context * c = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, N_EMBD, T);
        ggml_set_input(x);
        ggml_tensor * probs = ggml_soft_max(c, ggml_mul_mat(c, router, x));
        ggml_tensor * ids = ggml_argsort_top_k(c, probs, n_used);
        ggml_tensor * w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, n_exp, T), ids);
        { ggml_tensor * w2 = ggml_reshape_2d(c, w, n_used, T);
          ggml_tensor * den = ggml_repeat(c, ggml_sum_rows(c, w2), w2);
          w = ggml_reshape_3d(c, ggml_div(c, w2, den), 1, n_used, T); }
        ggml_tensor * y = experts_fwd(c, gate, up, down, x, ids, w, n_used);
        ggml_tensor * idsc = ggml_cont(c, ids); // top_k returns a strided view
        ggml_set_output(idsc); ggml_set_output(w); ggml_set_output(y);
        ggml_cgraph * gf = ggml_new_graph_custom(c, 1024, false);
        ggml_build_forward_expand(gf, y); ggml_build_forward_expand(gf, idsc); ggml_build_forward_expand(gf, w);
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(c, cpu);
        ggml_backend_tensor_set(x, xh.data(), 0, ggml_nbytes(x));
        ggml_backend_graph_compute(cpu, gf);
        ggml_backend_tensor_get(y, yref.data(), 0, ggml_nbytes(y));
        ggml_backend_tensor_get(idsc, ids_h.data(), 0, ggml_nbytes(idsc));
        ggml_backend_tensor_get(w, w_h.data(), 0, ggml_nbytes(w));
        ggml_backend_buffer_free(b); ggml_free(c);
    }

    // (b) EP: start the servers (shards 1, 2), build the shards
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    const char * eps[N_SH] = { nullptr, "127.0.0.1:50311", "127.0.0.1:50312" };
    for (int k = 1; k < N_SH; k++) {
        std::string ep = eps[k];
        std::thread([ep, cpu_dev]() { ggml_backend_dev_t d = cpu_dev; ggml_backend_rpc_start_server(ep.c_str(), nullptr, 2, 1, &d); }).detach();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    struct shard { ggml_backend_t be; ggml_context * c; ggml_backend_buffer_t buf; ggml_tensor *g, *u, *d, *x, *ids, *w, *y; ggml_cgraph * gf; std::vector<int32_t> idl; std::vector<float> part; };
    std::vector<shard> sh(N_SH);
    const size_t sl_gu = ggml_row_size(GGML_TYPE_Q2_K, N_EMBD) * N_FF, sl_d = ggml_row_size(GGML_TYPE_Q3_K, N_FF) * N_EMBD;
    std::vector<char> tmp;
    for (int k = 0; k < N_SH; k++) {
        shard & s = sh[k];
        s.be = k == 0 ? cpu : ggml_backend_rpc_init(eps[k], 0);
        if (!s.be) { fprintf(stderr, "no backend for shard %d\n", k); return 2; }
        s.c = ggml_init(ip);
        const int nk = std::max(cnt[k], 1);
        s.g = ggml_new_tensor_3d(s.c, GGML_TYPE_Q2_K, N_EMBD, N_FF, nk);
        s.u = ggml_new_tensor_3d(s.c, GGML_TYPE_Q2_K, N_EMBD, N_FF, nk);
        s.d = ggml_new_tensor_3d(s.c, GGML_TYPE_Q3_K, N_FF, N_EMBD, nk);
        s.x = ggml_new_tensor_2d(s.c, GGML_TYPE_F32, N_EMBD, T);
        s.ids = ggml_new_tensor_2d(s.c, GGML_TYPE_I32, n_used, T);
        s.w = ggml_new_tensor_3d(s.c, GGML_TYPE_F32, 1, n_used, T);
        s.y = experts_fwd(s.c, s.g, s.u, s.d, s.x, s.ids, s.w, n_used);
        ggml_set_output(s.y);
        s.gf = ggml_new_graph_custom(s.c, 1024, false);
        ggml_build_forward_expand(s.gf, s.y);
        s.buf = ggml_backend_alloc_ctx_tensors(s.c, s.be);
        for (int e = 0; e < n_exp; e++) if (owner[e] == k) { // the node loads only its own shard
            tmp.resize(sl_d);
            ggml_backend_tensor_get(gate, tmp.data(), (size_t) e * sl_gu, sl_gu); ggml_backend_tensor_set(s.g, tmp.data(), (size_t) slot[e] * sl_gu, sl_gu);
            ggml_backend_tensor_get(up,   tmp.data(), (size_t) e * sl_gu, sl_gu); ggml_backend_tensor_set(s.u, tmp.data(), (size_t) slot[e] * sl_gu, sl_gu);
            ggml_backend_tensor_get(down, tmp.data(), (size_t) e * sl_d,  sl_d);  ggml_backend_tensor_set(s.d, tmp.data(), (size_t) slot[e] * sl_d,  sl_d);
        }
        s.idl.resize(n_used * T);
        s.part.resize(N_EMBD * T);
    }
    // per-layer round: send x, ids (owned -> local slot, others -> -1), w; run; fetch partials
    std::vector<float> yep(N_EMBD * T, 0.f);
    for (int k = 0; k < N_SH; k++) {
        shard & s = sh[k];
        for (int i = 0; i < n_used * T; i++) s.idl[i] = owner[ids_h[i]] == k ? slot[ids_h[i]] : -1;
        ggml_backend_tensor_set_async(s.be, s.x, xh.data(), 0, ggml_nbytes(s.x));
        ggml_backend_tensor_set_async(s.be, s.ids, s.idl.data(), 0, ggml_nbytes(s.ids));
        ggml_backend_tensor_set_async(s.be, s.w, w_h.data(), 0, ggml_nbytes(s.w));
        ggml_backend_graph_compute_async(s.be, s.gf);
        ggml_backend_tensor_get_async(s.be, s.y, s.part.data(), 0, ggml_nbytes(s.y));
    }
    int n_pairs[N_SH] = {0};
    for (int i = 0; i < n_used * T; i++) n_pairs[owner[ids_h[i]]]++;
    for (int k = 0; k < N_SH; k++) {
        ggml_backend_synchronize(sh[k].be);
        if (getenv("EP_DROP") && atoi(getenv("EP_DROP")) == k) continue; // mutation check: the test must fail
        for (size_t i = 0; i < yep.size(); i++) yep[i] += sh[k].part[i];
    }
    double e = nmse(yep, yref);
    printf("EP vs single node: T=%d experts=%d used=%d partition=%d  pairs per shard %d/%d/%d  NMSE %.3e  %s\n", T, n_exp, n_used, part,
           n_pairs[0], n_pairs[1], n_pairs[2], e, e < 1e-10 ? "PASS" : "FAIL");
    fflush(stdout);
    _exit(e < 1e-10 ? 0 : 1);
}
