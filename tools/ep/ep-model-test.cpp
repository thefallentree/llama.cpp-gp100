// End-to-end check of the expert-parallel loader and graph (src/llama-ep.h, deepseek41): loads one DeepSeek-V4.1 GGUF twice, plain
// and with LLAMA_EP_PARTITION (this process is node 0; nodes 1.. are ggml-rpc servers started in this process, whose experts are
// placed on their RPC devices with tensor overrides and loaded by the servers from the shard files, GGML_RPC_FILE_ROOT), runs the same
// tokens through both and prints the NMSE of the logits.
// usage: ep-model-test MODEL.gguf PARTITION.part [-t tokens] [-g ngl] [-n nodes_local]
//   -n K: place the shard tensors of nodes 1..K-1 on RPC servers (default: all of them); -g: layers on the first GPU device (default 0: CPU)
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <sstream>

static std::vector<float> run(llama_model * model, const std::vector<llama_token> & toks, int n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = toks.size(); cp.n_ubatch = toks.size();
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "no context\n"); exit(2); }
    llama_batch b = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = toks.size();
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(2); }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> out((size_t) nv * toks.size());
    for (size_t i = 0; i < toks.size(); i++) memcpy(out.data() + i * nv, llama_get_logits_ith(ctx, i), nv * sizeof(float));
    llama_batch_free(b);
    llama_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL.gguf PARTITION.part [-t tokens] [-g ngl] [-n nodes]\n", argv[0]); return 2; }
    std::string model_path = argv[1], part_path = argv[2];
    int n_tok = 16, ngl = 0, n_rpc = -1;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-t") n_tok = atoi(argv[++i]); else if (a == "-g") ngl = atoi(argv[++i]); else if (a == "-n") n_rpc = atoi(argv[++i]);
    }
    int n_nodes = 0;
    { std::ifstream f(part_path); std::string l; while (std::getline(f, l)) { std::istringstream is(l); std::string k; is >> k; if (k == "nodes") is >> n_nodes; } }
    if (n_rpc < 0) n_rpc = n_nodes - 1;
    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * m0 = llama_model_load_from_file(model_path.c_str(), mp);
    if (!m0) { fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 2; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m0));
    std::mt19937 rng(7);
    std::vector<llama_token> toks(n_tok);
    for (auto & t : toks) t = rng() % nv;
    std::vector<float> ref = run(m0, toks, 256);
    llama_model_free(m0);

    // the EP model: servers in this process, shard tensors of node k on server k
    std::vector<ggml_backend_dev_t> devs;
    std::vector<std::string> pats(n_nodes);
    std::vector<llama_model_tensor_buft_override> ov;
    const char * root = getenv("GGML_RPC_FILE_ROOT");
    if (!root) { setenv("GGML_RPC_FILE_ROOT", "/", 1); }
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    std::vector<ggml_backend_dev_t> srv_devs;
    for (int k = 1; k <= n_rpc; k++) {
        std::string ep = "127.0.0.1:" + std::to_string(50400 + k);
        std::thread([ep, cpu_dev]() { ggml_backend_dev_t d = cpu_dev; ggml_backend_rpc_start_server(ep.c_str(), nullptr, 2, 1, &d); }).detach();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    for (int k = 1; k <= n_rpc; k++) {
        std::string ep = "127.0.0.1:" + std::to_string(50400 + k);
        ggml_backend_reg_t reg = ggml_backend_rpc_add_server(ep.c_str());
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        devs.push_back(dev);
        pats[k] = "blk\\.[0-9]+\\.ffn_(gate|up|down)_exps\\.ep" + std::to_string(k) + "\\.weight";
        ov.push_back({ pats[k].c_str(), ggml_backend_dev_buffer_type(dev) });
    }
    ov.push_back({ nullptr, nullptr });
    devs.push_back(nullptr);
    setenv("LLAMA_EP_PARTITION", part_path.c_str(), 1);
    mp.devices = devs.data();
    mp.tensor_buft_overrides = ov.data();
    llama_model * m1 = llama_model_load_from_file(model_path.c_str(), mp);
    if (!m1) { fprintf(stderr, "cannot load the EP model\n"); return 2; }
    std::vector<float> got = run(m1, toks, 256);
    double e = 0, n = 0;
    for (size_t i = 0; i < ref.size(); i++) { e += (double) (got[i] - ref[i]) * (got[i] - ref[i]); n += (double) ref[i] * ref[i]; }
    const double nmse = e / n;
    printf("EP model vs plain: %zu logits, %d nodes (%d on RPC servers), NMSE %.3e  %s\n", ref.size(), n_nodes, n_rpc, nmse, nmse < 1e-9 ? "PASS" : "FAIL");
    fflush(stdout);
    _exit(nmse < 1e-9 ? 0 : 1);
}
