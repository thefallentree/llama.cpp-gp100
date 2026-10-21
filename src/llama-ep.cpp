#include "llama-ep.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

static std::runtime_error ep_err(const std::string & what) {
    return std::runtime_error("expert-parallel partition: " + what);
}

std::shared_ptr<llama_ep_partition> llama_ep_partition::load(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        throw ep_err("cannot open " + path);
    }
    auto p = std::make_shared<llama_ep_partition>();
    std::string line;
    bool magic = false;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream is(line);
        std::string key;
        is >> key;
        if (key == "ds41-ep") {
            int v = 0;
            is >> v;
            if (v != 1) {
                throw ep_err("unknown version in " + path);
            }
            magic = true;
        } else if (!magic) {
            throw ep_err(path + " is not a partition file");
        } else if (key == "nodes") {
            is >> p->n_nodes;
        } else if (key == "experts") {
            is >> p->n_expert;
        } else if (key == "layers") {
            is >> p->n_layer;
            if (p->n_nodes < 1 || p->n_layer < 1) {
                throw ep_err("'layers' before 'nodes'");
            }
            p->shard_path.assign(p->n_nodes, "");
            p->shard_id.assign(p->n_nodes, 0);
            p->ids.assign(p->n_layer, std::vector<std::vector<int32_t>>(p->n_nodes));
        } else if (key == "slice") {
            is >> p->slice[0] >> p->slice[1] >> p->slice[2];
        } else if (key == "shard") {
            int k = 0;
            std::string id, pth;
            is >> k >> id;
            std::getline(is >> std::ws, pth);
            if (k < 1 || k >= p->n_nodes) {
                throw ep_err("bad shard node in " + path);
            }
            p->shard_id[k]   = strtoull(id.c_str(), nullptr, 16);
            p->shard_path[k] = pth;
        } else if (key == "L") {
            int il = -1, k = -1;
            is >> il >> k;
            if (il < 0 || il >= p->n_layer || k < 0 || k >= p->n_nodes) {
                throw ep_err("bad layer/node in " + path);
            }
            int e;
            while (is >> e) {
                if (e < 0 || e >= p->n_expert) {
                    throw ep_err("expert id out of range in " + path);
                }
                p->ids[il][k].push_back(e);
            }
        } else {
            throw ep_err("unknown key '" + key + "' in " + path);
        }
    }
    // every layer: the experts are a partition of 0 .. n_expert-1 (no layer without ids; a layer may leave a node empty? no: >= 1)
    for (int il = 0; il < p->n_layer; ++il) {
        std::vector<int> owner(p->n_expert, 0);
        int n = 0;
        for (int k = 0; k < p->n_nodes; ++k) {
            if (p->ids[il][k].empty()) {
                throw ep_err("layer " + std::to_string(il) + ": node " + std::to_string(k) + " owns no expert");
            }
            for (int e : p->ids[il][k]) {
                if (owner[e]++) {
                    throw ep_err("layer " + std::to_string(il) + ": expert " + std::to_string(e) + " has two owners");
                }
                n++;
            }
        }
        if (n != p->n_expert) {
            throw ep_err("layer " + std::to_string(il) + ": " + std::to_string(n) + " experts of " + std::to_string(p->n_expert) + " are owned");
        }
    }
    return p;
}

static uint64_t ep_align(uint64_t x) {
    return (x + 4095) & ~(uint64_t) 4095;
}

uint64_t llama_ep_partition::shard_offset(int node, int il, int kind) const {
    uint64_t off = 4096;
    for (int l = 0; l <= il; ++l) {
        for (int kd = 0; kd < 3; ++kd) {
            if (l == il && kd == kind) {
                return off;
            }
            off += ep_align((uint64_t) ids[l][node].size() * slice[kd]);
        }
    }
    return off;
}

uint64_t llama_ep_partition::shard_size(int node) const {
    uint64_t off = 4096;
    for (int l = 0; l < n_layer; ++l) {
        for (int kd = 0; kd < 3; ++kd) {
            off += ep_align((uint64_t) ids[l][node].size() * slice[kd]);
        }
    }
    return off;
}

void llama_ep_partition::table(int il, int node, int32_t * out) const {
    for (int e = 0; e < n_expert; ++e) {
        out[e] = -1;
    }
    const auto & v = ids[il][node];
    for (size_t j = 0; j < v.size(); ++j) {
        out[v[j]] = (int32_t) j;
    }
}
