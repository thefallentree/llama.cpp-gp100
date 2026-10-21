#pragma once

// Expert parallelism (DeepSeek-V4.1-Flash): the routed experts of every MoE layer are split over the nodes of a cluster.
// A partition file says which expert ids each node owns, in the order of the node's shard (hot first). Node 0 is the head
// (this process): its experts are gathered from the model file into the layer's normal expert tensors. Node k > 0 holds
// its experts in tensors named blk.N.ffn_{gate,up,down}_exps.epK.weight, placed on that node's devices with -ot; their
// bytes come from the node's shard file (written by scripts/ds41_ep_shard.py), which the node's rpc-server reads from
// its own disk (RPC_CMD_LOAD_FILE), so no weights travel over the network.
//
// Partition file (text):
//   ds41-ep 1
//   nodes K
//   experts N
//   layers L                 (the layers the file covers: 0 .. L-1)
//   slice GATE UP DOWN       (bytes of one expert of each matrix; checked against the model)
//   shard K ID PATH          (K >= 1: the shard file's u64 id in hex, and its path on node K)
//   L IL K e0 e1 e2 ...      (layer IL, node K: its expert ids, in shard order)
// Shard file: 4096-byte header "DS41EPSH" + u64 id (LE) + zeros; then for each layer 0 .. L-1 and gate, up, down: the node's
// experts of the matrix (n_k * slice bytes), each tensor starting on a 4096 boundary.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum llama_ep_kind { LLAMA_EP_GATE = 0, LLAMA_EP_UP = 1, LLAMA_EP_DOWN = 2 };

struct llama_ep_partition {
    int n_nodes  = 0;
    int n_expert = 0;
    int n_layer  = 0;
    uint64_t slice[3] = { 0, 0, 0 };

    std::vector<std::string> shard_path;  // [node]
    std::vector<uint64_t>    shard_id;    // [node]
    std::vector<std::vector<std::vector<int32_t>>> ids; // [layer][node]: expert ids in shard order

    static std::shared_ptr<llama_ep_partition> load(const std::string & path); // throws std::runtime_error

    int n_local(int il, int node) const { return (int) ids[il][node].size(); }

    // offset of the (layer, kind) tensor of a node in its shard file
    uint64_t shard_offset(int node, int il, int kind) const;
    uint64_t shard_size(int node) const; // the file's size

    // table[e] = slot of expert e in node's shard of layer il, or -1; n_expert entries
    void table(int il, int node, int32_t * out) const;
};
