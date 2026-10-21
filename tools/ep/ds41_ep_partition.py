#!/usr/bin/env python3
"""ds41_ep_partition.py - a popularity-balanced expert partition for expert-parallel DeepSeek-V4.1-Flash (src/llama-ep.h).

For every MoE layer the 384 experts are split over the nodes: node k gets SIZES[k] of them, chosen so that the expected number
of routed pairs per node is as equal as the sizes allow (longest-processing-time greedy over the experts by popularity), and
each node's list is in descending popularity (its hot-first order: the hot/cold split of the engine cuts a prefix).

Popularity: Xolotl's rank.bin (int32 (layer, expert) pairs, hottest first, over an agentic corpus; only the order is known, not the
counts). The model files are permuted in place with ds41_order.json (old ids in the new order, tools/ds41_order.py), so expert j of
a layer in the file is old expert ORDER[L][j]; without --order the file's ids are the ids of rank.bin. The weight of the expert of
within-layer rank r is (r + R0)^-ALPHA (assumed Zipf-like profile; replace it with measured per-expert counts via --counts).

usage: ds41_ep_partition.py --nodes 3 --sizes 120,132,132 [--layers 40] [--experts 384] [--rank rank.bin] [--order ds41_order.json]
         [--alpha 0.85] [--r0 8] [--counts counts.json] [--slice G U D | --gguf FIRST.gguf]
         [--shard K=/path/on/node/shard.bin ...] [--seed N] -o out.part
--counts: {"L": [count of expert e of layer L (file ids), ...]} measured with GGML_CUDA_EXPERT_CACHE_STATS or similar; overrides the profile.
--slice defaults to the DeepSeek-V4.1-Flash Q2_K file (gate/up Q2_K 5120x2304, down Q3_K 2304x5120); --gguf reads them from the model.
Prints, per layer group and per node, the expert count and the expected share of the pairs (the balance), and the load imbalance.
"""
import argparse, json, random, struct, sys

def read_rank(path, n_layer, n_exp):
    d = open(path, 'rb').read()
    v = struct.unpack('<%di' % (len(d) // 4), d)
    pos = {}
    for i in range(0, len(v), 2):
        pos[(v[i], v[i + 1])] = i // 2
    return pos

def gguf_slices(path, n_exp):
    sys.path.insert(0, __file__.rsplit('/', 1)[0])
    import ds41_gguf
    t = ds41_gguf.read_split(path)
    out = []
    for kind in ('gate', 'up', 'down'):
        ti = t['blk.0.ffn_%s_exps.weight' % kind]
        assert ti['dims'][-1] == n_exp, (kind, ti['dims'])
        out.append(ti['nbytes'] // n_exp)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', type=int, required=True)
    ap.add_argument('--sizes', required=True)
    ap.add_argument('--layers', type=int, default=40)
    ap.add_argument('--experts', type=int, default=384)
    ap.add_argument('--rank')
    ap.add_argument('--order')
    ap.add_argument('--alpha', type=float, default=0.85)
    ap.add_argument('--r0', type=float, default=8.0)
    ap.add_argument('--counts')
    ap.add_argument('--slice', nargs=3, type=int, default=None)
    ap.add_argument('--gguf')
    ap.add_argument('--shard', action='append', default=[])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--balance', choices=['equal', 'size'], default='equal', help='equal expected pairs per node (default) or proportional to the node sizes')
    ap.add_argument('-o', required=True)
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(',')]
    assert len(sizes) == a.nodes and sum(sizes) == a.experts, 'sizes must give %d experts over %d nodes' % (a.experts, a.nodes)
    slice_b = a.slice or (gguf_slices(a.gguf, a.experts) if a.gguf else [3870720, 3870720, 5068800])
    order = json.load(open(a.order)) if a.order else None
    pos = read_rank(a.rank, a.layers, a.experts) if a.rank else None
    counts = json.load(open(a.counts)) if a.counts else None
    rng = random.Random(a.seed)
    shard_path = {}
    for s in a.shard:
        k, p = s.split('=', 1)
        shard_path[int(k)] = p
    ids = [format(rng.getrandbits(64), '016x') for _ in range(a.nodes)]
    lines = ['ds41-ep 1', 'nodes %d' % a.nodes, 'experts %d' % a.experts, 'layers %d' % a.layers,
             'slice %d %d %d' % tuple(slice_b)]
    for k in range(1, a.nodes):
        lines.append('shard %d %s %s' % (k, ids[k], shard_path.get(k, '/models/ep/ds41-ep-%d.shard' % k)))
    tot_load = [0.0] * a.nodes
    stat = []
    for L in range(a.layers):
        if counts:
            w = [float(c) for c in counts[str(L)]]
        else:
            if pos is not None:
                old = order[str(L)] if order else list(range(a.experts))
                p = [pos.get((L, old[j]), 10 ** 9) for j in range(a.experts)]
            else:
                p = list(range(a.experts))           # the file is already hot-first
            by_pos = sorted(range(a.experts), key=lambda j: (p[j], j))
            r = {j: i for i, j in enumerate(by_pos)}
            w = [(r[j] + a.r0) ** -a.alpha for j in range(a.experts)]
        s = sum(w)
        w = [x / s * 6.0 for x in w]                  # expected pairs per token
        todo = sorted(range(a.experts), key=lambda j: (-w[j], j))
        got = [[] for _ in range(a.nodes)]
        load = [0.0] * a.nodes
        for j in todo:
            best = None
            for k in range(a.nodes):
                if len(got[k]) >= sizes[k]:
                    continue
                key = ((load[k] + w[j]) / (sizes[k] if a.balance == 'size' else 1.0), k)
                if best is None or key < best[0]:
                    best = (key, k)
            k = best[1]
            got[k].append(j)
            load[k] += w[j]
        for k in range(a.nodes):
            lines.append('L %d %d %s' % (L, k, ' '.join(str(x) for x in got[k])))
            tot_load[k] += load[k]
        stat.append(load)
    open(a.o, 'w').write('\n'.join(lines) + '\n')
    mean = sum(tot_load) / a.nodes
    print('wrote %s: %d nodes, sizes %s, slice bytes %s' % (a.o, a.nodes, sizes, slice_b))
    print('expected share of the pairs per node (all layers): %s' % ' '.join('%.3f' % (x / sum(tot_load)) for x in tot_load))
    worst = max(max(l) / (sum(l) / a.nodes) for l in stat)
    print('worst layer: busiest node carries %.2fx the mean (all-layer: %.3fx)' % (worst, max(tot_load) / mean))
    for k in range(1, a.nodes):
        n = sum(sizes[k] for _ in range(a.layers))
        print('node %d shard file: %.1f GB (%d experts x %d layers x %.2f MB)' % (k, n * sum(slice_b) / 1e9, sizes[k], a.layers, sum(slice_b) / 1e6))
    print('shard ids: %s' % ' '.join('%d=%s' % (k, ids[k]) for k in range(1, a.nodes)))

main()
