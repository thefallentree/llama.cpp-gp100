#!/usr/bin/env python3
"""The hot-first expert order per layer from an imatrix GGUF (llama-imatrix -o x.gguf): the per-expert routing
counts of blk.L.ffn_gate_exps.weight.counts, sorted descending. Writes ORDER.json for fn_permute_inplace.py
({"L": [old expert ids, hot first]}) and prints the coverage of the top-K experts (what a hot set of K catches).
usage: fn_glm_order.py IMATRIX.gguf ORDER.json [--ks 32,48,64,96,144]
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import numpy as np  # noqa: E402
import gguf  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith('--')]
ks = [32, 48, 64, 96, 144]
for a in sys.argv[1:]:
    if a.startswith('--ks='):
        ks = [int(x) for x in a[5:].split(',')]
if len(args) != 2:
    print(__doc__)
    sys.exit(2)
counts = {}
for t in gguf.GGUFReader(args[0]).tensors:
    m = re.match(r'blk\.(\d+)\.ffn_gate_exps\.weight\.counts', t.name)
    if m:
        counts[int(m.group(1))] = np.asarray(t.data, dtype=np.float64).ravel()
if not counts:
    raise SystemExit('no per-expert counts in the imatrix (needs an imatrix of a MoE model written as GGUF)')
order = {}
cov = []
for il in sorted(counts):
    c = counts[il]
    idx = np.argsort(-c, kind='stable')
    order[str(il)] = [int(i) for i in idx]
    cs = c[idx]
    tot = cs.sum() if cs.sum() > 0 else 1.0
    cov.append([cs[:k].sum() / tot for k in ks])
cov = np.array(cov)
with open(args[1], 'w') as f:
    json.dump(order, f)
n = len(next(iter(counts.values())))
print(f'{len(order)} layers, {n} experts each; order written to {args[1]}')
print('routing coverage of the top-K experts (mean over layers; min layer):')
for i, k in enumerate(ks):
    print(f'  top {k:4d} of {n}: {cov[:, i].mean() * 100:5.1f}%  (min {cov[:, i].min() * 100:5.1f}%)')
