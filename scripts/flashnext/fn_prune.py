#!/usr/bin/env python3
"""Remove routed experts from the trunk of a qwen4exp hybrid GGUF (hot/cold split, MTP block merged) by a published mask.

The mask gives, per trunk layer, which experts of the unpruned model stay (ISTA-DASLab's GSQ-RCO Coder release
publishes its 256-of-512 selection in tensor-allocation/*.rco-allocation.txt; its router rows are the original
router's rows under the mask, so the mask alone reproduces the pruned model on any quantization of the same base).
The hybrid file holds a layer's experts hot first in the order of the imatrix routing counts (scripts/fn_hybrid.py),
so the mask is mapped through that permutation; the kept experts go to one tensor per weight (all resident, no cold
tensor), the router keeps their rows, expert_count becomes the kept count, expert_hot_count goes. The MTP block
(blk.n_layer) is not pruned: nextn.expert_count keeps its expert count for the loader.

usage: fn_prune.py IN-00001-of-0000S.gguf OUT-00001-of-0000S.gguf MASK.npy IMATRIX.gguf [--stock STOCK.gguf]
MASK.npy: bool [n_layer, n_expert] in the original expert order. --stock: the unpruned file whose router rows are
compared with the permuted ones (a check of the permutation).
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, '/home/sunyc/models/llama.cpp-p100stack/gguf-py')
import gguf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('inp')
ap.add_argument('out')
ap.add_argument('mask')
ap.add_argument('imatrix')
ap.add_argument('--stock')
args = ap.parse_args()

mask = np.load(args.mask)
n_layer_mask, n_expert_all = mask.shape
keep_n = int(mask[0].sum())
assert all(int(m.sum()) == keep_n for m in mask), 'the GGUF format needs one expert count for all layers'

counts = {}
for t in gguf.GGUFReader(args.imatrix).tensors:
    m = re.match(r'blk\.(\d+)\.ffn_gate_exps\.weight\.counts', t.name)
    if m:
        counts[int(m.group(1))] = np.asarray(t.data, dtype=np.float64).ravel()
perm = {l: np.argsort(-c, kind='stable') for l, c in counts.items()}   # position -> original expert
assert len(perm) == n_layer_mask and all(len(p) == n_expert_all for p in perm.values()), (len(perm), n_layer_mask)

reader = gguf.GGUFReader(args.inp)
tens = {t.name: t for t in reader.tensors}
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
n_layer = int(reader.fields[f'{arch}.block_count'].contents())
n_nextn = int(reader.fields[f'{arch}.nextn_predict_layers'].contents()) if f'{arch}.nextn_predict_layers' in reader.fields else 0
n_trunk = n_layer - n_nextn
n_hot = int(reader.fields[f'{arch}.expert_hot_count'].contents()) if f'{arch}.expert_hot_count' in reader.fields else n_expert_all
assert n_trunk == n_layer_mask, (n_trunk, n_layer_mask)
assert int(reader.fields[f'{arch}.expert_count'].contents()) == n_expert_all

if args.stock:
    stock = {t.name: t for t in gguf.GGUFReader(args.stock).tensors}
    for l in range(n_trunk):
        s = stock[f'blk.{l}.ffn_gate_inp.weight']
        s = np.asarray(s.data)
        if s.dtype != np.float32:  # BF16 bits -> F32 (fn_requant stores the router as F32 this way)
            s = (s.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        s = s.reshape(n_expert_all, -1)
        h = np.asarray(tens[f'blk.{l}.ffn_gate_inp.weight'].data).reshape(n_expert_all, -1)
        assert np.array_equal(h, s[perm[l]]), f'layer {l}: the file is not permuted by the imatrix counts'
    print('permutation check against the stock router: ok')

# position lists of the kept experts (hot-first order kept)
keep = {}
for l in range(n_trunk):
    keep[l] = np.array([p for p in range(n_expert_all) if mask[l][perm[l][p]]])
    assert len(keep[l]) == keep_n
print(f'layers {n_trunk}, experts {n_expert_all} -> {keep_n}, hot {n_hot} in the file; kept among the hot: '
      f'mean {np.mean([(keep[l] < n_hot).sum() for l in keep]):.1f}')

writer = gguf.GGUFWriter(args.out, arch=arch, endianess=reader.endianess)
align = reader.fields.get(gguf.Keys.General.ALIGNMENT)
if align is not None:
    writer.data_alignment = align.contents()
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.') or field.name == f'{arch}.expert_hot_count':
        continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if field.name == 'split.tensors.count':
        val = val - 3*n_trunk
    if field.name == f'{arch}.expert_count':
        val = keep_n
    writer.add_key_value(field.name, val, vt, sub_type=st)
if n_nextn > 0:
    writer.add_uint32(f'{arch}.nextn.expert_count', n_expert_all)

order = []   # (name, array or callable, qtype)
for t in reader.tensors:
    m = re.match(r'blk\.(\d+)\.(ffn_(gate|up|down)_exps|ffn_gate_inp)\.weight$', t.name)
    mc = re.match(r'blk\.(\d+)\.ffn_(gate|up|down)_exps_cold\.weight$', t.name)
    if mc and int(mc.group(1)) < n_trunk:
        continue
    if m and int(m.group(1)) < n_trunk:
        l = int(m.group(1))
        if m.group(2) == 'ffn_gate_inp':
            a = np.asarray(t.data)
            assert a.shape[0] == n_expert_all
            order.append((t.name, np.ascontiguousarray(a[keep[l]]), t.tensor_type))
        else:
            cold = tens[t.name.replace('_exps.weight', '_exps_cold.weight')]
            assert cold.tensor_type == t.tensor_type
            def gather(t=t, cold=cold, l=l):
                a = np.concatenate([np.asarray(t.data), np.asarray(cold.data)], axis=0)
                assert a.shape[0] == n_expert_all, (t.name, a.shape)
                return np.ascontiguousarray(a[keep[l]])
            shape = (keep_n,) + tuple(np.asarray(t.data).shape[1:])
            order.append((t.name, (gather, shape, np.asarray(t.data).dtype), t.tensor_type))
    else:
        order.append((t.name, t.data, t.tensor_type))

for name, arr, qt in order:
    if isinstance(arr, tuple):
        fn, shape, dtype = arr
        writer.add_tensor_info(name, shape, dtype, int(np.prod(shape))*np.dtype(dtype).itemsize, qt)
    else:
        writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, qt)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
fd_in = os.open(args.inp, os.O_RDONLY)
for i, (name, arr, qt) in enumerate(order):
    if isinstance(arr, tuple):
        arr = arr[0]()
    writer.write_tensor_data(arr, tensor_endianess=reader.endianess)
    if i % 16 == 0:
        # keep the page cache for the serving model: drop what this pass has read and written
        writer.fout[0].flush()
        os.posix_fadvise(writer.fout[0].fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(fd_in, 0, 0, os.POSIX_FADV_DONTNEED)
writer.close()
os.close(fd_in)
print('written', args.out)
