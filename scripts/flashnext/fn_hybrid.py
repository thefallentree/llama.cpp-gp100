#!/usr/bin/env python3
"""Reorder qwen4exp routed experts hot-first per layer and split them into a hot and a cold tensor.

Experts are ranked per layer by routing counts from an imatrix GGUF. Expert tensors are permuted along the expert
axis and the router rows (ffn_gate_inp) with the same permutation, so routing is unchanged. The first H experts stay
in blk.N.ffn_{gate,up,down}_exps (meant for VRAM), the rest go to blk.N.ffn_{gate,up,down}_exps_cold (meant for
pinned host memory). The hot count is stored as <arch>.expert_hot_count.

usage: fn_hybrid.py IN-00001-of-0000S.gguf OUT-00001-of-0000S.gguf IMATRIX.gguf H [--dry-run]
"""
import argparse
import re
import sys

import numpy as np

sys.path.insert(0, '/home/sunyc/models/llama.cpp-flashnext/gguf-py')
import gguf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('inp')
ap.add_argument('out')
ap.add_argument('imatrix')
ap.add_argument('h', type=int)
ap.add_argument('--dry-run', action='store_true')
args = ap.parse_args()

counts = {}
for t in gguf.GGUFReader(args.imatrix).tensors:
    m = re.match(r'blk\.(\d+)\.ffn_gate_exps\.weight\.counts', t.name)
    if m:
        counts[int(m.group(1))] = np.asarray(t.data, dtype=np.float64).ravel()

perm = {}
cover = []
for l, c in counts.items():
    perm[l] = np.argsort(-c, kind='stable')
    cover.append(c[perm[l][:args.h]].sum() / max(c.sum(), 1))
print(f'layers {len(perm)}, hot {args.h}/{len(next(iter(counts.values())))}, routed coverage mean {np.mean(cover)*100:.2f}% min {np.min(cover)*100:.2f}%')
if args.dry_run:
    sys.exit(0)

reader = gguf.GGUFReader(args.inp)
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
writer = gguf.GGUFWriter(args.out, arch=arch, endianess=reader.endianess)
align = reader.fields.get(gguf.Keys.General.ALIGNMENT)
if align is not None:
    writer.data_alignment = align.contents()
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if field.name == 'split.tensors.count':
        val = val + 3*len(perm)
    writer.add_key_value(field.name, val, vt, sub_type=st)
writer.add_uint32(f'{arch}.expert_hot_count', args.h)

order = []   # (name, array, qtype)
for t in reader.tensors:
    m = re.match(r'blk\.(\d+)\.(ffn_(gate|up|down)_exps|ffn_gate_inp)\.weight$', t.name)
    if m and int(m.group(1)) in perm:
        p = perm[int(m.group(1))]
        a = np.asarray(t.data)
        assert a.shape[0] == len(p), (t.name, a.shape)
        a = a[p]
        if m.group(2) == 'ffn_gate_inp':
            order.append((t.name, np.ascontiguousarray(a), t.tensor_type))
        else:
            order.append((t.name, np.ascontiguousarray(a[:args.h]), t.tensor_type))
            order.append((t.name.replace('_exps.weight', '_exps_cold.weight'), np.ascontiguousarray(a[args.h:]), t.tensor_type))
    else:
        order.append((t.name, t.data, t.tensor_type))

for name, arr, qt in order:
    writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, qt)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for name, arr, qt in order:
    writer.write_tensor_data(arr, tensor_endianess=reader.endianess)
writer.close()
print('written', args.out)
