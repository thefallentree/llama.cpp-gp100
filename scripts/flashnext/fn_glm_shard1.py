#!/usr/bin/env python3
"""A new metadata shard for a split GLM-5.3-Flash GGUF (glm5-next): the original shard 1 (metadata only) rewritten
with the MTP sidecar's NextN block (blk.N.* tensors; token_embd, output and output_norm come from the trunk) and the
engine's keys: block_count N+1, nextn_predict_layers 1, expert_hot_count H, the per-layer arrays extended to N+1
layers from the sidecar. Shard 2 (the trunk tensors) is untouched; split.tensors.count is updated.
usage: fn_glm_shard1.py SHARD1-00001-of-00002.gguf MTP.gguf OUT-00001-of-00002.gguf --hot H
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('shard1')
ap.add_argument('mtp')
ap.add_argument('out')
ap.add_argument('--hot', type=int, required=True, help='expert_hot_count: hot experts per layer (LLAMA_EXPERT_HOT overrides at load)')
args = ap.parse_args()

trunk = gguf.GGUFReader(args.shard1)
mtp = gguf.GGUFReader(args.mtp)
arch = trunk.fields[gguf.Keys.General.ARCHITECTURE].contents()
assert arch == mtp.fields[gguf.Keys.General.ARCHITECTURE].contents(), 'architectures differ'
n_layer = int(trunk.fields[f'{arch}.block_count'].contents())
assert int(mtp.fields[f'{arch}.block_count'].contents()) == n_layer + 1, 'the MTP file must describe n_layer + 1 blocks'

mtp_tensors = []
for t in mtp.tensors:
    if t.name in ('token_embd.weight', 'output.weight', 'output_norm.weight'):
        continue   # the trunk's embeddings, head and final norm serve the draft
    assert t.name.startswith(f'blk.{n_layer}.'), t.name
    mtp_tensors.append(t)

writer = gguf.GGUFWriter(args.out, arch=arch, endianess=trunk.endianess)
align = trunk.fields.get(gguf.Keys.General.ALIGNMENT)
if align is not None:
    writer.data_alignment = align.contents()

override = {
    f'{arch}.block_count': n_layer + 1,
    f'{arch}.nextn_predict_layers': 1,
    f'{arch}.expert_hot_count': args.hot,
}
# per-layer arrays: the sidecar's copies cover n_layer + 1 entries
from_mtp = []
for name, f2 in mtp.fields.items():
    if name.startswith(f'{arch}.') and f2.types and f2.types[0] == gguf.GGUFValueType.ARRAY and len(f2.data) == n_layer + 1:
        from_mtp.append(name)
done = set()
for field in trunk.fields.values():
    name = field.name
    if name == gguf.Keys.General.ARCHITECTURE or name.startswith('GGUF.'):
        continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    if name in override:
        writer.add_key_value(name, override[name], vt if name != f'{arch}.expert_hot_count' else gguf.GGUFValueType.UINT32)
    elif name in from_mtp:
        f2 = mtp.fields[name]
        writer.add_key_value(name, f2.contents(), f2.types[0], sub_type=f2.types[-1])
    elif name == 'split.tensors.count':
        writer.add_key_value(name, int(field.contents()) + len(mtp_tensors), vt)
    else:
        writer.add_key_value(name, field.contents(), vt, sub_type=st)
    done.add(name)
for name in [f'{arch}.nextn_predict_layers'] + from_mtp:
    if name not in done and name in mtp.fields:
        f2 = mtp.fields[name]
        writer.add_key_value(name, f2.contents(), f2.types[0], sub_type=f2.types[-1])
        done.add(name)
if f'{arch}.expert_hot_count' not in done:
    writer.add_uint32(f'{arch}.expert_hot_count', args.hot)

for t in trunk.tensors:   # none in a metadata-only shard, kept for generality
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
for t in mtp_tensors:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in trunk.tensors:
    writer.write_tensor_data(t.data, tensor_endianess=trunk.endianess)
for t in mtp_tensors:
    writer.write_tensor_data(t.data, tensor_endianess=mtp.endianess)
writer.close()
print(f'written {args.out}: {len(trunk.tensors)} trunk + {len(mtp_tensors)} MTP tensors, block_count {n_layer + 1}, '
      f'expert_hot_count {args.hot}, per-layer arrays from the sidecar: {from_mtp}')
