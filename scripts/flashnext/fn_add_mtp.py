#!/usr/bin/env python3
"""Merge the qwen4exp MTP block (from an MTP-only GGUF) into shard 1 of a trunk GGUF, so that
`--spec-type draft-mtp` drafts with the target model itself (no -md): the draft shares the trunk's token embeddings,
its experts can live in VRAM with the trunk's, and no second model is loaded.

The MTP file's LM head is kept as blk.N.nextn.shared_head_head (a smaller head for drafting); its token embeddings
are dropped (the trunk's are used). block_count grows by one, nextn_predict_layers is set, and the per-layer arrays
(attention.recurrent_layers, attention.compress_ratios) are taken from the MTP file, which already covers n+1 layers.

usage: fn_add_mtp.py TRUNK-00001-of-0000S.gguf MTP.gguf OUT-00001-of-0000S.gguf [--no-head]
"""
import argparse
import sys

sys.path.insert(0, '/home/sunyc/models/llama.cpp-p100stack/gguf-py')
import gguf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('trunk')
ap.add_argument('mtp')
ap.add_argument('out')
ap.add_argument('--no-head', action='store_true', help='draft with the trunk LM head instead of the MTP file\'s')
args = ap.parse_args()

trunk = gguf.GGUFReader(args.trunk)
mtp = gguf.GGUFReader(args.mtp)
arch = trunk.fields[gguf.Keys.General.ARCHITECTURE].contents()
n_layer = int(trunk.fields[f'{arch}.block_count'].contents())
assert int(mtp.fields[f'{arch}.block_count'].contents()) == n_layer + 1, 'the MTP file must describe n_layer + 1 blocks'

mtp_tensors = []
for t in mtp.tensors:
    if t.name == 'token_embd.weight':
        continue
    if t.name == 'output.weight':
        if args.no_head:
            continue
        mtp_tensors.append((f'blk.{n_layer}.nextn.shared_head_head.weight', t))
        continue
    assert t.name.startswith(f'blk.{n_layer}.'), t.name
    mtp_tensors.append((t.name, t))

writer = gguf.GGUFWriter(args.out, arch=arch, endianess=trunk.endianess)
align = trunk.fields.get(gguf.Keys.General.ALIGNMENT)
if align is not None:
    writer.data_alignment = align.contents()

override = {
    f'{arch}.block_count': n_layer + 1,
    f'{arch}.nextn_predict_layers': 1,
}
from_mtp = [f'{arch}.attention.recurrent_layers', f'{arch}.attention.compress_ratios']
done = set()
for field in trunk.fields.values():
    name = field.name
    if name == gguf.Keys.General.ARCHITECTURE or name.startswith('GGUF.'):
        continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    if name in override:
        writer.add_key_value(name, override[name], vt)
    elif name in from_mtp and name in mtp.fields:
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

for t in trunk.tensors:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
for name, t in mtp_tensors:
    writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in trunk.tensors:
    writer.write_tensor_data(t.data, tensor_endianess=trunk.endianess)
for _, t in mtp_tensors:
    writer.write_tensor_data(t.data, tensor_endianess=mtp.endianess)
writer.close()
print(f'written {args.out}: {len(trunk.tensors)} trunk + {len(mtp_tensors)} MTP tensors, block_count {n_layer + 1}')
