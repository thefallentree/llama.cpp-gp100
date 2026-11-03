#!/usr/bin/env python3
r"""Rewrite one GGUF shard with selected non-expert weights re-encoded (default: dense -> Q8_0, router -> F32).

Routed experts, embeddings and norms are copied byte for byte. Q8_0 has a near-bandwidth width-1 CUDA mat-vec
on sm_60 and is close to lossless as a re-encoding target; K-quant/IQ/BF16 mat-vecs are issue-bound there.

usage: fn_requant.py IN.gguf OUT.gguf [--dry-run] [--dense q8_0|q4_0|f16] [--router f32|q8_0|keep] [--lowbit q8_0|g64]
                     [--keep TYPE,TYPE,...] [--skip REGEX] [--only REGEX]
--lowbit g64 re-encodes dense weights whose source is <= 4.5 bpw as Q4_1_G64 (sm_60 HFMA2 path) instead of Q8_0.
--keep copies dense weights of these source types unchanged (e.g. iq4_xs,iq4_nl,q2_0: their sm_60 mat-vecs
beat Q8_0 at widths 1-4, so re-encoding them only costs bandwidth).
--dense q4_0 (with --only) re-encodes the named dense weights as Q4_0 for the planar P4 mat-vec (half the bytes of
P8 at the same speed per byte); the mat-vecs that one launch computes (attn q+k+v, qkv+gate+alpha+beta) must share
a width, so convert them together, e.g.
  --dense q4_0 --router keep --only 'attn_(qkv|gate|q|k|v|output)\.weight|ssm_(out|alpha|beta)\.weight'
"""
import argparse
import re
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))  # this repo's gguf-py (Q2_0, Q4_16)
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as T  # noqa: E402

LOWBIT = {T.Q2_0, T.Q2_K, T.Q3_K, T.IQ4_XS, T.IQ4_NL, T.Q4_K, T.Q4_0}
CONVERTIBLE = {T.BF16, T.F16, T.Q2_0, T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K, T.IQ4_XS, T.IQ4_NL, T.Q4_0, T.Q5_0, T.Q8_0}


def deq_q2_0(raw: np.ndarray, ne0: int) -> np.ndarray:
    blocks = raw.reshape(-1, 18)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    codes = (blocks[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
    return ((codes.reshape(-1, 64).astype(np.float32) - 1.0) * d).reshape(-1, ne0)


def dequant(t) -> np.ndarray:
    ne0 = int(t.shape[0])
    raw = np.asarray(t.data)
    if t.tensor_type == T.F32:
        return raw.reshape(-1, ne0).astype(np.float32)
    raw = raw.view(np.uint8).reshape(-1, raw.view(np.uint8).size // int(np.prod(t.shape[1:])) if len(t.shape) > 1 else raw.size)
    if t.tensor_type == T.Q2_0:
        return deq_q2_0(raw, ne0)
    return gguf.quants.dequantize(raw, t.tensor_type).reshape(-1, ne0)


def quant_g64(f: np.ndarray) -> np.ndarray:
    rows, n = f.shape
    b = f.reshape(rows, n // 64, 64)
    vmin = b.min(-1)
    d = (b.max(-1) - vmin) / 15.0
    d[d == 0] = 1.0
    q = np.clip(np.round((b - vmin[..., None]) / d[..., None]), 0, 15).astype(np.uint8)
    out = np.empty((rows, n // 64, 36), dtype=np.uint8)
    out[..., 0:2] = d.astype(np.float16).view(np.uint8).reshape(rows, n // 64, 2)
    out[..., 2:4] = vmin.astype(np.float16).view(np.uint8).reshape(rows, n // 64, 2)
    for h in range(2):
        out[..., 4 + 16*h:20 + 16*h] = q[..., 32*h:32*h + 16] | (q[..., 32*h + 16:32*h + 32] << 4)
    return out.reshape(rows, -1)


def plan(t, dense: T, router: str, lowbit: str = 'q8_0', keep=frozenset()):
    n = t.name
    if t.tensor_type in keep and not n.endswith('ffn_gate_inp.weight'):
        return None
    if '_exps' in n or n in ('token_embd.weight', 'per_layer_token_embd.weight'):
        return None
    if len(t.shape) < 2 or int(t.shape[1]) < 8 or 'norm' in n:
        return None
    if n.endswith('ffn_gate_inp.weight'):
        if router == 'keep':
            return None
        return T.F32 if router == 'f32' else T.Q8_0
    if lowbit == 'g64' and t.tensor_type in LOWBIT and int(t.shape[0]) % 64 == 0 and int(t.shape[1]) % 8 == 0 \
            and t.name != 'output.weight':
        return T.Q4_1_G64
    if t.tensor_type in CONVERTIBLE and int(t.shape[0]) % 32 == 0:
        return dense
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('inp')
    ap.add_argument('out')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--dense', default='q8_0', choices=['q8_0', 'q4_0', 'f16'])
    ap.add_argument('--router', default='f32', choices=['f32', 'q8_0', 'keep'])
    ap.add_argument('--lowbit', default='q8_0', choices=['q8_0', 'g64'])
    ap.add_argument('--keep', default='', help='comma-separated source types to copy unchanged')
    ap.add_argument('--skip', default='', help='regex of tensor names to copy unchanged (e.g. shared_head_head)')
    ap.add_argument('--only', default='', help='regex: convert only the tensors whose name matches')
    ap.add_argument('--draft-vocab', default='', help='token id list: the MTP head (blk.N.nextn.shared_head_head) keeps these rows, a d2t tensor maps them back')
    args = ap.parse_args()
    keep = frozenset(T[k.strip().upper()] for k in args.keep.split(',') if k.strip())
    skip = re.compile(args.skip) if args.skip else None
    only = re.compile(args.only) if args.only else None
    dense = {'q8_0': T.Q8_0, 'q4_0': T.Q4_0, 'f16': T.F16}[args.dense]

    reader = gguf.GGUFReader(args.inp)
    conv = {}
    summary = {}
    for t in reader.tensors:
        nt = plan(t, dense, args.router, args.lowbit, keep)
        if skip is not None and skip.search(t.name):
            nt = None
        if only is not None and not only.search(t.name):
            nt = None
        if nt is not None and nt != t.tensor_type:
            conv[t.name] = nt
            key = (re.sub(r'blk\.\d+\.', 'blk.N.', t.name), t.tensor_type.name, nt.name)
            summary[key] = summary.get(key, 0) + t.n_bytes
    for (k, a, b), nb in sorted(summary.items(), key=lambda x: -x[1]):
        print(f'{nb/1e6:9.1f} MB  {a:7s} -> {b:5s}  {k}')
    print(f'converting {len(conv)} tensors, {sum(summary.values())/1e9:.2f} GB source')
    if args.dry_run:
        return

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
        if field.name == 'split.tensors.count' and args.draft_vocab:
            val = val + 1   # the d2t tensor
        writer.add_key_value(field.name, val, vt, sub_type=st)

    draft_ids = None
    if args.draft_vocab:
        ids = sorted({int(l.split()[0]) for l in open(args.draft_vocab) if l.strip() and not l.startswith('#')})
        draft_ids = np.array(ids, dtype=np.int64)

    data = {}
    for t in reader.tensors:
        if draft_ids is not None and t.name.endswith('nextn.shared_head_head.weight'):
            raw = np.asarray(t.data)
            n_vocab = int(t.shape[1])
            assert raw.shape[0] == n_vocab, (raw.shape, t.shape)
            keep = draft_ids[draft_ids < n_vocab]
            # the control tokens past the list (EOS, im_end, ...) stay draftable
            keep = np.unique(np.concatenate([keep, np.arange(max(0, n_vocab - 320), n_vocab, dtype=np.int64)]))
            arr = np.ascontiguousarray(raw[keep])
            data[t.name] = (arr, t.tensor_type)
            data['d2t'] = (keep, T.I64)
            writer.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, t.tensor_type)
            writer.add_tensor_info('d2t', keep.shape, keep.dtype, keep.nbytes, T.I64)
            print(f'draft head: {len(keep)} of {n_vocab} rows kept ({arr.nbytes/1e6:.1f} MB), d2t added')
            continue
        if t.name in conv:
            f = dequant(t)
            nt = conv[t.name]
            if nt == T.F32:
                data[t.name] = (f.astype(np.float32), nt)
            elif nt == T.F16:
                data[t.name] = (f.astype(np.float16), nt)
            elif nt == T.Q4_1_G64:
                data[t.name] = (quant_g64(f), nt)
            else:
                data[t.name] = (gguf.quants.quantize(f, nt), nt)
            arr, _ = data[t.name]
            writer.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, nt)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        if t.name in data:
            writer.write_tensor_data(data[t.name][0])
            if t.name.endswith('nextn.shared_head_head.weight') and 'd2t' in data:
                writer.write_tensor_data(data['d2t'][0])   # declared right after the head
        else:
            writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    print('written', args.out)


if __name__ == '__main__':
    main()
