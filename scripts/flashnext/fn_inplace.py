#!/usr/bin/env python3
"""Re-encode selected tensors of a GGUF shard in place: no second copy of a 40 GB file.

The data section is compacted in file order (every tensor's new offset is at or before its old one, so a tensor
is read whole and written before the next is touched), the header's tensor types and offsets are rewritten in
place (their fields are fixed-size), the file is truncated. A backup directory keeps the header and the original
bytes of the converted tensors, and --restore puts the file back byte for byte (the unconverted tensors are moved
back in reverse order). The untouched tensors are verified by hash after the conversion.

usage: fn_inplace.py FILE --to q4_0 --only REGEX --backup DIR [--dry-run]
       fn_inplace.py FILE --to q4_16 --from-release RELEASE.gguf --only REGEX --backup DIR [--dry-run]
       fn_inplace.py FILE --restore DIR
--to q4_16 --from-release: the matched tensors are re-encoded EXACTLY from the release's own Q3_K or Q2_0 tensors of
the same name (a fp16 scale per 16 weights takes Q3_K's sub-block scales and Q2_0's block scale as they are; the
tensor in FILE is whatever re-encoding of them, e.g. Q8_0); tensors of other types in the release are left alone.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, '/home/sunyc/models/llama.cpp-p100stack/gguf-py')
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as T  # noqa: E402

CONVERTIBLE = {T.BF16, T.F16, T.F32, T.Q8_0, T.Q4_0, T.Q5_0, T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K, T.IQ4_XS, T.IQ4_NL}


def pad(n: int, align: int) -> int:
    return (n + align - 1) // align * align


def file_offset(reader, arr) -> int:
    """the file offset of a memmap slice of the reader"""
    base = reader.data.__array_interface__['data'][0]
    return int(arr.__array_interface__['data'][0]) - base


def scan(path: str):
    """the tensors in file order with their header field positions; the reader is released before returning"""
    reader = gguf.GGUFReader(path)
    align = int(reader.alignment)
    data_offset = int(reader.data_offset)
    tensors = []
    for t in reader.tensors:
        f = t.field
        # parts: name length, name, n_dims, dims, type, offset
        type_pos = file_offset(reader, f.parts[-2])
        off_pos = file_offset(reader, f.parts[-1])
        assert f.parts[-2].dtype == np.uint32 and f.parts[-1].dtype == np.uint64, (t.name, f.parts[-2].dtype, f.parts[-1].dtype)
        rel = int(f.parts[-1][0])
        assert data_offset + rel == int(t.data_offset), (t.name, rel, t.data_offset)
        tensors.append({
            'name': t.name, 'type': int(t.tensor_type), 'shape': [int(x) for x in t.shape],
            'n_elements': int(t.n_elements), 'nbytes': int(t.n_bytes), 'rel': rel, 'type_pos': type_pos, 'off_pos': off_pos,
        })
    size = os.path.getsize(path)
    del reader
    tensors.sort(key=lambda x: x['rel'])
    # contiguous, as gguf.cpp requires
    run = 0
    for t in tensors:
        assert t['rel'] == run, (t['name'], t['rel'], run)
        run += pad(t['nbytes'], align)
    assert data_offset + run == size, (data_offset + run, size)
    return tensors, align, data_offset, size


def row_bytes(qtype, ne0: int) -> int:
    bs, ts = gguf.GGML_QUANT_SIZES[qtype]
    return ne0 // bs * ts


def dequant(raw: bytes, qtype, shape) -> np.ndarray:
    ne0 = shape[0]
    a = np.frombuffer(raw, dtype=np.uint8)
    if qtype == T.F32:
        return a.view(np.float32).reshape(-1, ne0)
    if qtype == T.F16:
        return a.view(np.float16).astype(np.float32).reshape(-1, ne0)
    if qtype == T.BF16:
        return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1, ne0)
    if qtype == T.Q2_0:  # this fork's type: 64 weights (code - 1)*d, gguf-py has no dequantizer for it
        blocks = a.reshape(-1, 18)
        d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        codes = ((blocks[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1, 64).astype(np.float32)
        return ((codes - 1.0) * d).reshape(-1, ne0)
    return gguf.quants.dequantize(a.reshape(-1, row_bytes(qtype, ne0)), qtype).reshape(-1, ne0)


def quant(f: np.ndarray, qtype) -> bytes:
    if qtype == T.F16:
        return f.astype(np.float16).tobytes()
    if qtype == T.F32:
        return f.astype(np.float32).tobytes()
    return np.ascontiguousarray(gguf.quants.quantize(f, qtype)).tobytes()


def blake(b) -> str:
    return hashlib.blake2b(b, digest_size=16).hexdigest()


# the exact re-encodings into Q4_16 (scales per 16 as fp16, codes 0..15 = value/scale + 8)
QK_K = 256


def q3_k_to_q4_16(raw: bytes, ne0: int) -> bytes:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 110)
    n_blocks = blocks.shape[0]
    hmask, rest = np.hsplit(blocks, [QK_K // 8])
    qs, rest = np.hsplit(rest, [QK_K // 4])
    scales, d = np.hsplit(rest, [12])
    d = d.view(np.float16).astype(np.float32)
    lscales, hscales = np.hsplit(scales, [8])
    lscales = (lscales.reshape((n_blocks, 1, 8)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))).reshape((n_blocks, 16))
    hscales = (hscales.reshape((n_blocks, 1, 4)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))).reshape((n_blocks, 16))
    sc = ((lscales & np.uint8(0x0F)) | ((hscales & np.uint8(0x03)) << np.uint8(4))).astype(np.int8) - np.int8(32)
    dl = d * sc.astype(np.float32)                                            # [n_blocks, 16]: the scale of each sub-block of 16
    ql = (qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))).reshape((n_blocks, 16, 16)) & np.uint8(3)
    qh = (hmask.reshape(n_blocks, -1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape((1, 1, 8, 1))).reshape((n_blocks, 16, 16)) & np.uint8(1)
    q = ql.astype(np.int16) - ((qh ^ np.uint8(1)).astype(np.int16) << 2)   # -4..3
    codes = (q + 8).reshape(-1, 16)
    assert codes.min() >= 0 and codes.max() <= 15
    out = np.concatenate([dl.reshape(-1, 1).astype(np.float16).view(np.uint8), gguf.quants.Q4_16.pack_codes(codes)], axis=-1)
    assert out.shape == (n_blocks * 16, 10)
    return np.ascontiguousarray(out).tobytes()


def q2_0_to_q4_16(raw: bytes, ne0: int) -> bytes:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    n_blocks = blocks.shape[0]
    d = blocks[:, :2].copy().view(np.float16)                                 # [n_blocks, 1], exact
    codes2 = ((blocks[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(n_blocks, 64)
    codes = (codes2.astype(np.int16) + 7).reshape(-1, 16)                    # (code - 1) + 8
    dl = np.repeat(d, 4, axis=1).reshape(-1, 1)                               # the block's scale for its 4 sub-blocks
    out = np.concatenate([dl.view(np.uint8), gguf.quants.Q4_16.pack_codes(codes)], axis=-1)
    assert out.shape == (n_blocks * 4, 10)
    return np.ascontiguousarray(out).tobytes()


EXACT_SOURCES = {T.Q3_K: q3_k_to_q4_16, T.Q2_0: q2_0_to_q4_16}


def convert(args) -> None:
    only = re.compile(args.only)
    to = T[args.to.upper()]
    tensors, align, data_offset, size = scan(args.file)
    release = None
    if args.from_release:
        assert to == T.Q4_16, '--from-release is the exact re-encoding into q4_16'
        rr = gguf.GGUFReader(args.from_release)
        release = {t.name: t for t in rr.tensors}
    plan = []
    run = 0
    total_src = 0
    for t in tensors:
        conv = only.search(t['name']) is not None and T(t['type']) in CONVERTIBLE and T(t['type']) != to \
            and t['shape'][0] % gguf.GGML_QUANT_SIZES[to][0] == 0 and len(t['shape']) >= 2
        if conv and release is not None:
            src = release.get(t['name'])
            conv = src is not None and src.tensor_type in EXACT_SOURCES and [int(x) for x in src.shape] == t['shape']
            if only.search(t['name']) is not None and not conv:
                print(f"  skip {t['name']}: release type {src.tensor_type.name if src is not None else 'absent'}")
        nbytes_new = t['n_elements'] // gguf.GGML_QUANT_SIZES[to][0] * gguf.GGML_QUANT_SIZES[to][1] if conv else t['nbytes']
        assert nbytes_new <= t['nbytes'], (t['name'], 'grows')
        plan.append({**t, 'convert': conv, 'type_new': int(to) if conv else t['type'], 'nbytes_new': nbytes_new, 'rel_new': run})
        run += pad(nbytes_new, align)
        if conv:
            total_src += t['nbytes']
    n_conv = sum(p['convert'] for p in plan)
    print(f'{n_conv} tensors to {to.name}, {total_src/1e9:.2f} GB -> file {size/1e9:.2f} -> {(data_offset + run)/1e9:.2f} GB')
    for p in plan:
        if p['convert'] and ('blk.0.' in p['name'] or 'blk.' not in p['name']):
            print(f"  {p['name']:40s} {T(p['type']).name:6s} -> {to.name} {p['nbytes']/1e6:8.1f} -> {p['nbytes_new']/1e6:8.1f} MB")
    if args.dry_run or n_conv == 0:
        return
    os.makedirs(args.backup, exist_ok=False)
    fd = os.open(args.file, os.O_RDWR)
    # the backup: the header and the converted tensors' bytes, and the hashes of all tensors
    with open(os.path.join(args.backup, 'header.bin'), 'wb') as fh:
        fh.write(os.pread(fd, data_offset, 0))
    for i, p in enumerate(plan):
        raw = os.pread(fd, p['nbytes'], data_offset + p['rel'])
        assert len(raw) == p['nbytes']
        p['hash'] = blake(raw)
        if p['convert']:
            with open(os.path.join(args.backup, f'{i:05d}.bin'), 'wb') as fb:
                fb.write(raw)
    manifest = {'file': os.path.abspath(args.file), 'size': size, 'align': align, 'data_offset': data_offset,
                'size_new': data_offset + run, 'tensors': plan}
    with open(os.path.join(args.backup, 'manifest.json'), 'w') as fm:
        json.dump(manifest, fm, indent=1)
    print(f'backup written to {args.backup}; converting')
    # the compaction, in file order
    for i, p in enumerate(plan):
        raw = os.pread(fd, p['nbytes'], data_offset + p['rel'])
        if p['convert']:
            f = dequant(raw, T(p['type']), p['shape'])
            if release is not None:
                src = release[p['name']]
                new = EXACT_SOURCES[src.tensor_type](np.asarray(src.data).tobytes(), p['shape'][0])
                # exact up to the fp16 rounding of a product scale: against the release's own values
                fr = dequant(np.asarray(src.data).tobytes(), src.tensor_type, p['shape'])
                g = dequant(new, to, p['shape'])
                rel = float(np.sqrt(((g - fr)**2).mean()) / max(1e-30, np.sqrt((fr**2).mean())))
                assert rel < 2e-3, (p['name'], 'not exact', rel)
                if i % 50 == 0 or 'blk.' not in p['name']:
                    print(f"  {p['name']:40s} {src.tensor_type.name} -> {to.name}: rel rms vs release {rel:.2e}, vs the file's {float(np.sqrt(((g - f)**2).mean()) / np.sqrt((f**2).mean())):.2e}")
            else:
                new = quant(f, to)
                # the re-encoding error against the source, as a check of the layout: within the step of the target
                g = dequant(new, to, p['shape'])
                err = float(np.abs(g - f).max())
                scale = float(np.abs(f).max())
                assert err <= 0.6*scale, (p['name'], err, scale)
                if i % 50 == 0 or 'blk.' not in p['name']:
                    print(f"  {p['name']:40s} max |err| {err:.3g} of max |w| {scale:.3g}")
            assert len(new) == p['nbytes_new'], (p['name'], len(new), p['nbytes_new'])
            raw = new
        if p['rel_new'] != p['rel'] or p['convert']:
            os.pwrite(fd, raw, data_offset + p['rel_new'])
        os.pwrite(fd, struct.pack('<I', p['type_new']), p['type_pos'])
        os.pwrite(fd, struct.pack('<Q', p['rel_new']), p['off_pos'])
    os.ftruncate(fd, data_offset + run)
    os.fsync(fd)
    os.close(fd)
    print('converted; verifying')
    verify(args.file, manifest)


def verify(path: str, manifest) -> None:
    tensors, align, data_offset, size = scan(path)
    assert size == manifest['size_new'], (size, manifest['size_new'])
    byname = {t['name']: t for t in tensors}
    fd = os.open(path, os.O_RDONLY)
    bad = 0
    for p in manifest['tensors']:
        t = byname[p['name']]
        ok = t['type'] == p['type_new'] and t['nbytes'] == p['nbytes_new'] and t['rel'] == p['rel_new'] and t['shape'] == p['shape']
        if ok and not p['convert']:
            ok = blake(os.pread(fd, t['nbytes'], data_offset + t['rel'])) == p['hash']
        if not ok:
            bad += 1
            print('  MISMATCH', p['name'])
    os.close(fd)
    print(f'verified {len(manifest["tensors"])} tensors, {bad} mismatches')
    assert bad == 0


def restore(args) -> None:
    with open(os.path.join(args.restore, 'manifest.json')) as fm:
        manifest = json.load(fm)
    plan = manifest['tensors']
    data_offset = manifest['data_offset']
    assert os.path.getsize(args.file) == manifest['size_new'], 'the file is not the converted one'
    fd = os.open(args.file, os.O_RDWR)
    # reverse file order: a tensor's old place is at or after its new one, and the tensors after it are back already
    for i in range(len(plan) - 1, -1, -1):
        p = plan[i]
        if p['convert']:
            with open(os.path.join(args.restore, f'{i:05d}.bin'), 'rb') as fb:
                raw = fb.read()
            assert len(raw) == p['nbytes']
        else:
            raw = os.pread(fd, p['nbytes'], data_offset + p['rel_new'])
        if p['convert'] or p['rel_new'] != p['rel']:
            os.pwrite(fd, raw, data_offset + p['rel'])
    with open(os.path.join(args.restore, 'header.bin'), 'rb') as fh:
        os.pwrite(fd, fh.read(), 0)
    os.ftruncate(fd, manifest['size'])
    os.fsync(fd)
    os.close(fd)
    tensors, align, data_offset2, size = scan(args.file)
    assert size == manifest['size'] and data_offset2 == data_offset
    byname = {t['name']: t for t in tensors}
    fd = os.open(args.file, os.O_RDONLY)
    bad = 0
    for p in plan:
        t = byname[p['name']]
        ok = t['type'] == p['type'] and t['rel'] == p['rel'] and t['nbytes'] == p['nbytes'] and \
            blake(os.pread(fd, t['nbytes'], data_offset + t['rel'])) == p['hash']
        if not ok:
            bad += 1
            print('  MISMATCH', p['name'])
    os.close(fd)
    print(f'restored and verified {len(plan)} tensors, {bad} mismatches')
    assert bad == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--to', default='q4_0')
    ap.add_argument('--from-release', default='', help='the release file whose Q3_K/Q2_0 tensors are re-encoded exactly (--to q4_16)')
    ap.add_argument('--only', default='')
    ap.add_argument('--backup', default='')
    ap.add_argument('--restore', default='')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.restore:
        restore(args)
        return
    assert args.only and (args.backup or args.dry_run), 'need --only and --backup'
    convert(args)


if __name__ == '__main__':
    main()
