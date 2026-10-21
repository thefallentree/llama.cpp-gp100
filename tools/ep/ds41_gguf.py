"""Minimal GGUF (v3) reader for the expert-parallel tools: tensor infos of a split model (no gguf python package needed)."""
import glob, os, re, struct

# ggml type id -> (block elements, bytes per block)
TYPES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
         10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
         30: (1, 2)}   # F32 F16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1 Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K BF16

def _str(f):
    n, = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8', 'replace')

def _skip_value(f, t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sizes:
        f.seek(sizes[t], 1)
    elif t == 8:
        n, = struct.unpack('<Q', f.read(8)); f.seek(n, 1)
    elif t == 9:
        et, n = struct.unpack('<IQ', f.read(12))
        if et in sizes:
            f.seek(sizes[et] * n, 1)
        else:
            for _ in range(n):
                _skip_value(f, et)
    else:
        raise ValueError('gguf value type %d' % t)

def read_file(path):
    """{name: {file, offset (absolute), dims (ne0 first), type, nbytes}} of one GGUF file"""
    out = {}
    with open(path, 'rb') as f:
        magic, ver, n_t, n_kv = struct.unpack('<4sIQQ', f.read(24))
        assert magic == b'GGUF' and ver == 3, (path, magic, ver)
        align = 32
        for _ in range(n_kv):
            key = _str(f)
            t, = struct.unpack('<I', f.read(4))
            if key == 'general.alignment' and t == 4:
                align, = struct.unpack('<I', f.read(4))
            else:
                _skip_value(f, t)
        infos = []
        for _ in range(n_t):
            name = _str(f)
            nd, = struct.unpack('<I', f.read(4))
            dims = struct.unpack('<%dQ' % nd, f.read(8 * nd))
            ty, off = struct.unpack('<IQ', f.read(12))
            infos.append((name, dims, ty, off))
        data = (f.tell() + align - 1) // align * align
    for name, dims, ty, off in infos:
        be, bb = TYPES.get(ty, (None, None))
        n_el = 1
        for d in dims:
            n_el *= d
        nbytes = n_el // be * bb if be else None
        out[name] = {'file': path, 'offset': data + off, 'dims': list(dims), 'type': ty, 'nbytes': nbytes}
    return out

def read_split(first):
    """all tensors of a model given as -00001-of-0000N.gguf (or a single file)"""
    m = re.match(r'^(.*)-(\d+)-of-(\d+)\.gguf$', first)
    files = [first]
    if m:
        n = int(m.group(3))
        files = ['%s-%05d-of-%05d.gguf' % (m.group(1), i, n) for i in range(1, n + 1)]
    out = {}
    for p in files:
        out.update(read_file(p))
    return out
