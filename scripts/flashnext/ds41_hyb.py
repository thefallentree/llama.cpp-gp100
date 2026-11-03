#!/usr/bin/env python3
"""Cut the routed experts of a DeepSeek-V4.1 GGUF into the hot/cold tensor pairs the engine loads.

For every layer L with a hot count h > 0 and each matrix X in (gate, up, down) the tensor
    blk.L.ffn_X_exps.weight        ne = [n_in, n_out, n_expert]
is replaced by two tensors of the same type, one right after the other:
    blk.L.ffn_X_exps.weight        the first h experts          ne = [n_in, n_out, h]
    blk.L.ffn_X_exps_cold.weight   the other n_expert - h       ne = [n_in, n_out, n_expert - h]
This is the layout llama_model_deepseek41 (src/models/deepseek41.cpp, the LLAMA_EXPERT_HOT block) cuts at load time
with split_single(), or finds ready made and re-cuts with resplit(): hot and cold in the same file, the same type
and dims but ne[2], and the cold data starting exactly where the hot data ends (hot bytes % general.alignment == 0,
checked here). Having the cold part as a tensor of its own lets llama-quantize requantize it (e.g. to Q2_0) while
the hot tensors keep their type. Expert ids are not changed (hot = ids [0, h), cold = ids [h, n_expert)), so the
router tensors (ffn_gate_inp, exp_probs_b) stay as they are. To put the hot experts first run
fn_permute_inplace.py before this script.

Metadata: <arch>.expert_hot_count (uint32) is set to --global-hot (default: the most common count of the list); the
loader takes it as the cut of every layer that LLAMA_EXPERT_HOT_LAYERS does not name (it re-cuts the pair there with
resplit(), so the per-layer counts of the file are NOT what it uses). The per-layer counts are also recorded in
<arch>.expert_hot_count_per_layer (uint32[n_layer], informational: the loader does not read it). Serve a file with
different counts per layer with LLAMA_EXPERT_HOT_LAYERS=<the list this script prints>; do not set LLAMA_EXPERT_HOT=0, that
asks for whole-layer tensors. split.tensors.count grows by 3 per cut layer. Every other KV entry and every other tensor is
copied byte for byte.

Output (default: the shard layout of the input is kept, output shard i holds what input shard i held, so a pair is
always inside one shard): OUTPUT is a directory (the input file names are reused), or a path ending in
-00001-of-0000N.gguf, or a path ending in .gguf (the -0000i-of-0000N suffix is added). --single writes one file.
Shards are written as NAME.partial and renamed when complete and synced; a rerun skips the shards that are already
complete for the same plan (their header bytes are compared), so an interrupted run resumes at shard granularity.
Memory: one 32 MiB buffer. Disk: the output needs the size of the input (the input is never touched).

Requantizing the cold tensors afterwards, e.g. to Q2_0, with the hot tensors and everything else untouched:
  ds41_hyb.py IN-00001-of-0000N.gguf OUT/ --hot-layers 140,140,... --write-quant-types types.txt
  llama-quantize --allow-requantize --keep-split --tensor-type-file types.txt OUT/NAME-00001-of-0000N.gguf Q/NAME.gguf Q2_K
(llama-quantize copies a tensor whose target type is its own type, but applies its mixture rules to every 2-D weight it has
no rule for, so types.txt has a line for each weight: the cold ones -> q2_0, the others -> the type they have; the final Q2_K
only sets general.file_type. --keep-split keeps every pair in its shard. Create the output directory first.)
Caveats: llama_model_loader::resplit() needs both halves of a pair to have the same type, even when the file already cuts
where it is asked to, so a Q2_0 cold half loads only with runs/fn-m2/k/ds41_mixed_pair_loader.patch; the engine's expert cache
and host tier address cold slices with the hot tensor's stride and type (expert-cache.cu, moe-host.cu, mmid-f16-sm60.cu).

usage: ds41_hyb.py INPUT-00001-of-0000N.gguf OUTPUT (--hot-layers SPEC | --hot N) [--global-hot N] [--single]
                   [--dry-run] [--force] [--no-verify] [--verify-data] [--write-quant-types FILE]
SPEC: "n,n,..." one count per layer from layer 0, or "il:n,..." (the LLAMA_EXPERT_HOT_LAYERS syntax: a bare number
continues after the previous layer, a later entry wins), or @FILE with either form (or a JSON list). 0 leaves a
layer uncut.
"""
import argparse
import collections
import json
import os
import re
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))  # this repo's gguf-py (Q2_0, Q4_16)
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as T  # noqa: E402
from gguf import GGUFValueType as VT  # noqa: E402

EXPERT_RE = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$')
COLD_RE = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)_exps_cold\.weight$')
SHARD_RE = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$')
SPLIT_KEYS = (gguf.Keys.Split.LLM_KV_SPLIT_NO, gguf.Keys.Split.LLM_KV_SPLIT_COUNT, gguf.Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT)
KV_START = 24            # magic, version, n_tensors, n_kv
CHUNK = 32 << 20
SYNC_EVERY = 1 << 30     # fdatasync + drop the page cache of what was written, so a 265 GB copy does not fill the memory


class Fail(Exception):
    pass


def pad(n: int, align: int) -> int:
    return (n + align - 1) // align * align


def gib(n: float) -> str:
    return f'{n / (1 << 30):.2f} GiB'


# ---------------------------------------------------------------------------------------------------- reading

@dataclass
class TInfo:
    name: str
    ne: Tuple[int, ...]     # ggml order: ne[0] is the fastest dim
    type: int               # raw ggml type id
    nbytes: int
    off: int                # absolute offset of the data in its file


@dataclass
class Shard:
    path: str
    version: int
    alignment: int
    size: int
    data_offset: int                                   # absolute offset of the data section
    kv_end: int                                        # = start of the tensor infos
    fields: List[Tuple[str, int, int, Tuple[int, ...]]]    # (key, start, end, value types) of the raw KV entries
    tensors: List[TInfo]
    arch: str = ''


def parse_shard(path: str) -> Shard:
    try:
        r = gguf.GGUFReader(path)
    except (ValueError, KeyError) as e:
        raise Fail(f'{path}: cannot parse the GGUF header ({e})')
    if r.endianess != gguf.GGUFEndian.LITTLE:
        raise Fail(f'{path}: big-endian GGUF is not supported')
    if not r.tensors:
        raise Fail(f'{path}: no tensors')
    fields = [f for f in r.fields.values() if not f.name.startswith('GGUF.')]
    kv_end = int(r.tensors[0].field.offset)
    ents = []
    for i, f in enumerate(fields):
        end = int(fields[i + 1].offset) if i + 1 < len(fields) else kv_end
        ents.append((f.name, int(f.offset), end, tuple(int(t) for t in f.types)))
    if not ents or ents[0][1] != KV_START:
        raise Fail(f'{path}: unexpected KV layout (first entry at {ents[0][1] if ents else None})')
    arch_f = r.fields.get(gguf.Keys.General.ARCHITECTURE)
    ts = [TInfo(t.name, tuple(int(x) for x in t.shape), int(t.tensor_type), int(t.n_bytes), int(t.data_offset)) for t in r.tensors]
    return Shard(path, int(r.fields['GGUF.version'].contents()), int(r.alignment), os.path.getsize(path), int(r.data_offset), kv_end,
                 ents, ts, arch_f.contents() if arch_f is not None else '')   # the reader (and its mapping) is dropped on return


def find_input_shards(first: str) -> List[str]:
    m = SHARD_RE.match(os.path.basename(first))
    if not m:
        return [first]
    prefix, idx, n = m.group(1), int(m.group(2)), int(m.group(3))
    if idx != 1:
        raise Fail(f'{first}: give the first shard (-00001-of-{n:05d}.gguf)')
    d = os.path.dirname(first)
    return [os.path.join(d, f'{prefix}-{i:05d}-of-{n:05d}.gguf') for i in range(1, n + 1)]


# ---------------------------------------------------------------------------------------------------- the hot list

def parse_hot_spec(spec: str) -> Dict[int, int]:
    """The LLAMA_EXPERT_HOT_LAYERS syntax; a JSON list and @FILE are accepted as well."""
    if spec.startswith('@'):
        try:
            with open(spec[1:]) as f:
                spec = f.read().strip()
        except OSError as e:
            raise Fail(f'{spec[1:]}: {e.strerror}')
    if spec.startswith('['):
        try:
            return {i: int(v) for i, v in enumerate(json.loads(spec))}
        except (ValueError, TypeError):
            raise Fail('bad JSON hot list')
    out: Dict[int, int] = {}
    il_next = 0
    for tok in re.split(r'[,\s]+', spec):
        if not tok:
            continue
        try:
            if ':' in tok:
                a, b = tok.split(':', 1)
                il, n = int(a), int(b)
            else:
                il, n = il_next, int(tok)
        except ValueError:
            raise Fail(f'bad hot list entry {tok!r}')
        if il < 0 or n < 0:
            raise Fail(f'bad hot list entry {tok!r}')
        out[il] = n
        il_next = il + 1
    return out


# ---------------------------------------------------------------------------------------------------- the plan

@dataclass
class OutTensor:
    name: str
    ne: Tuple[int, ...]
    type: int
    nbytes: int
    src: int        # input shard index
    off: int        # absolute offset in that shard


@dataclass
class OutFile:
    path: str
    tensors: List[OutTensor]
    header: bytes = b''
    size: int = 0
    in_shards: List[int] = field(default_factory=list)


def pack_str(s: str) -> bytes:
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def kv_scalar(key: str, vtype: int, fmt: str, value: int) -> bytes:
    return pack_str(key) + struct.pack('<I', int(vtype)) + struct.pack('<' + fmt, value)


def kv_u32_array(key: str, vals: List[int]) -> bytes:
    return pack_str(key) + struct.pack('<IIQ', int(VT.ARRAY), int(VT.UINT32), len(vals)) + struct.pack(f'<{len(vals)}I', *vals)


def build_header(version: int, kv: List[bytes], tensors: List[OutTensor], align: int) -> Tuple[bytes, int]:
    """The bytes up to the data section (padded), and the size of the data section."""
    off = 0
    ti = []
    for t in tensors:
        ti.append(pack_str(t.name) + struct.pack('<I', len(t.ne)) + struct.pack(f'<{len(t.ne)}Q', *t.ne) + struct.pack('<IQ', t.type, off))
        off += pad(t.nbytes, align)
    head = b'GGUF' + struct.pack('<IQQ', version, len(tensors), len(kv)) + b''.join(kv) + b''.join(ti)
    head += bytes(pad(len(head), align) - len(head))
    return head, off


def hot_layers_env(per_layer: List[int]) -> str:
    """LLAMA_EXPERT_HOT_LAYERS for these counts (the loader rejects entries below 1, so uncut layers are left out)."""
    if all(n > 0 for n in per_layer):
        return ','.join(str(n) for n in per_layer)
    return ','.join(f'{l}:{n}' for l, n in enumerate(per_layer) if n > 0)


def existing_ancestor(p: str) -> str:
    p = os.path.abspath(p)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return p


def type_name(t: int) -> str:
    try:
        return T(t).name
    except ValueError:
        return f'type{t}'


def make_plan(args) -> Tuple[List[Shard], List[OutFile], Dict[int, int], int, Dict[str, object]]:
    in_paths = find_input_shards(args.input)
    for p in in_paths:
        if not os.path.isfile(p):
            raise Fail(f'{p}: no such file')
    t0 = time.time()
    shards = [parse_shard(p) for p in in_paths]
    arch = shards[0].arch
    if not arch:
        raise Fail(f'{in_paths[0]}: no general.architecture')
    if arch != 'deepseek41':
        print(f'warning: architecture is {arch!r}, not deepseek41 (the pair layout is the same for glm5-next / qwen4exp)', file=sys.stderr)
    for s in shards:
        if s.arch not in ('', arch):
            raise Fail(f'{s.path}: architecture {s.arch!r} differs from the first shard')
    print(f'parsed {len(shards)} shard header(s) in {time.time() - t0:.1f} s: '
          f'{sum(len(s.tensors) for s in shards)} tensors, {gib(sum(s.size for s in shards))}')

    # the split metadata of the input
    n_in_tensors = sum(len(s.tensors) for s in shards)
    for i, s in enumerate(shards):
        by = {n: (a, b, ty) for n, a, b, ty in s.fields}
        if len(shards) > 1:
            if gguf.Keys.Split.LLM_KV_SPLIT_NO not in by or gguf.Keys.Split.LLM_KV_SPLIT_COUNT not in by:
                raise Fail(f'{s.path}: split.no / split.count missing')
    # the expert tensors
    experts: Dict[Tuple[int, str], Tuple[int, TInfo]] = {}   # (layer, gate|up|down) -> (shard, tensor)
    n_expert = 0
    for si, s in enumerate(shards):
        for t in s.tensors:
            if COLD_RE.match(t.name):
                raise Fail(f'{s.path}: {t.name} is already a cold tensor; give the file before the cut')
            m = EXPERT_RE.match(t.name)
            if not m:
                continue
            if len(t.ne) != 3:
                raise Fail(f'{t.name}: {len(t.ne)} dims, expected [n_in, n_out, n_expert]')
            if n_expert and t.ne[2] != n_expert:
                raise Fail(f'{t.name}: {t.ne[2]} experts, other layers have {n_expert}')
            n_expert = t.ne[2]
            experts[(int(m.group(1)), m.group(2))] = (si, t)
    if not experts:
        raise Fail('no blk.L.ffn_{gate,up,down}_exps.weight tensors in the input')
    layers = sorted({l for l, _ in experts})
    for l in layers:
        for k in ('gate', 'up', 'down'):
            if (l, k) not in experts:
                raise Fail(f'layer {l} has no ffn_{k}_exps.weight')
    print(f'{len(layers)} layers with experts ({layers[0]}..{layers[-1]}), {n_expert} experts each')

    # the hot counts
    if args.hot is not None:
        hot = {l: args.hot for l in layers}
    else:
        hot = parse_hot_spec(args.hot_layers)
    for l in sorted(hot):
        if l not in layers:
            print(f'warning: hot count given for layer {l}, which has no experts: ignored', file=sys.stderr)
            del hot[l]
    for l in layers:
        if l not in hot:
            if not args.allow_unlisted:
                raise Fail(f'layer {l} has experts but no hot count (--allow-unlisted leaves such layers uncut)')
            hot[l] = 0
        if hot[l] >= n_expert:
            raise Fail(f'layer {l}: hot count {hot[l]} must be below the {n_expert} experts')
    cut_layers = [l for l in layers if hot[l] > 0]
    if not cut_layers:
        raise Fail('every hot count is 0: nothing to cut')
    counts = collections.Counter(hot[l] for l in cut_layers)
    g = args.global_hot if args.global_hot is not None else min(c for c in counts if counts[c] == max(counts.values()))
    if not 1 <= g < n_expert:
        raise Fail(f'--global-hot {g} must be in [1, {n_expert - 1}]')

    # tensors per input shard, with the cut applied
    align0 = shards[0].alignment
    out_tensors: List[List[OutTensor]] = []
    n_cut = 0
    for si, s in enumerate(shards):
        if args.single and s.alignment != align0:
            raise Fail('--single needs one alignment for all shards')
        lst: List[OutTensor] = []
        for t in s.tensors:
            m = EXPERT_RE.match(t.name)
            h = hot.get(int(m.group(1)), 0) if m else 0
            if not h:
                lst.append(OutTensor(t.name, t.ne, t.type, t.nbytes, si, t.off))
                continue
            if t.nbytes % n_expert:
                raise Fail(f'{t.name}: {t.nbytes} bytes do not divide into {n_expert} experts')
            sl = t.nbytes // n_expert
            if (h * sl) % s.alignment:
                raise Fail(f'{t.name}: {h} experts are {h * sl} bytes, not a multiple of the alignment {s.alignment}; '
                           f'the cold tensor could not follow the hot one without padding (the loader needs that)')
            lst.append(OutTensor(t.name, (t.ne[0], t.ne[1], h), t.type, h * sl, si, t.off))
            lst.append(OutTensor(t.name.replace('_exps.weight', '_exps_cold.weight'), (t.ne[0], t.ne[1], n_expert - h), t.type,
                                 (n_expert - h) * sl, si, t.off + h * sl))
            n_cut += 1
        out_tensors.append(lst)
    n_out_tensors = n_in_tensors + n_cut

    # the key/value entries of every output file
    hot_key, arr_key = f'{arch}.expert_hot_count', f'{arch}.expert_hot_count_per_layer'
    n_layer_arr = (max(layers) + 1) if layers else 0
    per_layer = [hot.get(l, 0) for l in range(n_layer_arr)]

    def kv_entries(si: int, single: bool) -> List[bytes]:
        s = shards[si]
        with open(s.path, 'rb') as f:
            f.seek(KV_START)
            raw = f.read(s.kv_end - KV_START)
        out = []
        for name, a, b, types in s.fields:
            ent = raw[a - KV_START:b - KV_START]
            if name in (hot_key, arr_key):
                continue
            if name in SPLIT_KEYS:
                if single:
                    continue
                if name == gguf.Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT:
                    if types != (int(VT.INT32),):
                        raise Fail(f'{s.path}: {name} is not an int32')
                    ent = kv_scalar(name, VT.INT32, 'i', n_out_tensors)
            out.append(ent)
        if si == 0:
            out.append(kv_scalar(hot_key, VT.UINT32, 'I', g))
            out.append(kv_u32_array(arr_key, per_layer))
        return out

    # the output files
    outs: List[OutFile] = []
    if args.single:
        paths = [resolve_single_output(args.output, in_paths[0])]
        merged = [t for lst in out_tensors for t in lst]
        # the first shard carries the KV; the split keys are dropped
        for s in shards[1:]:
            lost = [n for n, _, _, _ in s.fields if n not in SPLIT_KEYS]
            if lost:
                print(f'warning: --single drops the KV entries of {s.path} other than split.*: {lost[:5]}', file=sys.stderr)
        of = OutFile(paths[0], merged, in_shards=list(range(len(shards))))
        of.header, dsize = build_header(shards[0].version, kv_entries(0, True), merged, align0)
        of.size = len(of.header) + dsize
        outs.append(of)
    else:
        paths = resolve_outputs(args.output, in_paths)
        for si, s in enumerate(shards):
            of = OutFile(paths[si], out_tensors[si], in_shards=[si])
            of.header, dsize = build_header(s.version, kv_entries(si, False), out_tensors[si], s.alignment)
            of.size = len(of.header) + dsize
            outs.append(of)
    info = dict(arch=arch, n_expert=n_expert, layers=layers, hot=hot, global_hot=g, n_cut=n_cut, n_in_tensors=n_in_tensors,
                n_out_tensors=n_out_tensors, in_bytes=sum(s.size for s in shards), per_layer=per_layer)
    return shards, outs, hot, g, info


def resolve_outputs(output: str, in_paths: List[str]) -> List[str]:
    n = len(in_paths)
    if is_dir_spec(output):
        return [os.path.join(output, os.path.basename(p)) for p in in_paths]
    if n == 1:
        return [output]
    m = SHARD_RE.match(output)
    if m:
        if int(m.group(2)) != 1 or int(m.group(3)) != n:
            raise Fail(f'{output}: expected -00001-of-{n:05d}.gguf')
        return [f'{m.group(1)}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]
    return [f'{output[:-5]}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]


def is_dir_spec(output: str) -> bool:
    """A directory (it is created): ends in /, exists as one, or is not a .gguf path."""
    return output.endswith('/') or os.path.isdir(output) or not output.endswith('.gguf')


def resolve_single_output(output: str, first_in: str) -> str:
    if is_dir_spec(output):
        m = SHARD_RE.match(os.path.basename(first_in))
        base = (m.group(1) + '.gguf') if m else os.path.basename(first_in)
        return os.path.join(output, base)
    return output


# ---------------------------------------------------------------------------------------------------- writing

class Progress:
    def __init__(self, total: int):
        self.total, self.done, self.t0, self.last = total, 0, time.time(), 0.0

    def add(self, n: int, label: str):
        self.done += n
        now = time.time()
        if now - self.last >= 5:
            self.last = now
            rate = self.done / max(now - self.t0, 1e-6)
            eta = (self.total - self.done) / max(rate, 1)
            print(f'  {label}: {gib(self.done)} / {gib(self.total)}  {rate / 1e6:.0f} MB/s  ETA {eta / 60:.1f} min', flush=True)


def write_file(of: OutFile, shards: List[Shard], prog: Progress, label: str):
    tmp = of.path + '.partial'
    fds = {si: os.open(shards[si].path, os.O_RDONLY) for si in of.in_shards}
    for fd in fds.values():
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except OSError:
            pass
    buf = bytearray(CHUNK)
    mv = memoryview(buf)
    since_sync = 0
    try:
        with open(tmp, 'wb') as fout:
            fout.write(of.header)
            align = shards[of.in_shards[0]].alignment
            for t in of.tensors:
                fd = fds[t.src]
                done = 0
                while done < t.nbytes:
                    want = min(CHUNK, t.nbytes - done)
                    got = os.preadv(fd, [mv[:want]], t.off + done)
                    if got <= 0:
                        raise Fail(f'{shards[t.src].path}: short read of {t.name} at {t.off + done}')
                    fout.write(mv[:got])
                    done += got
                    since_sync += got
                    prog.add(got, label)
                    if since_sync >= SYNC_EVERY:
                        since_sync = 0
                        fout.flush()
                        os.fdatasync(fout.fileno())
                        try:
                            os.posix_fadvise(fout.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        except OSError:
                            pass
                p = pad(t.nbytes, align) - t.nbytes
                if p:
                    fout.write(bytes(p))
            fout.flush()
            os.fsync(fout.fileno())
    finally:
        for fd in fds.values():
            os.close(fd)
    if os.path.getsize(tmp) != of.size:
        raise Fail(f'{tmp}: wrote {os.path.getsize(tmp)} bytes, expected {of.size}')
    os.replace(tmp, of.path)
    dfd = os.open(os.path.dirname(os.path.abspath(of.path)), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def is_complete(of: OutFile) -> bool:
    """A finished output of this plan: right size and the very same header bytes (renamed only after the sync)."""
    try:
        if os.path.getsize(of.path) != of.size:
            return False
        with open(of.path, 'rb') as f:
            return f.read(len(of.header)) == of.header
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------------- verification

def verify_outputs(outs: List[OutFile], shards: List[Shard], hot: Dict[int, int], n_expert: int):
    """Header level: the files parse, match the plan, and obey what resplit()/split_single() need."""
    n_pairs = 0
    for of in outs:
        r = parse_shard(of.path)
        if r.size != of.size:
            raise Fail(f'{of.path}: size {r.size}, planned {of.size}')
        if len(r.tensors) != len(of.tensors):
            raise Fail(f'{of.path}: {len(r.tensors)} tensors, planned {len(of.tensors)}')
        with open(of.path, 'rb') as f:
            if f.read(len(of.header)) != of.header:
                raise Fail(f'{of.path}: header differs from the plan')
        data0 = r.data_offset
        idx = {}
        for i, (p, t) in enumerate(zip(of.tensors, r.tensors)):
            idx[t.name] = i
            if (t.name, t.ne, t.type, t.nbytes) != (p.name, p.ne, p.type, p.nbytes):
                raise Fail(f'{of.path}: tensor {i} is {t.name} {t.ne} {t.type}, planned {p.name} {p.ne} {p.type}')
            if t.off % r.alignment or (t.off - data0) % r.alignment:
                raise Fail(f'{of.path}: {t.name} is not aligned')
        for name, i in idx.items():
            m = EXPERT_RE.match(name)
            if not m or not hot.get(int(m.group(1))):
                continue
            h, c = r.tensors[i], r.tensors[idx[name.replace('_exps.weight', '_exps_cold.weight')]]
            ok = (idx[c.name] == i + 1 and h.type == c.type and h.ne[:2] == c.ne[:2] and h.ne[2] == hot[int(m.group(1))]
                  and h.ne[2] + c.ne[2] == n_expert and c.off == h.off + h.nbytes and h.nbytes % r.alignment == 0
                  and h.nbytes // h.ne[2] == c.nbytes // c.ne[2])
            if not ok:
                raise Fail(f'{of.path}: {name}: the pair does not satisfy the loader rules')
            n_pairs += 1
    print(f'verified {len(outs)} file(s): {n_pairs} hot/cold pairs; headers match the plan, cold follows hot with no padding')


def verify_data(outs: List[OutFile], shards: List[Shard]):
    """Every output tensor against the input bytes it was cut from."""
    t0 = time.time()
    total = 0
    a, b = bytearray(CHUNK), bytearray(CHUNK)
    for of in outs:
        r = parse_shard(of.path)
        fo = os.open(of.path, os.O_RDONLY)
        fds = {si: os.open(shards[si].path, os.O_RDONLY) for si in of.in_shards}
        try:
            for p, t in zip(of.tensors, r.tensors):
                done = 0
                while done < p.nbytes:
                    n = min(CHUNK, p.nbytes - done)
                    ga = os.preadv(fo, [memoryview(a)[:n]], t.off + done)
                    gb = os.preadv(fds[p.src], [memoryview(b)[:n]], p.off + done)
                    if ga != gb or ga <= 0:
                        raise Fail(f'{of.path}: short read in {p.name}')
                    if (a == b) if ga == CHUNK else (a[:ga] == b[:ga]):
                        pass
                    else:
                        raise Fail(f'{of.path}: {p.name} differs from the input at +{done}')
                    done += ga
                total += p.nbytes
        finally:
            os.close(fo)
            for fd in fds.values():
                os.close(fd)
    print(f'data verified: {gib(total)} compared with the input in {time.time() - t0:.0f} s')


def write_quant_types(path: str, outs: List[OutFile], cold_type: str):
    """A llama-quantize --tensor-type-file: cold tensors -> cold_type, every other 2-D+ weight keeps its current type.
    llama-quantize does nothing to a tensor whose target type is its own type (copy), but without a line for a tensor
    it applies its own mixture rules (hc_*, F32/BF16 weights, the routed experts would be requantized)."""
    lines = [f'_exps_cold\\.weight$={cold_type}']
    seen = set()
    for of in outs:
        for t in of.tensors:
            if COLD_RE.match(t.name) or not t.name.endswith('.weight') or len([d for d in t.ne if d > 1]) < 2 or t.name in seen:
                continue
            seen.add(t.name)
            lines.append(f'^{re.escape(t.name)}$={type_name(t.type).lower()}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {path}: the cold tensors -> {cold_type}, {len(lines) - 1} weights keep their type '
          f'(llama-quantize --allow-requantize --tensor-type-file {path} IN OUT Q2_K; the Q2_K only sets general.file_type)')


# ---------------------------------------------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='the first shard (-00001-of-0000N.gguf) or a single GGUF')
    ap.add_argument('output', help='directory, shard path or .gguf path (see the module doc)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--hot-layers', help='per-layer hot counts: "n,n,..." or "il:n,..." (LLAMA_EXPERT_HOT_LAYERS syntax) or @FILE')
    g.add_argument('--hot', type=int, help='the same hot count for every layer')
    ap.add_argument('--global-hot', type=int, help='value of <arch>.expert_hot_count (default: the most common count)')
    ap.add_argument('--allow-unlisted', action='store_true', help='leave layers that have experts but no hot count uncut')
    ap.add_argument('--single', action='store_true', help='write one file instead of keeping the input shards')
    ap.add_argument('--force', action='store_true', help='overwrite outputs that exist and are not complete for this plan')
    ap.add_argument('--dry-run', action='store_true', help='print the plan, write nothing')
    ap.add_argument('--no-verify', action='store_true', help='skip the header check after writing')
    ap.add_argument('--verify-data', action='store_true', help='also compare every output byte with the input (reads both)')
    ap.add_argument('--min-free-gib', type=float, default=2.0, help='free space to keep beyond the output (default 2)')
    ap.add_argument('--write-quant-types', metavar='FILE', help='write a llama-quantize --tensor-type-file (cold -> --cold-type, rest unchanged)')
    ap.add_argument('--cold-type', default='q2_0', help='type for --write-quant-types (default q2_0)')
    args = ap.parse_args(argv)

    try:
        shards, outs, hot, g_hot, info = make_plan(args)
    except Fail as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    except OSError as e:
        print(f'error: {e.filename or ""}: {e.strerror}', file=sys.stderr)
        return 2

    in_real = {os.path.realpath(s.path) for s in shards}
    for of in outs:
        if os.path.realpath(of.path) in in_real:
            print(f'error: output {of.path} is an input file', file=sys.stderr)
            return 2

    n_layers_cut = info['n_cut'] // 3
    print(f'arch {info["arch"]}: {n_layers_cut} layers cut ({info["n_cut"]} tensors -> {2 * info["n_cut"]}), '
          f'{info["n_in_tensors"]} -> {info["n_out_tensors"]} tensors, expert_hot_count {g_hot}')
    print('LLAMA_EXPERT_HOT_LAYERS=' + hot_layers_env(info['per_layer']))
    if g_hot not in set(hot[l] for l in info['layers'] if hot[l]):
        print(f'note: --global-hot {g_hot} is not one of the layer counts')
    todo = []
    for of in outs:
        if is_complete(of):
            print(f'  {of.path}: complete for this plan, skipped')
            if os.path.exists(of.path + '.partial') and not args.dry_run:
                os.remove(of.path + '.partial')   # left by an interrupted rerun
        elif os.path.exists(of.path) and not args.force:
            print(f'error: {of.path} exists and is not the output of this plan (--force overwrites)', file=sys.stderr)
            return 2
        else:
            todo.append(of)
    need = sum(of.size for of in todo)
    d = os.path.dirname(os.path.abspath(outs[0].path))
    if not args.dry_run:
        os.makedirs(d, exist_ok=True)
    free = shutil.disk_usage(existing_ancestor(d)).free
    print(f'output {gib(sum(of.size for of in outs))} in {len(outs)} file(s), {gib(need)} to write; free in {d}: {gib(free)}')
    if args.write_quant_types:
        write_quant_types(args.write_quant_types, outs, args.cold_type)
    if args.dry_run:
        return 0
    if need + args.min_free_gib * (1 << 30) > free:
        print(f'error: not enough free space ({gib(free)} free, {gib(need)} + {args.min_free_gib} GiB needed)', file=sys.stderr)
        return 2

    t0 = time.time()
    prog = Progress(need)
    try:
        for n, of in enumerate(todo):
            if os.path.exists(of.path + '.partial'):
                os.remove(of.path + '.partial')
            label = f'{os.path.basename(of.path)} ({n + 1}/{len(todo)})'
            t1 = time.time()
            write_file(of, shards, prog, label)
            print(f'  wrote {of.path}: {gib(of.size)} in {time.time() - t1:.0f} s', flush=True)
        if todo:
            print(f'wrote {gib(need)} in {time.time() - t0:.0f} s ({need / max(time.time() - t0, 1e-6) / 1e6:.0f} MB/s)')
        if not args.no_verify:
            verify_outputs(outs, shards, hot, info['n_expert'])
        if args.verify_data:
            verify_data(outs, shards)
    except Fail as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    except OSError as e:
        print(f'error: {e.filename or ""}: {e.strerror} (the unfinished file is NAME.partial; a rerun starts that shard again)', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
