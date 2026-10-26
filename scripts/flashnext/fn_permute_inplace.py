#!/usr/bin/env python3
"""Reorder the routed experts of a MoE GGUF in place, layer by layer, so that each layer's experts come hot first
(the order the engine's hot/cold split cuts at load, LLAMA_EXPERT_HOT). Nothing is copied: the expert slices of
blk.L.ffn_{gate,up,down}_exps.weight, the rows of blk.L.ffn_gate_inp.weight (F32, F16 or BF16) and the entries of
blk.L.exp_probs_b.bias and blk.L.exp_probs_b_vl.bias (each when the file has it) are permuted with one slice buffer
(cycle following), so a 110 GB file needs no second copy and no extra disk.
Pure python (no numpy): it runs on a hypervisor with the file.

usage: fn_permute_inplace.py FILE.gguf ORDER.json [--dry-run | --verify-only]
FILE.gguf: a single file, or any shard of a split set (NAME-00001-of-0000N.gguf): all shards' headers are read and every
tensor is permuted in the shard that holds it (a layer's experts, its router and its biases may all be in different shards).
ORDER.json: {"3": [old expert ids in the new order], "4": [...], ...} from fn_glm_order.py (imatrix counts) or tools/ds41_order.py.
--dry-run: read the headers, check the order against every tensor and print the plan (shards, byte sizes). Writes nothing.
--verify-only: the dry run, then compare the file's slices with the fingerprints of the original slices (see below) and
the journal. Opens the files read-only. Without fingerprints it only checks the structure.
A journal FILE.gguf.permute.json records the layers done; rerunning skips them (a crash mid-layer leaves that layer
half permuted: the journal says 'in progress', restore the shards of that layer from the download then).
Before a layer is journaled as done, every slice is read back from disk and compared with the crc32 of the original slice
it should hold, and the crc32 of all original slices go to FILE.gguf.permute.fp.json (--verify-only checks a file against
them later; a copy needs the journal and the fingerprint file next to the first shard too). The journal and the fingerprints
belong to the first shard, whichever shard is given. SIGINT, SIGTERM and SIGHUP let the layer in progress finish, then stop
(a second signal aborts). Neither --dry-run nor --verify-only writes; an unknown flag is an error.
"""
import argparse
import collections
import errno
import fcntl
import json
import os
import re
import signal
import struct
import sys
import time
import zlib

# ggml type -> (name, block size, type size in bytes)
GGML_TYPES = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20), 8: ('Q8_0', 32, 34),
              10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144), 13: ('Q5_K', 256, 176),
              14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292), 23: ('IQ4_XS', 256, 136), 30: ('BF16', 1, 2),
              16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74), 22: ('IQ2_S', 256, 82), 20: ('IQ4_NL', 32, 18), 19: ('IQ1_S', 256, 50),
              29: ('IQ1_M', 256, 56), 21: ('IQ3_S', 256, 110), 18: ('IQ3_XXS', 256, 98)}
KV_FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}   # fixed-size value types
KV_SIZES = {vt: struct.calcsize('<' + fmt) for vt, fmt in KV_FORMATS.items()}
KV_STRING, KV_ARRAY = 8, 9
SPLIT_RE = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$')
LABELS = {'ffn_gate_exps': 'gate', 'ffn_up_exps': 'up', 'ffn_down_exps': 'down', 'ffn_gate_inp': 'router',
          'exp_probs_b': 'bias', 'exp_probs_b_vl': 'bias_vl'}


def type_info(name, ttype):
    info = GGML_TYPES.get(ttype)
    if info is None:
        raise SystemExit(f'{name}: ggml type {ttype} is not in the table')
    return info


def tensor_nbytes(name, dims, ttype):
    tname, bs, ts = type_info(name, ttype)
    if dims[0] % bs:
        raise SystemExit(f'{name}: ne0 {dims[0]} is not a multiple of the {tname} block {bs}')
    n = dims[0] // bs * ts
    for d in dims[1:]:
        n *= d
    return n


def tensor_slice_bytes(name, dims, ttype, axis):
    """bytes of one slice along `axis` (the dims after it must be 1)"""
    for d in dims[axis + 1:]:
        if d != 1:
            raise SystemExit(f'{name}: dims after axis {axis} are not 1: {dims}')
    if axis == 0 and type_info(name, ttype)[1] != 1:
        raise SystemExit(f'{name}: {type_info(name, ttype)[0]} is block quantized along ne0, it has no slices there')
    return tensor_nbytes(name, dims, ttype) // dims[axis]


class Header:
    """The GGUF header of one file: the scalar KV pairs (arrays are skipped), the tensor infos and the data section offset."""

    def __init__(self, f):
        self.f = f
        self.fname = getattr(f, 'name', 'GGUF')
        try:
            self._parse()
        except struct.error:
            raise SystemExit(f'{self.fname}: the GGUF header is truncated')

    def _parse(self):
        f = self.f
        f.seek(0)
        magic = f.read(4)
        if magic != b'GGUF':
            raise SystemExit(f'{self.fname}: not a GGUF file: {magic!r}')
        self.version, = struct.unpack('<I', f.read(4))
        if self.version < 3:
            raise SystemExit(f'{self.fname}: GGUF v{self.version}: only v3 is handled')
        n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
        self.kv = {}
        for _ in range(n_kv):
            key = self._str()
            vt, = struct.unpack('<I', f.read(4))
            value = self._value(vt)
            if value is not None:
                self.kv[key] = value
        self.alignment = self.kv.get('general.alignment', 32)
        if not isinstance(self.alignment, int) or self.alignment <= 0:
            raise SystemExit(f'{self.fname}: bad general.alignment {self.alignment!r}')
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            n_dims, = struct.unpack('<I', f.read(4))
            dims = list(struct.unpack('<' + 'Q' * n_dims, f.read(8 * n_dims)))
            ttype, offset = struct.unpack('<IQ', f.read(12))
            if name in self.tensors:
                raise SystemExit(f'{self.fname}: tensor {name} is listed twice')
            self.tensors[name] = (dims, ttype, offset)
        end = f.tell()
        self.data_start = (end + self.alignment - 1) // self.alignment * self.alignment

    def nbytes(self, name):
        dims, ttype, _ = self.tensors[name]
        return tensor_nbytes(name, dims, ttype)

    def slice_bytes(self, name, axis):
        dims, ttype, _ = self.tensors[name]
        return tensor_slice_bytes(name, dims, ttype, axis)

    def _str(self):
        n, = struct.unpack('<Q', self.f.read(8))
        if n > 1 << 30:
            raise SystemExit(f'{self.fname}: a string of {n} bytes: the header is corrupt')
        return self.f.read(n).decode('utf-8', 'replace')

    def _value(self, vt):
        """the value of a string or a scalar; None for an array (skipped)"""
        f = self.f
        if vt == KV_STRING:
            return self._str()
        if vt == KV_ARRAY:
            st, n = struct.unpack('<IQ', f.read(12))
            self._skip_array(st, n)
            return None
        if vt not in KV_FORMATS:
            raise SystemExit(f'{self.fname}: unknown KV value type {vt}')
        return struct.unpack('<' + KV_FORMATS[vt], f.read(KV_SIZES[vt]))[0]

    def _skip_array(self, st, n):
        f = self.f
        if st in KV_SIZES:
            f.seek(n * KV_SIZES[st], 1)
        elif st == KV_STRING:
            for _ in range(n):
                size, = struct.unpack('<Q', f.read(8))
                f.seek(size, 1)
        elif st == KV_ARRAY:
            for _ in range(n):
                inner, count = struct.unpack('<IQ', f.read(12))
                self._skip_array(inner, count)
        else:
            raise SystemExit(f'{self.fname}: unknown KV array element type {st}')


class Shard:
    def __init__(self, path, index):
        self.path = path
        self.index = index   # 1 based, as in the file name
        self.size = os.path.getsize(path)
        with open(path, 'rb', buffering=1 << 20) as f:
            self.header = Header(f)


class Tensor:
    """Where a tensor of a (split) GGUF lives: its shard, shape and type, and the file offset of its data in the shard."""

    def __init__(self, name, shard, dims, ttype, offset):
        self.name = name
        self.shard = shard
        self.dims = dims
        self.ttype = ttype
        self.offset = shard.header.data_start + offset

    @property
    def type_info(self):
        return type_info(self.name, self.ttype)

    @property
    def nbytes(self):
        return tensor_nbytes(self.name, self.dims, self.ttype)

    def slice_bytes(self, axis):
        return tensor_slice_bytes(self.name, self.dims, self.ttype, axis)


class Model:
    """A GGUF file or a split set: the headers of all shards and one name -> Tensor map."""

    def __init__(self, path):
        m = SPLIT_RE.match(path)
        if m:
            count = int(m.group(3))
            paths = [f'{m.group(1)}-{i:05d}-of-{count:05d}.gguf' for i in range(1, count + 1)]
            missing = [p for p in paths if not os.path.isfile(p)]
            if missing:
                raise SystemExit('the split set is incomplete, missing: ' + ', '.join(missing))
        else:
            paths = [path]
        self.shards = [Shard(p, i + 1) for i, p in enumerate(paths)]
        self.tensors = {}
        for sh in self.shards:
            for name, (dims, ttype, offset) in sh.header.tensors.items():
                if name in self.tensors:
                    raise SystemExit(f'{name} is in {self.tensors[name].shard.path} and in {sh.path}')
                self.tensors[name] = Tensor(name, sh, dims, ttype, offset)
        for sh in self.shards:
            count = sh.header.kv.get('split.count')
            if count is not None and count != len(self.shards):
                warn(f'{sh.path}: split.count is {count}, the file name says {len(self.shards)} shards')
        total = self.shards[0].header.kv.get('split.tensors.count')
        if total is not None and total != len(self.tensors):
            warn(f'split.tensors.count is {total}, the shards list {len(self.tensors)} tensors')

    def need(self, name):
        t = self.tensors.get(name)
        if t is None:
            raise SystemExit(f'missing {name}')
        return t


Item = collections.namedtuple('Item', 'tensor sb')   # sb: bytes of one slice (one expert / router row / bias entry)
Layer = collections.namedtuple('Layer', 'il order items')


def warn(msg):
    print(f'warning: {msg}', file=sys.stderr, flush=True)


def label(name):
    return LABELS.get(name.split('.', 2)[2].rsplit('.', 1)[0], name)


def build_plan(model, orders):
    """for each layer of the order: the tensors to permute and their slice sizes, all checked against the order"""
    plan = []
    for il in sorted(orders):
        order = orders[il]
        n_expert = len(order)
        if sorted(order) != list(range(n_expert)):
            raise SystemExit(f'layer {il}: the order is not a permutation of 0..{n_expert - 1}')
        items = []
        for suffix in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
            t = model.need(f'blk.{il}.{suffix}.weight')
            if len(t.dims) != 3 or t.dims[2] != n_expert:
                raise SystemExit(f'{t.name}: dims {t.dims}, the order has {n_expert} experts')
            items.append(Item(t, t.slice_bytes(2)))
        t = model.need(f'blk.{il}.ffn_gate_inp.weight')
        if len(t.dims) != 2 or t.dims[1] != n_expert:
            raise SystemExit(f'{t.name}: dims {t.dims}, expected [n_embd, {n_expert}]')
        items.append(Item(t, t.slice_bytes(1)))
        for suffix in ('exp_probs_b', 'exp_probs_b_vl'):
            t = model.tensors.get(f'blk.{il}.{suffix}.bias')
            if t is None:
                continue
            if len(t.dims) != 1 or t.dims[0] != n_expert or t.type_info[1] != 1:
                raise SystemExit(f'{t.name}: dims {t.dims} type {t.type_info[0]}, expected [{n_expert}] float')
            items.append(Item(t, t.slice_bytes(0)))
        for it in items:
            end = it.tensor.offset + it.tensor.nbytes
            if end > it.tensor.shard.size:
                raise SystemExit(f'{it.tensor.name}: ends at {end}, beyond the end of {it.tensor.shard.path} ({it.tensor.shard.size})')
        plan.append(Layer(il, order, items))
    return plan


def check_leftovers(model, plan):
    """per-expert tensors the plan does not cover: a hash routing table (tid2eid) holds expert ids and would need a remap"""
    handled = {it.tensor.name for lay in plan for it in lay.items}
    left = []
    for name in sorted(model.tensors):
        m = re.match(r'blk\.(\d+)\.(.+)$', name)
        if m and name not in handled and any(k in m.group(2) for k in ('_exps', 'exp_probs', 'gate_inp', 'tid2eid')):
            left.append(name)
    for name in left:
        if 'tid2eid' in name:
            raise SystemExit(f'{name}: a hash routing table holds expert ids, permuting the experts needs a remap this tool does not do')
    if left:
        warn(f'{len(left)} expert related tensors are not permuted (their layer is not in the order, or they are not known): '
             + ', '.join(left[:6]) + (' ...' if len(left) > 6 else ''))


def print_plan(model, plan, journal):
    h0 = model.shards[0].header
    print(f'{len(model.shards)} shard(s), {len(model.tensors)} tensors, arch {h0.kv.get("general.architecture", "?")}')
    for sh in model.shards:
        print(f'  shard {sh.index}: {sh.path}: {sh.size} B, GGUF v{sh.header.version}, {len(sh.header.tensors)} tensors, '
              f'data at {sh.header.data_start}, alignment {sh.header.alignment}')
    per_shard = collections.defaultdict(lambda: [0, 0])
    for lay in plan:
        cols = []
        for it in lay.items:
            t = it.tensor
            cols.append(f'{label(t.name)} s{t.shard.index} {t.type_info[0]} {t.nbytes} B')
            per_shard[t.shard.index][0] += 1
            per_shard[t.shard.index][1] += t.nbytes
        state = journal.get(str(lay.il))
        print(f'layer {lay.il:2d}: ' + ' | '.join(cols) + f' | order {len(lay.order)} ok' + (f' | journal: {state}' if state else ''))
    sl = sorted({(label(it.tensor.name), it.sb) for lay in plan for it in lay.items})
    print('bytes per slice: ' + ', '.join(f'{k} {v}' for k, v in sl))
    for idx in sorted(per_shard):
        print(f'  shard {idx}: {per_shard[idx][0]} tensors to permute, {per_shard[idx][1] / 1e9:.2f} GB')
    print(f'{len(plan)} layers, {sum(len(lay.items) for lay in plan)} tensors; '
          f'{sum(it.sb * len(lay.order) for lay in plan for it in lay.items) / 1e9:.1f} GB to permute')


def pread_exact(fd, size, offset):
    buf = os.pread(fd, size, offset)
    if len(buf) == size:
        return buf
    parts, got = [buf], len(buf)
    while got < size:
        more = os.pread(fd, size - got, offset + got)
        if not more:
            raise SystemExit(f'unexpected end of file at {offset + got}')
        parts.append(more)
        got += len(more)
    return b''.join(parts)


def pwrite_all(fd, data, offset):
    view = memoryview(data)
    while len(view):
        n = os.pwrite(fd, view, offset)
        view = view[n:]
        offset += n


def permute_slices(fd, base, slice_size, order, crcs=None):
    """In place: new slot j <- old slot order[j]. Cycle following with one slice buffer per cycle.
    crcs[i] gets the crc32 of the original content of slot i (every slot is read once; a fixed slot only if crcs is given)."""
    n = len(order)
    done = [False] * n
    moved = 0
    for start in range(n):
        if done[start]:
            continue
        if order[start] == start:
            done[start] = True
            if crcs is not None:
                crcs[start] = zlib.crc32(pread_exact(fd, slice_size, base + start * slice_size))
            continue
        saved = pread_exact(fd, slice_size, base + start * slice_size)
        if crcs is not None:
            crcs[start] = zlib.crc32(saved)
        # the cycle: start <- order[start] <- order[order[start]] <- ...
        j = start
        while True:
            src = order[j]
            if src == start:
                pwrite_all(fd, saved, base + j * slice_size)
                done[j] = True
                moved += 1
                break
            data = pread_exact(fd, slice_size, base + src * slice_size)
            if crcs is not None:
                crcs[src] = zlib.crc32(data)
            pwrite_all(fd, data, base + j * slice_size)
            done[j] = True
            moved += 1
            j = src
    return moved


def bad_slots(fd, base, slice_size, order, crcs):
    """the slots whose content is not the original slice order[j] (order None: slot j must still hold slice j)"""
    bad = []
    for j in range(len(crcs)):
        src = j if order is None else order[j]
        if zlib.crc32(pread_exact(fd, slice_size, base + j * slice_size)) != crcs[src]:
            bad.append(j)
    return bad


def order_crc(order):
    return zlib.crc32(struct.pack('<%dI' % len(order), *order))


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def open_shards(plan, flags):
    fds = {}
    for lay in plan:
        for it in lay.items:
            sh = it.tensor.shard
            if sh.path not in fds:
                fds[sh.path] = os.open(sh.path, flags)
                if flags & os.O_RDWR:
                    try:
                        fcntl.flock(fds[sh.path], fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError as e:
                        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                            raise SystemExit(f'{sh.path} is locked by another process (a second permute?)')
                        warn(f'{sh.path}: cannot lock: {e}')
    return fds


def drop_cache(fd, offset, length):
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def run(model, plan, path):
    journal_path = path + '.permute.json'
    fp_path = path + '.permute.fp.json'
    journal = load_json(journal_path)
    fps = load_json(fp_path).get('layers', {})
    unchecked = []
    for lay in plan:
        state = journal.get(str(lay.il))
        if state == 'in progress':
            shards = sorted({it.tensor.shard.path for it in lay.items})
            raise SystemExit(f'layer {lay.il} was left half permuted (journal): restore its tensors from the download first, in {", ".join(shards)}')
        if state == 'done':
            rec = fps.get(str(lay.il))
            if rec is None:
                unchecked.append(lay.il)
            elif rec['order_crc'] != order_crc(lay.order):
                raise SystemExit(f'layer {lay.il} is journaled as done with another order than ORDER.json: it would be skipped, not permuted again')
    if unchecked:
        warn(f'layers {unchecked} are journaled as done without fingerprints: skipped, not checked')
    fds = open_shards(plan, os.O_RDWR)
    stop = []

    def on_signal(signum, frame):
        if stop:
            raise KeyboardInterrupt
        stop.append(signum)
        print(f'signal {signum}: the layer in progress is finished first, then the run stops (send it again to abort at once)', file=sys.stderr, flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)
    t0 = time.time()
    n_done = 0
    try:
        for lay in plan:
            key = str(lay.il)
            if journal.get(key) == 'done':
                continue
            if stop:
                print(f'stopped before layer {lay.il}; rerun to continue')
                return 0
            journal[key] = 'in progress'
            save_json(journal_path, journal)
            t1 = time.time()
            moved = 0
            crcs = {}
            for it in lay.items:
                crcs[it.tensor.name] = [0] * len(lay.order)
                moved += permute_slices(fds[it.tensor.shard.path], it.tensor.offset, it.sb, lay.order, crcs[it.tensor.name])
            for fd in {fds[it.tensor.shard.path] for it in lay.items}:
                os.fsync(fd)
            nbad = 0
            for it in lay.items:
                fd = fds[it.tensor.shard.path]
                drop_cache(fd, it.tensor.offset, it.sb * len(lay.order))
                bad = bad_slots(fd, it.tensor.offset, it.sb, lay.order, crcs[it.tensor.name])
                if bad:
                    nbad += len(bad)
                    print(f'layer {lay.il}: {it.tensor.name}: {len(bad)} slots do not hold their slice after the permute: {bad[:8]}', file=sys.stderr)
            if nbad:
                raise SystemExit(f'layer {lay.il}: verification failed, the journal keeps it as in progress: restore the tensors from the download')
            fps[key] = {'order_crc': order_crc(lay.order), 'crcs': crcs}
            save_json(fp_path, {'layers': fps})
            journal[key] = 'done'
            save_json(journal_path, journal)
            n_done += 1
            print(f'layer {lay.il}: {moved} slices moved and verified in {time.time() - t1:.1f} s', flush=True)
        print(f'done in {time.time() - t0:.0f} s ({n_done} layers permuted)')
        return 0
    finally:
        for fd in fds.values():
            os.close(fd)


def verify_only(model, plan, path):
    journal = load_json(path + '.permute.json')
    fps = load_json(path + '.permute.fp.json').get('layers', {})
    if not fps:
        print(f'no fingerprints ({path}.permute.fp.json): only the structure was checked')
        return 0
    fds = open_shards(plan, os.O_RDONLY)
    t0 = time.time()
    n_bad = 0
    n_ok = 0
    try:
        for lay in plan:
            key = str(lay.il)
            rec = fps.get(key)
            state = journal.get(key)
            if rec is None:
                print(f'layer {lay.il}: no fingerprints' + (f' (journal: {state})' if state else ''))
                continue
            if state == 'in progress':
                print(f'layer {lay.il}: FAIL the journal says in progress')
                n_bad += 1
                continue
            if rec['order_crc'] != order_crc(lay.order):
                print(f'layer {lay.il}: FAIL the order differs from the one the file was permuted with')
                n_bad += 1
                continue
            want = lay.order if state == 'done' else None
            bad = []
            for it in lay.items:
                crcs = rec['crcs'].get(it.tensor.name)
                if crcs is None or len(crcs) != len(lay.order):
                    bad.append(f'{it.tensor.name}: no fingerprints')
                    continue
                slots = bad_slots(fds[it.tensor.shard.path], it.tensor.offset, it.sb, want, crcs)
                if slots:
                    bad.append(f'{it.tensor.name}: {len(slots)} slots {slots[:8]}')
            if bad:
                n_bad += 1
                print(f'layer {lay.il}: FAIL ({"permuted" if want else "original"}) ' + '; '.join(bad))
            else:
                n_ok += 1
                print(f'layer {lay.il}: ok ({"permuted" if want else "original"})')
    finally:
        for fd in fds.values():
            os.close(fd)
    print(f'verified {n_ok} layers, {n_bad} failed, in {time.time() - t0:.0f} s')
    return 1 if n_bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument('file', metavar='FILE.gguf')
    ap.add_argument('order', metavar='ORDER.json')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='print the plan, write nothing')
    mode.add_argument('--verify-only', action='store_true', help='check the file against the fingerprints, read only')
    args = ap.parse_args()
    with open(args.order) as f:
        orders = {int(k): v for k, v in json.load(f).items()}
    model = Model(args.file)
    plan = build_plan(model, orders)
    check_leftovers(model, plan)
    base = model.shards[0].path   # the journal belongs to the set, whichever shard was given
    print_plan(model, plan, load_json(base + '.permute.json'))
    if args.dry_run:
        print('dry run: nothing was written')
        return 0
    if args.verify_only:
        return verify_only(model, plan, base)
    return run(model, plan, base)


if __name__ == '__main__':
    sys.exit(main())
