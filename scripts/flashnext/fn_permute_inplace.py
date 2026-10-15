#!/usr/bin/env python3
"""Reorder the routed experts of a MoE GGUF in place, layer by layer, so that each layer's experts come hot first
(the order the engine's hot/cold split cuts at load, LLAMA_EXPERT_HOT). Nothing is copied: the expert slices of
blk.L.ffn_{gate,up,down}_exps.weight, the rows of blk.L.ffn_gate_inp.weight and the entries of blk.L.exp_probs_b.bias
are permuted with one slice buffer (cycle following), so a 110 GB file needs no second copy and no extra disk.
Pure python (no numpy): it runs on a hypervisor with the file.

usage: fn_permute_inplace.py FILE.gguf ORDER.json [--dry-run] [--verify-only]
ORDER.json: {"3": [old expert ids in the new order], "4": [...], ...} from fn_glm_order.py (imatrix counts).
A journal FILE.gguf.permute.json records the layers done; rerunning skips them (a crash mid-layer leaves that layer
half permuted: the journal says 'in progress', restore from the download then).
"""
import json
import os
import struct
import sys
import time

# ggml type -> (block size, type size in bytes)
GGML_TYPES = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20), 8: ('Q8_0', 32, 34),
              10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144), 13: ('Q5_K', 256, 176),
              14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292), 23: ('IQ4_XS', 256, 136), 30: ('BF16', 1, 2),
              16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74), 20: ('IQ2_S', 256, 82), 19: ('IQ1_S', 256, 50),
              29: ('IQ1_M', 256, 56), 21: ('IQ3_S', 256, 110), 18: ('IQ3_XXS', 256, 98)}
KV_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}   # fixed-size value types


class Header:
    """The GGUF header: KV pairs (skipped) and the tensor infos, plus the data section offset."""

    def __init__(self, f):
        self.f = f
        f.seek(0)
        magic = f.read(4)
        if magic != b'GGUF':
            raise SystemExit(f'not a GGUF file: {magic!r}')
        self.version, = struct.unpack('<I', f.read(4))
        if self.version < 3:
            raise SystemExit(f'GGUF v{self.version}: only v3 is handled')
        n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
        self.alignment = 32
        for _ in range(n_kv):
            key = self._str()
            vt, = struct.unpack('<I', f.read(4))
            value = self._value(vt)
            if key == 'general.alignment':
                self.alignment = value
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            n_dims, = struct.unpack('<I', f.read(4))
            dims = list(struct.unpack('<' + 'Q' * n_dims, f.read(8 * n_dims)))
            ttype, offset = struct.unpack('<IQ', f.read(12))
            self.tensors[name] = (dims, ttype, offset)
        end = f.tell()
        self.data_start = (end + self.alignment - 1) // self.alignment * self.alignment

    def _str(self):
        n, = struct.unpack('<Q', self.f.read(8))
        return self.f.read(n).decode('utf-8', 'replace')

    def _value(self, vt):
        f = self.f
        if vt == 8:
            return self._str()
        if vt == 9:
            st, n = struct.unpack('<IQ', f.read(12))
            v = None
            for _ in range(n):
                v = self._value(st)
            return v
        size = KV_SIZES[vt]
        raw = f.read(size)
        if vt == 4:
            return struct.unpack('<I', raw)[0]
        if vt == 10:
            return struct.unpack('<Q', raw)[0]
        return raw

    def nbytes(self, name):
        dims, ttype, _ = self.tensors[name]
        tname, bs, ts = GGML_TYPES[ttype]
        if dims[0] % bs:
            raise SystemExit(f'{name}: ne0 {dims[0]} is not a multiple of the {tname} block {bs}')
        n = dims[0] // bs * ts
        for d in dims[1:]:
            n *= d
        return n

    def slice_bytes(self, name, axis):
        """bytes of one slice along `axis` (the last dims after it must be 1)"""
        dims, _, _ = self.tensors[name]
        total = self.nbytes(name)
        for d in dims[axis + 1:]:
            if d != 1:
                raise SystemExit(f'{name}: dims after axis {axis} are not 1: {dims}')
        return total // dims[axis]


def permute_slices(fd, base, slice_size, order, buf_limit=64 << 20):
    """In place: new slot j <- old slot order[j]. Cycle following with one slice buffer per cycle."""
    n = len(order)
    assert sorted(order) == list(range(n)), 'order is not a permutation'
    done = [False] * n
    moved = 0
    for start in range(n):
        if done[start] or order[start] == start:
            done[start] = True
            continue
        # the cycle: start <- order[start] <- order[order[start]] <- ...
        saved = os.pread(fd, slice_size, base + start * slice_size)
        j = start
        while True:
            src = order[j]
            if src == start:
                os.pwrite(fd, saved, base + j * slice_size)
                done[j] = True
                moved += 1
                break
            data = os.pread(fd, slice_size, base + src * slice_size)
            os.pwrite(fd, data, base + j * slice_size)
            done[j] = True
            moved += 1
            j = src
    return moved


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    if len(args) != 2:
        print(__doc__)
        sys.exit(2)
    path, order_path = args
    dry = '--dry-run' in flags
    with open(order_path) as f:
        orders = {int(k): v for k, v in json.load(f).items()}
    journal_path = path + '.permute.json'
    journal = {}
    if os.path.exists(journal_path):
        with open(journal_path) as f:
            journal = json.load(f)
    with open(path, 'rb') as f:
        h = Header(f)
    size = os.path.getsize(path)
    print(f'{path}: GGUF v{h.version}, {len(h.tensors)} tensors, data at {h.data_start}, alignment {h.alignment}')
    plan = []
    for il in sorted(orders):
        order = orders[il]
        n_expert = len(order)
        items = []
        for suffix in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
            name = f'blk.{il}.{suffix}.weight'
            if name not in h.tensors:
                raise SystemExit(f'missing {name}')
            dims, ttype, off = h.tensors[name]
            if dims[2] != n_expert:
                raise SystemExit(f'{name}: {dims[2]} experts, the order has {n_expert}')
            items.append((name, h.data_start + off, h.slice_bytes(name, 2)))
        name = f'blk.{il}.ffn_gate_inp.weight'
        dims, ttype, off = h.tensors[name]
        if dims[1] != n_expert or ttype != 0:
            raise SystemExit(f'{name}: dims {dims} type {ttype}, expected [n_embd, {n_expert}] F32')
        items.append((name, h.data_start + off, h.slice_bytes(name, 1)))
        name = f'blk.{il}.exp_probs_b.bias'
        if name in h.tensors:
            dims, ttype, off = h.tensors[name]
            if dims[0] != n_expert or ttype != 0:
                raise SystemExit(f'{name}: dims {dims} type {ttype}')
            items.append((name, h.data_start + off, 4))
        for name, base, sb in items:
            if base + sb * n_expert > size:
                raise SystemExit(f'{name}: beyond the file end')
        plan.append((il, order, items))
    print(f'{len(plan)} layers, {sum(len(p[2]) for p in plan)} tensors; '
          f'{sum(it[2] * len(p[1]) for p in plan for it in p[2]) / 1e9:.1f} GB to permute')
    if dry:
        for il, order, items in plan[:2]:
            print(f'  layer {il}: ' + ', '.join(f'{n.split(".")[2]} {sb} B/slice' for n, _, sb in items) + f'; first new slots <- old {order[:8]}')
        return
    fd = os.open(path, os.O_RDWR)
    t0 = time.time()
    for il, order, items in plan:
        key = str(il)
        if journal.get(key) == 'done':
            continue
        if journal.get(key) == 'in progress':
            raise SystemExit(f'layer {il} was left half permuted (journal): restore the file first')
        journal[key] = 'in progress'
        with open(journal_path, 'w') as f:
            json.dump(journal, f)
        t1 = time.time()
        moved = 0
        for name, base, sb in items:
            moved += permute_slices(fd, base, sb, order)
        os.fsync(fd)
        journal[key] = 'done'
        with open(journal_path, 'w') as f:
            json.dump(journal, f)
        print(f'layer {il}: {moved} slices moved in {time.time() - t1:.1f} s', flush=True)
    os.close(fd)
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
