#!/usr/bin/env python3
"""ds41_ep_shard.py - write the shard file of one node of an expert-parallel partition (src/llama-ep.h).

Reads the model (a split GGUF, first file) and the partition file, and writes for the node the experts it owns, in the node's
order, layer by layer as gate, up, down (each tensor on a 4096 boundary) after a 4096-byte header "DS41EPSH" + the shard id.
The node's rpc-server reads the file from its own disk (GGML_RPC_FILE_ROOT) - the weights never travel over the network.
Plain python (no numpy, no gguf package); reads each expert slice with one pread and writes sequentially.

usage: ds41_ep_shard.py --partition P.part --node K --gguf FIRST.gguf -o /path/shard.bin|- [--layers N]
       (-o -: the shard goes to stdout, for `... | ssh pve-N 'cat > /mnt/gpuwork/models/ep/shard.bin'`)
       (--layers N writes only layers 0..N-1: the shard of a reduced-layer test; the loader's layer count must match)
"""
import argparse, os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds41_gguf

def load_partition(path):
    p = {'ids': {}, 'shard': {}}
    for line in open(path):
        t = line.split()
        if not t or t[0].startswith('#'):
            continue
        if t[0] in ('nodes', 'experts', 'layers'):
            p[t[0]] = int(t[1])
        elif t[0] == 'slice':
            p['slice'] = [int(x) for x in t[1:4]]
        elif t[0] == 'shard':
            p['shard'][int(t[1])] = int(t[2], 16)
        elif t[0] == 'L':
            p['ids'][(int(t[1]), int(t[2]))] = [int(x) for x in t[3:]]
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partition', required=True)
    ap.add_argument('--node', type=int, required=True)
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--layers', type=int)
    ap.add_argument('-o', required=True)
    a = ap.parse_args()
    p = load_partition(a.partition)
    assert 1 <= a.node < p['nodes'], 'node 0 is the head: it gathers its experts from the model file, it has no shard file'
    n_layer = a.layers or p['layers']
    t = ds41_gguf.read_split(a.gguf)
    kinds = ['gate', 'up', 'down']
    # layout must be that of llama_ep_partition::shard_offset: all p['layers'] layers are laid out (offsets of layer L do not depend
    # on later layers, so a shard of the first N layers is a prefix of the full one)
    def align(x):
        return (x + 4095) // 4096 * 4096
    srcs = {}
    for L in range(n_layer):
        for kd, kn in enumerate(kinds):
            ti = t['blk.%d.ffn_%s_exps.weight' % (L, kn)]
            n_exp = ti['dims'][-1]
            assert n_exp == p['experts'] and ti['nbytes'] // n_exp == p['slice'][kd], \
                'blk.%d %s: %d experts of %s bytes, the partition says %d of %d' % (L, kn, n_exp, ti['nbytes'] // n_exp, p['experts'], p['slice'][kd])
            srcs[(L, kd)] = ti
    fds = {}
    def fd_of(path):
        if path not in fds:
            fds[path] = os.open(path, os.O_RDONLY)
        return fds[path]
    out = sys.stdout.buffer if a.o == '-' else open(a.o, 'wb')   # '-': stream to a pipe (ssh pve-N 'cat > file'): no disk needed here
    hdr = b'DS41EPSH' + struct.pack('<Q', p['shard'][a.node])
    out.write(hdr + b'\0' * (4096 - len(hdr)))
    t0 = time.time()
    total = 4096
    for L in range(n_layer):
        ids = p['ids'][(L, a.node)]
        for kd in range(3):
            ti = srcs[(L, kd)]
            slc = p['slice'][kd]
            fd = fd_of(ti['file'])
            n = 0
            for e in ids:
                buf = os.pread(fd, slc, ti['offset'] + e * slc)
                assert len(buf) == slc, 'short read'
                out.write(buf)
                n += slc
            pad = align(n) - n
            out.write(b'\0' * pad)
            total += n + pad
        if L % 4 == 3 or L == n_layer - 1:
            print('layer %d done: %.1f GB, %.0f s' % (L, total / 1e9, time.time() - t0), flush=True, file=sys.stderr if a.o == '-' else sys.stdout)
    out.flush()
    if a.o != '-':
        out.close()
    print('wrote %s: %d bytes' % (a.o, total), file=sys.stderr if a.o == '-' else sys.stdout)

main()
