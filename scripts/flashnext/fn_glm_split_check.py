#!/usr/bin/env python3
"""Fingerprints of expert slices straight from a GGUF file, in the lines the engine prints with
LLAMA_EXPERT_SPLIT_CHECK=1 (llama-model.cpp, after the data is loaded): per MoE layer and per ffn_{up,gate,down}_exps,
the file's experts 0 and HOT-1 as the hot tensor's slices [0] and [HOT-1], the experts HOT and N-1 as the cold
tensor's slices [0] and [N-HOT-1]. Pure python (no numpy): it runs on a hypervisor next to the file. Compare with
`diff <(grep split-check engine.log | sort) <(sort this_output)`.
usage: fn_glm_split_check.py FILE.gguf HOT [FIRST_LAYER-LAST_LAYER]     (default: every layer with expert tensors)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fn_permute_inplace import Header  # noqa: E402


def fnv(b):
    h = 0xcbf29ce484222325
    for x in b:
        h = ((h ^ x) * 0x100000001b3) & 0xffffffffffffffff
    return h


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    path, hot = sys.argv[1], int(sys.argv[2])
    lo, hi = 0, 10**6
    if len(sys.argv) > 3:
        lo, hi = (int(x) for x in sys.argv[3].split('-'))
    with open(path, 'rb') as f:
        h = Header(f)
        names = sorted(n for n in h.tensors if re.match(r'blk\.\d+\.ffn_(up|gate|down)_exps\.weight$', n))
        for name in names:
            il = int(name.split('.')[1])
            if il < lo or il > hi:
                continue
            dims, _, off = h.tensors[name]
            n = dims[2]
            sl = h.slice_bytes(name, 2)
            base = h.data_start + off
            cold_name = name.replace('_exps.weight', '_exps_cold.weight')
            for e, label, idx in ((0, name, 0), (hot - 1, name, hot - 1), (hot, cold_name, 0), (n - 1, cold_name, n - hot - 1)):
                f.seek(base + e * sl)
                b = f.read(sl)
                print(f'split-check: {label}[{idx}] head {fnv(b[:4096]):016x} tail {fnv(b[-4096:]):016x} sum {sum(b)}')
            sys.stdout.flush()


if __name__ == '__main__':
    main()
