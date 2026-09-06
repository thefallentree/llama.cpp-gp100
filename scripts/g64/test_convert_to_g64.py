#!/usr/bin/env python3
"""Offline tests for convert_to_g64 (no GPU, no 27B rewrite)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_to_g64 as g64  # noqa: E402


def test_pack_roundtrip() -> None:
    rng = np.random.default_rng(0)
    vals = rng.standard_normal((8, 64)).astype(np.float32)
    wts = np.ones_like(vals)
    packed = g64.pack_g64(vals, wts)
    assert packed.shape == (8, 36)
    # dequant one group with the G64 layout: dm + two Q4_1 halves
    d = packed[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    m = packed[:, 2:4].copy().view(np.float16).astype(np.float32)[:, 0]
    rec = np.empty((8, 64), np.float32)
    for h in range(2):
        qs = packed[:, 4 + 16 * h : 4 + 16 * h + 16]
        rec[:, 32 * h : 32 * h + 16] = qs & 0x0F
        rec[:, 32 * h + 16 : 32 * h + 32] = qs >> 4
    rec = rec * d[:, None] + m[:, None]
    err = float(np.max(np.abs(rec - vals)))
    # 4-bit affine over 64 weights; random Gaussian should land under 1.0
    assert err < 1.0, err


def test_q4_1_then_g64_roundtrip() -> None:
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((4, 64)).astype(np.float32)
    # fake two Q4_1 blocks per 64-group
    blk = np.zeros((8, 20), np.uint8)
    for i, half in enumerate(raw.reshape(8, 32)):
        vmin, vmax = float(half.min()), float(half.max())
        d = max((vmax - vmin) / 15.0, 1e-8)
        m = vmin
        q = np.clip(np.rint((half - m) / d), 0, 15).astype(np.uint8)
        blk[i, 0:2] = np.array([d], dtype=np.float16).view(np.uint8)
        blk[i, 2:4] = np.array([m], dtype=np.float16).view(np.uint8)
        blk[i, 4:20] = q[:16] | (q[16:] << 4)
    vals = g64.dequant_q4_1(blk).reshape(-1, 64)
    packed = g64.pack_g64(vals, np.ones_like(vals))
    assert packed.shape == (4, 36)
    assert packed.dtype == np.uint8


class _T:
    def __init__(self, name, shape, type_id=2):  # Q4_1 = 2
        self.name = name
        self.shape = shape
        self.tensor_type = type_id


def test_attn_output_is_g64_not_skipped() -> None:
    row = g64.classify_tensor(_T("blk.3.attn_output.weight", (6144, 5120)))
    assert row["role"] == "g64", row
    assert row["a16"] is True
    head = g64.classify_tensor(_T("output.weight", (5120, 248320), type_id=8))
    assert head["role"] == "head"
    embd = g64.classify_tensor(_T("token_embd.weight", (5120, 248320)))
    assert embd["role"] == "embd"
    odd = g64.classify_tensor(_T("blk.0.weird.weight", (48, 5120)))
    assert odd["role"] == "fallback-q4_1"


def test_serve_argv() -> None:
    cmd = g64.serve_argv(
        Path("model.gguf"),
        draft=Path("draft.gguf"),
        mmproj=None,
        ctx=8192,
        ngram=True,
        graph_slots=8,
        checkpoints=8,
    )
    joined = " ".join(cmd)
    assert "--model model.gguf" in joined
    assert "--model-draft draft.gguf" in joined
    assert "--spec-ngram-mod-n-min 7" in joined
    assert "--device-draft CUDA1" in joined
    assert "Q4_K" not in joined


def main() -> int:
    test_pack_roundtrip()
    test_q4_1_then_g64_roundtrip()
    test_attn_output_is_g64_not_skipped()
    test_serve_argv()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
