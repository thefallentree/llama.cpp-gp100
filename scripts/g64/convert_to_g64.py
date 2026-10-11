#!/usr/bin/env python3
"""Convert any GGUF llama.cpp can load into a GP100-servable Q4_1_G64 mix.

Two convert paths:

  quantize  (default)  llama-quantize --pure Q4_1_G64
                       from F16/BF16/F32 (preferred) or --allow-requantize
  refit                Q4_1 -> Q4_1_G64 without an F32 round-trip
                       (same path that built the Qwen3.8 27B pack)

Neither path is Qwen-specific. Tensors whose K is not divisible by 64 fall
back to Q4_1 (llama-quantize) or are left alone (refit). token_embd defaults
to Q4_1 so a Q4_1 draft can share the exact embedding; output defaults to
Q8_0. Stock ggml-org llama.cpp refuses type 43 — serve with
thefallentree/llama.cpp-gp100 (sm_60 / Tesla P100 only).

See docs/backend/gp100-g64.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gguf-py"))
import gguf  # noqa: E402

G64_TYPE_ID = 43
A16_QK = 32
A16_ROWS = 4
G64_K = 64
G64_ROWS = 8

# Substring matches. Do NOT put "output.weight" here — it is a suffix of
# attn_output.weight and would skip every attention-out projection.
SKIP_NAME_SUBSTR = (
    "ffn_gate_inp.weight",
    "ssm_conv1d",
    "shortconv.conv.weight",
    "mmproj",
)

def _arch(reader: gguf.GGUFReader) -> str:
    field = reader.fields.get("general.architecture")
    if field is None:
        return "unknown"
    raw = field.parts[-1].tobytes()
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")


def _kv_u32(reader: gguf.GGUFReader, key: str, default: int | None = None) -> int | None:
    field = reader.fields.get(key)
    if field is None:
        return default
    try:
        val = field.contents()
        if isinstance(val, (list, tuple)):
            val = val[0]
        return int(val)
    except Exception:  # noqa: BLE001
        return default


def load_imatrix(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    data = path.read_bytes()
    off = 0
    (n,) = struct.unpack_from("<i", data, off)
    off += 4
    for _ in range(n):
        (ln,) = struct.unpack_from("<i", data, off)
        off += 4
        name = data[off : off + ln].decode()
        off += ln
        (ncall,) = struct.unpack_from("<i", data, off)
        off += 4
        (nval,) = struct.unpack_from("<i", data, off)
        off += 4
        vals = np.frombuffer(data, dtype=np.float32, count=nval, offset=off).copy()
        off += 4 * nval
        if ncall > 0:
            vals /= ncall
        out[name] = vals
    return out


def dequant_q4_1(blk: np.ndarray) -> np.ndarray:
    d = blk[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    m = blk[:, 2:4].copy().view(np.float16).astype(np.float32)[:, 0]
    qs = blk[:, 4:20]
    w = np.empty((blk.shape[0], 32), np.float32)
    w[:, :16] = qs & 0x0F
    w[:, 16:] = qs >> 4
    return w * d[:, None] + m[:, None]


def fit_group(vals: np.ndarray, wts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vmin = vals.min(1)
    vmax = vals.max(1)
    d = np.maximum((vmax - vmin) / 15.0, 1e-8)
    m = vmin
    for _ in range(2):
        q = np.clip(np.rint((vals - m[:, None]) / d[:, None]), 0, 15)
        sw = wts.sum(1)
        swq = (wts * q).sum(1)
        swq2 = (wts * q * q).sum(1)
        swv = (wts * vals).sum(1)
        swqv = (wts * q * vals).sum(1)
        det = sw * swq2 - swq * swq
        ok = det > 1e-12
        nd = np.where(ok, (sw * swqv - swq * swv) / np.where(ok, det, 1), d)
        nm = np.where(ok, (swq2 * swv - swq * swqv) / np.where(ok, det, 1), m)
        d = np.where(nd > 1e-8, nd, d)
        m = nm
    q = np.clip(np.rint((vals - m[:, None]) / d[:, None]), 0, 15).astype(np.uint8)
    return d.astype(np.float16), m.astype(np.float16), q


def pack_g64(vals: np.ndarray, wts: np.ndarray) -> np.ndarray:
    ng = vals.shape[0]
    d16, m16, q = fit_group(vals, wts)
    out = np.empty((ng, 36), np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(ng, 2)
    out[:, 2:4] = m16.view(np.uint8).reshape(ng, 2)
    for h in range(2):
        sub = q[:, 32 * h : 32 * h + 32]
        out[:, 4 + 16 * h : 4 + 16 * h + 16] = sub[:, :16] | (sub[:, 16:] << 4)
    return out


def _shape_ne(t) -> tuple[int, ...]:
    # ReaderTensor.shape is the GGUF dim list: shape[0] == ggml ne[0] == K.
    return tuple(int(x) for x in t.shape)


def classify_tensor(t) -> dict:
    ne = _shape_ne(t)
    ne0 = ne[0] if ne else 0
    ne1 = ne[1] if len(ne) > 1 else 1
    ne2 = ne[2] if len(ne) > 2 else 1
    name = t.name
    typ = int(t.tensor_type)
    type_name = gguf.GGMLQuantizationType(typ).name
    is_weight = name.endswith("weight")
    n_dims = len(ne)
    skip = (
        (not is_weight)
        or n_dims < 2
        or name.endswith("_norm.weight")
        or name == "output.weight"
        or "token_embd" in name
        or "per_layer_token_embd" in name
        or any(s in name for s in SKIP_NAME_SUBSTR)
    )
    k_ok = n_dims >= 1 and ne0 % G64_K == 0
    rows_a16 = n_dims >= 2 and ne1 % G64_ROWS == 0
    a16 = (
        is_weight
        and n_dims >= 2
        and k_ok
        and rows_a16
        and ne2 == 1
        and ne0 % A16_QK == 0
        and ne1 % A16_ROWS == 0
    )
    if "token_embd" in name:
        role = "embd"
    elif name == "output.weight":
        role = "head"
    elif skip:
        role = "leave"
    elif typ == gguf.GGMLQuantizationType.Q4_1_G64:
        role = "already-g64"
    elif k_ok:
        role = "g64"
    else:
        role = "fallback-q4_1"
    return {
        "name": name,
        "type": type_name,
        "type_id": typ,
        "ne": ne,
        "role": role,
        "a16": a16,
        "k_div64": k_ok,
        "rows_div8": rows_a16,
    }


def inspect_gguf(path: Path) -> dict:
    reader = gguf.GGUFReader(str(path))
    tensors = [classify_tensor(t) for t in reader.tensors]
    by_role: dict[str, int] = {}
    by_type: dict[str, int] = {}
    a16_n = 0
    for row in tensors:
        by_role[row["role"]] = by_role.get(row["role"], 0) + 1
        by_type[row["type"]] = by_type.get(row["type"], 0) + 1
        if row["a16"] and row["role"] in ("g64", "already-g64"):
            a16_n += 1
    q41_body = sum(1 for r in tensors if r["role"] == "g64" and r["type"] == "Q4_1")
    g64_body = sum(1 for r in tensors if r["role"] == "already-g64")
    ctx = None
    arch = _arch(reader)
    for key in (
        f"{arch}.context_length",
        "qwen35.context_length",
        "llama.context_length",
        "general.context_length",
    ):
        ctx = _kv_u32(reader, key)
        if ctx:
            break
    return {
        "path": str(path),
        "arch": arch,
        "n_tensors": len(tensors),
        "by_role": by_role,
        "by_type": by_type,
        "a16_eligible": a16_n,
        "q4_1_convertible": q41_body,
        "already_g64": g64_body,
        "refit_ok": q41_body > 0 and g64_body == 0,
        "context_length": ctx,
        "tensors": tensors,
    }


def print_inspect(report: dict, verbose: bool = False) -> None:
    print(f"file   {report['path']}")
    print(f"arch   {report['arch']}")
    if report.get("context_length"):
        print(f"ctx    {report['context_length']}")
    print(f"types  {report['by_type']}")
    print(f"roles  {report['by_role']}")
    print(
        f"A16    {report['a16_eligible']} tensors meet "
        f"ne0%{G64_K}==0 and ne1%{G64_ROWS}==0 (fast sm_60 HFMA2 path)"
    )
    print(f"refit  {'yes' if report['refit_ok'] else 'no'} "
          f"({report['q4_1_convertible']} Q4_1 body tensors)")
    fallback = [t for t in report["tensors"] if t["role"] == "fallback-q4_1"]
    if fallback:
        print(f"note   {len(fallback)} weight(s) have K not divisible by 64 "
              "→ llama-quantize will keep them Q4_1")
        for t in fallback[:12]:
            print(f"         {t['name']} ne={t['ne']} {t['type']}")
        if len(fallback) > 12:
            print(f"         … {len(fallback) - 12} more")
    miss_a16 = [
        t for t in report["tensors"]
        if t["role"] in ("g64", "already-g64") and not t["a16"]
    ]
    if miss_a16:
        print(f"note   {len(miss_a16)} G64-shaped tensor(s) miss the A16 "
              "row constraint (ne1%8!=0 or ne2!=1) → dequant+cuBLAS on GPU")
        for t in miss_a16[:8]:
            print(f"         {t['name']} ne={t['ne']}")
    if verbose:
        for t in report["tensors"]:
            flag = "A16" if t["a16"] else "   "
            print(f"  {flag} {t['role']:14} {t['type']:12} {t['ne']}  {t['name']}")


def find_llama_quantize(explicit: Path | None) -> Path | None:
    if explicit:
        return explicit if explicit.is_file() else None
    env = os.environ.get("LLAMA_QUANTIZE")
    if env and Path(env).is_file():
        return Path(env)
    for cand in (
        REPO / "build" / "bin" / "llama-quantize",
        REPO / "build" / "bin" / "Release" / "llama-quantize",
        Path(shutil.which("llama-quantize") or ""),
    ):
        if cand and cand.is_file():
            return cand
    return None


def run_quantize(
    src: Path,
    dst: Path,
    quantize: Path,
    *,
    imatrix: Path | None,
    output_type: str,
    embd_type: str,
    allow_requantize: bool,
    threads: int,
    dry_run: bool,
) -> None:
    cmd = [
        str(quantize),
        "--pure",
        "--output-tensor-type", output_type,
        "--token-embedding-type", embd_type,
    ]
    if imatrix:
        cmd += ["--imatrix", str(imatrix)]
    if allow_requantize:
        cmd.append("--allow-requantize")
    if dry_run:
        cmd.append("--dry-run")
    cmd += [str(src), str(dst), "Q4_1_G64", str(threads)]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def run_refit(src: Path, dst: Path, imatrix: Path | None) -> None:
    im = load_imatrix(imatrix) if imatrix else {}
    reader = gguf.GGUFReader(str(src))
    writer = gguf.GGUFWriter(
        str(dst),
        arch=_arch(reader),
        use_temp_file=True,
    )
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if len(field.types) != 1:
            continue
        val = 42 if key == "general.file_type" else field.contents()
        writer.add_key_value(key, val, field.types[0])
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or len(field.types) == 1:
            continue
        writer.add_key_value(key, field.contents(), field.types[0], sub_type=field.types[-1])

    nconv = 0
    nskip = 0
    for t in reader.tensors:
        data = np.asarray(t.data)
        row = classify_tensor(t)
        if row["role"] == "g64" and t.tensor_type == gguf.GGMLQuantizationType.Q4_1:
            blk = data.reshape(-1, 20)
            vals = dequant_q4_1(blk).reshape(-1, 64)
            ne0 = int(row["ne"][0])
            wt = im.get(t.name)
            if wt is not None and wt.size == ne0:
                wts = np.tile(wt.astype(np.float32), vals.shape[0] * 64 // ne0).reshape(-1, 64)
                wts = np.maximum(wts, 1e-4 * wts.max() if wts.max() > 0 else 1.0)
            else:
                wts = np.ones_like(vals)
            byte_shape = tuple(data.shape[:-1]) + (ne0 // 64 * 36,)
            packed = pack_g64(vals, wts).reshape(byte_shape)
            writer.add_tensor(t.name, packed, raw_dtype=gguf.GGMLQuantizationType.Q4_1_G64)
            nconv += 1
        else:
            writer.add_tensor(t.name, data, raw_dtype=gguf.GGMLQuantizationType(int(t.tensor_type)))
            if row["role"] == "g64":
                nskip += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(
        f"refit {nconv} Q4_1 tensors → Q4_1_G64 (type {G64_TYPE_ID}); "
        f"left {nskip} non-Q4_1 body tensors; imatrix={'yes' if im else 'no'}"
    )


def draft_is_gemma_mtp(draft: Path | None) -> bool:
    """True for a Gemma 4 assistant / shared-KV MTP GGUF.

    Filename heuristics first (offline tests, missing files). If the path
    exists, the GGUF architecture is authoritative.
    """
    if draft is None:
        return False
    name = draft.name.lower()
    if "gemma4-assistant" in name or name.startswith("mtp-gemma") or "-mtp-gemma" in name:
        return True
    if draft.is_file():
        try:
            return _arch(gguf.GGUFReader(str(draft))) == "gemma4-assistant"
        except Exception:  # noqa: BLE001
            return False
    return "mtp" in name and "gemma" in name


def serve_argv(
    model: Path,
    *,
    draft: Path | None,
    mmproj: Path | None,
    ctx: int,
    ngram: bool,
    graph_slots: int,
    checkpoints: int,
    spec: str = "auto",
) -> list[str]:
    # Gemma 4 MTP shares the target KV across two schedulers. GPU sampling
    # 500s on tensor-split Gemma (vector underflow). LoopSpec flags are
    # wrong here: the assistant is not a DFlash drafter.
    use_mtp = spec == "mtp" or (spec == "auto" and draft_is_gemma_mtp(draft))
    cmd = [
        "./build/bin/llama-server",
        "--model", str(model),
        "--jinja",
    ]
    if not use_mtp:
        cmd += ["--backend-sampling"]
    cmd += [
        "--graph-slots", str(graph_slots),
        "--ctx-checkpoints", str(checkpoints),
        "--checkpoint-min-step", "0",
        "--slot-prompt-similarity", "0.05",
        "--device", "CUDA0,CUDA1",
        "--split-mode", "tensor",
        "--tensor-split", "1,1",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--ctx-size", str(ctx),
        "--flash-attn", "on",
    ]
    if draft:
        if use_mtp:
            cmd += [
                "--model-draft", str(draft),
                "--device-draft", "CUDA0,CUDA1",
                "--cache-type-k-draft", "q8_0",
                "--cache-type-v-draft", "q8_0",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "2",
                "--spec-draft-n-min", "0",
                "--spec-draft-p-min", "0",
            ]
        else:
            cmd += [
                "--model-draft", str(draft),
                "--device-draft", "CUDA1",
                "--cache-type-k-draft", "q4_0",
                "--cache-type-v-draft", "q4_0",
                "--spec-type", "ngram-mod,draft-dflash" if ngram else "draft-dflash",
                "--spec-draft-n-max", "7",
                "--spec-draft-n-min", "0",
            ]
            if ngram:
                cmd += [
                    "--spec-ngram-mod-n-min", "7",
                    "--spec-ngram-mod-n-max", "7",
                ]
    if mmproj:
        cmd += ["--mmproj", str(mmproj)]
    return cmd


def cmd_inspect(args: argparse.Namespace) -> int:
    report = inspect_gguf(Path(args.src))
    if args.json:
        out = dict(report)
        if not args.verbose:
            out.pop("tensors", None)
        json.dump(out, sys.stdout, indent=2)
        print()
    else:
        print_inspect(report, verbose=args.verbose)
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    src = Path(args.src)
    dst = Path(args.dst)
    report = inspect_gguf(src)
    print_inspect(report)
    if report["already_g64"] and not args.force:
        print("already contains Q4_1_G64 tensors; pass --force to rewrite", file=sys.stderr)
        return 2
    mode = args.mode
    if mode == "auto":
        mode = "refit" if report["refit_ok"] and not args.allow_requantize else "quantize"
        print(f"mode   auto → {mode}")
    if mode == "refit":
        if not report["refit_ok"] and not args.force:
            print("refit needs a Q4_1 body; use --mode quantize --allow-requantize", file=sys.stderr)
            return 2
        if args.dry_run:
            print("dry-run: would refit Q4_1 body tensors in place of llama-quantize")
            return 0
        run_refit(src, dst, Path(args.imatrix) if args.imatrix else None)
    else:
        quantize = find_llama_quantize(Path(args.quantize) if args.quantize else None)
        if quantize is None:
            print(
                "llama-quantize not found. Build this fork (sm_60) or pass "
                "--quantize /path/to/llama-quantize. Stock ggml-org builds "
                "do not know Q4_1_G64.",
                file=sys.stderr,
            )
            return 2
        src_quantized = any(
            t["type"] not in ("F32", "F16", "BF16")
            for t in report["tensors"]
            if t["role"] == "g64"
        )
        allow = args.allow_requantize or src_quantized
        if src_quantized and not args.allow_requantize:
            print("note   source body is already quantized; enabling --allow-requantize "
                  "(quality is worse than F16→G64). Prefer a F16/BF16 GGUF.")
        run_quantize(
            src,
            dst,
            quantize,
            imatrix=Path(args.imatrix) if args.imatrix else None,
            output_type=args.output_tensor_type,
            embd_type=args.token_embedding_type,
            allow_requantize=allow,
            threads=args.threads,
            dry_run=args.dry_run,
        )
    if not args.dry_run and dst.is_file():
        print()
        print_inspect(inspect_gguf(dst))
        print()
        print("serve:")
        print(" ", " ".join(serve_argv(
            dst,
            draft=Path(args.draft) if args.draft else None,
            mmproj=Path(args.mmproj) if args.mmproj else None,
            ctx=args.ctx,
            ngram=not args.no_ngram,
            graph_slots=args.graph_slots,
            checkpoints=args.ctx_checkpoints,
            spec=args.spec,
        )))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    cmd = serve_argv(
        Path(args.model),
        draft=Path(args.draft) if args.draft else None,
        mmproj=Path(args.mmproj) if args.mmproj else None,
        ctx=args.ctx,
        ngram=not args.no_ngram,
        graph_slots=args.graph_slots,
        checkpoints=args.ctx_checkpoints,
        spec=args.spec,
    )
    print(" ".join(cmd))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ins = sub.add_parser("inspect", help="classify tensors / A16 eligibility")
    ins.add_argument("src")
    ins.add_argument("--verbose", action="store_true")
    ins.add_argument("--json", action="store_true")
    ins.set_defaults(func=cmd_inspect)

    conv = sub.add_parser("convert", help="write a Q4_1_G64 mix")
    conv.add_argument("src")
    conv.add_argument("dst")
    conv.add_argument("--mode", choices=("auto", "quantize", "refit"), default="auto")
    conv.add_argument("--quantize", type=Path, default=None, help="llama-quantize binary")
    conv.add_argument("--imatrix", type=Path, default=None)
    conv.add_argument("--output-tensor-type", default="q8_0")
    conv.add_argument("--token-embedding-type", default="q4_1")
    conv.add_argument("--allow-requantize", action="store_true")
    conv.add_argument("--force", action="store_true")
    conv.add_argument("--dry-run", action="store_true")
    conv.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 8))
    conv.add_argument("--draft", type=Path, default=None)
    conv.add_argument("--mmproj", type=Path, default=None)
    conv.add_argument("--ctx", type=int, default=131072)
    conv.add_argument("--graph-slots", type=int, default=8)
    conv.add_argument("--ctx-checkpoints", type=int, default=8)
    conv.add_argument("--no-ngram", action="store_true")
    conv.add_argument(
        "--spec",
        choices=("auto", "loopspec", "mtp"),
        default="auto",
        help="auto: gemma4-assistant draft → draft-mtp, else LoopSpec",
    )
    conv.set_defaults(func=cmd_convert)

    srv = sub.add_parser("serve", help="print a generic llama-server command")
    srv.add_argument("model")
    srv.add_argument("--draft", type=Path, default=None)
    srv.add_argument("--mmproj", type=Path, default=None)
    srv.add_argument("--ctx", type=int, default=131072)
    srv.add_argument("--graph-slots", type=int, default=8)
    srv.add_argument("--ctx-checkpoints", type=int, default=8)
    srv.add_argument("--no-ngram", action="store_true")
    srv.add_argument(
        "--spec",
        choices=("auto", "loopspec", "mtp"),
        default="auto",
        help="auto: gemma4-assistant draft → draft-mtp, else LoopSpec",
    )
    srv.set_defaults(func=cmd_serve)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
