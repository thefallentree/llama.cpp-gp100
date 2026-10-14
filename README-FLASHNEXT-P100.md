# Qwen3.8-Flash-Next on two Tesla P100s

This fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (branch `flashnext-p100-wip`) serves
**Qwen3.8-Flash-Next** — a 125B-parameter MoE with 6B active parameters and a 51B n-gram embedding table,
48 hybrid GDN/sparse-attention layers, 512 experts per layer with 10 used, plus an MTP draft block — on
**two Tesla P100 (16 GB, sm_60, 2016)** at 75–90 tokens/s of decode, 300–420 tokens/s of prefill and a 204,800-token
context, from the 2-bit expert release `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (Q2_0 experts, ~40 GB).
It is the production coding model behind an OpenAI-compatible `llama-server` on that box, used through the pi
coding agent and chat front ends.

Everything upstream llama.cpp does still works; the additions are a decode engine for exactly this model family
(architecture `qwen4exp`) on exactly this kind of hardware. They are not upstream-shaped and are not meant as a PR.

## What is different from upstream

Pascal has no tensor cores, no dp4a, 732 GB/s of HBM per card and 10 GB/s of PCIe between the cards. The model
does not fit one card. The branch adds, on top of upstream master (rebased regularly; the base is in `git log`):

| area | what | where |
|---|---|---|
| tensor-parallel decode | both cards run every layer on their half of the weights through the Meta backend (`-sm tensor`); the AllReduce is a flag-in-data exchange over PCIe peer memory fused with the hyper-connection epilogue; a decode round is one captured CUDA graph "window" of up to 8 tokens per card (~1,340 kernels, 19 ms) | `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-cuda/fn-engine.cu` |
| fused sm_60 engine | planar 8-bit (P8), 4-bit (P4) and exact 4-bit (Q4_16) weight layouts with mat-vecs that reach the memory bandwidth at 1–8 columns; fused MoE up/gate/SwiGLU/down over the hot experts; fused norms, GDN input side, routing | `ggml/src/ggml-cuda/fn-engine.cu`, `fn-engine.cuh`, `fattn-sel.cu` |
| expert cache | a per-layer budget of experts resident in VRAM (`LLAMA_EXPERT_HOT`, per-layer profile `LLAMA_EXPERT_HOT_LAYERS`), the rest in pinned host memory; slices are swapped between windows from routing counts; spare slots so no graph waits for an exchange | `ggml/src/ggml-cuda/expert-cache.cu` |
| host tier | cold experts of a decode window are computed on CPU workers pinned to the GPUs' NUMA node (`GGML_CUDA_MOE_HOST_CPUS`, `GGML_CUDA_MOE_HOST_THREADS`, `GGML_CUDA_HOST_NUMA`) from a mailbox in mapped pinned memory, so the GPU never idles on PCIe for a few cold rows | `ggml/src/ggml-cuda/moe-host.cu` |
| sparse attention | the QSA layers' top-k pool selection reads only the selected cells (`GGML_OP_FLASH_ATTN_SEL`, no dense KQ mask), the pooled indexer keys and the pools' member cells live in device tables of their own, a radix top-k; q8_0 K/V | `src/llama-memory-hybrid-idx.cpp`, `src/llama-kv-cache-dsa.cpp`, `ggml/src/ggml-cuda/fattn-sel.cu`, `top-k.cu` |
| MTP drafting | the model's own MTP block drafts (`--spec-type draft-mtp`, no second model), two draft steps chained on the device, a reduced draft vocabulary (`d2t`), the catch-up batch deferred into the first draft step, backend sampling | `common/speculative.cpp`, `src/llama-context.cpp` |
| memory | the 28.8 GB n-gram (PLE) table stays on the host and is mlock'd (`LLAMA_PLE_MLOCK`); short prompts (≤ `LLAMA_UBATCH_SMALL_MAX` tokens) are decoded in windows instead of the prompt path; the server's host prompt cache (`--cache-ram`) also saves the drafter's state so an agent's context survives other clients' requests | `src/llama-model.cpp`, `tools/server/` |
| model files | scripts that turn the release into the served file: dense weights re-encoded to Q8_0, experts reordered hot-first and split into hot/cold tensors by routing counts, the MTP block merged into the trunk, the attention tensors re-encoded exactly as Q4_16 in place | `scripts/flashnext/` |

Measured on the production box (2× P100, Xeon Gold 6132, 96 GB RAM, CUDA 12.9), build u65, context 204,800,
240 hot experts per layer, q8_0 K/V, temperature 0 unless stated:

| prompt | prefill | decode (novel / cached prompt) |
|---|---|---|
| 1K tokens | 318–395 tok/s | 75 / 90 tok/s |
| 16K | 410–418 | 78 / 82 |
| 64K | 415 | 75 / 68 |
| 200K | ~357 | ~60 |

KL divergence of the served file against the Q8-dense base: 0.0047 (the model's own rounding floor is ~0.004).
Quality through the same server: HumanEval / HumanEval+ (EvalPlus harness) 97.0 / 93.9 one-shot; LiveCodeBench
v6 (175 problems, 2025-01..04) 80.6% through a coding agent with a 10-minute budget per problem (93% of the
problems that did not hit the budget; the model card reports 91.9 with unlimited thinking).

## Build

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CUDA 12.x and a driver that supports it (the P100 runs with driver 580). `GGML_NATIVE=ON` (the default) is
used for the host tier's AVX-512 kernels.

## Model files

Start from the release's Q2_0 shards (`Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2}-of-00002.gguf`) and its
`mtp-Qwen3.8-Flash-Next-Q4_0.gguf`; an imatrix of the model on mixed text gives the routing counts
(`llama-imatrix -m ... -f calibration.txt -c 512 --chunks 200 -o imatrix.gguf`). Shard 2 holds experts only and
is shared by every variant (symlink it next to each output).

```sh
# 1. every dense weight to Q8_0, the routers to F32 (experts, embeddings, norms copied byte for byte)
python3 scripts/flashnext/fn_requant.py release-00001-of-00002.gguf q8d-00001-of-00002.gguf
# 2. experts hot-first per layer by the imatrix counts, split into hot (352 per layer) and cold tensors
python3 scripts/flashnext/fn_hybrid.py q8d-00001-of-00002.gguf hyb-00001-of-00002.gguf imatrix.gguf 352
# 3. the MTP block merged into the trunk (the draft shares the trunk's embeddings)
python3 scripts/flashnext/fn_add_mtp.py hyb-00001-of-00002.gguf mtp-Qwen3.8-Flash-Next-Q4_0.gguf model-00001-of-00002.gguf
# 4. the MTP block's dense weights as Q8_0 and its head reduced to a draft vocabulary (optional, +5% decode)
python3 scripts/flashnext/fn_requant.py model-00001-of-00002.gguf model-m8v-00001-of-00002.gguf --skip shared_head_head --draft-vocab draft-vocab.txt
# 5. the attention tensors re-encoded exactly from the release's Q3_K/Q2_0 as Q4_16, in place (a backup directory, --restore undoes it)
python3 scripts/flashnext/fn_inplace.py model-m8v-00001-of-00002.gguf --to q4_16 --from-release release-00001-of-00002.gguf --only 'attn_(qkv|gate|q|k|v)\.weight' --backup q4bak
```

The hot count (352) is what fits a 20K context; the server moves the hot/cold border at load (`LLAMA_EXPERT_HOT`),
so one file serves every context size. Plain Q4_0 for the dense trunk ruins the model (KLD 0.15); Q4_16 is the
exact re-encoding of the 60 attention tensors the release itself stores at Q3_K/Q2_0 (the script leaves every
other matched tensor at Q8_0). Steps 1–3 need disk for a second copy of shard 1 (~40 GB); step 5 works in place.

## Serving

The production command, from `flash.sh` on the box (two P100s on NUMA node 1, CPUs of that node for the host tier):

```sh
GGML_CUDA_GRAPHS_PRE_VOLTA=1 GGML_CUDA_ALLREDUCE=internal LLAMA_KV_PAD_GEOM=1 \
LLAMA_EXPERT_HOT=240 LLAMA_EXPERT_HOT_LAYERS=320,283,277,221,232,240,...   # per-layer hot budget (48 values), optional \
GGML_CUDA_MOE_HOST_CPUS=1,3,5,7,9,13,17,19,21,23,25,27 GGML_CUDA_MOE_HOST_THREADS=12 GGML_CUDA_HOST_NUMA=1 \
LLAMA_UBATCH_SMALL_MAX=96 \
taskset -c 11,15,39,43 ./llama-server -m model-m8v-00001-of-00002.gguf -ngl 999 -fa on --fit off \
    -sm tensor -ot 'per_layer_token_embd=CPU,ffn_(up|gate|down)_exps_cold=CUDA_Host' -lm none -lzm off \
    -t 8 -c 204800 -ub 1024 -ubd 128 -b 2048 --parallel 1 --cache-ram 8192 \
    -ctk q8_0 -ctv q8_0 --ctx-checkpoints 8 --checkpoint-min-step 0 --slot-prompt-similarity 0.05 \
    --spec-type draft-mtp --spec-draft-n-max 2 -bs --jinja --host 0.0.0.0 --port 8080
```

- `-sm tensor` with `-ot ... exps_cold=CUDA_Host` is the whole topology: dense weights and hot experts split over
  the cards, cold experts in pinned host memory, the n-gram table on the CPU.
- `GGML_CUDA_GRAPHS_PRE_VOLTA=1` enables CUDA graphs on Pascal (upstream gates them to Volta+);
  `GGML_CUDA_ALLREDUCE=internal` selects the in-engine AllReduce.
- Hot experts per layer: 312 fit a 20K context, ~300 at 69K, 240 at 204,800 (each 8 experts per layer are 250 MiB
  per card). A larger context costs decode speed through the cold share.
- `--parallel 1`: the engine decodes one sequence; a second slot serializes and halves both (measured).
- `--cache-ram 8192`: the host prompt cache; with it an agent's conversation is restored in ~0.5 s after another
  client's request instead of re-prefilled.
- `-ubd 128` (`--spec-draft-n-ubatch`): the MTP block's prompt ubatch, which needs far less compute buffer.

## Layout and history

- `QWEN38_FLASH_NEXT_ARCHITECTURE.md` on the box (not in this repo) is the engineering log: 86 numbered findings
  with the measurements behind every change, including the refuted ones.
- The branch is a squashed engine commit over upstream master plus a few later commits; previous histories are
  kept as `backup/flashnext-p100-wip-<sha>` branches on this fork.
- `tests/test-backend-ops.cpp` covers the added ops (`FLASH_ATTN_SEL`, the GDN decode columns, the top-k identity).

## Limitations

Single sequence, this model family only (`qwen4exp`: Qwen3.8-Flash-Next and its derivatives), sm_60/61 (the
kernels assume no dp4a and HFMA2 pairs), Linux, two cards (the AllReduce and the expert split are written for
exactly two). The decode is not run-to-run deterministic at temperature 0: the hot expert set evolves with the
traffic, and a cold expert computed on the host rounds differently from a hot one on the GPU (texts diverge after
some tens of tokens; the KL divergence does not move).
