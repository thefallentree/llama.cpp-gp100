# llama.cpp-gp100: llama.cpp for Tesla P100 (sm_60)

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for NVIDIA Pascal **sm_60**:
the Tesla P100 (GP100, 16 GB HBM2). The P100 has no tensor cores and no `dp4a`, but runs half-precision FMA
on `half2` pairs (HFMA2) at twice its fp32 rate. This fork serves large mixture-of-experts models on a few of
these cards: kernels built on HFMA2, experts split between VRAM and host memory, and a fused decode engine
that cuts the launch overhead that dominates decode on this hardware.

The fork tracks upstream llama.cpp. The branch `sm60` is upstream
[`785322d17`](https://github.com/ggml-org/llama.cpp/commit/785322d17f2d51e5185cf996a1f4e46d87640841)
plus two reverts and the fork's own commits, grouped by area (`git log 785322d17..sm60`). Everything upstream
does still works; the additions are selected on sm_60 or by their switches.

## Supported models and measured speeds

Measured on 2x Tesla P100 16 GB per node (CUDA 12.9, driver 580), with the production builds this branch was
cut from:

| model | setup | decode | prefill |
|---|---|---|---|
| **Qwen3.8-Flash-Next** (`qwen4exp`, 2-bit experts) | one node, 2x P100 tensor split, 262,144-token context, q8_0 K / q4_0 V, MTP drafting | ~10 ms/token at 1K context (~100 tok/s), ~12 ms/token at 200-255K (~85 tok/s) | ~550 tok/s (255K cold prompt in ~465 s) |
| **GLM-5.3-Flash** (`glm5-next`, Q2_K_S) | one node, 2x P100, 64K context, MTP drafting | 126 ms per draft round at 1K | 58.5 tok/s at 16K |
| **DeepSeek-V4.1-Flash** (`deepseek41`, UD-Q2_K_XL) | 2 nodes (4x P100) layer split over RPC, 32K context, DSpark draft | 157 ms per draft round, 72 ms/token | 56.7 tok/s at 16K |

Qwen3.8-Flash-Next: perplexity of the served file at c16384 is 2.8170; needle retrieval 5/5 at 16K and 256K;
median time to first token 0.97 s on `/completion` and 1.15 s on chat for 15-600-word requests.

## Key features

- **sm_60 kernels.** HFMA2 mat-vec and `MUL_MAT_ID` kernels for dense and expert weights (Q4_1_G64, Q4_1,
  Q4_K, Q8_0, Q2_0, Q4_0, Q2_K, Q3_K, Q6_K and the IQ1/IQ2/IQ3 i-quants), an HFMA2 tile GEMM for prompt
  batches, half2 KQ accumulation in flash attention.
- **Fused decode engine** (`ggml/src/ggml-cuda/fn-engine.cu`). Projections, MoE up/down, gated-delta-net and
  sparse-attention steps fused into few launches. A decode window is captured once and replayed as a CUDA
  graph. With `-sm tensor` both cards run every layer on half the weights, with a peer-to-peer all-reduce.
- **Hot/cold experts with a host tier.** The frequently routed experts of each layer stay in VRAM
  (`LLAMA_EXPERT_HOT`, per layer `LLAMA_EXPERT_HOT_LAYERS`), the rest in pinned host memory
  (`-ot "..._exps_cold=CUDA_Host"`). A decode step's cold pairs are computed by host threads pinned next to the
  GPUs (AVX-512/AVX2 kernels), with per-socket pools and an optional NUMA mirror. An **expert cache** swaps
  hot and cold experts by routing statistics while the server runs.
- **G64 / Q4_16 weight types.** `Q4_1_G64` (Q4_1 with one scale/min per 64 weights, 4.5 bpw) and `Q4_16`
  (an exact 5-bpw re-encoding of Q3_K / Q2_0 tensors for the engine). Convert any GGUF with
  `scripts/g64/convert_to_g64.py` (see `docs/backend/gp100-g64.md`).
- **Drafters.** The model's own MTP head (`--spec-type draft-mtp`), with draft steps chained on the device and a
  reduced draft vocabulary; DSpark (`--spec-type draft-dspark`, confidence cut `--spec-draft-p-cum`) and
  DFlash drafts. **Launch-ahead**: the next decode window starts on the device before the host has sampled
  its tokens.
- **Server prompt cache with drafter state.** `--cache-ram` saves and restores the drafter's per-sequence state
  with each prompt, so a long agent conversation is restored in seconds instead of re-prefilled;
  `--checkpoint-spread` keeps context checkpoints spread over the prompt.
- **Sparse attention.** `GGML_OP_FLASH_ATTN_SEL` attends over each token's selected cache rows (QSA / DSA
  indexers) on f16, q8_0 or q4_0 K/V; one-launch radix top-k.
- **RPC fixes** for layer splits over nodes: client and server graph caches stay coherent when buffers are
  freed, a remote device's pinned host buffer type is an `-ot` target.

## Type ids: a compatibility note

The fork's types use ids far above upstream's range: `GGML_TYPE_Q4_1_G64 = 99`, `GGML_TYPE_Q4_16 = 100`,
`LLAMA_FTYPE_MOSTLY_Q4_1_G64 = 99`. Older builds of this fork used 43/44, which upstream has since given to
`GGML_TYPE_F8_E4M3` (43). This branch reads upstream F8_E4M3 files as upstream does; **do not load upstream
F8_E4M3 GGUFs with older builds of this fork** (they would be read as G64), and retype files written by those
builds before loading them here.

## Build

CUDA 12.x and a driver that supports the P100 (580 is used). `GGML_NATIVE=ON` (the default) builds the host
tier's AVX-512 kernels for the build machine.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DGGML_CUDA_NO_VMM=ON \
    -DGGML_RPC=ON -DLLAMA_CURL=OFF
cmake --build build -j --target llama-server ggml-rpc-server llama-perplexity test-backend-ops
```

A CPU-only build (`-DGGML_CUDA=OFF`) builds and runs the CPU references of the added ggml ops and types.

## Serving examples

Environment and arguments are those of the production services, shortened where marked.

**Qwen3.8-Flash-Next, one node, 2x P100, 262K context:**

```sh
GGML_CUDA_GRAPHS_PRE_VOLTA=1 GGML_CUDA_ALLREDUCE=internal LLAMA_KV_PAD_GEOM=1 GGML_CUDA_HOST_NUMA=1 \
LLAMA_EXPERT_HOT=228 LLAMA_EXPERT_HOT_LAYERS=304,269,263,...   `# hot experts per layer (one value per layer)` \
GGML_CUDA_MOE_HOST_CPUS=1,3,5,7,9,13,17,19,21,23,25,27 GGML_CUDA_MOE_HOST_THREADS=24 \
GGML_CUDA_MOE_HOST_NUMA_POOLS=1 GGML_CUDA_MOE_HOST_MIRROR=1 \
GGML_CUDA_MOE_HOST_POOL_CPUS=0,2,4,6,8,10,12,14,16,18,20,22:1,3,5,7,9,13,17,19,21,23,25,27 \
GGML_NUMA_SCAN_OFF=1 GGML_CUDA_EC_SWAP_GRID=4 GGML_CUDA_EXPERT_CACHE_NOWAIT=1 \
GGML_CUDA_AR_COPY_CHUNK_BYTES=524288 LLAMA_IDX_CACHE_TYPE=q8_0 LLAMA_UBATCH_SMALL_MAX=96 \
GGML_CUDA_MMID16_NARROW128=1 GGML_CUDA_MOE_HOST_PROMPT=1 GGML_CUDA_MOE_HOST_PROMPT_MAX_TOK=256 \
taskset -c 11,15,39,43 ./llama-server -m Qwen3.8-Flash-Next-Q2_0-hyb3-00001-of-00002.gguf \
    -ngl 999 -fa on --fit off -sm tensor \
    -ot "per_layer_token_embd=CPU,ffn_(up|gate|down)_exps_cold=CUDA_Host" -lm none -lzm off \
    -t 8 -c 262144 -ub 1536 -ubd 512 -b 3072 --parallel 1 --cache-ram 24576 \
    -ctk q8_0 -ctv q4_0 --ctx-checkpoints 8 --checkpoint-min-step 0 --slot-prompt-similarity 0.05 \
    --spec-type draft-mtp --spec-draft-n-max 2 -bs --checkpoint-spread --jinja --host 0.0.0.0 --port 8080
```

- `-sm tensor` with `-ot ..._exps_cold=CUDA_Host` is the topology: dense weights and hot experts split over
  the two cards, cold experts in pinned host memory, the per-layer embedding table on the CPU.
- `GGML_CUDA_GRAPHS_PRE_VOLTA=1` enables CUDA graphs on Pascal (upstream enables them from Volta);
  `GGML_CUDA_ALLREDUCE=internal` selects the engine's all-reduce.
- The host-tier CPU lists name the cores of the NUMA nodes next to the GPUs; adjust them to the machine.
- `--parallel 1`: the engine decodes one sequence.

**GLM-5.3-Flash, one node, 2x P100 (layer split):**

```sh
GGML_CUDA_MOE_HOST_CPUS=<40 host cores> GGML_CUDA_MOE_HOST_THREADS=40 GGML_CUDA_MOE_HOST_NUMA_INTERLEAVE=1 \
GGML_CUDA_MOE_HOST_POOLS=1 LLAMA_EXPERT_HOT=48 LLAMA_EXPERT_HOT_MTP=1 LLAMA_UBATCH_SMALL_MAX=96 \
GGML_CUDA_MOE_HOST_PROMPT=1 LLAMA_SPEC_AHEAD=0 LLAMA_DRAFT_CHAIN=0 \
./llama-server -m GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf -ngl 999 -fa on --fit off -sm layer \
    -ot "_exps_cold=CUDA_Host" -lm none -lzm off -c 65536 -b 2048 -ub 512 -t 8 --parallel 1 --jinja \
    --spec-type draft-mtp --spec-draft-n-max 2 -bs --checkpoint-spread --cache-ram 2048 --host 0.0.0.0 --port 8080
```

**DeepSeek-V4.1-Flash, 2 nodes over RPC (layer split, DSpark draft on the head's second card):**

```sh
# node B: the rpc-server with its two cards (same host-tier environment as the head)
./ggml-rpc-server -H <node-B-address> -p 50052 -d CUDA0,CUDA1 -t 8

# node A: the head
GGML_CUDA_MOE_HOST_CPUS=<host cores> GGML_CUDA_MOE_HOST_THREADS=24 GGML_CUDA_MOE_HOST_NUMA_INTERLEAVE=1 \
GGML_CUDA_MOE_HOST_POOLS=1 GGML_CUDA_MOE_HOST_PROMPT=1 GGML_CUDA_MOE_HOST_HOT_MS=2000 \
LLAMA_EXPERT_HOT=52 LLAMA_EXPERT_HOT_LAYERS=0:47,1:47,...   `# layer:hot pairs` \
LLAMA_EXPERT_ORDER=ds41_expert_order.json LLAMA_UBATCH_SMALL_MAX=96 LLAMA_SPEC_AHEAD=0 LLAMA_DRAFT_CHAIN=0 \
LLAMA_DSV41_LAYER_SKIP=2 LLAMA_LAZY_WARM=1000 \
./llama-server -m DeepSeek-V4.1-Flash-UD-Q2_K_XL-00001-of-00007.gguf --rpc <node-B-address>:50052 \
    -ngl 999 -fa on --fit off -sm layer -ts 1,1,2,2 -lm none -lzm on -c 32768 -b 2048 -ub 128 -t 2 --parallel 1 \
    -ctk q8_0 -ctv q8_0 --jinja --chat-template-file models/templates/deepseek-ai-DeepSeek-V4.1-Flash.jinja \
    --spec-type draft-dspark -md DeepSeek-V4.1-Flash-DSpark-draft.gguf -ngld 99 -devd CUDA1 \
    --spec-draft-n-max 5 --spec-draft-p-min 0 --spec-draft-sampling probabilistic --spec-draft-p-cum 0.5 \
    --host 0.0.0.0 --port 8080
```

`-lm none` keeps a model larger than the host's memory limit from being mapped whole; `LLAMA_EXPERT_ORDER`
loads a stock file's experts hot-first without rewriting it.

## Repository notes

- `README-FLASHNEXT-P100.md`: the earlier notes for Qwen3.8-Flash-Next, with the recipe that turns the 2-bit
  release into the served file (`scripts/flashnext/`). Its speed table is from an older build at 204,800 tokens.
- `scripts/g64/`, `docs/backend/gp100-g64.md`: G64 conversion for any model.
- `tests/test-backend-ops` covers the added ops and types; `tests/test-backend-copy-order`,
  `tests/test-rpc-graph-cache.sh` and `tests/llama-context` cover the scheduler, RPC and context changes.
- Linux only; the kernels assume sm_60 (no `dp4a`, HFMA2). The decode engine and the all-reduce are written for
  one sequence on two cards per node.

For everything else (other backends, model support, the server API, bindings) see upstream's
[README](https://github.com/ggml-org/llama.cpp#readme) and [docs](https://github.com/ggml-org/llama.cpp/tree/master/docs).

## License

MIT, as upstream (see [LICENSE](LICENSE)).
