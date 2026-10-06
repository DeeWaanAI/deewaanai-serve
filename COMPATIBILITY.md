# Engine compatibility

DeeWaanAI Serve swaps only the MoE feed-forward dispatch and keeps the stock backbone, so
a model must satisfy the attention/rope assumptions of the continuous-batching path. A
model that violates these will fail a runtime pre-flight check rather than silently
produce wrong output.

## Required

1. MLX-loadable MoE exposing `switch_mlp` (SwitchGLU) under `blk.mlp` (OLMoE, Qwen2/3-MoE),
   `blk.feed_forward` (LFM2-MoE), or `blk.block_sparse_moe` (Mixtral).
2. NeoX (non-traditional) RoPE — `rotate_half` layout.
3. Full rotary (no partial rotary).
4. Uniform per-layer `self_attn.rope` with a standard KV cache (no sliding-window-only or
   interleaved attention; no conv/mamba state caches).
5. 4-bit affine-quantized experts (packed uint32 + float scales/biases, group 64) — the
   MLX 4-bit checkpoint layout that `dwa convert` reads verbatim.

## Verified compatible

| Model | Provider | Notes |
|---|---|---|
| OLMoE-1B-7B | AI2 | fidelity gates pass |
| Qwen1.5-MoE-A2.7B | Alibaba | greedy match 3/3 |
| Qwen3-30B-A3B | Alibaba | reference >RAM subject (3B active) |
| Mixtral-8x7B | Mistral | synthetic gate; 13B active → poor streaming subject |

## Known incompatible (fail pre-flight)

| Model | Reason |
|---|---|
| LFM2-8B-A1B | conv+attention hybrid |
| gpt-oss | partial rotary + sliding window |
| Llama-4-Scout | traditional rope |
| Command-R | traditional rope + sliding window |
| DeepSeek-V3 | traditional + partial rotary |
| DBRX | no `self_attn` attribute |
| Granite-4.0-H | mamba hybrid |
| EXAONE-MoE | sliding window (likely) |

## >RAM note

A model can be architecture-compatible yet too large to fully load for gating. Use the
layer-by-layer hotrank path and a same-quant cross-engine fidelity check for >RAM models.
