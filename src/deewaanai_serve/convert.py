# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Mug and Bewong Ltd (Company No. 14916888).
# DeeWaanAI(TM) is a trademark of Mug and Bewong Ltd. See LICENSE.
#!/usr/bin/env python3
"""
extract_mlx_experts.py — build pinned-engine expert slabs from an MLX 4-bit
checkpoint, VERBATIM (no re-quantization, no recompression).

Why this exists: the pinned engine (code/stage0/pinned_arm.py ExpertPool /
_StackProj) consumes MLX's standard 4-bit affine layout — packed uint32 weight
+ float scales + float biases, group_size 64 — which is EXACTLY how mlx-community
4-bit checkpoints store routed experts. Producing slabs for a new model needs no
quantization math: we copy each expert's (weight, scales, biases) byte-for-byte
into per-expert slab files the engine expects: layer{LL}_expert{EEE}.safetensors
with keys {gate,up,down}_proj.{weight,scales,biases}.

Two checkpoint layouts are handled:
  A) PER-EXPERT (e.g. Qwen2-MoE):  model.layers.L.mlp.experts.E.{proj}.{part}
  B) STACKED (OLMoE / LFM2-MoE):   model.layers.L.{mlp|feed_forward}.switch_mlp.{proj}.{part}
     with leading dim == num_experts; we de-stack along dim 0.
Bytes are never recomputed in either case → the pinned arm's numerics equal
stock MLX 4-bit inference for the same experts (verified by the T2 gate).

Usage:
  python extract_mlx_experts.py --mlx-dir <mlx-4bit-dir> --out <experts-dir>
"""
import argparse, re
from collections import defaultdict
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")
# Mixtral names experts w1/w2/w3; mlx_lm maps them to gate/down/up at load
# (mixtral.py:212). Normalize everything to the canonical gate/up/down the
# pinned engine reads.
PROJ_ALIAS = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj",
              "gate_proj": "gate_proj", "up_proj": "up_proj", "down_proj": "down_proj"}
_PROJ_ALT = "|".join(PROJ_ALIAS)
_CONTAINER = r"(?:mlp|feed_forward|block_sparse_moe)"

# A) per-expert: model.layers.L.<container>.experts.E.{proj}.{part}
PER_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.%s\.experts\.(\d+)\.(%s)\.(%s)$"
    % (_CONTAINER, _PROJ_ALT, "|".join(PARTS)))
# B) stacked: model.layers.L.<container>.switch_mlp.{proj}.{part}
STACKED_RE = re.compile(
    r"^model\.layers\.(\d+)\.%s\.switch_mlp\.(%s)\.(%s)$"
    % (_CONTAINER, _PROJ_ALT, "|".join(PARTS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(dict)          # (layer, expert) -> {"proj.part": tensor}
    stacked = defaultdict(dict)          # (layer, proj)   -> {"part": tensor[E,...]}
    shards = sorted(Path(args.mlx_dir).glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors in {args.mlx_dir}")

    for sh in shards:
        with safe_open(str(sh), framework="pt") as f:
            for k in f.keys():
                ma = PER_EXPERT_RE.match(k)
                if ma:
                    L, E, proj, part = int(ma.group(1)), int(ma.group(2)), ma.group(3), ma.group(4)
                    grouped[(L, E)][f"{PROJ_ALIAS[proj]}.{part}"] = f.get_tensor(k)
                    continue
                mb = STACKED_RE.match(k)
                if mb:
                    L, proj, part = int(mb.group(1)), mb.group(2), mb.group(3)
                    stacked[(L, PROJ_ALIAS[proj])][part] = f.get_tensor(k)

    # Expand stacked layout into per-expert groups (de-stack along dim 0).
    if stacked:
        # num_experts per layer inferred from weight leading dim
        layers = sorted({L for (L, _) in stacked})
        for L in layers:
            w0 = stacked[(L, PROJS[0])].get("weight")
            if w0 is None:
                raise SystemExit(f"layer {L}: stacked switch_mlp missing weight")
            E = int(w0.shape[0])
            for e in range(E):
                for proj in PROJS:
                    d = stacked[(L, proj)]
                    missing = [p for p in PARTS if p not in d]
                    if missing:
                        raise SystemExit(f"layer {L} {proj}: missing {missing} in stacked layout")
                    for part in PARTS:
                        grouped[(L, e)][f"{proj}.{part}"] = d[part][e].contiguous()

    if not grouped:
        raise SystemExit("no expert tensors matched — wrong model layout?")

    bad = [k for k, v in grouped.items() if len(v) != 9]
    if bad:
        raise SystemExit(f"{len(bad)} experts incomplete (need 9 tensors), e.g. {bad[:3]}")

    n = 0
    for (L, E), state in sorted(grouped.items()):
        save_file(state, str(out / f"layer{L:02d}_expert{E:03d}.safetensors"))
        n += 1
    print(f"wrote {n} expert slabs to {out}")
    k0 = sorted(grouped)[0]; s0 = grouped[k0]
    for kk in sorted(s0)[:3]:
        t = s0[kk]; print("  sample", kk, tuple(t.shape), t.dtype)


if __name__ == "__main__":
    main()
