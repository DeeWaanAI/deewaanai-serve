#!/usr/bin/env python3
"""
prefetch_equiv_test.py — prove predictive prefetch is output-neutral.

Loads the pinned engine, pins a FRACTION of experts per layer from a measured
hotrank (so the cold streaming path is exercised), generates greedy tokens on a
few prompts, and prints a hash of the outputs. Run it twice — PREFETCH=0 and
PREFETCH=1 — and the hashes must match. Prefetch only changes WHEN expert bytes
are read, never WHICH experts run (native router stays authoritative), so any
divergence is a bug.

Usage:
  PREFETCH=0 code/venv_stage0/bin/python code/prefetch_equiv_test.py ...
  PREFETCH=1 code/venv_stage0/bin/python code/prefetch_equiv_test.py ...
"""
import argparse, hashlib, math, sys
from pathlib import Path
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent / "stage0"))
from pinned_arm import PinnedOlmoe
import pinned_arm
import mlx_lm

PROMPTS = [
    "Explain how mixture-of-experts models work in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
    "用中文解释什么是人工智能。",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--experts-dir", required=True)
    ap.add_argument("--hotrank", required=True)
    ap.add_argument("--pin-fraction", type=float, default=0.5)
    ap.add_argument("--experts", type=int, required=True)   # num_experts per layer
    a = ap.parse_args()
    print("PREFETCH_ENABLED =", pinned_arm.PREFETCH_ENABLED)
    arm = PinnedOlmoe(Path(a.ref_dir), Path(a.experts_dir),
                      cold_capacity=2048, memory_limit_gb=13.0)
    # select hot subset exactly like pinned_server does
    import json
    act = json.loads(Path(a.hotrank).read_text())
    rank_of = {int(k): (v.get("ranked_experts") or v.get("top_experts") or [])
               for k, v in act["layers"].items()}
    n_keep = max(1, math.ceil(a.pin_fraction * a.experts))
    per_layer = {}
    for l in arm.moe_layer_ids:
        ranked = list(rank_of.get(l, []))
        if len(ranked) < a.experts:
            seen = set(ranked); ranked += [e for e in range(a.experts) if e not in seen]
        per_layer[l] = sorted(ranked[:n_keep])
    arm.pin_experts(per_layer)
    # enable cross-step predictive prefetch to use the same ranking
    arm.pool.set_ranking(rank_of, {l: n_keep for l in arm.moe_layer_ids})
    print(f"resident/layer={n_keep}/{a.experts}, layers={len(arm.moe_layer_ids)}")

    outs = []
    for p in PROMPTS:
        o = mlx_lm.generate(arm.model, arm.tokenizer, prompt=p, max_tokens=40, verbose=False)
        outs.append(o["text"] if isinstance(o, dict) else o)
    blob = "\n".join(outs).encode()
    print("OUTPUT SHA256:", hashlib.sha256(blob).hexdigest())
    print("first output head:", repr(outs[0][:80]))

if __name__ == "__main__":
    main()
