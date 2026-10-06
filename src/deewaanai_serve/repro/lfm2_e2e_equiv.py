#!/usr/bin/env python3
"""
lfm2_e2e_equiv.py — end-to-end numeric gate on REAL weights.

Proves the full LFM2 path is faithful: stock mlx_lm.generate vs the pinned
model (moe_submodule(feed_forward) hooking + extracted per-expert slabs, all
experts resident). Identical greedy tokens = extraction + wiring + ordering all
correct on the real checkpoint (the synthetic block gate already proved the
kernel; this proves the data path).

Usage:
  python lfm2_e2e_equiv.py --ref-dir <mlx-4bit> --experts-dir <slabs> --experts 32
"""
import argparse, sys
from pathlib import Path
import mlx.core as mx
import mlx_lm
sys.path.insert(0, str(Path(__file__).resolve().parent / "stage0"))
from pinned_arm import PinnedOlmoe  # noqa: E402

PROMPTS = [
    "Explain how mixture-of-experts models work in simple terms.",
    "Write a Python function to compute fibonacci numbers.",
    "用中文解释什么是人工智能。",
]
MAXTOK = 40

def gen(model, tok, prompt):
    out = mlx_lm.generate(model, tok, prompt=prompt, max_tokens=MAXTOK,
                          verbose=False)
    return out["text"] if isinstance(out, dict) else out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--experts-dir", required=True)
    ap.add_argument("--experts", type=int, required=True)
    a = ap.parse_args()

    # reference: stock (generate all, then free before loading the pinned copy)
    smodel, stok = mlx_lm.load(a.ref_dir)
    refs = [gen(smodel, stok, p) for p in PROMPTS]
    del smodel
    import gc; gc.collect(); mx.clear_cache()

    # pinned: same weights, switch_mlp swapped to fully-resident PinnedSwitchMLP
    arm = PinnedOlmoe(Path(a.ref_dir), Path(a.experts_dir),
                      cold_capacity=1, memory_limit_gb=12.0)
    arm.pin_all(a.experts)
    print(f"pinned moe_layer_ids: {len(arm.moe_layer_ids)} layers, "
          f"resident ~{arm.resident_total_mb():.0f} MB")

    n_pass = 0
    for i, p in enumerate(PROMPTS):
        pin = gen(arm.model, arm.tokenizer, p)
        ref = refs[i]
        ok = ref == pin
        n_pass += ok
        print(f"[prompt {i}] {'MATCH' if ok else 'DIFFER'}")
        if not ok:
            print("  ref:", repr(ref[:160]))
            print("  pin:", repr(pin[:160]))
    print("-"*60)
    if n_pass == len(PROMPTS):
        print(f"RESULT: PASS — {n_pass}/{len(PROMPTS)} greedy generations identical (real weights).")
        return 0
    print(f"RESULT: FAIL — {n_pass}/{len(PROMPTS)} identical.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
