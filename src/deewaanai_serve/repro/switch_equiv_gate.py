#!/usr/bin/env python3
"""
switch_equiv_gate.py — generic switch_mlp equivalence gate (any MoE dims).

Generalizes test_t2_switch_equivalence.py: proves our fully-pinned fast path
(PinnedSwitchMLP.build_resident_from_slabs -> ResidentSwitch) reproduces stock
mlx_lm SwitchGLU BIT-EXACTLY at a given model's expert dimensions, in both the
decode regime (T*top_k < 64, do_sort False) and the prefill regime
(T*top_k >= 64, do_sort True — exercises the sorted_indices asymmetry).

Synthetic weights (both arms consume identical tensors) → $0, model-agnostic,
and independent of the real checkpoint. Run it for OLMoE (regression) and LFM2
before trusting any LFM2 measurement.

Usage:
  python switch_equiv_gate.py --hidden 2048 --inter 1792 --experts 32 --topk 4 --group 64
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent / "stage0"))
from pinned_arm import PinnedSwitchMLP, ExpertPool  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear  # noqa: E402

BITS = 4

def quantize_experts(E, out_f, in_f, G):
    scale = (1.0 / in_f) ** 0.5
    fp = mx.random.uniform(low=-scale, high=scale, shape=(E, out_f, in_f)); mx.eval(fp)
    out = mx.quantize(fp, group_size=G, bits=BITS, mode="affine")
    w, s = out[0], out[1]
    bi = out[2] if len(out) > 2 else None
    mx.eval(w, s, *([bi] if bi is not None else []))
    return w, s, bi

def build_stock(H, I, E, G, quant):
    stock = SwitchGLU(input_dims=H, hidden_dims=I, num_experts=E, bias=False)
    for name, out_f, in_f in (("up_proj", I, H), ("gate_proj", I, H), ("down_proj", H, I)):
        q = QuantizedSwitchLinear(in_f, out_f, E, bias=False, group_size=G, bits=BITS, mode="affine")
        w, s, bi = quant[name]; q.weight = w; q.scales = s
        if bi is not None: q.biases = bi
        mx.eval(q.weight, q.scales, q.biases); setattr(stock, name, q)
    return stock

def build_pinned(H, I, E, G, TOPK, quant):
    pool = ExpertPool(Path("/nonexistent"), H, I, cold_capacity=1)
    mlp = PinnedSwitchMLP(pool=pool, layer_id=0, hidden=H, inter=I, num_experts=E, group=G)
    slab = {e: {f"{p}.{part}": quant[p][i][e] for p in ("up_proj","gate_proj","down_proj") for i,part in enumerate(("weight","scales","biases")) if quant[p][i] is not None} for e in range(E)}
    mlp.build_resident_from_slabs(slab)
    assert mlp._P == mlp.num_experts
    return mlp

def run_case(T, H, E, TOPK, stock, ours):
    mx.random.seed(1234 + T)
    x = mx.random.normal(shape=(T, H)).astype(mx.bfloat16)
    idx = mx.argsort(mx.random.uniform(shape=(T, E)), axis=-1)[:, :TOPK].astype(mx.int32)
    mx.eval(x, idx)
    a = stock(x, idx); b = ours(x, idx); mx.eval(a, b)
    equal = mx.array_equal(a, b).item()
    diff = mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item()
    reg = "decode" if idx.size < 64 else "prefill"
    print(f"  T={T:<4} k={TOPK} sort={str(idx.size>=64):<5} {reg:<7} -> " + ("BIT-EXACT" if equal else f"DIVERGENT max|Δ|={diff:.3e}"))
    return bool(equal)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--inter", type=int, required=True)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--name", default="model")
    a = ap.parse_args()
    H,I,E,G,TOPK = a.hidden,a.inter,a.experts,a.group,a.topk
    print("="*72); print(f"switch_mlp equivalence gate — {a.name}"); print(f"dims hidden={H} inter={I} experts={E} topk={TOPK} group={G}"); print("="*72)
    mx.random.seed(7)
    quant = {"up_proj":quantize_experts(E,I,H,G), "gate_proj":quantize_experts(E,I,H,G), "down_proj":quantize_experts(E,H,I,G)}
    stock = build_stock(H,I,E,G,quant); ours = build_pinned(H,I,E,G,TOPK,quant)
    res = [run_case(T,H,E,TOPK,stock,ours) for T in (1,4,8,16,64)]
    print("-"*72)
    if all(res): print(f"RESULT: PASS — {sum(res)}/{len(res)} bit-identical at {a.name} dims."); return 0
    print(f"RESULT: FAIL — {results.count(False) if False else res.count(False)}/{len(res)} diverge. Do NOT trust {a.name} measurements."); return 1

if __name__ == "__main__":
    sys.exit(main())
