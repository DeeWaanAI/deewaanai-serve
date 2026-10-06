#!/usr/bin/env python3
"""
union_analysis.py — measured "reuse fragmentation" curve from frozen expert traces.

Question (lit Phase-2 T5 / qs-inequality): as we add CONCURRENT requests to one
batch, how fast does the union of experts that must be RESIDENT grow? Slow
(sub-linear) growth => a pinned hot set amortizes across users (pinned wins).
Fast growth to the full pool => reuse fragments, pinning buys nothing (big-model
failure mode). This is the mechanism behind the RAM/model-size ratio.

Input : results/telemetry/<model>.records.jsonl  (prompt -> layer -> {expert:count})
Output: results/telemetry/<model>_union.json     (frozen derivative, not a source)

Deterministic: fixed RNG seed. No edits to raw data.
"""
import json, sys, random, statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RECORDS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"results/telemetry/olmoe_trace400.records.jsonl"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT/"results/telemetry/olmoe_trace400_union.json"
SEED = 12345
KS = [1,2,4,8,12,16,24,32,48,64]
COVERAGES = [1.0, 0.95, 0.90]   # fraction of batch traffic that must be resident
N_SAMPLES = 400

def load():
    prompts = []
    for line in open(RECORDS):
        line=line.strip()
        if not line: continue
        d=json.loads(line)
        # per-layer: {expert:int -> count}
        layers={int(li):{int(e):int(c) for e,c in lv.items()} for li,lv in d["l"].items()}
        prompts.append(layers)
    return prompts

def union_for_batch(batch, coverage):
    """Mean over layers of #experts resident to cover `coverage` of that layer's
    batch traffic. Returns (mean_resident_per_layer, mean_total_resident)."""
    n_layers=len(batch[0])
    per_layer=[]
    for li in range(n_layers):
        agg=defaultdict(int)
        for p in batch:
            for e,c in p[li].items(): agg[e]+=c
        total=sum(agg.values())
        if coverage>=1.0:
            per_layer.append(len(agg))
        else:
            order=sorted(agg, key=lambda e:-agg[e]); run=0; need=0
            for e in order:
                run+=agg[e]; need+=1
                if run>=coverage*total: break
            per_layer.append(need)
    return statistics.mean(per_layer), sum(per_layer)

def main():
    prompts=load(); P=len(prompts)
    # observed expert pool per layer = max distinct experts seen at any layer
    n_experts_layer=max(len(lv) for p in prompts for lv in p.values())
    rng=random.Random(SEED)
    result={"schema":"union-v1","records":str(RECORDS.name),"n_prompts":P,
            "experts_per_layer":n_experts_layer,"n_layers":len(prompts[0]),
            "seed":SEED,"samples":N_SAMPLES,"coverage_targets":COVERAGES,"curve":{}}
    print(f"OLMoE trace: {P} prompts, {len(prompts[0])} layers, ~{n_experts_layer} experts/layer")
    print(f"{'k':>3} | {'resident@100%':>13} | {'resident@95%':>12} | {'resident@90%':>12} | {'pool_frac@100%':>14}")
    for cov in COVERAGES:
        result["curve"][str(cov)]={}
    for k in KS:
        if k>P: break
        stats={c:[] for c in COVERAGES}
        for _ in range(N_SAMPLES):
            batch=rng.sample(prompts,k)
            for c in COVERAGES:
                mean_pl,tot=union_for_batch(batch,c)
                stats[c].append((mean_pl,tot))
        row={}
        for c in COVERAGES:
            mean_pl=statistics.mean(s[0] for s in stats[c])
            row[c]=round(mean_pl,2)
            result["curve"][str(c)][str(k)]=round(mean_pl,2)
        pool_frac=row[1.0]/n_experts_layer
        print(f"{k:>3} | {row[1.0]:>13.2f} | {row[0.95]:>12.2f} | {row[0.90]:>12.2f} | {pool_frac:>13.1%}")
    OUT.write_text(json.dumps(result,indent=2)); print("wrote",OUT)

if __name__=="__main__":
    main()
