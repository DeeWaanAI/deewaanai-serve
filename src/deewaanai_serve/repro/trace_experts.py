#!/usr/bin/env python3
"""
trace_experts.py — prefill-only MoE expert-activation profiler (numerics-safe).

For each prompt we run ONE forward pass (prefill), intercepting the router's
top-k selection at every MoE layer. We never generate tokens and never run the
expert FFNs' outputs into a loss — we only record WHICH experts each token
activates. This is far cheaper than serving and gives the real, measured traffic
statistics the paper needs:

  - per-layer expert-visit frequency  -> Zipf alpha + "top-X% experts cover Y%
    of traffic" (directly sizes the DeeWaanAI pinned working set / RAM),
  - per-prompt (layer, expert) set      -> pairwise Jaccard overlap, which lets
    us construct concurrency mixes at a TARGET expert-overlap (the rigorous
    replacement for the identical/distinct proxy).

Model-agnostic: it monkeypatches any block exposing `.switch_mlp(x, indices)`
(OLMoE, Qwen3-MoE, Mixtral, Moonlight/DeepSeek-style) and/or `.experts` — the
selected indices are captured, then the original call runs unchanged, so the
model's numerics are untouched.

Usage:
  python3 trace_experts.py --model-dir <mlx-4bit-dir> --prompts prompts.txt \
      --out <json> [--max-prompts 500] [--max-len 512]
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import KVCache


class _Recorder:
    """Wrapper around a switch_mlp module, reassigned onto the block as
    `mlp.switch_mlp`. The block's `self.switch_mlp(x, inds)` then resolves to us
    via attribute lookup — monkeypatching an nn.Module's __call__ directly is
    bypassed by the metaclass, which is why the first version captured nothing."""
    def __init__(self, mod, li, records):
        self._mod, self._li, self._records = mod, li, records
    def __call__(self, x, indices, *a, **k):
        try:
            arr = mx.array(indices).astype(mx.int32)
            mx.eval(arr)
            self._records.setdefault(self._li, []).append(arr)  # [T,k]
        except Exception:
            pass
        return self._mod(x, indices, *a, **k)
    def __getattr__(self, name):
        return getattr(self._mod, name)


def instrument(model):
    """Wrap every MoE `switch_mlp` so its call records the chosen indices."""
    records = {}
    for li, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        sm = getattr(mlp, "switch_mlp", None) if mlp else None
        if sm is None:
            continue
        mlp.switch_mlp = _Recorder(sm, li, records)
    return records


def prompt_sets_from_records(records, n_layers_present):
    """Build, per token position summed to per-prompt expert SETS. Since we
    append one [T,k] tensor per layer per forward (one prompt at a time), we
    also track prompt boundaries by call count == number of forwards."""
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompts", required=True, help="file, one prompt per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-prompts", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--records", default=None,
                    help="append-only JSONL, one line per traced prompt with the "
                         "per-layer expert COUNTS. Aggregation always folds this "
                         "file, so any checkpoint is a valid stopping point.")
    ap.add_argument("--resume", action="store_true",
                    help="skip prompt indices already present in --records")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="run no forwards; build --out purely from --records")
    ap.add_argument("--seq-out", default=None,
                    help="optional JSONL of PER-TOKEN ordered expert sequences "
                         "{i, l:{layer:[[t0 experts],[t1 experts],...]}} for the "
                         "NEA predictability study (D-044).")
    args = ap.parse_args()

    prompts = [ln.strip() for ln in Path(args.prompts).read_text().splitlines()
               if ln.strip()][: args.max_prompts]
    if not prompts:
        print("no prompts", file=sys.stderr); return 1

    rec_path = (Path(args.records) if args.records
                else Path(args.out).with_suffix(".records.jsonl"))
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if rec_path.exists():
        for ln in rec_path.read_text().splitlines():
            try:
                done.add(int(json.loads(ln)["i"]))
            except Exception:
                continue
        print(f"records: {len(done)} prompts already captured in {rec_path}", file=sys.stderr)

    todo = [i for i in range(len(prompts)) if i not in done]
    if args.aggregate_only:
        todo = []
    elif not todo:
        print("nothing to trace; will aggregate", file=sys.stderr)

    records = {}
    num_layers = 0
    if todo:
        model, tok = mlx_lm.load(args.model_dir)
        num_layers = len(model.layers)
        records = instrument(model)

    t0 = time.perf_counter()
    seqf = open(args.seq_out, "a") if args.seq_out else None
    with rec_path.open("a") as rf:
        for n, i in enumerate(todo):
            p = prompts[i]
            ids = tok.encode(p)[: args.max_len]
            if not ids:
                continue
            caches = [KVCache() for _ in range(num_layers)]
            try:
                out = model(mx.array([ids]), caches)
                mx.eval(out)
            except Exception as e:
                print(f"forward failed on prompt {i}: {e}", file=sys.stderr)
                for li in records:
                    records[li].clear()
                continue
            # Compact per-prompt record: layer -> {expert: visit count}. Enough
            # for the §4 ranking (counts) AND §5 overlap (non-zero keys = set).
            entry = {"i": i, "l": {}}
            seq_entry = {"i": i, "l": {}} if args.seq_out else None
            for li, arrl in records.items():
                if not arrl:
                    continue
                arr = arrl[-1]
                arr = arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(-1, arr.shape[-1])
                rows = arr.tolist()
                c = Counter()
                for row in rows:
                    c.update(row)
                entry["l"][str(li)] = {str(e): int(v) for e, v in c.items()}
                if seq_entry is not None:
                    # per-token ordered expert selection (NEA predictability, D-044)
                    seq_entry["l"][str(li)] = [[int(e) for e in row] for row in rows]
            rf.write(json.dumps(entry) + "\n")
            rf.flush()                     # durable: a crash loses at most 1 prompt
            if seqf is not None:
                seqf.write(json.dumps(seq_entry) + "\n")
                seqf.flush()
            for li in records:
                records[li].clear()         # raw tensors are no longer needed
            if n % 25 == 0:
                print(f"  traced {n+1}/{len(todo)} (prompt {i})", file=sys.stderr, flush=True)

    if seqf is not None:
        seqf.close()

    # ── Aggregate by FOLDING THE RECORDS FILE (not RAM), so any checkpoint of
    #    this file is a complete, valid input for §4's hot-set selection. ──────
    layer_freq = {}
    per_prompt_sets = []
    n_fwd = 0
    for ln in rec_path.read_text().splitlines():
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        sig = set()
        for li_s, counts in rec.get("l", {}).items():
            li = int(li_s)
            freq = layer_freq.setdefault(li, Counter())
            for e_s, v in counts.items():
                e = int(e_s)
                freq[e] += int(v)
                sig.add((li, e))
        per_prompt_sets.append(sig)
        n_fwd += 1
    if not layer_freq:
        print("no records to aggregate", file=sys.stderr); return 1
    print(f"folding {n_fwd} prompts from {rec_path}", file=sys.stderr)

    out = {
        "model_dir": str(args.model_dir), "n_prompts_traced": n_fwd,
        "seconds": round(time.perf_counter() - t0, 1),
        "layers": {}, "global_freq": Counter(),
    }
    for li, freq in layer_freq.items():
        total = sum(freq.values())
        order = [e for e, _ in freq.most_common()]
        cum = 0; cover = []
        for rank, e in enumerate(order, 1):
            cum += freq[e]; cover.append(round(cum / total, 4))
        top = order[: min(len(order), 32)]
        out["layers"][str(li)] = {
            "num_experts_used": len(freq), "total_visits": total,
            "top_experts": top,
            "coverage_topk": cover[: 32],           # fraction of traffic in top-k
            # FULL observed ranking (experts that never fired are absent; §4's
            # selector pads them deterministically at the tail). Needed because
            # P>25% on a 128-expert layer exceeds the truncated fields above.
            "ranked_experts": order,
            "coverage_full": cover,
            "zipf_alpha": _fit_zipf(freq),
        }
        for e, c in freq.items():
            out["global_freq"][str(e)] = out["global_freq"].get(str(e), 0) + c

    out["pairwise_jaccard"] = _jaccard_sample(per_prompt_sets, sample=20000)
    # serialise (Counter -> dict)
    out["global_freq"] = dict(out["global_freq"])
    out["_params"] = {"max_prompts": args.max_prompts, "max_len": args.max_len,
                      "prompts_file": str(args.prompts), "records": str(rec_path)}
    # ATOMIC: an armed §4 chain polls for this file and starts on sight, so a
    # half-written JSON must never be observable.
    tmp = Path(str(args.out) + ".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(args.out)
    print(f"wrote {args.out}  ({n_fwd} prompts, "
          f"{sum(len(v) for v in per_prompt_sets[:1]) if per_prompt_sets else 0}+ experts/layer)")
    return 0


def _fit_zipf(freq):
    """Crude Zipf exponent from rank-frequency (log-log linear fit) on a layer."""
    items = sorted(freq.values(), reverse=True)
    n = len(items)
    if n < 3:
        return None
    import math
    xs = [math.log(i + 1) for i in range(n)]
    ys = [math.log(c) for c in items]
    m = n
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = (m * sxx - sx * sx)
    if den == 0:
        return None
    slope = (m * sxy - sx * sy) / den      # ~ -alpha
    return round(-slope, 3)


def _jaccard_sample(sets, sample):
    import random
    if len(sets) < 2:
        return None
    random.seed(1234)
    js = []
    n = len(sets)
    for _ in range(min(sample, n * (n - 1) // 2)):
        i, j = random.randrange(n), random.randrange(n)
        if i == j:
            continue
        a, b = sets[i], sets[j]
        u = len(a | b)
        if u:
            js.append(len(a & b) / u)
    if not js:
        return None
    js.sort()
    q = lambda f: js[min(len(js) - 1, int(f * len(js)))]
    return {"n_pairs": len(js), "min": round(js[0], 3), "p25": round(q(.25), 3),
            "median": round(q(.5), 3), "p75": round(q(.75), 3), "max": round(js[-1], 3)}


if __name__ == "__main__":
    sys.exit(main())
