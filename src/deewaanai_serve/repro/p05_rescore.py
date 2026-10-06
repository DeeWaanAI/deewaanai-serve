#!/usr/bin/env python3
"""
p05_rescore.py — Recompute concurrency ceilings from FROZEN ramp request logs.

Methodology (D-033): the SLO "usable ceiling" is the highest concurrency level
at which the 5th percentile of per-user decode throughput stays >= the SLO
floor (10 tok/s). The prior pipeline flagged a level unstable on per-user MIN
(a single worst request), which mis-ranks arms; median over-states the tail.
p05 is the fairness statistic. Raw data is read-only; nothing here edits it.

Output (frozen derivatives, not sources):
  results/p05_rescore/p05_ceilings.json   per-arm ceiling under 3 statistics
  results/p05_rescore/p05_levels.csv      per-arm per-level p05/median/min table

Deterministic: no RNG. Percentile = numpy 'linear' interpolation, reimplemented
in pure python so the result does not depend on numpy version.
"""
import json, csv, glob, os, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
# optional: argv[1]=ramp dir, argv[2]=out dir  (defaults = original frozen run)
RAMP = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"results/from_box_final/ramp"
OUT  = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT/"results/p05_rescore"
OUT.mkdir(parents=True, exist_ok=True)

SLO_FLOOR = 10.0  # tok/s per user, from meta/levels slo_floor_tok_s

def pct_linear(sorted_vals, q):
    """numpy.percentile(..., interpolation='linear') equivalent. q in [0,100]."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(pos // 1); hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

def load_requests(fp):
    """-> {level: [tok_per_s_decode, ...]} for ok requests only."""
    by_level = {}
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("ok"):
                continue
            v = r.get("tok_per_s_decode")
            if v is None:
                continue
            by_level.setdefault(r["level"], []).append(float(v))
    return by_level

def ceiling(levels_stats, floor):
    """Highest level whose statistic >= floor. levels_stats: {level: value}."""
    ok = [lv for lv, val in sorted(levels_stats.items())
          if val == val and val >= floor]  # val==val filters NaN
    return max(ok) if ok else 0

def main():
    arms = {}
    rows = []
    for fp in sorted(glob.glob(str(RAMP / "*.requests.jsonl"))):
        base = os.path.basename(fp).replace(".requests.jsonl", "")
        by_level = load_requests(fp)
        if not by_level:
            continue
        p05, med, mn = {}, {}, {}
        for lv, vals in by_level.items():
            sv = sorted(vals)
            p05[lv] = round(pct_linear(sv, 5), 3)
            med[lv] = round(statistics.median(sv), 3)
            mn[lv]  = round(sv[0], 3)
            rows.append({"arm": base, "level": lv, "n": len(vals),
                         "p05_tok_s": p05[lv], "median_tok_s": med[lv],
                         "min_tok_s": mn[lv],
                         "p05_meets_slo": p05[lv] >= SLO_FLOOR})
        arms[base] = {
            "levels_tested": sorted(by_level),
            "ceiling_p05":    ceiling(p05, SLO_FLOOR),
            "ceiling_median": ceiling(med, SLO_FLOOR),
            "ceiling_min":    ceiling(mn,  SLO_FLOOR),
            "p05_by_level":   {str(k): p05[k] for k in sorted(p05)},
        }

    # pair pinned vs llama per (model)
    result = {
        "schema": "p05-rescore-v1",
        "generated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "slo_floor_tok_s": SLO_FLOOR,
        "percentile_method": "linear (numpy-equivalent), pure python",
        "note": "ceilings from frozen *.requests.jsonl; raw data unmodified",
        "arms": arms,
    }
    (OUT / "p05_ceilings.json").write_text(json.dumps(result, indent=2))
    with open(OUT / "p05_levels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm","level","n","p05_tok_s",
                            "median_tok_s","min_tok_s","p05_meets_slo"])
        w.writeheader(); w.writerows(rows)

    # console summary
    print(f"{'arm':40s} {'ceil_p05':>8s} {'ceil_med':>8s} {'ceil_min':>8s}")
    for a in sorted(arms):
        r = arms[a]
        print(f"{a:40s} {r['ceiling_p05']:>8d} {r['ceiling_median']:>8d} {r['ceiling_min']:>8d}")
    print(f"\nwrote {OUT/'p05_ceilings.json'}")
    print(f"wrote {OUT/'p05_levels.csv'}")

if __name__ == "__main__":
    main()
