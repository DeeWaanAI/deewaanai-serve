#!/usr/bin/env python3
"""verify_two_way.py — the enforcement gate (exit 0 required before any
paper number is trusted). Checks, with zero trust in derivatives:

1. RAW CHECK   : every DP's source file exists AND its sha256 equals the
                 current FROZEN_MANIFEST entry (edits to raw data after
                 freeze are detected here).
2. DP RE-RESOLVE: json-scalar → navigate key segments, compare value;
                 jsonl-field → re-read line index, compare field.
3. DERIVATION  : re-run analyze_derived.py; results/derived/summary.json
                 must come out byte-identical (deterministic analysis).
4. CLAIM DIRECTION: every mention's dp_ids exist; verbatim quote present
                 in its doc; derived formula evaluates True against the
                 re-resolved component values.

Pipeline order (run_pipeline.sh): analyze → datapoints → claims → freeze → verify.
"""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "paper" / "claim_map"
MANIFEST = ROOT / "results" / "FROZEN_MANIFEST.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_by_segments(obj, segs):
    for s in segs:
        if isinstance(obj, list):
            obj = obj[int(s)]
        else:
            obj = obj[s]
    return obj


def eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(fa) and math.isnan(fb):
        return True
    return abs(fa - fb) <= 1e-12


def main() -> int:
    fails = []
    manifest = {e["path"]: e["sha256"]
                for e in json.loads(MANIFEST.read_text())["files"]}
    dp = json.loads((MAP / "datapoints.json").read_text())
    pm = json.loads((MAP / "paper-claims.json").read_text())

    # 3. deterministic derivation re-run
    before = (ROOT / "results/derived/summary.json").read_bytes()
    subprocess.run([sys.executable, str(ROOT / "code/paper/analyze_derived.py")],
                   check=True, capture_output=True)
    after = (ROOT / "results/derived/summary.json").read_bytes()
    if before != after:
        fails.append(("DERIVATION", "analyze_derived.py re-run produced "
                                    "different bytes — analysis not deterministic"))

    # 1+2. datapoints
    values = {}
    for d in dp["datapoints"]:
        src = ROOT / d["source_file"]
        if not src.exists():
            fails.append((d["dp_id"], "missing source"))
            continue
        if d["source_file"] not in manifest:
            fails.append((d["dp_id"], "not in FROZEN_MANIFEST (run freeze first)"))
            continue
        if sha256(src) != manifest[d["source_file"]]:
            fails.append((d["dp_id"], f"RAW FILE CHANGED after freeze: {d['source_file']}"))
            continue
        if d["kind"] == "json-scalar":
            try:
                v = get_by_segments(json.loads(src.read_text()), d["key_segments"])
            except Exception:
                fails.append((d["dp_id"], "key path not found"))
                continue
            if not eq(v, d["value"]):
                fails.append((d["dp_id"], f"value mismatch raw={v} map={d['value']}"))
            values[d["dp_id"]] = v
        elif d["kind"] == "jsonl-field":
            try:
                line = src.read_text().splitlines()[d["line_index"]]
                row = json.loads(line)
                v = row[d["field"]]
            except Exception as e:
                fails.append((d["dp_id"], f"jsonl resolve error {e}"))
                continue
            if not eq(v, d["value"]):
                fails.append((d["dp_id"], f"jsonl value mismatch raw={v} map={d['value']}"))
            if d["verbatim_excerpt"] not in line:
                fails.append((d["dp_id"], "jsonl excerpt not verbatim in line"))
            values[d["dp_id"]] = v

    # 4. claims
    ids = set(values)
    docs = {}
    for m in pm["mentions"]:
        for pid in filter(None, [m.get("dp_id")] + m.get("component_dp_ids", [])):
            if pid not in ids:
                fails.append((m["mention_id"], f"dp missing: {pid}"))
        doc = m["doc"]
        if doc not in docs:
            docs[doc] = (ROOT / doc).read_text()
        if m["verbatim_quote"] not in docs[doc]:
            fails.append((m["mention_id"], "verbatim quote not in doc"))
        if m["kind"] == "derived" and m.get("formula"):
            try:
                comp = [values[p] for p in m["component_dp_ids"]]
                ok = eval(m["formula"], {"__builtins__": {}},
                          {"round": round, "all": all, "abs": abs,
                           **{f"v{i+1}": c for i, c in enumerate(comp)}})
                if ok is not True:
                    fails.append((m["mention_id"], f"formula evaluates {ok}"))
            except Exception as e:
                fails.append((m["mention_id"], f"formula error {e}"))

    if fails:
        for f in fails[:40]:
            print("FAIL", *f)
        print(f"{len(fails)} failures / {dp['count']} datapoints, "
              f"{pm['count']} mentions")
        return 1
    print(f"OK: {dp['count']} datapoints verified against frozen raw data; "
          f"{pm['count']} claims resolved; derivation byte-deterministic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
