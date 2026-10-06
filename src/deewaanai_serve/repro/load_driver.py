#!/usr/bin/env python3
"""
Concurrency load driver — Gate-1 harness (§2 H1/H2 protocol).

Arm-agnostic: speaks OpenAI-compatible /v1/chat/completions with streaming,
so it drives llama-server, mlx_lm.server, and our pinned-arm server alike.
Reused UNCHANGED on the rental box (§7 step 1: harness developed locally).

Metrics per level (pre-registered definitions):
  - TTFT: request send → first content token received (streaming).
  - per-user tok/s: output tokens / decode window (first→last token) per req.
  - aggregate tok/s: total output tokens / level wall window.
  - error rate: failed or timed-out requests / total.
  - level FAILED (per §2) if: server unreachable, TTFT > 60 s, or err > 5%.

Resilience (owner may walk away; network/battery may die):
  - stdlib only (no pip deps, no venv needed).
  - Append-only JSONL: every request result is flushed to disk immediately;
    level summaries are checkpointed. On restart, completed levels are
    skipped (--resume, default on).
  - No sleep-polling loops; single sequential pass; nice'd by the wrapper.
  - Exit codes: 0 = all levels done, 10 = ramp ended by failure level
    (that IS the H2 result), 1 = harness bug.

Usage:
  python3 load_driver.py --arm-label llama_t1 \
      --api-base http://127.0.0.1:8080 --model olmoe \
      --levels 1,2,4 --max-tokens 50 --reps 1 \
      --out ../../results/ramp/
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TTFT_LIMIT_S = 60.0      # §2 pre-registered failure definition
ERR_LIMIT = 0.05         # §2 pre-registered
MIN_PER_WORKER = 3       # requests per concurrent worker per level (keeps high
                         # levels properly sampled, not capped at len(prompts))

_write_lock = threading.Lock()
_FULL_TEXT = True        # store full prompt + full output text (paper proof)


def load_prompts() -> list[str]:
    """Pre-registered TEST_PROMPTS from config.py (ast-parsed; no torch import)."""
    import ast
    here = Path(__file__).resolve().parent.parent
    src = (here / "config.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "TEST_PROMPTS":
                    return list(ast.literal_eval(node.value))
    raise RuntimeError("TEST_PROMPTS not found in code/config.py")


def append_jsonl(path: Path, rec: dict) -> None:
    with _write_lock:
        with path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


def one_request(api_base: str, model: str, prompt: str, max_tokens: int,
                timeout_s: float, arm: str, level: int, req_id: str) -> dict:
    """One streaming chat-completion. Returns a metric record."""
    url = api_base.rstrip("/") + "/v1/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,          # greedy, matching Stage-0 reference
        "stream": True,
    }).encode()
    t_send = time.perf_counter()
    t_first = None
    n_chunks = 0
    chars = 0
    text_parts: list[str] = []
    err = None
    try:
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:                 # SSE: iterate lines
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                # OpenAI-compat servers put streamed text in delta.content, but
                # llama.cpp with a reasoning template (e.g. Qwen3) streams the
                # thinking text in delta.reasoning_content with content=null.
                # Count whichever is present as a generated token so decode
                # speed is measured for every arm; note the field for rigor.
                piece = delta.get("content")
                if not piece:
                    piece = delta.get("reasoning_content")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_chunks += 1
                    chars += len(piece)
                    text_parts.append(piece)
        t_done = time.perf_counter()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            TimeoutError, json.JSONDecodeError) as e:
        t_done = time.perf_counter()
        err = f"{type(e).__name__}: {e}"[:300]

    ttft = (t_first - t_send) if t_first else None
    decode_window = (t_done - t_first) if t_first else None
    rec = {
        "arm": arm, "level": level, "req_id": req_id,
        "prompt_chars": len(prompt),
        "prompt_text": prompt if _FULL_TEXT else None,
        "ok": err is None and n_chunks > 0,
        "error": err,
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "wall_s": round(t_done - t_send, 4),
        "decode_s": round(decode_window, 4) if decode_window else None,
        "out_chunks": n_chunks,
        "out_chars": chars,
        "tok_per_s_decode": (n_chunks / decode_window)
            if decode_window and decode_window > 0 else None,
        "text_head": "".join(text_parts)[:120],
        "text_full": "".join(text_parts) if _FULL_TEXT else None,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return rec


def probe_server(api_base: str) -> bool:
    """Is the server alive? (models endpoint)."""
    try:
        with urllib.request.urlopen(api_base.rstrip("/") + "/v1/models",
                                    timeout=5):
            return True
    except Exception:
        return False


def run_level(args, prompts: list[int], level: int, out_dir: Path,
              reqlog: Path) -> dict:
    """Fire requests through `level` workers.
    --prompt-mode identical -> every concurrent user sends the SAME prompt
    (maximal expert-set overlap = the "all users hit the same experts" case).
    distinct -> users cycle the corpus (natural/low overlap). Jobs are filled to
    at least level*MIN_PER_WORKER so high concurrency is genuinely exercised."""
    mode = getattr(args, "prompt_mode", "distinct")
    base = [prompts[0]] if mode == "identical" else prompts
    want = max(len(base) * args.reps, level * MIN_PER_WORKER)
    jobs = [(f"{mode}_{i}", base[i % len(base)]) for i in range(want)]
    results: list[dict] = []

    def work(job):
        rid, p = job
        rec = one_request(args.api_base, args.model, p, args.max_tokens,
                          args.timeout_s, args.arm_label, level, rid)
        append_jsonl(reqlog, rec)
        return rec

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=level) as ex:
        for rec in ex.map(work, jobs):
            results.append(rec)
    window = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    tu = [r["tok_per_s_decode"] for r in ok if r["tok_per_s_decode"]]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    total_tokens = sum(r["out_chunks"] for r in ok)
    alive = probe_server(args.api_base)

    err_rate = len(errs) / max(1, len(results))
    ttft_max = max(ttfts) if ttfts else None
    hard_fail = []          # §2 pre-registered failure definitions
    if not alive:
        hard_fail.append("server_unreachable")
    if ttft_max is not None and ttft_max > TTFT_LIMIT_S:
        hard_fail.append(f"ttft>{TTFT_LIMIT_S}s")
    if err_rate > ERR_LIMIT:
        hard_fail.append(f"error_rate>{ERR_LIMIT:.0%}")
    # SLO quality boundary: worst-user decode speed below the floor (only when
    # the server is still alive — otherwise it's already a hard failure).
    floor = getattr(args, "min_user_tok_s", 0.0) or 0.0
    user_min = min(tu) if tu else None
    quality_fail = bool(alive and floor > 0 and user_min is not None
                        and user_min < floor)
    fail_reasons = hard_fail + (["user_tps<floor"] if quality_fail else [])

    summary = {
        "arm": args.arm_label, "level": level,
        "n_requests": len(results), "n_ok": len(ok), "n_err": len(errs),
        "error_rate": round(err_rate, 4),
        "aggregate_tok_s": round(total_tokens / window, 2) if window > 0 else None,
        "per_user_tok_s_median": round(statistics.median(tu), 2) if tu else None,
        "per_user_tok_s_min": round(user_min, 2) if user_min is not None else None,
        "ttft_median_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_max_s": round(ttft_max, 3) if ttft_max is not None else None,
        "level_window_s": round(window, 2),
        "server_alive_after": alive,
        "slo_floor_tok_s": floor or None,
        "hard_stable": len(hard_fail) == 0,      # server alive & within §2 limits
        "quality_stable": len(hard_fail) == 0 and not quality_fail,
        "stable": len(fail_reasons) == 0,        # ramp continues only if both hold
        "fail_reasons": fail_reasons,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    append_jsonl(out_dir / f"{args.arm_label}.levels.jsonl", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-label", required=True,
                    help="e.g. llama_t1 | mlxstock_t1 | pinned_t1")
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", default="local")
    ap.add_argument("--levels", default="1,2,4",
                    help="comma-separated concurrency levels, ascending")
    ap.add_argument("--max-tokens", type=int, default=50)
    ap.add_argument("--reps", type=int, default=1,
                    help="full prompt-set repetitions per level")
    ap.add_argument("--min-user-tok-s", type=float, default=0.0,
                    help="SLO floor: worst-user decode tok/s below this marks "
                         "the level quality-FAILED (usable-users ceiling). "
                         "0=disabled (only hard-failure stops the ramp).")
    ap.add_argument("--prompt-mode", choices=["distinct", "identical"],
                    default="distinct",
                    help="identical = all concurrent users share one prompt "
                         "(max expert overlap); distinct = corpus rotation.")
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-full-text", action="store_true",
                    help="omit full prompt/output text from JSONL (saves space)")
    ap.add_argument("--provenance", default=None,
                    help="free-form JSON string or @file merged into run metadata")
    args = ap.parse_args()

    global _FULL_TEXT
    _FULL_TEXT = not args.no_full_text

    prompts = load_prompts()
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    reqlog = args.out / f"{args.arm_label}.requests.jsonl"

    # ── run provenance (paper-grade: who/what/when produced these numbers) ──
    import os
    import platform
    import subprocess
    def try_run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=10).stdout.strip()[:200]
        except Exception:
            return None
    meta = {
        "arm": args.arm_label, "api_base": args.api_base,
        "model_param": args.model, "levels": levels,
        "max_tokens": args.max_tokens, "reps": args.reps,
        "prompts": prompts,
        "host": platform.node(), "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "software_version": os.environ.get("SOFTVER", "v1-mem17+backlog256"),
        "pinned_compile": os.environ.get("PINNED_COMPILE", "0"),
        "python": platform.python_version(),
        "mlx_version": None, "git_commit_harness": None,
        "cpu_count": os.cpu_count(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import mlx.core as _mx
        meta["mlx_version"] = getattr(_mx, "__version__", None)
    except Exception:
        pass
    gc = try_run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"])
    meta["git_commit_harness"] = gc or try_run(
        ["git", "-C", str(Path(__file__).resolve().parents[2] /
                          "repo_apple-silicon-moe-serving"),
         "rev-parse", "--short", "HEAD"])
    if args.provenance:
        try:
            extra = (json.loads(Path(args.provenance.lstrip("@")).read_text())
                     if args.provenance.startswith("@")
                     else json.loads(args.provenance))
            meta.update(extra)
        except Exception as e:
            meta["provenance_error"] = str(e)[:200]
    (args.out / f"{args.arm_label}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))

    done_levels: set[int] = set()
    if not args.no_resume:
        lf = args.out / f"{args.arm_label}.levels.jsonl"
        if lf.exists():
            for line in lf.read_text().splitlines():
                try:
                    done_levels.add(json.loads(line)["level"])
                except Exception:
                    continue
    if done_levels:
        print(f"resume: skipping completed levels {sorted(done_levels)}",
              flush=True)

    if not probe_server(args.api_base):
        print(f"FATAL: server not reachable at {args.api_base}", flush=True)
        return 1

    print(f"arm={args.arm_label} levels={levels} prompts={len(prompts)} "
          f"max_tokens={args.max_tokens} reps={args.reps}", flush=True)

    ramp_failed = False
    for level in levels:
        if level in done_levels:
            continue
        s = run_level(args, prompts, level, args.out, reqlog)
        print(json.dumps(s), flush=True)
        if not s["stable"]:
            print(f"LEVEL {level} FAILED ({s['fail_reasons']}) — ramp stops "
                  f"here per §2 (this is the H2 result, not a harness error)",
                  flush=True)
            ramp_failed = True
            break
    return 10 if ramp_failed else 0


if __name__ == "__main__":
    sys.exit(main())
