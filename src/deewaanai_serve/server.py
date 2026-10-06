# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Mug and Bewong Ltd (Company No. 14916888).
# DeeWaanAI(TM) is a trademark of Mug and Bewong Ltd. See LICENSE.
#!/usr/bin/env python3
"""
pinned_server.py — Gate-1 optimized-arm server: the pinned-expert engine
serving N concurrent users with continuous batched decoding.

OpenAI-compatible API (/v1/chat/completions with SSE streaming, /v1/models)
so the same harness (load_driver.py) drives stock arms and this arm alike.

Batching design (numerically faithful, stock backbone untouched):
  * Each arriving request is PREFILLED SOLO through the stock model path
    (byte-identical to single-user serving, positions 0..L-1).
  * Sequences join a shared "world" coordinate at world position W. Their
    prefilled KEYS get one extra rotation R(W - L): rope rotations compose,
    so a key rotated at p then by (W-L) behaves exactly as a key natively
    at p + (W-L). Every real token of a row shifts by the same constant, so
    all query→key RELATIVE distances equal solo serving. Batch-generated
    tokens are rope'd by the stock attention code at the uniform scalar
    offset W (all rows share it), preserving that invariant forever.
  * Left-pad columns [0, join_W - L) of each row are excluded by an additive
    -inf mask delivered through make_mask() on our batch cache object — the
    stock create_attention_mask path defers to it without touching model code.
  * Membership changes (join/finish) rebuild between steps; KV buffers are
    preallocated per layer with growth headroom, mimicking stock KVCache.
  * MoE: PinnedSwitchMLP resident gather path handles T = batch size with
    zero host syncs when fully pinned.

Resilience: single engine thread performs all MLX ops (no contention),
bounded queues per request, SIGTERM flushes stats. Run under
nohup+caffeinate by the caller; nothing here survives sleep, but all
measurements live client-side (load_driver JSONL), which resumes.
"""
from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
import os
import uuid
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).parent

from .engine import PinnedOlmoe


# ═══════════════════════════════════════════════════════════════════════════
# Batched KV machinery
# ═══════════════════════════════════════════════════════════════════════════

class SharedState:
    """World coordinate + membership for the running batch."""
    def __init__(self):
        self.W = 0                     # next uniform decode position
        self.rows: list[Seq] = []      # batch row order == list order
        self._mask = None
        self._mask_W = -1

    def pad_mask(self):
        """Additive mask [B,1,1,W+1] excluding each row's left-pad columns.
        Built once per world position, reused by all 16 layers."""
        if self._mask_W != self.W:
            pads = mx.array([[r.join_W - r.pre_len] for r in self.rows])
            cols = mx.arange(0, self.W + 1)
            keep = cols[None, :] >= pads            # [B, W+1]
            self._mask = mx.where(
                keep[:, None, None, :],
                mx.array(0.0, mx.bfloat16),
                mx.array(float("-inf"), mx.bfloat16))
            self._mask_W = self.W
        return self._mask


class BatchKVCache:
    """Stock-cache-API object owning one layer's batched k/v buffers.
    All rows share the scalar write position SharedState.W; W advances
    once per engine step AFTER the full forward."""

    STEP = 256

    def __init__(self, shared: SharedState):
        self.shared = shared
        self.keys = None
        self.values = None

    @property
    def offset(self) -> int:
        return self.shared.W

    @property
    def cap(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update_and_fetch(self, keys, values):
        prev = self.shared.W
        B, H, add, D = keys.shape
        if prev + add > self.cap:
            self._grow_cols(prev + add + self.STEP)
        self.keys[..., prev:prev + add, :] = keys
        self.values[..., prev:prev + add, :] = values
        return (self.keys[..., :prev + add, :],
                self.values[..., :prev + add, :])

    def make_mask(self, N, return_array=False, window_size=None):
        if N == 1:
            # pads exist iff some row joined after the batch origin
            for r in self.shared.rows:
                if r.join_W != r.pre_len:
                    m = self.shared.pad_mask()
                    # Match the model's attention dtype: OLMoE is bfloat16,
                    # Qwen2-MoE runs float16 — SDPA requires the mask to promote
                    # to the output dtype, so cast to the cached-key dtype.
                    if self.keys is not None and m.dtype != self.keys.dtype:
                        m = m.astype(self.keys.dtype)
                    return m
            return None          # no pads anywhere → stock None path
        return None

    # -- structural maintenance (called between steps) ------------------------
    def attach_row(self, k, v, shift: int, W: int):
        """Append one prefilled sequence as a new last row. k,v: [1,H,L,D]."""
        L = k.shape[2]
        cap = max(W + self.STEP, self.cap)
        if self.keys is None:
            self.keys = mx.zeros((1, k.shape[1], cap, k.shape[3]), k.dtype)
            self.values = mx.zeros((1, v.shape[1], cap, v.shape[3]), v.dtype)
        else:
            if self.keys.shape[2] != cap:
                self._grow_cols(cap)
            zr = mx.zeros((1, self.keys.shape[1], self.keys.shape[2],
                           self.keys.shape[3]), self.keys.dtype)
            self.keys = mx.concatenate([self.keys, zr], axis=0)
            self.values = mx.concatenate([self.values, zr], axis=0)
        row = self.keys.shape[0] - 1
        self.keys[row:row + 1, :, shift:shift + L, :] = k
        self.values[row:row + 1, :, shift:shift + L, :] = v

    def drop_rows(self, keep: list[int]):
        if self.keys is None:
            return
        if not keep:
            self.keys = self.values = None
            return
        idx = mx.array(keep)
        self.keys = mx.take(self.keys, idx, axis=0)
        self.values = mx.take(self.values, idx, axis=0)

    def reset(self):
        self.keys = self.values = None

    def _grow_cols(self, cap: int):
        pad_shape = self.keys.shape[:2] + (cap - self.cap, self.keys.shape[3])
        pad = mx.zeros(pad_shape, self.keys.dtype)
        self.keys = mx.concatenate([self.keys, pad], axis=2)
        self.values = mx.concatenate([self.values, pad], axis=2)

    # -- stock-API compat probes (unused by OLMoE, guarded anyway) -----------
    def trim_to_size(self, *a, **k):
        return 0

    def is_trimmable(self):
        return False

    def is_quantized(self):
        return False


class Seq:
    """One in-flight request."""
    __slots__ = ("req_id", "pre_len", "join_W", "q", "n_gen", "done",
                 "max_tokens", "_next_tok", "_first_tok", "_pre_k", "_pre_v",
                 "_raw_k")

    def __init__(self, req_id: str, out_q: queue.Queue):
        self.req_id = req_id
        self.pre_len = 0
        self.join_W = 0
        self.q = out_q
        self.n_gen = 0
        self.done = False
        self.max_tokens = 50
        self._next_tok: int | None = None
        self._first_tok: int | None = None
        self._pre_k = None
        self._pre_v = None
        self._raw_k = None


# ═══════════════════════════════════════════════════════════════════════════
# Engine: solo prefill → batched decode
# ═══════════════════════════════════════════════════════════════════════════

def reposition_keys(k, shift: int, base: float, scale: float = 1.0):
    """Move a post-rope cached key block from positions 0..L-1 to
    shift..shift+L-1 EXACTLY: un-rope by θ_p, re-rope by θ_{p+shift}.
    (Naive rope(k, offset=shift) would yield θ_{2p+shift} — wrong.)
    NeoX half-split layout, computed in float32, non-traditional only."""
    if shift == 0:
        return k
    L = k.shape[2]
    D = k.shape[3]
    half = D // 2
    inv = 1.0 / (base ** (mx.arange(0, D, 2, dtype=mx.float32) / D))   # [half]
    p = mx.arange(L, dtype=mx.float32)[:, None]
    ang0 = p * inv
    ang1 = (p + shift) * inv * scale
    ang0 = ang0 * scale
    c0, s0 = mx.cos(ang0)[None, None], mx.sin(ang0)[None, None]
    c1, s1 = mx.cos(ang1)[None, None], mx.sin(ang1)[None, None]
    x = k.astype(mx.float32)
    x1, x2 = x[..., :half], x[..., half:]
    u1 = x1 * c0 + x2 * s0          # inverse rotation
    u2 = x2 * c0 - x1 * s0
    y1 = u1 * c1 - u2 * s1          # forward rotation at new angles
    y2 = u2 * c1 + u1 * s1
    return mx.concatenate([y1, y2], axis=-1).astype(k.dtype)


class Engine(threading.Thread):
    def __init__(self, arm: PinnedOlmoe, eos_ids: set[int], max_batch: int,
                 max_tokens_default: int, stats_path: Path | None):
        super().__init__(daemon=True, name="engine")
        self.arm = arm
        self.model = arm.model
        self.tok = arm.tokenizer
        self.eos_ids = eos_ids
        self.max_batch = max_batch
        self.max_tokens_default = max_tokens_default
        self.n_layers = arm.n_layers

        self.pending: list[tuple[Seq, mx.array]] = []
        self.shared = SharedState()
        self.layer_caches: list[BatchKVCache] = []
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        # v3: cap how many sequences we prefill per reconcile pass so a burst of
        # new arrivals cannot serially stall in-flight decodes. 0/unset = eager
        # (pre-v3 behavior: prefill every admissible row this pass).
        self.prefill_budget = int(os.environ.get("PINNED_PREFILL_BUDGET", "0"))
        self.stats = {"steps": 0, "prefills": 0, "reconciles": 0,
                      "batch_hist": {}, "tokens_total": 0}
        self.stats_path = stats_path

    def submit(self, seq: Seq, prompt_ids: mx.array):
        with self.lock:
            self.pending.append((seq, prompt_ids))

    # ---- solo prefill (stock path, exact numerics) --------------------------
    def _prefill(self, seq: Seq, ids: mx.array) -> int:
        from mlx_lm.models.cache import KVCache
        caches = [KVCache() for _ in range(self.n_layers)]

        class Cap:
            def __init__(self, inner):
                self.inner, self.calls = inner, []
            def __call__(self, x, offset=0):
                self.calls.append(x)        # [q_pre, k_pre]
                return self.inner(x, offset=offset)

        caps = []
        for blk in self.model.model.layers:
            c = Cap(blk.self_attn.rope)
            blk.self_attn.rope = c
            caps.append(c)
        try:
            x = mx.expand_dims(ids, 0)                     # [1, L]
            logits = self.model(x, caches)
            first = int(mx.argmax(logits[:, -1], axis=-1).item())
            mx.eval(logits)
        finally:
            for blk, c in zip(self.model.model.layers, caps):
                blk.self_attn.rope = c.inner
        seq._pre_k = [c.keys[..., :c.offset, :] for c in caches]
        seq._pre_v = [c.values[..., :c.offset, :] for c in caches]
        seq._raw_k = [c.calls[1] for c in caps]            # pre-rope keys
        seq.pre_len = int(caches[0].offset)
        seq._first_tok = first
        self.stats["prefills"] += 1
        return first

    # ---- membership maintenance ---------------------------------------------
    def _reconcile(self) -> bool:
        """Reap finished rows, admit pending ones. Returns True if any row
        was attached (so the caller can expect progress)."""
        sh = self.shared
        self.stats["reconciles"] = self.stats.get("reconciles", 0) + 1

        alive = [r for r in sh.rows if not r.done]
        if sh.rows and not alive:                       # batch drained: reset
            sh.rows = []
            sh.W = 0
            self.layer_caches = []
        elif len(alive) != len(sh.rows):                # partial dropoff
            keep = [i for i, r in enumerate(sh.rows) if not r.done]
            for c in self.layer_caches:
                c.drop_rows(keep)
            sh.rows = alive

        with self.lock:
            items = list(self.pending)
        attach: list[Seq] = []
        remaining: list[tuple[Seq, mx.array | None]] = []

        if not sh.rows and items:
            # fresh batch: take a group, set world origin = max prefill len.
            # v3 budget caps prefills this pass; the rest stay pending and are
            # admitted on later passes while decode keeps stepping.
            take = self.max_batch if self.prefill_budget <= 0 else min(self.max_batch, self.prefill_budget)
            group = items[:take]
            remaining = items[take:]
            for s, ids in group:
                if s._first_tok is None:
                    self._prefill(s, ids)
                attach.append(s)
            if attach:
                sh.W = max(s.pre_len for s in attach)
        elif items:
            room = self.max_batch - len(sh.rows)
            if self.prefill_budget > 0:
                room = min(room, self.prefill_budget)
            for s, ids in items:
                if len(attach) >= room:
                    remaining.append((s, ids))
                    continue
                if s._first_tok is None:
                    self._prefill(s, ids)
                if s.pre_len <= sh.W:
                    attach.append(s)          # fits left-padded at W
                else:
                    remaining.append((s, None))   # parked; prefill kept
        with self.lock:
            processed = {id(s) for s, _ in items}
            arrived_later = [x for x in self.pending if id(x[0]) not in processed]
            self.pending = remaining + arrived_later

        rope0 = self.model.model.layers[0].self_attn.rope
        assert not getattr(rope0, "traditional", False), \
            "reposition assumes NeoX (non-traditional) rope"
        base = float(getattr(rope0, "base", 10000.0))
        for seq in attach:
            first = seq._first_tok
            if seq.done:                       # client vanished pre-attach
                seq.q.put(("done", None))
                continue
            if first in self.eos_ids or seq.max_tokens <= 1:
                seq.q.put(("tok", first))
                seq.q.put(("done", None))
                seq.done = True
                continue
            seq.join_W = sh.W
            shift = sh.W - seq.pre_len
            seq._next_tok = first
            seq.q.put(("tok", first))
            seq.n_gen += 1
            sh.rows.append(seq)
            if not self.layer_caches:                   # fresh batch
                self.layer_caches = [BatchKVCache(sh)
                                     for _ in range(self.n_layers)]
            for li in range(self.n_layers):
                raw = seq._raw_k[li]
                # ONE exact rotation from the pre-rope capture == what a
                # native shifted prefill would have cached (no double-round)
                k = rope0(raw, offset=shift)
                self.layer_caches[li].attach_row(
                    k, seq._pre_v[li], shift, sh.W)
            del seq._pre_k, seq._pre_v, seq._raw_k
        sh._mask_W = -1                                 # invalidate mask
        return bool(attach)

    # ---- one batched decode step ---------------------------------------------
    def _step(self):
        sh = self.shared
        rows = sh.rows
        x = mx.array([[r._next_tok] for r in rows])     # [B,1]
        logits = self.model(x, self.layer_caches)
        toks = mx.argmax(logits[:, -1], axis=-1).tolist()
        sh.W += 1
        sh._mask_W = -1
        for r, t in zip(rows, toks):
            r._next_tok = int(t)
            r.q.put(("tok", int(t)))
            r.n_gen += 1
            if t in self.eos_ids or r.n_gen >= r.max_tokens:
                r.done = True
                r.q.put(("done", None))
        self.stats["steps"] += 1
        self.stats["tokens_total"] += len(rows)
        key = str(len(rows))                 # TRUE batch size (was clamped at 8)
        self.stats["max_batch_seen"] = max(self.stats.get("max_batch_seen", 0), len(rows))
        self.stats["batch_hist"][key] = \
            self.stats["batch_hist"].get(key, 0) + 1

    # ---- main loop ------------------------------------------------------------
    def run(self):
        while not self.stop_flag.is_set():
            sh = self.shared
            with self.lock:
                has_pending = bool(self.pending)
            if has_pending or any(r.done for r in sh.rows):
                self._reconcile()
            if not sh.rows:
                time.sleep(0.003)
                continue
            self._step()
        for r in self.shared.rows:
            r.q.put(("done", None))
        self._flush_stats()

    def _flush_stats(self):
        if self.stats_path:
            try:
                out = dict(self.stats)
                pool = dict(getattr(getattr(self.arm, "pool", None), "stats", {}) or {})
                resident = pool.get("stack_hits", 0) + pool.get("pin_hits", 0)
                req = resident + pool.get("cold_hits", 0) + pool.get("cold_misses", 0)
                pool["served_requests"] = req
                pool["resident_hit_rate"] = round(resident / req, 4) if req else None
                pool["mean_cold_fetch_ms"] = round(
                    1000.0 * pool.get("cold_seconds", 0.0) / pool["cold_misses"], 2) \
                    if pool.get("cold_misses") else 0.0
                pool["cold_streamed_GB"] = round(pool.get("cold_bytes", 0) / 1e9, 3)
                out["pool"] = pool
                out["pin_cfg"] = getattr(self.arm, "pin_cfg", None)
                try:
                    out["resident_mb"] = round(self.arm.resident_total_mb(), 1)
                except Exception:
                    pass
                self.stats_path.write_text(json.dumps(out, indent=2))
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# HTTP layer (stdlib; OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════════

def build_http_server(engine: Engine, model_name: str):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                self._json(200, {"object": "list", "data":
                                 [{"id": model_name, "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                msgs = req.get("messages", [])
                max_tok = int(req.get("max_tokens")
                              or engine.max_tokens_default)
                prompt = engine.tok.apply_chat_template(
                    msgs, add_generation_prompt=True)
                ids = engine.tok.encode(prompt) \
                    if isinstance(prompt, str) else list(prompt)
            except Exception as e:
                return self._json(400, {"error": f"bad request: {e}"[:200]})

            rid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
            q: queue.Queue = queue.Queue()
            seq = Seq(rid, q)
            seq.max_tokens = max(1, max_tok)
            engine.submit(seq, mx.array(ids))

            stream = bool(req.get("stream", True))
            deadline_s = 30 + 30 * max_tok          # generous watchdog bound
            if not stream:
                toks = []
                while True:
                    try:
                        kind, payload = q.get(timeout=deadline_s)
                    except queue.Empty:
                        seq.done = True
                        break
                    if kind == "done":
                        break
                    toks.append(payload)
                out = engine.tok.decode(toks)
                return self._json(200, {
                    "id": rid, "object": "chat.completion",
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant",
                                             "content": out},
                                 "finish_reason": "stop"}],
                    "usage": {"completion_tokens": len(toks)}})

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    try:
                        kind, payload = q.get(timeout=deadline_s)
                    except queue.Empty:
                        seq.done = True
                        break
                    if kind == "done":
                        break
                    piece = engine.tok.decode([payload])
                    chunk = {"id": rid, "object": "chat.completion.chunk",
                             "choices": [{"index": 0,
                                          "delta": {"content": piece}}]}
                    self.wfile.write(
                        f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                seq.done = True     # client gone; retire at next reconcile

    return Handler, ThreadingHTTPServer


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", type=Path, required=True)
    ap.add_argument("--experts-dir", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=50)
    ap.add_argument("--memory-limit-gb", type=float,
                    default=float(os.environ.get("PINNED_MEM_GB", 6.0)),
                    help="MLX GPU memory cap; must exceed the pinned model size "
                         "(Stage-0 OLMoE ~5GB -> 6; Qwen3-30B ~17GB -> ~17-18, "
                         "kept under iogpu.wired_limit).")
    ap.add_argument("--stats-out", type=Path, default=None)
    # ── §4 offload controls ────────────────────────────────────────────────
    ap.add_argument("--pin-fraction", type=float, default=1.0,
                    help="share of experts kept RESIDENT; the rest stream cold "
                         "from disk slabs (1.0 == v1/v2 fully-resident)")
    ap.add_argument("--hotrank", type=Path, default=None,
                    help="§6 activations JSON; supplies the MEASURED per-layer "
                         "expert frequency ranking that chooses the hot set")
    ap.add_argument("--cold-capacity", type=int, default=1,
                    help="LRU slots for cold-streamed experts (only meaningful "
                         "when --pin-fraction<1; stays 1 for fully-resident so "
                         "existing v1/v2 numbers are unaffected)")
    args = ap.parse_args()

    partial = args.pin_fraction < 0.999
    if partial and args.hotrank is None:
        raise SystemExit(
            "§4 REFUSAL: --pin-fraction<1 needs --hotrank (a §6 measured ranking). "
            "Pinning an arbitrary subset would silently measure a configuration "
            "the appliance would never run.")
    if partial and not args.hotrank.exists():
        raise SystemExit(f"§4 REFUSAL: --hotrank not found: {args.hotrank}")

    arm = PinnedOlmoe(args.ref_dir, args.experts_dir,
                      cold_capacity=(args.cold_capacity if partial else 1),
                      memory_limit_gb=args.memory_limit_gb)
    t0 = time.perf_counter()
    if not partial:
        arm.pin_all(arm.num_experts)
        pin_cfg = {"pin_fraction": 1.0, "mode": "fully-resident"}
    else:
        import math
        act = json.loads(args.hotrank.read_text())
        rank_of = {int(k): (v.get("ranked_experts") or v.get("top_experts") or [])
                   for k, v in act["layers"].items()}
        n_keep = max(1, math.ceil(args.pin_fraction * arm.num_experts))
        per_layer = {}
        for l in arm.moe_layer_ids:
            ranked = list(rank_of.get(l, []))
            if len(ranked) < arm.num_experts:      # never-observed experts pad last
                seen = set(ranked)
                ranked += [e for e in range(arm.num_experts) if e not in seen]
            per_layer[l] = sorted(ranked[:n_keep])
        arm.pin_experts(per_layer)
        # Give the pool the full ranking + resident count so cross-step
        # predictive prefetch (D-042) can warm the next tier.
        arm.pool.set_ranking(rank_of, {l: n_keep for l in arm.moe_layer_ids})
        pin_cfg = {"pin_fraction": args.pin_fraction, "mode": "partial-offload",
                   "resident_experts_per_layer": n_keep, "total_resident": n_keep * len(per_layer),
                   "hot_source": str(args.hotrank), "cold_capacity": args.cold_capacity}
    arm.pin_cfg = pin_cfg
    print(f"pinned {pin_cfg['mode']} P={pin_cfg['pin_fraction']} in "
          f"{time.perf_counter()-t0:.1f}s; active={mx.get_active_memory()/1e6:.0f}MB",
          flush=True)

    tok = arm.tokenizer
    # --- numerics-guarded mx.compile of the MoE block (PINNED_COMPILE=1) ----
    # v2 lever. Design D-022 said "+mx.compile decode", but that is IMPOSSIBLE
    # on mlx 0.32: mx.compile rejects the stateful KVCache argument
    # ("Function arguments must be trees of arrays or constants... received
    # type mlx_lm.models.cache.KVCache" — measured at startup, 2026-09-25).
    # So v2 compiles the fully-resident MoE block instead (arrays in/out), the
    # dominant per-token cost (48 layers x 3 gather_qmm). Hard gate: eager vs
    # compiled must agree BIT-EXACT on a per-layer random probe, else we run
    # eager — guaranteeing v1<->v2 numbers stay numerically comparable.
    if os.environ.get("PINNED_COMPILE") == "1":
        ok, detail, n_on = False, "", 0
        try:
            for l in arm.moe_layer_ids:
                mlp = arm.model.layers[l].mlp.switch_mlp
                if not mlp.enable_compile():
                    detail = f"L{l} not fully resident — MoE block not compilable"
                    break
                x = mx.random.normal((8, mlp.hidden)).astype(mx.bfloat16)
                idx = mx.random.randint(0, mlp.num_experts, (8, getattr(mlp, "top_k", 8)))
                mlp.disable_compile(); ea = mlp(x, idx)
                mlp.enable_compile();    cb = mlp(x, idx)
                mx.eval(ea, cb)
                d = float(mx.max(mx.abs(ea.astype(mx.float32)
                                       - cb.astype(mx.float32))).item())
                if d != 0.0:
                    detail = f"L{l} eager!=compiled Δ={d}"
                    break
                n_on += 1
            else:
                ok = True
            if ok:
                print(f"PINNED_COMPILE=ON ({n_on} MoE blocks compiled, "
                      f"bit-exact guard passed)", flush=True)
            else:
                for l in arm.moe_layer_ids:
                    arm.model.layers[l].mlp.switch_mlp.disable_compile()
                print(f"PINNED_COMPILE=OFF ({detail}) — running eager", flush=True)
        except Exception as e:
            for l in arm.moe_layer_ids:
                try:
                    arm.model.layers[l].mlp.switch_mlp.disable_compile()
                except Exception:
                    pass
            import traceback
            tb = traceback.format_exc().strip().splitlines()[-1]
            print(f"compile unavailable ({type(e).__name__}: {str(e)[:120]}) "
                  f"— eager | {tb[:160]}", flush=True)

    eos = set(getattr(tok, "eos_token_ids", None) or {tok.eos_token_id})

    engine = Engine(arm, eos, args.max_batch, args.max_tokens, args.stats_out)
    engine.start()

    Handler, Server = build_http_server(engine, "pinned-olmoe")
    Server.request_queue_size = 256   # default backlog 5 -> RSTs overflow conns
    srv = Server(("127.0.0.1", args.port), Handler)
    print(f"pinned_server listening on :{args.port}", flush=True)

    def shutdown(*_):
        print("pinned_server: shutting down — flushing stats", flush=True)
        engine.stop_flag.set()
        engine._flush_stats()
        sys.exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    srv.serve_forever()


if __name__ == "__main__":
    main()
