# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Mug and Bewong Ltd (Company No. 14916888).
# DeeWaanAI(TM) is a trademark of Mug and Bewong Ltd. See LICENSE.
#!/usr/bin/env python3
"""
Stage 0 — pinned expert arm (Metal/MLX).

Port of the v4_engine lineage mechanisms to Apple Silicon:
  1. EXPERT SEPARATION — every expert lives in its own small safetensors slab
     on disk (produced by split_olmoe_experts.py); the stacked switch_mlp
     tensors are dropped from the model at build time.
  2. HOT-EXPERT PINNING — a configured set of experts is materialized once
     and held for the whole session; cold experts are lazy-loaded on router
     selection into a bounded LRU. mx.set_memory_limit caps the working set
     so the owner's 16 GB laptop can never be swamped.
  3. AFFINITY DISPATCH — hooks kept for H3 (rental stage): PinnedSwitchMLP
     accepts per-request domain labels via the block; single-user Stage 0
     degenerates to plain top-k routing.

Design guarantee: the backbone (embedding, attention, routers, norms,
lm_head) is the STOCK mlx_lm OLMoE module loaded verbatim — only
`mlp.switch_mlp` is replaced. So router numerics (precise softmax,
argpartition top-k, raw score weighting) are byte-identical to the
reference arm; the pinning/lazy machinery is the only variable.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Predictive prefetch (D-041): overlap the SSD cold-expert reads that cause the
# streaming cliff. Worker threads do pure-numpy safetensors file reads (thread-
# safe, no MLX); the main thread does the numpy->mx conversion. The native
# router stays authoritative, so prefetch can only waste I/O — never change
# which experts run or their outputs. Off unless PREFETCH=1.
import os as _pf_os
PREFETCH_ENABLED = _pf_os.environ.get("PREFETCH") == "1"
PREFETCH_WORKERS = int(_pf_os.environ.get("PREFETCH_WORKERS", "8"))
# Cross-step predictive prefetch (D-042): after each step, warm the next-tier
# experts (just outside the resident set) in a background thread so the NEXT
# step's likely selections hit cache instead of blocking on a disk read.
CROSS_STEP = _pf_os.environ.get("CROSS_STEP") == "1" and PREFETCH_ENABLED
PREFETCH_LOOKAHEAD = int(_pf_os.environ.get("PREFETCH_LOOKAHEAD", "12"))
# I/O budgeting (D-045): cap speculative background reads PER DECODE STEP and
# pause the worker while the critical path is doing real cold I/O, so an
# accurate predictor cannot starve the requests actually being served.
PREFETCH_BUDGET = int(_pf_os.environ.get("PREFETCH_BUDGET", "16"))
import threading as _threading
import queue as _queue

# Import custom Metal kernel (if available)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "metal_kernel"))
    import metal_kernel
    CUSTOM_METAL_KERNEL_AVAILABLE = True
    if __import__("os").environ.get("DISABLE_METAL_KERNEL") == "1":
        CUSTOM_METAL_KERNEL_AVAILABLE = False  # force validated mx.gather_qmm path

    # Initialize the Metal kernel
    if CUSTOM_METAL_KERNEL_AVAILABLE:
        metallib_path = str(Path(__file__).parent.parent.parent / "metal_kernel" / "efficient_expert_gather.metallib")
        if Path(metallib_path).exists():
            metal_kernel.initialize_metal(metallib_path)
            print(f"Custom Metal kernel initialized from {metallib_path}")
        else:
            print(f"Warning: Metal kernel metallib not found at {metallib_path}")
            CUSTOM_METAL_KERNEL_AVAILABLE = False
except ImportError as e:
    CUSTOM_METAL_KERNEL_AVAILABLE = False
    print(f"Custom Metal kernel not available: {e}")

from mlx_lm.models.cache import KVCache

# Debug: FORCE_PARTIAL_PATH=1 routes an all-resident model through the partial
# branch (zero cold misses) to tell "cold kernel differs" apart from "the
# partial branch itself differs". Inert unless the env var is set.
import os as _os
_FORCE_PARTIAL = _os.environ.get("FORCE_PARTIAL_PATH") == "1"
# D-031 fix: serve cold slots with the SAME kernel as the resident stack/stock.
# Set COLD_GATHER_QMM=0 to reproduce the pre-fix quantized_matmul behavior.
_COLD_GATHER_QMM = _os.environ.get("COLD_GATHER_QMM", "1") == "1"
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort


# ═══════════════════════════════════════════════════════════════════════════
# Per-expert module (mirrors stock quantized FFN semantics)
# ═══════════════════════════════════════════════════════════════════════════

class Expert(nn.Module):
    """down( silu(gate(x)) * up(x) ) — matches SwitchGLU + HF OLMoE."""

    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    @staticmethod
    def _qlinear(w, scales, biases, in_f, out_f):
        # w: uint32-packed [out, in/8]; scales/biases: [out, in/group]
        # NOTE: slab 'biases' is the affine quantization zero-point (per-group),
        # NOT a per-output additive bias. Always bias=False for OLMoE.
        group = in_f // scales.shape[1]
        q = nn.QuantizedLinear(in_f, out_f, bias=False,
                               group_size=group, bits=4, mode="affine")
        q.weight = w
        q.scales = scales
        if biases is not None:
            q.biases = biases
        mx.eval(q.weight, q.scales)
        if biases is not None:
            mx.eval(q.biases)
        return q

    @classmethod
    def from_slab(cls, slab: dict, hidden: int, inter: int) -> "Expert":
        ex = cls(hidden, inter)
        for name, out_f, in_f in (("gate_proj", inter, hidden),
                                  ("up_proj", inter, hidden),
                                  ("down_proj", hidden, inter)):
            w = slab[f"{name}.weight"]
            s = slab.get(f"{name}.scales")
            b = slab.get(f"{name}.biases")
            if w.dtype in (mx.uint32, mx.uint8) and s is not None:
                setattr(ex, name, cls._qlinear(w, s, b, in_f, out_f))
            else:
                lin = nn.Linear(in_f, out_f, bias=False)
                lin.weight = w
                setattr(ex, name, lin)
        return ex

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ═══════════════════════════════════════════════════════════════════════════
# Expert pool: pinned set + bounded LRU + usage stats
# ═══════════════════════════════════════════════════════════════════════════

class ExpertPool:
    def __init__(self, experts_dir: Path, hidden: int, inter: int,
                 cold_capacity: int = 256):
        self.dir = Path(experts_dir)
        self.hidden, self.inter = hidden, inter
        self.pinned: dict[tuple[int, int], Expert] = {}
        self.cold: OrderedDict[tuple[int, int], Expert] = OrderedDict()
        self.cold_capacity = cold_capacity
        self.usage: dict[tuple[int, int], int] = {}
        self.stats = {"loads": 0, "cold_hits": 0, "pin_hits": 0,
                      "loaded_bytes": 0, "evictions": 0,
                      # §4 offload metrics: filled ONLY on a true disk miss, so
                      # fully-resident (v1/v2) runs never touch this path.
                      "cold_misses": 0, "cold_seconds": 0.0, "cold_bytes": 0,
                      # Selections served by the resident STACK. build_resident
                      # pops pool.pinned entries once stacked, so pin_hits alone
                      # would report 0% residency for every partial run.
                      "stack_hits": 0}
        self._path_cache: dict[tuple[int, int], Path] = {}
        # Cross-step predictive prefetch state (D-042).
        self._rank_of: dict[int, list[int]] = {}
        self._n_keep: dict[int, int] = {}
        self._pq: _queue.Queue = _queue.Queue()
        self._queued: set[tuple[int, int]] = set()
        self._qlock = _threading.Lock()
        self._staged: dict[tuple[int, int], dict] = {}   # key -> torch tensors (worker)
        self._slock = _threading.Lock()
        self._pref_stop = _threading.Event()
        self._pref_thread = None
        # I/O budgeting (D-045): speculative reads left this step + critical flag.
        self._spec_left = PREFETCH_BUDGET
        self._critical = False

    def set_ranking(self, rank_of: dict[int, list[int]], n_keep: dict[int, int]):
        """Give the pool the per-layer traffic ranking so it can predict the
        next tier to warm. Starts the background prefetcher when CROSS_STEP."""
        self._rank_of = rank_of
        self._n_keep = n_keep
        if CROSS_STEP and self._pref_thread is None:
            self._pref_thread = _threading.Thread(target=self._prefetch_worker, daemon=True)
            self._pref_thread.start()

    def _prefetch_worker(self):
        """Background: read predicted slabs (torch, thread-safe) into _staged.
        Never touches MLX. The main thread converts staged->mx on drain()."""
        while not self._pref_stop.is_set():
            # I/O budgeting (D-045): never compete with the critical path's real
            # cold reads; yield until it finishes.
            if self._critical:
                time.sleep(0.002)
                continue
            try:
                key = self._pq.get(timeout=0.2)
            except _queue.Empty:
                continue
            with self._qlock:
                self._queued.discard(key)
            l, e = key
            if key in self.pinned or key in self.cold:
                continue
            try:
                with self._slock:
                    present = key in self._staged
                if present:
                    continue
                td = self._read_slab(l, e)
                with self._slock:
                    if len(self._staged) < 2 * self.cold_capacity:
                        self._staged[key] = td
                        self.stats["prefetch_staged"] = self.stats.get("prefetch_staged", 0) + 1
            except Exception:
                pass  # a failed speculative read is harmless

    def submit_predictive(self, layer_id: int, candidates=None):
        """Enqueue likely-next experts for background warming.

        candidates (D-044 temporal/context predictor): explicit expert ids the
        caller believes are likely next step (e.g. this step's selections + the
        running per-layer context top-k). If None, fall back to the static
        hotrank next-tier (D-042)."""
        if not CROSS_STEP:
            return
        if candidates is None:
            ranked = self._rank_of.get(layer_id)
            if not ranked:
                return
            keep = self._n_keep.get(layer_id, 0)
            candidates = ranked[keep: keep + PREFETCH_LOOKAHEAD]
        for e in candidates:
            if self._spec_left <= 0:      # per-step speculative I/O budget (D-045)
                break
            key = (layer_id, int(e))
            if key in self.pinned or key in self.cold:
                continue
            with self._qlock:
                if key in self._queued:
                    continue
                self._queued.add(key)
            self._pq.put(key)
            self._spec_left -= 1

    def drain_staged(self):
        """Main thread: move worker-staged torch tensors into the cold LRU as mx
        arrays (byte-identical to mx.load). Called at the top of a decode step."""
        if not CROSS_STEP:
            return
        self._spec_left = PREFETCH_BUDGET   # fresh speculative budget each step
        with self._slock:
            items = list(self._staged.items()); self._staged.clear()
        for (l, e), td in items:
            if (l, e) in self.pinned or (l, e) in self.cold:
                continue
            slab = self._slab_to_mx(*td)
            self.cold[(l, e)] = Expert.from_slab(slab, self.hidden, self.inter)
            self.stats["loads"] += 1
            self.stats["prefetch_hits"] = self.stats.get("prefetch_hits", 0) + 1
        while len(self.cold) > self.cold_capacity:
            self.cold.popitem(last=False)
            self.stats["evictions"] += 1

    def slab_path(self, layer: int, expert: int) -> Path:
        """Resolve the on-disk slab path across splitter padding conventions.

        The legacy OLMoE splitter wrote `layer{LL}_expert{EE}` (2-digit expert)
        while the generalized split_moe_experts.py writes `layer{LL}_expert{EEE}`
        (3-digit expert). Probe the known widths and cache the match; fall back
        to the canonical 3-digit form for paths not yet materialized."""
        key = (layer, expert)
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached
        for ew in (3, 2):
            cand = self.dir / f"layer{layer:02d}_expert{expert:0{ew}d}.safetensors"
            if cand.exists():
                self._path_cache[key] = cand
                return cand
        default = self.dir / f"layer{layer:02d}_expert{expert:03d}.safetensors"
        self._path_cache[key] = default
        return default

    def _load(self, layer: int, expert: int) -> Expert:
        path = self.slab_path(layer, expert)
        slab = mx.load(str(path))
        self.stats["loads"] += 1
        self.stats["loaded_bytes"] += path.stat().st_size
        return Expert.from_slab(slab, self.hidden, self.inter)

    def pin(self, layer: int, expert: int):
        key = (layer, expert)
        if key not in self.pinned:
            self.pinned[key] = self._load(layer, expert)

    def get(self, layer: int, expert: int) -> Expert:
        key = (layer, expert)
        self.usage[key] = self.usage.get(key, 0) + 1
        if key in self.pinned:
            self.stats["pin_hits"] += 1
            return self.pinned[key]
        if key in self.cold:
            self.stats["cold_hits"] += 1
            self.cold.move_to_end(key)
            return self.cold[key]
        t_cold = time.perf_counter()
        ex = self._load(layer, expert)
        dt = time.perf_counter() - t_cold
        self.stats["cold_misses"] += 1
        self.stats["cold_seconds"] += dt
        try:
            self.stats["cold_bytes"] += self.slab_path(layer, expert).stat().st_size
        except OSError:
            pass
        self.cold[key] = ex
        while len(self.cold) > self.cold_capacity:
            self.cold.popitem(last=False)
            self.stats["evictions"] += 1
        return ex

    def resident_mb(self) -> float:
        keys = list(self.pinned) + list(self.cold)
        return sum(self.slab_path(l, e).stat().st_size for (l, e) in keys) / 1e6

    # ---- predictive prefetch (D-041) --------------------------------------
    # safetensors dtype -> (numpy view dtype, mlx dtype). bf16/u32 have no direct
    # numpy dtype, so we read the raw bytes as a same-itemsize numpy type and
    # .view() into the true MLX dtype — byte-identical to mx.load, no arithmetic.
    _ST_DTYPE = {
        "BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, mx.float16),
        "F32": (np.float32, mx.float32), "F64": (np.float64, mx.float64),
        "U32": (np.uint32, mx.uint32), "I32": (np.int32, mx.int32),
        "I64": (np.int64, mx.int64), "U8": (np.uint8, mx.uint8),
        "I8": (np.int8, mx.int8), "BOOL": (np.uint8, mx.bool_),
    }

    def _read_slab(self, layer: int, expert: int):
        """WORKER thread: pure file I/O + header parse. No MLX (not thread-safe),
        no torch arithmetic. Returns (bytes, header, data_start). The disk read —
        the actual bottleneck — runs here, off the critical path."""
        import json as _json, struct
        with open(str(self.slab_path(layer, expert)), "rb") as fh:
            buf = fh.read()
        hlen = struct.unpack("<Q", buf[:8])[0]
        header = _json.loads(buf[8:8 + hlen])
        return buf, header, 8 + hlen

    def _slab_to_mx(self, buf, header, data_start):
        """MAIN thread: zero-copy numpy-frombuffer + MLX view. No conversion
        arithmetic, so it is ~as cheap as mx.load (which cannot run off-thread)."""
        out = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            npdt, mxdt = self._ST_DTYPE[info["dtype"]]
            shape = info["shape"]
            s, e = info["data_offsets"]
            cnt = 1
            for d in shape:
                cnt *= d
            arr = np.frombuffer(buf, dtype=npdt, count=cnt,
                                offset=data_start + s).reshape(shape)
            a = mx.array(arr)
            if a.dtype != mxdt:
                a = a.view(mxdt)
            out[name] = a
        return out

    @staticmethod
    def _t2mx(t):  # legacy torch path kept for reference; unused by fast reader
        import torch
        if t.dtype == torch.bfloat16:
            return mx.array(t.to(torch.float32).numpy()).astype(mx.bfloat16)
        return mx.array(t.numpy())

    def _ensure_executor(self):
        if getattr(self, "_executor", None) is None:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS)
        return self._executor

    def prefetch_parallel(self, pairs):
        """Overlap the SSD reads for a set of cold experts needed THIS step.
        Worker threads do the file I/O concurrently (torch, no MLX); the main
        thread does the torch->mx conversion and builds Expert objects into the
        cold LRU. Correctness-neutral: loads exactly the experts the router
        already chose, just concurrently instead of serially."""
        if not PREFETCH_ENABLED:
            return
        todo = [p for p in pairs if p not in self.pinned and p not in self.cold]
        if not todo:
            return
        ex = self._ensure_executor()
        t0 = time.perf_counter()
        self._critical = True          # pause speculative reads (D-045)
        try:
            results = list(ex.map(lambda pe: self._read_slab(*pe), todo))
        finally:
            self._critical = False
        for (layer, expert), td in zip(todo, results):
            slab = self._slab_to_mx(*td)
            self.cold[(layer, expert)] = Expert.from_slab(slab, self.hidden, self.inter)
            self.stats["loads"] += 1
            self.stats["prefetched"] = self.stats.get("prefetched", 0) + 1
        while len(self.cold) > self.cold_capacity:
            self.cold.popitem(last=False)
            self.stats["evictions"] += 1
        self.stats["prefetch_seconds"] = self.stats.get("prefetch_seconds", 0.0) + (time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════════════
# Resident stack: pinned slabs assembled into gather-dispatch buffers
# ═══════════════════════════════════════════════════════════════════════════

class _StackProj(nn.Module):
    """gather-dispatch projection over a resident stack of experts
    (same kernel + argument order as stock QuantizedSwitchLinear)."""

    def __init__(self):
        super().__init__()
        self.weight = None
        self.scales = None
        self.biases = None
        self.group_size: int = 64
        self.bits: int = 4
        self.mode: str = "affine"

    def set_stacked(self, w, s, b, group: int):
        """Assign the stacked tensors and force device materialization."""
        self.weight, self.scales, self.biases = w, s, b
        self.group_size = group
        mx.eval(w, s, b)

    def __call__(self, x, indices, sorted_indices: bool = False):
        """Same signature/contract as stock QuantizedSwitchLinear.__call__.
        `sorted_indices` MUST mirror stock: it selects the segmented gather-qmm
        kernel, whose accumulation order differs from the generic path. Omitting
        it (default False) while stock passed True produced a ~1e-6 divergence
        at large prefill — see test_t2_switch_equivalence.py (T2 gate)."""
        
        # Use custom Metal kernel if available
        if CUSTOM_METAL_KERNEL_AVAILABLE and not sorted_indices:
            try:
                # Extract dimensions
                # x shape: [batch, 1, 1, in_dim] or [batch, in_dim]
                if x.ndim == 4:
                    batch_size = x.shape[0]
                    in_dim = x.shape[3]
                    x_flat = x.reshape(batch_size, in_dim)
                else:
                    batch_size = x.shape[0]
                    in_dim = x.shape[1]
                    x_flat = x
                
                # weights shape: [num_experts, out_dim, in_dim_packed]
                num_experts = self.weight.shape[0]
                out_dim = self.weight.shape[1]
                in_dim_packed = self.weight.shape[2]
                
                # indices shape: [batch, k] or [T, k]
                k = indices.shape[-1]
                
                # Convert to numpy and extract pointers
                x_np = np.ascontiguousarray(np.array(x_flat))
                weights_np = np.ascontiguousarray(np.array(self.weight))
                scales_np = np.ascontiguousarray(np.array(self.scales))
                biases_np = np.ascontiguousarray(np.array(self.biases))
                indices_np = np.ascontiguousarray(np.array(indices))
                
                x_ptr = x_np.__array_interface__['data'][0]
                weights_ptr = weights_np.__array_interface__['data'][0]
                scales_ptr = scales_np.__array_interface__['data'][0]
                biases_ptr = biases_np.__array_interface__['data'][0]
                indices_ptr = indices_np.__array_interface__['data'][0]
                
                # Call custom kernel
                result = metal_kernel.efficient_expert_gather_qmm(
                    x_ptr, weights_ptr, scales_ptr, biases_ptr, indices_ptr,
                    batch_size, k, out_dim, in_dim, num_experts, 
                    self.group_size, self.bits
                )
                
                # Keep references to prevent garbage collection
                self._last_refs = (x_np, weights_np, scales_np, biases_np, indices_np)
                
                return result
            except Exception as e:
                # Fall back to mx.gather_qmm on error
                print(f"Custom Metal kernel failed: {e}, falling back to mx.gather_qmm")
        
        # Default: use mx.gather_qmm
        return mx.gather_qmm(
            x, self.weight, self.scales, self.biases,
            rhs_indices=indices, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
            sorted_indices=sorted_indices)


class ResidentSwitch(nn.Module):
    """Stock SwitchGLU numerics over a RESIDENT SUBSET of experts: weights
    stacked once from pinned slabs; a device LUT maps global expert ids to
    stack slots. Routing/score weighting stay entirely on-device (no per-
    layer host sync) for the hot path."""

    def __init__(self):
        super().__init__()
        self.up_proj = _StackProj()
        self.gate_proj = _StackProj()
        self.down_proj = _StackProj()

    def __call__(self, x, slot_idx):
        """Mirror stock SwitchGLU contract: x [T,D], slot_idx [T,k] → [T,k,D].
        Includes sort for large batches (prefill), skips for small (decode)."""
        x = mx.expand_dims(x, (-2, -3))  # [T, 1, 1, D]
        do_sort = slot_idx.size >= 64
        idx = slot_idx
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, slot_idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(nn.silu(x_gate) * x_up, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, slot_idx.shape)
        return x.squeeze(-2)


# ═══════════════════════════════════════════════════════════════════════════
# Sparse MoE block: resident-stack fast path + per-expert cold fallback
# ═══════════════════════════════════════════════════════════════════════════

class PinnedSwitchMLP(nn.Module):
    """Drop-in replacement for stock `mlp.switch_mlp`."""

    def __init__(self, pool: ExpertPool, layer_id: int, hidden: int,
                 inter: int, num_experts: int, group: int):
        super().__init__()
        self.pool = pool
        self.layer_id = layer_id
        self.hidden, self.inter = hidden, inter
        self.num_experts = num_experts
        self.group = group
        self.resident: ResidentSwitch | None = None
        self.lut: mx.array | None = None       # device int32 [num_experts]
        self.lut_host: list[int] | None = None  # host mirror for partial path
        self.gid_of_slot: list[int] = []
        self._P: int = 0  # number of stacked resident experts
        self._cfn = None  # v2: mx.compile'd fully-pinned block (None = eager)
        self.stack_bytes: int = 0  # resident bytes of the stacked experts
        # D-044 temporal/context predictor: running per-layer expert usage so
        # cross-step prefetch warms recently-used (likely-reused) experts.
        from collections import Counter as _Counter
        self._ctx = _Counter()

    # ---- v2: compile the MoE block (not the whole forward) ------------------

    def enable_compile(self):
        """Compile the fully-pinned fast path. Legal for mx.compile because it
        is arrays-in/arrays-out — unlike model(x, cache), whose stateful
        KVCache args mlx 0.32 rejects. Shape-specialised (default) so each
        batch size gets its own trace."""
        if self.resident is None or self._P != self.num_experts:
            return False  # partial/offloaded path has host syncs -> not compilable
        rs, lut = self.resident, self.lut
        self._cfn = mx.compile(lambda x, indices: rs(x, lut[indices]))
        return True

    def disable_compile(self):
        self._cfn = None

    # ---- stack construction ------------------------------------------------

    def _assemble_from_tensors(self, ids: list[int],
                               proj_tensors: dict[str, list]):
        """Shared finalization: stack per-proj tensors, build LUT, wire
        ResidentSwitch. `proj_tensors` maps 'up_proj'/'gate_proj'/'down_proj'
        → list of (weight, scales, biases) tuples in `ids` order."""
        P = len(ids)
        group = self.group
        rs = ResidentSwitch()
        for proj_name in ("up_proj", "gate_proj", "down_proj"):
            tensors = proj_tensors[proj_name]
            w = mx.stack([t[0] for t in tensors])
            s = mx.stack([t[1] for t in tensors])
            b = mx.stack([t[2] for t in tensors])
            mx.eval(w, s, b)
            getattr(rs, proj_name).set_stacked(w, s, b, group)
        # Free source tensors
        del proj_tensors
        # Build LUT: global expert id → slot index (-1 = cold)
        lut_np = [-1] * self.num_experts
        for slot, gid in enumerate(ids):
            lut_np[gid] = slot
        self.lut = mx.array(lut_np, dtype=mx.int32)
        self.lut_host = lut_np
        mx.eval(self.lut)
        self.gid_of_slot = ids
        self.resident = rs
        self._P = P
        self._cfn = None  # stack rebuilt -> old compiled closure is stale
        self.stack_bytes = sum(
            int(t.nbytes)
            for p in (rs.up_proj, rs.gate_proj, rs.down_proj)
            for t in (p.weight, p.scales, p.biases) if t is not None)

    def build_resident(self):
        """Stack the currently-pinned Expert modules into gather-dispatch
        buffers, then free the individual Expert objects."""
        ids = sorted(e for (l, e) in self.pool.pinned if l == self.layer_id)
        if not ids:
            return
        proj_tensors: dict[str, list] = {}
        for proj_name in ("up_proj", "gate_proj", "down_proj"):
            triplet = []
            for e in ids:
                params = self.pool.pinned[(self.layer_id, e)][proj_name].parameters()
                triplet.append((params["weight"], params["scales"],
                                params["biases"]))
            proj_tensors[proj_name] = triplet
        self._assemble_from_tensors(ids, proj_tensors)
        # Free individual Expert modules — the stack replaces them
        for e in ids:
            self.pool.pinned.pop((self.layer_id, e), None)

    def build_resident_from_slabs(self, slab_data: dict[int, dict]):
        """Build resident stack directly from raw slab dicts (avoids
        materializing per-expert nn.Module objects → lower peak memory).
        `slab_data` maps global expert id → slab dict (from mx.load)."""
        ids = sorted(slab_data.keys())
        proj_tensors: dict[str, list] = {}
        for proj_name in ("up_proj", "gate_proj", "down_proj"):
            triplet = []
            for gid in ids:
                slab = slab_data[gid]
                triplet.append((slab[f"{proj_name}.weight"],
                                slab[f"{proj_name}.scales"],
                                slab[f"{proj_name}.biases"]))
            proj_tensors[proj_name] = triplet
        self._assemble_from_tensors(ids, proj_tensors)

    # ---- forward -----------------------------------------------------------

    def __call__(self, x, indices):
        """Drop-in replacement for stock SwitchGLU: x [T,D], indices [T,k].
        Some archs (qwen3_moe) pass a leading batch dim (x [B,T,D], indices
        [B,T,k]); stock SwitchGLU tolerates it, so we flatten, recurse on the
        2-D case, and restore the leading shape on the way out."""
        if x.ndim != 2:
            lead = indices.shape[:-1]
            y = self.__call__(x.reshape(-1, x.shape[-1]),
                              indices.reshape(-1, indices.shape[-1]))
            return y.reshape(*lead, y.shape[-2], y.shape[-1])
        T = x.shape[0]
        k = indices.shape[-1]

        # ── FULLY-PINNED fast path: zero host syncs ──
        if (self.resident is not None and self._P == self.num_experts
                and not _FORCE_PARTIAL):   # debug: force partial w/ zero cold misses
            self.pool.stats["stack_hits"] += T * k
            if self._cfn is not None:
                return self._cfn(x, indices)      # v2: compiled MoE block
            return self.resident(x, self.lut[indices])

        # ── PARTIAL-PINNED path: hot gather + cold per-expert fallback ──
        self.pool.drain_staged()   # harvest cross-step prefetched experts (D-042)
        idx_h = indices.tolist()  # single sync point
        lut_host = self.lut_host if self.lut_host is not None else [-1] * self.num_experts
        slot_rows = [[lut_host[e] for e in row] for row in idx_h]

        out = mx.zeros((T * k, self.hidden), dtype=x.dtype)

        # Hot experts: batch via resident stack
        if self.resident is not None:
            hot = mx.array(slot_rows, dtype=mx.int32)
            has_hot = bool(mx.any(hot >= 0).item())
            if has_hot:
                self.pool.stats["stack_hits"] += int(mx.sum(hot >= 0).item())
                # Clamp cold entries to slot 0; mask them out after
                hot_masked = mx.where(hot >= 0, hot, 0)
                y = self.resident(x, hot_masked).reshape(T * k, self.hidden)
                sel = (hot >= 0).reshape(-1).astype(x.dtype)
                out = out + y * sel[:, None]

        # Cold experts: per-expert loop (only those not in the stack)
        groups: dict[int, list[tuple[int, int]]] = {}
        for t in range(T):
            for kk in range(k):
                if slot_rows[t][kk] < 0:
                    groups.setdefault(idx_h[t][kk], []).append((t, kk))
        # Predictive prefetch (D-041): overlap this step's cold-expert SSD reads
        # concurrently instead of serially, so pool.get() below hits the cache.
        if PREFETCH_ENABLED and groups:
            self.pool.prefetch_parallel([(self.layer_id, e) for e in groups])
        # Cross-step (D-044 temporal/context predictor): warm, in the background,
        # the experts this step selected (likely reused next step, recall .38) plus
        # the running per-layer context top-k (recall .35) — NOT the static
        # popularity next-tier (recall .24, D-042), which under-predicts.
        cur = set()
        for row in idx_h:
            cur.update(row)
        self._ctx.update(cur)
        cand = set(cur)
        cand.update(e for e, _ in self._ctx.most_common(PREFETCH_LOOKAHEAD * 2))
        self.pool.submit_predictive(self.layer_id, candidates=cand)
        for e, pos in groups.items():
            ex = self.pool.get(self.layer_id, e)   # load/evict accounting unchanged
            rows = mx.array([p[0] for p in pos])
            slots = mx.array([p[0] * k + p[1] for p in pos])
            xr = mx.take(x, rows, axis=0)
            if _COLD_GATHER_QMM:
                # Same kernel as the resident stack + stock (D-031): each cold
                # projection goes through gather_qmm as a 1-slot stack instead
                # of per-expert quantized_matmul. 'biases' is the affine
                # zero-point here (see Expert._qlinear), matching _StackProj.
                def _qmm(proj, a):
                    return mx.gather_qmm(
                        mx.expand_dims(a, (-2, -3)),
                        mx.expand_dims(proj.weight, 0),
                        mx.expand_dims(proj.scales, 0),
                        mx.expand_dims(proj.biases, 0),
                        rhs_indices=mx.zeros((a.shape[0], 1), dtype=mx.int32),
                        transpose=True, group_size=proj.group_size, bits=proj.bits,
                        mode=getattr(proj, "mode", "affine"),
                        # same rule as ResidentSwitch.do_sort: this flag picks a
                        # different segmented kernel (D-018/D-019), so it must
                        # match the resident path's accumulation order.
                        sorted_indices=(a.shape[0] >= 64)).reshape(a.shape[0], -1)
                y = _qmm(ex.down_proj,
                         nn.silu(_qmm(ex.gate_proj, xr)) * _qmm(ex.up_proj, xr))
            else:
                y = ex(xr)
            out = out.at[slots].add(y)
        return out.reshape(T, k, self.hidden)


# ═══════════════════════════════════════════════════════════════════════════
# Model assembly: stock backbone + swapped MoE
# ═══════════════════════════════════════════════════════════════════════════

def moe_submodule(blk):
    """Return the decoder-block submodule that owns `switch_mlp`.

    Architecture-agnostic lookup: OLMoE exposes the MoE block as `blk.mlp`,
    LFM2-MoE (mlx_lm lfm2_moe) as `blk.feed_forward`, and Mixtral as
    `blk.block_sparse_moe`. All share the SwitchGLU (`switch_mlp`) contract, so
    the pinned gather path applies unchanged once we find the right container.
    Returns None for dense layers."""
    for attr in ("mlp", "feed_forward", "block_sparse_moe"):
        sub = getattr(blk, attr, None)
        if sub is not None and hasattr(sub, "switch_mlp"):
            return sub
    return None


def assert_attention_compatible(model):
    """Fail loudly if any attention layer violates the pinned_server batch-attach
    assumptions (see COMPATIBILITY.md). The world-coordinate rope-shift trick
    requires: per-layer `self_attn.rope`, NeoX (non-traditional) rope, full
    rotary (no partial_rotary_factor < 1), and no sliding/interleaved window.
    Silent wrong-answers on incompatible archs are worse than a crash."""
    for l, blk in enumerate(model.layers):
        attn = getattr(blk, "self_attn", None)
        if attn is None:
            continue  # conv/hybrid layer with no self_attn -> unsupported later
        rope = getattr(attn, "rope", None)
        if rope is None:
            raise RuntimeError(
                f"layer {l}: self_attn has no .rope — pinned_server batch-attach "
                "requires a standard rope'd attention layer (see COMPATIBILITY.md).")
        if getattr(rope, "traditional", False):
            raise RuntimeError(
                f"layer {l}: traditional (HF) rope is unsupported by the "
                "world-coordinate reposition (needs NeoX non-traditional).")
        prf = getattr(getattr(model, "args", None), "partial_rotary_factor", 1.0)
        if prf is not None and prf < 1.0:
            raise RuntimeError(
                f"partial_rotary_factor={prf} unsupported — reposition assumes "
                "full-rotary keys (DeepSeek/DBRX/Moonlight class).")
        if getattr(attn, "sliding_window", None) or getattr(attn, "window_size", None):
            raise RuntimeError(
                f"layer {l}: sliding-window attention is unsupported by the "
                "uniform-KV batch-attach (gpt-oss/Cohere/EXAONE class).")
    return True


class PinnedOlmoe:
    def __init__(self, ref_dir: Path, experts_dir: Path,
                 cold_capacity: int = 256, memory_limit_gb: float = 6.0):
        import json as _json
        import mlx_lm
        mx.set_memory_limit(int(memory_limit_gb * 1e9))  # laptop protection
        model, self.tokenizer = mlx_lm.load(str(ref_dir))
        self.model = model
        assert_attention_compatible(model)  # fail loud on unsupported archs
        cfg = model.args
        self.n_layers = len(model.layers)
        # MoE expert count is named differently per arch (Qwen/OLMoE: num_experts;
        # DeepSeek/Moonlight: n_routed_experts; Mixtral: num_local_experts).
        self.num_experts = (getattr(cfg, "num_experts", None)
                            or getattr(cfg, "n_routed_experts", None)
                            or getattr(cfg, "num_local_experts", None))
        self.experts_dir = Path(experts_dir)

        # Qwen3-MoE stores expert FFN width separately (moe_intermediate_size);
        # OLMoE reuses intermediate_size. group_size comes from config.json;
        # both fall back to the OLMoE defaults → OLMoE path numerically unchanged.
        moe_inter = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
        group = 64
        try:
            cfgj = _json.loads((Path(ref_dir) / "config.json").read_text())
            q = cfgj.get("quantization") or cfgj.get("quantization_config") or {}
            group = int(q.get("group_size", 64))
        except Exception:
            pass
        self.moe_inter, self.group = moe_inter, group

        self.pool = ExpertPool(experts_dir, cfg.hidden_size, moe_inter,
                               cold_capacity)
        self.moe_layer_ids = []
        for l, blk in enumerate(model.layers):
            sub = moe_submodule(blk)
            if sub is None:
                continue  # dense MLP layer (Qwen3 mixes dense + MoE) — leave stock
            sub.switch_mlp = PinnedSwitchMLP(
                self.pool, l, cfg.hidden_size, moe_inter,
                self.num_experts, group=group)
            self.moe_layer_ids.append(l)

    def __call__(self, inputs, cache):
        return self.model(inputs, cache)

    # ---- pinning API -------------------------------------------------------

    def pin_experts(self, per_layer_ids: dict[int, list[int]]):
        """Pin a subset of experts (load via pool), then build stacks."""
        for l, es in per_layer_ids.items():
            for e in es:
                self.pool.pin(l, e)
        # Build resident stacks from the loaded Experts (MoE layers only —
        # mixed dense/MoE archs such as DeepSeek/GLM have dense layers whose
        # mlp has no switch_mlp, so iterating range(n_layers) would crash).
        for l in self.moe_layer_ids:
            mlp = moe_submodule(self.model.layers[l]).switch_mlp
            mlp.build_resident()

    def resident_total_mb(self) -> float:
        """§4 RAM cost: stacked hot experts + anything still in the pool
        (pinned leftovers + cold LRU). pool.resident_mb() alone reads ~0 once
        build_resident() has moved the pinned experts into stacks."""
        stacks = sum(moe_submodule(self.model.layers[l]).switch_mlp.stack_bytes
                     for l in self.moe_layer_ids)
        return stacks / 1e6 + self.pool.resident_mb()

    def pin_all(self, n_experts: int):
        """Pin ALL experts using direct slab → stack (low peak memory).
        Avoids materializing 1024 Expert nn.Module objects.
        Iterate MoE layers only — dense layers (mixed archs like DeepSeek/GLM)
        have no switch_mlp and no expert slabs."""
        for l in self.moe_layer_ids:
            mlp = moe_submodule(self.model.layers[l]).switch_mlp
            # Load slabs as raw dicts, keyed by expert id
            slab_data: dict[int, dict] = {}
            for e in range(n_experts):
                path = self.pool.slab_path(l, e)
                slab = mx.load(str(path))
                self.pool.stats["loads"] += 1
                self.pool.stats["loaded_bytes"] += path.stat().st_size
                slab_data[e] = slab
            mlp.build_resident_from_slabs(slab_data)
            del slab_data


# Model-agnostic alias: the class now generalizes OLMoE *and* Qwen3-MoE
# (both share the SwitchGLU contract). PinnedOlmoe kept for Stage-0 parity.
PinnedMoE = PinnedOlmoe


# ═══════════════════════════════════════════════════════════════════════════
# Greedy generation with decode-only timing (Stage-0 gate metric)
# ═══════════════════════════════════════════════════════════════════════════

def generate_greedy(arm: PinnedOlmoe, prompt_ids: list[int], max_new: int = 50,
                    eos_ids: tuple[int, ...] = ()):
    cache = [KVCache() for _ in range(arm.n_layers)]
    x = mx.array([prompt_ids])
    t0 = time.perf_counter()
    logits = arm(x, cache)
    tok = mx.argmax(logits[:, -1], axis=-1).item()
    t_prefill = time.perf_counter() - t0

    seq = [tok]
    mx.synchronize()
    t1 = time.perf_counter()
    for _ in range(max_new - 1):
        x = mx.array([[tok]])
        logits = arm(x, cache)
        tok = mx.argmax(logits[:, -1], axis=-1).item()
        seq.append(tok)
        if tok in eos_ids:
            break
    mx.synchronize()
    dt = time.perf_counter() - t1
    n = len(seq) - 1
    return {
        "tokens": seq,
        "n_decode": n,
        "decode_s": dt,
        "tg_tok_s": n / dt if dt > 0 else float("nan"),
        "prefill_s": t_prefill,
    }
