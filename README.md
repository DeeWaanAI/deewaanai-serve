# DeeWaanAI Serve

**Open research on serving Mixture-of-Experts models on Apple Silicon (MLX).**

> ⚠️ **Status: research prototype / proof of concept — not a production tool.**
> The concurrency finding (pinning hot experts serves more simultaneous users) is measured
> and reproducible. The "run a model bigger than RAM" result is a **feasibility demo only** —
> on a single laptop SSD it streams at ~2 tok/s, which is not usable for real workloads.
> We publish the method, the numbers, and the honest limits so others can build on them.

> This is a **serving system**, not a model. It keeps a traffic-ranked subset of a
> Mixture-of-Experts model's experts resident in unified memory, dispatches via a
> gather kernel, and streams cold experts from disk with continuous batching — behind a
> standard OpenAI-compatible HTTP API.

Built by Youssef Hariri — maintained under DeeWaanAI, a brand of Mug and Bewong Ltd (UK Company No. 14916888). Accompanying paper:
*"When Pinning Pays: A Measured Crossover Curve and a Feasibility Result for
Mixture-of-Experts Serving on Consumer Unified Memory"* (see `docs/` and `CITATION.cff`).

---

## What it does — measured results

All numbers are from the paper and are reproducible from the frozen data in this repo
(see *Reproducing the paper*). Metric: **p05 concurrency ceiling** — the most simultaneous
users served at a ≥ 10 tok/s per-user floor, same-machine, run to failure.

**Serve more users (model fits comfortably in RAM):**

| Model | RAM / ratio | Ours | llama.cpp | Result |
|---|---|---|---|---|
| OLMoE-1B-7B | 16 GB / 3.8× | **16** | 8 | ~2× more concurrent users |
| Qwen1.5-MoE-A2.7B | 16 GB / 1.8× | 4 | 8 | advantage gone |
| Qwen3-30B-A3B | 24 GB / 0.9× | 4 | 4 | tie |

The win is real but **ratio-conditional**: it appears above roughly a 2–3× RAM-to-model
ratio and vanishes as the model approaches RAM.

**Run a model that doesn't fit (feasibility, not speed):**

- On a 16 GB Mac, llama.cpp's GPU path **cannot load** the 18 GB Qwen3-30B-A3B (Metal
  out-of-memory); this engine **streams it** at ≈ 2.1 tok/s single-user.
- Within-step **parallel prefetch** is a real **2.1×** speedup over serial streaming.
- Streaming preserves quality: perplexity stays within **0.41%** of fully-resident.
- Honest negatives: *predictive* prefetch and trading expert-residency for KV **do not**
  help on a single consumer SSD — disk bandwidth is the wall.

---

## Requirements

- **Apple Silicon** Mac (M-series) running macOS — the engine uses MLX/Metal and will not
  run on Linux/Windows.
- Python **3.12**.
- A MoE model that is **MLX 4-bit** and uses **NeoX (non-traditional) RoPE with uniform,
  full-attention layers**. Verified compatible: OLMoE-1B-7B, Qwen1.5-MoE, Qwen3-30B-A3B,
  Mixtral-8x7B. Incompatible (fail-closed): Llama-4, DeepSeek, gpt-oss, Cohere, DBRX,
  mamba/conv hybrids. See `COMPATIBILITY.md`.

## Install

```bash
git clone https://github.com/DeeWaanAI/deewaanai-serve.git
cd deewaanai-serve
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quickstart

Model weights are **not** included (see *Weights and licensing*). Pull an MLX 4-bit MoE
yourself, build slabs, then serve.

```bash
# 1. Get a compatible model (example: OLMoE, Apache-2.0) — downloads to ./models
huggingface-cli download mlx-community/OLMoE-1B-7B-0924-Instruct-4bit \
  --local-dir models/olmoe-mlx-4bit

# 2. Convert to pinned-engine expert slabs (verbatim; no re-quantization)
dwa convert --mlx-dir models/olmoe-mlx-4bit --out models/olmoe-slabs

# 3. Serve (fully resident for a model that fits)
dwa serve --ref-dir models/olmoe-mlx-4bit --experts-dir models/olmoe-slabs --port 8080

# 4. Use it (OpenAI-compatible)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"pinned-olmoe","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":40}'
```

To run a model **larger than RAM**, add `--pin-fraction 0.1` (streams the rest from disk;
expect low single-digit tok/s on one SSD). The server exposes `/v1/models` and
`/v1/chat/completions` (with SSE streaming).

## Reproducing the paper

- `docs/P1_v0.1.pdf` / `.tex` — the paper; `docs/PAPER_PLAN.md` — the outline.
- `results/` — the **frozen derived data** behind every number: per-experiment p05
  ceilings (`results/ceilings/`), router-telemetry summaries (`results/telemetry/`), both
  perplexity-gate runs (`results/ppl/`), the self-verifying headline spine
  (`results/derived/paper_headline_numbers.json`), and the figure PNGs
  (`results/figures/`). No model weights and no bulky per-request logs are included.
- `src/deewaanai_serve/repro/` — the analysis and fidelity-gate scripts that produced the
  above (`p05_rescore.py`, `union_analysis.py`, `trace_experts.py`,
  `extract_headline_numbers.py`, `verify_two_way.py`, `switch_equiv_gate.py`,
  `lfm2_e2e_equiv.py`, `prefetch_equiv_test.py`, `make_crossover_figs.py`). They were
  authored against the research tree, so point them at this `results/` directory to
  re-derive the spine and figures. Re-running the full concurrency ramps requires a
  compatible model (see Quickstart).

## Weights and licensing

This repository contains **only DeeWaanAI-authored code and derived result data**. It
does **not** redistribute any model weights. The `dwa convert` step operates on a
checkpoint **you** download, and you are responsible for the upstream model's license
(for example, Qwen models carry their own terms that restrict redistribution/commercial
use; OLMoE and Mixtral are Apache-2.0).

## License

**AGPL-3.0-only** — see `LICENSE`. The AGPL requires that anyone who runs a modified
version of this software as a network service makes their source available. Copyright (c) 2026 Mug and Bewong Ltd.

## Trademark

**DeeWaanAI™** and the DeeWaanAI logo are trademarks of Mug and Bewong Ltd (Company No. 14916888). This license grants no
rights to use the DeeWaanAI name or logo; forks must use a different name.

## Citation

Cite the paper + data: **DOI [10.5281/zenodo.23182261](https://doi.org/10.5281/zenodo.23182261)** — see `CITATION.cff` and https://zenodo.org/records/23182261.
