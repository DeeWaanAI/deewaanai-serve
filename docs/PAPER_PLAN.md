# Consolidated Paper — Section / Subsection Plan (v1)

**Program:** DeeWaanAI — pinned-expert MoE serving on Apple Silicon
**Target:** arXiv preprint (unbounded; all results incl. negatives in-body, full methodology chain)
**Draft format:** LaTeX (`.tex`), built from this Markdown plan. Version scheme: `paper/P1_v0.X.tex`, bump on data-corrected rounds.
**Governing rule (research-paper-rigor):** every number printed must trace to a frozen file → committed script → raw data. Numbers that exist only in prose are defects; reconcile before first compile (see §0 and §10).

---

## Working title (candidates)

1. *When Pinning Pays: A Measured Crossover Curve and a Feasibility Result for Mixture-of-Experts Serving on Consumer Unified Memory*
2. *Pinned-Expert Orchestration for MoE Inference on Apple Silicon: Concurrency Crossovers, >RAM Streaming, and What Prefetch Doesn't Fix*
3. *Run Bigger Than RAM, Or Run More At Once: Two Regimes of MoE Serving on a Laptop*

**One-sentence thesis:** On Apple-Silicon unified memory, pinning hot MoE experts doubles the concurrent users llama.cpp can serve *only while the model is a small fraction of RAM*, and — in the opposite regime — a streaming hot-set engine *runs models that llama.cpp's GPU path cannot load at all*, though bounded to a feasibility (not speed) result by the single-SSD bandwidth wall.

---

## Narrative arc (three acts, unified)

- **Act I — fits-RAM crossover (§5).** Controlled same-machine study: pinned orchestration serves up to 2× the concurrency of llama.cpp above a ~2–3× RAM/model ratio; the advantage collapses to a tie and mildly reverses as the model approaches RAM. Mechanism is measured, not assumed.
- **Act II — beyond-RAM pivot (§6).** The business regime: models larger than RAM. Streaming is a cliff; within-step *parallel* I/O is the lever (2.1×), *predictive* prefetch is not (clean negative), residency-for-KV eviction doesn't help (negative), and the headline is *feasibility* — llama.cpp's Metal path OOMs where our engine runs.
- **Act III — synthesis (§7–8).** Position vs the 2025-26 working-set / prefetch literature; state plainly which pillars we have (working set + streaming) and which we lack (prediction, scheduling, cache-aware routing); bound every claim to its hardware and statistic.

---

## §0 Pre-draft reconciliation checklist (DO FIRST — integrity gate)

These are the claim-vs-frozen-data discrepancies I flagged while auditing. Each must be resolved to ONE authoritative statistic + ONE frozen source before it appears in prose.

1. **>RAM single-user speed.** D-052 prose says "2.32 tok/s"; frozen `results/ram_vs_llama/p05/p05_ceilings.json` says `ours_pf` L1 = **2.123** (p05). Pick the statistic (recommend p05, consistent with the rest) and cite the file; do not print the unreconciled 2.32.
2. **D-050 A/B/C prefetch numbers.** Report quotes medians (A 1.14 / B 2.38 / C 0.79); frozen `results/ram30b2/p05/` gives p05 L1 (A 0.909 / B 1.934 / C 0.372). The **2.1× ratio holds under both** — state which statistic and cite.
3. **Resident-growth curve.** `FINAL_FINDINGS.md` says "36→57 of 64 as concurrency 1→64"; `EXPERT_REVIEW_REPORT.md` says "36→56 as 1→32 (90% cov: 31→51)". Re-read `results/telemetry/olmoe_trace400_union.json` → `code/union_analysis.py` and fix one wording.
4. **Streaming-cliff table (D-040).** Report says resident 40/60/80/100% → 5.6/6.1/10.8/111 tok/s; frozen `results/pinfraction/p05/` shows `pf0.4` no-prefetch L1 = 4.546 and `pf1.0` L1 = 117.7. Reconcile to frozen values (the shape — cliff — is robust; the point estimates must match the file).
5. **Qwen3-30B crossover point (4 vs 4).** This is the Gate-1 rental (24 GB M4) figure. Confirm the frozen source (likely `results/rental_instance/p1_2` or `results/p05_rescore`). This is the only curve point NOT on the 16 GB laptop — disclose the cross-machine caveat explicitly.
6. **Rental (Gate-1) numbers.** The rental OLMoE p05 ceiling is legitimately **12 vs 8** (24 GB M4) — a real Gate-1 result, distinct from the laptop crossover's 16 vs 8. What IS stale: the min-based "worst-user 25.4 vs 10.7" framing and any "8 vs 4" from the Sept-26 docs, superseded by p05 (D-033). Report rental on p05 too.
7. **Pending fidelity gate for >RAM.** Same-quant cross-engine greedy match (ours-streamed-Q4 vs llama mmap-Q4) is **not done** (llama OOMs on the box). The paper must state this as an open evidence gap, not imply it passed.
8. **PPL gate = two runs, report the right one.** `PPL_OLMoE.json` (pre-fix) FAILS because the cold-stream path used a different math kernel; `PPL_OLMoE_POSTFIX2.json` (post-fix, kernel parity restored, D-029→D-032) PASSES, max ΔPPL 0.41% at P=0.1 → 0.09% at P=0.8 → 0.00% at P=1.0. Cite the post-fix as the result and the pre-fix as the caught confound — never cite the pre-fix FAIL as the finding.
9. **Gate-1 folded in as its own section (Option A).** A "Gate-1: exploratory campaign" section (v1 ceilings, v2/v3 nulls, flat-routing, PPL two-run, bandwidth/attribution, kernel negative) placed after Method, re-scored on the same p05 statistic via `results/p05_rescore`.
10. **Backups = GitHub only.** Name GitHub (plus local) as the durable archive; the object-storage mirror is being cleared and is not the archive of record.

---

## §1 Abstract (≈200–250 words)

Main points (one sentence each):
1. Problem: MoE serving on consumer unified memory (≤24 GB), where the incumbent (llama.cpp) and research systems behave differently across two regimes.
2. System: a pinned hot-expert orchestration engine on stock `mlx_lm` + MLX gather + continuous batching; evaluated by **p05 concurrency ceiling** under a 10 tok/s/user SLO floor, same-machine, uncensored.
3. Act I result: crossover — 2× concurrency at high RAM/model ratio (OLMoE 3.8× → 16 vs 8 users), gone by ~1.8× (Qwen1.5-MoE loss) and a tie at 0.9× (Qwen3-30B). Mechanism from router telemetry (Jaccard 0.59, sub-linear resident growth, 71% single-request skew).
4. Act II result: >RAM feasibility — llama.cpp's Metal path OOMs on the 18 GB model on a 16 GB Mac; our engine streams it (~2.1 tok/s); within-step parallel prefetch gives 2.1× over serial, predictive prefetch and residency-for-KV eviction are negatives.
5. Positioning + rigor: first measured crossover curve on M-series under a per-user SLO; every number claim→frozen-file traceable.

**Framing (business-reader, skill §F):** "the slowest person in a 16-way chat is still served ≥10 tok/s where llama.cpp's tail breaks by 12"; "a model that literally does not fit can still run — a capability, not a speedup."

---

## §2 Introduction

- 2.1 Motivation: MoE economics on consumer hardware; DeeWaanAI appliance context (why small-active MoE, why unified memory, why per-user tail matters).
- 2.2 The gap: no published **RAM/model-size crossover curve** and no **>RAM concurrency/feasibility** result on M-series under a per-user SLO (ground in §3 literature; state the owned gap).
- 2.3 Research questions:
  - **RQ1:** When (as a function of RAM/model ratio) does hot-expert pinning beat llama.cpp on concurrent users?
  - **RQ2:** What mechanism produces or destroys that advantage? (measured, not assumed)
  - **RQ3:** Can an MoE larger than RAM be *served at all* on the device, and does pinning/streaming add value over the incumbent's offload?
  - **RQ4:** Which serving levers (parallel I/O vs prediction vs KV-residency trade) move the needle on single-SSD consumer hardware?
- 2.4 Contributions (bulleted, each with the section that proves it):
  1. Measured crossover curve (3 current models) + bounded operating guidance.
  2. Router-telemetry mechanism explaining the ~2× bound and its collapse.
  3. >RAM feasibility result + 2.1× parallel-prefetch win.
  4. Three disciplined negatives: predictive prefetch, residency-for-KV eviction, custom Python/MLX kernel (appendix).
  5. A reproducible claim→evidence methodology (1,728 DP registry + 52-decision log) as a template for small-team systems research.
- 2.5 Scope and non-claims (pre-empt reviewer over-reading): single-user speed is *not* claimed; custom-kernel speedup is *not* claimed; >RAM is *feasibility*, not concurrency; claims are same-machine ratios, not absolute tok/s.

---

## §3 Background & Related Work

- 3.1 MoE inference mechanics: sparse routing, top-k experts, why small-active ≠ small-footprint; 4-bit affine layout, `mx.gather_qmm`.
- 3.2 Working-set / memory-bounded serving: WiSP (2606.21868), MawForge (2607.09686), fine-grained offload (2502.05370), DuoServe-MoE (2509.07379).
- 3.3 Predictive prefetch & scheduling (levers we measure against): SpecPrefetch (2607.24787), ExpertFlow (DAC'26, industry — no verified arXiv ID, flag as such), Edge0 (project, not paper), Klotski (2502.06888).
- 3.4 Cache-aware routing: MoCE (2412.00099); pre-registered negative "Cacheable by Design?" (2608.18261) → our thrash-cliff parallel.
- 3.5 Memory/KV contention theory: **the *qs* inequality** (2603.08960) — the formal frame for our OLMoE-win / Qwen-loss swing.
- 3.6 Tail latency & fairness metrics: FairBatching (2510.14392), Niyama (2503.22762), SLO-aware sliding window (2606.05933) — justify the p05 ceiling.
- 3.7 Bandwidth/roofline: Memory-Bound-not-Bandwidth-Limited (2605.30571), prefill/decode (2512.22066), Ecco (2505.06901) — why single-user stays slow.
- 3.8 Apple-Silicon-specific: NPU MoE (2604.18788), multi-node EP (2506.23635).
- 3.9 Position table: "which of the six pillars does each system implement; which do we."
  **Data/verification:** cite each arXiv ID to `results/lit_review_phase2/` (24 verified finalists); every citation re-checked against primary registry before draft.

---

## §4 System, Metric & Methodology

- 4.1 Engine (SUT): stock `mlx_lm` backbone; only `switch_mlp`→`PinnedSwitchMLP` swapped; resident hot stack + gather-QMM dispatch; cold experts streamed from verbatim per-expert slabs (`extract_mlx_experts.py`, no re-quantization).
- 4.2 Continuous batching: "world-coordinate" rope-shift trick; its assumptions → the NeoX-uniform-rope / full-attention compatibility boundary (link `COMPATIBILITY.md`; frame honestly as a limitation, §8.3).
- 4.3 Baseline & controls: llama.cpp (`llama-server`) incumbent; ablation controls B2 (--no-mmap), B3 (MLX stock, isolates framework), B4 (naive-LRU, isolates hot-set intelligence). Note which were run vs only designed (Mixtral/B2-B4 partial — disclose).
- 4.4 **Metric definition — p05 concurrency ceiling** (the paper's measurement spine): highest N such that the 5th-percentile per-user decode ≥ 10 tok/s; why p05 over min/median; uncensored ceilings (run to failure); percentile method (linear, numpy-equivalent).
- 4.5 Fidelity gate (before any ceiling is trusted): synthetic bit-exact equivalence (`switch_equiv_gate.py`) + end-to-end greedy match (`lfm2_e2e_equiv.py`, 3/3) + PPL streaming gate (ΔPPL ≤ 0.5%).
- 4.6 Experimental protocol / frozen corpus: WildChat-3k subset (sha dc3f8f7e…); identical reps, context budget, quant across arms.
- 4.7 Reproducibility & claim→evidence chain: `results/FROZEN_MANIFEST.json` (sha256), `paper/claim_map/datapoints.json` (1,728 DPs), `paper-claims.json`, `DECISIONS.md` (D-001…D-052). Two-way verifier must be GREEN before any number is quoted.
- 4.8 **Stage-0 validation numbers** (already linked, 14 claims): baseline pp512 1261.4±39.5, tg128 132.1±0.2, gate thresholds, 5/5 parity. Establish that the harness itself is sound.
  **Data:** `results/derived/summary.json` via `code/paper/analyze_derived.py`; `code/p05_rescore.py`.

---

## §5 Experiment I — The Concurrency Crossover (fits-RAM)

- 5.1 Design: same-machine per model, fixed per-user context budget for both arms, llama given enough slots it is not queue-capped, run to failure.
- 5.2 **Headline table** (p05 ceilings, ratio = device-RAM ÷ model-4bit-GB):

  | Model | Provider | 4-bit | RAM/Ratio | Pinned | llama | Advantage | Frozen source |
  |---|---|---|---|---|---|---|---|
  | OLMoE-1B-7B | AI2 | 4.2 GB | 16 GB / 3.8× | **16** | 8 | **2.0× win** | `results/crossover/p05/p05_ceilings.json` |
  | Qwen1.5-MoE-A2.7B | Alibaba | 8.8 GB | 16 GB / 1.8× | **4** | 8 | **0.5× (loss)** | `results/crossover_qwen15/p05/p05_ceilings.json` |
  | Qwen3-30B-A3B | Alibaba | 18 GB | 24 GB / 0.9× | **4** | 4 | **1.0× (tie)** | `results/p05_rescore/…` (rental — verify, cross-machine caveat) |

- 5.3 Per-level shape (the signature): llama ~2× faster at 1–2 users (p05 64.7 vs 34.0 tok/s OLMoE) but tail collapses below floor by 12 users; pinned holds to 16. We lose single-stream speed, win concurrency — only at high ratio. (Fig A p05 curves.)
- 5.4 **Mechanism (measured):** OLMoE 64 experts / top-8, 400 prompts — median pairwise Jaccard **0.59**; sub-linear resident growth (≈36→57/64 with concurrency; reconcile §0.3); single request touches **45/64 = 71%** of pool. Sub-linear growth ⇒ concurrency amortization (the win); mild skew ⇒ bound near 2×; big models ⇒ coverage set exceeds RAM (win dies). (Fig B resident-growth curve.)
- 5.5 Fidelity confirmation for these arms (§4.5 gates pass on OLMoE + Qwen1.5-MoE).
- 5.6 **Confounds, stated honestly:** Qwen1.5-MoE "loss" mixes ratio effect + our lower/noisier per-user throughput at 60×24 expert dims; *direction* robust, *exact threshold* not. Qwen3-30B point is cross-machine.
- 5.7 **Strategic implication (quantified):** "Choose a MoE whose 4-bit footprint is ≲1/4 of RAM. At 3.8× you double served users; below ~2× expect no concurrency benefit from pinning."
  **Data:** ceilings via `code/p05_rescore.py`; ramps `results/crossover*/{pinned,llama}_*.requests.jsonl` via `code/harness/run_crossover.sh`; telemetry `results/telemetry/olmoe_trace400*.json` via `code/union_analysis.py`.

---

## §6 Experiment II — Serving MoE Bigger Than RAM (feasibility)

- 6.1 Hypothesis flip & subject selection: filter on **active params** (≤4B/token), not footprint → Qwen3-30B-A3B (3B active) is the reference; Mixtral (13B active) is the discarded subject (kept as a negative data point).
- 6.2 **Finding — streaming is a cliff, not a slope (D-040):** resident % 40/60/80/100% → per-user ≈4.5 / … / 117.7 tok/s, ceiling 0/0/0/12 (frozen `results/pinfraction/p05/`). Cause: synchronous serial cold-load latency. (Fig D.)
- 6.3 **Finding — within-step parallel prefetch turns cliff into slope (D-041):** output-neutral (SHA `5da5f215…`); pf0.4 ceiling 0→2, L1 4.546→12.77 tok/s; the shipped lever. (Fig E; `results/pinfraction/p05/` `*_pf` arms.)
- 6.4 **Finding — real >RAM A/B/C (D-050):** Qwen3-30B-A3B (18 GB > 16 GB), pin 0.1 — per-user tok/s (statistic per §0.2): A serial < B parallel **(≈2.1× win)** > C predictive (hurts). `results/ram30b2/p05/`.
- 6.5 **Negative — prediction doesn't pay on single SSD (D-042/045/046/050):** cross-step predictive prefetch shows no gain on cacheable models and *reduces* throughput on the >RAM model; parallelism already captures the overlap; ~62% mispredicted reads are wasted traffic on a bandwidth-bound SSD.
- 6.6 **Negative — residency-for-KV eviction refuted (D-051):** evicting experts to free KV *lowered* the p05 ceiling 12→4; the binding constraint is per-user **speed**, not KV capacity. Keep experts fully resident on this hardware. `results/reskv/p05/`.
- 6.7 **The product test (D-052):** ours streams the 18 GB model (~2.1 tok/s, §0.1) on a 16 GB Mac; llama.cpp `-ngl 99` **OOMs**, `-ngl 20` **0/10** succeed ("Metal OutOfMemory"). Claim = *feasibility* ("can run vs cannot run at all"), NOT speed (both far under the 10 tok/s floor at pin 0.1). `results/ram_vs_llama/`.
- 6.8 Predictability analysis (why prediction is weak here): aggregate top-10 coverage 28%, Zipf α 0.44, entropy 5.86/6.0; per-token recall — static 0.244, temporal 0.381, context 0.353, topic +0.015. Signal is temporal/contextual, not topical — motivates a learned predictor on higher-bandwidth hardware (future work).
- 6.9 Fidelity caveats: D-047 cross-residency near-tie rounding divergence (resident ≠ bit-exact to cold-stream); >RAM same-quant cross-engine gate **pending** (§0.7). State as open, not passed.
- 6.10 **Strategic implication (quantified):** "Big-model-on-small-hardware is today a *capability* story bounded by SSD bandwidth (~2.4 tok/s ceiling on a laptop). The next doubling is hardware (parallel NVMe), not cleverer prediction."
  **Data:** sweeps `code/harness/run_pinfraction_sweep.sh`; prefetch impl `code/stage0/pinned_arm.py`; correctness `code/prefetch_equiv_test.py`; predictability `code/{predictability_analysis,sequence_predictability}.py`.

---

## §7 Discussion

- 7.1 The two regimes as one story: crossover (fits-RAM concurrency) and feasibility (>RAM capability) are the same mechanism (bounded working set + amortization) read at opposite ends of the RAM/model ratio; the *qs* inequality predicts both the ~2× bound and its collapse.
- 7.2 Why our numbers are conservative & honest: same-machine ratios only; low reps acknowledged; uncensored ceilings; negatives reported.
- 7.3 Limitations: architecture support (NeoX-uniform-rope only — excludes Llama-4/DeepSeek/gpt-oss/Cohere/DBRX/Granite); single-SSD bandwidth wall; cross-machine Qwen3-30B point; Qwen1.5-MoE confound; pending >RAM fidelity gate.
- 7.4 What the field says we're missing (six-pillar synthesis): we implement pillar 1 (working set + streaming); we lack prefetch (properly budgeted on fast media), token scheduling, cache-aware routing — our engine is roughly the baseline those 2025-26 systems improve on.

---

## §8 Threats to Validity (dedicated — arXiv gives room)

- 8.1 Construct: is p05 ceiling the right SLO proxy? (defend vs median/min, cite fairness lit).
- 8.2 Internal: low reps (n per level), statistic drift between prose and frozen files (§0 items), Metal-kernel→gather_qmm fallback (measures orchestration, not bespoke kernel).
- 8.3 External: model family (Qwen/OLMoE only cleanly compatible); hardware (M1 Pro 16 GB / one M4 24 GB); OS paging not directly controlled.
- 8.4 Conclusion-formation: the crossover *threshold* value vs *direction* (only the direction is robust).

---

## §9 Related-positioning conclusion + future work

- 9.1 Conclusion (answer RQ1–RQ4 directly, one line each).
- 9.2 Future work tied to the decisive experiments: multi-NVMe >RAM test (does prediction help once bandwidth headroom exists?); NEA learned next-expert predictor; traditional-rope/partial-rotary generalization; productionizing the appliance operating point.

---

## §10 Figures & Tables inventory

- T1 = §5.2 crossover table.
- T2 = §6.2/6.3 cliff→slope table.
- T3 = §6.4 A/B/C prefetch table.
- T4 = §6.6 reskv eviction ceilings.
- T5 = COMPATIBILITY matrix (compatible vs fail vs reason).
- T6 = related-work six-pillar position matrix.
- Fig A = OLMoE p05 curves (pinned vs llama) — `results/crossover/figures/figA_p05_curves.svg`.
- Fig B = resident-growth sub-linear curve — `.../figB_resident_growth.svg`.
- Fig C = advantage vs RAM/model ratio (3 points) — `results/crossover_qwen15/figC_advantage_vs_ratio.svg`.
- Fig D = residency cliff — `results/pinfraction/figD_residency_cliff.svg`.
- Fig E = cliff→slope with prefetch — `results/pinfraction/figE_prefetch_cliff_to_slope.svg`.
- (Add) Fig F = >RAM A/B/C per-user bars; Fig G = predictability recall bars. **New figure scripts needed** (rigor: figure data-tables must be written into `results/derived/`).

---

## §A Appendices

- **A1 Reproducibility / artifact map:** per-claim `claim → frozen file → script`; `FROZEN_MANIFEST.json` (sha256); models with HF revisions; `code/paper/run_pipeline.sh` = enforcement gate.
- **A2 AI-usage disclosure & integrity log** (this is a *strength* of the paper, per skill §D/F):
  - Corrections cascade as numbered dated stories: D-012 (pp512 1262.2→1261.4, reference 140.8→127.6 — console memory replaced by frozen data); D-033 (min→p05, censored→uncensored, 12v8→16v8); D-048→D-049→D-050 (the "prefetch hurts" over-claim retracted as an implementation bug, then rebuilt to a real 2.1× win); D-008a (a "1.16 ratio" headline killed as an unequal-load artifact).
  - "No LLM made unsupervised decisions"; every protocol/decision human-authored-reviewed-approved.
  - Pointer to the claim map as the enforcement artifact.
- **A3 Negative-result appendix — the custom-kernel detour:** `mx.gather_qmm` at 9.5 GB/s (≈8% of ~120 GB/s peak); a vectorized Python/MLX `efficient_gather_qmm` was **33× slower** (0.03×), max diff 1.4e-6; diagnosis = kernel fusion (llama.cpp dequantizes in-register in one pass). Conclusion that pivoted the program from custom-kernel speed to orchestration. **Frame as: measured why the obvious optimization fails → redirected effort; a decision-quality case study.**
- **A4 Gate-1 rental context:** 24 GB EC2 Mac M4, $1.23/hr×24 h, v1/v2/v3 + §6 traces; the cross-machine Qwen3-30B point; 3-way data backup (laptop / GitHub / S3).

---

## Suggested writing order (fastest to a verifiable draft)

1. §4 Methodology + §0 reconciliation (lock the statistic and sources first — everything else cites it).
2. §5 crossover results + mechanism (cleanest, self-contained contribution).
3. §6 >RAM results (the negatives fall out naturally).
4. §3 related work (position each claim against the six pillars).
5. §1–2 abstract/intro (write last so claims match proven results).
6. §7–9 discussion/limits/conclusion.
7. Appendices A1–A4.
8. Wire `paper-claims.json` to the draft sentences; run the two-way verifier; require GREEN before any number stands.
