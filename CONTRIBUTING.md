# Contributing to DeeWaanAI Serve

Thanks for your interest! DeeWaanAI Serve is AGPL-3.0 open source. A few things to know
before you send a change.

## Before you start

- Target platform is **Apple Silicon + MLX**. PRs should keep the engine runnable on
  macOS/Python 3.12 with the pinned dependency versions in `pyproject.toml`.
- The serving path assumes **NeoX (non-traditional) RoPE, uniform full-attention** MoE
  blocks. See `COMPATIBILITY.md`. New architecture support (traditional rope, partial
  rotary, sliding-window/conv state) is welcome but needs the fidelity gates to pass.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Verifying your change

The project's rule is that no number ships that isn't traceable to frozen data.

```bash
# Fidelity gates (must pass before any performance claim):
python -m deewaanai_serve.repro.switch_equiv_gate      # synthetic bit-exact
python -m deewaanai_serve.repro.prefetch_equiv_test    # output-neutral SHA
# Re-score ceilings from frozen logs:
python -m deewaanai_serve.repro.p05_rescore <results-dir>
```

## Contributor License Agreement (required)

Because DeeWaanAI may offer this software under both AGPL and a separate commercial
license, **every contribution must be covered by our [CLA](CLA.md)**. On your first pull
request, sign the CLA (we use a bot; follow the prompt). This keeps the code's licensing
options open as the community grows — it does not take away your rights to your own
contributions.

We also ask for a `Signed-off-by` trailer (DCO) on commits.

## Pull requests

- One logical change per PR; link an issue where possible.
- Include the frozen-data provenance for any new benchmark.
- Report negative results honestly — they are valued here.

## Code of conduct

Be kind, be rigorous, give credit.
