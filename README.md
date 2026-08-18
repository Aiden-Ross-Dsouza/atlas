# ATLAS: Measure Fitness, Don't Infer the Regime

**Routing persistent adapters for continual JEPA world models**

*NeurIPS 2026 Workshop on Continual World Models — Idea Track*

---

## Overview

ATLAS places a library of small **charts** (adapters) on a permanently frozen JEPA world model and runs three rules over one normalised quantity — the **Unexplained Motion Fraction (UMF)**:

| Rule | What it does |
|------|-------------|
| **SELECT** | pick the chart with lowest unexplained motion on a fresh chunk |
| **REFINE** | one SGD step on the selected chart, after scoring (AdaJEPA) |
| **EXPAND** | commit a new chart only after verifying on unseen data that it closes the deficit |

No rewards, no task labels, no regime boundaries — only the agent's own latent prediction error.

---

## Repository structure

```
atlas/
├── atlas/            # Core package: score, chart, library, router, expand, loop, …
├── scripts/          # CLI entry points: download, gates, E0–E5, tables, figures
├── configs/          # YAML hyperparameters and regime definitions
├── modal/            # Modal cloud GPU app (for contributors running on Modal)
├── tests/            # Unit tests (pytest, no GPU required)
├── data/             # gitignored — dataset lives here
├── ckpts/            # gitignored — checkpoints live here
├── logs/             # gitignored — per-episode JSONL logs
├── atlas_out/        # gitignored — experiment outputs (.pt, .parquet, .pdf)
├── hub/              # gitignored — torch.hub cache
├── pyproject.toml
├── environment.yml   # conda env for local dev
└── .env.example      # copy to .env and fill in ATLAS_HOME
```

---

## Setup

### Prerequisites

- Conda (Miniconda or Anaconda)
- CUDA-capable GPU (24–32 GB VRAM recommended; ViT-S/14 + depth-6 predictor)
- **No HF login required** — checkpoints are public via `torch.hub`

> **Do NOT install MuJoCo 2.1 / `d4rl` / `mujoco-py`.**
> Push-T uses `pymunk`. PointMaze is cut from the experimental plan.

### 1a. Conda (Linux / macOS — recommended for GPU servers)

```bash
git clone https://github.com/Aiden-Ross-Dsouza/atlas.git
cd atlas
conda env create -f environment.yml   # creates 'atlas' env with Python 3.10 + ffmpeg
conda activate atlas
uv pip install -e .          # installs atlas + jepa-wms (git dep) + all deps
uv pip install -e ".[dev]"   # adds pytest, ruff
```

### 1b. Plain venv (Windows — when conda is not available)

```powershell
git clone https://github.com/Aiden-Ross-Dsouza/atlas.git
cd atlas
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install uv
uv pip install -e .
uv pip install -e ".[dev]"
```

### 3. Configure paths

```bash
cp .env.example .env
# Edit .env and set ATLAS_HOME to the absolute path of this repo.
# All other paths (data/, ckpts/, logs/, atlas_out/) derive from ATLAS_HOME.
```

On Windows, use forward slashes: `ATLAS_HOME=C:/path/to/atlas`

### 4. Download checkpoints and dataset

```bash
python scripts/download_data.py
```

Downloads:
- `dino_wm_pusht` checkpoint via `torch.hub.load('facebookresearch/jepa-wms', 'dino_wm_pusht')` → `ckpts/`
- Push-T dataset via jepa-wms download utility → `data/pusht_noise/`

### 5. Day-2 reconnaissance (before writing chart code)

```bash
python scripts/dump_params.py   # inspect predictor parameter names and LN vs AdaLN
```

---

## Substrate: AdaJEPA vs JEPA-WM

> **Critical design decision (implementation plan §0):** AdaJEPA's released code builds on PLDM (a small ResNet encoder). JEPA-WM/DINO-WM use a **frozen DINOv2 ViT-S/14**. Our thesis is about frozen foundation-model backbones.
>
> **Every arm — ATLAS and all baselines — runs inside the `jepa-wms` substrate** with the same frozen DINOv2 encoder, same ViT predictor, same CEM planner. Our AdaJEPA baseline corresponds to their `predlast+encfrozen` variant.

---

## Experiments

Run in this order. **E0 and E1 gate the project (days 3–5).**

| Experiment | Command | GPU-h | Gate |
|-----------|---------|-------|------|
| **E0** Adapter capacity | `python scripts/run_e0.py` | ~3 | adapter kind chosen |
| **E1** Fitness routing ← **THE GATE** | `python scripts/run_e1.py --charts atlas_out/e0/<kind>` | ~4 | T1 → GO / PIVOT |
| **E5** Cross-policy (supp.) | `python scripts/run_e5.py --charts atlas_out/e0/<kind>` | ~3 | heatmap |
| **E4+E3** Continual stream + ladder | `python scripts/run_e4.py` | ~21 | F1, F2b, T2 |
| **E2** Appearance vs dynamics | `python scripts/run_e2.py` | ~6 | F2a |

### Correctness gates (run before trusting results)

```bash
python scripts/smoke_gates.py --all
```

| Gate | Tests |
|------|-------|
| G1 | Library {c₀} → trajectory bit-identical to frozen |
| G2 | Over-refine on W; score on W' → X doesn't auto-win |
| G3a | New regime → probe commits |
| G3b | Noise → probe rejects |
| G4 | 20 rollouts per regime differ statistically |
| G5 | Two arms, same seed → identical start states |
| G6 | Static chunk → UMF returns None |

### Regenerate tables and figures from logs

```bash
python scripts/make_tables.py --all    # T1, T2
python scripts/make_figures.py --all   # F1, F2 as .pdf
```

---

## Hyperparameters (fixed, not swept)

| Symbol | Meaning | Value |
|--------|---------|-------|
| `τ` | UMF adequacy threshold | 0.5 |
| `q` | strikes to arm probe | 3 |
| `m` | hysteresis margin | 0.05 |
| `n_probe` | probe fitting steps | 20 |
| `K_max` | library cap | 10 |
| lr | refinement | 5e-4 |
| CEM | 200 × 10 opt steps, horizon 25 | — |

---

## Running on Modal (cloud GPU)

See [modal/README.md](modal/README.md) for full instructions.

```bash
pip install modal && modal token new
modal volume create atlas-data
modal run modal/modal_app.py::download_data
modal run modal/modal_app.py::run_e0
modal run modal/modal_app.py::run_e4
```

---

## Integration point (jepa-wms planning loop)

The experiment scripts raise `NotImplementedError` at the point where the
**jepa-wms CEM planning eval loop** must be wired in.
All ATLAS logic (scoring, routing, expansion, refinement) is complete in
`atlas/`. The integration task is to call it from inside the `evals/simu_env_planning/`
rollout loop:

```python
# Inside the replan loop:
step_info = atlas.loop.atlas_step(library, expander, predictor,
                                   encoder_output, actions, current_idx, cfg)
current_idx = step_info.selected_idx
library[current_idx].apply_(predictor)
plan = cem_planner(predictor, ...)
library[current_idx].restore_(predictor, library.c0)

# After executing 5 actions and collecting new chunk:
atlas.loop.atlas_refine(library[current_idx], predictor,
                         new_encoder_output, new_actions, cfg.lr)
```

---

## Tests

```bash
pytest tests/ -v          # no GPU required
pytest tests/ --cov=atlas # with coverage
```

---

## Scope-cut ladder

If time runs short, cut in this order:
1. Sensitivity run
2. E5 cross-policy
3. E2 cell D
4. Episodes/segment: 20 → 15
5. Drop the `atlas_fixed` arm

**E0 + E1 + E2 + E3/E4 with T1/T2/F1/F2 is the complete paper.**

---

## Citation

```bibtex
@misc{atlas2026,
  title  = {ATLAS: Measure Fitness, Don't Infer the Regime},
  author = {[Authors]},
  year   = {2026},
  note   = {NeurIPS 2026 Workshop on Continual World Models},
}
```

---

## References

- Terver et al., *JEPA-WM*, arXiv:2512.24497, 2025
- Wang, Bounou, LeCun, Ren, *AdaJEPA*, arXiv:2606.32026, 2026
- Zhou et al., *DINO-WM*, arXiv:2411.04983, 2024
