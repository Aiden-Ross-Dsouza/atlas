# ATLAS — Implementation Plan v2

*Companion to proposal v7. ~11 days, single GPU (24–32 GB). Push-T only.*

**Changes from v1:** cut to five experiments (one supplementary); baselines restructured into a monotone ablation ladder; E0 reduced from 18 runs to 6; S1 stream, PointMaze, Wall, goal-progress routing, and the large hyperparameter sweeps removed. **Total compute drops from ~55 to ~37 GPU-h.**

---

## 0. The decision that shapes everything

**AdaJEPA and JEPA-WM are different substrates.** AdaJEPA's released code builds on PLDM: its encoder is *"a small ResNet whose stages, in order, are five residual blocks (rb1–rb5), an optional pooling head, and a projection head."* JEPA-WM/DINO-WM use a **frozen DINOv2 ViT-S/14** with a ViT predictor.

Our thesis is about frozen *foundation-model* backbones, and the public per-environment checkpoints live in `jepa-wms`. Therefore:

> **Every method — ATLAS and all baselines — runs inside the `jepa-wms` substrate with the same frozen DINOv2 encoder, same predictor, same CEM planner. AdaJEPA is reimplemented there following its published hyperparameters and released code.**

Because DINO-WM freezes DINOv2 entirely, our AdaJEPA baseline corresponds to their **`predlast+encfrozen`** variant — a configuration they themselves ablate. **State this in the paper's baseline description.** Comparing all arms on one substrate is more controlled than comparing across codebases.

---

## 1. Repositories

| Repo | URL | Use | Required |
|---|---|---|---|
| **jepa-wms** | `github.com/facebookresearch/jepa-wms` | **Primary substrate** — models (`app/plan_common/models`), datasets (`app/plan_common/datasets`), planning eval (`evals/simu_env_planning`), `hubconf.py` | **Yes — fork it** |
| **AdaJEPA** | `github.com/agentic-learning-ai-lab/adajepa` | Reference for the plan–act–adapt loop; **PushT visual-corruption generators** to port for E2 | **Yes** (read + port) |
| **DINO-WM** | `github.com/apple/ml-dino-wm` | Push-T env internals — to locate the physics parameters | **Yes** (read) |
| DINOv2 | auto via TorchHub | encoder | Automatic |
| PLDM / DINOv3 / RoboCasa | — | not used | **Skip** |

Fork `jepa-wms` → `atlas-wm`; add everything under a new `atlas/` package. **One upstream file is modified** (a single hook in the planning rollout). Keeps the diff reviewable.

---

## 2. Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
conda create -n atlas python=3.10 ffmpeg=7 -c conda-forge -y && conda activate atlas
git clone https://github.com/<you>/atlas-wm.git && cd atlas-wm
uv pip install -e . && uv pip install -e ".[dev]"
uv pip install statsmodels scipy pandas          # McNemar, bootstrap, tables
python -c "import torchcodec; print('ok')"

export JEPAWM_HOME=/path/to/workspace
export JEPAWM_DSET=$JEPAWM_HOME/datasets
export JEPAWM_LOGS=$JEPAWM_HOME/logs
export JEPAWM_CKPT=$JEPAWM_HOME/ckpts
export ATLAS_OUT=$JEPAWM_HOME/atlas_out
export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl   # fallback: osmesa

source ~/.bashrc && cd $JEPAWM_HOME/atlas-wm && python setup_macros.py
```

**Do not install MuJoCo 2.1 / `d4rl` / `mujoco-py`.** PointNaze is cut; Push-T uses `pymunk`.

---

## 3. Checkpoints and data

```python
model, prep = torch.hub.load('facebookresearch/jepa-wms', 'dino_wm_pusht')   # PRIMARY
model, prep = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_pusht')   # optional check
```

| Model | HF file | Direct fallback |
|---|---|---|
| **DINO-WM Push-T** | `dino_wm_pusht.pth.tar` | `dl.fbaipublicfiles.com/jepa-wms/pt_dino-wm.pth.tar` |
| JEPA-WM Push-T | `jepa_wm_pusht.pth.tar` | `dl.fbaipublicfiles.com/jepa-wms/pt_jepa-wm.pth.tar` |

Keys: `['encoder','predictor','heads','opt','scaler','epoch','batch_size','lr','amp']`. Decoder heads not needed.

```bash
python src/scripts/download_data.py --dataset pusht     # → $JEPAWM_DSET/pusht_noise/
```

DINOv2 ViT-S/14 @ 224, predictor depth 6, `D=384`. Dataset is DINO-WM's, re-hosted unmodified.

**Release:** chart libraries (`.pt`), UMF traces (`.parquet`), per-episode JSONL logs with seeds, regime configs. A few hundred MB.

---

## 4. Code layout

```
atlas-wm/atlas/
├── score.py      # UMF + informative-chunk gating                    (C3)
├── chart.py      # Chart: ln_act | lora4 | full; apply_/restore_/clone
├── library.py    # add / evict / clone / utilisation
├── router.py     # umf | e1 | sdyn | random | oracle   (+ sobs, optional)
├── expand.py     # library strike counter + fixability probe          (C2)
├── loop.py       # atlas_step(): the prequential controller           (C1)
├── adajepa.py    # AdaJEPA + Persistent-AdaJEPA in this substrate
├── regimes.py    # PhysicsRegime wrapper + visual corruptions
├── streams.py    # S2 driver, paired seeding
├── harness.py    # E0/E1/E5 offline harness (no ATLAS loop)
├── stats.py      # McNemar, paired bootstrap, normalised recovery
└── plots.py      # F1, F2
scripts/  run_e0.py … run_e5.py, make_tables.py, make_figures.py, smoke_gates.py
configs/atlas/  e0.yaml e1.yaml e2.yaml e4.yaml regimes/pusht.yaml
```

Single upstream hook, in `evals/simu_env_planning/`:

```python
chart = atlas.loop.atlas_step(state, library, chunk, cfg)
world_model.apply_chart(chart)          # in-place adapter swap
```

---

## 5. Charts

### 5.1 Day-2 reconnaissance — before writing any chart code

```python
model, prep = torch.hub.load('facebookresearch/jepa-wms', 'dino_wm_pusht')
for n, p in model.predictor.named_parameters():
    if p.ndim <= 1 or 'norm' in n: print(f"{n:60s} {tuple(p.shape)}")
print('predictor params:', sum(p.numel() for p in model.predictor.parameters()))
```

| Surface | Selector | Expected params |
|---|---|---|
| **LN affine + action encoder** | `.norm`/`.ln` with `ndim==1`, plus the action-conditioning module | ≈ **10.4 k** |
| **LoRA r=4** on attention `qkv`+`proj` | injected | ≈ **55 k** |
| **Full predictor** | all | ≈ **1.8 M** |

> ⚠️ **JEPA-WM's best Push-T config is `predAdaLN`** (`pt_..._predAdaLN_ftprop_depth6_...`), i.e. *adaptive* LayerNorm. If normalisation is produced by a conditioning MLP there are no free LN affine parameters, and the primary chart surface does not exist as specified. **This is why `dino_wm_pusht` (plain LN, `pred_dino_wm`) is the primary checkpoint.** If you later use the JEPA-WM checkpoint, target the AdaLN conditioning MLP's output projection instead — equally small, equally valid, different code path.

### 5.2 API

```python
class Chart:
    KINDS = ('ln_act', 'lora4', 'full')          # only three; E0 picks one
    def __init__(self, predictor, kind='ln_act'): ...
    def apply_(self, predictor)  / restore_(self, predictor)
    def clone(self)              # identity-preserving deep copy
    def save(path) / load(path) / n_params()
```

Two non-negotiables: **identity initialisation** (LoRA `B=0`; LN at pretrained values) so a clone starts exactly as capable as its parent and gate G1 passes; and **apply/restore rather than model copies** — swapping ~10 k floats is instant, twenty predictor copies are not.

---

## 6. Regimes

### 6.1 Push-T dynamics variants (primary, matched appearance)

Push-T is `pymunk`-based. Confirm attribute paths in `ml-dino-wm`, then:

| Regime | Parameter | Value | Appearance |
|---|---|---|---|
| **R0** default | — | shipped | — |
| **R1** light block | T-block `body.mass`, `body.moment` | ×0.2 | unchanged |
| **R2** high damping | `space.damping` | see note | unchanged |

> ⚠️ In `pymunk`, `space.damping` is a per-second **retention** factor (1.0 = no damping), so "more damping" means **decreasing** it (e.g. 0.9 → 0.3). Get the sign right on day 3.

```python
class PhysicsRegime(gym.Wrapper):
    def __init__(self, env, mass_scale=1., damping_scale=1.): ...
    def reset(self, **kw):
        obs = self.env.reset(**kw)
        self._apply_physics()      # ESSENTIAL: many envs rebuild the space on reset
        return obs
```

**S2 uses R0 and R1 only.** R2 exists for E0 (two regimes, to check adapter capacity on a second kind of shift).

### 6.2 Visual corruptions (E2 only)

Port blur / salt-and-pepper / dark / colour-change from AdaJEPA as an `ObservationWrapper`, so physics is untouched.

### 6.3 The 2×2 (E2)

| Cell | Appearance | Dynamics | Build |
|---|---|---|---|
| A control | same | same | R0 vs R0 |
| **B decisive** | same | differ | R0 vs R1 |
| **C over-expansion test** | differ | same | R0 vs R0+colour |
| D realistic | differ | differ | R0 vs R1+colour |

---

## 7. Experiments

### 7.0 Calibrate the budget on day 2 — before fixing episode counts

AdaJEPA's published planning hyperparameters, which we adopt: **CEM 200 samples × 10 opt steps, subplanner horizon 25, 5 executed actions per replan, ≤30 MPC steps.** That is 2,000 candidate rollouts per replan, ~6 replans per episode.

```bash
python scripts/run_e4.py --arm frozen --episodes 3 --profile
# prints sec/episode, peak VRAM, predictor forwards/replan
```

$$\text{GPU-h} = \frac{\text{sec/ep} \times \text{episodes} \times \text{segments} \times \text{seeds} \times \text{arms}}{3600}$$

**At 30 s/episode:** E4 = 20 ep × 6 segments × 3 seeds × 7 arms = 2,520 episodes ≈ **21 GPU-h**.

**If > 40 s/episode, cut in this order** (uniformly across all arms, and report it):
1. CEM opt steps 10 → 6
2. Episodes/segment 20 → 15
3. Max MPC steps 30 → 20 *(AdaJEPA's own ablation setting)*
4. Drop the ATLAS-fixed-library arm

### 7.1 E0 — adapter capacity (RQ0). *Days 3–4, ~3 GPU-h*

`{ln_act, lora4, full}` × `{R1, R2}` = **6 offline fine-tunes**, ~2 k steps each, Adam, predictor lr `5e-4` (AdaJEPA's value). Evaluate UMF on held-out regime trajectories and planning success in-regime.

```bash
python scripts/run_e0.py --kinds ln_act lora4 full --regimes R1 R2 --steps 2000 \
    --out $ATLAS_OUT/e0
```

**Pre-registered rule:** smallest kind reaching ≥ 90 % of `full` on **both** metrics, in **both** regimes. That kind is used everywhere downstream.

**T5 (supplementary)** — Adapter | Params | KB | ΔUMF (R1/R2) | Success (R1/R2) | % of full

### 7.2 E1 — fitness routing (RQ1). *Days 4–5, ~4 GPU-h.* **THE GATE**

Charts from E0, frozen. Per episode: 2 warmup replans under `c₀` → score all charts → select → plan the rest. **Same start states, goals and env seeds across every router.**

Routers: `umf` · `e1` (one-step) · `sdyn` · `random` (analytic `mean_c SR(c)`) · `oracle_id`. *(Goal-progress dropped; `sobs` supplementary.)*

```bash
python scripts/run_e1.py --charts $ATLAS_OUT/e0/<kind> --routers umf e1 sdyn random oracle_id \
    --episodes 60 --seeds 3 --out $ATLAS_OUT/e1
```

**T1 (body)**

| Router | SR | Routing acc. | Oracle gap | Normalised recovery [95 % CI] |
|---|---|---|---|---|
| UMF (ours) | | | | |
| one-step `e₁` | | | | |
| S-dyn | | | | |
| Random | | — | | 0 by definition |
| Oracle-ID | | 1.00 | 0 | 1 by definition |

**Pass criterion (pre-registered):** normalised recovery ≥ 0.8, reported only when `SR_oracle − SR_random ≥ 10 pp`.

> **GO / PIVOT.** ≥ 0.8 → continue. Near 0 → the paper becomes *"prediction-error routing does not transfer to frozen visual latent spaces,"* with T1's `e₁`-vs-UMF comparison as the substantive content. Six days remain; E2 and E4 still run and remain interesting.

### 7.3 E2 — appearance vs dynamics (RQ2). *Day 9, ~6 GPU-h*

Routers: `umf` vs `sdyn` (+ `sobs` if time). 40 episodes × 3 seeds × 4 cells.

```bash
python scripts/run_e2.py --cells A B C D --routers umf sdyn --episodes 40 --seeds 3 \
    --out $ATLAS_OUT/e2
```

**Two numbers carry the experiment:**
- **Cell B** (same look, different physics): UMF routing accuracy ≫ S-dyn's early-episode accuracy.
- **Cell C** (different look, same physics): **charts committed = 0** for ATLAS.

**F2a** — grouped bars, routing accuracy by cell and router.

### 7.4 E3 + E4 — expansion ladder and continual stream (RQ3, RQ4). *Days 7–8, ~21 GPU-h*

**One stream, seven arms.** S2 = `A,B,A,B,A,B` (R0/R1), 20 episodes per segment × 3 seeds, paired seeding: episode `i` of segment `s` uses seed `hash(s,i)` for **every** arm.

| # | Arm | Adapts | Persists | Library+routing | Expands | Verifies |
|---|---|---|---|---|---|---|
| 1 | Frozen | | | | | |
| 2 | AdaJEPA | ✓ | | | | |
| 3 | Persistent-AdaJEPA *(ours)* | ✓ | ✓ | | | |
| 4 | ATLAS-fixed-library | ✓ | ✓ | ✓ | | |
| 5 | ATLAS-detect-only | ✓ | ✓ | ✓ | ✓ | |
| 6 | **ATLAS** | ✓ | ✓ | ✓ | ✓ | ✓ |
| 7 | Oracle-ID | — | — | oracle | — | — |

Arms 4→5→6 *are* E3, the expansion ablation; arms 1→2→3→6 are the adaptation ladder. **One run produces both results.**

```bash
python scripts/run_e4.py --stream s2 --arms frozen adajepa adajepa_persist \
    atlas_fixed atlas_detect atlas oracle_id --episodes 20 --seeds 3 --out $ATLAS_OUT/e4
```

**T2 (body) — ladder, recall and expansion**

| Arm | SR overall | SR first visit A | SR final revisit A | paired Δ [CI] | McNemar *p* | Charts committed | Probes rejected |
|---|---|---|---|---|---|---|---|
| Frozen | | | | ≈ 0 | | 0 | — |
| AdaJEPA | | | | ≈ 0 (resets) | | — | — |
| Persistent-AdaJEPA | | | | **< 0 expected** | | — | — |
| ATLAS-fixed-library | | | | | | 0 | — |
| ATLAS-detect-only | | | | | | **> 2 expected** | — |
| **ATLAS** | | | | **> 0 expected** | | **≈ 2 expected** | |
| Oracle-ID | | | | ceiling | | — | — |

True regime count for S2 is **2**. Detect-only over-committing while ATLAS lands near 2 is the C2 result.

### 7.5 E5 — cross-policy diagnostic. *Day 5, ~3 GPU-h, supplementary*

`M[i,j]` = chart `i`'s UMF on chunks generated by chart `j`'s plans. Report per-column argmin accuracy and a column-normalised heatmap. Half a day; answers the sharpest objection without consuming a core slot.

### 7.6 Baseline configurations

| Arm | Configuration |
|---|---|
| Frozen | `dino_wm_pusht`, no adaptation |
| AdaJEPA | 1 grad step/replan; buffer = 5 most recent transitions; predictor lr `5e-4`; Adam; encoder frozen (= `predlast+encfrozen`); **re-init from pretrained each episode** |
| Persistent-AdaJEPA *(ours)* | as above, **no per-episode re-init**. Labelled as our modification |
| ATLAS-* | same loss, lr, optimiser, buffer size as AdaJEPA — **only the library/routing/expansion differ** |

Planner constant everywhere: CEM 200 × 10, horizon 25, 5 executed actions, ≤30 MPC steps.

### 7.7 Hyperparameters — fixed, not swept

| Symbol | Meaning | Value |
|---|---|---|
| `τ` | UMF adequacy threshold | 0.5 |
| `q` | strikes to arm the probe | 3 |
| `m` | hysteresis margin | 0.05 |
| `n_probe` | probe fitting steps | 20 |
| `K_max` | library cap | 10 |
| refine steps | per check | 1 (AdaJEPA's) |
| min-motion | informative-chunk gate | 10th pct of training displacement |
| chart lr | refinement | 5e-4 (AdaJEPA's) |

**One sensitivity run on day 10 if time:** `τ ∈ {0.3,0.5,0.8}` × `q ∈ {2,3,5}` on a single seed. Nine short runs, supplementary table. *(v1's full sweep is cut.)*

---

## 8. Statistics

```python
def normalised_recovery(sr_fit, sr_oracle, sr_random, min_spread=0.10):
    spread = sr_oracle - sr_random
    return None if spread < min_spread else (sr_fit - sr_random) / spread

def paired_bootstrap(a, b, n=10_000, seed=0):
    """a, b: per-episode binary outcomes in the SAME episode order."""
    d = a - b
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    return d.mean(), np.percentile(d[idx].mean(1), [2.5, 97.5])

def mcnemar_paired(a, b):
    from statsmodels.stats.contingency_tables import mcnemar
    return mcnemar([[((a==1)&(b==1)).sum(), ((a==1)&(b==0)).sum()],
                    [((a==0)&(b==1)).sum(), ((a==0)&(b==0)).sum()]], exact=True).pvalue
```

Every table reports **Δ with CI**, never two bare means. 20 ep × 6 segments × 3 seeds = **360 paired episodes per arm** — a 10 pp paired difference is comfortably detectable; the same difference unpaired would not be.

---

## 9. Gates — run before trusting anything

```bash
python scripts/smoke_gates.py --all
```

| Gate | Day | Test | Catches |
|---|---|---|---|
| **G1 identity** | 4 | Library `{c₀}` only ⇒ trajectory **bit-identical** to frozen (same seed); `UMF(c₀)` on clean data ≪ 1 | any error in the chart apply/restore path |
| **G2 prequential** | 6 | Over-refine chart X on `W` (50 steps), score all charts on the **next** window `W′` — X must not automatically win | leakage; every routing result would be an artifact |
| **G3a probe fires** | 6 | Genuinely new regime ⇒ probe passes, chart commits | expansion is dead code; the stream would measure a single-chart method |
| **G3b probe discriminates** | 6 | Inject observation noise (unfixable) ⇒ probe **rejects**, nothing commits | C2 would be vacuous |
| **G4 regimes real** | 3 | 20 random-action rollouts per regime ⇒ trajectories differ visibly and statistically | a fake shift invalidates everything downstream |
| **G5 pairing** | 3 | Two arms, same seeds ⇒ identical initial states and goals per episode index | statistics would be underpowered |
| **G6 denominator** | 3 | Chunks with displacement below the gate return `None`; no score, no strike, no probe; confirm no pathological behaviour when the agent is stationary | UMF blow-up near zero motion |

---

## 10. Figures

| ID | Content | Script |
|---|---|---|
| **F1** *(body)* | **Money plot.** Success (rolling mean, window 5) vs. episode across S2; regime boundaries dashed; ★ chart committed, ○ probe fired but rejected. All 7 arms, 95 % CI band over 3 seeds | `plots.money_plot` |
| **F2** *(body)* | **(a)** 2×2 routing accuracy, UMF vs S-dyn, by cell. **(b)** library size vs. episode for arms 4/5/6, horizontal line at true regime count = 2 | `plots.two_panel` |
| S1 | UMF traces: one line per chart across the stream, selected chart shaded | `plots.umf_traces` |
| S2 | Cross-policy heatmap (E5) | `plots.crosspolicy` |
| S3 | UMF vs success scatter with Kendall τ — C3 validation | `plots.umf_vs_sr` |

Matplotlib only, colour-blind-safe, `.pdf` 300 dpi, fonts ≥ 8 pt at column width.

---

## 11. Day-by-day

| Day | Deliverable | Done when |
|---|---|---|
| **1** | Fork, env, `dino_wm_pusht` loads, `pusht` downloaded | `torch.hub.load` works |
| **2** | Reproduce frozen Push-T planning; **profile one episode**; dump predictor params (§5.1) | ±3 pp of published; budget computed; LN vs AdaLN confirmed |
| **3** | `regimes.py`, `streams.py` (paired seeding), `score.py` | **G4, G5, G6 pass** |
| **3–4** | **E0** — 6 fine-tunes | adapter kind chosen; T5 |
| **4** | `chart.py`, `library.py` | **G1 passes** |
| **4–5** | **E1** routing | **T1 → GO / PIVOT** |
| **5** | **E5** cross-policy (supplementary) | heatmap |
| **6** | `router.py`, `expand.py`, `loop.py`, `adajepa.py`; all 7 arms wired | **G2, G3a, G3b pass** |
| **7–8** | **E4 + E3** stream, 7 arms × 3 seeds *(runs overnight)* | F1, F2b, T2 |
| **9** | **E2** 2×2 | F2a |
| **10** | Sensitivity run; all figures; polish | assets final |
| **11** | Finish 4 pages | submitted |

**Draft §1–3 of the paper from day 6** while streams run — none of it depends on results.

**Scope-cut ladder:** sensitivity run → E5 → E2 cell D → 20→15 episodes/segment → drop arm 4. **E0 + E1 + E2 + E3/E4 with T1/T2/F1/F2 is the complete paper.**

---

## 12. Compute budget

| Item | GPU-h |
|---|---|
| E0 capacity (6 fine-tunes) | ~3 |
| E1 routing | ~4 |
| E5 cross-policy | ~3 |
| **E3 + E4 stream** (7 arms × 6 segments × 20 ep × 3 seeds) | **~21** |
| E2 2×2 | ~6 |
| **Total** | **~37** |

Peak VRAM ~6–10 GB (CEM batch 200 × horizon 25, ViT-S/14 + depth-6 predictor; scoring is forward-only). Backprop touches ≤55 k adapter params and **never the ViT encoder**.

---

## 13. Risks

| Risk | Detect by | Response |
|---|---|---|
| **E1 fails (RQ1 weak)** | Day 5, T1 | Pivot to the negative result; T1 already contains the `e₁`-vs-UMF comparison. Six days remain |
| Predictor uses AdaLN, no free LN affine | Day-2 dump | `dino_wm_pusht` (plain LN) is primary for exactly this reason; else target the AdaLN conditioning MLP |
| Physics edits don't change dynamics | **G4**, day 3 | Different parameter; verify by rendering. Never proceed with a fake shift |
| Episodes slower than 40 s | Day-2 profile | Cut per §7.0, uniformly across arms, and report it |
| Probe never/always fires | **G3**, day 6 | Adjust `τ`, `q`; if degenerate, report the sensitivity as the finding |
| S-dyn ties UMF | E1/E2 | Report honestly; finding becomes *route by dynamics, not appearance* |
| TorchHub `HTTP 503` | Any day | `rm uv.lock && uv sync`, or the direct `fbaipublicfiles` URLs (§3) |
| MuJoCo/EGL render failure | Day 1–2 | `MUJOCO_GL=osmesa` |

---

## 14. Release checklist

- [ ] Fork of `jepa-wms` with `atlas/` added; upstream diff = one hook
- [ ] `uv.lock` pinned + conda env YAML
- [ ] Regime configs with exact physics values
- [ ] Chart libraries (`.pt`) for every experiment
- [ ] Per-episode JSONL: seed, segment, regime, selected chart, UMF for all charts, success
- [ ] `smoke_gates.py` in CI (G1–G6 on every commit)
- [ ] `make_tables.py --all && make_figures.py --all` regenerates every number from the logs
- [ ] README documenting the AdaJEPA substrate difference (§0) and our reimplementation choices
