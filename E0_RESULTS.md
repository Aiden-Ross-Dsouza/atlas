# E0 Results — Adapter Capacity (Final Run, 2026-08-23)

## Update: planning-success half — validated end-to-end on Modal (L4 GPU)

`modal run modal/modal_e0_planning.py --kind ln_act --regime R1 --episodes 1 --num-samples 100`
completed successfully: checkpoint downloaded, `chart_ln_act_R1.pt` applied, 1 episode ran
(`success=False steps=30 replans=1 wall_time=136.7s`, peak GPU memory 6.64GB). Getting here required
two Modal-specific fixes (`code-review.md` Bug #8) on top of the memory finding below — both are
resolved now. Real (non-extrapolated) cost data point: **136.7s per episode at num_samples=100**
(1/3 of the published spec's 300); scaling roughly linearly with num_samples, full spec extrapolates
to **~7 minutes/episode** on an L4 — much cheaper than the earlier ~14.4 min/episode estimate, which
was extrapolated from a slower, memory-constrained local 6GB GPU rather than measured directly. At
L4's $0.80/hr, that's roughly **$0.09/episode** at full spec — not yet validated at num_samples=300
itself, only extrapolated from the num_samples=100 run above.

## Earlier note: planning-success half — script written, needs a bigger GPU

`scripts/run_e0_planning.py` (new) implements the missing Success column below. It's
correctness-tested at reduced CEM settings (small `num_samples`), but the published spec
(`num_samples=300`) OOMs on this machine's 6GB GPU — a single planning call tries to allocate
13.5GB+. Extrapolated: **~14.4 minutes per episode** at full spec (one CEM planning call already
covers a full 30-step episode with the shipped config's `horizon=6`/`num_act_stepped=6`/
`frameskip=5`). Adapter kind barely matters for this cost — `ln_act`/`full` ≈160ms, `lora4` ≈200ms
per 30-step forward rollout (LoRA's extra low-rank matmul). Full findings: `code-review.md` Bug #7.
**Needs Modal** (or another GPU with meaningfully more than 6GB) to run at spec.

## Status: E0 is only half finished — read this before trusting the ranking below

E0's own definition (implementation plan §7.1) has **two** deliverables: "Evaluate UMF on held-out
regime trajectories **and planning success in-regime**." Table T5's columns confirm this:
`Adapter | Params | KB | ΔUMF (R1/R2) | Success (R1/R2) | % of full`. Only the UMF half is done —
the **Success** column requires the CEM planner actually driving episodes, and that integration
(`atlas.loop.atlas_step()` wired into `evals/simu_env_planning/`'s rollout loop) does not exist yet
(the single biggest blocked item in this project — see `claude.md` §0.1). This is **not** the same
gap as E1 (a separate, later experiment that tests whether UMF-based *routing among multiple
charts* works, using whichever adapter kind E0 selects) — E0 itself is incomplete on its own terms,
independent of E1 ever running.

**Why the UMF half is still worth having despite this:** it's a standard, deliberately cheap proxy
used to prune the adapter-kind search space before spending expensive planner-driven compute
validating the winner — the proposal names this exact tradeoff directly (§3.2, citing JEPA-WM:
*"models which unroll many actions faithfully do not thereby succeed at planning... its correlation
with success is imperfect"*). But a proxy is only as good as how well its input distribution
resembles what it's a proxy for — see the new caveat below, which is a real, unresolved gap in that
resemblance, not just the usual "proxies are imperfect" disclaimer.

Full technical detail behind every fix mentioned here lives in `code-review.md` (Bugs #1–#6f),
`REGIME_DESIGN_REVIEW.md` (mass-cancellation physics), and `ACTION_SAMPLING_REVIEW.md`
(action-sampling literature review). This document is the results-focused summary: what changed,
what the final numbers are, and how to read them.

## What changed since the first (invalid) E0 run

The first E0 run (results now archived at `atlas_out/e0_pre_regime_fix_2026-08-22/`) predates six
fixes, applied in this order as they were discovered:

1. **Predictor-state contamination** (`scripts/run_e0.py`) — the shared predictor was never reset
   between fine-tunes, so later charts trained on top of earlier charts' weights instead of the
   pretrained checkpoint. Fixed: snapshot + reload before every kind/regime.
2. **`ln_act`/`lora4` parameter-selection gaps** (`atlas/chart.py`) — `ln_act` missed ~half the
   LayerNorms (an unnamed `Sequential`-indexed one); `lora4` missed the attention output
   projection (`to_out`, not matched by the original name list). Fixed via structural
   (`isinstance`) detection and an expanded name list.
3. **R1's mass-based regime shift is physically dead at any scale** — Push-T's pusher is a
   `pymunk.Body.KINEMATIC`; Chipmunk2D's solver makes a struck dynamic body's post-collision
   velocity algebraically independent of its own mass. Re-targeted R1 → `shape.friction`,
   R2 → `shape.elasticity` (both were 0.0, unset, in the shipped env). Also fixed an unrelated bug
   in the same file: `atlas/regimes.py` imported `gymnasium` instead of legacy `gym`, so
   `PhysicsRegime` had never actually worked against the real environment before this.
4. **Trajectory sampling rarely produced contact** — fully random per-step actions produced
   agent-block contact in only 13–17% of rollouts. Re-targeted to a persistent random-target +
   proportional-aim scheme (matching how every public Push-T lineage — IBC, Diffusion Policy,
   DINO-WM — actually generates data), plus a rejection-sampling retry. Now 100% contact across
   30-seed validation for all three regimes.
5. **Action-scale calibration mismatch** — the aimed sampling's raw actions had std ~2.2x the
   checkpoint's own `preprocessor.action_std`, producing genuinely out-of-distribution model
   inputs after normalization. Fixed with a tuned `ACTION_GAIN=0.25`. This shrank per-step
   displacement, which then required lengthening trajectories (`traj_len` 10 → 50) to keep UMF's
   denominator (observed motion) large enough to be well-behaved.
6. **GPU memory / `Chart.load()` bugs** — the longer trajectories OOM'd training on a 6GB GPU
   (backprop scales with rollout length), so training and eval trajectory lengths were decoupled
   (`--train-traj-len 10`, `--eval-traj-len 50`). Separately, `Chart.load()` corrupted `lora4`
   charts' `_param_names` on reload, crashing the resume path and leaving the predictor
   permanently stuck mid-parametrized; fixed by recomputing `_param_names` structurally instead of
   trusting the saved file's dict keys, and by widening `evaluate_e0_chart()`'s `try/finally` to
   cover `chart.apply_()` too.

## Two diagnostic checks run before trusting any of this

**Per-horizon error decomposition** (does error explode with rollout length, suggesting the model
itself is broken?): No. Ratio was highest at horizon 1 (small early denominator) and *improved*
with longer horizons — ruling out open-loop error compounding as the dominant issue, and pointing
instead at the denominator-sensitivity mechanism that fix #4/#5 above address.

**Action-scale audit** (is the model being fed actions consistent with what it was trained on?):
No, initially — raw action std was ~2.2x the checkpoint's calibration, confirmed and fixed as #5
above.

## Final results — all three regimes, three adapter kinds

| Regime | Kind | Train Loss | Eval Loss | Eval UMF |
|---|---|---|---|---|
| R0 (default) | `ln_act` | 0.0392 | 0.7497 | 0.6738 |
| R0 (default) | `lora4` | 0.0218 | 0.7581 | 0.6758 |
| R0 (default) | `full` | 0.0015 | 0.8831 | 0.8117 |
| R1 (high friction) | `ln_act` | 0.0466 | 0.5608 | 0.6801 |
| R1 (high friction) | `lora4` | 0.0263 | 0.6729 | 0.8766 |
| R1 (high friction) | `full` | 0.0035 | 0.9166 | 1.2976 |
| R2 (high restitution) | `ln_act` | 0.0497 | 0.8611 | 1.2398 |
| R2 (high restitution) | `lora4` | 0.0293 | 0.8917 | 1.3796 |
| R2 (high restitution) | `full` | 0.0026 | 0.8805 | 1.6705 |

## Baselines (untrained/identity chart, same pipeline, same regime physics)

| Regime | Baseline UMF |
|---|---|
| R0 | 0.7457 |
| R1 | 1.2771 |
| R2 | 0.5873 |

## Reduction vs. baseline (negative = fine-tuning made it worse)

| Regime | `ln_act` | `lora4` | `full` |
|---|---|---|---|
| R0 | **+9.6%** | **+9.4%** | −8.9% |
| R1 | **+46.7%** | **+31.4%** | −1.6% |
| R2 | −111.1% | −134.9% | −184.4% |

## Interpretation

- **`ln_act` (the smallest adapter, ~10.7k params) is the most consistent performer** — it's the
  best or tied-best kind in every regime, and the only kind that never catastrophically regresses.
- **`full` (20.8M params) never wins**, and is the worst performer in R1 and R2. Its train loss is
  always near-zero (0.0015–0.0035) — classic overfitting to just 3 short training trajectories.
  Full capacity memorizes the specific training rollouts rather than learning something that
  generalizes to the 2 held-out eval trajectories.
- **R2 (elasticity shift) is a real negative result, not a bug**: every adapter kind gets *worse*
  than doing nothing. R2's baseline (0.587) is already the best of the three baselines — there's
  little room for improvement and, with so little training data, more room for fine-tuning to pull
  the model away from an already-decent zero-shot fit than to improve it.
- **The pre-registered rule** ("smallest kind reaching ≥90% of `full`'s gain, in both regimes")
  **cannot be applied as literally stated** — `full` isn't the best performer anywhere, so there's
  no positive "full's gain" to measure 90% of in R1 or R2. Per `claude.md` §1.8 ("a failed
  pre-registered criterion is a result — report it, don't fix it"), this is reported as-is rather
  than reinterpreted to produce a winner.

## Caveats — read before treating any of this as final

1. **Extremely small sample.** 3 training trajectories, 2 held-out eval trajectories, per
   regime — every "Eval UMF" above is the average of just 2 numbers. This is a smoke-test-scale
   signal, not a statistically powered result.
2. **Single seed, single run.** No repeated trials, no confidence intervals. A different random
   seed could plausibly shift which kind "wins" a given regime, especially for the close R0/R1
   comparisons between `ln_act` and `lora4`.
3. **Training and eval trajectory lengths differ** (10 vs. 50 steps) — a deliberate, documented
   tradeoff forced by this machine's 6GB GPU (`code-review.md` Bug #6e), not a methodological
   choice made for its own sake. On a larger GPU, both could use the same (longer) length.
4. **`evaluate_e0_chart()` still doesn't apply the informative-chunk motion gate**
   (`compute_motion_gate`) — not consequential here since the new sampling reliably produces large
   displacement, but worth revisiting if a future eval set produces lower-motion chunks.
5. **R0 was not part of the original E0 design** (implementation plan §7.1 specifies 3 kinds × 2
   regimes, R1/R2 only) — trained here as a deliberate, user-requested addition for a fuller
   picture, not a plan correction.
6. **The trajectories used to train and evaluate every adapter above are scripted, not
   planner-generated — this is a real, unresolved gap, not just the usual "proxies are imperfect"
   disclaimer.** `load_regime_trajectories()`'s target-directed-walk-plus-noise scheme is
   goal-directed in spirit, but it is not CEM's actual behavior: CEM samples ~200 candidate action
   sequences per replan and picks whichever minimizes latent distance-to-goal *under the current
   (possibly chart-adapted) predictor* — a fundamentally different action distribution than "walk
   toward one fixed random point with Gaussian noise." An adapter that best fits our scripted walk
   is not guaranteed to be the adapter that best fits CEM's actual candidate-rollout statistics.
   Concretely: **the ranking above (`ln_act` > `lora4` > `full`) is a screen based on one specific,
   non-planner action distribution, and could plausibly change once real planner-driven trajectories
   are available.** This can only be resolved once the CEM-planner integration exists — either by
   re-running this half of E0 against planner-generated trajectories, or by treating E1's
   planner-driven results as the real test of whether this ranking held up. Until then, treat
   `ln_act` as "the current best guess," not "the confirmed answer."

## Raw artifacts

- `atlas_out/e0/results.json`, `atlas_out/e0/results.md` — machine-readable and Table-T5-formatted
  versions of the results table above.
- `atlas_out/e0/chart_{kind}_{regime}.pt` — the 9 trained charts.
- `atlas_out/e0/loss_{kind}_{regime}.json` — full 2000-step training loss curves.
- `atlas_out/e0_pre_regime_fix_2026-08-22/` — the archived, invalid first run (predates all fixes
  above), kept for reference/comparison only.
