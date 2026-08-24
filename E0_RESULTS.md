# E0 Results — Adapter Capacity (Final Run, 2026-08-23)

## Update (2026-08-24, part 3): CEM config was wrong for everything below — corrected, and the headline finding does not cleanly replicate

**Everything in "part 2" and the graduated-screen table further below (the 18-episode sweep, the
baseline-vs-adapter comparisons, the CEM-cost-ranking diagnostic) was run under the wrong CEM
planner configuration.** `scripts/run_e0_planning.py` used `num_samples=300, iterations=30,
horizon=6, num_act_stepped=6` — pulled from jepa-wms's own shipped eval YAML and mislabeled "the
published spec" in this script's own comments. The implementation plan's actual, explicitly
project-wide planner budget (§7.0, restated in §7.6: *"Planner constant everywhere"*) is **CEM 200
samples × 10 opt steps, subplanner horizon 25, 5 executed actions per replan, ≤30 MPC steps"** —
a materially different setup: the old config collapsed to a single one-shot replan per episode;
the correct one runs ~6 replans per episode with real closed-loop correction. A second bug was
found fixing this: `num_act_stepped` must be **1** (one model-chunk = `FRAMESKIP=5` raw actions),
not 5 — `num_act_stepped=5` empirically gives only ~2 replans/episode (25 raw actions/replan), not
the ~6 the plan describes; `num_act_stepped=1` gives exactly 6, confirmed by direct measurement.
(`scripts/run_e1.py` has the same `num_act_stepped=5` bug — not fixed there, out of scope for this
file.) Both fixed in `scripts/run_e0_planning.py`/`modal/modal_e0_planning.py`; new results below
and going forward use the corrected config. Old results are kept below for the record (each row
carries its own config values) but should not be read as current findings.

**First corrected-config check (baseline vs. `ln_act`, R1, episode 0 — the same task instance the
part-2 diagnostic used):**

| | block_pos_diff (success <20) | block_angle_diff (success <0.35) | Result |
|---|---|---|---|
| baseline | 38.99 | 0.049 | Fail |
| ln_act | 48.71 | 0.301 | Fail |

n=1 each (a second episode was cut short by an unrelated operational issue — a killed local Modal
CLI launcher process eventually propagated a cancellation to the remote job, ~40 min into a ~42 min
episode; not a code bug, more episodes to follow). **The part-2 headline result — baseline cleanly
solves episode 0, every adapter fails it identically — does not replicate under the corrected
config.** Under real closed-loop CEM (6 corrective replans instead of 1 one-shot), *neither*
baseline nor `ln_act` succeeds on this instance. `ln_act` is still worse on both metrics (further
from the goal, much worse angle), so the qualitative direction (adapter underperforms baseline) may
still hold, but the dramatic "clean win vs. complete failure" contrast reported in part 2 was very
likely an artifact of the broken single-shot config, not a robust effect. **Treat part 2's rank-
correlation-≈0 / "ranking inversion" finding as not yet re-validated** — it may still be real, but
needs to be re-measured under this corrected config before being trusted, per the same caution.

Also run alongside this: a controlled retraining check on `ln_act`/R1 with 5 trajectories × 30 steps
(vs. the original 3×10) to test whether E0's original charts were simply undertrained. Result:
**eval UMF got worse, not better** (0.68 → 1.11) — richer training on the same scripted-walk action
distribution did not help, and if anything hurt. This weakens the "just undertrained" hypothesis and
is more consistent with a genuine distribution-mismatch explanation (E0's training actions don't
resemble what CEM actually explores) — though this chart's *planning* success (not just UMF) hasn't
been checked yet.

## Update (2026-08-24, part 2): confirmed mechanism — E0 fine-tuning distorts CEM's cost ranking

**Superseded by the CEM-config correction above — read that note first.** The diagnostic below was
run under the wrong planner config; the mechanism it describes may still be real but has not yet
been re-validated under the corrected setup.

Following up on the finding below (baseline solves episode 0 at R1; every adapter fails it
identically), `scripts/diagnose_cem_costs.py` (new, one-off diagnostic — monkey-patches
`CEMPlanner.cost_function` at runtime, no jepa-wms files touched) captured CEM's actual
per-candidate costs for `baseline` vs. `ln_act`, same episode (seed=0, R1), full published spec
(num_samples=300, iterations=30).

**Key fact this relies on:** CEM's planner always starts from `mean=0, std=var_scale` and draws
candidates via a generator seeded from `cfg.local_seed` (hardcoded to 0). Since only the *predictor*
is adapted — the DINOv2 encoder is frozen and identical either way — iteration 0 samples the exact
same 300 candidate action sequences under both models. Verified directly: `max abs diff: 0.0` between
the two runs' iteration-0 action tensors. This makes the iteration-0 costs a clean, same-input,
different-model comparison — not an artifact of different units or different candidates.

**Result — the two models don't just disagree by a shift, they disagree about which actions are
even good:**
- Spearman rank correlation between baseline's and `ln_act`'s costs for the identical 300
  candidates: **0.089 (p=0.12)** — statistically indistinguishable from zero.
- Baseline's best candidate (cost 0.378 under baseline) is ranked a reasonable #13/300 by `ln_act`.
- **`ln_act`'s best candidate (cost 0.333 — its own top pick) is ranked #110/300 by baseline**
  (baseline scores it 0.622, worse than baseline's own average of 0.652 across all 300). The
  specific action the adapter is most confident is good is, by the frozen model's own (empirically
  validated) judgment, a mediocre-to-bad choice.
- By the final iteration, each model has converged to its own preferred region: baseline's elite
  cluster is tighter and lower (mean 0.076, max 0.218) than `ln_act`'s (mean 0.117, max 0.411) —
  roughly 1.5-2x worse, measured in the adapter's own cost units, not just in real-world outcome.

**Conclusion:** this is not a case of "the adapter is slightly worse everywhere." `ln_act`'s CEM
search converges *confidently* (tight, low-variance elite set by iteration 30 — the search itself
works correctly) toward an action that its own distorted cost landscape rates as excellent, but
which both the frozen model's cost function and the real environment agree is mediocre. This matches
exactly what happened in reality: baseline executed a real working push (14px from goal, success);
`ln_act` executed its confidently-chosen action and the block barely moved at all (32px — unchanged
from the initial displacement, failure). E0's offline fine-tuning objective (open-loop prediction
error on 3 training trajectories × 10 steps) measurably distorts the predictor's counterfactual
cost/ranking function that CEM depends on, even though it can simultaneously *improve* the offline
UMF metric (as it did for `ln_act` at R1). **Minimizing prediction error does not necessarily
preserve planning competence** — a citable finding in its own right (the proposal's §3.2 already
names this tension in general terms; this is now direct, quantified evidence of it in this substrate).

**Not yet checked:** whether this generalizes beyond this one episode/adapter/regime (`ln_act`/R1/
seed=0) — this diagnostic was deliberately run on a single, cheap, clean instance to establish the
mechanism, not to characterize how often or how severely it occurs. Raw captures:
`atlas_out/e0_planning/cem_diagnostics/{baseline,ln_act}_R1_seed0.json`.

## Update (2026-08-24, part 1): two real bugs in the planning-success harness fixed; new finding — trained adapters may be *hurting* real CEM planning at R1/R2

**Superseded by the CEM-config correction in part 3 above** — the dataset-goal/angle-wrap fixes
described here are still correct and still in effect, but the 18-episode graduated-screen table
this section leads to was run under the wrong CEM planner config. Read part 3 first.

The first planning-success runs (previous section below) showed 0% success across every kind/regime
combo, including the frozen baseline. Investigation found two real bugs in `run_e0_planning.py`
itself (not in `atlas/`'s core ATLAS logic), both now fixed:

1. **Success check coupled to an irrelevant quantity.** `PushTWrapper.eval_state()` (jepa-wms's own
   utility) compares `state[:4]` = `[agent_x, agent_y, T_x, T_y]` together — meaningful only when
   goal/init states come from a correlated real trajectory (`goal_source=dset`). Our goals were
   independently random (`sample_random_init_goal_states`), so the "target pusher position" baked
   into the goal had nothing to do with where the pusher would sensibly end up after actually placing
   the block — success required both to align by pure chance, which essentially never happens.
   **Fixed:** `run_e0_planning.py::block_success()` now compares only the T-block's position/angle
   (indices 2-4), matching Push-T's actual objective (and IBC/Diffusion Policy's own metric).
2. **Angle-wrap bug, both upstream and inherited.** `generate_state()` draws the goal angle as
   `randn()*2pi - pi` — an unbounded Gaussian, not a wrapped angle — so a raw angle difference could
   span several full rotations. The wrap formula (`min(d, 2pi-d)`), valid only within one rotation,
   produced garbage (even negative) results for larger differences. **Fixed:** proper fold-into-range
   formula (`abs((d+pi) % 2pi - pi)`), verified against edge cases including multi-rotation input.
3. **Root cause underneath both:** independently-random init/goal pairs are frequently far apart
   (167-223px commonly), asking CEM to solve in one 30-step, no-replan shot — a near-impossible
   one-shot control problem regardless of adapter quality, producing a floor effect that looked like
   "0% success everywhere" but was actually "the benchmark itself was unsolvable." **Fixed:** goals
   are now sampled from real recorded Push-T trajectory segments (`data/pusht_noise/train/states.pth`
   — only the 175MB raw-state file is needed, not the full 7GB dataset with visual/action/token data).
   `init_state = states[ep, offset]`, `goal_state = states[ep, offset+30]`, both padded with zero
   velocity to match `with_velocity=True`. Verified: median displacement dropped from ~167-223px to
   ~35px across 50 samples — a plausible, solvable one-shot task.

### Graduated validation (per a staged plan: fix → smoke-test → small screen → decide)

**Frozen baseline (no chart) at R0:** 2/2 success, real margins (`block_pos_diff` 9.5/14.6,
well under the 20px threshold) — confirms the fixed benchmark is genuinely solvable.

**Small 2-3-episode screen, ln_act/lora4/full × R1/R2 (18 episodes) vs. frozen baseline at R1/R2
(6 episodes), same seeds for direct comparison:**

| | Ep0 | Ep1 | Ep2 | Success |
|---|---|---|---|---|
| Baseline R1 (no adapter) | 14.2 / 0.05 **T** | 12.9 / 0.14 **T** | 62.2 / 0.70 F | **2/3** |
| Baseline R2 (no adapter) | 9.3 / 0.07 **T** | 19.7 / 0.15 **T** | 62.2 / 0.70 F | **2/3** |
| ln_act R1 | 32.4 / 0.05 F | 19.5 / 0.18 **T** | 62.2 / 0.70 F | 1/3 |
| lora4 R1 | 32.4 / 0.05 F | 39.9 / 0.39 F | 25.3 / 0.33 F | 0/3 |
| full R1 | 32.4 / 0.05 F | 17.8 / 0.44 F | 25.1 / 0.05 F | 0/3 |
| ln_act R2 | 32.4 / 0.05 F | 45.8 / 0.97 F | 31.1 / 1.16 F | 0/3 |
| lora4 R2 | 32.4 / 0.05 F | 45.8 / 0.97 F | 62.2 / 0.70 F | 0/3 |
| full R2 | 32.4 / 0.05 F | 14.0 / 0.67 F | 62.2 / 0.70 F | 0/3 |

(values are `block_pos_diff` / `block_angle_diff`; success threshold is pos<20, angle<0.35 rad)

**Baseline: 4/6 (67%) across R1+R2. All adapters combined: 1/18 (5.6%).**

### Two structural patterns in the data

- **Episode 2 fails identically everywhere** (`62.2/0.70`, byte-exact across all 8 runs — every
  kind, every regime, including baseline). Init/goal sampling depends only on the episode seed, not
  regime/chart, and the final position exactly equals the *initial* displacement (the block never
  moved). This looks like a genuinely hard/unreachable task instance at this horizon (agent starts
  too far from the block to make contact and complete the push in one 30-step, no-replan shot) —
  independent of model quality, not evidence of anything broken.
- **Episode 0 is the real finding.** Baseline solves it cleanly in both regimes (14.2 then 9.3px);
  every trained adapter fails it identically (32.4px — again exactly the initial displacement, zero
  progress). Applying *any* chart turned a solvable episode into a complete failure on this specific
  instance, in both R1 and R2.

### Interpretation — a real, reportable result, not a broken pipeline

The near-miss numbers throughout are sane (no garbage/NaN/floor effect) — this isn't a code bug in
the new harness. But it directly contradicts E0's own UMF-based ranking: the offline UMF screen found
`ln_act`/`lora4` *improving* over baseline at R1 (Reduction vs. baseline: ln_act +46.7%, lora4 +31.4%),
yet here, under real CEM-driven planning, every adapter does *worse* than no adapter at all. This is
exactly the risk `E0_RESULTS.md`'s caveat #6 flagged as hypothetical — "the ranking above is a screen
based on one specific, non-planner action distribution, and could plausibly change" — now observed
directly rather than just anticipated.

**Leading hypothesis (not yet confirmed):** E0's fine-tuning optimizes open-loop prediction error on
the *observed* held-out trajectory (3 training rollouts × 10 steps, itself a very small/narrow action
distribution — see caveat #1). CEM, however, scores hundreds of *counterfactual* candidate action
sequences that were never part of that training data. An adapter can plausibly get better at
predicting the one trajectory it saw while getting *worse* at correctly ranking the many hypothetical
action sequences CEM actually needs to compare — i.e., minimizing prediction error does not
necessarily preserve planning competence. This would itself be a legitimate, citable finding for
ATLAS (the proposal's own §3.2 already flags this exact tension in general terms), not just a
methodology problem.

**Not yet confirmed because we don't have the evidence for the mechanism.** `run_e0_planning.py`
currently logs only the *executed* action's outcome — not the CEM planner's internal candidate
rankings/costs. To confirm or rule out the hypothesis directly, the next diagnostic step (not yet
run) is: for episode 0 specifically (same seed, same CEM `local_seed`), instrument the planner to log
the top-K candidate action sequences and their predicted costs under the frozen baseline vs. an
adapter, and check whether the adapter inverts the cost ranking (rates a no-contact/bad action as
better than the correct push). This needs code changes to capture (jepa-wms's `CEMPlanner` doesn't
expose this by default) — not yet implemented as of this write-up.

**Other hypotheses not ruled out:** too little/narrow E0 training data (30 transitions/regime) to
safely modify a predictor CEM depends on; the fine-tuning objective/hyperparameters causing
over-adaptation to the tiny training set; a residual implementation issue in chart apply/restore
(considered less likely given `full`/`ln_act`/`lora4` all show the same qualitative pattern, and
Chart apply/restore already passed its own bug fixes in `code-review.md` Bugs #1/#3/#4/#6f).

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
