# E0 Results — Adapter Capacity (Final Run, 2026-08-23)

## 🔴 P4 CAPACITY MATRIX COMPLETE (2026-08-25) — E0 fails; capacity is not the bottleneck

All four arms on calibrated **R2 (`{"damping": 0.5}`)**, N=20, P2d-filtered sampler,
`nas=6`. Pairing verified: `init_block_pos_diff` and `init_agent_block_dist` identical
across all four arms on all 20 episode indices, 0 mismatches. `regime_config` confirmed
`{'damping': 0.5}` in every JSONL.

Sources: `atlas_out/e0_v3_planning_dataset_baseline/`, `e0_v3_planning_dataset_ln_act/`,
`e0_v4_planning_lora4/`, `e0_v4_planning_full/`.

| Arm | trainable params | SR | Δ vs baseline | knock-aways | mean progress | zero-contact eps |
|---|---:|---:|---|---:|---:|---:|
| baseline (`c0`) | 0 | 45.0% (9/20) | — | 5/20 | +19.6px | 0 |
| **`ln_act`** | 10,764 | **50.0% (10/20)** | +5.0pp, CI [−10, +20] | **2/20** | **+37.3px** | 0 |
| `lora4` | 118,176 | 40.0% (8/20) | −5.0pp, CI [−20, +10] | 4/20 | +29.9px | 0 |
| `full` | 20,800,884 | 20.0% (4/20) | **−25.0pp, CI [−45, −10]** | 6/20 | **−2.9px** | **3** |

**Normalised recovery** against the 10-episode recoverable set `[2,3,5,10,12,14,15,17,18,19]`
(see R0 confound section below): `ln_act` **1/10**, `lora4` **1/10** (ep19), `full` **0/10**.

### Verdict — pre-registered criteria

- **§7.1's capacity rule ("smallest kind reaching ≥90% of `full`'s gain") cannot be applied
  as stated.** `full`'s gain is **negative** (−25.0pp). There is no winner to name. Report
  this as an inapplicable rule, not as a rule satisfied by the smallest arm by default.
- **E0's ≥15pp bar: FAILED.** The best arm (`ln_act`) reaches +5.0pp with a CI spanning zero,
  and recovers one tenth of the regime-opened gap.
- The **only statistically real effect in the matrix** is `full`'s degradation — the sole
  comparison whose CI excludes zero, and it points *against* the adapter.

### Mechanism — what `full` actually did

`full` is the only arm with **zero-contact episodes** (7, 14, 19: `block_pos_diff` byte-identical
to `init_block_pos_diff`), and the only arm with **negative mean progress** (−2.9px overall,
**−32.0px on easy episodes**). It is not overshooting a correction — CEM's plans stopped being
goal-directed. Fine-tuning 20.8M parameters on 20 trajectories degraded the frozen predictor's
goal-relevant structure outright. Its offline eval UMF (0.728, near the "no better than static"
ceiling, despite the lowest train loss) flagged this before planning ever ran.

**Honest scoping of this claim:** `full`'s failure is **confounded with training-set size**. It
is evidence that *capacity without proportionate data* hurts on this substrate — **not** a clean
demonstration that capacity per se is harmful. Do not write the stronger claim.

### ⚠️ Post-hoc, exploratory — NOT a pre-registered result

Splitting on initial block displacement at 90px (a threshold chosen **after** seeing the data —
garden-of-forking-paths hazard, treat as hypothesis only):

| Split | Arm | SR | mean final err | mean progress | knock-aways |
|---|---|---:|---:|---:|---:|
| Easy (<90px, n=11) | baseline | 7/11 | 55.8 | +1.8 | 2 |
| | **`ln_act`** | **8/11** | **30.9** | **+26.8** | **1** |
| | `lora4` | 6/11 | 55.6 | +2.0 | 3 |
| | `full` | 3/11 | 89.7 | −32.0 | 4 |
| Hard (≥90px, n=9) | baseline | 2/9 | 99.3 | +41.2 | 3 |
| | `ln_act` | 2/9 | 90.3 | +50.2 | **1** |
| | **`lora4`** | 2/9 | **76.6** | **+63.9** | **1** |
| | `full` | 1/9 | 107.8 | +32.7 | 2 |

`ln_act` and `lora4` appear to have **complementary competences** — `ln_act` owns short-range
correction, `lora4` reduces long-range error substantially (ep0 120→67.5, ep3 157→60.0,
ep5 82→46.9, ep15 104→64.1) without crossing the 20px success threshold. Aggregate SR hides
this because the effects cancel. **SR is identical (2/9) on the hard split** — the difference is
visible only in the continuous metric. This does **not** rescue E0's failed SR test. If it
survives a pre-registered replication it is an argument *for* a routed library; until then it is
a hypothesis.

### Interpretation

Across four capacity levels spanning three orders of magnitude, no adapter clears the bar, and
the largest actively regresses. Combined with the open-loop nature of the replay training data
(trajectories never show the model recovering from an overshoot), the evidence favours **the
training signal, not model capacity, as the bottleneck**. That is a reportable negative result
under §1.8 — not something to fix by tuning.

---

## 🟢 R0 CONFOUND CHECK (2026-08-25) — headroom is REAL and quantified; the "dilution" theory was wrong

Frozen baseline at **R0** on the identical P2d-filtered sampler, N=20, all four arms confirmed
paired (20/20 episode indices, no mismatches).

**R0 = 19/20 = 95.0%.** Versus R2 baseline's 45.0% → **+50.0pp, 95% CI [+30.0, +70.0]**. The
confound closes decisively in the "headroom is real" direction, and the gap is *larger* than the
pre-P2d calibration estimate of 40pp. The benchmark is sound.

### The recoverable set — E0's real denominator

Episodes R0 solves but the shifted baseline loses: **10 of 20** — `[2,3,5,10,12,14,15,17,18,19]`.

| Arm | recovers | of 10 |
|---|---|---|
| `ln_act` × dataset | ep17 | **1/10** |
| `ln_act` × hybrid | ep18 | **1/10** |

**Normalised recovery = (10 − 9)/(19 − 9) = 10%.** This is the number to report — the same
statistic E1 uses (`atlas/stats.py::normalised_recovery`), with R0-frozen as the ceiling. It is
far more informative than "+5pp with a CI touching zero": the chart recovers **one tenth** of
the gap the regime opened.

### Where the headroom lives — and a retracted claim

| init displacement | n | R0 | R2 baseline | R2 + dataset chart | recoverable |
|---|---|---|---|---|---|
| 0–80px | 11 | 11/11 | 7/11 | 8/11 | 4 |
| 80–120px | 2 | 1/2 | 1/2 | 1/2 | 0 |
| **120–300px** | **7** | **7/7** | **1/7** | **1/7** | **6** |

**🔻 RETRACTION.** The P3 section below theorised that the 120–300px episodes were
*intrinsically* hard and were diluting the measurement, and proposed adding a difficulty
**ceiling** to the sampler. **That was wrong.** R0 solves 7/7 of them. They are not hard — they
are exactly where damping is lethal, and they hold **6 of the 10** recoverable episodes.
Adding that ceiling would have deleted most of the benchmark's discriminative power. **Do not
add a `--max-block-pos-diff`.** `min_block_pos_diff=40` with no ceiling is correct as-is.

### The actual diagnosis

Damping-0.5 **selectively destroys long-range pushing** (7/7 → 1/7) while leaving short pushes
largely intact (11/11 → 7/11). That is consistent with the overshoot mechanism: the predictor
under-estimates post-contact glide, and that error compounds with distance travelled.

`ln_act` recovered **1 episode in the easy bucket and 0 of the 6 in the hard bucket** — it
fixes what barely needed fixing. Mechanistically this follows from E0's protocol: at
`num_act_stepped=6` there is **one replan**, so the whole 30-raw-step episode is planned
open-loop from t=0 off a 6-model-step prediction, and error compounds across the horizon.
Short pushes need the first step or two right; long pushes need all six. A 10.7k-parameter
LayerNorm nudge reduces the error (knock-aways 5→2, +21.3px mean) without surviving six
compounding steps.

**Two candidate explanations, both still open:**

| # | Explanation | Test |
|---|---|---|
| A | **Capacity** — 10.7k params cannot carry the correction; `lora4` (118k) / `full` (20.8M) might | **P4**, now defensible: confound closed, headroom quantified, and E0 cannot be reported complete on one adapter kind |
| B | **Horizon compounding** — any residual error kills a 6-step open-loop plan; closed-loop replanning would truncate it | **P5's re-baseline, run early**: frozen R0 and R2 at `nas=1` (6 replans). Already required work — see below |

Artifacts: `atlas_out/e0_v3_baseline_R0/{baseline_R0.jsonl,baseline_R0_summary.json}`.

---

## 🔴 P3 DECISION POINT (2026-08-25) — E0 **FAILS** its pre-registered criterion

**Headline: no chart cleared the pre-registered bar (≥15pp SR gain with a paired-bootstrap CI
excluding 0). Per `CLAUDE.md` §1.8 this is reported as a result, not fixed. E1 is not being
run; P4 is not being launched.**

Regime R2 = **damping 0.5** (calibrated in `E0_RECOVERY_PLAN.md` §0.5), N=20 paired episodes,
P2d-filtered sampler, all three arms confirmed paired (20/20 episode indices match).

| Arm | SR | vs baseline | 95% CI (paired bootstrap) |
|---|---|---|---|
| baseline (frozen) | 9/20 = 45.0% | — | — |
| `ln_act` × dataset-trained | 10/20 = 50.0% | +5.0pp | [0.0, +15.0] — touches zero |
| `ln_act` × hybrid-trained | 8/20 = 40.0% | −5.0pp | [−25.0, +10.0] — wrong direction |

### But "the chart does nothing" is NOT what the episode-level data says

Four independent signals, all pointing the same way for the **dataset-trained** chart:

1. **It is a strict superset of baseline.** Successes: baseline `{1,4,6,7,8,9,11,13,16}`,
   dataset chart `{…same nine…, 17}`. Gained ep17, **lost nothing**. McNemar discordant pairs
   b=1, c=0. Noise flips episodes in *both* directions; this did not.
2. **Knock-aways 5/20 → 2/20** — the mechanism damping-0.5 was chosen to expose, and a
   prediction **pre-registered before the run** (`E0_RECOVERY_PLAN.md` §0.5). It was confirmed.
3. **Mean final block distance improved +21.3px across the 10 shared failures** (better on
   7/10; ep12 alone by 107px — the +198px catastrophe from calibration).
4. The **hybrid** chart shows none of this: gained ep18 but lost ep1 and ep8 (b=1, c=2), no
   knock-away improvement (6/20, *worse* than baseline). This is the profile of a chart that
   is genuinely not helping — and it makes signals 1–3 harder to dismiss as noise.

### Why the improvement didn't convert into success rate

**The benchmark is diluted by intrinsically hard episodes.** Success by initial displacement:

| init_block_pos_diff | n | baseline | dataset chart | hybrid |
|---|---|---|---|---|
| 0–80px | 11 | 7/11 | **8/11** | 5/11 |
| 80–120px | 2 | 1/2 | 1/2 | 1/2 |
| **120–300px** | **7** | **1/7** | **1/7** | 2/7 |

Seven of twenty episodes ask for a 120–300px block displacement within 30 raw steps. Baseline
and chart both solve 1 of 7 — that third of the sample contributes almost no discriminative
power, it just drags both arms toward each other. `min_block_pos_diff=40` sets a floor on task
difficulty but **no ceiling**.

And the dataset chart's failures sit far from the 20px cliff — final distances
`[29, 49, 86, 86, 88, 91, 102, 121, 131, 205]`. Only one is near the threshold, so a
marginally better chart would not convert many of them either.

**Read together: the effect is real but small, and the instrument is too blunt to resolve it.**
At N=20 paired binary episodes, the minimum detectable effect is roughly 3 episodes (15pp) —
the pre-registered bar was, in hindsight, near the noise floor of its own sample size.

### The open confound that must be resolved before anything else

**R0's success rate on the P2d-filtered sampler was never measured.** P2d changed the sampler
(ep2 now resolves to a different, reachable pair), and only R2 was re-baselined. So we cannot
currently distinguish:

- **R0 ≈ 85%** → the 40pp damping headroom is real, the task is fine, and E0's negative result
  stands on its own terms (the chart is genuinely too weak).
- **R0 ≈ 50%** → the filtered sampler is intrinsically hard regardless of physics, damping is
  no longer what causes failure, and P3 was measuring task difficulty rather than regime
  adaptation — a **measurement** problem, not an E0 result.

The 1/7 rate on hard episodes for *both* arms makes the second possibility live. One 20-episode
frozen-baseline run at R0 settles it. **Do that before interpreting this result further.**

### What this settles regardless

**The open-loop-vs-closed-loop question (P2b) is answered: dataset replay wins.** The hybrid
(closed-loop) collector produced a worse chart on every axis — SR, McNemar, and knock-aways.
This contradicts the prior expectation recorded in `E0_RECOVERY_PLAN.md` §0.1/P2b that
closed-loop data would better represent deployment. Recorded as a negative result.

### Artifacts

Modal volume `atlas-data`: `atlas_out/e0_v3_dataset/`, `atlas_out/e0_v3_hybrid/` (charts +
loss/val curves + seed manifests), `atlas_out/e0_v3_planning_dataset_baseline/`,
`atlas_out/e0_v3_planning_dataset_ln_act/`, `atlas_out/e0_v3_planning_hybrid_ln_act/`
(per-episode JSONL + summaries).

---

## 🟡 SUPERSEDED BY P3 ABOVE (2026-08-25, post T1–T8 fix): first real post-fix results — frozen baseline + 3 trained charts

**Status: preliminary (n=10 paired episodes per cell, not T10's full ~20-seed design). Real, trustworthy
numbers under the repaired pipeline — first data since the rollout fix that can be cited at all — but
not yet statistically powered for a final verdict. All runs on Modal (L4), substrate config
(`num_samples=300, iterations=30, horizon=6, num_act_stepped=6`), same 10 paired seeds across every
cell (`atlas_out/e0_planning_filtered/`), real (init, goal) pairs sampled from `data/pusht_noise/train/`.**

### Bug found and fixed before any of this was trustworthy: trivial init/goal pairs

The first frozen-baseline@R0 sanity run (n=10, unfiltered sampling) measured 90% SR — matching
DINO-WM's published number almost exactly, which looked like a clean pass. It wasn't: 5/10 episodes
finished in ≤8 raw steps (one in a single action), because `sample_dataset_init_goal()` picks init/goal
states from the *same* real demo episode 30 raw timesteps apart, and real demos have idle/repositioning
stretches where the block barely moves over that window — sometimes landing init and goal already
within `block_success()`'s own 20px/π-9 threshold, making "success" free. **Fixed:**
`sample_dataset_init_goal()` (`scripts/run_e0_planning.py`) now retries (up to `max_tries=20`) until the
sampled pair has ≥`min_block_pos_diff=40px` of real block displacement (new CLI flag
`--min-block-pos-diff`). Also added `total_contacts` logging to `run_episode()`'s result dict (mirrors
the contact counter `run_e0.py`'s trajectory generator already had) so episode difficulty and
real-vs-trivial contact are auditable straight from `episodes.jsonl`, not just inferred from step counts.
**Any planning-success number measured before this fix (including this file's own numbers below) should
be treated as inflated and not compared against the numbers in this section.**

### Frozen baseline, filtered sampling, n=10 per regime

| Regime | Success | Notes |
|---|---|---|
| R0 (default) | 8/10 (80%) | 1 genuine zero-contact failure (ep2: `total_contacts=0`, final diff == init diff to the decimal — pusher never touched the block in 30 steps) |
| R2 (elasticity 0.9) | 8/10 (80%) | **Episode-for-episode identical outcome pattern to R0** (same 2 fail, same 8 succeed, very close final diffs) — elasticity barely perturbs frozen behavior for these tasks |
| R1 (friction 0.8) | 7/10 (70%) | **Real divergence from R0**: episode 8 (init=53.2px) succeeds under R0/R2 but fails under R1 (30 steps, 13 contacts, stalls at 47.5px) — friction is doing something R0/R2 aren't, on an identical task instance |

Seeds are literally paired across regimes (`sample_dataset_init_goal` is regime-independent), so this
is a genuine controlled comparison, not just three separate samples: e.g. seed 0 (init=91.5px) fails on
all three regimes with nearly identical final diffs (~81–97px); seed 1 (init=45.8px) succeeds on all
three. R1's episode 8 is the one real behavioral difference in this n=10 sample — good news for the
project: a regime a frozen model genuinely can't handle is exactly what a chart needs to exist for.

### Charts trained (T9, `atlas_out/e0_v2/`) — real-data replay + early stopping, both implemented this session

Two T9 requirements landed: (1) `scripts/run_e0.py::load_regime_trajectories(..., source="dataset")`
replays real `data/pusht_noise/train/` demo action sequences under the shifted regime instead of the
old scripted aimed-walk sampler (validated: replaying under R0 reproduces the original recording to
~1e-5; contact rate 17/17 for both R1 and R2); (2) `atlas/harness.py::run_e0_finetune()` gained
early stopping on a held-out validation split (`val_trajectories`/`eval_every`/`patience`) — keeps the
*best*-validation-loss chart snapshot, not the final step's (the original bug this fixes: `full` reached
train loss 0.0015 over 2000 steps on 30 transitions with zero validation signal).

**⚠️ Open methodological concern, raised 2026-08-25, not yet resolved:** real-data replay is
open-loop — the replayed action sequence never reacts to what the block actually does under the new
(R1/R2) physics, since it's just a fixed real R0-recorded sequence. A closed-loop policy (like the
original scripted sampler, which recomputes its action from the *live* simulated position every step)
is reactive to the actual regime-shifted dynamics; open-loop replay isn't, and later actions in a
replayed trajectory can become progressively mismatched to what's actually happening as the trajectory
diverges from the original recording. This may be part of why `lora4` (below) failed the way it did.
Proposed fix, not yet implemented: sample real *initial states* from the dataset (for state diversity)
but drive the rollout forward with the closed-loop scripted policy instead of blind replay.

| Chart | Regime | Train traj / len | Eval UMF | Early-stopped at | Planning SR (n=10, paired) | vs. baseline |
|---|---|---|---|---|---|---|
| `ln_act` | R1 | 20 × 25 | 0.2078 | step 150 (best @ step 25) | 7/10 (70%) | same rate, but **fixed** the real ep8 friction-failure; introduced a razor-thin new miss on ep3 (21.6px vs. 20px threshold) |
| `ln_act` | R2 | 20 × 25 | 0.1232 | step 475 (best @ step 350) | 8/10 (80%) | **identical outcome pattern to baseline, episode-for-episode** — no behavioral change, consistent with R2 barely perturbing the frozen model in the first place |
| `lora4` | R1 | 10 × 15† | 0.2420 | step 150 (best @ step 25) | **4/10 (40%)** | **substantially worse.** Distinctive failure mode: *more* contact than any other chart (up to 35) but pushes the block the *wrong way* on 2 episodes (ended up further from goal than the start) |

† `lora4` OOM'd on Modal's 22GB L4 at the same 20×25 trajectories `ln_act` used (its LoRA
parametrization has real extra activation overhead per forward pass); retried at 10×15. **This is a
real confound** — `lora4`'s worse result may reflect materially less/shorter training data rather than
(or in addition to) the architecture itself or the open-loop data concern above. Re-run at a matched
trajectory budget before treating 40% as a verdict on `lora4`.

**Not yet done:** `full` chart (third kind the pre-registered rule needs); re-running `lora4` at a fair
trajectory budget; resolving the open-loop-replay concern; extending past n=10 for statistical power;
the formal `scripts/run_e0_matrix.py` (T10) this was all a manual preview of.

---

## ⚠️ SUPERSEDED (2026-08-25, `E0_IMPLEMENTATION_PLAN.md` T1–T8): everything below this line is invalidated

**Root cause, found after this file's "part 3" update below:** `atlas/score.py::_open_loop_rollout`
— used by both UMF scoring and E0 chart fine-tuning — did not unroll the checkpoint the way it
actually unrolls. Four stacked defects: (1) time base wrong by 5× (the rollout looped over RAW action
count while the encoded action features only had `raw/frameskip` valid entries, so most steps re-fed
a stale action and compared a 5-frames-ahead prediction against a 1-frame-ahead target); (2) proprio
hard-zeroed, when this checkpoint's predictor structurally *requires* real proprio (concatenated into
the token channel width — `forward_pred(proprio=None)` is a channel-width `RuntimeError`, not a
graceful no-proprio path, confirmed empirically); (3) context window fixed at 1 frame instead of the
checkpoint's `ctxt_window=2`; (4) a correct implementation (`EncPredWM.unroll()`) already shipped in
the checkpoint's own wrapper and went unused. Full diagnosis: `E0_DIAGNOSIS_AND_PLAN.md`.

**This invalidates:** every UMF number below (the `ln_act > lora4 > full` ranking, the 18-episode
sweep, the baseline-vs-adapter comparisons, the CEM-cost-ranking diagnostic, the "richer retraining
made it worse" result), all 9 charts in `atlas_out/e0/*.pt`, and this file's own "part 3" CEM-config
correction below (that correction is real and still holds — the planner config bug it describes was
a *second*, independent bug — but the UMF/rollout numbers it was measured against are still invalid).

**Fixed, verified on real hardware (`E0_IMPLEMENTATION_PLAN.md` T1–T5):** `_open_loop_rollout`
rewritten on `EncPredWM.unroll()`; real proprio threaded through `score.umf()`,
`harness.run_e0_finetune()`/`run_e1_episode()`, `router.py`, and `expand.py`; E0's trajectory
generation chunk-aligned to the model time base with real captured proprio
(`load_regime_trajectories`); the informative-chunk motion gate (G6) wired into both E0 and E1's UMF
calls. Real 30-raw-step R1 trajectory, frozen model, identity chart: **UMF = 0.227** (< 1.0, vs. this
file's pre-fix online 24–52 / offline 0.67–1.67 range). A tiny live 5-step fine-tune shows real
gradient flow (loss 0.099 → 0.077). All available gates (G2, G3a, G3b, G5, G6) pass; G1/G4 still need
a live env to actually exercise (see `CLAUDE.md` §0.1 and `E0_IMPLEMENTATION_PLAN.md`'s Final Gate
section for the current honest status).

**Not yet done:** T9 (retrain all charts through the repaired pipeline) and T10 (the chart×regime
planning matrix — E0's real Success column) both require explicit approval per
`E0_IMPLEMENTATION_PLAN.md`'s 🛑 STOP gates before spending GPU budget on them. Until T9 lands, there
is no valid E0 chart to report a Success column for, and every number below should be read as
historical record of the pre-fix pipeline, not a current finding.

---

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
