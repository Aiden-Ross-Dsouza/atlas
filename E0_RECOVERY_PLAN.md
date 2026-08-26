# E0 Recovery Plan — executable task list

*Written 2026-08-25, after the T1–T8 rollout fix and the first trustworthy post-fix results.*
*Read `CLAUDE.md` first (it binds you), then `E0_RESULTS.md`'s 🟢 CURRENT section, then this.*
*This file supersedes `E0_IMPLEMENTATION_PLAN.md`'s T9/T10 ordering. T1–T8 and T12 are done.*

> **STATUS (2026-08-26): E0 IS COMPLETE, and E2 HAS ALSO ALREADY RUN — read `E2_RESULTS.md`.**
> Three independent rescue hypotheses for E0 tested; all three rejected. Regimes settled:
> **R1 = friction 2.0**, **R2 = damping 0.5** (§0.3–§0.5). The pre-registered P3 test failed,
> and every follow-up attempt to save it — bigger charts, better training data, a refined
> metric — failed too. **Full closing analysis: §0.8 below.** Separately, **E2 (routing/selector
> quality) has already run and found the project's one clean positive result: UMF-based routing
> correctly discriminates a dynamics shift from an appearance shift on R2 (Cell B accuracy 0.880
> vs. S-dyn's 0.324), while correctly committing nothing on appearance alone (Cell C). This is
> orthogonal to E0's failure — it validates the *selector*, not the *library*: the chart it
> routes to still doesn't improve planning success. Full detail, limitations, and the R1-vs-R2
> mechanism: `E2_RESULTS.md`, not duplicated here.** If you are picking this project up fresh,
> read `HANDOFF.md` first — it index-links every results doc and flags what's session-verified
> vs. relayed.
>
> **The five-arm R2 matrix, all paired on the same 20 episodes, `{"damping": 0.5}` throughout:**
>
> | Arm | Params | SR | vs baseline | Knock-aways | Zero-contact |
> |---|---|---|---|---|---|
> | baseline (frozen) | 0 | 45.0% | — | 5/20 | 0 |
> | **`ln_act`/dataset** | 10,764 | **50.0%** | +5.0pp (CI touches 0) | **2/20 (best)** | 0 |
> | `lora4`/dataset | 118,176 trainable | 40.0% | −5.0pp | 4/20 | 0 |
> | `ln_act`/closed_loop | 10,764 | 35.0% | −10.0pp | 6/20 (worst, tied) | 1 |
> | `full`/dataset | 20,800,884 | 20.0% | −25.0pp (CI excludes 0) | 6/20 (worst, tied) | 3 |
>
> **The two rescue hypotheses this matrix kills:**
> - **Capacity** (10.7k → 118k → 20.8M): monotonically *worse* above the smallest chart.
> - **Training signal** (open-loop replay → reactive-but-goal-blind hybrid → live on-policy CEM
>   collection): the *most* sophisticated collector (`closed_loop`) scored worst of the three
>   `ln_act` variants — worse knock-aways (6 vs. 2), worse mean progress (+10.8px vs. +37.3px),
>   and the only arm besides `full` to produce a zero-contact episode. On-policy data was
>   supposed to be the most promising remaining lever; it wasn't.
> - **The metric** (reported same day, not independently re-run in this file's own session):
>   localizing UMF to only the moving tokens was tried as a fix for the UMF-vs-success
>   inversion (§0.4/§0.6 already noted UMF and planning success disagree in this substrate) —
>   it made the inversion *worse*, not better. UMF still discriminates coarsely (it correctly
>   flagged `closed_loop` as the worst offline UMF, 0.4971, matching its near-worst SR) but not
>   finely (it still ranks `lora4` above `ln_act`, backwards from the real result).
>
> **Two corrections to numbers already in circulation, both now fixed:**
> 1. **`lora4`'s reported parameter count was wrong.** `e0_v6_R1/results.json` (and by
>    extension anything computed from `Chart.n_params()`'s naive sum) reports 10,292,640 —
>    that figure sums ALL stored tensors, including 12 frozen base-weight copies
>    (10,174,464 elements) kept only as restore references. The real trainable capacity is
>    **118,176** (12 × `lora_A` + 12 × `lora_B`), which is what every capacity comparison in
>    this file and in `E0_RESULTS.md` already used — do not let `results.json`'s raw number
>    into any external write-up.
> 2. **The local backup of `chart_full_R2.pt` was corrupted** (`PytorchStreamReader failed
>    reading file data/17` — almost certainly truncated by a batch-download loop hitting a
>    shell timeout mid-transfer). Re-pulled from the Modal volume and verified it now
>    deserializes correctly (`torch.load` succeeds, `dict` with `kind`/`params` keys as
>    expected). **If you have an older local clone of this repo's `atlas_out/`, re-pull this
>    file before trusting it.**
>
> **R1 charts are now trained and on the volume** (`atlas_out/e0_v6_R1/chart_{ln_act,lora4}_R1.pt`),
> unblocking E2 (needs a real `chart_R1` for library index 1, not a placeholder):
>
> | Chart | Train loss | Eval loss | Eval UMF |
> |---|---|---|---|
> | `ln_act` R1 | 0.0645 | 0.2479 | 0.2845 |
> | `lora4` R1 | 0.0438 | 0.2542 | 0.2876 |

---

## 0. Why this plan exists

The rollout bug is fixed and hardware-verified. The first trustworthy numbers then exposed a
different problem, which no amount of chart training can fix:

| Regime | Frozen baseline SR (n=10, paired, filtered seeds) |
|---|---|
| R0 (default) | 8/10 |
| R2 (elasticity 0.9) | 8/10 — episode-for-episode identical outcome pattern to R0 |
| R1 (friction 0.8) | 7/10 — one genuine behavioural difference (ep8) |

**The regime shift is worth ~10 percentage points.** That is the entire headroom any chart
could ever recover. Two structural consequences:

1. **E0 cannot demonstrate adapter benefit.** `ln_act`×R1 fixed the one real R1 failure
   (ep8) and lost a razor-thin new one (ep3: 21.6px against a 20px threshold), netting 7/10
   — which is what a 10pp-headroom benchmark produces whether or not the chart works.
2. **E1 is dead on arrival.** Its pre-registered gate needs `SR_oracle − SR_random ≥ 10pp`,
   and `atlas/stats.py::normalised_recovery` returns `None` below that **by design**
   (`stats.py:35`). If total regime-induced degradation *is* 10pp, the oracle-random spread
   is necessarily smaller. E1 produces no reportable number at any episode count.

So the critical path is **not** "train more charts." It is: establish that the benchmark can
show a difference at all, then run one decisive comparison.

### 0.1 The data-source question — settled, do not relitigate

A concern was raised that because `dino_wm_pusht` was trained on `data/pusht_noise`, drawing
E0's trajectories from that dataset makes everything too easy for the frozen model, and that
the original synthetic sampler existed to probe genuinely unseen situations.

The goal is right. The lever is not connected. Three reasons, all checkable in the code:

- **The scripted sampler is not out-of-distribution either.** `scripts/run_e0.py:254-268`:
  `ACTION_GAIN=0.25` was tuned specifically so scripted actions' std (`~[0.215, 0.204]`)
  matches the checkpoint's own `preprocessor.action_std` (`~[0.202, 0.200]`). Being
  off-distribution was diagnosed as a **bug** (`code-review.md` #6d) and deliberately
  removed. Synthetic init states come from `env.seed(); env.reset()` — PushTEnv's own random
  reset, the same generator behind the demos. "Synthetic" here means "not literally
  recorded," not "unseen distribution."
- **The novelty is supposed to come from the physics, not the data source.** A friction
  change is a *dynamics* change the predictor cannot have learned from any amount of R0 data,
  whichever actions condition it. Deliberately feeding an odd action distribution would
  instead produce a model that is bad because it is confused, which an adapter could "fix" by
  re-calibrating to a weird action prior — not the claim ATLAS wants to make.
- **`E0_IMPLEMENTATION_PLAN.md` T9 explicitly directs the dataset-replay path.** It replays
  real recorded *actions* under R1/R2 physics: R0 replay reproduces the recording to ~1e-5,
  so any larger divergence under R1/R2 *is* the regime shift, not familiar data.

**Therefore: do not revert `--data-source` to `scripted`, and do not return to the pre-fix
synthetic runs.** Two narrower concerns behind the worry *are* real and are handled below:
open-loop replay does not represent deployment (→ P2b), and train/eval episode overlap would
be genuine leakage (→ P2c).

### 0.3 P0 result (ran 2026-08-25) — the shift is visible, and R2 is mechanically dead

Frozen predictor, identity chart, no planner, n=8 **matched** real episodes per regime:

| Regime | eval UMF | mean numerator Σ‖ẑ−z‖² | mean displacement | mean n_contacts |
|---|---|---|---|---|
| R0 | 0.1578 | 170,897 | 345.04 | 38.5 |
| R1 (friction 0.8) | 0.2188 | **224,571 (+31%)** | 351.95 | 35.6 |
| R2 (elasticity 0.9) | 0.1543 | 166,971 (−2%) | 346.50 | 38.1 |

**Conclusions:**

1. **R1 is real but under-powered.** +31% numerator with displacement flat (+2%), so it is not
   a denominator artifact. But absolute UMF stays at 0.22 — well under 1.0 — so the frozen
   model still predicts R1 competently, which is exactly why the planning gap is only 10pp.
   The axis works; the magnitude does not.
2. **R2 is not a weak shift, it is no shift**, and the mechanism explains why it cannot be
   fixed by raising the value. `pusht_env.py:687` hardcodes `self.space.damping = 0` in
   `_setup()`, called on every `reset()`. In pymunk that is the fraction of velocity retained
   per second, so the block's velocity is annihilated every step: Push-T is fully quasi-static,
   the block moves only while actively pushed. Restitution governs *bounce-back velocity after
   impact*, which is destroyed before it can displace anything. Elasticity is physically unable
   to express itself in the observable state at any value.
3. **The same line identifies the right replacement axis.** `space.damping` itself changes the
   block's behaviour on **every** post-contact timestep rather than only during the handful of
   contact steps friction touches, and all 18,500 demos behind this checkpoint were generated
   at `damping=0`, so gliding dynamics are genuinely unseen. The pusher is `KINEMATIC` with its
   velocity set explicitly by PD control each sim step (`pusht_env.py:489-491`), so damping
   affects the **block only** — the action interface and the visuals are untouched.

**Therefore R2 is redefined: elasticity → damping.** This is a second regime-design correction
in the same class as the mass→friction one, and for the same reason (a mechanism that provably
cannot express itself in this environment). Write it up as a new section in
`REGIME_DESIGN_REVIEW.md` and as a §6.1b correction in `ATLAS_implementation_plan_v2.md`.

**Why this is required, not optional.** E1's library is `{c0, chart_R1, chart_R2}`. If R2 ≡ R0
then `chart_R2` ≈ `c0`, the routing problem is degenerate, and `SR_oracle − SR_random` collapses
mechanically no matter how good the router is. E1 needs two genuinely distinct regimes to have
a denominator at all. (An earlier draft of this plan said to keep R2 as a control — that was
wrong for this reason.)

**Method note for whoever re-runs P0:** this plan originally claimed a shared `seed_offset`
forces all three regimes to replay the same real episodes. That is **false** —
`run_e0.py:164` bakes a per-regime base (`R0:2000, R1:0, R2:1000`) in *before* adding
`seed_offset`, so equal offsets give different absolute seeds and different episodes.
Compensate the offset per regime so absolute seeds match, and assert the resulting
`(episode_idx, offset)` lists are identical across regimes before trusting any comparison.

### 0.4 P1 Step-3 screening result (ran 2026-08-25) — candidates selected

Same UMF diagnostic as §0.3 (frozen predictor, identity chart, no planner), same 8 matched
real episodes, extended across the candidate grid. `set_regime_config` and the
`_apply_physics` damping branch landed in `atlas/regimes.py`; the damping write-back was
verified empirically to survive `reset()` before being trusted.

| Candidate | eval UMF | numerator Σ‖ẑ−z‖² | numerator vs R0 | **UMF vs R0** | displacement | mean contacts |
|---|---|---|---|---|---|---|
| R0 | 0.1578 | 170,897 | — | 1.00× | 345.0 | 38.5 |
| friction 0.8 (current R1) | 0.2188 | 224,571 | +31.4% | 1.39× | 351.9 | 35.6 |
| **friction 2.0** | 0.2632 | 262,651 | +53.7% | **1.67×** | 344.9 | 34.5 |
| friction 5.0 | 0.2612 | 263,411 | +54.1% | 1.66× | 345.7 | 33.9 |
| damping 0.9 | 0.5011 | 653,858 | +282.6% | 3.18× | 427.8 | 14.6 |
| **damping 0.5** | 0.3890 | 477,834 | +179.6% | **2.47×** | 393.1 | 13.3 |
| damping 0.1 | 0.2111 | 253,846 | +48.5% | 1.34× | 370.9 | 26.5 |

**Rank by UMF, not the raw numerator.** Damping inflates observed displacement (345 → 371–428),
and the raw numerator does not normalize that out while UMF does — which is the whole reason
§0.3 insisted on reporting displacement alongside. Reading the numerator column alone inverts
one conclusion: **damping 0.1 is the *weakest* candidate tested (1.34×), below even the current
friction 0.8**, because most of its numerator gain is just the block travelling further.

**Selected:**

| Regime | Value | Rationale |
|---|---|---|
| **R1** | friction **2.0** | friction saturates here — 5.0 is indistinguishable (1.66× vs 1.67×), so there is nothing above 2.0 worth an episode |
| **R2** | damping **0.5** | 2.47×, the strongest candidate that is not the extreme; damping 0.9 (3.18×) held **in reserve**, used only if 0.5 lands above the target band |
| dropped | elasticity (any), friction 5.0, damping 0.1 | no shift / saturated / weakest-on-UMF respectively |

**On the contact collapse (38.5 → 13.3 at damping 0.5) — real, but confounded; do not
over-read it.** These trajectories are *open-loop dataset replay*: R0-recorded actions replayed
under shifted physics. Under damping the block glides away from where the demo expected it, so
the demo's later actions miss — fewer contacts follow near-mechanically from replaying a stale
action sequence, independently of whether the task is tractable. A closed-loop CEM planner
chases the block instead. So this is **not** by itself evidence of a floor effect.

**Disambiguate it in Step 4 rather than guessing:** `run_e0_planning.py:281` already logs
`total_contacts` per episode. If the planner's contact count under damping 0.5 stays near R0's
while open-loop replay's collapsed, the collapse was a collector artifact and damping is
tractable. If planner contacts collapse too, the floor-effect risk is real and the fallback is
damping 0.1 — accepting the weaker shift — or a shorter `min_block_pos_diff`. This same
comparison is the cheapest available evidence for P2b's open-loop-vs-closed-loop question, so
record it either way.

### 0.5 P1 Step-4 result (ran 2026-08-25) — **regimes are SETTLED. P1 is closed.**

Three frozen-baseline cells, 15 paired episodes each, `--kind baseline` (no chart anywhere).
Pairing verified: `init_block_pos_diff` identical across all 15 episode indices in all three
cells. ~155 s/episode on an L4, peak GPU 6.45 GB (confirming this could not have run on the
6 GB local card).

| Cell | SR | vs R0 | mean contacts |
|---|---|---|---|
| R0 control | 13/15 = **86.7%** | — | 12.5 |
| R1 friction 2.0 | 10/15 = **66.7%** | −20.0pp | 15.3 |
| R2 damping 0.5 | 7/15 = **46.7%** | −40.0pp | 6.9 |

#### The shape of the degradation matters more than its size

Excluding ep2 (dead in all three cells — see P2d):

| Cell | SR | successes: steps / contacts | failures | knock-aways | mean damage |
|---|---|---|---|---|---|
| R0 | 13/14 | 17.4 / 13.7 | 1 | 1 | +5.7px |
| friction 2.0 | 10/14 | 17.0 / 12.7 | 4 | 1 | +5.0px |
| **damping 0.5** | 7/14 | **8.4 / 3.0** | 7 | **4** | **+104px** |

("knock-away" = final `block_pos_diff` **greater** than `init_block_pos_diff`, i.e. the run left
the block further from the goal than it started.)

Under damping, successes get *faster* (8.4 steps / 3.0 contacts vs. R0's 17.4 / 13.7 — one good
shove carries the block home) while failures get *catastrophic*: 4 of 7 knock the block away,
averaging +104px. The individual cases:

```
ep12  init  43.0 -> 241.6  (+198.6)  ang 1.82  ctc 33    [R0: success at 10.8]
ep 8  init  53.2 -> 145.4  ( +92.2)  ang 0.65  ctc  4    [R0: success at 12.6]
ep 0  init  91.5 -> 177.6  ( +86.0)  ang 0.85  ctc 12    [R0: fail at 97.2]
ep 3  init 156.0 -> 195.0  ( +39.1)  ang 0.95  ctc 11    [R0: success at 16.5]
```

This is **systematic overshoot**. The predictor was trained entirely at `damping=0`, so it
believes the block stops dead on contact; CEM therefore commands pushes calibrated for a block
that does not glide, and under damping the block sails past the goal. Friction's failures look
like R0's instead — diffuse, small margins (ep3 misses at 22.7 against a 20px threshold), no
mechanism.

**That is the property that makes damping the right regime.** A specific, one-directional,
*learnable* model error is something an adapter can repair. Diffuse incompetence is not.

It also **resolves §0.4's open question**: the contact collapse is real with a live planner
(12.5 → 6.9, −45%), not purely a replay artifact — but it is milder than open-loop replay's
38.5 → 13.3 (−65%), so the planner does partially compensate. Both effects are real.

#### Decisions

| # | Decision |
|---|---|
| 1 | **R1 = friction 2.0** (−20pp). Not a failed candidate — a *milder* regime with a **different mechanism** (tangential contact vs. post-contact glide). E1 needs regimes that are **distinct** more than it needs two hard ones. Friction saturates at 2.0 (§0.4: 5.0 measured identical), so this is the axis's ceiling. |
| 2 | **R2 = damping 0.5** (−40pp), accepted **despite missing the pre-registered ≤40% cutoff by one episode** (46.7%; 6/15 would have been exactly 40%). Justification below. |
| 3 | **Damping 0.9 will NOT be run.** 40pp is already 4× E1's 10pp gate, and ep12's +198px / 33-contact thrashing at 0.5 indicates 0.9 risks the floor effect the band's *lower* bound exists to prevent. Spending GPU to make the benchmark harder in a direction we do not need is a bad trade. |
| 4 | **P3's decisive chart test moves from R1 to R2 (damping 0.5)** — most headroom, and its failure mode is mechanistically the one a chart should fix. |

Net: a clean three-way gradient across two distinct physical mechanisms —
**R0 86.7% / R1 66.7% / R2 46.7%**.

#### Justifying decision 2 — record this before any chart trains

The rule's *purpose* was headroom, and 40pp satisfies it four times over (E1 needs 10pp).
Critically, accepting a **weaker** shift than the rule demanded is the **conservative**
direction: it makes the benchmark *harder* for ATLAS to win on, not easier. Goalpost-moving is
making a benchmark easier after seeing results; this does the opposite, and it is decided on
frozen-baseline evidence only, with no chart trained, loaded, or consulted. Write it into
`REGIME_CALIBRATION.md` as a dated, documented deviation carrying this reasoning, **before**
P3 begins.

#### Pre-register this before P3 — stronger than "did SR go up"

> A working damping chart should **specifically reduce the knock-away failures** (the 4
> episodes where the block ends further from the goal than it started), by correcting the
> predictor's under-estimate of post-contact travel. Report knock-away count and mean damage
> alongside SR for every arm.

If SR improves but knock-aways do not, the chart is not fixing the mechanism the regime was
chosen to expose — report that, rather than banking a lucky SR bump.

#### Follow-up work this creates

- Set `REGIME_CONFIGS` in `atlas/regimes.py` to the calibrated values (`R1: {"friction": 2.0}`,
  `R2: {"damping": 0.5}`) so the **defaults are the calibrated regimes** and nothing downstream
  depends on remembering a `--regime-config` flag.
- Copy the three calibration JSONLs out of the session scratchpad into `atlas_out/e0_calib_*/`
  — they currently live only in a session-temporary directory.
- Write `REGIME_CALIBRATION.md` (the rule, the table, the pairing check, decisions 1–4, the
  deviation justification) and the `REGIME_DESIGN_REVIEW.md` / §6.1b elasticity→damping note.
- **Re-verify after changing the defaults**: re-run the P0-style UMF diagnostic and confirm
  R1/R2 reproduce §0.4's friction-2.0 (1.67×) and damping-0.5 (2.47×) rows.

### 0.6 P3 result (ran 2026-08-25) — E0 fails; one confound must be resolved before deciding

**Full result, tables, and episode-level analysis: `E0_RESULTS.md`'s 🔴 P3 DECISION POINT
section.** Summary and the decision it forces:

- **E0 fails the pre-registered criterion.** Baseline 45.0%, `ln_act`/dataset 50.0%
  (+5pp, CI [0, +15]), `ln_act`/hybrid 40.0%. Neither clears ≥15pp with a CI excluding 0.
- **But four signals say the dataset chart's effect is real, just small**: it is a strict
  superset of baseline (gained ep17, lost nothing; McNemar b=1 c=0), knock-aways fell 5/20 →
  2/20 (the **pre-registered** mechanism prediction, confirmed), mean final distance improved
  +21.3px across shared failures, and the hybrid chart — which shows none of this — behaves
  exactly like a chart that isn't helping.
- **The instrument is too blunt.** 7 of 20 episodes ask for a 120–300px push in 30 raw steps;
  baseline and chart both solve 1/7. `min_block_pos_diff=40` floors task difficulty but sets no
  ceiling, so a third of the sample carries almost no discriminative power. At N=20 paired
  binary episodes the minimum detectable effect is ~3 episodes — the 15pp bar sat near its own
  sample's noise floor.

#### ✅ RESOLVED — R0 confound check ran 2026-08-25

**R0 = 19/20 = 95.0%** on the identical filtered sampler. R0 − R2 = **+50.0pp, CI [+30, +70]**.
Headroom is real and larger than calibration estimated. **Full analysis: `E0_RESULTS.md`'s
🟢 R0 CONFOUND CHECK section.** Key consequences:

- **E0's real metric is normalised recovery = 1/10 = 10%.** The recoverable set (R0 solves,
  shifted baseline fails) is exactly 10 episodes; the dataset chart recovers one.
- **🔻 The "dilution" theory below is RETRACTED, and the difficulty-ceiling proposal with it.**
  R0 solves 7/7 of the 120–300px episodes — they are not intrinsically hard, they hold 6 of the
  10 recoverable episodes, and capping difficulty would have gutted the benchmark. **Do not add
  `--max-block-pos-diff`.** The goalpost-hazard section below is therefore moot; no sampler
  change is warranted or permitted.
- **The diagnosis:** damping selectively destroys long-range pushing (7/7 → 1/7) while short
  pushes survive (11/11 → 7/11). `ln_act` recovered 1 easy episode and **0 of the 6 hard ones**.

Two open explanations, and the runs that separate them, are tabulated in `E0_RESULTS.md` —
**capacity** (→ P4) versus **horizon compounding** (→ run P5's `nas=1` re-baseline early).
Both are now sanctioned; see the status banner at the top of this file.

<details><summary>Original (superseded) confound-check instructions and goalpost-hazard note</summary>

#### 🔴 Do this next, before anything else: the R0 confound check

**R0's SR on the P2d-filtered sampler was never measured.** P2d changed the sampler and only R2
was re-baselined, so the 55% failure rate cannot currently be attributed:

| If R0 on the filtered sampler is… | Then… |
|---|---|
| **≈ 85%** | the 40pp damping headroom is real, the benchmark is sound, and E0's negative result stands on its own terms — the chart is genuinely too weak. Decide P4 on that basis. |
| **≈ 50%** | the filtered sampler is intrinsically hard regardless of physics; damping is no longer what causes failure, and P3 measured task difficulty rather than regime adaptation. That is a **measurement bug**, not an E0 result, and it must be repaired before any conclusion stands. |

The 1/7 hard-episode rate for *both* arms makes the second case live. One run settles it:

```bash
modal run --detach modal/modal_e0_planning.py \
    --kind baseline --regime R0 --episodes 20 --out-subdir e0_v3_baseline_R0
```

~20 episodes, ~50 min, no charts. **Do not interpret P3 further or launch P4 until this
returns.**

#### If the second case holds — the goalpost hazard, stated plainly

The repair would be to bound episode difficulty (a `--max-block-pos-diff` ceiling alongside the
existing floor), concentrating episodes in the band where the regime shift actually decides the
outcome. **Changing the sampler after seeing chart results is exactly the shape of
goalpost-moving**, so it is only legitimate under the same discipline P1 followed:

1. justified by the **frozen-baseline** R0 diagnostic, not by any chart result;
2. the ceiling chosen from frozen-baseline behaviour alone, pre-registered in writing before
   any chart is re-evaluated;
3. **every** arm re-run on the new sampler, baseline included — no mixing of old and new numbers.

If those three cannot be satisfied, report E0's negative result as it stands and stop.

</details>

### 0.7 P4 capacity matrix (ran 2026-08-25) — `lora4` result: capacity is not the bottleneck

**Training.** `lora4` and `full` both trained on `dataset` (the P3 winner), R2, identical budget
to P3's `ln_act` (20 train trajs × 25 length, 8 val trajs). Both confirmed `regime_config ==
{"damping": 0.5}` via `e0_seed_manifest.json` before evaluation. **No OOM on either** — P2a's
fix holds under the real Modal config, not just the local 6GB smoke test.

| Kind | Params | Train loss | Eval UMF |
|---|---|---|---|
| `ln_act` (P3) | 10,764 | 0.131 | 0.336 |
| `lora4` | 118,176 trainable | **0.055** | 0.329 |
| `full` | 20,800,884 | **0.035 (lowest)** | **0.728 (worst — near the 1.0 no-better-than-static ceiling)** |

`full`'s combination — best train loss, worst eval loss — is a textbook overfit: 20.8M
parameters memorizing 20 training trajectories rather than learning the correction. This is
independent evidence against the capacity theory, before any planning episode was spent on it.

**`lora4` planning result, N=20, paired, identical protocol to P3:**

| Arm | SR | vs baseline | vs `ln_act` |
|---|---|---|---|
| baseline | 9/20 = 45.0% | — | — |
| `ln_act` | 10/20 = 50.0% | +5.0pp | — |
| **`lora4`** | **8/20 = 40.0%** | **−5.0pp, CI [−20, +10]** | **−10.0pp, CI [−30, +10]** |

Pairing verified: `init_block_pos_diff` identical across all 20 episode indices in all three
arms (baseline, `ln_act`, `lora4`).

**Knock-aways revert:** `lora4` 4/20 — almost back to baseline's 5/20, worse than `ln_act`'s
2/20. Whatever mechanism `ln_act` was correcting, `lora4` does less of it, not more.

**Episode-level pattern is the clearest signal.** `ln_act`'s one distinguishing win over
baseline was ep17 (strict superset, McNemar b=1 c=0 — see §0.6). `lora4` **loses that exact
episode**, plus ep1 and ep4 (all losses vs. baseline too), and gains only ep19 — solved by
neither baseline nor `ln_act`. A real capacity-driven improvement should look like `ln_act`'s
pattern scaled up (more wins, same losses). Instead it looks scrambled — consistent with noise,
not a bigger version of the same fix.

**Reading, combining all three signals:** `full`'s offline overfit, `lora4`'s worse-not-better
planning result, and `lora4`'s scrambled (not superset) episode pattern all point the same
direction. This matches the mechanistic prediction made before these results existed (see the
user's framing, recorded here for the record): the failure is *not* representational capacity —
10.7k LN params are already expressive enough for a low-dimensional, one-directional "stop
believing the block halts on contact" correction. The likely bottleneck is that the training
signal itself is incomplete: open-loop replay trajectories never show the model *recovering*
from an overshoot, so more parameters just fit the same incomplete correction more precisely
(and, at `full`'s scale on only 20 trajectories, overfit it outright).

**`full`'s planning evaluation:** launched 2026-08-25 as the formality §7.1's pre-registered
rule requires (defined relative to `full`'s gain) — not because a different outcome is
expected, given its offline UMF already sits near the no-better-than-static ceiling.

#### nas=1 diagnostic — attempted, cancelled, no result

A cheap-looking diagnostic to separate "capacity" from "horizon compounding" (frozen baseline
only, R2, `num_act_stepped=1` instead of 6 — 6 replans/episode instead of 1) was proposed and
launched. **The cost estimate was wrong.** `num_act_stepped=1` requires a full CEM search (300
samples × 30 iterations) at *every* replan, so 6 replans/episode costs ~6× a `nas=6` episode's
wall time, not the same. Real measured rate: **877.67s/episode** (vs. ~150s/episode for every
`nas=6` run in this project) → **~4h52m for 20 episodes**, not the "~50 min" originally
estimated. **Cancelled by the user after 1/20 episodes** (`ap-Mn8BxQ03n2VARhp2wIAMp2`, stopped
via `modal app stop --yes`) — no usable result exists from this attempt. The capacity-vs-horizon
question is being resolved by the P4 matrix itself instead (see above): `lora4`'s
worse-than-`ln_act` result already argues against capacity, independent of whether horizon
compounding is also a contributing factor. **If a future session wants this diagnostic, budget
~5 hours or cut episode count substantially (e.g. N=6-8) before launching — do not assume
`nas=6` timing transfers.**

#### Settled regardless: the open-loop question (P2b)

**Dataset replay beats the closed-loop hybrid collector.** The hybrid chart was worse on SR,
McNemar, and knock-aways. This contradicts the expectation recorded in §0.1/P2b that
closed-loop data would better represent deployment — recorded as a negative result, and it
means `--data-source dataset` stays the default.

### 0.8 `closed_loop` result (ran 2026-08-26) — **E0 is complete; three rescue hypotheses, three rejections**

**Training.** `ln_act` trained on `closed_loop`-collected data: the live CEM planner (frozen,
pristine predictor — no chart applied during collection) replans against the shifted R2
physics every model chunk (`num_act_stepped=1`), so the resulting 20 train + 8 val trajectories
contain the model's own real overshoot *and* its own attempted correction — the ingredient
`dataset` (blind replay) and `hybrid` (reactive but from an unrelated scripted heuristic, not
the model itself) both structurally lack. Collection budget deliberately cheaper than eval
(100×10 vs. the substrate's validated 300×30 — collection needs reactive trajectories, not
optimal ones; documented deviation, ~180 CEM searches total). Confirmed
`regime_config == {"damping": 0.5}` via `e0_seed_manifest.json` before evaluating. Smoke-tested
first (2 train + 1 val trajectories, ~8 min): contact rate 3/3, confirmed the collector was
genuinely interacting with the shifted physics before the real 20+8 run was launched.

**A methodology note for offline-metric comparisons across data sources:** `closed_loop`'s
offline eval UMF (0.4233) is **not directly comparable** to `ln_act`/dataset's 0.336 or
`lora4`'s 0.329 — those two share the same eval distribution (dataset-replay trajectories),
while `closed_loop`'s eval set is drawn from live planner rollouts, plausibly a harder
distribution (more varied, more extreme states) on its own terms. A higher UMF there is not by
itself evidence of a worse chart. The one internally-valid comparison is each arm's own
train→eval loss ratio (a generalisation-gap proxy): `closed_loop` sits at 4.2×, `ln_act`/dataset
at 2.6×, `lora4` at 5.9×, `full` at 17.0× — `closed_loop` is not an outlier on this axis. **The
only number that actually resolves anything is planning success, evaluated identically across
every arm** (see table in the status banner above).

**Planning result: 7/20 = 35.0%, worse than baseline (45.0%) and worse than `ln_act`/dataset
(50.0%).** `closed_loop − baseline` = −10.0pp, CI [−30%, +10%]; `closed_loop − ln_act/dataset` =
−15.0pp, CI [−35%, +5%] — both touch zero, but both point the wrong direction for the
hypothesis. Knock-aways: 6/20, tied with `full` for the worst of any arm — the exact mechanism
`closed_loop` was built to fix (recovering from overshoot) got *worse*, not better. Episode-level:
baseline beat `closed_loop` on 3 episodes it previously lost (1, 4, 8); `closed_loop` only won
back 1 (19) — a losing trade, the inverse of `ln_act`/dataset's clean superset pattern (§0.6).
All 20 episodes confirmed paired (`init_block_pos_diff` identical) against every other R2 arm.

**Why the most sophisticated data source did worst is itself informative, not just a null
result.** A speculative but consistent reading: `closed_loop`'s collector used a cheap CEM
budget (100×10) specifically so collection stayed affordable — but that means the "corrections"
in its training data are themselves drawn from a *weaker* planner than the one ultimately being
evaluated (300×30). The chart may have learned to imitate a noisier, less competent recovery
attempt than the eval-time planner would ever need. This is a plausible explanation, not a
verified one — nothing in this session tested it directly.

**Combined with capacity (§0.7) and the metric refinement mentioned in the status banner, this
closes out the three live explanations for E0's failure that this project could identify and
cheaply test.** None rescues it. `ln_act`/dataset remains the best-performing arm found, with a
real but small (+5pp, not statistically significant at N=20) effect — the finding stands as:
**no capacity level, no data-collection strategy, and no cheap metric refinement recovers
planning competence on this substrate under a damping shift.** Per `CLAUDE.md` §1.8, this is
reported as the result, not chased further with a fourth speculative fix.

**Local-artifact hygiene, discovered during this close-out (see status banner for the fix):**
`chart_full_R2.pt`'s local backup was corrupted (truncated by a batch `modal volume get` loop
hitting a shell timeout mid-transfer) and has been re-pulled and verified. `lora4`'s
`Chart.n_params()`-derived parameter count (10,292,640, still what `e0_v6_R1/results.json`
reports) double-counts 12 frozen base-weight restore copies; the real trainable capacity used
throughout every table in this file and `E0_RESULTS.md` is the correct **118,176**.

**What's next: E2, not E1.** E1 requires an oracle-random SR gap ≥10pp to report anything —
with every chart tested landing at or below baseline, that gap will not clear 10pp regardless of
routing algorithm (see the original day-one argument in §0, now confirmed rather than
overturned by everything since). **E2 tests something different and still open**: given a
library with a real (if weak) `chart_R1` and `chart_R2`, does a selector correctly route between
regimes at all — a question about the *selection mechanism*, not chart quality, and the one
claim in the proposal E0 has not already settled. R1 charts (`ln_act`, `lora4`, both trained,
on the volume at `atlas_out/e0_v6_R1/`) unblock this at an estimated ~$1 of Modal spend.

### 0.2 Execution rules

- Work in order. P0 → P1 → P2 → P3 → P4 → P5 → P6. **P1, P3, P4, P6 are 🛑 STOP tasks**:
  do the code, report, wait for explicit approval before spending GPU budget.
- **Never claim a gate or check passed without pasting the actual output.**
- Never `git commit` / `git push`, never `reset --hard` / `clean -fd` / `checkout --`.
- Do not change any `CLAUDE.md` §1.7 hyperparameter (τ=0.5, q=3, m=0.05, n_probe=20,
  K_max=10, chart lr=5e-4).
- All Modal runs use `--detach`.
- Report each task as: what changed, which files, what you ran, the output, what you did
  *not* run.

---

## P0 — Is the regime shift visible to the predictor at all?

**Cost: minutes. No planner, no CEM, no charts. Do this before anything else.**

This is the cheapest decisive measurement left in the project and it has never been run
post-fix. We have `identity chart, one R1 trajectory → UMF 0.227` from T1's verification but
**no R0 comparison**, so we do not currently know whether the frozen predictor can even see
the regime shift.

### What to write

A throwaway script in the scratchpad (**not** in `scripts/`). It reuses three things that
already exist and need no modification:

```python
# 1. Load the checkpoint the same way scripts/run_e0.py:441-460 does
#    (local hub, source="local"); keep BOTH wrapper and inner model.
# 2. For each regime in ("R0", "R1", "R2"):
#        trajs = load_regime_trajectories(wrapper, prep, regime,
#                    num_trajs=8, traj_len=50, device=device,
#                    seed_offset=0, source="dataset")
#    IMPORTANT: seed_offset must be IDENTICAL across regimes so every regime
#    replays the SAME real episodes/offsets and only the physics differs.
#    Assert it: the episode_idx/offset lists must match across all three.
# 3. chart = Chart(wm.predictor, "ln_act")   # identity by construction, CLAUDE.md §1.4
# 4. evaluate_e0_chart(wrapper, chart, trajs, motion_gate=None)
```

`evaluate_e0_chart` is `scripts/run_e0.py:322`. Import it, don't reimplement it.

### What to report — the ratio alone is not enough

UMF divides by observed displacement (`atlas/score.py:106`). Higher friction moves the block
*less*, which shrinks the denominator and can inflate UMF for reasons unrelated to prediction
quality. So report **all four** per regime:

| Column | How |
|---|---|
| eval UMF | `evaluate_e0_chart`'s second return value |
| raw numerator `Σ‖ẑ−z‖²` | inline the numerator from `score.py:103` in your script |
| mean observed displacement | `(z[-1]−z[0]).norm(p="fro")` per trajectory, averaged |
| mean `n_contacts` | already in each trajectory dict (`run_e0.py:316`) |

Paste the table.

### How to read it

- **R1 numerator materially worse than R0** (and not explained by displacement) → the shift
  is real and visible to the predictor. The 10pp planning gap is then a *task/measurement*
  problem (30 raw steps, one-shot planning, brief contacts), and P1 targets exactly that.
- **R1 ≈ R0 on both numerator and UMF** → the predictor cannot see the shift. No adapter can
  help, P1's sweep becomes the whole question, and the friction axis is probably finished.
  **Report and escalate before running P1.**
- **R2 ≈ R0** is expected — it already looks like a non-shift in the planning results.

---

## P1 ✅ CLOSED — Calibrate the regime against the frozen baseline

**All steps ran 2026-08-25. Outcome in §0.5: R1 = friction 2.0, R2 = damping 0.5, damping 0.9
not run. Do not re-run any part of P1.** The rest of this section is the method record.

**This was benchmark design, not tuning.** No chart was trained, loaded, or consulted anywhere
in this task — selection depended on frozen-baseline behaviour only, which is what keeps it
clear of `CLAUDE.md` §1.8.

### Step 1 — Pre-register, before running anything

Create `REGIME_CALIBRATION.md` at repo root containing this rule, verbatim, **before** the
first run:

> Choose the *smallest* physics perturbation for which the frozen baseline holds R0 at ≥ 70%
> SR and drops the shifted regime to ≤ 40% SR, on the same ≥ 15 paired filtered seeds. Ties
> broken toward the smaller perturbation. Selection depends only on frozen-baseline SR; chart
> results play no part in it.

### Step 2 — Make regime strength configurable

`atlas/regimes.py:55-59` currently hardcodes `REGIME_CONFIGS`. Add an override hook so each
candidate's config is logged into its own JSONL instead of being hand-edited between runs.
Smallest correct change:

```python
# atlas/regimes.py
def set_regime_config(name: str, cfg: dict) -> None:
    """Override a regime's physics parameters for calibration sweeps (P1).
    Values are logged per-run; this is a benchmark-design knob, not a tuned
    hyperparameter (see REGIME_CALIBRATION.md)."""
    REGIME_CONFIGS[name] = dict(cfg)
```

Then in `scripts/run_e0_planning.py::main`, add `--regime-config` taking a JSON string
(e.g. `'{"friction": 2.0}'`), call `set_regime_config(args.regime, json.loads(...))` before
constructing `PhysicsRegime`, and add the resolved config to the summary JSON written at
`run_e0_planning.py:450-455` **and** to every per-episode JSONL record
(`run_e0_planning.py:423`). Thread the same flag through
`modal/modal_e0_planning.py::run_e0_planning` and its `main` entrypoint.

### Step 3 — Screen candidates cheaply first

Run **P0's UMF diagnostic** on each candidate before spending planning episodes on it. A
candidate whose frozen numerator is indistinguishable from R0 will not produce a planning gap
either — drop it without paying for episodes.

**✅ Step 3 is DONE — ran 2026-08-25, results and selection in §0.4. Do not re-run it.**
Selected: **R1 = friction 2.0**, **R2 = damping 0.5** (damping 0.9 in reserve). Elasticity,
friction 5.0, and damping 0.1 are dropped. Skip to Step 4.

The grid below is kept for the record and for the implementation notes underneath it.

| Regime | Candidate configs | Why |
|---|---|---|
| **R1** (keep friction) | `{"friction": 2.0}`, `{"friction": 5.0}` | the one original axis P0 proved is visible (+31% numerator); just under-powered at 0.8 |
| **R2** (**replace** elasticity → damping) | `{"damping": 0.9}`, `{"damping": 0.5}`, `{"damping": 0.1}` | changes the block's behaviour on **every** post-contact timestep, not just contact steps; unseen by all 18,500 training demos, which were all generated at `damping=0` |

Do **not** sweep elasticity. P0 showed R2's numerator *below* R0's, and the mechanism explains
why (§0.3): with `space.damping = 0` any restitution velocity is annihilated before it
displaces anything. It is not a weak shift, it is no shift, and no value fixes that.

### Implementing the damping branch

Add to `PhysicsRegime._apply_physics` (`atlas/regimes.py:90-99`):

```python
if "damping" in self._cfg:
    self.env.space.damping = self._cfg["damping"]
```

Notes:
- `_apply_physics` runs *after* `reset()`, and `reset()` calls `_setup()` which recreates
  `self.space` and re-hardcodes `damping = 0` (`pusht_env.py:685-687`) — so setting it here is
  both necessary and correctly ordered. `gym.Wrapper` proxies attribute **reads**, so
  `self.env.space` resolves to the fresh space. **Verify empirically** (`CLAUDE.md` §9) by
  asserting `env.space.damping` post-reset before relying on it.
- Prefer this over `PushTEnv(damping=...)` (a real constructor param, `pusht_env.py:365`,
  applied at `:436-437`): the wrapper route keeps regime physics in one place and avoids
  touching all four `PushTEnv(...)` construction sites (`run_e0.py:190`,
  `run_e0_planning.py:356`, `run_e1.py:294`, and the harness).
- pymunk damping is velocity **retained per second**, so `0` = instant stop and `1` = no decay.
  With `sim_hz=100`, `control_hz=10`, one control step retains `damping**0.1` — e.g. 0.93 at
  `damping=0.5`. Even 0.9 is a large change from 0; expect these to bite hard, and expect to
  pick the *largest* (gentlest) value that satisfies the rule.

### Guard against overshooting

The pre-registered rule targets ≤ 40% SR for the shifted regime, but a regime that drives the
baseline to ~0% is as useless as one that changes nothing — there is no recoverable signal in
a floor effect either. Treat **~20–40%** as the target band, and if every candidate on an axis
lands below ~10%, step the perturbation down rather than accepting it.

### Step 4 — Planning episodes for survivors

Use `scripts/run_e0_planning.py --kind baseline` **unchanged otherwise** — its filtered
`sample_dataset_init_goal()` (`min_block_pos_diff=40`) and block-only `block_success()` are
the two fixes that made the current baseline numbers trustworthy at all.

Three cells, 15 episodes each, all frozen baseline (`--kind baseline`, no chart anywhere):

```bash
# R0 control -- same seeds; both candidates compare against this
modal run --detach modal/modal_e0_planning.py \
    --kind baseline --regime R0 --episodes 15 --out-subdir e0_calib_R0

# R1 candidate: friction 2.0
modal run --detach modal/modal_e0_planning.py \
    --kind baseline --regime R1 --episodes 15 \
    --regime-config '{"friction": 2.0}' --out-subdir e0_calib_fric2

# R2 candidate: damping 0.5
modal run --detach modal/modal_e0_planning.py \
    --kind baseline --regime R2 --episodes 15 \
    --regime-config '{"damping": 0.5}' --out-subdir e0_calib_damp05
```

Escalate to `'{"damping": 0.9}'` **only** if damping 0.5 lands above the target band. Per the
pre-registered rule, the smallest perturbation that satisfies the band wins — do not run 0.9
speculatively.

Seeds are `range(episodes)` and `sample_dataset_init_goal` is regime-independent, so cells
are paired by construction — but **verify** it by diffing `init_block_pos_diff` per episode
index across cells. Paste that check.

### Step 5 — Escalate if nothing bites

If no candidate reaches ≥ 30pp degradation, **stop and report.** That is a genuine, citable
finding about Push-T (kinematic pusher + brief contacts + a 30-step horizon leaves little for
physics to change), and it forces a scope decision the user must make: a different shift axis
(block geometry, action scale/actuation), or reporting E0 as a negative result per
`CLAUDE.md` §1.8. **Do not proceed to P3 on a weak regime hoping statistics will rescue it.**

---

## P2 — Two recipe fixes (code only, no approval needed)

Both are cheap, and both must land before any chart trains on the calibrated regime so that
P3's comparison is fair.

### P2a — Remove the OOM that confounded `lora4`

**The bug.** `atlas/harness.py:159-177` accumulates `total_loss` across **every** trajectory
and calls `backward()` once, so all trajectories' autograd graphs are alive simultaneously.
Memory scales O(N trajectories). That is why `lora4` OOM'd on the 22GB L4 at the same 20×25
budget `ln_act` used and had to be retrained at 10×15 — which is the confound behind its
4/10 planning result. Fixing this removes the confound rather than measuring around it.

**The fix.** Move `backward()` inside the trajectory loop, scaled by `1/len(trajectories)`:

```python
# atlas/harness.py, inside the `for step in pbar:` loop
optimizer.zero_grad()
total_loss = 0.0
for traj in trajectories:
    ...                                  # unchanged: z_ctxt, _open_loop_rollout, loss
    (loss / len(trajectories)).backward()   # was: total_loss = total_loss + loss
    total_loss += loss.item()               # detached scalar, for logging only
optimizer.step()
avg_loss = total_loss / len(trajectories)
```

Gradients are mathematically identical (sum of per-trajectory grads either way); peak memory
becomes O(1 trajectory).

**Verify, and paste both:**
1. `lora4` trains at `--num-train-trajs 20 --train-traj-len 25` without OOM.
2. `ln_act`'s loss curve at fixed settings/seed is unchanged to float tolerance versus the
   current code (run ~20 steps before and after, diff `loss_ln_act_*.json`).

### P2b — Add a closed-loop `hybrid` data source

**Why.** At deployment there are no recorded actions — a chart must be refined on what the
agent actually did under the new physics. Open-loop replay of an R0-recorded action sequence
never reacts to how the shifted physics diverge, and is a plausible contributor to `lora4`'s
failure mode (more contact than any other chart, but wrong-direction pushes). This is the one
part of the "synthetic data" instinct that is genuinely load-bearing.

**The change.** `scripts/run_e0.py::load_regime_trajectories` already contains both halves.
Add `source="hybrid"`:

- **Init**: the `dataset` branch's path (`run_e0.py:195-212`) — sample a real episode/offset,
  `base_env.reset_to_state = init_state7`. Gives real, diverse, on-distribution start states.
- **Actions**: the `scripted` branch's reactive loop (`run_e0.py:239-275`) — recompute each
  action from the **live** post-shift agent position, so it reacts to the changed physics.
  Keep `ACTION_GAIN=0.25` exactly as is; do not retune it.

Everything downstream (frame subsampling, action chunking, proprio capture via
`world_model.encode()`, the zero-contact retry) is shared and needs no change. Add `"hybrid"`
to `DataSource` (`run_e0.py:35`) and to `--data-source`'s `choices` (`run_e0.py:426`), and
thread it through `modal/modal_e0_planning.py::run_e0_train`'s `data_source` param (it is
already a passthrough string, so likely no change needed — verify).

**This is an additional source, not a replacement.** `dataset` stays the default until P3
rules between them.

**Verify:** generate one hybrid R1 trajectory and assert
`encoder_output.shape[0] == actions.shape[0] + 1`, `actions.shape[1] == 10`, `n_contacts > 0`,
and that the recorded `episode_idx`/`offset` are populated. Paste the shapes.

### P2c — Audit train/eval episode overlap

`run_e0.py` (`source="dataset"`) draws episodes from `data/pusht_noise/train/`, and
`run_e0_planning.py::sample_dataset_init_goal` samples init/goal pairs from the **same
split**. If a chart trains on the same real episodes its planning eval is drawn from, that is
leakage.

Check `atlas_out/e0_v2/e0_seed_manifest.json`'s `episode_idx` values against the episodes
that `sample_dataset_init_goal` resolves to for planning seeds `0..19` (reproduce them by
calling it with `np.random.RandomState(seed)` for each seed — it is deterministic). If they
overlap, add a held-out episode-index split (e.g. training draws from even indices, planning
from odd) and record it in the manifest.

**Log the outcome either way** — it needs to be an auditable fact, like the existing
E0-vs-E1 seed-disjointness check.

### P2d — Filter out unreachable episodes (kill ep2)

**The evidence.** In P1's Step-4 data, episode 2 is dead in **all three** cells: `total_contacts
= 0`, `block_pos_diff` exactly equal to `init_block_pos_diff` (62.2 → 62.2), byte-identical
across R0, friction 2.0 and damping 0.5. The planner never touches the block — the agent spawns
too far away to reach it within 30 raw steps.

**Why it matters.** It is a constant that can never discriminate between arms, and it burns
1/15 = 6.7% of every future run's budget. Worse, it silently drags every reported SR toward each
other, shrinking exactly the gaps P3 and P4 are trying to measure.

**The fix.** `scripts/run_e0_planning.py::sample_dataset_init_goal` already retries on
`min_block_pos_diff`. Add an agent-to-block reachability condition to the same retry loop:
reject pairs where the agent's start position is too far from the block to plausibly make
contact within the step budget.

Derive the threshold from the data rather than guessing — compute agent-block distance for
ep2 versus the episodes that did make contact, and set the cut where they separate. Expose it
as a CLI flag (`--max-agent-block-dist`) with the same documented-default treatment
`--min-block-pos-diff` already has, and log the resolved value into the summary JSON.

**Consequence — this changes the task, so baselines must be re-measured.** §0.5's
86.7 / 66.7 / 46.7 are **calibration-only** numbers, valid for the regime decision that was made
on them, but they are *not* the baselines P3 reports against. Fold the re-baseline into P3's
`c0` arm (which runs anyway) rather than paying for a separate run.

**Verify:** sample 50 pairs, print the agent-block distance distribution, and confirm no
ep2-style unreachable pair survives.

---

## P3 🛑 — The decisive E0 comparison

**Requires approval.** Run this on **calibrated R2 (damping 0.5)** only — one regime, one
adapter kind, the narrowest test that can kill or clear E0. Calibrated R1 (friction 2.0) is
needed for E1's library and enters at P4; do not fan out to it here.

**Why R2 and not R1** (changed per §0.5): R2 has 40pp of headroom versus R1's 20pp, and its
failure mode is systematic overshoot from under-predicted glide — a specific, one-directional
model error that is exactly what a dynamics chart should be able to correct. R1's failures are
diffuse and look like R0's, so a chart succeeding there would be much harder to attribute.

### Step 1 — Train two `ln_act` charts at identical budget

Only the data source differs. Everything else — trajectory count, length, val split,
optimizer schedule, early stopping — must match exactly.

```bash
modal run --detach modal/modal_e0_planning.py::run_e0_train \
    --kinds ln_act --regimes R2 --data-source dataset \
    --num-train-trajs 20 --train-traj-len 25 --num-val-trajs 8 \
    --out-subdir e0_v3_dataset

modal run --detach modal/modal_e0_planning.py::run_e0_train \
    --kinds ln_act --regimes R2 --data-source hybrid \
    --num-train-trajs 20 --train-traj-len 25 --num-val-trajs 8 \
    --out-subdir e0_v3_hybrid
```

Once `REGIME_CONFIGS` carries the calibrated values (§0.5 follow-up), `--regimes R2` resolves to
damping 0.5 with no extra flag. **Verify that it does** — a chart trained under elasticity 0.9
and evaluated under damping 0.5 would be a silent, invalidating mismatch, and nothing
downstream would catch it. Print the resolved config from both the training and evaluation
processes and confirm they match.

### Step 2 — Evaluate three arms on N=20 paired filtered seeds

```bash
for arm in "baseline e0_v3_dataset" "ln_act e0_v3_dataset" "ln_act e0_v3_hybrid"; do
  set -- $arm
  modal run --detach modal/modal_e0_planning.py \
      --kind $1 --regime R2 --episodes 20 \
      --charts-subdir $2 --out-subdir e0_v3_planning_$2_$1
done
```

The `baseline` arm doubles as P2d's re-baseline on the filtered sampler — §0.5's 46.7% was
measured before the reachability filter and must not be reused here.

Per episode, `episodes.jsonl` already logs: `success`, `block_pos_diff`, `block_angle_diff`,
`total_contacts`, `init_block_pos_diff`, `init_block_angle_diff`, `steps`, `replans`. Add the
chart's eval UMF alongside if not already present.

**Also report the knock-away metric** (§0.5's pre-registration): per arm, the count of episodes
with `block_pos_diff > init_block_pos_diff` and their mean damage. The baseline arm should show
roughly 4 of 7 failures knocking away, averaging ~+104px; a chart that is genuinely correcting
the glide mis-prediction should cut both.

### Step 3 — Apply the pre-registered decision rule

Write this into `REGIME_CALIBRATION.md` (or a sibling `E0_DECISION.md`) **before** looking at
Step 2's output:

- **Best `ln_act` beats `c0` by ≥ 15pp with a paired-bootstrap CI excluding 0**
  (`atlas/stats.py::paired_bootstrap`, paired on episode index — never an unpaired test)
  → **E0 passes, proceed to P4.** Report the knock-away metric alongside: an SR gain that
  leaves knock-aways untouched means the chart is not fixing the mechanism R2 was chosen to
  expose, and should be reported as such rather than as a clean pass.
- **Otherwise → stop.** E0 has failed under this protocol. Report it as a result per
  `CLAUDE.md` §1.8 and escalate to the user. Do **not** run E1, and do **not** start sweeping
  adapter kinds hoping one rescues it.

The dataset-vs-hybrid arm settles the open-loop question outright. **Keep whichever wins on
planning success, not on UMF** — `E0_RESULTS.md` already documents UMF and planning success
disagreeing in this substrate.

---

## P4 🛑 — Capacity matrix (only if P3 passed)

**Requires approval.** Now — and only now — `scripts/run_e0_matrix.py` (the old T10) is worth
writing.

Train `lora4` and `full` on the **winning data source** at **exactly** P3's trajectory count,
length, val split, optimizer schedule, and early-stopping settings. P2a is what makes that
budget parity achievable; without it `lora4` OOMs and the comparison is confounded again.

Matrix: `{c0, ln_act, lora4, full}` × **{calibrated R1, calibrated R2}**, N=20 paired seeds,
chart held **fixed for the whole episode**. R2 (damping) enters here because E1's library
needs a chart that is genuinely distinct from both `c0` and `chart_R1` — see §0.3. Reuse `run_e0_planning.py`'s episode loop, `block_success()`, and
`sample_dataset_init_goal()` — all three are correct and are not part of any known bug.

Report Δ with paired CI per `CLAUDE.md` §5 — never two bare means. Report real parameter
counts via `chart.n_params()` (`ln_act` 10,764 · `lora4` 118,176 trainable / 10,292,640 stored
· `full` 20,800,884), not tensor counts.

Apply plan §7.1's pre-registered rule to **planning success**, not offline UMF: the smallest
kind reaching ≥ 90% of `full`'s planning-success gain. If `full` shows no positive gain, the
rule cannot be applied as stated — report that honestly (this already happened once,
`E0_RESULTS.md`'s superseded section) rather than reinterpreting it into a winner.

The UMF-vs-success scatter with Kendall τ (proposal §8, supplementary figure S3) falls out of
these same logs for free — generate it.

---

## P5 — Fix E1's protocol (code only, no approval needed; do before any E1 run)

E1 currently measures a **different task** than E0, so its numbers would not be comparable
even if it ran. Three divergences, all confirmed in the code:

| # | Issue | Location | Fix |
|---|---|---|---|
| 1 | random init/goal instead of E0's filtered real dataset pairs | `atlas/harness.py:320` (`goal_utils.sample_random_init_goal_states`) | import and use `run_e0_planning.py::sample_dataset_init_goal` |
| 2 | `eval_state()` — includes agent position, exactly the bug E0 already fixed | `atlas/harness.py:378` | import and use `run_e0_planning.py::block_success` |
| 3 | episode runs up to **150 raw steps** (`n_replans_target = 30//6 = 5`, each replan executing `6×5 = 30` raw actions) vs. E0's 30 | `scripts/run_e1.py:238`, `:83-89` | set `CEM_NUM_ACT_STEPPED = 1`, keep `CEM_HORIZON = 6` |

Fix 3 also resolves the warmup-vs-routing deadlock that `run_e1.py:69-82` flags as an open
issue: at `num_act_stepped=1`, one replan executes `1×frameskip = 5` raw actions, so
`MAX_MPC_STEPS=30` gives **6 replans = 2 warmup + 4 routed** inside a 30-raw-step episode.
E1 becomes genuinely receding-horizon. (E0 at `nas=6` is one-shot within the episode — fine
for E0, but it structurally cannot exercise routing, which is why E1 needs its own value.)

Record this as a **second documented deviation** from the substrate config, in the same style
as `ATLAS_implementation_plan_v2.md` §7.0a: E0 keeps `nas=6` (substrate fidelity, one-shot),
E1 uses `nas=1` (routing requires multiple decision points). Note it in `run_e1.py`'s comment
block, replacing the open-issue text there.

**Then re-baseline the frozen model under the new protocol.** A 6-replan closed-loop episode
is not E0's 1-replan task; E0's baseline SR is **not** E1's baseline SR and the two must never
be compared directly. Run `--kind baseline` under E1's protocol at N=20 before any routing run.

**Verify:** `python scripts/smoke_gates.py --gate G5` (pairing), and paste one episode's
`raw_steps_per_replan` — it must read `[5, 5, 5, 5, 5, 5]`, not `[25, 25, ...]` or `[30]`.

---

## P6 🛑 — E1 smoke, then the real run

**Requires approval, and only if P3 passed and P5 landed.**

First measure throughput — `scripts/profile_episode.py` was implemented for real in T8 but its
output was never recorded anywhere, and the whole E1 budget depends on it:

```bash
python scripts/profile_episode.py --episodes 3
```

Then smoke (2–3 paired episodes, all five routers). It must verify **all** of:

- identical init/goal per episode index across every router (gate G5);
- exactly 2 warmup replans, routing from replan 3 onward (`selected_trace`);
- `umf_trace` finite and O(1) on informative chunks, `None` on gated ones;
- `oracle_id` selects the calibrated regime's own chart;
- `random` reproducible across two runs at the same seed;
- every router shares the same episode length and success rule.

Then the full run at whatever the measured throughput affords:

```bash
python scripts/run_e1.py --charts atlas_out/<winner> --kind <winner-kind> \
    --routers umf e1 sdyn random oracle_id --episodes <N> --seeds 3 \
    --out atlas_out/e1
```

`oracle_id` **and** `random` must both be in `--routers` or T1 is all `nan` — that is exactly
why `atlas_out/e1_verify/T1.md` currently reads `nan`.

Report the pre-registered criterion honestly, including "denominator below 10pp, not
reportable" if that is the outcome.

---

## Verification summary

Run the relevant gate for whatever you touched; `--all` at milestones. **Gate names are
uppercase** (`--gate G6`, not `g6`).

| After | Run | Must show |
|---|---|---|
| P0 | ✅ done — the scratch diagnostic | 4-column table per regime, matched episodes asserted |
| P1 | ✅ done — paired baseline cells | R0-vs-candidate SR, `init_block_pos_diff` diffed per episode index |
| `regimes.py` defaults | re-run the P0 diagnostic | R1/R2 reproduce §0.4's 1.67× / 2.47× rows |
| P2a | 20-step train, before/after | no OOM at 20×25 for `lora4`; `ln_act` curve unchanged |
| P2b | one hybrid trajectory | `enc.shape[0] == acts.shape[0]+1`, `acts.shape[1] == 10`, `n_contacts > 0` |
| P2b/c | `python scripts/smoke_gates.py --gate G6` | low-motion chunk returns `None` |
| P2d | 50 sampled pairs | agent-block distance distribution; no unreachable pair survives |
| P3 | `atlas/stats.py::paired_bootstrap` | Δ with CI, knock-away counts, plus per-episode JSONL |
| P5 | `python scripts/smoke_gates.py --gate G5` | `raw_steps_per_replan == [5]*6` |
| Before "done" | `python scripts/smoke_gates.py --all` | see caveat below |

**G1 caveat:** `gate_g1` still has the gymnasium-API bug (`env.reset(seed=...)` + 5-tuple
`step()` against a legacy-`gym` env — the same bug T12 #11 fixed in `gate_g4` but explicitly
did not fix here), **and** even once fixed its current test design never applies the chart to
the predictor or calls the model at all — it only re-runs the same env rollout twice and
checks determinism. So `--all` passing does **not** currently mean chart apply/restore is
verified. Either fix G1 to actually exercise `Chart.apply_`/`restore_`, or report it as
skipped. Do not let `--all` read as a clean pass while G1 is vacuous.

---

## Do not do

- Do not revert `--data-source` to `scripted`, or return to the pre-fix synthetic runs (§0.1).
- Do not sweep or train on **elasticity** — P0 proved the mechanism cannot express itself in
  this env (§0.3). R2 is redefined as a damping shift.
- **Do not run damping 0.9**, and do not re-open regime calibration. P1 is closed (§0.5); the
  regimes are R1 = friction 2.0 and R2 = damping 0.5.
- Do not reuse §0.5's 86.7 / 66.7 / 46.7 as P3/P4 baselines — they predate P2d's reachability
  filter. Re-measure via P3's `c0` arm.
- Do not change any `CLAUDE.md` §1.7 hyperparameter.
- **Do not run E1.** E0 is complete (§0.8) and failed under every variant tested (capacity,
  data source, metric refinement) — the oracle-random gap E1 needs (≥10pp) will not clear that
  bar with charts that don't beat baseline. This is settled, not pending.
- Do not use `atlas_out/e0/*.pt` (pre-rollout-fix, invalidated) for anything but wiring smoke
  tests.
- Do not use a local `atlas_out/e0_v4_full/chart_full_R2.pt` without re-pulling it from the
  Modal volume first — the original local copy was corrupted (§0.8); a re-pull has been
  verified to load correctly.
- Do not cite `Chart.n_params()`'s raw sum for `lora4` (10,292,640, still what
  `e0_v6_R1/results.json` reports) as its capacity — it double-counts 12 frozen base-weight
  restore copies. The real trainable capacity, used everywhere else in this file and
  `E0_RESULTS.md`, is **118,176** (§0.8).
- Do not `git commit` or `git push`.
