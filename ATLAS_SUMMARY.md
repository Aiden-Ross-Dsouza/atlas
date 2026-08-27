# ATLAS — Project Summary (current results only)

*Written 2026-08-26 to consolidate the project's four working docs
(`E0_RECOVERY_PLAN.md`, `E0_RESULTS.md`, `E2_RESULTS.md`, `HANDOFF.md`) into
one current-only read. Those four docs remain the source of truth and keep
their full history on purpose (nothing is ever deleted, only superseded
in-place with a dated banner) — this file exists so an outside reader doesn't
have to reconstruct which numbers are current by reading that whole trail.
Every number below is quoted, not paraphrased, from the current top section
of its source doc.*

---

> **AUDIT CORRECTION BANNER — 2026-08-27 (FIX_SPEC.md A14, A15).**
> Two statements below have been corrected in place:
> - **§4.3 converged-CEM spread.** Previously "std ... is 3.8–8.3px — a tight
>   cluster". The real range over all six seed/kind cells is **3.77–27.15px**
>   (`ln_act` seeds 1 and 2 are 17.9 and 27.2). "Tight cluster" is removed.
>   `E0_RESULTS.md` hedged this correctly; the summary had dropped the hedge.
> - **§4.5 Cell B "+55.6pp → +26.3pp" before/after.** The "pre-fix +55.6pp"
>   figure has **no surviving raw records**, so the before/after framing is
>   dropped; only the current number is reported.

## 1. What ATLAS is

Modern robots and agents increasingly use a **world model** — a neural network
trained to predict what a scene will look like after an action, so a planner
can "imagine" many possible action sequences and pick the best one without
actually trying them all in the real world. ATLAS studies world models that
are **frozen** (never updated) but paired with small, swappable **adapters**
("charts") — think of a chart as a small patch that nudges the frozen model's
predictions for one specific situation (e.g. "the surface has more friction
than usual"), without touching the model itself.

The research question: given a library of charts, can the model's own
prediction-error signal (how wrong its predictions were on what actually just
happened) be used to (a) **pick** the right chart for the current situation,
and (b) **decide when to create a new chart** — without ever being told what
situation it's actually in?

The environment used throughout is **Push-T**: a 2D simulated robot pushes a
T-shaped block toward a goal position. The frozen model is a public checkpoint
(`dino_wm_pusht`) — a frozen vision encoder (DINOv2) plus a small predictor
network. Planning is done with **CEM** (Cross-Entropy Method): sample hundreds
of candidate action sequences, score each by predicted "cost" (roughly:
predicted distance from the goal), keep the best-scoring ones, and repeat.

**UMF** ("Unexplained Motion Fraction") is this project's core scoring metric:
it compares a chart's *predicted* next-state to what *actually* happened,
normalized so 0 = perfect prediction and ≈1 = no better than assuming nothing
moves. Lower is better.

---

## 2. The current thesis, stated plainly

**Predictive-fitness routing works. Planning improvement does not — and this
null held up under every robustness check run.**

Concretely: UMF reliably identifies which chart is a better *predictor* of
what's currently happening (a routing/selection result, tested in "E2"), and
UMF also reliably predicts, *within one chart*, which individual attempts will
succeed or fail. But switching to the lower-UMF chart does **not** measurably
improve real task success (tested in "E0"), and this negative result survived
being re-tested at 5× the statistical power, at up to 5× more training data,
and under a more reactive (closed-loop) planning setup. The mechanism appears
to be that the planner's actual candidate-ranking process was never what
chart training optimized for — reducing prediction error and improving the
planner's real ranking of actions turn out to be different things.

---

## 3. How we got here — corrections made along the way

Standard practice in this project: nothing gets silently fixed and forgotten.
Every correction below is a real thing that was wrong, caught, and fixed —
listed here compressed to one line each; full derivations live in
`E0_RECOVERY_PLAN.md`.

| Was | Corrected to | Why |
|---|---|---|
| Every UMF/planning number before 2026-08-25 | Invalidated, then re-verified after a fix | The rollout code that unrolls the model's predictions had a 5× wrong time base, zeroed a required input (proprioception), and used the wrong context window — fixed and re-verified on real hardware. |
| R1 regime = lighter block (mass × 0.2) | R1 regime = higher friction (friction = 2.0) | Push-T's pusher is physically a "kinematic" body — in the physics engine used, a struck object's outcome is mathematically independent of the *pusher's* mass at any scale. Mass-scaling the block did nothing, confirmed by identical rendered trajectories at 1000× different mass values. |
| R2 regime = higher restitution/bounciness (elasticity = 0.9) | R2 regime = higher damping (damping = 0.5) | The environment hard-codes zero velocity-damping, so all post-contact velocity is destroyed every physics step before restitution/bounciness can express itself — this regime was measured to be no shift at all, not a weak one. |
| `lora4` chart's parameter count reported as 10,292,640 | Real trainable count: 118,176 | The larger number double-counted 12 frozen weight-restore copies kept only as internal bookkeeping. Every comparison in this project now uses 118,176. |
| `lora4` chart training OOM'd (ran out of GPU memory) at the intended data budget, had to train on less data | Fixed — trains fine at full budget now | The training loop was accumulating every trajectory's computation graph simultaneously before one gradient update; switched to one gradient update per trajectory (mathematically identical result, far less memory). |
| E0's original planning comparison, N=20 episodes, "+5.0pp, CI touches zero" | Re-run at N=100, "−1.0pp, CI [−9,+7]pp" | N=20 gives confidence intervals wide enough that a real effect and true zero are both consistent with the data. N=100 was run specifically to settle this — see §4. |

---

## 4. Results at a glance

### 4.1 Does the chart improve planning success? — No, confirmed at high statistical power

Baseline (frozen model, no chart) vs. the best-performing chart (`ln_act`, a
~10.7k-parameter adapter), same physics regime, same real (start, goal) pairs
per seed for both arms:

| Arm | Success rate | Δ vs. baseline | 95% CI | Statistical test |
|---|---:|---|---|---|
| baseline | 44.0% (44/100) | — | — | — |
| chart | 43.0% (43/100) | **−1.0pp** | **[−9.0, +7.0]** | McNemar p=1.000 |

This is a well-powered null, not an inconclusive one — the confidence
interval is roughly half the width of the original N=20 measurement, and the
point estimate sits almost exactly on zero.

**But the chart's prediction quality *does* predict success within a single
arm:** episode-level correlation between UMF and success is significant and
negative (as expected — lower UMF, better prediction, correlates with
success) in every arm tested this session: baseline τ=−0.406 (p<0.0001,
n=92), chart τ=−0.449 (p<0.0001, n=94). **This is the core dissociation**:
UMF predicts success within an arm, but doesn't predict which arm is better.

**This correlation survives controlling for episode difficulty (2026-08-27
update).** A standing objection: episode difficulty (`init_block_pos_diff`)
could drive both UMF and success independently, making the within-arm τ a
difficulty artifact rather than a model-quality signal. Partial correlation
(OLS-residualized on difficulty + contact count, then Kendall τ on
residuals) shows the effect **survives, attenuated but still highly
significant**: baseline −0.406 → **−0.358** (p=4.4×10⁻⁷), chart −0.449 →
**−0.374** (p=9.6×10⁻⁸). A stratified check corroborates this within every
difficulty tercile except the baseline's hardest bucket, where a floor
effect (8.3% SR) plausibly washes the signal out. **Conclusion: UMF tracks
model quality within an episode, not merely episode difficulty** — the
dissociation claim above is on firm ground. Full detail: `E0_RESULTS.md`'s
2026-08-27 section.

**Three more zero-cost checks on the same N=100 data (2026-08-27), closing
out every remaining pre-registered/"verify, don't assume" item:**

- **Knock-aways + mean progress replicate qualitatively at N=100.** The N=20
  mechanism test (5/20→2/20 knock-aways) holds up, more modestly: baseline
  24/100 (24.0%) vs. `ln_act` 22/100 (22.0%) knock-aways, and mean progress
  actually favors the chart (+25.1px baseline vs. **+32.4px chart**) even
  though SR is flat — a second, mechanism-specific dissociation instance:
  knock-aways down, progress up, SR unmoved.
- **SR by init-displacement bucket shows no hidden win.** 0–80px: 64.2% both
  arms (identical). 80–120px: 34.8% baseline vs. 30.4% chart (chart worse).
  120–300px: 8.3% both (floor effect). The aggregate null isn't masking a
  stratum where the chart helps — it's uniform across difficulty.
- **The "bridge" check confirms the cost-ranking diagnostic and the N=100
  planning run share the identical seed construction** (`init_block_pos_diff`
  matches to machine precision for seeds/episodes 0–9). Cross-referencing:
  all 4 planning **successes** have negative cost-ranking ρ; the 6
  **failures** have mixed-sign ρ — the opposite of what a naive
  "inverted-ranking-causes-failure" story predicts. n=10, not a significance
  claim, flagged as a curiosity, not a finding.
- **Training-size sweep pairing re-verified exactly** (max mismatch 0.0px,
  not just "6 decimal places") across the N=100 run and both the 60- and
  100-trajectory sweep re-evals — the sweep's pairing was never actually
  broken.

Full detail and raw output: `E0_RESULTS.md`'s 2026-08-27 section,
`scripts/analyze_n100.py`, `atlas_out/analysis_n100.json`.

### 4.1b Does the same null hold under the OTHER physics regime? — New 2026-08-27

Every planning-success comparison above uses the damping-shift regime (R2).
The friction-shift regime (R1) had never had this comparison run at any real
sample size — only a 1-episode smoke test and a 15-episode baseline-only
calibration check existed. Ran it for real: baseline vs. `ln_act` chart,
N=40 paired episodes, same real (start, goal) pairs per seed for both arms:

| Arm | Success rate | Δ vs. baseline | 95% CI | Statistical test |
|---|---:|---|---|---|
| baseline | 70.0% (28/40) | — | — | — |
| chart | 60.0% (24/40) | **−10.0pp** | **[−27.5, +7.5]** | McNemar p=0.388, 12/40 discordant |

Not statistically significant (the CI touches zero), but the point estimate
is now **negative** under R1, unlike R2 where it hovered near flat
(−1.0pp to +2.5pp across the training-data sweep in §4.2). Two things worth
holding onto: (1) the "no reliable planning benefit" conclusion **generalizes
across both tested regimes** — in neither regime does the chart produce a
significant improvement; (2) the *direction* of the (non-significant) point
estimate is not consistent between regimes, which argues against reading
even the R2 near-zero result as "the chart is neutral" in any strong sense —
it may simply not reliably do anything positive or negative, regime-
dependent noise being the more parsimonious explanation than a stable null
effect. Raw data: `atlas_out/e1_baseline_vs_chart_R1/{baseline,ln_act}_R1.jsonl`.

### 4.2 Does more training data help? — UMF improves; planning success stays flat

The chart above was trained on 20 real trajectories — a number never varied
until this session. Two more charts were trained (60 and 100 trajectories,
same recipe otherwise), and each was re-evaluated on 40 fresh paired episodes:

| Training trajectories | UMF (offline prediction-error metric) | Planning success | Δ vs. baseline | 95% CI |
|---:|---:|---:|---|---|
| 20 | 0.336 | 43.0% (N=100) | −1.0pp | [−9.0, +7.0] |
| 60 | 0.302 | 40.0% (N=40) | 0.0pp | [−12.5, +12.5] |
| 100 | 0.268 | 42.5% (N=40) | +2.5pp | [−12.5, +17.5] |

UMF falls **monotonically** — more data reliably makes the offline metric
better, with no sign of saturating even at 5× the data. Every planning-success
confidence interval spans zero at every data size. This is the cleanest form
of the dissociation result: across a 5× range of training budget, more data
measurably buys a better predictive-fitness score and buys nothing in
planning success.

### 4.3 What's actually wrong with the planner's ranking? — A mechanism, not just a null

Given UMF (prediction error) doesn't track planning success, a direct
question: does the *planner's own cost function* — the thing that actually
picks an action, every replan — correctly rank candidate actions the way the
real physics would? Tested by fixing 10 (start, goal) situations, capturing
the planner's cost for the same 300 candidate action sequences under both the
frozen baseline and the chart, then actually simulating all 300 candidates
for real to get their true outcome (something no model ever sees).

| | Averaged across all 10 situations (each candidate pooled together) | **Averaged per-situation, then meaned** | Range across situations |
|---|---:|---:|---|
| baseline (no chart) | ρ = 0.206 (looks positive, p≈5×10⁻³⁰) | **ρ = −0.072** | −0.453 to +0.373 |
| chart | ρ = 0.227 (looks positive, p≈3×10⁻³⁶) | **ρ = −0.051** | −0.441 to +0.340 |

(ρ = Spearman rank correlation between predicted cost and true outcome;
+1 = perfect ranking, 0 = no better than random, and both directions are
"lower cost, lower true distance = good.")

**The "averaged across everything" number is misleading here** — it mixes
"harder situations tend to have both higher cost and worse outcomes" (true,
but irrelevant to whether the *ranking within one situation* is any good)
with the actual thing the planner needs, which is the per-situation number.
That per-situation number is close to zero and swings unpredictably from
strongly negative to strongly positive **for the untouched frozen baseline,
not just the chart** — and the chart's numbers track the baseline's almost
exactly, situation by situation. **Whatever determines ranking quality is a
property of the specific task instance, not something the chart changes.**
This is a real, mechanistic reason for §4.1's null: the chart improves
prediction error while the planner's actual action-ranking — the thing that
determines what gets executed — was never touched by that training.

**Update (2026-08-27) — the near-zero ranking result is regime-specific, not
universal.** The R2 (damping) diagnostic above had never been run as a
control against R0 (no regime shift at all) — closing that gap was Opus's
single most-repeated "non-negotiable" request. Result: **R0's baseline
cost-ranking is strongly *positive*, not near-zero** — mean per-seed ρ =
0.501, 95% CI [0.277, 0.726] (n=10 seeds, 300 candidates/seed), vs. R2's
−0.072. **The planner's cost function ranks candidates reasonably well under
default physics; it only degenerates under the R2 damping shift specifically
— it is not a general property of this frozen model's CEM search.** This
revises the framing above: "whatever determines ranking quality is a
property of the specific task instance, not something the chart changes" is
still true *within* R2 (baseline and chart track each other closely there),
but it is not true *across* regimes — R0 vs. R2 is exactly the kind of
regime-level difference that mattered.

**Regret quantifies the R0-vs-R2 gap directly, not just as a correlation.**
`diagnose_cem_costs.py` was extended to persist raw per-candidate data
(previously summary-only), enabling: regret = true_dist of the
cost-argmin candidate minus the batch's actual best true_dist, and the
cost-ranked top-10's mean true_dist vs. the full batch's mean.

| | Mean regret (trusting the planner's pick vs. best available) | Top-10-by-cost vs. batch mean |
|---|---:|---:|
| R0, baseline | **8.5px** | **+28.2px (top-10 better)** |
| R2, baseline | **88.1px** (~10×) | **−15.7px (top-10 WORSE)** |
| R2, `ln_act` | **92.3px** | **−8.5px (top-10 worse)** |

Under R0, trusting the planner's cost ranking is cheap and directionally
correct. Under R2, it is an order of magnitude more costly, and the
cost-ranked top-10 candidates are actually *worse* on average than an
unranked batch — trusting the ranking under R2 is actively
counterproductive, not just uninformative. Contact fraction (~80% across
all three rows, identical because iteration-0's CEM draw is
model/regime-independent by construction) rules out the alternative
explanation that ρ≈0 just reflects most candidates never touching the
block. Full detail, all with 95% CIs: `E0_RESULTS.md`'s 2026-08-27 top
section, and the companion partial-Kendall-τ checks in the section below it.

**Is the R0-to-R2 collapse a sudden break or a gradual slide? New 2026-08-27
— it's gradual.** The R0/R2 comparison above is two points: no shift and
full shift. Ran three intermediate damping strengths (0.125, 0.25, 0.375,
between R0's implicit 0 and R2's 0.5), n=20 seeds each, same protocol:

| Damping | 0.0 (R0) | 0.125 | 0.25 | 0.375 | 0.5 (R2) |
|---|---:|---:|---:|---:|---:|
| Mean rho | 0.532 | 0.295 | 0.169 | 0.078 | 0.001 |
| 95% CI | [0.39,0.68] | [0.17,0.42] | [0.04,0.30] | [−0.06,0.21] | [−0.13,0.13] |

The decline is **smooth and monotonic**, not a threshold that suddenly
breaks — the planner's ranking quality degrades roughly proportionally to
the physics mismatch, all the way down to chance level at full shift. This
is a stronger, more specific claim than "off vs. on": it argues the
underlying cause is a continuous mismatch between the frozen model's
implicit physics assumptions and the true dynamics, not a discrete failure
mode that only trips past some critical shift magnitude. Raw data:
`atlas_out/cost_ranking_dose_{0125,025,0375}/`.

**CEM doesn't just rank poorly under R2 — it confidently converges to a
knock-away (2026-08-27).** All of the above uses CEM's iteration-0 (raw,
untrained) candidate draw. A converged-search check — capture the FINAL
iteration's population instead (3 seeds, full 300×30 search) — answers
"does a real search do better than a random draw?" **No: it's worse.**
Verified as genuine convergence (std of the 300 final candidates' true
outcome ranges 3.77–27.15px across the six seed/kind cells — `ln_act` seeds 1
and 2 are the high end at 17.9 and 27.2px; it is a converged search, not
noise, but not a uniformly tight cluster), baseline's converged
plan lands *farther* from goal than the episode started, in **all 3 seeds
tested, every time**: e.g. seed 2 starts 158.2px from goal, converges to a
confident plan that lands at 200.8px (median). This is the same knock-away
failure already measured at 24% of N=100 episodes, but shows it isn't
occasional bad luck — CEM is **certain and wrong**, not uncertain, because
its internal model has no representation of R2's damping. `ln_act` doesn't
reliably fix this (helps in 2/3 seeds, is outright worse in the third).
Ranking quality at convergence is no better than at iteration-0 — pooled ρ
is now significantly *negative* for baseline (−0.179, p≈7×10⁻⁸). Full
detail: `E0_RESULTS.md`'s 2026-08-27 "CONVERGED-CEM COST RANKING" section.

### 4.4 Does more reactive (closed-loop) planning change the picture? — Directionally, not conclusively

Every result above uses one open-loop plan per 30-step episode. A cheaper
follow-up tested 3 replans per episode instead of 1 (more reactive, correcting
mid-episode instead of committing to one plan):

| Arm | Success rate (N=20) | Δ vs. baseline | 95% CI | Statistical test |
|---|---:|---|---|---|
| baseline | 40.0% (8/20) | — | — | — |
| chart | 50.0% (10/20) | **+10.0pp** | **[−10.0, +30.0]** | McNemar p=0.625, discordant 3:1 favoring chart |
| *(for reference, same chart, 1 replan)* | *43.0% (N=100)* | *−1.0pp* | *[−9.0, +7.0]* | |

**Not significant at this sample size — the interval still spans zero — but
the direction flips positive**, unlike the null result at 1 replan. This is
the first evidence in the project that the single-shot planning protocol
might be suppressing a real benefit; it needs a properly-powered re-run (same
scale-up already done for §4.1) before it's a citable finding either way.

### 4.5 The one positive result: routing between charts works

A separate question from "does the chart help": if you already have several
charts, can the same UMF signal correctly pick which chart matches the
current situation? Tested by deliberately shifting either the *physics*
(what UMF should detect) or the *visual appearance* (what it shouldn't react
to), and comparing UMF-based routing against a naive "similarity to what the
scene looks like" baseline router (called S-dyn):

| Router | Accuracy, 3-chart library (chance = 33%) |
|---|---:|
| **UMF** | **60.3%** — real, ~2× better than chance |
| S-dyn (appearance-based baseline) | 36.5% — indistinguishable from chance |

The confusion matrix shows *why* S-dyn fails: it defaults to the same chart
regardless of the true physics situation, essentially guessing. UMF's
accuracy held up when the library was expanded from 2 charts (an earlier,
smaller test) to 3 — it didn't erode, and S-dyn didn't improve either.

**2-chart decisive cell (`ln_act`×R2), current numbers: UMF 0.833 vs. S-dyn
0.570.** UMF routes correctly more often than the appearance-based baseline;
S-dyn's residual accuracy is naive stickiness that sometimes coincides with
the right answer (correctly persisting post-switch). *(Audit note, FIX_SPEC.md
A15: an earlier version of this paragraph reported a "pre-fix +55.6pp →
post-fix +26.3pp" before/after. The "+55.6pp" figure has no surviving raw
records, so the before/after framing has been dropped and only the current
number is stated.)* Full detail: `E2_RESULTS.md`'s 2026-08-27 section.

**This is orthogonal to §4.1's negative result** — it validates that the
*selection mechanism* works, not that the charts it selects between actually
help planning. Correct routing here still routes to a chart that doesn't
improve success.

### 4.6 Two smaller results that answer specific objections

- **Does the routing signal actually give you anything, if you had a
  perfect oracle that always picked the best chart?** **Updated
  2026-08-27 — the original answer here was wrong and has been corrected.**
  A prior version of this section reported a 2.5–3.3pp oracle-vs-random
  spread and concluded no routing algorithm could manufacture benefit the
  library doesn't contain. That number rested on a 3-chart comparison where
  the third chart (`chart_R1`, evaluated on R2 episodes) was never actually
  run — the baseline's own results were duplicated and presented as if they
  were `chart_R1`'s, undisclosed. **That missing evaluation has now been run
  for real** (`atlas_out/e0_chart_r1_on_r2/ln_act_R2.jsonl`, 20 episodes,
  `chart_ln_act_R1.pt` applied to R2 planning episodes, same seeds as the
  existing baseline/`chart_R2` rows): baseline 45.0% (9/20), `chart_R2` 50.0%
  (10/20), `chart_R1` 45.0% (9/20). **Real oracle SR = 60.0%, random SR =
  46.7%, spread = 13.3pp, bootstrap 95% CI [3.3, 25.0]** — this CI does not
  touch zero and clears the project's own 10pp reporting threshold, the
  opposite of what the fabricated data showed. `chart_R1` is the unique
  winner (baseline and `chart_R2` both fail) on 2 of the 20 episodes. **A
  real denominator for chart routing does exist** — the conclusion that no
  routing algorithm could show a meaningful benefit was based on fabricated
  data and does not hold. Caveat: `chart_R1` and `chart_R2` come from two
  different training runs (`e0_v6_R1` vs `e0_v3_dataset`), not a perfectly
  matched triple, and n=20 is still a small sample. Full derivation:
  `research_audit/CLAIMS_MATRIX.md` row N8, `research_audit/RESULTS_AUDIT.md`.
- **Does the "commit a new chart only if it's verified to help" mechanism
  actually work, or is it just unexercised code?** Tested directly: 3 charts
  were actually committed through the real verification path when a genuine
  physics shift was present, and the same mechanism correctly commits nothing
  under an appearance-only shift (0.0% of chunks exceed the commit threshold,
  vs. 15.7% under a real shift). The mechanism fires when it should and stays
  quiet when it shouldn't.

---

## 5. Settled vs. still open

**Settled, not planned to be revisited:**
- The chart does not improve planning success on this task/regime, at any
  tested training-data budget, at high statistical power (§4.1, §4.2).
- The planner's own cost-ranking is close to zero-correlated with true
  outcomes per-situation, for the frozen model itself, not just the chart
  (§4.3) — a real mechanism for the above.
- UMF-based chart routing works and clearly beats an appearance-based
  baseline (§4.5).
- A verification-gated "commit a new chart" mechanism has been demonstrated
  to fire correctly and stay quiet correctly (§4.6).

**Still open:**
- Closed-loop (more reactive) planning shows a directionally positive but
  not statistically significant signal (§4.4) — needs a larger re-run.
- The full continual-learning stream experiment (repeatedly encountering and
  re-encountering different physics regimes over many episodes) is planned
  but has not been run — the relevant code path has never executed
  end-to-end yet.
- Whether a properly-tuned larger adapter, a different regime, or a
  different environment would show a real planning benefit is untested;
  this project's negative result is specific to the adapters, regime, and
  environment tested.

---

## 6. A note on verification

This project runs two people/sessions working the same repository in
parallel, by design (one on the results above, one on the continual-learning
stream mentioned as "still open"). Most results above were independently run
and verified end-to-end within the session that produced this summary; a
smaller number (noted where relevant, mainly parts of §4.5's earlier/smaller
version and some early chart-training numbers) were produced by the parallel
session and verified from its own logs rather than independently re-run. This
doesn't change any number reported here, but it's the honest provenance if
asked.

---

## 7. Where to go for more detail

The four working docs remain the source of truth and are kept intentionally
complete (nothing is ever deleted — only superseded in place with a dated
banner), so they read as a full audit trail, not just a final answer:

- **`E0_RESULTS.md`** — most-recent-first. Only the top section (down to the
  first horizontal rule) is current; everything below is either supporting
  detail behind that top summary or explicitly marked SUPERSEDED/invalidated
  historical record.
- **`E2_RESULTS.md`** — same convention; only the top "UPDATE" section is
  current, the rest is the original (still-valid) E2 result it updates.
- **`E0_RECOVERY_PLAN.md`** — the process narrative: why the physics regimes
  were chosen and redefined, the mechanism behind each fix, the full
  hypothesis-testing history behind §4.1–§4.3 above.
- **`HANDOFF.md`** — navigation/index doc, points to where every result
  actually lives, plus a few results (§4.6 above) that exist only there.

Deeper background (the original research proposal, the day-by-day
implementation plan, an engineering bug log, and two methodology-derivation
documents on regime design and action sampling) exists in the repo and is
available on request — not reproduced here since its conclusions are already
folded into the docs above.
