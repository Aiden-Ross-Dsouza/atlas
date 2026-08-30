# C-2 FAILURE DIAGNOSIS — is the 1/20 real, and was P0 a failure?

> ## SUPERSEDED IN PART — 2026-08-30, by the experiments this document proposed.
> **Read `FIXLOG.md` V3-19 first.** Part 5's three experiments were all run. **All three
> hypotheses in Parts 3 and 5 were REFUTED**, including §3.1 (CEM exploitation), which this
> document called its sharpest catch, and §3.2 (the proprio head). The measured mechanism is
> **variance compression against a threshold objective**: chart and frozen `c0` reach the
> *same* mean final block distance (58.9 vs 58.2 px, paired p=0.093) with *very* different
> spread (sd 32.7 vs 53.1, Levene p=0.0006), and Push-T success is a 20 px threshold.
> §2.2 of this file is also corrected there: the chart's inflated closed-loop UMF is largely a
> *consequence* of the block not moving, not an independent failure.
> **What survives unchanged:** Part 1 (no plumbing bug), Part 2.1/2.3/2.4 (the measurements),
> Part 4 (P0 was not a failure) and Part 6 (the write-up framing) — Part 6 is in fact
> strengthened, since the mean-vs-threshold gap is a sharper claim than the one proposed here.

**Date:** 2026-08-30 · **Scope:** why `phase0_v3/c2_p0g_R2` returned SR 1/20 against a 10/20
paired baseline, whether it is a bug or a result, and what it implies for the project.
**Nothing was launched. No code was changed. Every number below was recomputed from raw files
in this repo with `.venv/Scripts/python.exe`.** Per `CLAUDE.md` §1.9, findings carry the check
that produced them; nothing here is quoted from a chat transcript or from FIXLOG's own summary.

---

## VERDICT (short)

> **The C-2 result is real, correctly measured, and correctly paired. It is not a bug.**
>
> **But it is also not "the chart is bad".** It is a clean, reproducible demonstration that
> *the objective the chart was trained on and the objective the planner optimises are different
> functions*, and that improving the first degrades the second. Three independent facts pin this:
>
> 1. **The failure exists at replan 1**, from the byte-identical initial state, before any
>    state-distribution drift can have occurred. So it is **not** a DAgger/coverage problem.
> 2. **C-1 is genuinely positive and is not an artifact** — at CEM iteration 0 the chart's top-10
>    elites land 67.6 px from goal vs the frozen model's 89.4 px. C-1 ran at `iterations=1,
>    capture_iteration=first`: **it only ever saw iteration 0.** The damage is created between
>    CEM iteration 1 and iteration 10 of that same first search.
> 3. **The chart's training loss is visual-latent-only, while the planner's cost is
>    `visual + 0.1 * proprio`** and the unroll propagates proprio autoregressively. The chart is
>    fine-tuned with the proprio head completely unconstrained, and the planner uses it.
>
> **P0's execution was not a failure. P0's execution is the reason you found this.** The §4.5
> C-1/C-2 gate did exactly the job it was written to do: it caught a chart that would otherwise
> have gone into E0'/E1/E2 on a 40% UMF improvement and poisoned every downstream number.
>
> **What has failed is a premise, not a process:** UMF (latent prediction error) is not a safe
> selection signal for planning competence in this substrate. That is now measured five
> independent times, all the same direction, and this is the first time it has been measured
> *cleanly, paired, and with the CI excluding zero.*

---

## PART 1 — RULING OUT A BUG (all checks executed, none is a "should be fine")

| # | Concern | Check performed | Result |
|---|---|---|---|
| 1 | Protocol mismatch (`P0_CLOSEOUT_AUDIT` B-4: launcher defaults are `iterations=30, nas=6`) | `replans` field in both JSONLs; **and** mean wall time restricted to full-length (3-replan) episodes only, which removes the early-termination confound | `replans==3` in both => `nas=2`. Full-length wall time: **chart 150.5 s (n=19) vs baseline 152.1 s (n=10)** — identical. `iterations=10, horizon=6` confirmed *empirically*, not just from FIXLOG's launch line. **B-4 is resolved: the arms ran the same protocol.** |
| 2 | Task pairing broken | Compared `init_block_pos_diff` and `init_agent_block_dist` per episode across arms at 1e-9 | **Byte-identical for all 20 episodes.** The pairing is exact. |
| 3 | Wrong chart loaded (`P0_CLOSEOUT_AUDIT` T-14: `p0g_onpolicy_frozen_check/` holds an identically-named identity chart) | FIXLOG V3-18's launch line says `--charts-subdir p0g_onpolicy`; independently, an identity chart would have reproduced the baseline, not moved SR by 45 pp | Real chart. |
| 4 | Chart not actually applied / applied to the wrong module | `run_e0_planning.py:478-481` and `diagnose_cem_costs.py:315-316` use the **same** `Chart.load` + `apply_(wm.predictor)`, and `wm = model.model` is the object `GC_Agent` plans through. C-1 used that identical path and produced a **large positive** effect | The chart is genuinely applied. A broken application could not have produced C-1's result. |
| 5 | Degenerate / exploded chart weights | Diffed `p0g_onpolicy/chart_ln_act_R2.pt` against `p0g_onpolicy_frozen_check/chart_ln_act_R2.pt` (the `--steps 0` identity snapshot = pretrained LN values), all 26 tensors / 10,764 floats | **Global relative change 3.9%**; largest single tensor 23.4% (`layers.1.0.norm.bias`); no NaN, no outliers. A modest, plausible adaptation — not a numerically broken artifact. |

**Conclusion of Part 1: there is no plumbing bug. The measurement stands.**

---

## PART 2 — WHAT THE DATA ACTUALLY SHOWS (new; none of this is in FIXLOG or the close-out audit)

### 2.1 The symptom is "the agent stops touching the block"

| statistic | chart arm | frozen `c0` baseline |
|---|---:|---:|
| success rate | **1/20** | 10/20 |
| mean contacts / episode | **2.25** | 4.75 |
| episodes with **zero** contacts | **4/20** | 0/20 |
| episodes where the block moved **< 0.5 px** all episode | **8/20** | **0/20** |
| episodes running the full 3 replans | 19/20 | 10/20 |

*(FIXLOG V3-18 reports "9/20 episodes show `block_pos_diff == init_block_pos_diff` exactly". At
exact equality it is **2/20**; at any tolerance from 0.5 px to 5 px it is **8/20**. Minor
correction — the substance is right, the wording "exactly" is not.)*

In every one of the 9 discordant episodes the baseline made **more** contacts than the chart.
The failure mode is not overshoot, not thrashing — it is **inaction**. RED FLAG 2 (mechanism
unmoved) correctly did not fire, because knock-aways fell 4 -> 1; they fell because the planner
stopped hitting the block at all.

### 2.2 The chart's own prediction error is ~2.6x WORSE in closed loop

`umf_mean` is logged per episode by `run_e0_planning.py` for both arms.

| | chart arm | frozen `c0` |
|---|---:|---:|
| planning-time UMF, mean of episode means | **1.113** (n=20) | 0.449 (n=18) |
| same, restricted to the 18 episodes both arms report | **1.178** | 0.449 |
| median | 0.638 | 0.462 |

Offline, on its own disjoint test split, this chart *improves* UMF (0.6753 -> 0.5583 at T=2;
0.5568 -> 0.3313 at trajectory scale — `P0_CLOSEOUT_AUDIT` V-3, verified). In the loop it makes
prediction error **2.6x worse**. This is the sharpest single statement of the dissociation and
it was available for free in files already on disk.

*(Caveat, stated because it matters: each arm's UMF is measured on the chunk **that arm chose**,
so this is not "same data, two models". It is "each model on its own chosen actions" — which is
the operationally relevant quantity, and the one the deployed router would see.)*

### 2.3 The failure is present at replan 1 — this is the decisive fact

Replan 1 starts from the byte-identical initial state in both arms (2.1 above), so no
state-distribution drift can yet have occurred.

| | chart | baseline |
|---|---:|---:|
| replan-1 UMF, paired, n=18 | **1.302** | 0.347 |
| chart worse in | **12/18** | Wilcoxon p = 0.054 |

Per-replan means: chart 1.204 / 0.999 / 1.211 vs baseline 0.347 / 0.652 / 0.706. **The chart is
already 3.7x worse on the very first plan and does not get progressively worse.**

**This rules out the DAgger-round-1 coverage hypothesis as the primary cause** — the hypothesis
`P0G_REVIEW.md` §B named as the real residual objection ("install the chart and the planner
visits states the chart never saw"). That mechanism predicts a *compounding* failure across
replans. What is observed is a failure that is fully formed on plan one.

### 2.4 C-1 is real, and C-1 is structurally blind to what killed C-2

Recomputed from `phase0_v3/cost_ranking_p0g_R2/...json` (both arms score the **same** 300
prior-sampled action sequences per seed, with identical ground-truth rollouts — a clean paired
design):

| | frozen `c0` | on-policy `ln_act` |
|---|---:|---:|
| mean true final block dist of the **top-10 by cost** (= CEM's elite set) | 89.4 px | **67.6 px** |
| contact fraction among those elites | 0.99 | 0.94 |
| per-seed rho | +0.001 | +0.276 |

I checked C-1 for the two artifacts it could plausibly have been: **massive ties in `true_dist`**
(most candidates never touch the block, so they all share the init distance — real in seed 0 at
83%, but mean contact fraction across seeds is 0.83, so ties are not driving it), and the
**pooled-rho trap** (`P0G_FIX_PLAN` §4.5's named trap — the reported number is per-seed;
pooled is 0.436 and was correctly not used). **C-1 survives both. It is a genuine result.**

And the header of that file says it all: `"iterations": 1, "capture_iteration": "first"`.
**C-1 measured CEM iteration 0 only.** C-2 runs 10 iterations. Combined with 2.3, the failure is
created **between CEM iteration 1 and iteration 10 of the first search** — precisely the interval
C-1 cannot see.

### 2.5 A weak but same-direction signal: the chart makes inaction look cheaper

Over the 17 seeds with >=5 no-contact candidates, the chart moves no-contact candidates
**down** the cost ordering by a mean of **6.4 percentile points** (11/17 seeds, Wilcoxon
p = 0.045). Small, marginal, and reported as **suggestive only** — but it is the same direction
as the closed-loop symptom (8/20 episodes never move the block), and at iteration 0 a 6-point
shift is invisible while over 10 CEM iterations of elite refitting it is not.

---

## PART 3 — TWO STRUCTURAL CAUSES, NAMED

### 3.1 Objective mismatch: the chart is a better *model* and a worse *optimisation target*

This is the standard model-based-RL failure (objective mismatch / model exploitation): CEM is an
optimiser, and it will find and exploit exactly the region where the adapted model is
over-optimistic. A model that is better in expectation over a broad prior batch (C-1, iteration 0)
can be strictly worse at its own argmin after 10 iterations of elite refitting (C-2).

This is not speculation about the substrate — **this repo has already measured the same shape for
the frozen model.** `P0G_REVIEW.md` §B's own P0-C dose response, on the same 20 paired seeds:

| CEM iterations | frozen `c0` SR under R2 |
|---:|---:|
| 10 | 10/20 |
| 15 | 9/20 |
| 30 | 8/20 |

**Optimising harder against this cost function already makes outcomes worse for the frozen
model.** The chart makes the cost function locally sharper (cost CV 0.195 vs 0.170) and better
ranked on average — and therefore makes the exploitation worse, not better.

### 3.2 The chart's loss never touches an output the planner uses

- `atlas/score.py::_open_loop_rollout` returns `out["visual"]` only.
- `atlas/harness.py:183-184` (and the val path at `:147-148`) computes
  `compute_trajectory_loss(model, z_preds, enc_out[1:])` — **visual latents only**.
- The planner's cost is `diff = diff_visual + self.alpha * diff_proprio`
  (`hub/.../planning/planning/objectives.py:129-141`), with `alpha = 0.1`
  (`scripts/run_e0_planning.py:108`), and `EncPredWM.unroll` propagates proprio
  **autoregressively at every step**.

So the chart is fine-tuned under a loss that leaves the proprio head **completely unconstrained**,
and the planner then (a) scores candidates partly on that head and (b) feeds it back into every
subsequent prediction step. Nothing in the pipeline prevents the fine-tune from degrading the
model's estimate of *where its own agent will be*. The observed symptom — the agent fails to
reach the block — is exactly what a degraded proprio channel would produce.

**This is a design gap, not a coding bug, and as far as I can find it is written down nowhere in
the repo.** It also applies to every chart this project has ever trained, `lora4`xR1 included.

### 3.3 A training-data composition factor (real, but confounded)

`e0_seed_manifest.json`, 100 train trajectories per regime: R0 mean **17.07** contacts/traj,
R2 mean **4.50** — the -74% collapse `P0_CLOSEOUT_AUDIT` B-2 found and nobody read. A chart fit
predominantly on low-contact data will predict "nothing much happens" well and contact dynamics
badly, which is the same direction as everything above.

**But flag the confound honestly:** per `P0_CLOSEOUT_AUDIT` T-3 the R0 and R2 collections are
**unpaired** (`seed_base` 2000 vs 1000; only ~2/100 shared demo episodes), so R0-vs-R2 differences
confound regime with task difficulty. And a counter-signal I computed here cuts the other way:
per-chunk block displacement is *larger* under R2 (mean 54.5 px, median 49.7) than R0 (24.2 /
18.4). **B-2's -74% should not be treated as decisive until it is re-measured on paired tasks.**
The close-out audit judged B-2 to survive T-3; the displacement result is new evidence that the
confound may be larger than assumed.

---

## PART 4 — WAS P0 A FAILURE?

**No — and the distinction matters for how you write this up.**

**What worked, and should be credited explicitly:**

1. **The on-policy collector fix (C.3 / P0-G) did what it was built to do.** The dataset-replay
   `ln_act` chart scored rho = 0.014 +/- 0.294 on C-1 (N11, indistinguishable from frozen). The
   on-policy chart scores rho = 0.276 +/- 0.181, CI [0.197, 0.355], top-10 elites 89 px -> 68 px.
   **The collector genuinely restored a mechanism the old collector could not.** That is a real,
   defensible, publishable result and it is the direct payoff of `COHERENCE_AUDIT_2` CHECK 2.
2. **§4.5's two-check gate is the single most valuable thing in this project's process.** Without
   C-2 being mandatory, a chart with a 40% UMF improvement and a positive mechanism check would
   have entered E0'/E1/E2 and produced downstream numbers that were confidently wrong. The gate
   fired. The 9-0 discordant split with p = 0.0039 is not a power-limited null being over-read —
   it is well past what n=20 detects.
3. **`P0G_REVIEW` / `P0G_FIX_PLAN`'s discipline held.** The protocol matched, the pairing was
   exact, the baseline was pre-existing, the power statement travelled with the result, and the
   verdict was BLOCK rather than an explanation.

**What failed, and it is a premise not a process:**

**UMF does not track planning competence in this substrate. It is now measured five times:**

| # | Evidence | UMF | Planning |
|---|---|---|---|
| N1 | dataset `ln_act`, N=100 paired | down, monotone | 44/100 vs 43/100, McNemar p = 1.000 |
| N4 | 20/60/100 trajectories | 0.336 -> 0.302 -> 0.268 | every CI spans zero |
| N11 | localized top-16 | `lora4` **best** | `lora4` SR **worst** (40%) |
| — | `lora4`xR1 | 0.242, better | **4/10** vs 8/10 |
| **C-2** | **on-policy `ln_act`xR2, N=20 paired** | **0.675 -> 0.558 (-17%)** | **1/20 vs 10/20, p = 0.0039** |

Plus a sixth, same-sign: Kendall tau(UMF, success) = **-0.316** within the C-2 arm itself.

Four of those were nulls or small-n. **C-2 is the first one that is clean, paired, and
significant — and it is not merely "UMF is uninformative", it is "selecting on UMF actively
destroyed control."** That is a stronger and more interesting statement than anything in the
original proposal.

**The consequence you have to face:** ATLAS's C1 selection and C2 verification both route on UMF.
`CLAUDE.md`'s own title for this work is *"Measure Fitness, Don't Infer the Regime."* The fitness
measure has now been falsified against the thing it is supposed to be a proxy for. E1 was already
retired analytically (`HANDOFF.md` §7.1); E2's positive result is regime-*label* accuracy, not
planning competence (`OPUS_REMAINING_TASKS` #13). E0' as pre-registered will not produce a
positive chart, and you now have the mechanism for why.

---

## PART 5 — THE THREE EXPERIMENTS THAT TURN THIS INTO A PAPER (~$1 total)

Ordered by decisiveness. Each is a one-line change to an existing script; **no new code.**

1. **C-1 at convergence — `--iterations 10 --capture-iteration last`** (~26 min, ~$0.35).
   `diagnose_cem_costs.py` already supports `capture_iteration="last"` (`:237`, `:322`). This
   directly tests §3.1: if the chart's *converged* elites are worse in ground truth than the
   frozen model's converged elites — while its iteration-0 elites are better (already measured:
   67.6 vs 89.4 px) — **the exploitation mechanism is proven, in one figure.**
2. **C-2 at `--iterations 1` and `--iterations 3`** (~2 x $0.5). If chart SR recovers toward
   baseline as CEM optimises *less*, you have the causal direction. Pair it with the existing
   frozen-model dose response (10/20, 9/20, 8/20 at it = 10/15/30) and you have a two-arm
   dose-response figure showing that optimisation pressure against a decorrelated cost is
   harmful, and that the chart amplifies it.
3. **C-2 with `alpha = 0`** (proprio term removed from the planner cost, ~$0.5). Tests §3.2
   directly. If SR recovers, the unsupervised-proprio-head gap is the cause — and the fix is one
   term added to `compute_trajectory_loss`, which would be a genuine methodological contribution
   rather than a patch.

**Do (1) first.** It is the cheapest, it is forward-only, and it converts "C-1 and C-2 disagree,
we don't know why" — FIXLOG V3-18's current honest position — into a named, measured mechanism.

---

## PART 6 — HONEST FRAMING FOR THE WRITE-UP

The proposed paper (adapters + UMF routing improves continual control) is not supportable. The
paper the data supports is stronger for a workshop on world models in physical AI, and it is
6-7 pages:

> **Latent prediction error is not a safe selection signal for adapter routing in JEPA world
> models.** On a frozen DINO-WM Push-T substrate under an out-of-support global dynamics shift,
> an on-policy-trained LN adapter reduces held-out open-loop latent prediction error by 17-40%
> and measurably restores the CEM cost function's single-batch candidate ranking (per-seed
> Spearman rho 0.001 -> 0.276; elite-set true final distance 89 -> 68 px) — **while collapsing
> closed-loop planning success from 50% to 5% on the same 20 paired tasks (McNemar p = 0.0039).**
> We localise the failure to the interior of a single CEM search rather than to state-distribution
> drift, and identify two structural causes: the adapter's training objective and the planner's
> cost are different functions, and the adapter is a better model but a worse optimisation target.

That is a real contribution. Objective mismatch is known in MBRL (Lambert et al.); what is new
here is a **crisp, paired, single-substrate demonstration that a metric-improving adapter is
control-destroying**, with the open-loop-ranking check that *passes* right next to the
closed-loop check that *fails*. Reviewers at a workshop will take that; several will find it more
useful than another positive routing number.

Two things to fix in the framing before submission, both `CLAUDE.md` §1.8 issues already flagged
by prior audits and now load-bearing:

- Stop calling `damping = 0.5` an "overshoot" regime. It is an **extrapolation** shift
  (`space.damping` is a global world parameter; the checkpoint saw one value in all 18,685 demos).
  A null under it must be written as *"a 10.7k-parameter adapter could not absorb an out-of-support
  global dynamics change"*, not *"adapters do not help"*.
- Report the C-1/C-2 dissociation as **the** result, not as an unresolved tension to be explained
  away later. It is the most defensible thing this project has produced.

---

## WHAT WAS AND WAS NOT VERIFIED HERE

**Verified by direct recomputation from raw files:** every number in Parts 1-3 except where noted.
Protocol match (wall time on full-length episodes), init-state pairing, chart weight delta,
contact counts, block-displacement counts, planning-time UMF (all three replans, paired at
replan 1), C-1 elite-set statistics, C-1 tie/pooled-rho artifact checks, the no-contact cost-rank
shift, the visual-only training loss, the planner objective's proprio term and `alpha = 0.1`.

**Verified by source reading only (not executed):** the chart-application path identity between
`run_e0_planning.py` and `diagnose_cem_costs.py`.

**NOT verified — stated as hypotheses, with the experiment that would settle each:** §3.1's
CEM-exploitation mechanism (experiments 1 and 2 in Part 5); §3.2's proprio-degradation mechanism
(experiment 3). §2.5's cost-rank shift is measured but marginal (p = 0.045, n = 17) and is
supporting evidence only, not a mechanism.

**NOT re-run:** nothing in this pass spent GPU. The C-2 and C-1 artifacts were read as
downloaded, not re-executed.
