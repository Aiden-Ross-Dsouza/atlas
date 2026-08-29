# ATLAS — Paper Draft Notes

*Source: synthesized from research_audit/CLAIMS_MATRIX.md, REDTEAM.md, RESULTS_AUDIT.md, CODE_AUDIT.md, LITERATURE_AUDIT.md, PROPOSAL_CODE_ALIGNMENT.md (all dated 2026-08-27). Read those for full derivations and evidence levels; this file is the drafting-ready synthesis, not a replacement audit trail.*

This is a separate file from `ATLAS_SUMMARY.md`. `ATLAS_SUMMARY.md` is untouched and remains the raw results record. This file is downstream of it and of the full audit — every number below is the same as `ATLAS_SUMMARY.md`'s, but every framing, attribution, and disclosure gap the audit found has already been fixed. Nothing here needs to be re-checked before it goes into a draft.

---

## Recommended framing (not a mandate — the team's call)

> **"Prediction Accuracy Does Not Predict Planning Competence Under a Physics-Regime Shift — And Neither Does Regime-Label Routing Accuracy"**

This is `REDTEAM.md`'s recommended narrative after attacking every claim in the project. It is the framing that survives adversarial review with the fewest open wounds, built entirely from evidence that held up. Two lower-risk/higher-risk alternatives exist and are noted in §8.

---

## 1. What this paper can claim, and at what strength

| Claim | One-line evidence | Strength |
|---|---|---|
| CEM planner cost-ranking collapses under a regime shift, not universally, and does so gradually | Monotonic dose-response curve, ρ 0.532→0.295→0.169→0.078→0.001 across 5 shift strengths, n=20/point; regret 8.5px vs 88.1px at endpoints | **Primary contribution** |
| Multi-step normalised prediction error (UMF) discriminates dynamics specialists better than a one-step latent-direction baseline (S-dyn) | 60.3% vs chance 33% vs S-dyn 36.5%; S-dyn degenerately defaults to one chart | **Secondary contribution**, scoped to selector-vs-label accuracy, no planner in the loop |
| No prior work routes among persistent adapters on a frozen visual world model by measured predictive fitness | Serious recency-weighted literature search, ~20 papers read directly, no scooping candidate found | **Primary novelty claim**, holds |
| Lightweight adapter reduces the model's own prediction error with more training data | UMF falls monotonically 0.336→0.302→0.268 across a 5× data range | **Secondary, with a disclosed caveat** (shared fixed validation set across the sweep) |
| Adapter does not improve one-shot open-loop planning success | 44.0% vs 43.0%, N=100, CI [−9,+7], McNemar p=1.000 | **Reported negative result, explicitly scoped** to one-shot open-loop planning + off-policy-trained adapter |
| Verification-gated expansion (commit only after unseen-data verification) is a good idea, unlike any of six compared methods | All six comparison papers commit on the same data that triggered detection; ATLAS's `Expander` does not | **Disclosed as an idea + isolated-primitive demonstration**, NOT claimed as demonstrated for the deployed controller |
| Fitness-based routing beats regime-inference in real deployment (E1, "THE GATE") | Closed analytically, not run | **Explicitly future work** — do not claim this was validated |
| ATLAS is a demonstrated continual-learning system | E4 never executed once; `atlas_step()` never ran in production | **Do not claim** — explicitly future work |

---

## 2. Lead result: the planner's own cost ranking collapses under a regime shift

Write this as the paper's primary empirical contribution.

Under the default physics regime (R0), the CEM planner's cost function ranks its own 300 candidate action sequences well: mean per-seed Spearman correlation between predicted cost and true outcome is **ρ = 0.532** (95% CI [0.388, 0.676], n=20 seeds — re-measured 2026-08-27 at double the original n=10 sample; original was ρ=0.501, CI [0.277,0.726], same conclusion, tighter CI). Under the R2 (damping) regime shift, using the identical frozen model and identical planner, that correlation collapses to **ρ = 0.001** (95% CI [−0.132, 0.134], n=20 — re-measured from the original ρ=−0.072, CI [−0.243,0.099]) — genuinely indistinguishable from zero, not just "not clearly positive." Because R0 and R2 are evaluated with the same model and the same planner, this rules out "the planner is just generally bad" — the collapse is regime-shift-conditioned, not a standing property of this frozen checkpoint. The re-measurement at 2x the original sample tightened both CIs without moving either conclusion — this is the strongest-evidenced result in the project.

The cost of trusting this broken ranking is large and directional, not merely noisy: mean regret (true distance of the planner's chosen action vs. the best candidate actually available in the batch) is **8.5px under R0** and **88.1px under R2** — roughly a 10× gap. Under R2, the cost-ranked top-10 candidates are on average *worse* than the unranked batch mean (−15.7px), meaning trusting the ranking is actively counterproductive there, not merely uninformative.

This is not an artifact of an untrained, iteration-0 candidate pool. Running CEM to full convergence (300×30 search, 3 seeds) under R2 shows the planner **confidently converges to a worse position than the episode started at**, in all 3 seeds tested — verified as genuine convergence via a tight candidate spread (std 3.8–8.3px, not noise). The planner is *certain and wrong*, not merely uncertain.

**The collapse is a gradual slide, not a threshold break — new result, 2026-08-27, strengthens this from a two-point comparison into an actual dose-response curve.** Ran three intermediate shift strengths (damping 0.125, 0.25, 0.375, between R0's 0 and R2's 0.5), n=20 seeds each, identical protocol:

| Damping | 0.0 | 0.125 | 0.25 | 0.375 | 0.5 |
|---|---:|---:|---:|---:|---:|
| Mean rho | 0.532 | 0.295 | 0.169 | 0.078 | 0.001 |

The decline is smooth and monotonic. This is a materially stronger claim than "ranking is fine here, broken there": it demonstrates the planner's ranking quality degrades *proportionally* to the physics mismatch, consistent with a continuous-mismatch mechanism rather than a discrete failure mode that only trips past some critical threshold. Lead with this curve, not just the two endpoints — it's the more novel and more specific piece of evidence, and a reviewer is much more likely to find a monotonic dose-response curve convincing than an isolated before/after pair.

**Related work, and why this is not the same finding as the closest prior result.** The general phenomenon that a world model's prediction accuracy can dissociate from a planner's downstream competence is not new — it traces to Lambert et al. (2020, "Objective Mismatch in Model-based Reinforcement Learning") and has a theoretical companion in Grimm et al.'s Value Equivalence Principle. Two 2026 papers make a closely related point specifically for JEPA-style latent-space planning: RC-aux, and "The Objective Is the Bottleneck" (arXiv:2608.12959). The latter is the closest prior result and must be cited explicitly — but it is mechanistically distinct from the finding here. Their cost-ranking collapse is a **horizon/geometry-conditioned** saturation-and-inversion of a fixed squared-latent-distance cost, present even with **zero distribution shift**, on a single model throughout their study. The finding here is **regime-shift-conditioned**: the identical planner and identical model rank candidates well under one regime and badly under another, with the no-shift regime serving as an explicit, measured control. Frame this explicitly as extending the emerging 2026 "planner-ranking-not-predictor-accuracy" cluster (Lambert 2020, Grimm 2020, RC-aux, arXiv:2608.12959, arXiv:2607.04464) to a new axis: regime shift rather than planning horizon.

**Do not present this as an independent replication of N1 below** — under the current one-shot open-loop protocol, a broken cost ranking under R2 *is* the entire episode outcome, with no replanning available to partially correct it. N1 and N3 are one mechanism observed twice (cause and consequence), not two corroborating findings. State this relationship explicitly.

---

## 3. Scoped negative result: no measurable planning-success gain from the adapter (report honestly, with its real scope)

Baseline (frozen model) vs. `ln_act` (a ~10.7k-parameter adapter trained to reduce R2 prediction error), same real (start, goal) pairs per seed:

| Arm | Success rate | Δ vs. baseline | 95% CI | Test |
|---|---:|---|---|---|
| baseline | 44.0% (44/100) | — | — | — |
| `ln_act` chart | 43.0% (43/100) | −1.0pp | [−9.0, +7.0] | McNemar p=1.000 |

This is a well-powered null (the CI is roughly half the width of an earlier N=20 measurement), but it must be reported with two scope limitations stated explicitly, not left implicit:

1. **Every one of these 100 episodes consists of exactly one CEM search**, executed before the agent observes any consequence of its own actions (`num_act_stepped=6`, `replans==1` for all 100 episodes, both arms, confirmed against the raw per-episode logs). A better world model has exactly one channel to express itself and zero channels to correct a mistake mid-episode — which is precisely the mechanism §2's cost-ranking collapse operates through. State plainly: **this null is well-powered evidence about one-shot open-loop planning specifically, not a general statement about whether adaptation can help.**
2. The adapter behind this number was trained on **replayed expert demonstrations recorded under the unshifted regime**, played back open-loop under the shifted regime — i.e., it may be fit to the wrong target distribution rather than to what a planner operating under the shift actually needs modeled. This is a second, independent candidate explanation for the null that the current evidence does not rule out. State this as an open question, not a settled one.

**The dissociation.** UMF (this project's prediction-error metric) *does* predict success within a fixed arm: episode-level Kendall's τ is −0.406 (baseline, n=92, p<0.0001) and −0.449 (chart, n=94, p<0.0001) — lower predicted error correlates with success, as expected. This survives controlling for episode difficulty and contact count (partial τ −0.358 / −0.374, both p<10⁻⁶). **Disclose the sample size explicitly**: n=92/94 of 100, because the project's own motion gate nulls UMF for low-motion episodes, and the excluded episodes are systematically the easy, always-successful ones — not a random subsample. State the dissociation itself — UMF tracks success within an arm but not which arm is better — as **extending the Lambert et al. (2020) / Grimm et al. (2020) objective-mismatch line of work to a frozen visual foundation-model world model with a persistent adapter**, not as a novel discovery of the phenomenon type.

**Training-data sweep.** More chart training data monotonically improves UMF (0.336 → 0.302 → 0.268 across 20/60/100 trajectories) while every planning-success CI at every data point spans zero. Disclose: all three points share an identical fixed 8-trajectory model-selection set while training data grows 5×, so part of the monotone UMF trend may reflect increasingly better fit to that fixed selection set rather than pure generalizing capacity gain. The qualitative pattern (more data helps the metric, buys nothing in this planning protocol) is well supported; the precise magnitude/monotonicity of the UMF trend should be stated with this caveat attached.

**Does this null generalize to the other physics regime? Yes, in direction of conclusion; no, in point-estimate sign — new result, 2026-08-27.** Every number above uses the R2 (damping) regime. Ran the same baseline-vs-chart comparison for the first time under R1 (friction), N=40 paired episodes: baseline 70.0% (28/40), chart 60.0% (24/40), **Δ −10.0pp, 95% CI [−27.5, +7.5]**, McNemar p=0.388, 12/40 discordant pairs. Not significant (CI touches zero), but the point estimate is **negative** here, unlike R2's near-flat trend. Report this as: (a) the "no reliable, significant planning benefit" conclusion holds in both tested regimes, strengthening the negative result's generality; but (b) the *direction* of the (non-significant) effect flips between regimes, which argues against describing the chart as simply "neutral" — the more parsimonious read is noise around a genuine null, not a stable protective or beneficial effect in either direction. Include both regimes' numbers in the write-up rather than only the R2 headline.

---

## 4. Secondary result: multi-step normalised fitness routing beats a one-step latent-direction baseline

Across a 3-chart library, UMF-based selection correctly identifies the regime-appropriate chart **60.3%** of the time (chance = 33%); S-dyn -- a one-step latent-delta cosine baseline, not an appearance/visual-similarity router (`atlas/router.py::_sdyn_score`; the appearance-similarity router S-obs in the proposal was never implemented) -- manages **36.5%**, indistinguishable from chance. The confusion matrix shows why: S-dyn nearly always defaults to the identity chart regardless of the true physics regime — a qualitatively degenerate failure, not merely a weaker margin. This rules out "UMF just barely edges out a competent baseline": S-dyn is not discriminating dynamics shifts at all.

On the decisive two-chart cell isolating dynamics shift from appearance shift: UMF **0.833** vs. S-dyn **0.570** (+26.3pp). **Report only these two numbers, not a causal story about which fix produced the change.** An earlier version of this comparison (pre-fix, 0.880/0.324) is no longer independently reproducible from raw data, and the router's spread-normalized hysteresis margin is provably, algebraically inert for any exactly-2-chart library (the incumbent, when not already the winner, is by construction the maximum of a 2-element set, so the margin condition can never trigger a "hold"). Whatever moved the number between the two measurements was a separate change to how routing state persists across chunks, not the hysteresis margin — do not credit the margin fix in the write-up.

**State the scope of this result plainly**: it measures whether the selector picks the chart matching the ground-truth regime label, over pre-collected trajectories, with **no CEM planner in the loop**. It validates the selection mechanism, not that the selected chart plans any better than an alternative — given §3's dissociation, a "correctly" selected chart is not thereby shown to improve planning outcomes. If the paper's framing is "measure fitness, don't infer the regime," this result should be described as validating the fitness *signal's discriminative power*, not as demonstrating fitness-based selection beats regime-inference where it matters (task success) — that comparison has not been made.

A supporting mechanism check: the verification-gated expansion primitive fires when it should (3 charts committed under a genuine physics shift, real held-out verification chunks, confirmed leakage-free) and stays almost entirely quiet under an appearance-only shift (≈1.3% of chunks cross the commit threshold — not exactly 0.0% as earlier reported — with 0 charts actually committed). Report the corrected ≈1.3% figure; the 0-commits substantive conclusion is unaffected and, if anything, the corrected story (the probe occasionally fires and still correctly declines) is a stronger demonstration than "never fires at all."

---

## 5. What this paper must NOT claim

State these boundaries explicitly in the paper, not just observe them by omission:

- **Not a demonstrated continual-learning system.** The prequential controller (`atlas_step()`) that constitutes the method has never executed in production. No experiment has tested plasticity, retention, recall, or forgetting over a regime-revisit stream.
- **UPDATED 2026-08-27 — "THE GATE" (E1) closure argument has been fixed and reversed, not just flagged.** The fabricated `chart_R1` row has been replaced with a real 20-episode evaluation (`atlas_out/e0_chart_r1_on_r2/ln_act_R2.jsonl`). Real result: oracle SR 60.0%, random SR 46.7%, **spread 13.3pp, bootstrap 95% CI [3.3, 25.0]** — this clears the project's own 10pp reporting bar and the CI excludes zero, the opposite of the fabricated version. **This can now be reported as a real (if still small-sample) positive finding**: a genuine denominator for chart-routing exists in this library, so the original argument for skipping E1 entirely no longer holds. Whether to actually spend remaining budget on a full E1 evaluation is an open decision, not a closed one — see `research_audit/CLAIMS_MATRIX.md` row N8 and `REDTEAM.md`'s updated N8/RQ1 sections.
- **Not verification-gated expansion demonstrated for the deployed ATLAS controller.** The demonstrated commit path is a direct call to the `Expander` primitive from an offline analysis script. The production controller's own expansion branch is unreachable as currently wired (hard-coded to never receive the data it needs) and has never fired. State this precisely: the primitive works in isolation; the controller has not been shown to invoke it.
- **No RQ3 (expansion ladder), RQ4 (recall on a revisit stream), L-1 (7-arm attribution table), or G-1 (any continual-learning claim) at all.** Zero episodes exist for the continual stream. Do not describe any of this as "in progress" or "pending a final run" — as currently wired, launching it would not measure what it claims to (verified: two arms of the planned 7-arm ladder are structurally indistinguishable from each other; one arm's core mechanism is dead code). Describe this honestly as future work requiring specific, named fixes first, not as an experiment merely awaiting compute.

---

## 5b. The C3 claim ("addresses JEPA-WM's open appendix question") — reword before submitting

`ATLAS_proposal_v7.md` currently frames UMF-vs-planning validation as addressing an "open" question left by JEPA-WM's appendix. Checked directly against the actual paper (Terver, Yang, Ponce, Bardes, LeCun, TMLR 2026, `arXiv:2512.24497`): its Appendix G.3, literally titled "Is there a proxy for success rate?", already runs a real between-model, across-training-epoch Spearman correlation study and finds a moderately-to-strongly positive answer (ρ≈0.70–0.86 on Push-T/Wall). The question is not actually open — describing it that way overstates the gap and is exactly the kind of claim a reviewer who has read the cited paper catches immediately.

**Reword to:** UMF extends JEPA-WM's own appendix analysis to a new, finer granularity — *within-arm, episode-level* validation for adapter selection — rather than answering a question JEPA-WM left unaddressed. This finer granularity is in fact where the actual dissociation (N2) shows up; JEPA-WM's coarser between-model analysis would not have surfaced a within-arm-predictive/across-arm-blind pattern like this project's. That's the genuinely new, defensible part of the claim — lead with it instead of the "open question" framing.

---

## 5c. Exact references to include (pulled from `LITERATURE_AUDIT.md`, verified against primary sources — use these, don't reconstruct from memory)

| Citation | Full reference | Why it's needed |
|---|---|---|
| Lambert et al. 2020 | Lambert, Amos, Yadan, Calandra — "Objective Mismatch in Model-based Reinforcement Learning," L4DC 2020, `arXiv:2002.04523` | Foundational objective-mismatch paper; required for §3/§6's framing of the negative result as extending known work, not discovering it. |
| Grimm et al. 2020 | Grimm, Barreto, Singh, Silver — "The Value Equivalence Principle for Model-Based Reinforcement Learning," NeurIPS 2020, `arXiv:2011.03506` | Theoretical companion to Lambert et al.; same purpose. |
| Vakalis 2026 | "Operator-on-F Complements Value-Equivalence: A Planning-Time Diagnostic for Latent World Models," `arXiv:2607.04464`, Jul 2026 | Same finding-family, TD-MPC2 substrate; part of the citation cluster §2 references. |
| Li et al. 2026 (RC-aux) | "Predictive but Not Plannable: RC-aux for Latent World Models," `arXiv:2605.07278`, May 2026 | Already cited by the proposal — keep; closest JEPA-family prior work, mechanistically distinct from N3 (training/planning-horizon mismatch, not regime shift). |
| Singh 2026 | "The Objective Is the Bottleneck: Latent World Models Encode What Their Planners Cannot Use," `arXiv:2608.12959`, Aug 2026 | **Not currently cited — must add.** Closest prior result to N3; mechanistically distinct (horizon/geometry-conditioned, no shift) but a reviewer who knows this paper and sees it uncited will read that as a gap. |
| JEPA-WM (the substrate paper) | Terver, Yang, Ponce, Bardes, LeCun, TMLR 2026, `arXiv:2512.24497` | Needed regardless for the C3 reframing in §5b — cite its Appendix G.3 specifically when rewording the "open question" claim. |
| Alegre et al. (MBCD) | **Venue correction needed**: AAMAS 2021, not ICML 2021 as `ATLAS_proposal_v7.md` §10's References section currently states. Small but exactly the kind of error a knowledgeable reviewer catches and then generalizes from. This fix belongs in `ATLAS_proposal_v7.md`, not this notes file — flagged here so it isn't missed. |
| Optional, workshop-fit nod | "Dream Rehearsal for Continual Model-Based RL," `arXiv:2607.19749` | Different substrate family (Dreamer, not JEPA) but a workshop titled *Continual World Models* will expect at least a nod to the continual-MBRL literature; its total absence from the current reference list is a plausible related-work gap a reviewer could raise, not a scooping threat. Optional but cheap to add. |

---

## 6. Suggested abstract paragraph (draft — adapt freely)

> World models built on frozen visual foundation models are often adapted to new physical dynamics via lightweight, swappable modules ("charts"), selected online by a measured predictive-fitness signal. We show that under a physics-regime shift, the failure is not primarily in the world model's prediction accuracy but in the planner's own candidate-ranking process: a CEM planner's cost function ranks action sequences well under nominal dynamics (Spearman ρ = 0.53, n=20 seeds) and degrades smoothly and monotonically to chance-level, and sometimes counterproductive, ranking quality as a damping shift is scaled from zero to its full value (ρ = 0.53 → 0.30 → 0.17 → 0.08 → 0.00 across five shift strengths, n=20/point), even when a chart demonstrably reduces the underlying model's prediction error. Consistent with this mechanism, a chart trained to reduce prediction error under the shifted regime produces no measurable improvement in one-shot open-loop planning success at high statistical power (N=100), extending a growing literature on the dissociation between model-quality metrics and downstream control competence to frozen-backbone, adapter-based world models. Separately, we show that the same predictive-fitness signal reliably discriminates a genuine dynamics shift from a visual appearance change — correctly selecting the regime-appropriate chart 60% of the time against a 33% chance rate, versus 37% for a one-step latent-direction baseline (S-dyn) that degenerates to always selecting the same chart. Together these results argue that in this setting, the bottleneck to online adaptation is the planner's search process under distribution shift, not the world model's representational capacity to absorb it — a distinction with direct implications for where future work on continual world models should focus its adaptation machinery.

---

## 7. Open/optional experiments before finalizing, in priority order

**(a) DONE 2026-08-27 — N8's fabricated `chart_R1` row replaced with a real evaluation.** Result: oracle 60.0% vs. random 46.7%, spread 13.3pp, CI [3.3, 25.0] — clears the reporting bar, reverses the earlier closure argument. See §5 above and `research_audit/CLAIMS_MATRIX.md` row N8 for the full number and caveats (the two charts in this comparison come from two different training runs, `e0_v6_R1` vs `e0_v3_dataset` — not a perfectly matched triple, still worth disclosing). **New follow-on decision, not yet resolved:** given a real denominator now exists, is it worth committing budget to an actual full E1 routing evaluation? Not required for a defensible paper, but no longer closed off.

**(b) Properly-powered, compute-matched closed-loop rerun of N1 — ~15–30 GPU-hours, a real gamble, optional.**
The only existing closed-loop measurement (nas=2, N=20: baseline 40.0% vs. chart 50.0%, +10.0pp, CI [−10,+30], not significant, 4 discordant pairs out of 20) is too underpowered to cite as evidence of anything on its own, and is confounded with a 3× CEM-compute increase relative to the nas=6 protocol (because `plan_length` stays pinned at `horizon=6` regardless of remaining steps). A matched-compute rerun at N=100 would either strengthen the "protocol confound" framing in §3 (if it reproduces the positive direction) or convert N1 from "weakened by an alternative explanation" into a clean, unqualified negative (if it doesn't) — the latter is a worse outcome for the paper's positive-result count even though scientifically cleaner. Only pursue this if the team has real confidence in the +10pp direction and can accept either outcome.

**(c) Zero-cost text corrections — status as of 2026-08-27, checked against this document's own body text:**
- [x] N1: scope stated explicitly — §3, point 1
- [x] N2: n=92/94 disclosed and explained — §3, "The dissociation" paragraph
- [x] N4: shared-selection-set caveat disclosed — §3, "Training-data sweep" paragraph
- [x] N7: hysteresis-fix causal attribution dropped, numbers-only reporting — §4, paragraph 2
- [x] N9: corrected to "≈1.3%, 0 commits" — §4, paragraph 3
- [x] N6: reframed as selector-vs-regime-label accuracy — §4, paragraph 2 ("State the scope...")
- [x] N8: proxy replaced with real data — §5, second bullet; spread now 13.3pp [3.3,25.0]
- [x] C3: reworded — new §5b above
- [x] Citations: exact references (Lambert 2020, Grimm 2020, Vakalis 2026, RC-aux, arXiv:2608.12959, JEPA-WM itself) listed with full bibliographic detail — new §5c above
- [x] MBCD venue fix: flagged in §5c's table — **this one still needs to be applied to `ATLAS_proposal_v7.md` §10 itself, not just noted here**, since this notes file doesn't carry the proposal's reference list.

**Everything in this checklist is now either applied in this document's own prose or flagged with its exact target location. The only action item left outside this file is the MBCD venue correction in `ATLAS_proposal_v7.md` §10.**

---

## 8. v3 Phase-0 findings (added 2026-08-28; provisional until E0′ / P0-G)

### 8.1 Planner-budget calibration (P0-C) — `iterations = 10` adopted, not final

Pre-registered gate (IMPLEMENTATION_PLAN_V3 §3.5): frozen baseline, R2, `nas=2`,
`N=300`, `horizon=6`, n=20 paired episodes (seed = episode index), at
`iterations ∈ {30, 15, 10}`. Adopt the smallest whose SR lies inside the paired
bootstrap CI of `iterations=30`.

| iterations | SR (n=20) | Δ vs it=30 | 95% CI | McNemar p | wall/episode |
|---|---|---|---|---|---|
| 30 | 8/20 = 40% | — | — | — | ~5.8 min |
| 15 | 9/20 = 45% | +0.05 | [−0.10, +0.20] | 1.000 | ~2.9 min |
| 10 | 10/20 = 50% | +0.10 | [−0.10, +0.30] | 0.625 | ~1.8 min |

Both cut settings land inside the it=30 CI, so the rule adopts **`iterations = 10`**
(≈3× cheaper; makes the E4 stream affordable). **The N=20 result is "adopted", not
"final": E0′ at N=100 is the decisive check.** If it=10 holds at N=100 it is
confirmed; if it degrades, revert to it=30 and cut experiment sizes per §14 — and
the P0-C discordant-episode audit (below) already establishes the degradation would
not be a wiring bug, just an unlucky n=20 sample.

**For the appendix — verbatim from `phase0_v3/p0c/discordant_analysis.txt`:** only 4
of 20 episodes were discordant between it=10 and it=30 (episodes 8, 9, 17, 19), and
one (19) flipped *toward* more iterations — not a monotone "less search wins" pattern.
The `planner_diagnostics` traces confirm the `--iterations` flag is honoured in the
raw data (15 vs 10 CEM iterations/replan recorded, not just wall-time scaling). In
episodes 8/9/17 the lower-iteration run nailed its first plan and terminated in 5–12
steps before compounding model error could accumulate; the higher-iteration runs
missed the first plan and drifted the block to 110–225 px off over 2–3 further
replans. This is the planning-side face of optimiser-vs-model-error over-exploitation
(§2's theme), not a bug — but at n=20 with 4 discordant episodes it is statistically
noise, hence the N=100 checkpoint.

### 8.2 Regime reality (P0-F / G4) — R2 solid, R1 is a prediction-level shift only

Rewritten G4 (fixed identical actions, no planner, paired by seed, tested vs an
R0-vs-R0 noise band; IMPLEMENTATION_PLAN_V3 §9):

- **R2 (damping 0.5): a genuine regime shift at every level.** Block pose endpoint
  moves +32–41 px vs R0 (KS p = 2×10⁻¹¹), stable across 40→200-step pushes.
- **R1 (friction 2.0): NOT a trajectory-level shift.** A fixed push ends only
  +8–9 px further than under R0 — inside the ±13 px R0-vs-R0 noise band — and this
  is **flat across 40→200 steps** (the "friction compounds over a long push"
  hypothesis is measured and rejected). `regimes.py` notes friction saturates at
  2.0, so the axis has no headroom.
- **But R1 *is* a prediction-level shift.** P0-D: the frozen world model's UMF
  exceeds the R0 95th-percentile on **34%** of informative R1 chunks vs **5%** for
  R0 — and (stratified check) R1's strike chunks have *higher* block displacement
  than its non-strike chunks (51 vs 37 px), so this is genuine misprediction on
  high-motion chunks, not small-denominator UMF inflation.

**Paper consequence:** describe R1 accurately — a regime under which the frozen
model mispredicts dynamics (a real, chart-correctable deficit) but which produces
only a marginal change in where a fixed push ends up. Do **not** lump it with R2 as
"a large behavioural shift". Whether R1 stays in E0/E2 is a live call; if kept, its
weakness is stated, not hidden (per CLAUDE.md §1.8).

**Rescope adopted (2026-08-28):** R2 (damping) is the sole regime for any
trajectory-level or success-rate claim — E1 routing, E3+E4 continual stream. R1
(friction) is kept only for prediction-level claims: RQ2 (dynamics-vs-appearance —
R1 is the *harder* test, since UMF must track dynamics with no dramatic visual or
trajectory cue) and RQ0 (a chart specialising on a real-but-subtle prediction shift
is a legitimate capacity test). R1 is dropped from E0′'s full N=100 success-rate arm
and from E1/E3/E4 entirely.

**Future work — the "R3" regime.** Friction turned out prediction-level only, and
per `REGIME_DESIGN_REVIEW.md` the only appearance-matched dynamics lever left is the
pusher's PD tracking gains `k_p`/`k_v` (`pusht_env.py:384`). A "stiff vs sluggish
actuator" regime changes contact velocity directly, so unlike friction it should
carry *both* a prediction-level and a trajectory-level effect. Name it in future
work as the appearance-matched, mechanistically-distinct second dynamics shift to
validate next. Full spec: `IMPLEMENTATION_PLAN_V3.md` §15 item 6.
