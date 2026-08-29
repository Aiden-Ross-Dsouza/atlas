# Prediction Accuracy Does Not Predict Planning Competence Under a Physics-Regime Shift

*NeurIPS 2026 Workshop on Continual World Models, Idea Track (2–4 pages excluding references). Double-blind; anonymized throughout.*

<!--
TEMPLATE NOTE (not part of the paper text):
LaTeX preamble confirmed by the authors as \usepackage[dblblindworkshop, final]{neurips_2026}.
No paper checklist required for this workshop. Appendices do not count toward the page limit.

LENGTH STATUS (for the coordinating session, not the paper): main text is ~3,300 words including
tables and figure placeholders. That is roughly 5 pages set in neurips_2026, so a further cut of
~25-30% is needed to hit the 4-page limit. The cut ladder, in the order that costs the paper least:
  1. Figure 2 (confusion matrices) -> Appendix; keep the two accuracy numbers inline. (~0.4 page)
  2. Section 4.4 -> two sentences, full paragraph to Appendix. (~90 words)
  3. Related Work paragraph 2 -> compress the three 2026 papers into one sentence each. (~120 words)
  4. Section 3's "What was actually executed" -> a three-item inline list. (~100 words)
  5. Section 5 "Other scope limits" -> Appendix, keeping a one-line pointer. (~90 words)
Do NOT cut: the scope conditions attached to Section 4.2's null, the Section 4.3 "Scope" paragraph,
or the first two Limitations paragraphs. Those are the disclosures the audit found were missing, and
removing them re-creates the overclaims this draft exists to avoid.
-->

---

## Abstract

Adapting a frozen-backbone world model to new physical dynamics is usually treated as a representation problem: fit a lightweight swappable module until prediction error comes down. In a Push-T manipulation setting we find that framing locates the bottleneck in the wrong place. Under nominal dynamics a CEM planner ranks its own 300 candidate action sequences informatively (mean per-seed Spearman rho 0.532, 95% CI [0.388, 0.676], n = 20 seeds). Under a damping shift, same frozen model and same planner, that ranking collapses to chance (rho 0.001, CI [-0.132, 0.134]), and the chosen action carries 88.1px of mean regret against the best candidate in its own batch versus 8.5px without the shift. Consistently, a 10.7k-parameter adapter that lowers the model's latent prediction error changes nothing measurable in one-shot open-loop planning success (44.0% vs 43.0%, N = 100 paired episodes, CI [-9.0, +7.0]pp, McNemar p = 1.000). The same prediction-error signal does discriminate a dynamics shift from a 100%-of-pixels appearance shift, selecting the regime-matched adapter 60.3% of the time against 33% chance, where S-dyn -- a baseline that matches only the direction of the first predicted latent step -- reaches 36.5% and degenerates to one fixed adapter. We have not run a continual-stream evaluation, and we state plainly what that leaves untested.

<!-- Drew from: research_audit/PAPER_DRAFT_NOTES.md §1, §2, §3, §4, §6; research_audit/RESULTS_AUDIT.md §2(a),(d),(g), §12; research_audit/CLAIMS_MATRIX.md N1, N3, N6. -->

---

## 1. Introduction

A frozen visual foundation model plus a learned latent predictor is now a standard world-model recipe, and the standard way to make one cope with a change in physics is to attach a small adapter and train it until latent prediction error falls. The implicit assumption is that prediction error is what stands between the agent and competent behaviour. For a persistent library of adapters routed online, that assumption is load-bearing twice: for whether adapting helps at all, and for whether measured predictive fitness is the right thing to route on.

We built a system to test it: a library of persistent adapters ("charts") on a frozen DINOv2 encoder and JEPA-style predictor, selected by measured action-conditioned rollout error on transitions none has trained on, refined after selection, and expanded only once a candidate demonstrates on a subsequent unseen chunk that it closes the deficit (Section 3). What follows is what the measurements said, which is not what the design predicted.

Under our physics-regime shift the failure sits in the planner's ranking process, not the predictor's accuracy: the same CEM planner driving the same frozen model ranks its candidates informatively with no shift and at chance with one, and trusting the ranking under the shift is worse than ignoring it. That reframes the adapter result. A 10.7k-parameter adapter does lower prediction error, monotonically with training data, and its per-episode error correlates with per-episode success within a fixed arm, but it buys nothing in success rate. Our planning protocol is one-shot and open-loop, which we treat as a scope limit rather than a general claim, since an improved model then has one channel to express itself and none to correct a mistake.

Contributions:

- **A regime-shift-conditioned measurement of planner cost-ranking collapse, with a no-shift control.** rho 0.532 (R0) vs 0.001 (R2), n = 20 seeds per regime, regret 8.5px vs 88.1px, identical model and planner throughout. This extends work on planner-ranking-versus-predictor-accuracy [Lambert et al. 2020; Grimm et al. 2020; Li et al. 2026; Singh 2026; Vakalis 2026] along a new axis: distribution shift rather than planning horizon or cost geometry.
- **A well-powered, explicitly scoped null on adapter-driven planning gain,** 44.0% vs 43.0% at N = 100 paired episodes, alongside a 5x training-data sweep in which prediction error falls monotonically (0.336 to 0.268) and every planning CI spans zero. We present this as replicating objective mismatch in a new substrate, not discovering it.
- **Evidence that the normalised multi-step UMF signal discriminates dynamics specialists** where a single-step latent-direction baseline (S-dyn) does not: 60.3% vs 36.5% against 33% chance over a 3-adapter library, with the baseline failing degenerately rather than narrowly. (Both signals see the scene only through the frozen encoder; the contrast is multi-step normalised error vs. one-step direction cosine, not fitness vs. appearance.)
- **An explicit account of what remains untested,** including that no continual-stream evaluation has run and the prequential controller has never executed end to end.

<!-- Drew from: research_audit/PAPER_DRAFT_NOTES.md §1, §2, §3, §5; research_audit/REDTEAM.md "What survives cleanly"; research_audit/LITERATURE_AUDIT.md §6, §7, §9. -->

---

## 2. Related Work

**Objective mismatch.** That a model-quality metric can dissociate from control performance is established. Lambert et al. [2020] found near-zero-to-weak correlation between validation log-likelihood and episode reward in classical MBRL, and built a model whose likelihood stayed high while reward fell; Grimm et al. [2020] give the value-equivalence account of why raw prediction accuracy is the wrong target. We replicate this rather than discover it. What is new is the substrate (a frozen visual foundation model with a persistent adapter), the training-budget sweep, and the localisation of the failure to planner ranking under a regime shift.

**Planner ranking in latent world models.** RC-aux [Li et al. 2026] shows accurate short-horizon prediction coexisting with a latent geometry poorly aligned to planning, attributed to a training-versus-planning horizon mismatch. Vakalis [2026] shows reward-prediction error staying flat across a TD-MPC2 model-size sweep while planning return collapses. Closest to us, Singh [2026] shows a CEM-planned latent world model whose predictor stays informative while the squared-latent-distance objective saturates and inverts with goal distance; that failure is horizon- and geometry-conditioned and appears with zero distribution shift. Ours is shift-conditioned, with the no-shift regime as a measured control: same family, different mechanism. The substrate paper's appendix [JEPA-WM] already asks whether embedding-prediction loss proxies for success rate and answers positively at a between-model, across-training-epoch granularity. We do not claim that question was open; we extend it to a within-arm, episode-level granularity for adapter selection, where Section 4.2's dissociation appears and a between-model analysis would not surface it.

**Predictive-fitness routing and module expansion.** Routing among dynamics models by prediction likelihood is not new: it goes back to multiple-model adaptive control [Narendra et al. 1997] and appears in change-point-detecting MBRL [Alegre et al. 2021, AAMAS]. [NEEDS CONFIRMATION: exact Narendra reference and year; the audit records this line as spanning roughly 1992–2003 without pinning one paper.] We claim novelty only for transferring that mechanism to a frozen high-dimensional visual latent space with no calibrated likelihood; nearby module-library methods route on input or feature similarity instead [DPCore; CLARE; WorMI; Dynamic TMoE; ShiftEx]. On expansion, we read the trigger mechanism of six comparison methods [DEN; CN-DPM; Dynamic TMoE; ShiftEx; MBCD; CLARE] and found each commits a new module on the data that triggered detection, with no future-held-out verification step; Appendix G tabulates the mechanisms. We present verification-gated expansion as a design position supported by that gap, not a demonstrated result. Continual model-based RL is otherwise dominated by replay on reconstruction-based backbones [arXiv:2607.19749; arXiv:2606.27374]; ours is modular rather than replay-based, but we have no continual-stream result to compare against theirs.

<!-- Drew from: research_audit/LITERATURE_AUDIT.md §1, §3, §4, §6, §7, §8, §9; research_audit/PAPER_DRAFT_NOTES.md §5c; research_audit/REDTEAM.md Section C. Note: LITERATURE_AUDIT §5's softening of the C3 "open question" framing is applied in Section 5 rather than here. -->

---

## 3. Method, and What Was Actually Run

**Substrate.** A public Push-T checkpoint pairing a frozen DINOv2 encoder with a JEPA-style action-conditioned latent predictor and a CEM planner [JEPA-WM]. Encoder and planner are frozen everywhere; gradients reach only adapter parameters. Charts come in three families, spanning 10,764 to 20.8M trainable parameters (Appendix B); all results below use the smallest, `ln_act`.

**Fitness signal.** For a chart *c* and an observed chunk *Q* we score the unexplained motion fraction

> UMF(*c*; *Q*) = Σ_k ‖ẑ_k^c − z_k‖² / Σ_k ‖z_k − z₀‖²

the action-conditioned open-loop rollout error normalised by the null model that nothing moves. The denominator makes scores comparable across chunks of different motion magnitude; chunks whose displacement falls below the 10th percentile of training displacement return no score, since it is then near zero.

**The three rules.** SELECT: apply each chart in turn, score UMF on the chunk just observed, take the argmin subject to a hysteresis margin. REFINE: one SGD step on the selected chart, strictly after scoring, so no chart is scored on data it has just been fit to. EXPAND: count consecutive informative chunks where the selected chart's UMF exceeds tau; after *q* strikes, fit a candidate on the accumulated deficit chunks and commit only if it beats both tau and the incumbent on the *next*, unseen chunk. Hyperparameters are in Appendix A.

**What was actually executed.** Results come from three offline protocols, not the composed controller. (i) *Planning evaluation*: charts trained offline, then evaluated by CEM planning (300 samples, 30 iterations, 10 elites, horizon 6) in a fixed regime. Each episode is 30 raw steps and, at the substrate's own validated 6 executed model steps per replan, is **exactly one CEM search executed before the agent observes any consequence of its own actions**; we confirmed one replan per episode across all 100 episodes of both arms. (ii) *Routing evaluation*: SELECT scored against the ground-truth regime label over pre-collected trajectories with **no planner in the loop**, which measures the selector, not the outcome of selecting. (iii) *Expansion diagnostic*: the expansion primitive called directly from an offline script at a relaxed strike count (Appendix F). REFINE has never executed in a production run, and the controller composing all three rules has never executed end to end. We report what these protocols measured and nothing beyond it.

**Regimes.** R0 is the environment default; R1 sets pusher-block friction to 2.0 and R2 sets space damping to 0.5, both unset in the shipped environment. These replace an originally specified mass shift we found provably inert, since the pusher is a kinematic body and the block's post-collision velocity is independent of its own mass at any scale. Appearance shift is a 100%-of-pixels darkening, chosen over a hue rotation we measured to alter only 5.6% of pixels on this environment's near-white renders. Appendix D records both regime corrections.

<!-- Drew from: research_audit/PROPOSAL_CODE_ALIGNMENT.md items A, B, F, G, H, I, J; research_audit/EXPERIMENT_STATUS.md §1, §2; research_audit/PAPER_DRAFT_NOTES.md §3, §5; atlas/score.py (UMF definition); E2_RESULTS.md (corruption magnitudes). Anonymization: no repo paths, file names, or vendor names in prose. -->

---

## 4. Experiments and Results

Every comparison below is paired: the same seed fixes the same initial state and the same goal state for every arm. We report differences with confidence intervals from a paired bootstrap, and McNemar's exact test for paired binary outcomes.

### 4.1 Does the planner's own cost ranking survive the shift? (No.)

In one planning situation we draw the planner's 300 candidate action sequences at its first CEM iteration, record each candidate's predicted cost, and separately roll each one out in the true environment for its true final block-to-goal distance. The question is how well predicted cost ranks true outcome.

| Regime | Mean per-seed Spearman rho | 95% CI | Regret of the chosen action | Top-10-by-cost vs batch mean |
|---|---:|---|---:|---:|
| R0 (no shift) | **0.532** | [0.388, 0.676] | 8.5px | +28.2px better |
| R2 (damping shift) | **0.001** | [-0.132, 0.134] | 88.1px | **-15.7px worse** |
| R2, with `ln_act` chart | 0.014 | [-0.115, 0.143] | 92.3px | -8.5px worse |

*n = 20 seeds per regime, 300 candidates per seed; contact occurred in 80.0% of candidate rollouts in all three rows.*

Under R0 the ranking is informative; under R2 it is indistinguishable from chance, with a tight CI centred on zero rather than merely including it. Both rows use the same frozen model and the same planner, so the control rules out "this planner is simply bad." The collapse is directional rather than noisy: mean regret against the best candidate in the planner's own batch is roughly 10x larger under the shift, and the cost-ranked top decile ends farther from the goal than an unranked draw. The adapter does not repair the ranking.

Nor is this an unconverged-pool artifact. Run to full convergence under R2, the baseline's final plan lands farther from the goal than the episode started in **3 of 3 seeds**, with a spread across its 300 final candidates of 4.7–8.3px. The planner is confident and wrong, not uncertain. The chart arm's spreads are wider on two of three seeds, so Appendix C gives per-seed values rather than one range.

[FIGURE 1: two-panel scatter, one panel per regime (R0, R2). x-axis = CEM predicted cost of a candidate action sequence; y-axis = true final block-to-goal distance in pixels after rolling that candidate out in the environment. 300 points per seed, pooled over 20 seeds, with the per-seed Spearman rho distribution shown as an inset strip. Source: the cost-ranking diagnostic runs, baseline arm, capture at CEM iteration 0.]

### 4.2 Does an adapter that lowers prediction error improve planning success? (Not under this protocol.)

| Arm | Success rate | Δ vs. baseline | 95% CI | McNemar |
|---|---:|---:|---|---|
| Frozen baseline | 44.0% (44/100) | — | — | — |
| `ln_act` chart (10.7k params) | 43.0% (43/100) | -1.0pp | [-9.0, +7.0]pp | p = 1.000 |

*N = 100 paired episodes under R2; pairing verified episode-by-episode on initial state and goal.*

Two scope conditions belong inside this null, not under it. **Every one of these episodes is a single open-loop CEM plan**, so a better model has one channel to express itself and none to correct a mistake, which is exactly Section 4.1's mechanism; this is evidence about one-shot open-loop planning, not about whether adaptation can help in general. And the adapter was trained on expert demonstrations recorded under the unshifted regime and replayed open-loop under the shifted one, so it may be fit to the wrong target distribution rather than to what a planner under the shift needs modelled. Our data rules out neither.

**The dissociation.** Within a fixed arm, prediction error tracks success: episode-level Kendall's tau between mean UMF and success is -0.406 (baseline) and -0.449 (chart), both p < 10^-4, both surviving partial correlation on episode difficulty and contact count (-0.358, -0.374). These use n = 92 and 94 of 100, because the motion gate returns no score for low-motion episodes and one replan covers the whole episode. **The excluded episodes are not a random subsample**: they are the small-displacement ones, 100% of which succeeded in both arms, so the correlation is measured over the harder ~93%. The signal ranks episodes within an arm and is blind to which arm is better.

**Training-data sweep.** Across 20, 60 and 100 training trajectories, held-out UMF falls monotonically (0.336, 0.302, 0.268) while every planning-success CI spans zero (Appendix E). One caveat belongs with the trend: all three points share a fixed 8-trajectory model-selection set while training data grows 5x, so part of the monotone improvement may be better fit to that fixed set rather than capacity gain.

### 4.3 Does the fitness signal discriminate dynamics from appearance?

Over a 3-chart library (identity, an R1-trained chart, an R2-trained chart), routed against uncorrupted trajectories from each of R0, R1 and R2:

| Router | Accuracy | vs. chance (0.333) |
|---|---:|---:|
| UMF (measured predictive fitness) | **60.3%** | +27.0pp |
| S-dyn (action-conditioned latent-delta cosine similarity) | 36.5% | +3.2pp |

*120 decisions per true regime, 83–104 ungated after the motion gate.*

The baseline's failure is qualitative rather than marginal: S-dyn picks the identity chart at nearly the same rate whatever the true regime (61, 62, 65 across rows), so it is not discriminating dynamics at all, while UMF's diagonal dominates every row. On the two-chart cell isolating dynamics shift from appearance shift, UMF reaches 0.833 against S-dyn's 0.570. We report those two numbers only; an earlier measurement of that cell is not reproducible from surviving records and we make no causal claim about what moved it.

[FIGURE 2: side-by-side 3x3 confusion matrices, rows = true regime (R0/R1/R2), columns = selected chart, one matrix for the UMF router and one for the S-dyn router, cell values as counts, shading by row-normalised rate. Source: the 3-chart routing evaluation, 720 per-decision records, gated decisions excluded.]

**Scope.** This measures whether the selector picks the chart matching the ground-truth regime *label*, over pre-collected trajectories, with no planner in the loop. Given Section 4.2's dissociation, a "correctly" selected chart is not thereby shown to plan better. Read it as evidence about the fitness signal's discriminative power, not as a demonstration that fitness-based selection beats regime inference on task success. That comparison has not been made.

The expansion primitive behaves consistently with the same picture: 0 charts committed under the appearance shift, 3 committed under the dynamics shift, through the real fit-then-verify path with the verification chunk confirmed disjoint from the fitting data. Appendix F gives the threshold-crossing rates and the diagnostic strike setting, which deviates from our pre-registered one.

### 4.4 Is there anything for a router to win?

A router can only recover success some chart in the library actually delivers. Over 20 paired R2 planning episodes with a 3-chart library (identity 45.0%, R2-chart 50.0%, R1-chart 45.0%), a per-episode oracle reaches 60.0% against 46.7% for uniform random selection: a **spread of 13.3pp, bootstrap 95% CI [3.3, 25.0]**, clearing the 10pp denominator threshold we pre-registered as the condition for reporting a normalised-recovery score at all. The R1 chart is the unique winner on 2 of 20 episodes, so this is not a single-episode artifact. A routing denominator therefore exists here. Whether any router captures it is untested, and the two non-identity charts come from training runs with slightly different recipes, so this is not a matched triple.

<!-- Drew from: research_audit/RESULTS_AUDIT.md §2(a)(b)(c)(d)(e)(g)(h), §8, §11, §12; research_audit/CLAIMS_MATRIX.md N1, N2, N3, N3b, N4, N6, N7, N8, N9; research_audit/PAPER_DRAFT_NOTES.md §2, §3, §4, §5; research_audit/REDTEAM.md Sections A and B. Corrections applied here vs. source docs: N9's "0.000" corrected to ~1.3%; N7's hysteresis causal attribution dropped; N8's fabricated proxy replaced with the real 20-episode evaluation; n=20 cost-ranking numbers used, not the older n=10. -->

---

## 5. Discussion and Limitations

What this work does not establish, including where that undercuts its own design premise.

**No continual-learning experiment has been run.** Every number here comes from independent episodes in one fixed regime with an offline-trained adapter, or from offline routing accuracy over pre-collected trajectories. There is no stream, no regime revisit, and no retention, recall or forgetting measurement. The controller composing SELECT, REFINE and EXPAND has never executed end to end and REFINE has never run in production. For a workshop on continual world models this is the gap that matters most, and we state it rather than calling the experiment pending.

**Verification-gated expansion is demonstrated only as an isolated primitive.** The commit path we exercised is a direct call to the expansion routine from an offline script. In the controller as wired, the branch that would invoke it cannot receive the next-chunk data it needs, so it has never fired there. The idea is supported by a literature gap; its behaviour inside the deployed controller is not evidence we have. Relatedly, we inspected the 7-arm ablation ladder we had specified and found two arms behaviourally identical as coded, one adjacent pair differing by four mechanisms rather than one, and one arm's verification path unreachable. Run as-is that ladder would return a wiring artifact that reads as a finding, so we name the defects rather than the schedule.

**The planning protocol is the most likely alternative explanation for the null.** One open-loop plan per episode is the configuration behind every planning number here, and receding-horizon replanning is what standard control-theoretic reasoning says buys robustness to model error. We found no empirical citation quantifying this for CEM-planned latent world models, so we offer it as reasoned inference. Our only closed-loop measurement (3 replans, N = 20) points the other way at +10.0pp, CI [-10.0, +30.0], p = 0.625 on 4 discordant pairs, and is confounded with roughly 3x the search compute per episode. That is an untested direction, not a counterweight.

**Adapter-capacity conclusions are confounded with training budget.** The three adapter families under R2 were not trained at recorded, matched budgets, and the largest was never trained under R1 at all. Our pre-registered rule (smallest family reaching 90% of the full predictor's gain in both metrics and both regimes) therefore has no denominator in one regime, and the full predictor's gain went negative in the other. The honest statement is descriptive: at the budgets and protocol used, no adapter family produced a measurable planning gain. We do not report a pre-registered criterion as met or missed.

**Other scope limits.** The routing result defines correctness by regime label, in tension with a design premise about not needing to infer the regime. Both expansion diagnostics ran at a relaxed strike count. Two of our internal correctness gates turned out on inspection to assert nothing, so "the gates pass" is not independent evidence; the properties they were meant to check were verified separately against raw data. Our novelty claim rests on a serious recency-weighted search, not a proof of absence.

<!-- Drew from: research_audit/REDTEAM.md (all of Sections A-D, "What survives cleanly", "The strongest honest paper"); research_audit/CLAIMS_MATRIX.md G-1, S-2, S-5, S-6, S-7, RQ0, RQ3, RQ4, L-1, C2, C4; research_audit/EXPERIMENT_STATUS.md §3, §4, §5; research_audit/PROPOSAL_CODE_ALIGNMENT.md items A.2, C.1, C.2, F.3, G.4, H.3; research_audit/LITERATURE_AUDIT.md §9 Point 3 (open-loop citation gap). -->

---

## 6. Conclusion

In this Push-T setting a physics-regime shift breaks the planner's ability to rank its own candidate action sequences before it meaningfully limits the model's ability to predict, and a lightweight adapter that measurably improves the latter changes nothing measurable about success under one-shot open-loop planning. The same prediction-error signal is nonetheless discriminative enough to separate a dynamics shift from an appearance shift over a small adapter library. We read this as an argument for putting adaptation machinery where the failure actually is, and we report it without the continual-stream evaluation the broader system was designed for.

<!-- Drew from: research_audit/PAPER_DRAFT_NOTES.md §1 and §2; no new claims introduced. -->

---

## Appendix A. Fixed hyperparameters

Set before running and not swept.

| Symbol | Meaning | Value |
|---|---|---:|
| tau | UMF adequacy threshold | 0.5 |
| q | consecutive informative strikes to arm the probe | 3 (diagnostics at 1, disclosed) |
| m | hysteresis margin | 0.05 |
| n_probe | probe fitting steps | 20 |
| K_max | library cap | 10 |
| chart lr | refinement and offline fine-tuning (Adam) | 5e-4 |
| motion gate | informative-chunk threshold | 10th percentile of training displacement |
| offline chart training | max steps / patience / eval interval | 2000 / 5 / 25 |
| CEM | samples / iterations / elites / horizon | 300 / 30 / 10 / 6 |
| CEM | executed model steps per replan | 6 (planning eval) |
| Environment | frameskip / raw steps per episode | 5 / 30 |

**Note on `m`.** The margin is applied to the fraction of each replan's own chart-to-chart score spread rather than as an absolute score margin, because absolute 0.05 is scale-dependent across the routers compared. This is a deviation from the margin as originally specified. It is also algebraically inert for any exactly-two-chart library: an incumbent that is not the argmin is by construction the maximum of a two-element set, so the relative gap is exactly 1.0 and the margin can never hold the router. Two-chart results in this paper should be read as pure argmin routing.

## Appendix B. Adapter families

| Family | Trainable parameters | Surface |
|---|---:|---|
| `ln_act` | 10,764 | predictor LayerNorm affine parameters plus the action encoder |
| `lora4` | 118,176 | rank-4 LoRA on predictor projection matrices |
| `full` | 20,800,884 | all predictor parameters (capacity ceiling) |

Held-out UMF measured after offline fine-tuning: `ln_act` x R2 0.336, `lora4` x R2 0.329, `full` x R2 0.728, `ln_act` x R1 0.285, `lora4` x R1 0.288. `full` x R1 was never trained. The R2 trio's training budgets were not recorded and are not confirmed matched, and at least one family is documented as having been retrained at a reduced budget at some point; `full`'s outlying value is equally consistent with a budget confound or with overfitting a 20.8M-parameter model on ~100 model transitions. These numbers are reported as descriptive, not as a capacity ordering.

## Appendix C. Converged-CEM per-seed detail (R2)

| Seed | Initial block-goal distance | Baseline median final distance | Baseline candidate spread (sd) | Chart median final distance | Chart candidate spread (sd) |
|---:|---:|---:|---:|---:|---:|
| 0 | 91.5px | 119.7px | 7.0px | 89.8px | 3.8px |
| 1 | 45.8px | 62.6px | 8.3px | 54.4px | 17.9px |
| 2 | 158.2px | 200.8px | 4.7px | 231.7px | 27.2px |

All three baseline seeds converge to a plan whose median outcome is farther from the goal than the episode's own starting distance. The chart arm's spreads are markedly wider on two of three seeds, which we report rather than summarising as a single range.

## Appendix D. Regime definitions and two corrections

R0 is the environment default. R1 sets pusher-block friction to 2.0; R2 sets space damping to 0.5. Both parameters were unset (and therefore at their zero/identity defaults) in the shipped environment.

These replace two earlier specifications, and we record both corrections because they change what the experiments mean. The originally specified mass shift is inert at any scale: the pusher is a kinematic body, so the block's post-collision velocity is algebraically independent of its own mass, which we confirmed empirically over a 10^6 range of mass scalings producing byte-identical trajectories. An elasticity shift was then specified and also dropped, because the environment hard-codes zero velocity damping, so restitution has no channel through which to express itself. R1 therefore means "high friction," not "light block," everywhere in this paper.

Separately, our initial trajectory collector used per-step uniform-random actions and produced agent-block contact in only ~13–17% of rollouts. All trajectory data here uses a persistent-target proportional-aim collector, which reaches contact in 100% of a 30-seed validation set.

## Appendix E. Training-data sweep, full table

| Training trajectories | Held-out UMF | Planning success | Δ vs. paired baseline | 95% CI | McNemar |
|---:|---:|---:|---:|---|---|
| 20 | 0.336 | 43.0% (43/100) | -1.0pp | [-9.0, +7.0] | p = 1.000 |
| 60 | 0.302 | 40.0% (16/40) | 0.0pp | [-12.5, +12.5] | p = 1.000 |
| 100 | 0.268 | 42.5% (17/40) | +2.5pp | [-12.5, +17.5] | p = 1.000 |

Each row is paired against the same baseline episodes at matching episode index, pairing verified on initial state and goal. The 60- and 100-trajectory arms have recorded train/eval seed manifests confirming a disjoint 8-trajectory evaluation set, identical across both sizes. The 20-trajectory arm has no saved manifest, so its held-out status rests on the procedure description rather than recorded seeds. All three arms select their checkpoint on that same fixed 8-trajectory set, which is the caveat noted in Section 4.2.

## Appendix F. Expansion diagnostic detail

| Condition | Ungated chunks with UMF > tau = 0.5 | Charts committed at q = 1 | Charts committed at q = 3 |
|---|---:|---:|---:|
| R0, uncorrupted | 0.0–2.2% | not run | 0 |
| R0 + 100%-of-pixels darkening (appearance only) | ~1.3% | 0 | 0 |
| R2 damping shift (dynamics only) | 15.7% | 3 | 0 |

Both commit diagnostics ran at *q* = 1 rather than the pre-registered *q* = 3. At *q* = 3 nothing commits in any condition, because three consecutive strikes at the observed 15.7% per-chunk rate is a ~0.4% event, so the pre-registered setting has no discriminative power at this episode length. The appearance-only row is the substantive one: the threshold is crossed occasionally and the verification step still declines to commit, which is a stronger demonstration of the gate working than a row where it never fires. Two further precision notes. The verification chunk is an independently seeded trajectory not yet folded into the deficit pool, confirmed by tracing the caller loop. And the incumbent that a candidate must beat is chosen as the argmin over the verification chunk itself, which gives the incumbent a look-ahead advantage; this biases against committing, so it cannot inflate the 3-commit count.

## Appendix G. Expansion triggers in six comparison methods

Each row records the mechanism we read in the paper itself, and whether a new module is verified on data other than the data that triggered its creation.

| Method | Expansion trigger | Verifies on future unseen data before committing? |
|---|---|---|
| DEN | current task's own training loss exceeds a fixed threshold; group-sparsity prunes back | No |
| CN-DPM | Dirichlet-process responsibility routes samples to a short-term buffer; a new expert is trained on that buffer once it fills | No |
| Dynamic TMoE | MMD distribution-shift detector fires; a heterogeneous expert is instantiated and pre-trained immediately | No |
| ShiftEx | MMD covariate-shift detection at the party level; new expert trained on the triggering cohort | No |
| MBCD | multivariate CUSUM on a log-likelihood ratio crosses threshold; a fresh dynamics model and policy are initialised at once | No |
| CLARE | z-score of per-layer autoencoder reconstruction error against prior tasks; new adapter trained on the current task's own data | No |

Two of these operate outside world-model dynamics prediction (CLARE on vision-language-action policies, DPCore, cited in Section 2 for routing rather than expansion, on image classification under corruption), so they are precedents for the mechanism family, not direct competitors on this task. A concurrent 2026 line of work on transferring optimisation strategies across world-model backbones adopts a verify-before-accept principle of the same shape, at the level of whole-model training recipes across research campaigns rather than deployment-time adaptation modules.

## Appendix H. Statistical procedure

Paired designs throughout: episode *i* uses a seed that fully determines both the initial state and the goal state, identically across arms, verified episode-by-episode. Differences in success rate are reported with a paired bootstrap CI (10,000 resamples) and McNemar's exact test. Spearman correlations for cost ranking are computed per seed and averaged, with a normal-approximation CI over the per-seed distribution, rather than pooled across seeds, so that between-seed difficulty variation does not inflate the estimate. The partial Kendall correlations in Section 4.2 are computed by OLS-residualising both variables on episode difficulty and contact count; the point estimates are the quantity we rely on, and we note that the residualised p-value is anticonservative because it does not correct for the estimated regression coefficients.

[NEEDS CONFIRMATION: total compute used for the reported experiments, stated in a form that does not identify the hardware or provider. A GPU-hour total would be conventional to report here and we do not currently have one recorded.]

<!-- Drew from: research_audit/PROPOSAL_CODE_ALIGNMENT.md items B, B.1, E, G.1, H.1, H.3; research_audit/RESULTS_AUDIT.md §2(e), §10; research_audit/CLAIMS_MATRIX.md P-4, P-5; EXPERIMENT_STATUS.md §2 (parameter counts); CLAUDE.md §0.1 (regime-correction history, action-sampling correction). Anonymization check: no vendor, cluster, GPU model, repo path, or internal filename appears in appendix prose. -->
