# ATLAS — Claims Matrix

**Last updated: 2026-08-27, pass 3 — N8/RQ1 RE-MEASURED with real data (see those rows below): the fabricated `chart_R1` proxy that closed E1 without a run has been replaced with an actual 20-episode evaluation, and the conclusion REVERSES — the real oracle-minus-random spread (13.3pp, CI [3.3,25.0]) clears the 10pp bar the fabricated version (2.5-3.3pp) missed. This is the single most consequential update since the audit completed. Pass 2 (evidence levels from the four completed audit passes, `REDTEAM.md`'s full adversarial pass) remains otherwise current — see the "Pass 2 evidence-level summary" section below for everything else.**

---

## What this file is

One row per major claim the ATLAS project makes or would make in a paper.
Written during a pre-submission audit for the NeurIPS 2026 Workshop on
Continual World Models (Idea Track, 2-4 pages, non-archival, deadline
29 Aug 2026 AoE).

This file is the register that the other `research_audit/` documents point
back to. When another audit document says a finding "threatens C1" or
"threatens N2", the identifier refers to a row below.

**Read `.claude/skills/research-audit/SKILL.md` for the L0-L7 evidence-level
definitions before using this file.** In brief: L0 asserted, L1 specified,
L2 code exists, L3 code runs correctly, L4 raw results exist, L5 statistic
independently recomputed, L6 claim actually supported, L7 claim survives
adversarial attack.

Two level columns, deliberately:

- **Claimed** — the level the project's own documents implicitly assert by
  the way they state the claim. Project documents were written by prior
  Claude Code sessions and are themselves unverified claims, so this column
  is *what is being asserted*, never evidence.
- **Verified** — the level this audit can actually defend. Set to L0 on this
  pass for everything; the four audit agents fill it in.

---

## Section A — Claims from the proposal (`ATLAS_proposal_v7.md`)

These are the claims the project set out to make. Several are now in tension
with the project's own results; that tension is the point of the audit.

| ID | Exact claim (paraphrased tightly; source section) | Tested by | Claimed | Verified | Status |
|---|---|---|---|---|---|
| **C1** | Selecting among persistent adaptation modules by their **measured** action-conditioned rollout error on transitions none has trained on works, and works better than routing by similarity to a stored fingerprint. To our knowledge no module-library method for frozen visual foundation models routes this way. (§2 Primary) | E1, E2 | L4 | L0 | PENDING |
| **C1-novelty** | The *mechanism* (prediction-error model-bank switching) is explicitly disclaimed as novel (Narendra 1992-2003, MBCD 2021). The novelty claimed is narrowly its transfer to a **frozen high-dimensional visual latent space with no calibrated likelihood**. (§2, §3.5) | Literature only | L0 | L0 | PENDING |
| **C2** | Commit a new module only after **demonstrating on future unseen data** that it closes the deficit. To the best of our knowledge no existing method verifies on future unseen data before committing. Stated as a hypothesis (RQ3), not an established result. (§2 Secondary) | E3 (inside E4) | L2 | L0 | PENDING |
| **C2-probation** | A verified chart arrives already fitted and wins on merit, so the forced-execution probation period earlier designs needed is deleted. (§2) | E3/E4 | L0 | L0 | PENDING |
| **C3** | UMF — L2 latent prediction error normalised by observed latent motion — is a useful normalised predictive-fitness score. Novelty claimed only for (a) the choice of *predicted stasis* as the null model and (b) **validating it against planning success**, addressing JEPA-WM's open appendix question. (§2 Supporting) | E0, E1 | L4 | L0 | PENDING |
| **C4** | The Deployment Stream protocol — plasticity / retention / recall, the parameter-vs-system-level retention distinction, and a cross-policy competence diagnostic — is a useful contribution. (§2 Supporting) | E4, E5 | L1 | L0 | PENDING |
| **RQ0** | A lightweight adapter can absorb a physical dynamics shift at all. Pre-registered decision rule: use the smallest adapter class reaching >= 90% of full-predictor gain in **both** UMF reduction **and** planning success, in **both** regimes. (§5, §7 E0) | E0 | L4 | L0 | PENDING |
| **RQ1** | UMF identifies the competent chart, and does so better than dynamics-fingerprint routing. Pre-registered pass criterion: normalised recovery `(SR_umf - SR_rand)/(SR_oracle - SR_rand) >= 0.8`, **reported only when the oracle-minus-random denominator >= 10 pp**. Declared "THE GATE" for the whole project. (§5, §7 E1) | E1 | L1 | L0 | PENDING |
| **RQ2** | UMF responds to *dynamics* rather than *appearance*. Decisive measurements: Cell B (same appearance, different physics) UMF routing accuracy >> S-dyn; Cell C (different appearance, same physics) charts committed = 0. (§5, §7 E2) | E2 | L4 | L0 | PENDING |
| **RQ3** | Verification-gated expansion beats detect-and-spawn: ATLAS commits about 2 charts (the true regime count for stream S2) where detect-only commits more than 2 and fixed-library commits 0. (§5, §7 E3) | E3 | L2 | L0 | PENDING |
| **RQ4** | A persistent chart library delivers recall on an A,B,A,B,A,B stream — success on the final revisit to regime A exceeds success on the first visit, with paired delta > 0 for ATLAS and < 0 for Persistent-AdaJEPA. (§5, §7 E4) | E4 | L2 | L0 | PENDING |
| **L-1** | **The ladder attributes gain to a mechanism.** Each of the 7 arms differs from its neighbour by exactly one mechanism (adapts / persists / library+routing / expands / verifies), so one table attributes any gain to a specific mechanism rather than to "our system". Called "the paper's central table". (§7 E4, plan §7.4) | E4 | L2 | L0 | PENDING |
| **P-1** | Charts are disjoint parameter sets, so updating one **cannot** alter another's parameters — parameter-level retention is *guaranteed*, system-level retention is not. (§1, §6) | Code + G1 | L2 | L0 | PENDING |
| **P-2** | Strict prequential order: scoring always precedes refinement, so every score is a post-refinement verdict on fresh data and no chart can win by construction. (§6, gate G2) | Code + G2 | L3 | L0 | PENDING |
| **P-3** | Every method runs on one shared substrate: same frozen DINOv2 encoder, same predictor, same CEM planner, including the AdaJEPA baseline. (plan §0, CLAUDE.md §1.1) | Code | L2 | L0 | PENDING |
| **P-4** | The physics regimes are real shifts, verified to change trajectories visibly and statistically (gate G4). R1 = friction 2.0, R2 = damping 0.5, after two documented corrections from the originally-specified mass and elasticity parameters. (plan §6.1a, gate G4) | G4 | L1 | L0 | PENDING |
| **P-5** | Paired seeding: episode i of segment s uses seed hash(s,i) for **every** arm, giving 20 ep x 6 segments x 3 seeds = **360 paired episodes per arm** (plan §8), which is what makes a 10pp paired difference detectable. | G5 + E4 | L2 | L0 | PENDING |

---

## Section B — Claims the project's *current results* actually make

These are not in `ATLAS_proposal_v7.md`. They emerged during execution and
are what `ATLAS_SUMMARY.md` (2026-08-26/27) now presents as the project's
findings. **If a paper is written in the next 60 hours, these — not
Section A — are most of what it will claim.** They therefore need auditing
at least as hard as the proposal's claims, and they have never been through
a pre-registration of any kind.

| ID | Exact claim (source: `ATLAS_SUMMARY.md` section) | Evidence cited by the project | Claimed | Verified | Status |
|---|---|---|---|---|---|
| **N1** | The chart does **not** improve planning success, at high statistical power. Baseline 44.0% (44/100) vs `ln_act` 43.0% (43/100), delta −1.0pp, 95% CI [−9.0,+7.0], McNemar p=1.000. Described as "a well-powered null, not an inconclusive one". (§4.1) | `atlas_out/e0_planning_n100`, `analysis_n100.json` | L5 | L0 | PENDING |
| **N2** | **The dissociation.** UMF predicts success *within* an arm (Kendall tau −0.406 baseline, −0.449 chart, both p<1e-4) but does not predict *which arm is better*. Survives partial correlation controlling for episode difficulty and contact count (−0.358, −0.374). (§4.1) | same | L5 | L0 | PENDING |
| **N3** | **Mechanism.** The CEM planner's own cost ranking over candidate action sequences is near-zero-correlated with true outcomes under R2 (per-seed mean rho −0.072, CI [−0.243,+0.099]) but strongly positive under R0 (+0.501, CI [+0.277,+0.726]). Regret 8.5px (R0) vs 88.1px (R2). Under R2 the cost-ranked top-10 candidates are *worse* than the unranked batch mean. Therefore the ranking degeneracy is **regime-specific, not a general property of the frozen model**. (§4.3) | `atlas_out/cost_ranking_R0`, `cost_ranking_R2_v2`, `cost_ranking_R2_converged` | L5 | L0 | PENDING |
| **N3b** | At CEM *convergence* (not iteration 0) the baseline's plan under R2 lands **farther from the goal than the episode started**, in all 3 seeds tested, with a tight candidate spread (std 3.8-8.3px) — i.e. CEM is "certain and wrong", not uncertain. (§4.3) | `atlas_out/cost_ranking_R2_converged` | L5 | L0 | PENDING |
| **N4** | More training data monotonically improves UMF (0.336 -> 0.302 -> 0.268 at 20/60/100 trajectories) and buys nothing in planning success (all CIs span zero). Presented as "the cleanest form of the dissociation result". (§4.2) | `atlas_out/e0_train_sweep_60/100`, `e0_planning_sweep_60/100` | L5 | L0 | PENDING |
| **N5** | Closed-loop planning (3 replans, `nas=2`) flips the direction positive: baseline 40.0% vs chart 50.0%, +10.0pp, CI [−10.0,+30.0], McNemar p=0.625, N=20. Explicitly labelled not significant and needing a powered re-run. (§4.4) | `atlas_out/e0_planning_nas2` | L4 | L0 | PENDING |
| **N6** | **The one positive result.** UMF-based routing over a 3-chart library achieves 60.3% accuracy vs chance 33% and vs S-dyn's 36.5%; the confusion matrix shows S-dyn defaults to `c0` regardless of true regime. (§4.5) | `atlas_out/e2_confusion_matrix` | L5 | L0 | PENDING |
| **N7** | Post-hysteresis-fix Cell B: UMF 0.833 vs S-dyn 0.570 (+26.3pp), down from the pre-fix +55.6pp. Direction survives, margin roughly halves. (§4.5) | `atlas_out/e2_R2_posthysteresis` etc. | L5 | L0 | PENDING |
| **N8** | A *perfect* oracle over the real chart library beats random chart selection by only 2.5-3.3pp — below the project's own 10pp reporting threshold. Therefore **no routing algorithm can manufacture benefit the library does not contain**, and E1 is closed analytically without being run. (§4.6, `HANDOFF.md` §7.1) | paired bootstrap over 20 R2 episodes | L5 | L0 | PENDING |
| **N9** | The verification-gated expansion mechanism has been demonstrated to fire correctly (3 charts committed through the real path under a genuine physics shift) and to stay quiet correctly (0.0% of chunks exceed threshold under an appearance-only shift). (§4.6, `HANDOFF.md` §7.6) | `atlas_out/e2_R2_cellB_q1`, `e2_R2_cellC_q1` | L4 | L0 | PENDING |

---

## Section C — Claims about the project's own process (self-audit targets)

These are claims *about the evidence*, made by prior Claude Code sessions.
Under this audit's operating rules they are L0 assertions until checked, and
several of them are load-bearing for whether Sections A and B can be
believed at all.

| ID | Claim | Where asserted | Verified | Status |
|---|---|---|---|---|
| **S-1** | The pre-2026-08-25 rollout bug is fully fixed, and every number produced after that date is trustworthy. Four specific defects named: 5x wrong time base, hard-zeroed proprio, wrong context window, unused correct implementation. | `CLAUDE.md` §0.1 | L3 | **VERIFIED (holds for what was checked)** — see Section C table below |
| **S-2** | All headless gates (G2, G3a, G3b, G5, G6) pass post-fix; G1 was rewritten and now passes; G4 is the only skipped gate. | `CLAUDE.md` §0.1, `HANDOFF.md` §7.2 | L3 | **PARTIALLY FALSE at the time first checked, NOW FIXED (FIX_SPEC.md C1/C2, 2026-08-28)** — see Section C table below |
| **S-3** | `CLAUDE.md` §0.1's status section is accurate as of its own "last checked 2026-08-25" stamp. Known to be stale on at least G1 by `HANDOFF.md`'s own admission; extent of further staleness unknown. | `CLAUDE.md` §0.1 | L2 | **CONFIRMED stale** — see Section C table below |
| **S-4** | E0 is "closed" as a negative result and three independent rescue hypotheses were tested and rejected. | `E0_RECOVERY_PLAN.md` status banner | L2 | **FRAMING OVERSTATED** — see Section C table below |
| **S-5** | `OPUS_REMAINING_TASKS.md` item 10 asserts the `closed_loop` result was "tested with a broken instrument" — four stacked train/deploy mismatches — and that `E0_RECOVERY_PLAN.md`'s current framing of it as a clean rejection is wrong. **This retraction is listed as still outstanding (Section B, write-up-only), i.e. the incorrect framing is still standing in the source docs.** | `OPUS_REMAINING_TASKS.md` #10 | L3 | **CONFIRMED TRUE** — see Section C table below |
| **S-6** | `OPUS_REMAINING_TASKS.md` item 22 asserts that the 15pp bar E0 was judged against was **invented during the recovery process** and appears in neither the proposal nor the implementation plan, and sits close to the minimum detectable effect of its own N=20 sample. Also listed as an outstanding, un-applied correction. | `OPUS_REMAINING_TASKS.md` #22 | L3 | **CONFIRMED TRUE (FIX_SPEC.md D2, 2026-08-28)** — see Section C table below |
| **S-7** | `OPUS_REMAINING_TASKS.md` item 13 asserts E2 defines routing "correctness" by **regime label** (`correct_idx = 1` whenever regime != R0), not by demonstrated planning competence — which is in direct tension with the paper's own title, "Measure Fitness, Don't Infer the Regime". Also outstanding. | `OPUS_REMAINING_TASKS.md` #13 | L3 | **CONFIRMED TRUE (FIX_SPEC.md D2, 2026-08-28)** — see Section C table below |
| **S-8** | Two Claude Code sessions worked this repository in parallel on the same checkout, and some results were relayed between them rather than independently reproduced. | `HANDOFF.md` §0, §3, `ATLAS_SUMMARY.md` §6 | L3 | **CONFIRMED TRUE, with a documented nuance** (FIX_SPEC.md D2, 2026-08-28) — see Section C table below |

---

## Section D — The claim the project has NOT tested at all

Recorded here because it is the largest gap between what the paper is
*named for* and what has been *run*, and because no row above captures it.

| ID | Claim | Status |
|---|---|---|
| **G-1** | **Nothing in this project has ever tested continual learning.** Every result on disk comes from single-regime, single-episode-population evaluations (E0's planning comparisons) or from offline routing accuracy over collected trajectories with no planner in the loop (E2). The continual stream (S2 = A,B,A,B,A,B), which is the sole experiment that would exercise plasticity, retention, recall, forgetting, expansion, and the 7-arm attribution ladder, **has never been executed end to end.** `atlas/loop.py::atlas_step()` — the prequential controller that *is* the ATLAS method — has never run in production. Consequently RQ4, RQ3, C2, C4, L-1 and P-5 have zero empirical support of any kind, and the venue being targeted is a **continual** world models workshop. | NOT RUN |

---

## Open questions this pass could not resolve

Listed explicitly rather than guessed at. These are for the audit agents and
for the user.

1. Does `scripts/run_e4.py` actually work? It is 323 lines and no longer
   raises `NotImplementedError` (contradicting `CLAUDE.md` §0.1, which says
   it does), but `atlas_out/` contains no `e4` directory. Implemented-but-
   never-run is the highest-risk state for correctness.
2. Is the closed-loop replan path correct? N5 depends on it entirely, and
   the user who commissioned this audit has independently expressed doubt
   about it. If the multi-replan path has a stale-context bug, N5 is not
   evidence of anything.
3. Is the near-universal use of `num_act_stepped=6` (one CEM search per
   30-step episode, i.e. fully open-loop) a defensible protocol for a paper
   about *adaptive* world models, or does it structurally preclude the
   effect the project set out to measure? This is a scientific-validity
   question, not a bug question, and it bears on N1, N4, and G-1.
4. Are ~2000 offline SGD steps on 20-100 trajectories per chart consistent
   with a method whose stated premise is on-the-fly adaptation (AdaJEPA:
   one SGD step, 5-transition buffer)? If the charts are trained in a
   fundamentally different regime from the one the method describes, the
   capacity result (RQ0) may not be measuring what the method needs.
5. Is the dissociation (N2/N3) novel, or a known objective-mismatch result
   rediscovered? `LITERATURE_AUDIT.md` must answer this; it determines
   whether the negative result is publishable.

---

## Pass 2 evidence-level summary (2026-08-27, written by the coordinating session from the four completed audit files)

**Read the source file for full detail on any row — this is a compressed index, not a replacement.**

### Section A (proposal claims)

| ID | Verified | One-line basis |
|---|---|---|
| C1 | **L6, holds** | `LITERATURE_AUDIT.md`: no scooping paper found after a serious search (incl. a dedicated supplementary pass). E2 (`N6`) is the actual positive evidence: UMF routing 60.3% vs chance 33% vs S-dyn 36.5%, independently recomputed exactly (`RESULTS_AUDIT.md` pass 1 claim (g)). Caveat: E2 has **no CEM planner in the loop** — this validates the *selector*, not "routing improves planning," which C1 as stated implies. |
| C1-novelty | **L6, holds** | Mechanism-not-novel disclaimer (Narendra, MBCD) reconfirmed correct; transfer-to-frozen-visual-latent-space novelty reconfirmed by two independent literature passes. |
| C2 | **L4, mechanism demonstrated, not evaluated** | `expand.py` fires correctly once (E2's q=1 diagnostic, `CODE_AUDIT.md` §2.7/L.6 confirms no data leakage). Never evaluated against detect-only or fixed-library (needs E4, never run). RQ3 comparison does not exist. |
| C3 | **L4→L3 mixed** | UMF-vs-planning validation exists (N2) but the *capacity* half (RQ0/E0) is confounded — see N1 below. |
| C4 | **L1** | Deployment Stream protocol is specified in code (`atlas/streams.py`) but has never executed (E4 never run). |
| RQ0 | **L4, decision rule inapplicable** | `full`'s gain went negative, so the pre-registered ≥90%-of-full rule is undefined (confirmed by direct data re-check, `RESULTS_AUDIT.md` §9 — this is an honest self-report, not a gap). The project substituted an ad hoc 15pp bar (S-6, still unresolved in write-up). **Additionally now confirmed:** E0's training data itself does not match the deployment protocol (S-5, see below) — RQ0 may not describe the artifact ATLAS actually deploys. |
| RQ1 | **REOPENED 2026-08-27 pass 3 — the analytic-closure argument no longer holds on real data** | See N8 below. The fabricated `chart_R1` proxy has been replaced with a real 20-episode evaluation; the real oracle-minus-random spread is 13.3pp [3.3,25.0], which clears the 10pp bar that previously justified skipping E1. Whether to actually run a real E1 routing evaluation is now an open decision, not a closed one. |
| RQ2 | **L5, Cell B holds; Cell C mostly holds with one factual correction** | N6/N7 below. Cell C's "0.000 chunks exceed τ" is **factually wrong** (true rate ≈1.28%, `RESULTS_AUDIT.md` §8) — the "0 commits" conclusion itself is unaffected. |
| RQ3 | **L0, not run** | No E4 output exists. Additionally, `CODE_AUDIT.md` §2.1 found the ATLAS arm's verification path is **hard-coded dead** in the current E4 code (`next_encoder_output=None`) — if E4 were launched as-is today, RQ3 would report "ATLAS: 0 commits," a false negative baked in by a bug, not a finding. Must be fixed before any E4 launch. |
| RQ4 | **L0, not run** | Same — no E4 output. Additionally `CODE_AUDIT.md` §2.9 found the first-visit-vs-final-revisit delta this claim needs **cannot be paired** under the current seeding scheme even if E4 is run as-is. |
| L-1 (7-arm ladder attributes gain to one mechanism) | **L2, contradicted** | Two independent static-trace confirmations (`CODE_AUDIT.md` §2.2 + `PROPOSAL_CODE_ALIGNMENT.md` L.1) that arm 2 (AdaJEPA) and arm 3 (Persistent-AdaJEPA) are **behaviourally identical** as currently coded — `AdaJEPA.reset()` is never called in production. The "persistence" rung of the central table is vacuous as-is. Also: arms 4/5/6 skip refinement when `current_idx==0` while arms 2/3 always refine (`CODE_AUDIT.md` §2.8) — the arm-3→arm-4 rung differs by two mechanisms, not one. **Both are fixable (E4 never ran, nothing on disk is contaminated) but both must be fixed before L-1 can be claimed.** |
| P-1 (disjoint charts) | **L3, true with one caveat** | Verified true for same-kind libraries. `chart.restore_()` does not restore pretrained weights for `ln_act`/`full` (only `lora4`) — currently masked by full-overwrite in practice, but falsifies the literal wording if ever exercised differently (`CODE_AUDIT.md` §2.5). |
| P-2 (strict prequential order) | **L2, gate is vacuous** | `CODE_AUDIT.md` §4.2: **Gate G2 asserts nothing** — every prior claim that "G2 passes" (`CLAUDE.md`, `HANDOFF.md`) is asserting nothing. The underlying code order was not independently re-derived as correct or incorrect beyond the gate's own (empty) check. |
| P-3 (shared frozen substrate) | **L3, true** | Encoder freeze verified at all checked entry points, no gradient path found. |
| P-4 (regimes are real physics shifts) | **L2 for reality, L4 for persistence-safety** | Gate G4 (the direct empirical check) still never run. But the specific contamination hazard this audit worried about — regime settings leaking across a reset — is now **ruled out** by two independent direct reads of the actual `pusht_env.py::_setup()` (`CODE_AUDIT.md` §9.2 + `PROPOSAL_CODE_ALIGNMENT.md` L.5): every reset fully rebuilds the pymunk space/shapes at hard-coded defaults, so R1/R2 physics cannot bleed into a later episode. This clears E4's single biggest safety concern. |
| P-5 (paired seeding, 360 episodes/arm) | **L2** | Design verified correct in isolation (`stats.py` functions clean, `paired_seed` clean). The 360-episode design itself does not exist (needs E4). Also: `make_tables.py` pairs arms by equal length, not equal key set (`CODE_AUDIT.md` §2.10) — a resume-driven length mismatch would silently misalign the pairing. |

### Section B (the project's actual current results)

| ID | Verified | One-line basis |
|---|---|---|
| N1 | **L5 arithmetic confirmed, but the protocol it's measured under is a serious confound** | Baseline 44.0% vs `ln_act` 43.0%, CI/McNemar reproduce. **But every one of the 100 episodes used exactly ONE open-loop CEM plan (`replans==1` for all, confirmed against JSONL, `CODE_AUDIT.md` §1.1)** — the planner never reacts to its own actions. This is the single most likely reviewer objection: "well-powered null" is well-powered about one-shot open-loop planning specifically, not about adaptive world modelling in general. |
| N2 | **L5, holds** | Kendall tau values reproduce; partial-correlation caveat noted (`CODE_AUDIT.md` §3.6: the partial-Kendall p-value itself is not statistically valid, but the point estimates are not in question). |
| N3 / N3b | **L5, holds, STRENGTHENED with more data AND a new dose-response curve 2026-08-27, confirmed genuinely novel vs. closest prior work** | `analyze_cost_ranking.py` mechanism confirmed sound (`CODE_AUDIT.md` §9.4, one immaterial CI-approximation caveat). **Re-measured at n=20 seeds/regime (double the original n=10), on real Modal GPU runs**: R0 (no shift) baseline mean per-seed rho **0.532, CI [0.388, 0.676]** (was 0.501, CI [0.277,0.726]); R2 (shifted) baseline mean rho **0.001, CI [−0.132, 0.134]**, chart rho **0.014, CI [−0.115, 0.143]** (was −0.072/−0.051 at n=10). Both CIs tightened; R2's estimate moved from "slightly negative, CI touching zero" to "tightly bounded around exactly zero" — a cleaner demonstration of chance-level ranking, not a weaker one. Raw data: `atlas_out/cost_ranking_R0/` and `atlas_out/cost_ranking_R2_v2/` (3 files each, seeds 0-9/10-14/15-19). **NEW: dose-response curve, n=20/point, 3 intermediate damping strengths between R0 and R2** — rho falls smoothly and monotonically: 0.532 (0.0) → 0.295 (0.125) → 0.169 (0.25) → 0.078 (0.375) → 0.001 (0.5). This is a **genuinely new, stronger claim** than the original two-point comparison: the collapse is a gradual degradation proportional to physics mismatch, not a threshold effect — evidence for a continuous-mismatch mechanism rather than a discrete failure mode. Raw data: `atlas_out/cost_ranking_dose_{0125,025,0375}/`. `LITERATURE_AUDIT.md` §9 point 1: arXiv:2608.12959 is the closest prior result and is **not the same finding** — different mechanism (horizon/geometry-conditioned cost inversion under no shift, vs. N3's regime-shift-conditioned collapse with a clean no-shift control, now further strengthened by the dose-response shape their paper doesn't test). Must be cited, does not need to be retracted. |
| N4 | **L4, holds, mildly confounded** | UMF trend real; `PROPOSAL_CODE_ALIGNMENT.md` item 5 (pass 1) notes the reported UMF is partly selected against a fixed 8-trajectory validation set while training data grows 5x, so part of the monotone trend may be monotone selection-set overfitting rather than pure capacity gain. Not independently re-quantified. |
| N5 | **L5 arithmetic confirmed, causal story confounded** | +10.0pp is real (N=20, not significant, as already disclosed). But `CODE_AUDIT.md` §1.5: because `plan_length` stays pinned at `horizon=6` regardless of `steps_left`, `nas=2` runs **3x the CEM search compute** of `nas=6` for the same 30 raw steps. The paired within-nas=2 comparison is fair; the "closed-loop feedback helps" narrative is not yet separated from "more planner compute helps." The closed-loop *mechanism itself* (context re-encoding) is independently verified clean (`CODE_AUDIT.md` §1.2) — this is a confound in the write-up's causal claim, not a bug in the code. |
| N6 | **L5, holds, strongest positive result in the project** | 60.3% vs chance 33% vs S-dyn 36.5%, exactly reproduced. Not materially exposed to the hysteresis-inertness finding below (K=3 library, not proven inert the way K=2 is). |
| N7 | **L5 arithmetic confirmed, causal attribution wrong** | The numbers (0.833 vs 0.570) are correct, but `CODE_AUDIT.md` §6.1 + `RESULTS_AUDIT.md` §10: the spread-normalised hysteresis margin (`m=0.05`) is **mathematically proven inert for any 2-chart library** (algebraic proof, not empirical) — and all three "*_posthysteresis" runs use exactly 2-chart libraries. **The margin fix could not have caused the reported improvement in any of these three runs.** Whatever actually moved the numbers was a separate "sequential hysteresis" fix. Any write-up crediting the m=0.05 margin specifically for N7 is making a claim the code cannot support. |
| N8 | **RE-MEASURED 2026-08-27 pass 3 with REAL data — conclusion REVERSED, L5** | Original finding (still true as a critique of the write-up): the 3-chart oracle-minus-random spread's `chart_R1` row was **not a real evaluation** — the baseline/c0 array duplicated as a stand-in, undisclosed in `HANDOFF.md` §7.1, entire 20-episode comparison driven by one discordant episode. **This has now been fixed by actually running the missing evaluation** (`atlas_out/e0_chart_r1_on_r2/ln_act_R2.jsonl`, 20 real episodes, `chart_ln_act_R1.pt` from `e0_v6_R1` applied to R2 planning episodes, same seeds/init-goal pairs as the existing baseline/chart_R2 rows in `e0_v3_planning_dataset_*`). **Real result: baseline 45.0% (9/20), chart_R2 50.0% (10/20), chart_R1 45.0% (9/20). Real 3-chart oracle SR = 60.0%, random SR = 46.7%, spread = 13.3pp, bootstrap 95% CI [3.3, 25.0].** This CI does **not** touch zero and **clears** the project's own pre-registered 10pp reporting threshold — the opposite of the fabricated version's 2.5-3.3pp (CI touching 0.0). `chart_R1` is the unique winner (baseline and chart_R2 both fail) on 2/20 episodes (ep. 10, 19) — real, non-trivial discordant signal, not a single-episode artifact this time. **Consequence: the argument used to close E1 without running it ("no routing algorithm can manufacture benefit the library doesn't contain") no longer holds on real data — a genuine denominator exists.** Caveat to disclose: `chart_R1` (from `e0_v6_R1`, the corrected/T9 pipeline) and `chart_R2` (from `e0_v3_dataset`, an earlier pipeline) were trained via slightly different recipes — not a perfectly matched triple, though both are real, valid, post-rollout-fix charts. Still n=20 (thin, though no longer single-episode-driven). **Open decision for the team: does this justify actually running a real E1 routing evaluation now?** See `REDTEAM.md` for the pre-existing analysis this reopens. |
| N9 | **L4, mechanism valid, one factual error in the write-up** | Data-leakage check clean (`CODE_AUDIT.md` §9.3: verification chunk is genuinely independent). Cell B's raw episode file is confirmed permanently absent (L4 ceiling, not fixable retroactively). Cell C's "0.000 chunks exceed τ" is **factually wrong** — true rate ≈1.28%, confirmed at two independent sample sizes (`RESULTS_AUDIT.md` §8) — though the "0 commits" conclusion this claim exists to support is unaffected, and the corrected mechanistic story (probe *did* fire once and correctly declined) is arguably a *stronger* demonstration than the currently-written "never fires" story. |
| **N10 (new, 2026-08-27)** | **L5, genuinely new result — first R1-regime planning-success comparison ever run at real N.** Every prior planning-success comparison (N1, N4, N5) used only the R2 (damping) regime. Ran baseline vs. `ln_act` chart under R1 (friction) at N=40 paired episodes for the first time: baseline 70.0% (28/40), chart 60.0% (24/40), delta **−10.0pp, 95% CI [−27.5, +7.5]** (touches zero, not significant), McNemar p=0.388, 12/40 discordant pairs — a real, informative sample, not a single-episode artifact. Not significant, but the point estimate is **negative** under R1, unlike R2's near-flat trend (−1.0pp to +2.5pp across N4's sweep). Supports the "no reliable planning benefit" conclusion generalizing across both tested regimes, while undermining any reading of the R2 result as "the chart is neutral/harmless" — the direction isn't stable across regimes, consistent with noise around a true null rather than a genuine protective or beneficial effect. Raw data: `atlas_out/e1_baseline_vs_chart_R1/{baseline,ln_act}_R1.jsonl`. |

### Section C (process/self-report claims)

| ID | Verified | One-line basis |
|---|---|---|
| S-1 (rollout fix complete/correct) | **L3, holds for what was checked** | `CODE_AUDIT.md` §5.1: time base, proprio threading, output alignment all independently verified clean. |
| S-2 (all headless gates pass) | **PARTIALLY FALSE at the time this row was first written; NOW FIXED (FIX_SPEC.md C1/C2, 2026-08-28)** | Was: G2 vacuous (asserts nothing, §4.2), G5 tests something different from what `CLAUDE.md` claims (§4.3) — both "passed" without testing the property named. **Now:** `scripts/smoke_gates.py::gate_g2` rewritten on structured (regime-perturbation) data driven through `_open_loop_rollout`, with a real assertion (a held-out score cannot be at least as good as the leaked training-window score) — shown to FAIL on a deliberately leaked chart (`umf_cx_Wp <= umf_cx_W` → AssertionError) and PASS on the correct held-out scoring. `gate_g5` rewritten to build two real `PushTEnv` instances at the paired seed and assert identical init state/visual/goal — shown to FAIL on deliberately mismatched seeds and PASS on paired seeds. Also fixed this pass: G1 extended to refined charts + `kind="full"` (C3), G8 loads via `source="local"` not remote (C8). G3a remains genuinely flaky (order-dependent, pre-existing, unrelated to this fix — see `FIXLOG.md` "Discovered, not fixed"). |
| S-3 (CLAUDE.md §0.1 stale) | **CONFIRMED true, more extensively than CLAUDE.md's own AUDIT NOTE already states** — see S-2 above for one gate-level instance not previously flagged. |
| S-4 (E0 "closed" as negative result) | **L2, framing overstated** | See S-5 — the `closed_loop` rejection specifically was not a clean test. |
| S-5 (closed_loop "tested with a broken instrument") | **CONFIRMED TRUE, all four sub-allegations verified** | `CODE_AUDIT.md` §9.1: collected off-policy (frozen predictor), ~9x less CEM compute at collection (100x10) vs eval (300x30), opposite-extreme replan frequency (nas=1 collection vs nas=6 eval), only 20 trajectories, reused across all three chart kinds. `E0_RECOVERY_PLAN.md`'s "closed_loop was cleanly rejected" framing is **not supportable as written**. Not a code bug — disclosed inline in comments — but a write-up framing error that should be corrected before submission. |
| S-6 (15pp bar invented, not in either design doc) | **CONFIRMED TRUE (FIX_SPEC.md D2, 2026-08-28)** | `grep -n "15pp" ATLAS_proposal_v7.md ATLAS_implementation_plan_v2.md` returns **zero matches in both files** — the string does not appear in either design document. It appears only in `E0_RECOVERY_PLAN.md` (lines 318, 327, 909), which is the recovery-process document, confirming the bar was introduced there, not pre-registered. `E0_RECOVERY_PLAN.md:327` itself already states the bar "sat near its own [minimum detectable effect]" at N=20 — self-consistent with S-6's second sub-claim. |
| S-7 (E2 correctness = regime label, not planning competence) | **CONFIRMED TRUE (FIX_SPEC.md D2, 2026-08-28)** | `scripts/run_e2.py:262`: `correct_idx = 0 if cond == "A" else cfg["correct_b"]` — a static per-cell config lookup (`CELL_CONFIGS`), assigned before any router runs and independent of any planning outcome. `grep -n "GC_Agent\|CEMPlanner\|planner" scripts/run_e2.py` returns **zero matches** — confirms no planner object exists anywhere in this script, so "correct" cannot be, and is not, derived from demonstrated planning competence. Direct tension with the paper's title ("Measure Fitness, Don't Infer the Regime") stands as described. |
| S-8 (two parallel sessions, relayed results) | **CONFIRMED TRUE (FIX_SPEC.md D2, 2026-08-28)** | `HANDOFF.md` §3 (lines 63-91) states this in its own voice: "several results in the docs above came from the *other* agent's work, relayed into this session's chat by the user, and this session did not independently re-run them" — explicitly naming `e0_v6_R1/` chart training, `umf_locality.json`, and all of `E2_RESULTS.md`/`atlas_out/e2*/` as relayed rather than reproduced by the session that wrote the summary docs. **Important nuance already in the same document (2026-08-26 correction, lines 81-91): each relayed item WAS independently derived from raw logs once, by the session that actually produced it** — e.g. E2's numbers were re-read from `e2_episodes.jsonl` per-condition, catching a real 0.828-vs-0.880 pooled-vs-per-condition discrepancy in the process. So S-8's factual claim (parallel sessions, relaying) is confirmed true, but it does not by itself mean the relayed numbers are unverified — only that neither session cross-verified the other's work. |
| **S-9** | `E0_RECOVERY_PLAN.md`'s own prescribed "P5" code fix for E1's harness (dataset init/goal, correct success criterion, `num_act_stepped=1`) was **never applied to `atlas/harness.py`/`scripts/run_e1.py`**, despite the doc's own "do before any E1 run" framing (`PROPOSAL_CODE_ALIGNMENT.md` L.6). Does not contaminate N8 (which bypasses this function entirely). **CONFIRMED CONSEQUENTIAL 2026-08-27: this bug was actually hit.** A real reduced E1 run (20 episodes × 3 routers, `atlas_out/e1_reduced_v2/`) produced **0% success for all three routers, including `oracle_id`** (the perfect-hindsight upper bound) — 0/60 total. Root cause confirmed by reading `atlas/harness.py:328` (`goal_utils.sample_random_init_goal_states`, independently random goals) combined with `atlas/harness.py:386` (`goal_utils.eval_state()`, whose own code comment in `run_e0_planning.py:209-213` explains it requires the agent's own position to match the goal too — meaningful only for goals drawn from a correlated real trajectory, "pure noise" otherwise). With independently random goals, the agent-position term makes success require the pusher to land within 20px of an unrelated random point — near-impossible by construction, independent of which chart is applied. **This run's 0% numbers are not evidence about routing quality and must not be cited as a result.** Fixing this (porting the same dataset-init/goal + `block_success()` fix E0 already validated) is required before any E1 run can produce interpretable data. |

### Section D

G-1 unchanged: still true, still the largest gap. **Additionally now sharpened**: even if E4 is launched, the current code would not actually test G-1's claim correctly without first fixing the two L-1-threatening bugs above (dead verification path, arm-2=arm-3) — so "E4 has never run" is not the only blocker; "E4 as currently coded would not measure what it claims to measure" is a second, independent blocker that must be fixed regardless of budget/time decisions.

---

*Evidence levels above are subject to revision once `REDTEAM.md` completes its adversarial pass — treat this section as "best evidence assembled," not "final."*
