# PAPER_FACT_CHECK.md

Adversarial fact-check of `PAPER_DRAFT.md` against `research_audit/` (CLAIMS_MATRIX, RESULTS_AUDIT, LITERATURE_AUDIT, REDTEAM, EXPERIMENT_STATUS, PAPER_DRAFT_NOTES, PROPOSAL_CODE_ALIGNMENT) and against the raw artifacts in `atlas_out/`. Line numbers refer to `PAPER_DRAFT.md` as of this pass.

*Produced 2026-08-27 by the `paper-fact-checker` agent. Its tool allowlist is Read/Grep only, so this file was transcribed verbatim from its report by the coordinating session. No fixes have been applied to the draft.*

## Summary counts

| Section | Issues | Notes |
|---|---:|---|
| A. BLOCKING: claim exceeds evidence | 10 | 6 are "claim exceeds evidence"; 4 are lesser (precision/framing) |
| B. BLOCKING: anonymization | 3 | 1 must-fix before conversion; 2 advisory |
| C. Unsourced claims | 5 | 1 is a table column presented as measured but inferred |
| D. Overclaiming by exclusion | 7 | D1 and D2 are the serious ones |
| E. Related Work fairness | 3 | none fatal; E1 is the substantive one |
| F. Hype language | 3 | draft is unusually clean here |
| G. Length and section budget | 1 (ordered cut list + do-not-cut list) | ~25-30% over |
| H. Style: em-dashes and filler | 3 | exactly 1 em-dash; no AI filler transitions found |

**Two findings are submission-blocking on their own: A1 (the paper's primary contribution rests on numbers with no raw data on this machine) and A2 (Section 4.4 rests on a file that does not exist).**

---

## A. BLOCKING: claim exceeds evidence

### A1 (SEVERE, primary contribution). The n=20 cost-ranking numbers have no raw backing on disk; only n=10 exists.

Quoted, Abstract (line 27): *"Under nominal dynamics a CEM planner ranks its own 300 candidate action sequences informatively (mean per-seed Spearman rho 0.532, 95% CI [0.388, 0.676], n = 20 seeds). Under a damping shift, same frozen model and same planner, that ranking collapses to chance (rho 0.001, CI [-0.132, 0.134])"*

Same numbers recur at: Contributions bullet 1 (line 43, "rho 0.532 (R0) vs 0.001 (R2), n = 20 seeds per regime"); Section 4.1 table rows R0/R2/R2-with-chart (lines 94-96); the table footnote (line 98, "n = 20 seeds per regime"); the Figure 1 caption (line 104, "pooled over 20 seeds").

Source it is supposed to rest on: `CLAIMS_MATRIX.md` N3 ("Re-measured at n=20 seeds/regime... Raw data: `atlas_out/cost_ranking_R0/` and `atlas_out/cost_ranking_R2_v2/` (3 files each, seeds 0-9/10-14/15-19)") and `RESULTS_AUDIT.md` §12.

**Independently verified this pass, the audit's own citation is wrong about the files.** `atlas_out/cost_ranking_R0/` contains exactly one file, `cost_ranking_R0_seeds0-1-2-3-4-5-6-7-8-9.json`; `atlas_out/cost_ranking_R2_v2/` contains exactly one file, `cost_ranking_R2_seeds0-1-2-3-4-5-6-7-8-9.json`. Not 3 files each. Reading those files directly:

- R0 baseline: `"mean_of_per_seed_rhos": 0.5011629611936655`, `"ci95_of_mean_seed_rho": [0.27652, 0.72580]`, `"pooled_n": 3000` (= 10 seeds x 300 candidates).
- R2 baseline: `"mean_of_per_seed_rhos": -0.07196586376157219`, `"pooled_n": 3000`. R2 chart: `-0.05103495078619711`.

These are exactly `RESULTS_AUDIT.md` §2(d)'s n=10 numbers, which are L5. The n=20 numbers exist only as prose in `RESULTS_AUDIT.md` §12 / `CLAIMS_MATRIX.md` N3, describing seeds 10-19 run on a remote GPU volume whose artifacts are not present locally.

Evidence level defensible right now: **L1 (procedure specified and asserted), not L4/L5.** The n=10 version is L5 and fully defensible.

Fix (choose one, do not paper over): (i) recover the seeds 10-19 JSON files, place them in those directories, and have someone re-run the recomputation before submission; or (ii) report the n=10 numbers throughout (rho 0.501, CI [0.277, 0.726]; rho -0.072, CI [-0.243, +0.099]; chart -0.051), which are independently verified, and drop "n = 20" everywhere including the Figure 1 caption. Note that option (ii) also weakens the sentence *"indistinguishable from chance, with a tight CI centred on zero rather than merely including it"* (line 100): at n=10 the R2 CI is [-0.243, +0.099], which is not centred on zero and is not tight, so that sentence must be rewritten, not just renumbered.

Related: the regret figures (8.5px / 88.1px / 92.3px) and the top-decile deltas (+28.2 / -15.7 / -8.5px) in the same table are verified at L5 **at n=10 only** (`RESULTS_AUDIT.md` §2(d)). Presenting them under an "n = 20 seeds per regime" footnote attributes them to a sample they were not computed on.

### A2 (SEVERE). Section 4.4's entire result rests on a file that does not exist on this machine.

Quoted, Section 4.4 (line 142): *"Over 20 paired R2 planning episodes with a 3-chart library (identity 45.0%, R2-chart 50.0%, R1-chart 45.0%), a per-episode oracle reaches 60.0% against 46.7% for uniform random selection: a **spread of 13.3pp, bootstrap 95% CI [3.3, 25.0]**, clearing the 10pp denominator threshold we pre-registered as the condition for reporting a normalised-recovery score at all. The R1 chart is the unique winner on 2 of 20 episodes, so this is not a single-episode artifact."*

Source: `CLAIMS_MATRIX.md` N8, `RESULTS_AUDIT.md` §11, `REDTEAM.md` N8/RQ1 update, `PAPER_DRAFT_NOTES.md` §5 and §7(a), all four cite `atlas_out/e0_chart_r1_on_r2/ln_act_R2.jsonl` (20 episodes).

**Independently verified this pass:** `atlas_out/e0_chart_r1_on_r2/ln_act_R2.jsonl` does not exist (Read returns "File does not exist"). `atlas_out/e0_chart_r1_on_r2_smoke/ln_act_R2.jsonl` exists and is **empty (0 bytes)**. A content search across `atlas_out/` for per-episode records containing `"episode": 19` returns 25 files, none of them an R1-chart-on-R2 planning file. The two other arms in the row (identity 45.0%, R2-chart 50.0%) *are* backed: `e0_v3_planning_dataset_baseline/baseline_R2.jsonl` and `e0_v3_planning_dataset_ln_act/ln_act_R2.jsonl`, both verified at L5 in `RESULTS_AUDIT.md` §7(j-1).

This is the exact slot in which `RESULTS_AUDIT.md` §7(j-2) previously found a **fabricated** data point (the baseline array duplicated as a stand-in for chart_R1, reproducing the old numbers 4-for-4). Pass 3 says the real evaluation was run and downloaded; the downloaded artifact is not here. Until it is, the paper cannot distinguish "real run whose file was misplaced" from "the same fabrication recurring."

Evidence level defensible right now: **L0-L1 for the 13.3pp spread, the CI, the 60.0%/46.7% rates, and the "unique winner on 2 of 20 episodes" sentence.** The 2-chart version (oracle 50.0%, random 47.5%, spread +2.5pp, CI [0.0, +7.5]) is L5 and is the only version currently defensible.

Fix: locate and restore `e0_chart_r1_on_r2/ln_act_R2.jsonl` and have the bootstrap recomputed locally before this paragraph ships. If it cannot be restored, **cut Section 4.4 entirely** rather than downgrading it: the 2-chart fallback (2.5pp, CI touching zero) does not clear the paper's own pre-registered 10pp bar and therefore supports the opposite of what Section 4.4 says.

### A3. Section 4.4's CI is presented as a conventional significance statement; it is structurally one-sided.

Quoted (line 142): *"a **spread of 13.3pp, bootstrap 95% CI [3.3, 25.0]**"*

Source: `RESULTS_AUDIT.md` §7(j-3) proves that per-episode `d_i = oracle_i - random_i >= 0` by construction for any library at any sample size, so every bootstrap resample mean is non-negative and the interval can never contain a negative value. "CI excludes zero" therefore carries far less inferential weight here than the notation implies.

Evidence level: the arithmetic is whatever A2 leaves it at; the *interpretation* is over-strong regardless.

Fix: one clause disclosing the non-negativity floor, e.g. that oracle-minus-random is non-negative episode-wise by construction so the interval is one-sided and should be read as an effect-size range, not a hypothesis test.

### A4. Section 4.3's expansion sentence asserts a commit count whose per-chunk records do not exist, and reads as a system claim.

Quoted (line 138): *"The expansion primitive behaves consistently with the same picture: 0 charts committed under the appearance shift, 3 committed under the dynamics shift, through the real fit-then-verify path with the verification chunk confirmed disjoint from the fitting data."*

Source: `RESULTS_AUDIT.md` §1 and §8 both confirm `atlas_out/e2_R2_cellB_q1/` contains **only** `e2_summary.json` and no `e2_episodes.jsonl`; the "3 charts committed" figure is explicitly **L4-incomplete** and "cannot be independently re-derived from per-chunk decisions in this directory, because those per-chunk decisions were never saved." The disjointness half *is* clean (`CODE_AUDIT.md` §9.3, `REDTEAM.md` N9 Attack 2).

Evidence level: **L4 (summary only), not L5**, for "3 committed"; the "0 committed" for the appearance shift is L4 as well.

Fix: attach the provenance caveat (per-chunk commit log for the 3-commit run was not retained; only the run summary survives), or move the count to Appendix F where the caveat can sit next to it. Section 3 already scopes this to the primitive, so the wording risk is modest, but the number's status must not be silently upgraded.

### A5. The Cell B routing numbers are stated as measured fact; the audit rates them L4, not L5.

Quoted (line 132): *"On the two-chart cell isolating dynamics shift from appearance shift, UMF reaches 0.833 against S-dyn's 0.570."*

Source: `RESULTS_AUDIT.md` §2(h) explicitly flags this sub-claim: "I did not re-derive these three summary numbers from the per-episode `e2_episodes.jsonl` in this pass... flagging as **L4, not L5**, for this specific sub-claim."

Fix: low-cost. The raw file `atlas_out/e2_R2_posthysteresis/e2_episodes.jsonl` exists (1944 records), so this can simply be recomputed and upgraded to L5. Otherwise no wording change is strictly required, but the coordinating session should know this number has not been re-derived from raw records the way 60.3%/36.5% has.

### A6. The S-dyn baseline is repeatedly described as an "appearance-similarity baseline." It is a dynamics fingerprint.

Quoted, Abstract (line 27): *"where an appearance-similarity baseline reaches 36.5% and degenerates to one fixed adapter."* Also Contributions bullet 3 (line 45): *"where an appearance-similarity baseline does not."*

The draft's own Section 4.3 table (line 128) defines it correctly: *"S-dyn (action-conditioned latent-delta cosine similarity)."* `CLAIMS_MATRIX.md` C1 and `REDTEAM.md` N6 Attack 2 both call it a **dynamics-fingerprint** baseline; `LITERATURE_AUDIT.md` §4 groups fingerprint-style routing under input/feature similarity but nothing in the audit calls S-dyn an appearance metric.

Why this matters at claim strength: the headline in Section 4.3 is "does the fitness signal discriminate dynamics from appearance?" If the competitor is relabelled as an appearance-based method, beating it on a dynamics-vs-appearance task looks near-tautological and the baseline looks like a strawman. The real, stronger result is that a *dynamics*-fingerprint baseline fails degenerately.

Evidence level: the 36.5% is L5; the *characterization* is unsupported by any audit row.

Fix: use "dynamics-fingerprint baseline" or "action-conditioned latent-delta similarity baseline" in the abstract and contributions, matching the table.

### A7 (lesser). "Dominated by replay" is a field-level generalization built on two abstract-only reads.

Quoted, Section 2 (line 58): *"Continual model-based RL is otherwise dominated by replay on reconstruction-based backbones [arXiv:2607.19749; arXiv:2606.27374]"*

Source: `LITERATURE_AUDIT.md` §8 tags both papers `[SECONDARY, abstract only]` and characterizes them as papers a reviewer "could reasonably ask why this isn't discussed," not as evidence about the composition of the subfield.

Fix: soften to "recent continual model-based RL in this space includes replay-based approaches on reconstruction backbones [refs]", or cite them as a nod without the "dominated" quantifier.

### A8 (lesser). The excluded-episode sentence conflates two different episode sets.

Quoted, Section 4.2 (line 117): *"**The excluded episodes are not a random subsample**: they are the small-displacement ones, 100% of which succeeded in both arms"*

`RESULTS_AUDIT.md` §2(b): baseline drops 8 episodes (6,7,31,42,43,46,86,98), all 8 successful; chart drops 6 (7,31,41,43,86,89), all 6 successful. Those are different sets with partial overlap. "100% of which succeeded in both arms" implies one set evaluated under both arms.

Fix: "the dropped episodes in each arm (8 and 6 respectively) succeeded in every case."

### A9 (lesser). "Consistently," risks presenting N1 and N3 as independent corroboration.

Quoted, Abstract (line 27): *"Consistently, a 10.7k-parameter adapter that lowers the model's latent prediction error changes nothing measurable..."*

`REDTEAM.md` N3 Attack 3 is explicit: "N1 and N3 must not be presented as independently corroborating evidence, that would double-count one phenomenon," and `PAPER_DRAFT_NOTES.md` §2 says "State this relationship explicitly." Section 4.2 does gesture at it ("which is exactly Section 4.1's mechanism"), and Section 5 does not. The abstract's "Consistently" leans the wrong way.

Fix: one clause somewhere in main text stating that under one-shot planning the ranking collapse *is* the episode outcome, so these are one mechanism observed as cause and as consequence, not two results.

### A10 (lesser). Regime reality is asserted; the gate that would have tested it never ran.

Quoted, Section 3 (line 78) and Appendix D (line 216): *"R1 sets pusher-block friction to 2.0 and R2 sets space damping to 0.5, both unset in the shipped environment."*

`CLAIMS_MATRIX.md` P-4: "**L2 for reality**... Gate G4 (the direct empirical check) still never run." `EXPERIMENT_STATUS.md` §4: G4 is "**NOT RUN**, the only acknowledged skipped gate... Regime reality is instead argued from the separate design analysis."

The draft has indirect empirical support (the R0-vs-R2 planner divergence, the R1-vs-R2 routing differences), which is arguably stronger than G4 would have been. But nothing in the paper says the regimes' behavioural reality was argued analytically rather than gate-verified.

Fix: one clause in Appendix D, or fold into Section 5's "Other scope limits."

---

## B. BLOCKING: anonymization

### B1 (MUST FIX before conversion). Every section ends with an HTML provenance comment naming internal audit files, and the header carries a note addressed to the drafting process.

Locations: lines 5-21 (TEMPLATE NOTE / LENGTH STATUS block), 29, 48, 60, 80, 144, 162, 170, 263.

Examples of what they contain: `research_audit/PAPER_DRAFT_NOTES.md §1`, `research_audit/RESULTS_AUDIT.md §2(a)`, `research_audit/REDTEAM.md`, `research_audit/PROPOSAL_CODE_ALIGNMENT.md items A, B, F`, `E2_RESULTS.md`, `CLAUDE.md §0.1`, `atlas/score.py (UMF definition)`, plus "LaTeX preamble confirmed by the authors as..." and "LENGTH STATUS (for the coordinating session, not the paper)".

These are internal project file paths and an internal working-process naming convention. They are invisible in rendered Markdown and would be invisible as LaTeX `%` comments, but they are exactly the class of artifact that survives a careless Markdown-to-LaTeX conversion or a `pandoc` pass into a visible block, and they persist in the source if source is ever shared. Line 144's comment is the most sensitive: it narrates which numbers were corrected relative to internal documents, including the phrase "N8's fabricated proxy replaced with the real 20-episode evaluation."

Fix: delete all nine comment blocks at conversion time. Keep the provenance mapping in a **separate** non-submitted file. This is a hard gate before PDF generation.

### B2 (advisory, clean). Prose-level anonymization checked and found clean.

Checked the whole draft, main text and all eight appendices, for: author names, institution names, acknowledgments, repository or project URLs, "our code at", GPU model / cluster / cloud-provider names, and internal experiment identifiers. Findings:

- No author or institution names anywhere. No acknowledgments section. No URLs of any kind.
- No compute-provider identification. The cloud GPU provider used for the runs (named in `RESULTS_AUDIT.md` §12) appears nowhere in the draft.
- Appendix H line 261's `[NEEDS CONFIRMATION: total compute used... stated in a form that does not identify the hardware or provider]` is the correct instinct. Recommendation: report a GPU-hour total and a generic accelerator class only.
- No internal experiment IDs (E0/E1/E2/E4, RQ0-RQ4, N1-N10, Cell A-D) leak into prose; the draft consistently uses descriptive names. This was done well.
- The project's own codename does not appear. Third-party model/paper names (DINOv2, Push-T, JEPA-WM, Lambert, Grimm, Singh, Vakalis, Li, Narendra, DEN, CN-DPM, MBCD, CLARE, DPCore, WorMI, ShiftEx, Dynamic TMoE) are all citations of others and are not an anonymization issue.

### B3 (advisory). "Charts" is an internal term used in prose.

Line 37: *"a library of persistent adapters ("charts")"*, and throughout. This is the project's internal vocabulary, not identifying on its own, but it is the term that would match this project if a reviewer searched for it. Judgment call; not blocking. If the group has any prior public artifact using "charts" for this construct, drop the term and say "adapters."

---

## C. Unsourced claims

### C1. The 80.0% contact fraction is attributed to n=20.

Quoted, Section 4.1 footnote (line 98): *"n = 20 seeds per regime, 300 candidates per seed; contact occurred in 80.0% of candidate rollouts in all three rows."*

`RESULTS_AUDIT.md` §2(d) verifies 80.03% for all three rows **at n=10**. No source exists for the contact fraction at n=20. Subsumed under A1 if the n=20 issue is resolved; otherwise the footnote is unsourced as stated.

### C2. Appendix F's `q = 3` column reports measured zeros for runs that were never made.

Quoted, Appendix F table (lines 236-238), column *"Charts committed at q = 3"*: values `0`, `0`, `0`, with the R0/uncorrupted row's `q = 1` cell honestly marked "not run."

The prose immediately after (line 240) reveals these are inferred, not measured: *"At q = 3 nothing commits in any condition, because three consecutive strikes at the observed 15.7% per-chunk rate is a ~0.4% event."* `EXPERIMENT_STATUS.md` §2 says the same thing as a probabilistic argument. No `q=3` E2 run appears in `RESULTS_AUDIT.md` §1's inventory.

Fix: mark the `q = 3` column as inferred (e.g. "0 (expected, ~0.4% event; not run)"), or drop the column and keep the prose argument. As printed, a reader will take three measured zeros as evidence.

### C3. The Narendra citation has no verified entry in the literature audit at all.

Quoted, Section 2 (line 58): *"it goes back to multiple-model adaptive control [Narendra et al. 1997]... [NEEDS CONFIRMATION: exact Narendra reference and year; the audit records this line as spanning roughly 1992-2003 without pinning one paper.]"*

The draft's own flag is correct and honest. For the record: `LITERATURE_AUDIT.md` contains **no** entry for Narendra in any of its `[VERIFIED]` / `[SECONDARY]` / `[SEARCHED, NOT FOUND]` lists; the only trace is `CLAIMS_MATRIX.md` C1-novelty's parenthetical "(Narendra 1992-2003, MBCD 2021)". So the status is weaker than "needs the exact year pinned": no one has read any Narendra paper for this project.

Fix: either have someone verify one specific reference, or cite the line generically ("multiple-model adaptive control, from the 1990s") without a specific paper, and lean on MBCD/Alegre 2021 (which *is* `[VERIFIED]` from the local PDF, AAMAS 2021, the draft's venue is correct) as the concrete precedent.

### C4. The receding-horizon robustness argument has no citation at all.

Quoted, Section 5 (line 156): *"receding-horizon replanning is what standard control-theoretic reasoning says buys robustness to model error. We found no empirical citation quantifying this for CEM-planned latent world models, so we offer it as reasoned inference."*

The hedge exactly matches `LITERATURE_AUDIT.md` §9 Point 3, which is good. But that same point recommends citing a standard MPC/RHC reference (Kwon & Han or equivalent) generically for the feedback-robustness rationale. The draft cites nothing.

Fix: add the generic MPC/RHC citation. Keep the "reasoned inference" hedge exactly as written.

### C5. Appendix B's four non-headline UMF values are outside anything `RESULTS_AUDIT.md` recomputed.

Quoted, Appendix B (line 202): *"`ln_act` x R2 0.336, `lora4` x R2 0.329, `full` x R2 0.728, `ln_act` x R1 0.285, `lora4` x R1 0.288."*

`RESULTS_AUDIT.md` §2(c) independently recomputed only the `ln_act` x R2 sweep values (0.3357 / 0.30229 / 0.26776). The other four are traceable to `PROPOSAL_CODE_ALIGNMENT.md` H.3's table (0.3286 / 0.7280 / 0.2845 / 0.2876), which reads them off `results.json` files rather than recomputing them from raw records: **L4, not L5**. Not a problem: Appendix B already labels them "descriptive, not... a capacity ordering," and correctly carries the budget-confound and `full` x R1-never-trained disclosures from H.3. Recorded here only so the coordinating session knows these five numbers sit one level below the headline ones. **No fix required.**

---

## D. Overclaiming by exclusion

### D1 (SERIOUS). The second physics regime's planning result is missing entirely from the paper, and its point estimate is negative.

`CLAIMS_MATRIX.md` **N10** (added 2026-08-27, L5): first-ever R1-regime planning comparison, N=40 paired, baseline 70.0% (28/40) vs chart 60.0% (24/40), **delta -10.0pp, 95% CI [-27.5, +7.5]**, McNemar p=0.388, 12/40 discordant. Raw data verified present on this machine: `atlas_out/e1_baseline_vs_chart_R1/ln_act_R1.jsonl`. `PAPER_DRAFT_NOTES.md` §3 instructs explicitly: *"Include both regimes' numbers in the write-up rather than only the R2 headline."*

The draft never mentions R1 planning at all. Section 4.2, the abstract, and Appendix E all report only R2. N10's own framing in the claims matrix is that it "undermin[es] any reading of the R2 result as 'the chart is neutral/harmless', the direction isn't stable across regimes."

Why this is overclaiming-by-exclusion rather than a mere omission: the draft's negative result is stated as a flat, benign null (-1.0pp), and the accompanying scope framing ("this is evidence about one-shot open-loop planning, not about whether adaptation can help in general") reads more charitably to the method than the full two-regime picture supports.

Fix: add the R1 row to Appendix E and one sentence to Section 4.2 giving the N=40 R1 numbers and stating that the non-significant direction flips between regimes, consistent with noise around a true null.

### D2 (SERIOUS). The G-1 disclosure is present in main text but incomplete on the named-experiment facts.

What the draft does say, Section 5 (line 152): *"**No continual-learning experiment has been run.** Every number here comes from independent episodes in one fixed regime with an offline-trained adapter, or from offline routing accuracy over pre-collected trajectories. There is no stream, no regime revisit, and no retention, recall or forgetting measurement. The controller composing SELECT, REFINE and EXPAND has never executed end to end and REFINE has never run in production. For a workshop on continual world models this is the gap that matters most, and we state it rather than calling the experiment pending."*

**Verdict: substantially satisfied.** It is in main text, not the appendix; it is the first item under Limitations; it names the never-executed controller and the never-executed REFINE; Section 3 (line 76) independently repeats it; the abstract's last sentence and Contributions bullet 4 both flag it; the Conclusion (line 168) closes on it. This is handled about as well as the draft could handle it.

**What is still missing** (`CLAIMS_MATRIX.md` G-1, `EXPERIMENT_STATUS.md` §1 and §5, `RESULTS_AUDIT.md` §5):

- It never says that the routing-in-the-loop experiment the project itself declared "THE GATE" **produced zero episodes beyond smoke scale** (`e1_smoke` = 5 lines, `e1_verify` = 1 line; the specified 180 episodes do not exist). Section 4.4's "Whether any router captures it is untested" is the closest the draft gets, and it reads as a scoping choice rather than as an experiment that was never run.
- It never says that E3/E4/E5 produced **zero episodes of any kind** (`atlas_out/` contains no `e3*`/`e4*`/`e5*` directory, confirmed by direct listing in `RESULTS_AUDIT.md` §5).
- Section 5's ladder paragraph (line 154) says the 7-arm ladder has wiring defects but never says it has produced **no episodes**; a reader could infer it ran and returned artifacts.

Fix: three clauses. "The routing-in-the-loop evaluation we designed as the project's decision gate has never been run beyond a smoke test." "The continual stream, the expansion ladder and the cross-policy diagnostic have produced zero episodes." Attach "which has never produced an episode" to the ladder sentence.

### D3. E1's second, unresolved structural blocker is omitted.

`EXPERIMENT_STATUS.md` §1, E1 row: "**A second, still-unresolved blocker remains regardless:** at `num_act_stepped=6` one replan covers a whole 30-step episode, leaving no room for E1's specified 'two warmup replans then route' structure (implementation plan §7.0a flags this as needing 'a real decision', never made)."

Section 4.4's *"Whether any router captures it is untested"* (line 142) makes this sound like a budget/time decision. It is a protocol incompatibility: the routing experiment as specified **cannot be run** at the planner configuration every number in the paper uses.

Fix: one clause in Section 4.4 or Section 5 stating that the routing experiment as specified is incompatible with the one-replan protocol used throughout, so it is not merely unrun.

### D4. The 3-chart routing headline carries an unresolved hysteresis caveat that the draft discloses only for the 2-chart case.

Appendix A's note on `m` (line 192) is excellent and covers K=2 exactly as `CODE_AUDIT.md` §6.1 / `RESULTS_AUDIT.md` §10 prove. But `REDTEAM.md` N6 Attack 3 leaves the K=3 case open: "SURVIVES with an open, unresolved caveat... `RESULTS_AUDIT.md` explicitly did not re-simulate how often the K=3 condition actually binds in the 720-record confusion-matrix data. **Outcome: flag as unverified rather than clean.**"

The 60.3%/36.5% headline in the abstract, Contributions bullet 3, and Section 4.3 is a K=3 result. Appendix A's note ends with "Two-chart results in this paper should be read as pure argmin routing" and says nothing about three-chart results.

Fix: one sentence in Appendix A: at K=3 the margin can take intermediate values and is non-trivial but permissive; how often it binds in the reported run has not been re-simulated.

### D5. The reported UMF numbers are optimistically biased by a two-way split, and the draft discloses only half of it.

Section 4.2 (line 119) discloses the shared fixed 8-trajectory selection set across the sweep. What it does not disclose: `PROPOSAL_CODE_ALIGNMENT.md` H.2 (corroborated in `REDTEAM.md` N1 Attack 3) finds the **same** 8-trajectory set is used both for early-stopping checkpoint selection (consulted up to 80 times at `--eval-every 25 --patience 5`) and as the final reported `eval_umf`. That is a two-way split reported as held-out; it biases *every* UMF number in the paper, including Appendix B's table and the 0.336 anchor, not just the monotone trend.

Fix: extend the Section 4.2 caveat, or add one line to Appendix B: the reported held-out UMF is measured on the same set that selected the checkpoint, so it is optimistically biased.

### D6. The evaluated artifact is trained in a different regime from the method the paper describes.

Section 3 (line 74) describes REFINE as "one SGD step on the selected chart." Section 3 (line 76) says charts were "trained offline." What is not stated: the offline training is up to 2000 SGD steps (Appendix A does give this, to the draft's credit) over what amounts to ~100 model transitions seen 2000 times (`PROPOSAL_CODE_ALIGNMENT.md` H.1). `CLAIMS_MATRIX.md` open question 4 asks directly whether this is "consistent with a method whose stated premise is on-the-fly adaptation," and warns "the capacity result may not be measuring what the method needs."

Fix: one clause acknowledging that the evaluated charts were produced by a heavyweight offline procedure, not by the one-step online refinement the method specifies, so the planning results characterize the adapter class rather than the online adaptation rule.

### D7. Two smaller unresolved items absent from Limitations.

- `CLAIMS_MATRIX.md` P-1 / `CODE_AUDIT.md` §2.5: `chart.restore_()` does not restore pretrained weights for `ln_act` or `full` (only `lora4`); currently masked by full-overwrite in practice, but it falsifies the literal "charts are disjoint / restore is exact" property. `REDTEAM.md` S-2 adds that the rewritten G1 only covers unrefined charts and so cannot detect this. The draft's Section 3 describes apply-and-restore without caveat.
- Section 5's gate sentence (line 160) covers the two vacuous gates well. It does not mention that G1's coverage is limited to unrefined charts.

Fix: fold both into the existing "Other scope limits" sentence, or into Appendix A/B. Low priority relative to D1-D3.

---

## E. Related Work fairness

### E1. Five competing module-library methods are collapsed into "input or feature similarity"; two of them are more sophisticated than that.

Quoted, Section 2 (line 58): *"nearby module-library methods route on input or feature similarity instead [DPCore; CLARE; WorMI; Dynamic TMoE; ShiftEx]."*

`LITERATURE_AUDIT.md` §4, from direct reads:
- **CLARE**: routing is "autoencoder reconstruction-error argmin per layer (not raw 'feature similarity' per se)"; the audit says "reconstruction error is more precise than feature similarity."
- **Dynamic TMoE**: the router is "GRU-based, not a plain memoryless router"; the audit calls the plain "memory router" label an underspecification.
- DPCore, WorMI and ShiftEx are characterized correctly (feature statistics distance, prototype Wasserstein distance, MMD signature matching).

Fix: either name the mechanisms in one compressed clause ("stored feature statistics, prototype distance, MMD signatures, or per-layer reconstruction error"), or add "input, feature or reconstruction-error similarity." Appendix G already handles CLARE and Dynamic TMoE precisely on the expansion axis, so this is a main-text-only imprecision.

### E2. Lambert et al. is characterized slightly weaker than the audit found it.

Quoted, Section 2 (line 54): *"Lambert et al. [2020] found near-zero-to-weak correlation between validation log-likelihood and episode reward in classical MBRL"*

`LITERATURE_AUDIT.md` §6 item 1 (VERIFIED, direct read): "near-zero-to-weak correlation... (Fig. 3, Spearman-like rho as low as -0.06 to **0.59** depending on dataset)". The top of that range is moderate, not weak.

Fix: "correlation ranging from near-zero to moderate depending on dataset," or give the range. Minor, but this is the paper's own lineage citation and understating it slightly inflates the novelty of the dissociation the draft elsewhere correctly calls a replication.

### E3. The substrate paper's appendix is characterized correctly but without the number that makes it a fair comparison.

Quoted, Section 2 (line 56): *"The substrate paper's appendix [JEPA-WM] already asks whether embedding-prediction loss proxies for success rate and answers positively at a between-model, across-training-epoch granularity. We do not claim that question was open."*

This is the correct application of `LITERATURE_AUDIT.md` §5 / `REDTEAM.md` C3: the softening the audit asked for has been done, and done well. The only fairness note: the audit records the actual strength (Spearman ~0.82-0.86 on Push-T, ~0.70-0.81 on Wall, ~0.39-0.47 on Metaworld). "Answers positively" is accurate; giving one number would be fairer to the competitor and costs three words. Optional.

**Nothing else in Related Work makes a competitor sound weaker than the audit found it.** The Appendix G table's uniform "No" column is fully supported at L6 by `LITERATURE_AUDIT.md` §3 (all six papers read directly). The Singh/Vakalis/Li characterizations match §7 and §6 exactly, including the load-bearing horizon-vs-shift distinction. The verdi nod in Appendix G is accurate and appropriately anonymized. MBCD's venue is correctly given as AAMAS, applying the audit's §3 correction.

---

## F. Hype language

The draft is unusually disciplined here. Searched for: novel, novelty, powerful, robust, significant/significantly, state-of-the-art, dramatic(ally), substantial(ly), clearly, strongly, remarkable, crucial. Results:

- **"powerful," "robust," "state-of-the-art," "dramatically," "substantially," "significant," "significantly" do not appear anywhere in the draft.** "Significant" is notably absent even where the draft reports p-values, which is correct.
- "novelty" appears twice (lines 58, 160), both scoped: *"We claim novelty only for transferring that mechanism to a frozen high-dimensional visual latent space"* (supported at L6 by `LITERATURE_AUDIT.md` §1 / `REDTEAM.md` C1) and *"Our novelty claim rests on a serious recency-weighted search, not a proof of absence"* (exactly the audit's own hedge). No fix.

Three items to flag:

### F1. "well-powered" (line 44, Contributions; line 27's abstract equivalent).
Quoted: *"**A well-powered, explicitly scoped null on adapter-driven planning gain,** 44.0% vs 43.0% at N = 100 paired episodes."* The number and CI are in the same sentence, and the scope qualifier is present, so this passes the rule. But `REDTEAM.md` N1 restricts the power claim to one-shot open-loop planning specifically, which the draft honors in Section 4.2. Keep as is; do not let it drift to "well-powered null on adaptation" in any edit.

### F2. "The planner is confident and wrong, not uncertain." (line 102)
This is a rhetorical claim resting on **3 seeds** (`cost_ranking_R2_converged`, verified L5 in `RESULTS_AUDIT.md` §2(e)). The draft does state "3 of 3 seeds" in the preceding sentence, which satisfies the number-in-the-adjacent-sentence rule, and it honestly discloses that the chart arm's spreads are wider on two of three seeds. Borderline acceptable. If anything is cut for length, this sentence is a candidate.

### F3. "tight CI centred on zero" (line 100).
Quoted: *"under R2 it is indistinguishable from chance, with a tight CI centred on zero rather than merely including it."* Backed by the number in the table, **but only under the n=20 numbers challenged in A1**. At the on-disk n=10 values (rho -0.072, CI [-0.243, +0.099]) this sentence is false. Contingent on A1's resolution.

---

## G. Length and section budget

Target: 2-4 pages excluding references, appendices uncounted. Drafter self-reports ~3,384 words main text, ~5 pages. Sections 1-6 must lose roughly 25-30%.

**Ordered cut list** (largest saving first, cheapest first within a tier). This mostly agrees with the drafter's own ladder in lines 12-17, with two changes noted:

1. **Figure 2 (line 134, confusion matrices) to Appendix.** Keep 60.3% / 36.5% and the S-dyn column-0 counts (61, 62, 65) inline; those two facts carry the whole argument. ~0.4 page. Largest single win.
2. **Section 4.4 to Appendix, or cut outright.** Note this is a *change from the drafter's ladder*, which does not list 4.4 at all. Given A2 (the underlying file is missing), this paragraph should not be in main text in its current form regardless of length. If the file is recovered, two sentences inline plus the detail in appendix. ~120 words plus the risk reduction.
3. **Related Work paragraph 2 (line 56) compressed to one sentence per 2026 paper.** ~120 words. The horizon-vs-shift distinction from Singh must survive the compression in some form (see do-not-cut list).
4. **Section 3's "What was actually executed" (line 76) to a three-item inline list.** ~100 words. The three protocol names and the "no planner in the loop" / "never executed end to end" clauses must survive verbatim.
5. **Section 5's "Other scope limits" (line 160) to Appendix, with a one-line pointer.** ~90 words. Note that D4/D6/D7 above want *more* material here, so the appendix version should absorb those rather than being trimmed.
6. **Section 4.1's converged-CEM paragraph (line 102) trimmed to one sentence**, with Appendix C carrying the per-seed table (it already does). ~60 words.
7. **Appendix A's note on `m` stays in the appendix** (it already is) but the Section 3 SELECT sentence's "subject to a hysteresis margin" can drop the mechanism detail. ~15 words.

**Must NOT be cut, removing any of these re-creates an overclaim the draft exists to avoid:**

- Section 4.2's two scope conditions paragraph (line 115), both the one-open-loop-plan condition and the wrong-target-distribution condition. This is `REDTEAM.md` N1 Attacks 1 and 2; without it the null reads as "adaptation does not work."
- Section 4.3's "Scope" paragraph (line 136). This is `CLAIMS_MATRIX.md` S-7 / `REDTEAM.md` N6 Attack 1; without it the routing result is silently claimed as a planning-competence result while measuring regime labels.
- Section 5's first two paragraphs (lines 152, 154): the continual-learning gap and the expansion-primitive-versus-controller distinction. `CLAIMS_MATRIX.md` G-1 and `REDTEAM.md` N9 Attack 1 / C2. These are the highest-cost omissions possible at this venue.
- Section 4.2's n=92/94 non-random-exclusion disclosure (line 117). `RESULTS_AUDIT.md` §2(b).
- Section 4.2's shared-selection-set caveat (line 119). `REDTEAM.md` N4 Attack 1.
- Section 2's "We replicate this rather than discover it" (line 54) and "We do not claim that question was open" (line 56). `LITERATURE_AUDIT.md` §6 calls the first one binary: get it right or the paper is scoopable in the abstract's first sentence.
- Section 4.3's "an earlier measurement of that cell is not reproducible from surviving records and we make no causal claim about what moved it" (line 132). `REDTEAM.md` N7, the sharpest kill in the audit; restoring any causal attribution here is the single most credibility-costly edit available.
- Appendix A's note on `m` (line 192): appendix, so it costs no page budget; do not delete it to tidy the appendix.

---

## H. Style: em-dashes and filler

### H1. Exactly one em-dash in the entire draft, and it is a table placeholder.
Line 110: the `Frozen baseline` row's three empty comparison cells contain U+2014.
Fix: replace with `n/a` or a plain hyphen. The prose is otherwise em-dash-free, which is a notable achievement given the source documents are saturated with them.

### H2. En dashes appear only in numeric ranges, acceptable.
Lines 3 ("2-4 pages"), 98/130 ("83-104"), 102 ("4.7-8.3px"), 202, 220 ("~13-17%"), 236 ("0.0-2.2%"). These are U+2013 in range position, standard typography, not em-dashes. No fix needed unless the user's rule is meant to cover all long dashes, in which case convert to "to" or a hyphen.

### H3. No AI-flavored filler transitions found.
Searched for: Moreover, Furthermore, Notably, Importantly, Additionally, In conclusion, It is worth noting, key insight, crucial. **Zero hits in prose.** Two mild transitions worth a look, neither a template phrase:
- "Consistently," opening the abstract's fourth sentence (line 27), flagged substantively at A9, not merely stylistically.
- "Nor is this an unconverged-pool artifact." (line 102), a real argumentative move, not filler. Keep.

---

## Appendix: what was independently verified against disk in this pass

- `atlas_out/cost_ranking_R0/`: 1 file, seeds 0-9, `mean_of_per_seed_rhos` = 0.5011629611936655, CI [0.27652, 0.72580], `pooled_n` 3000. (Contradicts CLAIMS_MATRIX N3's "3 files each" and the n=20 numbers.)
- `atlas_out/cost_ranking_R2_v2/`: 1 file, seeds 0-9, baseline `mean_of_per_seed_rhos` = -0.07196586376157219, chart -0.05103495078619711, `pooled_n` 3000.
- `atlas_out/e0_chart_r1_on_r2/ln_act_R2.jsonl`: **does not exist**.
- `atlas_out/e0_chart_r1_on_r2_smoke/ln_act_R2.jsonl`: exists, **0 bytes**.
- `atlas_out/e0_chartR1_on_R2/`: no readable episode data.
- Content search across `atlas_out/` for per-episode records with `"episode": 19` returns 25 files; none is an R1-chart-on-R2 planning file. `e0_v3_planning_dataset_baseline/baseline_R2.jsonl` and `e0_v3_planning_dataset_ln_act/ln_act_R2.jsonl` (the two backed arms of Section 4.4's triple) are present, as is `e1_baseline_vs_chart_R1/ln_act_R1.jsonl` (the N10 result the draft omits, D1).
