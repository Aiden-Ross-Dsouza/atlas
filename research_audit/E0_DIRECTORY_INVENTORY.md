# ATLAS — E0 training-output directory inventory (FIX_SPEC.md D4)

Generated 2026-08-28. Every directory under `atlas_out/` whose name starts
with `e0`, classified **current** / **superseded** / **smoke**, so the
release bundle is defensible (a reviewer downloading the artifact tarball
should be able to tell which directories back a paper number and which are
scratch).

**Evidence labels used below:**
- **[disk]** — verified directly from this session (`ls`, mtimes, file
  contents, or an independent recomputation from the raw JSONL/JSON).
- **[doc]** — inferred from cross-referencing `E0_RESULTS.md`,
  `ATLAS_SUMMARY.md`, `CLAUDE.md`, `HANDOFF.md`, or `FIXLOG.md`'s own
  citations of this directory. Not independently re-run this pass.
- **[name]** — inferred from the directory's own name (e.g. `*_smoke`,
  `*_pre_regime_fix_*`).

| Directory | Verdict | Basis |
|---|---|---|
| `e0` | **SUPERSEDED** | **[doc]** `FIXLOG.md`'s "Deliberately NOT fixed" table: `atlas_out/e0/results.json` records `params=26/12/69` — parameter *group* counts, an old bug, explicitly flagged historical/superseded there. **[disk]** Full 9-chart set (3 kinds x 3 regimes) + `T5.md`, but no `.jsonl` planning records — training-only, and no `test`/`val` seed disjointness (pre-A4). |
| `e0_pre_regime_fix_2026-08-22` | **SUPERSEDED** | **[name]+[doc]** Name states it directly; predates the 2026-08-23 regime-design fix (`REGIME_DESIGN_REVIEW.md`) that retargeted R1/R2 onto friction/damping. Every chart here was trained against the OLD (physically-dead, mass-based) R1/R2 definitions. |
| `e0_calib_fric2` | **SMOKE** | **[name]+[disk]** Single `baseline_R1.jsonl`, no chart file — a friction-calibration probe (`REGIME_DESIGN_REVIEW.md`/`E0_RECOVERY_PLAN.md`'s friction-saturation sweep), not a trained chart or a reported result. |
| `e0_chart_r1_on_r2` / `e0_chart_r1_on_r2_smoke` / `e0_chartR1_on_R2` | **SMOKE** | **[disk]** Cross-regime diagnostic (an R1-trained chart evaluated on R2 data) — a robustness probe, not a pre-registered E0 cell. `*_smoke` is explicitly named as such; the other two are the same diagnostic at different scales. |
| `e0_contact_check`, `e0_contact_check2` | **SMOKE** | **[disk]** `e0_contact_check` is empty; `e0_contact_check2` has charts but exists to validate the contact-rate/aimed-walk fix (`ACTION_SAMPLING_REVIEW.md`, Part A-viii), not to produce a reported chart. |
| `e0_earlystop_smoke`, `e0_realdata_smoke`, `e0_scripted_regress_smoke`, `e0_t3_smoke` | **SMOKE** | **[name]** All four are explicitly named `*_smoke` — pipeline-validation runs for the early-stopping / real-data-replay / scripted-regression / T3 mechanisms, not reported results. |
| `e0_planning` | **SUPERSEDED** | **[disk]** Only 1 record in `baseline_R1.jsonl` (confirmed by `red_flag_sweep.py`, D1) — a single-episode smoke of the planning harness before it was fixed, mtime 2026-08-24 (pre-rollout-fix, invalidated per `CLAUDE.md`'s CRITICAL banner). |
| `e0_planning_n100` | **CURRENT** | **[disk]** The canonical N=100 baseline-vs-`ln_act` result. Independently re-verified this session (D3): baseline 24/100 (24.0%) knock-aways, `ln_act` 22/100 (22.0%), mean progress +25.08px/+32.43px — matches `E0_RESULTS.md` §A.1 exactly. This is the file `analyze_n100.py`/N1 are computed from. |
| `e0_planning_nas2` | **CURRENT (diagnostic)** | **[doc]** The `num_act_stepped=2` arm behind N5's confounded +10pp finding (Part C-2/E-D). Still current — it is the input to a flagged-as-confounded analysis, not a superseded number. |
| `e0_planning_sweep_60`, `e0_planning_sweep_100` | **CURRENT** | **[disk]** Merged-shard planning sweeps (verified clean by A13: 0 content mismatches, contiguous). Back the N4 training-size trend. |
| `e0_train_sweep_60`, `e0_train_sweep_100` | **CURRENT** | **[disk]** The chart-*training* half paired with the two directories above (same 60/100-trajectory sweep, training not planning). |
| `e0_v3_baseline_R0` | **CURRENT** | **[disk]** R0 baseline planning eval, used for the cross-regime baseline comparison (R0 vs R1 vs R2) `CLAUDE.md` §0.1 cites. |
| `e0_v3_dataset` | **CURRENT, flagged (A9)** | **[doc]** The 20-trajectory `ln_act`/R2 chart anchoring N4 and the chart behind N1. `results.json` has no seed manifest recorded (A9's finding) — current and reported, but its held-out status rests on prose, not an artifact. Not superseded — no better artifact exists. |
| `e0_v3_hybrid` | **CURRENT, flagged (OPUS #11)** | **[doc]** The hybrid-collector chart. `OPUS_REMAINING_TASKS.md` #11 flags its conclusion as confounding two variables at once. Current on disk, not retrained/replaced — the caveat is about interpretation, not about the artifact being stale. |
| `e0_v3_planning_dataset_baseline`, `e0_v3_planning_dataset_ln_act`, `e0_v3_planning_hybrid_ln_act` | **CURRENT** | **[disk]** N=20 planning evals for the dataset-collector baseline/chart and the hybrid-collector chart — inputs to Part A-iv's dissociation table (UMF vs SR ranking) and the T5 capacity table. |
| `e0_v4_full`, `e0_v4_lora4` | **CURRENT, flagged (A10)** | **[doc]** The R2 `full`/`lora4` chart pair. `A10`/`PROPOSAL_CODE_ALIGNMENT.md` H.3 flag these as trained at unrecorded, unmatched budgets (`lora4` documented as retrained once at a reduced budget after an OOM). Current — no matched-budget retrain has landed (A10 option (b), not done this pass). |
| `e0_v4_planning_full`, `e0_v4_planning_lora4` | **CURRENT** | **[disk]** N=20 planning evals for the above two charts — the `full`=0.200 SR / `lora4`=0.400 SR entries in Part A-iv's table. |
| `e0_v5_closed_loop`, `e0_v5_planning_closed_loop` | **CURRENT, S-5 caveat applies** | **[doc]** The closed-loop-collector chart and its planning eval. S-5 (verified this session, D2): the collection instrument itself has four stacked train/deploy mismatches (9x CEM budget gap, opposite replan-frequency extreme, non-reactive-across-kinds collection). The chart/eval files are current (nothing supersedes them), but `E0_RECOVERY_PLAN.md`'s "clean rejection" framing of this result is not supportable — report the result, not the framing. |
| `e0_v6_R1` | **CURRENT** | **[disk]+[doc]** The canonical R1 chart pair (`ln_act`, `lora4`), touched by A11's `params_trainable` fix this session. This is the "🟢 CURRENT" pair `CLAUDE.md` §0.1 points to for the current R1 UMF/planning numbers. |
| `e0_v6_R1_results.json` (loose file, no directory) | **CURRENT** | **[disk]** A loose top-level results file duplicating/summarizing `e0_v6_R1/results.json` — also received A11's `params_trainable` fix. Not a directory; listed for completeness since D4 asked for every `e0*` artifact to be characterised. |

## Summary for the release bundle

**Safe to keep as the reported-results set:** `e0_planning_n100`,
`e0_planning_sweep_{60,100}`, `e0_train_sweep_{60,100}`, `e0_v3_baseline_R0`,
`e0_v3_dataset`, `e0_v3_hybrid`, `e0_v3_planning_*`, `e0_v4_*`, `e0_v5_*`,
`e0_v6_R1*`, `e0_planning_nas2`.

**Recommend excluding or clearly marking `superseded/` in any public release
tarball:** `e0`, `e0_pre_regime_fix_2026-08-22`, `e0_planning` (1-record
smoke), and everything named `*_smoke`/`*_check*`/`*_on_r2*`
(`e0_calib_fric2`, `e0_chart_r1_on_r2*`, `e0_chartR1_on_R2`,
`e0_contact_check*`, `e0_earlystop_smoke`, `e0_realdata_smoke`,
`e0_scripted_regress_smoke`, `e0_t3_smoke`) — none of these back a number in
`E0_RESULTS.md`/`ATLAS_SUMMARY.md`/`PAPER_DRAFT.md`.

This inventory is **read-only** — no directory was moved, renamed, or
deleted as part of D4. Any reorganisation is a release-packaging decision
for a later phase, not this one.
