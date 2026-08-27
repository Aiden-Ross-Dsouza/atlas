# ATLAS — FIXLOG

**Last updated: 2026-08-27 — file created during Phase 0. No fixes applied yet.**

## What this file is

The complete record of every code change made while executing
`SUBMISSION_PLAN.md`. Written for a future Claude Code session with zero memory
of the conversation that produced it.

This repository's operating rule (`CLAUDE.md`, Research Audit section) is **no
silent fixes**: a bug found during the audit is documented and left in place,
because fixing and re-running silently destroys the provenance of every number
already on disk. `SUBMISSION_PLAN.md` Part B is the one authorised exception —
each fix in that register was documented *before* being applied, and each gets
a row here recording what actually changed.

**If a number in `atlas_out/` was produced before a fix that affects it, that
number is superseded, not deleted.** The superseding is done in place in the
owning results document with a dated banner, and the row below records which
document was amended.

## How to read a row

| Field | Meaning |
|---|---|
| **ID** | The register ID from `SUBMISSION_PLAN.md` Part B (A1-A15, B1-B14, C1-C10, D1-D10) |
| **Date** | When the fix landed |
| **Defect** | What was wrong, at `file:line` |
| **Change** | What was actually done — not what was intended |
| **Claims affected** | Claim IDs from `CLAIMS_MATRIX.md` (N1-N10, C1-C4, RQ0-RQ4, P-1..P-5, S-1..S-9, L-1, G-1) |
| **Re-run needed** | yes/no, and if yes, which artifacts were regenerated |
| **Verification** | The named assertion, and **whether it was confirmed to fail before the fix**. An assertion that passes both before and after is vacuous and does not count — this is exactly the defect that made gates G2 and G5 worthless in this repo. |
| **Verified by** | Who ran the check. A sub-agent's "done" report is not verification; the orchestrating session must re-run it. |

---

## Fixes applied

### Phase 1 Tier A — Stage 1 (code + zero-GPU analysis only), 2026-08-27

Dispatched: A5, A6, A7, A8, A11, A12, A13 (full) + A1, A2, A3, A4 (code only,
re-runs deferred to Stage 2) + A14, A15 (docs). No GPU/Modal re-runs performed.

---

### A1 — Hysteresis margin inert at K=2

- **Date:** 2026-08-27
- **Defect:** `atlas/router.py:96-97` — `relative_gap = (current-best)/spread`,
  `spread = max(valid)-min(valid)`. At K=2 a non-argmin incumbent *is* the max,
  so `relative_gap = (max-min)/(max-min) = 1.0 > m=0.05` always; router is pure
  argmin, hysteresis never binds.
- **Change:** normaliser changed to the incumbent's own score:
  `relative_gap = (current-best)/current if current > 0 else 0.0`
  (`atlas/router.py:94-100`). `m=0.05` unchanged (Sec1.7 non-negotiable — only
  the normaliser changed). In-code comment rewritten, cites FIX_SPEC A1. Also
  added `relative_gap` / `hysteresis_held` to the returned info dict (feeds A3).
- **Claims affected:** C-1, C-2, N9 (E2 routing accuracy + hysteresis binding
  rate). Numbers not yet regenerated — Stage 2.
- **Re-run needed:** yes (Stage 2) — all E2 configs, new directory names.
- **Superseded numbers:** none yet; E2 SR / Cell-B numbers in `ATLAS_SUMMARY.md`
  §4.5 and `E2_RESULTS.md` will need supersede banners after the Stage 2 re-run.
- **Verification:** unit assertion (`scratchpad/assert_a1.py`): 2-chart score
  dict, incumbent worse by 1% must HOLD, by 20% must SWITCH.
  Failed before fix: **yes** — BEFORE: `by 1% -> idx 0 (SWITCH)`, `by 20% -> idx 0`,
  ASSERTION FAIL. AFTER: `by 1% -> idx 1 (HOLD)`, `by 20% -> idx 0 (SWITCH)`,
  ASSERTION PASS.
- **Known side effect (flagged, not a defect):** for routers whose incumbent
  score can be <= 0 (`sdyn`: negative cosine sim) the `else 0.0` branch makes
  `relative_gap = 0 < m`, i.e. always HOLD. This is the spec's exact prescribed
  formula; its effect on `sdyn` routing accuracy will be visible in the Stage 2
  E2 re-run and must be checked there.
- **Verified by:** atlas-fixer agent (Stage 1). Orchestrator must re-run.

### A2 — `maybe_expand()` picks the incumbent on the verification chunk

- **Date:** 2026-08-27
- **Defect:** `atlas/expand.py:131-133` — incumbent = argmin UMF over the
  held-out `next_encoder_output`, then candidate must beat it on that same
  chunk: a look-ahead advantage for the incumbent.
- **Change:** incumbent now = argmin summed UMF over `self._deficit_chunks` via
  new helper `_argmin_umf_over_chunks()` (`atlas/expand.py`). Verification
  comparison still on the held-out chunk. Old-style index still computed and
  stored in `Expander._last_probe_debug` (`incumbent_idx`, `old_incumbent_idx`,
  `incumbent_changed`) for audit. `run_e2.py` writes it per episode.
- **Claims affected:** N9 (charts committed / rejected in E2).
- **Re-run needed:** yes (Stage 2, combined E2 re-run) — will report how many
  E2 decisions change.
- **Superseded numbers:** none yet.
- **Verification:** unit assertion (`scratchpad/assert_a2.py`): fake `compute_umf`
  where chart A is best on the deficit chunk and chart B best on the held-out
  chunk; incumbent must switch from B(idx1) to A(idx0).
  Failed before fix: **yes** — BEFORE: `argmin over: heldout chunk, idx=1`,
  ASSERTION FAIL. AFTER: `{'incumbent_idx': 0, 'old_incumbent_idx': 1,
  'incumbent_changed': True}`, ASSERTION PASS.
- **Verified by:** atlas-fixer (Stage 1).

### A3 — Per-chunk expander state never logged

- **Date:** 2026-08-27
- **Defect:** `atlas_out/e2_R2_cellB_q1/` etc. hold only summaries; per-chunk
  expander decisions behind "N charts committed" were never written (N9 stuck
  at L4; `PAPER_FACT_CHECK` D4 "unverified").
- **Change:** `scripts/run_e2.py` per-episode loop now appends a
  `record_type="expansion"` record with `strikes`, `probe_outcome`,
  `relative_gap`, `hysteresis_held`, `committed`, `incumbent_debug`,
  `library_size`. Router info dict extended (see A1) to carry
  `relative_gap`/`hysteresis_held`.
- **Claims affected:** N9; `PAPER_FACT_CHECK` D4.
- **Re-run needed:** yes (Stage 2) — every E2 config + a new q=3 run, new dir
  names.
- **Verification:** deferred to Stage 2 (needs the E2 re-run to produce the
  JSONL). Code path exercised indirectly by `scratchpad/assert_a2.py`
  (`_last_probe_debug` populated).
- **Verified by:** atlas-fixer (Stage 1) — code only.

### A4 — Reported `eval_umf` measured on the checkpoint-selection set

- **Date:** 2026-08-27
- **Defect:** `scripts/run_e0.py` — the 8 val trajectories (seed_offset 10_000)
  served both early stopping (consulted up to ~80x) and the final reported UMF.
- **Change:** added `--num-test-trajs` (default 8) drawn at `seed_offset=20_000`
  (disjoint from train offset 0 and val offset 10_000). `eval_umf` now reported
  from the test set; `val_umf` reported alongside so bias = `eval_umf - val_umf`
  is measurable. `e0_seed_manifest.json` now has a `test` block. Both the fresh
  and the `[Resume]` result branches updated.
- **Claims affected:** N4 and every E0 UMF number.
- **Re-run needed:** yes (Stage 2) — **re-score only** (forward-only, no
  planner) the already-saved charts; no retraining. Not run in Stage 1.
- **Verification:** deferred to Stage 2 (assertion: manifest shows three
  disjoint seed ranges; `test_umf` + `val_umf` both in results.json). Code
  compiles; `py_compile` clean.
- **Verified by:** atlas-fixer (Stage 1) — code only.

### A5 — `analyze_n100.py` never computed N1's statistics

- **Date:** 2026-08-27
- **Defect:** script never called `paired_bootstrap` / `mcnemar_paired`; N1's CI
  and McNemar p were not reproducible from `analysis_n100.json`.
- **Change:** added `paired_success_stats()` (imports both functions from
  `atlas.stats` **unmodified**), wired as `report["A5_paired_success_stats"]`.
  Regenerated `atlas_out/analysis_n100.json` (pre-fix copy saved to
  `scratchpad/analysis_n100.PREFIX.json`).
- **Claims affected:** N1.
- **Re-run needed:** done — `atlas_out/analysis_n100.json` regenerated (zero-GPU).
- **Superseded numbers:** none — values match what the paper already states.
- **Verification:** regenerated file contains
  `delta_ci95_pp: [-9.0, 7.0]` and `mcnemar_p: 1.0`. Confirmed:
  `n_paired=100, sr_baseline=0.44, sr_chart=0.43, delta=-0.01`.
- **Verified by:** atlas-fixer (Stage 1).

### A6 — Oracle−random CI structurally one-sided

- **Date:** 2026-08-27
- **Defect:** `d_i = oracle_i - random_i >= 0` by construction; no bootstrap
  resample can be negative, "CI excludes zero" is vacuous.
- **Change:** added **new** function `oracle_gap_permutation(per_chart_successes,
  n=10_000, seed=0)` to `atlas/stats.py`. Existing functions untouched. Returns
  `(observed_gap, p_value, null_distribution_summary)`.
- **SPEC DEVIATION (flagged):** FIX_SPEC A6 says "permute chart labels within
  each episode". That is a literal no-op for this statistic (max_c / mean_c over
  a column are invariant under relabelling values already present). Implemented
  instead the non-degenerate null the assertion actually requires: each chart's
  outcome vector shuffled independently across episodes (destroys cross-chart
  specialisation, preserves marginals hence random SR). Documented in the
  function docstring.
- **Claims affected:** N8.
- **Re-run needed:** no (analysis function; callers add it when they report N8).
- **Verification:** assertion — identical-charts synthetic library must give
  `gap ~ 0`, `p ~ 1.0`.
  Failed before fix: **yes** — BEFORE: `ImportError: cannot import name
  'oracle_gap_permutation'`. AFTER: identical-charts `observed_gap=0.0000
  p=1.0000`; N8 real data `observed_gap=0.1333 p=1.0000` (null mean 0.382 —
  i.e. the real 3-chart library's oracle gap is *below* chance-independent
  charts: the charts are redundant, not complementary. Reportable finding).
- **Verified by:** atlas-fixer (Stage 1).

### A7 — Partial-Kendall p-value invalid

- **Date:** 2026-08-27
- **Defect:** `scripts/analyze_n100.py::partial_kendall` — Kendall's analytic
  null ignores the estimated OLS coefficients, so `partial_p` is
  anticonservative.
- **Change:** `partial_p` is now a permutation test on the residuals (10_000
  permutations of `y_resid`, `p = (#|tau_perm| >= |tau_obs|| + 1)/(n+1)`). Point
  estimates retained and still reported (`partial_tau`). Analytic value kept as
  `partial_p_analytic` for comparison.
- **Claims affected:** the within-episode UMF/quality dissociation claim
  (`ATLAS_SUMMARY.md` §4.1 bullet, `E0_RESULTS.md` 2026-08-27).
- **Re-run needed:** done — `analysis_n100.json` regenerated.
- **Superseded numbers:** point estimates unchanged (-0.358, -0.374). The
  p-values reported alongside them change: baseline analytic 4.4e-7 ->
  permutation 1.0e-4; chart analytic 9.6e-8 -> permutation 1.0e-4 (floor of the
  permutation resolution — still far below 0.05). Permutation p >= analytic p in
  both cases, as A7 requires. `ATLAS_SUMMARY.md` §4.1 quotes
  `p=4.4×10⁻⁷ / p=9.6×10⁻⁸` — flagged for a Phase 6 supersede note; not edited
  in Stage 1 (only A14/A15 were in doc scope).
- **Verification:** regenerated JSON: `partial_p (permutation) = 9.999e-05` for
  both arms, `>= partial_p_analytic` (4.4e-7, 9.6e-8). PASS.
- **Verified by:** atlas-fixer (Stage 1).

### A8 — `sr_by_bucket` silently drops episodes above 300px

- **Date:** 2026-08-27
- **Defect:** `scripts/analyze_n100.py::sr_by_bucket` — `pd.cut(bins=(0,80,120,
  300))` sends overflow to NaN, dropped by `groupby(observed=True)`; per-bucket
  n need not sum to episode count.
- **Change:** `edges` is now finite lower edges only `(0,80,120)` with an
  implicit `+inf` top bucket (`"120+px"`); added
  `assert out["n"].sum() == len(df)` with a diagnostic message.
- **Claims affected:** the "SR by bucket shows no hidden win" claim
  (`ATLAS_SUMMARY.md` §4.1).
- **Re-run needed:** done — `analysis_n100.json` regenerated.
- **Superseded numbers:** none — current N=100 data has 0 episodes >300px, so
  the top bucket is `120+px` with n=24 and SR unchanged (0.083). Assertion
  passes (53+23+24 = 100). It would have fired on a dataset with an overflow
  episode.
- **Verification:** ran `analyze_n100.py --all`; no assertion error; bucket
  label now `120+px`, n sums to 100.
- **Verified by:** atlas-fixer (Stage 1).

### A11 — Artifacts record the discredited LoRA parameter count

- **Date:** 2026-08-27
- **Defect:** `atlas_out/e0_v4_lora4/results.json`,
  `atlas_out/e0_v6_R1/results.json`, `atlas_out/e0_v6_R1_results.json` record
  `params=10292640`; `Chart.n_params()` sums `_params`, which at construction
  holds the 12 frozen full base matrices. Real trainable count is 118,176
  (`PAPER_DRAFT.md` App. B).
- **Change:** added `Chart.n_trainable_params()` (`atlas/chart.py`) — for
  `lora4` sums only `.lora_A`/`.lora_B`; otherwise `== n_params()`.
  `scripts/run_e0.py` now records `params_stored` + `params_trainable` in both
  result branches. Regenerated the three JSON artifacts adding both keys plus an
  `_audit_note_A11` (via `scratchpad/a11_regen.py`); original `params` field
  kept verbatim.
- **Claims affected:** N3 / the E0 capacity comparison; `PAPER_DRAFT.md` App. B
  (already correct).
- **Re-run needed:** no (metadata only; no metric recomputed).
- **Superseded numbers:** `params=10292640` in the three artifacts is now
  accompanied by `params_trainable=118176`. `E0_RESULTS.md` §"Limitations" item
  1 already documents this discrepancy — no new banner needed there.
- **Verification:** assertion — `n_trainable_params()` returns 118,176 for
  `lora4` and 10,764 for `ln_act`, and differs from `n_params()` for `lora4`.
  Failed before fix: **yes** — BEFORE: `AttributeError: 'Chart' object has no
  attribute 'n_trainable_params'`; `n_params()` = 10,292,640 for lora4.
  AFTER (real `dino_wm_pusht` predictor): `ln_act n_params()=10764
  n_trainable_params()=10764`, `lora4 n_params()=10292640
  n_trainable_params()=118176`, `full 20800884/20800884`. ASSERTION PASS.
- **Verified by:** atlas-fixer (Stage 1).

### A12 — Dead CLI dispatches that silently no-op

- **Date:** 2026-08-27
- **Defect:** `scripts/make_tables.py` offered `--table T5` but `main()`
  dispatched only T1/T2 (silent no-op); `scripts/make_figures.py` accepted only
  `{F1,F2}` while `atlas/plots.py` defines `umf_traces`/`crosspolicy`/`umf_vs_sr`
  that nothing dispatched.
- **Change:** implemented `make_t5()` (E0 capacity table from
  `{e0_dir}/results.json`; raises `FileNotFoundError` if that file is absent;
  `ΔUMF`/`Success`/`% of full` render `—` when their inputs are missing rather
  than vanishing). Added `--e0-dir` and the T5 dispatch. In `make_figures.py`
  added `make_s1/make_s2/make_s3` and `S1/S2/S3` choices: an explicit `--fig Sx`
  raises a clear error when its data is missing; `--all` reports and continues.
- **Claims affected:** none directly (tooling correctness).
- **Re-run needed:** no.
- **Verification:** assertion — every value in each `choices` list produces
  output or raises. `--table T5` -> writes `atlas_out/e0/T5.md` (9 rows).
  `--fig S1` -> `FileNotFoundError` (no e4 log). `--fig S2` -> `FileNotFoundError`
  (no e5 matrix). `--fig S3` -> writes `atlas_out/figures/S3_umf_vs_sr.pdf`.
- **Verified by:** atlas-fixer (Stage 1).

### A13 — `merge_planning_shards.py` does not check contiguity

- **Date:** 2026-08-27
- **Defect:** merging rejects duplicates but an upstream-missing episode passes
  silently.
- **Change:** added `--episode-start` (default 0); after sort, assert merged
  episode indices `== range(episode_start, episode_start+n)`, else `ValueError`
  naming the missing indices.
- **Claims affected:** any planning number built from a merged shard set (N1,
  N4, sweep tables).
- **Re-run needed:** no (guard only; existing merges verified clean 2026-08-27).
- **Verification:** synthetic test — shards `[0,1,2,3]`+`[5,6,7]` (gap at 4) ->
  `ValueError: ... missing [4]`. Shards `[0,1,2,3]`+`[4,5,6,7]` -> merges 8
  episodes cleanly.
- **Verified by:** atlas-fixer (Stage 1).

### A14 — `ATLAS_SUMMARY.md` overstates converged-CEM spread

- **Date:** 2026-08-27
- **Defect:** `ATLAS_SUMMARY.md` §4.3 — "std ... is 3.8–8.3px — a tight
  cluster". Real range over all six seed/kind cells is 3.77–27.15px (`ln_act`
  seeds 1 and 2 are 17.9 and 27.2). `E0_RESULTS.md` hedged correctly.
- **Change:** restated with the full range; removed "tight cluster"; added a
  dated AUDIT CORRECTION BANNER after the doc header.
- **Claims affected:** the §4.3 converged-CEM knock-away claim (framing only —
  the knock-away direction is unaffected).
- **Re-run needed:** no.
- **Superseded numbers:** "3.8–8.3px" -> "3.77–27.15px" in `ATLAS_SUMMARY.md`
  only. `E0_RESULTS.md` unchanged (already correct).
- **Verification:** text edit; grep confirms old string gone, banner present.
- **Verified by:** atlas-fixer (Stage 1).

### A15 — `ATLAS_SUMMARY.md` Cell-B before/after has no raw records

- **Date:** 2026-08-27
- **Defect:** `ATLAS_SUMMARY.md` §4.5 — "pre-fix +55.6pp -> post-fix +26.3pp"
  before/after; the "+55.6pp" figure has no surviving raw records.
- **Change:** dropped the before/after framing; the paragraph now states only
  the current 2-chart decisive-cell numbers (UMF 0.833 vs S-dyn 0.570) with an
  inline audit note. Covered by the same dated banner as A14.
- **Claims affected:** C-1 framing.
- **Re-run needed:** no. (Note: A1/A2/A3 Stage 2 E2 re-run will itself
  supersede the 0.833/0.570 numbers — a fresh banner will be added then.)
- **Superseded numbers:** "+55.6pp -> +26.3pp" removed from `ATLAS_SUMMARY.md`.
- **Verification:** text edit; grep confirms "55.6" gone from the file.
- **Verified by:** atlas-fixer (Stage 1).

---

*Original placeholder: Phase 0 (setup) complete; Phase 1 Stage 1 landed
2026-08-27. Stage 2 (E2 + E0 re-scores) not yet dispatched.*

<!--
Row template — copy this, do not reformat it:

### <ID> — <one-line summary>

- **Date:** YYYY-MM-DD
- **Defect:** <what was wrong> (`path/to/file.py:LINE`)
- **Change:** <what was actually done>
- **Claims affected:** <IDs, or "none">
- **Re-run needed:** <no | yes — artifacts regenerated: ...>
- **Superseded numbers:** <none | which doc got a dated banner, and which numbers>
- **Verification:** <assertion>. Failed before fix: <yes/no/n-a>. Passes after: <yes/no>.
- **Verified by:** <orchestrating session / agent name + independent re-run>
-->

---

## Discovered, not fixed

### Phase 1 Tier A Stage 1 (2026-08-27)

- **`ATLAS_SUMMARY.md` §4.1 quotes analytic Kendall p-values** (`p=4.4×10⁻⁷`,
  `p=9.6×10⁻⁸`) that A7 replaces with permutation p-values (`~1.0e-4`, still
  well below 0.05, direction/point-estimates unchanged). Not edited in Stage 1
  — doc scope was A14/A15 only. Flag for the Phase 6 `PAPER_DRAFT.md` /
  `ATLAS_SUMMARY.md` supersede pass. (`ATLAS_SUMMARY.md:111`)
- No new code defects found while implementing Stage 1. The `sdyn` +
  `current_score <= 0` interaction noted under A1 is a consequence of the
  authorised A1 formula, not a separate bug — recorded there.

## Deliberately NOT fixed

Recorded so a future session does not "helpfully" fix them and invalidate the
archive.

| Item | Why it stays |
|---|---|
| `atlas/score.py::umf` | Produced every UMF number on disk. Additions only (a new test-split flag is additive; the metric itself is frozen). |
| `atlas/stats.py`'s existing functions | `paired_bootstrap`, `mcnemar_paired`, `normalised_recovery` were independently verified clean (`CODE_AUDIT.md` §3.1-3.3). New functions may be added alongside them; existing ones are not touched. |
| `scripts/run_e0_planning.py`'s planning loop | Produced every planning-success number on disk. |
| `atlas_out/e0/results.json`, `e0_pre_regime_fix_2026-08-22/` | Record `params=26/12/69` (parameter *group* counts, an older bug). Historical and already superseded — left as the archive of what was actually run. |
| The `full` × R1 chart | Never trained; a prior explicit decision not to. Its absence means the pre-registered "≥90% of full's gain" rule is *undefined* for R1, which must be reported as such, not as "not met". |
