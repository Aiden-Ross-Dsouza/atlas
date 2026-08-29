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

### Phase 2 — Gates, latent hazards, unfinished audit items, 2026-08-28

Dispatched: C1, C2, C3, C4 (scope-changed, see row), C5, C6, C7, C8, C9
(skip C10 per instruction — Phase 5 routes around E1's harness) + D1, D2,
D3, D4, D10 (D5-D7 already closed by SUBMISSION_PLAN.md Part A; D8 folds
into Phase 3, not this pass; D9 not attempted — lower priority, flagged
below).

### C1 — Gate G2 asserted nothing

- **Date:** 2026-08-28
- **Defect:** `scripts/smoke_gates.py` (pre-fix `gate_g2`) built `W`/`W'`
  from i.i.d. `torch.randn` (no learnable structure), over-refined via a
  hand-rolled `forward_pred` loop bypassing `_open_loop_rollout`, computed
  both UMFs, then `if umf_cx is not None and umf_c0 is not None: pass` —
  printed `PASSED` unconditionally regardless of the values.
- **Change:** rewrote `gate_g2` (and added helper `_g2_make_regime_chunk`) to
  build `W`/`W'` from two INDEPENDENT structured "regimes" (predictor-weight
  perturbations, same mechanism as `gate_g3a`), over-refine `cx` on `W` ONLY
  via the production `_open_loop_rollout`/`_make_z_ctxt` path (300 real Adam
  steps), then assert the leakage signature: the genuinely held-out score on
  `W'` must be strictly worse than the reference (leaked) score on `cx`'s own
  training window `W`. A reference-only `umf_cx_W` is computed purely to
  make this assertion possible, never used to make a decision.
- **Claims affected:** P-2, S-2 ("all headless gates pass").
- **Re-run needed:** no (gate script only).
- **Verification:** `python scripts/smoke_gates.py --gate G2` →
  `PASSED (umf_cx on training W=0.0056 [leaked, reference-only], umf_cx on
  held-out W'=0.7241, umf_c0 on held-out W'=0.5870)`. **Failed before fix:
  yes** — demonstrated via `scratchpad/g2_demo_leak.py`, which deliberately
  feeds `cx`'s score on its OWN training window `W` in place of the
  held-out `W'` score (simulating the leak this gate exists to catch):
  `leaked 'W-prime' score = 0.0058, reference training-window score =
  0.0058` → `AssertionError: G2 DEMO FAILED (expected): leaked score on
  'W-prime' (0.0058) <= reference training score (0.0058)`. Passes after:
  yes, on the correct (non-leaked) scoring path.
- **Verified by:** orchestrating session, direct run (both before/after
  shown side by side above).

### C2 — Gate G5 was a tautology

- **Date:** 2026-08-28
- **Defect:** `scripts/smoke_gates.py` (pre-fix `gate_g5`) called
  `paired_seed()` with `arm="atlas"` and `arm="frozen"` and asserted the two
  integers were equal — true by inspection, since `paired_seed()`'s `arm`
  parameter is never referenced in its body. No env was built, no seed was
  sampled.
- **Change:** rewrote `gate_g5` (+ new helper `_g5_build_and_reset`) to
  construct two REAL `PushTEnv(render_size=224, with_velocity=True)`
  instances at the seed `paired_seed()` produces, `env.seed(s); env.reset()`
  on each, and assert the raw init state vector, rendered visual
  observation, and `goal_pose` are all `np.array_equal` across the two
  independently constructed envs.
- **Claims affected:** P-2 (G5 pairing), S-2.
- **Re-run needed:** no.
- **Verification:** `python scripts/smoke_gates.py --gate G5` → `PASSED
  (seed=3612771448, init_state=[128. 327. 272. 338. ...], goal=[256. 256.
  0.785...])`. **Failed before fix: yes** — demonstrated via
  `scratchpad/g5_demo_fail.py`, calling `_g5_build_and_reset` at two
  DELIBERATELY mismatched seeds (111, 222): `AssertionError: G5 DEMO FAILED
  (expected): mismatched seeds produced different init states ([390. 414.
  312. 186. ...] vs [184. 95. 215. 114. ...])`. Passes after: yes, on paired
  seeds.
- **Verified by:** orchestrating session, direct run.

### C3 — G1 only tested unrefined charts, never `kind="full"`

- **Date:** 2026-08-28
- **Defect:** `gate_g1` looped over `("ln_act", "lora4")` only (no `"full"`),
  and only ever tested a chart that had never been refined — where
  `restore_()` and `apply_()` are trivially identical (see C4), so the
  restore check was vacuous for the interesting case.
- **Change:** extended the kind loop to `("ln_act", "lora4", "full")`, and
  added a second block per kind that fits a chart for real (20 real Adam
  steps via `_open_loop_rollout`/`_make_z_ctxt`, structured data), confirms
  the refinement measurably changed the predictor's output (sanity check,
  itself an assertion), then confirms the NEW `Chart.restore_pretrained_()`
  (C4) returns the REFINED predictor to bit-identical pretrained weights and
  output.
- **Claims affected:** P-1, P-4 (identity/apply-restore correctness).
- **Re-run needed:** no.
- **Verification:** `python scripts/smoke_gates.py --gate G1` → `PASSED`
  for all three kinds including the new refined-chart block. Not a
  "must-fail" row (C3 is a coverage extension, not a dead-gate rewrite);
  verified instead by inspecting that the refined-output sanity assertion
  (`torch.equal(out_frozen, out_refined)` must be False) actually fires as a
  real check, confirmed by temporarily setting `lr=0`/`steps=0` in a
  scratch run and observing it correctly raise (chart didn't move ->
  assertion would be vacuous) before restoring the real 1e-2/20-step config.
- **Verified by:** orchestrating session.

### C4 — `chart.restore_()` does not restore pretrained weights (scope-changed)

- **Date:** 2026-08-28
- **Defect:** `atlas/chart.py::Chart.restore_()` for `kind in
  {"ln_act","full"}` is literally `self.apply_(predictor)` — it RE-APPLIES
  this chart's own current `_params`, which is NOT the predictor's
  pretrained state once the chart has been refined
  (`update_from_predictor_()` mutates `_params` in place). Falsifies P-1's
  literal wording; `atlas/loop.py:247`'s comment ("restore predictor to
  chart's baseline weights") was false.
- **SCOPE DEVIATION from FIX_SPEC.md C4, per explicit task instruction:**
  FIX_SPEC prescribes renaming `restore_` to `reapply_`. **Not done** — 10
  production call sites (`atlas/loop.py`, `atlas/expand.py`,
  `scripts/run_e0.py`, ...) depend on the current name/behaviour, and a
  rename touches files that produced numbers already on disk. Instead:
  added `Chart.restore_pretrained_(predictor, pretrained_state)` as a NEW,
  additive method (`atlas/chart.py`); left `restore_()` completely
  unchanged; fixed only the false comment at `atlas/loop.py:247`.
- **Change:** `atlas/chart.py` — new method `restore_pretrained_()`. For
  `kind="lora4"` it delegates to `restore_()` (parametrization removal is
  already non-destructive over the pretrained base weight — genuinely
  correct already). For `kind in {"ln_act","full"}` it copies from a
  caller-supplied `pretrained_state` dict (e.g. `predictor.state_dict()`
  captured before any chart was ever applied). `atlas/loop.py:246-253`:
  replaced the false comment with an accurate one; `atlas_refine()`'s
  behaviour is UNCHANGED (still calls the old `restore_()`, since that is
  its documented, depended-upon contract).
- **Claims affected:** P-1 (identity/restore correctness, literal wording).
- **Re-run needed:** no.
- **Verification:** `scratchpad`-style unit test (paraphrased in this
  session's report): built a synthetic `nn.LayerNorm`-based predictor,
  applied+refined+updated a chart (mutating params away from pretrained),
  then: `restore_() == pretrained? False` (confirms the pre-existing defect
  is real, not already fixed elsewhere), `restore_() == chart's own
  (refined) values (re-apply)? True` (confirms the exact failure mode),
  `restore_pretrained_() == pretrained? True` (confirms the new method
  works). **Failed before fix: yes** (old `restore_()` genuinely does not
  restore pretrained weights, verified, not just asserted). Passes after:
  yes, via the new method.
- **Verified by:** orchestrating session, direct run.

### C5 — `base_env.step()` bypasses the wrapper, silent for VisualCorruption

- **Date:** 2026-08-28
- **Defect:** `atlas/harness.py:381`, `atlas/harness_e4.py:263`,
  `scripts/run_e0_planning.py:310`, `scripts/profile_episode.py:137`,
  `scripts/diagnose_cem_costs.py:119` all step `base_env` directly rather
  than the (possibly wrapped) `env`/`regime` object. Harmless for
  `PhysicsRegime` (never touches observations), but for `VisualCorruption`
  (a `gym.ObservationWrapper`) this silently skips `.observation()` — a
  corrupted planning run would measure against the CLEAN image while
  believing it corrupted, with no error.
- **Change:** added `atlas.regimes.assert_no_bypassed_corruption(regime)` —
  walks the wrapper chain and raises `RuntimeError` if a `VisualCorruption`
  (kind != "none") is present. Called once at entry in
  `atlas/harness.py::run_e1_episode`, `atlas/harness_e4.py::run_e4_episode`,
  and inside the shared `scripts/run_e0_planning.py::prepare_with_visual`
  helper (also covers `profile_episode.py`/`diagnose_cem_costs.py`, which
  both call `prepare_with_visual`).
- **Claims affected:** E2 Cell C/D validity (appearance-only shift).
- **Re-run needed:** no — E0/E1 never use `VisualCorruption` (confirmed:
  only `run_e2.py` constructs one), so no existing number is affected; this
  is a guard against a FUTURE silent failure mode.
- **Verification:** direct unit test — `assert_no_bypassed_corruption` on a
  `PhysicsRegime`-only chain: no raise. On
  `VisualCorruption(PhysicsRegime(...), "blur")`: `RuntimeError` raised with
  the expected message. Confirmed both branches.
- **Verified by:** orchestrating session, direct run.

### C6 — `lora4` online refinement raises (empty param list)

- **Date:** 2026-08-28
- **Defect:** `atlas/loop.py::atlas_refine` and
  `atlas/expand.py::_fit_candidate` both built the optimizer's param list as
  `[p for n, p in predictor.named_parameters() if n in chart._param_names]`.
  For `kind="lora4"`, `apply_()` REPLACES the base weight's
  `named_parameters()` entry with a parametrization, so the base name no
  longer appears there — the filter selects ZERO parameters, and
  `optim.Adam([])` raises `ValueError: optimizer got an empty parameter
  list`. `atlas/harness.py`'s offline path already avoided this (selects by
  `requires_grad` after enabling `lora_A`/`lora_B`).
- **Change:** both sites now branch on `chart.kind`/`candidate.kind ==
  "lora4"` and select params by name suffix (`"lora_A" in n or "lora_B" in
  n`), matching `harness.py:117-125`'s already-correct pattern. `ln_act`/
  `full` paths unchanged.
- **Claims affected:** any future `atlas` (online, verification-reachable)
  or `atlas_detect` arm run with `kind="lora4"` — currently E0's `lora4`
  charts are only ever refined offline (`harness.py`), so no number on disk
  used the broken path; this blocks it for Phase 3/5.
- **Re-run needed:** no.
- **Verification:** real `dino_wm_pusht` predictor, `kind="lora4"`. OLD
  filter: `old filter param count: 0` -> `optim.Adam([])` ->
  `ValueError: optimizer got an empty parameter list`. **Failed before fix:
  yes** (reproduced against the literal pre-fix filter, not a hypothetical).
  NEW: `atlas_refine(c, model, enc, actions, proprio_ctxt=prop)` ->
  `atlas_refine SUCCEEDED, loss= 2.952293872833252`. Passes after: yes.
- **Verified by:** orchestrating session, direct run (both shown side by
  side above).

### C7 — `diagnose_cem_costs.py` writes only at the end

- **Date:** 2026-08-28
- **Defect:** `scripts/diagnose_cem_costs.py::main` accumulated `per_seed`
  in memory across the whole `seeds x kinds` sweep and wrote the JSON output
  only once, at the very end (`out_path.write_text(...)` after the loop) —
  the documented cause of two lost dose-response runs (SUBMISSION_PLAN.md
  C7).
- **Change:** compute `out_path`/`incremental_path` (a sibling `.jsonl`)
  before the seed loop; after each seed's record is built, append it to
  `incremental_path` immediately (`open(..., "a")`); the final aggregated
  `.json` (with pooled statistics, which genuinely need all seeds) is still
  written once at the end, unchanged in content. A stale `.jsonl` from a
  prior attempt is deleted at the start of a fresh run so appends never mix
  two runs.
- **Claims affected:** none directly (tooling robustness) — protects E-A
  (Phase 4) and any future dose-response run using this script.
- **Re-run needed:** no.
- **Verification:** `python -m py_compile scripts/diagnose_cem_costs.py` ->
  clean. Logic verified by inspection (append happens immediately after
  `per_seed.append`, inside the seed loop, before the pooled-summary code
  that requires all seeds) — not re-run end-to-end this pass (requires GPU/
  live env), consistent with Phase 2's zero-GPU scope.
- **Verified by:** orchestrating session (compile-checked; not
  GPU-exercised — flagged below).

### C8 — `smoke_gates.py` loaded the REMOTE hub, not the patched local copy

- **Date:** 2026-08-28
- **Defect:** `scripts/smoke_gates.py::main` called
  `torch.hub.load("facebookresearch/jepa-wms", "dino_wm_pusht", ...)`
  without `source="local"` — resolved against the remote GitHub repo spec,
  not the patched local checkout at `HUB_PATH`
  (`hub/hub/facebookresearch_jepa-wms_main`) every production script uses.
- **Change:** `torch.hub.load(_HUB_PATH, "dino_wm_pusht", source="local",
  ...)`, matching `run_e0_planning.py`/`diagnose_cem_costs.py`'s pattern.
- **Claims affected:** S-2 (gate correctness against the actual production
  code path).
- **Re-run needed:** no.
- **Verification:** `python scripts/smoke_gates.py --all` after the change:
  model loads from `hub/hub/facebookresearch_jepa-wms_main` (confirmed by
  the `[INFO] Loaded encoder and predictor` line and no network HEAD
  request beyond the checkpoint URL check already present in production);
  G1/G2/G3b/G6 all PASS. (G3a intermittently fails — pre-existing,
  order-dependent flakiness independent of this fix, see "Discovered, not
  fixed" below.)
- **Verified by:** orchestrating session, direct run.

### C9 — `configs/atlas/*.yaml` loaded by nothing, two disagree with real defaults

- **Date:** 2026-08-28
- **Defect:** No script reads `configs/atlas/{default,e0,e1,e2,e4}.yaml`;
  `configs/regimes/pusht.yaml` already carried a "documentation only"
  header (per its own note), the `configs/atlas/*` files did not. Checked
  each against the real script argparse defaults: `e4.yaml` disagrees on
  TWO values — `iterations: 10` vs `scripts/run_e4.py`'s
  `CEM_ITERATIONS = 30` (the code comment there admits "cut to 10 ... after
  profiling" was decided but never applied), and `seeds: 1` vs `--seeds`
  default `3`. `default.yaml`, `e0.yaml`, `e1.yaml`, `e2.yaml` all matched
  their scripts' real defaults exactly.
- **Change:** added a "documentation only, nothing loads this" header
  (matching `configs/regimes/pusht.yaml:3-5`'s convention) to all five
  files, stating per-file whether a disagreement was found, and for
  `e4.yaml` naming both disagreements explicitly with `file:line`
  citations.
- **Claims affected:** none directly — these files are not cited as
  evidence anywhere; this closes a latent trap for a future session that
  might otherwise trust them as live config.
- **Re-run needed:** no.
- **Verification:** `grep -c` each script's real argparse defaults against
  each yaml's stated values (shown in this session's report); confirmed the
  two `e4.yaml` disagreements and confirmed no disagreement in the other
  four files.
- **Verified by:** orchestrating session.

### D1 — Red-flag sweep never run as a dedicated pass

- **Date:** 2026-08-28
- **Change:** new script `scripts/red_flag_sweep.py` (read-only, additive) —
  scans every `atlas_out/**/*.jsonl` (35 files) for (1) exact-zero variance
  on fields whose name suggests per-episode/per-seed variation, (2)
  per-group episode counts (grouped by arm/router/kind/regime/seed_run), and
  (3) None/NaN in every numeric field. Report written to
  `research_audit/RED_FLAG_SWEEP.md`, with a manual-triage section at the
  top explaining every finding.
- **Result:** all 35 files scanned; every raw finding traced to a documented
  mechanism on inspection (motion-gate `None`s, mixed per-seed-summary vs
  per-episode record types in `e2_*` files producing the "unequal group
  size" / "None episode" findings, and two smoke tests' deliberately fixed
  seeds). **No new, unexplained red flag found.** No `atlas_out/e4*`
  directory exists yet (E4/S2 has produced zero episodes), so the
  zero-variance-across-seeds check — the one that would have caught
  SUBMISSION_PLAN.md's A-ix/B12 Modal fake-multi-seed bug — could not be
  exercised against real E4 data; flagged in the report to re-run after
  Phase 5.
- **Claims affected:** none corrected (no new defect found); strengthens
  confidence in existing E0/E1/E2 JSONL data.
- **Re-run needed:** N/A (diagnostic script, re-runnable any time, cheap).
- **Verification:** script run, output inspected record-by-record for the
  representative case of each finding category (shown in this session's
  report).
- **Verified by:** orchestrating session.

### D2 — `CLAIMS_MATRIX.md` Section C's S-1..S-8 verified

- **Date:** 2026-08-28
- **Change:** `research_audit/CLAIMS_MATRIX.md` — updated the Section C
  summary table's `Verified`/`Status` columns (were all `L0`/`PENDING`) and
  added code/doc-cited verdicts for S-6, S-7, S-8 (the three the doc itself
  flagged as "not independently re-derived"; S-1/S-4/S-5 already had
  verdicts from a prior pass, retained; S-2 updated to reflect this
  session's C1/C2 gate fixes; S-3 retained).
- **Result:** S-6 (15pp bar invented) **CONFIRMED TRUE** —
  `grep -n "15pp" ATLAS_proposal_v7.md ATLAS_implementation_plan_v2.md`
  returns zero matches in both; only appears in `E0_RECOVERY_PLAN.md`. S-7
  (E2 correctness = regime label) **CONFIRMED TRUE** —
  `scripts/run_e2.py:262`'s `correct_idx` is a static per-cell config
  lookup, and `grep -n "GC_Agent\|CEMPlanner\|planner" scripts/run_e2.py`
  returns zero matches (no planner exists in this script at all). S-8 (two
  parallel sessions, relayed results) **CONFIRMED TRUE, with the documented
  nuance already in `HANDOFF.md` §3** — relayed items were each
  independently derived from raw logs once, by the producing session, just
  not cross-verified by the other session. S-2 updated to reflect gates G2/
  G5 are no longer vacuous (this session's C1/C2).
- **Claims affected:** S-1 through S-8 directly; all downstream claims that
  cite gate-passing status or "E0 is closed" framing indirectly.
- **Re-run needed:** no (documentation update, backed by direct code/file
  checks recorded above).
- **Verification:** every S-6/S-7/S-8 verdict above is a direct grep/read
  result, pasted into `CLAIMS_MATRIX.md`'s own row, not a restated
  assertion.
- **Verified by:** orchestrating session.

### D3 — OPUS item 1's N=100 knock-away claim re-verified from raw JSONL

- **Date:** 2026-08-28
- **Change:** independently recomputed knock-away counts and mean progress
  directly from `atlas_out/e0_planning_n100/{baseline_R2,ln_act_R2}.jsonl`
  (not via `analyze_n100.py`, to avoid trusting the same code path twice) —
  `knock_away = block_pos_diff > init_block_pos_diff` (matching
  `analyze_n100.py::knock_away_progress`'s own documented definition).
- **Result:** baseline 24/100 (24.0%), `ln_act` 22/100 (22.0%); mean
  progress baseline +25.08px, `ln_act` +32.43px. **Matches `E0_RESULTS.md`
  §A.1 exactly** ("baseline 24/100 (24.0%) ... `ln_act` 22/100 (22.0%) ...
  baseline +25.1px vs. `ln_act` +32.4px"). The N=20-to-N=100 gap
  `OPUS_REMAINING_TASKS.md` item 1 flagged as unchecked is closed: the N=100
  number is genuine, independently reproduced from raw records, not merely
  copied from the doc that reports it.
- **Claims affected:** the A.1 knock-away/mean-progress dissociation claim.
- **Re-run needed:** no.
- **Verification:** shown in full in this session's report (command +
  output).
- **Verified by:** orchestrating session, direct computation.

### D4 — Full inventory of `e0*` directories

- **Date:** 2026-08-28
- **Change:** new file `research_audit/E0_DIRECTORY_INVENTORY.md` —
  every `atlas_out/e0*` directory (29 directories + 1 loose file)
  classified current/superseded/smoke, each with an evidence label
  ([disk]/[doc]/[name]) and citation.
- **Result:** 4 directories SUPERSEDED (`e0`, `e0_pre_regime_fix_2026-08-22`,
  `e0_planning`, and implicitly anything under those regimes), 11 SMOKE
  (calibration/diagnostic/pipeline-validation runs, none backing a reported
  number), 15 CURRENT (several flagged with an existing caveat — A9's
  missing seed manifest, A10's unmatched budgets, OPUS #11's hybrid
  confound, S-5's closed-loop instrument problem — but not superseded, since
  no better artifact exists).
- **Claims affected:** release-bundle defensibility; no paper number
  changed.
- **Re-run needed:** no (read-only inventory; nothing moved/deleted).
- **Verification:** every verdict cross-checked against at least one of:
  `ls`/mtime, a direct D1/D3-style recomputation, or an existing FIXLOG/
  CLAIMS_MATRIX citation — shown inline in the inventory file itself.
- **Verified by:** orchestrating session.

### D10 — Assert the "never batch multiple episodes" invariant

- **Date:** 2026-08-28
- **Defect:** `atlas/harness.py:413-415`, `atlas/harness_e4.py:288-290`,
  `scripts/run_e0_planning.py:338-340` all do
  `enc["visual"].squeeze(0).squeeze(1).flatten(1, 2)` immediately after
  `world_model.encode(...)` — safe only because `visual_t`/`proprio_t`'s
  batch dim is always 1 by call-site construction (Part A-vi), an
  unasserted invariant.
- **Change:** added `assert visual_t.shape[0] == 1 and proprio_t.shape[0]
  == 1, f"D10: expected a single-episode batch (dim0=1), got ..."`
  immediately before the `.encode(...)` call at all three sites.
- **Claims affected:** none (the invariant already held everywhere checked;
  this converts a silent-corruption risk into a loud failure if it is ever
  violated).
- **Re-run needed:** no.
- **Verification:** `python -m py_compile atlas/harness.py
  atlas/harness_e4.py scripts/run_e0_planning.py` -> clean. Not exercised
  against a real multi-episode-batch input (none exists; that is the point
  of the assertion — it should never fire in current usage).
- **Verified by:** orchestrating session (compile-checked; assertion logic
  reviewed, not adversarially triggered since no call site currently
  batches).

---

## Discovered, not fixed (Phase 2, 2026-08-28)

- **`scripts/smoke_gates.py::gate_g3a` is genuinely flaky, order-dependent,
  pre-existing, and NOT introduced by this session's C1/C2/C8 changes.**
  Confirmed: `python scripts/smoke_gates.py --gate G3a` alone PASSED 3/3
  consecutive runs; `python scripts/smoke_gates.py --all` (which runs G1 ->
  G2 -> G3a -> G3b -> G6 in one process) FAILED G3a 2/2 times with `only
  0/3 strikes recorded`. `gate_g3a` sets no `torch.manual_seed` of its own
  (unlike `gate_g1`/`gate_g2`), and the predictor's dropout layers are never
  put into `.eval()` mode anywhere in this file, so running G1/G2 first
  consumes global RNG state and leaves the predictor's forward passes
  non-deterministic in a way that measurably changes whether G3a's
  REL_SCALE=0.3 regime perturbation clears `tau=0.5` on the first `q=3`
  chunks. Out of scope for C1/C2 (register rows do not name G3a) — flagged
  here per FIX_SPEC rule 5 rather than silently fixed. **Recommended fix
  for a future pass:** either seed G3a explicitly (`torch.manual_seed(...)`
  at its top, matching G1/G2's pattern) or call `predictor.eval()` before
  each gate that must be deterministic.
- **C7's fix (`diagnose_cem_costs.py`'s incremental JSONL write) was
  compile-checked and logic-reviewed but not exercised end-to-end against a
  real GPU run** — Phase 2 is zero-GPU by mandate. Flagged so Phase 4 (E-A,
  which depends on this script) re-verifies the incremental file actually
  appears and grows during a real sweep before trusting it under time
  pressure.
- **`e0_v3_hybrid`'s methodology caveat (OPUS #11, two variables confounded
  at once) and `e0_v4_{full,lora4}`'s unmatched-budget caveat (A10) are
  RESTATED, not re-investigated, in `E0_DIRECTORY_INVENTORY.md` (D4).**
  Fixing either is out of this row's scope (A10 already has its own Phase 1
  register entry; OPUS #11 is a Phase 6 write-up item).
- **D9 (consolidate 470KB of AI-authored markdown into `EVIDENCE_LEDGER.md`
  as the single source of truth) was NOT attempted this pass** — explicitly
  lower priority in `SUBMISSION_PLAN.md`'s own Part B4 framing ("assign to
  Phase 2 if time"), and this session's time went to the higher-priority
  C1/C2 dead-gate rewrites and D1-D4/D10 first. Still open.
- **D8 (audit `smoke_e4.py`'s assertions / `modal_e4.py`) intentionally not
  duplicated here** — `SUBMISSION_PLAN.md` explicitly folds it into Phase
  3's spec (already covered in detail by `FIX_SPEC.md`'s B13 entry);
  re-doing it in Phase 2 would risk drifting from Phase 3's authoritative
  version.

---

*Original placeholder: Phase 0 (setup) complete; Phase 1 Stage 1 landed
2026-08-27. Stage 2 (E2 + E0 re-scores) landed 2026-08-28 — see "Phase 1
Tier A Stage 2" section, below the Phase 2/3 sections in file order.*

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

### Phase 3 Step 0 — B3 motion-gate calibration, 2026-08-28

**STEP 0 — MEASUREMENT ONLY, NO FIX APPLIED.** Per explicit task scope: this
entry records a measurement made to answer FIX_SPEC.md B3's go/no-go
question before any recalibration is implemented. No production file
(`scripts/run_e4.py`, `atlas/score.py`, or any other) was modified.

- **Question:** `scripts/run_e4.py:182-190` computes `motion_gate` as the
  10th percentile of **whole-`traj_len=10`** displacement (frameskip=5 ->
  2 model-step chunks) from **3** trajectories under regime_a (R0), then
  applies that single threshold via `score.py::umf`'s gate check
  (`observed_displacement <= motion_gate -> return None`) to **1-model-step**
  chunks throughout the whole S2 stream at `nas=1` (both R0 and R2
  segments). Does that gate swallow essentially all nas=1 chunks, silently
  collapsing arms 4-7 into arm 1?
- **Method:** `scratchpad/measure_motion_gate.py` (throwaway, not added to
  `atlas/` or `scripts/`). Loaded the real `EncPredWM` wrapper
  (`dino_wm_pusht`, local hub, same pattern as `run_e0_planning.py`/
  `smoke_gates.py` post-C8). (1) Reproduced `run_e4.py:182-190` EXACTLY:
  `load_regime_trajectories(model, prep, "R0", num_trajs=3, traj_len=10,
  seed_offset=20_000)`, `compute_motion_gate` at the 10th percentile. (2)
  Independently collected 30 real trajectories under R0 and 30 under R2 at
  `traj_len=30` (= `MAX_MPC_STEPS`, one full S2 episode), and for each
  computed all 6 individual single-model-step displacements
  `||z[k+1]-z[k]||_F` (k=0..5) — the exact same quantity and chunk
  granularity `score.py::umf`'s gate check (`score.py:87`, `T=1` at
  `nas=1`) compares against `motion_gate`. 180 samples per regime.
- **Result:**
  - Current gate (exact `run_e4.py` call): raw 3-trajectory whole-2-model-step
    displacements `[310.87, 393.19, 345.35]` -> 10th percentile =
    **317.77**.
  - R0 nas=1 (1-model-step) chunk displacement distribution (n=180): min
    25.27, p5 46.82, p10 56.86, p25 82.88, p50 114.96, p75 160.63, p90
    210.99, max 278.55, mean 125.83.
  - R2 nas=1 chunk displacement distribution (n=180): min 70.01, p5 89.67,
    p10 103.85, p25 122.89, p50 157.47, p75 192.81, p90 223.45, max 331.57,
    mean 160.30.
  - Fraction of real nas=1 chunks at/below the current gate (317.77):
    **R0: 100.00% (180/180)**, **R2: 99.44% (179/180)**.
- **Verdict: NO-GO as currently calibrated.** The gate is computed from a
  quantity (2-model-step cumulative displacement) that is structurally
  larger than the 1-model-step quantity it is applied to, so its 10th
  percentile sits above nearly the entire real nas=1 distribution's max
  (R0 max 278.55, R2 max 331.57, vs gate 317.77). At `nas=1` in production,
  `umf()` would return `None` for essentially every chunk, `Expander.record()`
  would return early on nearly every call, and arms 4-7 would silently
  collapse into arm 1 (frozen) while `run_e4.py` writes a plausible-looking
  success table with no error. **Phase 5's nas=1 protocol cannot launch on
  the current gate calibration; B3's part (b) (recalibrate at the chunk
  size actually scored, over >=30 trajectories) is required, not optional.**
- **Claims affected:** none yet (E4 has never run — nothing on disk is
  contaminated). This measurement gates whether Phase 3's B3 fix and Phase
  5's launch can proceed as currently speced.
- **Re-run needed:** n/a (measurement, not a fix).
- **Verification:** N/A — Step 0 is measurement-only per explicit task
  scope; no before/after assertion pair applies. Raw command + full stdout
  shown in this session's report.
- **Verified by:** orchestrating session, direct run (real `dino_wm_pusht`
  checkpoint, real `PhysicsRegime`-wrapped `PushTEnv`, CUDA).

---

### Phase 3 Step 1 — B4+C6 combined, B3 recalibration, B13, 2026-08-28

Dispatched (coordinator-authorized continuation of Phase 3, following Step 0):
B4+C6 (combined, same lines), B3 (using Step 0's own measurement as input),
B13. Order: B3 landed before B4's integration assertion was retested,
because B3's broken gate was independently masking B4 (see B4 row below).

### B4 + C6 — ATLAS arms silently never refine ln_act/full charts (combined fix)

- **Date:** 2026-08-28
- **Defect:** `scripts/run_e4.py:165-168` calls `requires_grad_(False)` on
  `wm.encoder.parameters()` and `wm.predictor.parameters()` once, up front,
  before any chart is ever applied. `atlas/loop.py::atlas_refine` and
  `atlas/expand.py::_fit_candidate` then build an Adam optimizer over a
  subset of `predictor.named_parameters()` selected by `chart._param_names`
  (`ln_act`/`full`) — for `ln_act`/`full`, `chart.apply_()` only
  `.data.copy_()`s into those SAME (frozen) tensors, so they stay
  `requires_grad=False` forever. C6 (already landed in Phase 2) fixed a
  SEPARATE defect on the same lines: for `kind="lora4"`, `apply_()`
  replaces the base weight's `named_parameters()` entry with a
  parametrization, so the name-membership filter selects zero params and
  `optim.Adam([])` raises; C6 already switched `lora4` to a name-suffix
  filter (`"lora_A" in n or "lora_B" in n`).
- **DISCOVERY — the observed failure mode is NOT what FIX_SPEC.md
  describes, though the root cause and prescribed fix are the same.**
  FIX_SPEC.md B4 says "`loss.backward()` raises." Empirically it does
  **not** raise: `run_e4.py`'s freeze loop only touches
  `wm.encoder`/`wm.predictor`, never `wm.action_encoder`/
  `wm.proprio_encoder` — confirmed both still `requires_grad=True`,
  untouched by the loop. `_open_loop_rollout`'s forward pass threads
  action/proprio embeddings through those still-trainable modules, so the
  loss tensor keeps a valid `grad_fn` even when every chart-selected
  predictor param has `requires_grad=False`. `loss.backward()` therefore
  **succeeds silently**, computing real (unwanted, unused) gradients for
  `action_encoder`/`proprio_encoder` while leaving every actual chart
  parameter's `.grad` as `None` — Adam's `step()` is then a total no-op
  for those params. **The chart is never refined, with no error anywhere.**
  This is the same defect B4 names (frozen-vs-optimized param mismatch)
  with a worse (silent, not crashing) symptom than documented. Per rule 5
  ("if you find a new bug... stop"), judged this is NOT a new/separate
  defect — it is B4 itself, observed directly rather than via the
  documented crash — so the prescribed fix (re-enable `requires_grad` on
  the chart's own selected surface) was applied as authorized, and this
  discovery is recorded here rather than treated as an out-of-scope find.
  `lora4` was confirmed NOT affected by the crash-or-no-op question at all
  in the pre-fix state, because `_inject_lora`'s freshly-constructed
  `nn.Parameter(A)`/`nn.Parameter(B)` default to `requires_grad=True`
  regardless of the up-front freeze.
- **Change:** `atlas/loop.py::atlas_refine` and
  `atlas/expand.py::_fit_candidate` — immediately after building `params`
  (the existing C6 kind-branch retained unchanged), added
  `for p in params: p.requires_grad_(True)` before constructing/using the
  Adam optimizer. C6's suffix-based `lora4` filter is untouched.
- **Claims affected:** any E4/S2 arm 4-6 (`atlas_fixed`/`atlas_detect`/
  `atlas`) result with `kind` in `{"ln_act", "full"}` — E4 has never run,
  so nothing on disk is affected; this blocks Phase 5 from producing a
  meaningless "0 adaptation, no error" table.
- **Re-run needed:** no (E4 has never run).
- **Verification:** two layers.
  **(1) Isolated, direct call** (`scratchpad/assert_b4.py`) — freezes
  `wm.encoder`/`wm.predictor` exactly as `run_e4.py:165-168` does, builds a
  real chart, calls `atlas_refine` on one real R0 chunk, compares
  `chart._params` before vs. after.
  BEFORE (unfixed): `kind=ln_act` → `atlas_refine returned loss=0.0360...`,
  `chart params changed after refine: False` →
  `RESULT kind=ln_act: UNCHANGED/RAISED (refine did NOT work)`.
  `kind=lora4` (unaffected by this defect, as predicted) →
  `chart params changed after refine: True` → `RESULT kind=lora4: CHANGED`.
  AFTER (fixed): `kind=ln_act` → `chart params changed after refine: True`
  → `RESULT kind=ln_act: CHANGED (refine worked)`. `kind=lora4` unchanged
  (still `CHANGED`, as expected — this fix is additive for that kind).
  **Failed before fix: yes** (`ln_act`, the case B4 actually affects).
  **(2) Integration assertions** (FIX_SPEC.md's literal commands, run with
  a reduced CEM budget `--num-samples 20 --iterations 2 --horizon 2` for
  wall-clock reasons — this does not change the pass/fail logic, only
  search quality):
  (a) `python scripts/run_e4.py --arms atlas --episodes 2 --seeds 1
  --kind ln_act` → runs to completion, 12 episodes logged, no traceback.
  Note: this integration command did NOT reproduce a crash even
  pre-B4-fix, because B3's (also-broken, also fixed this session) gate
  independently prevented the router from ever selecting a non-`c0` chart
  index, so `atlas_refine` (gated behind `state.current_idx != 0` in
  `harness_e4.py:313`) was never invoked at all — confirmed directly:
  before B3's fix, `--arms atlas --episodes 3` produced `selected_trace`
  values of all-0 or all-1 depending on episode, but with `refine_loss`
  fields either `None` or populated ONLY on episodes where the (still
  broken) gate happened to allow a switch, and B4 was not yet exercisable
  through this path in isolation from B3. Once B3 was fixed first (see
  below), a rerun of this exact integration command with B4 STILL
  reverted (`git`-free rollback via `Edit` to the pre-fix line, not
  `git stash` — the repo has many other Phase-1/2 in-flight uncommitted
  edits) was attempted; the direct isolated test above is the reproducible
  before/after evidence of record, since the integration path's manifestation
  depends on router randomness across two confounded bugs. AFTER (both B3
  and B4 fixed): (a) `ln_act` → runs to completion (`Summary:
  atlas_out\...\e4_summary.json` written, 1 success out of 12 episodes,
  no traceback). (b) `lora4` (using `--charts atlas_out/e0_v4_lora4`,
  since the default `--charts atlas_out/e0_v3_dataset` has no
  `chart_lora4_R2.pt` on disk — an unrelated, pre-existing data-path gap,
  not a B4/C6 defect) → also runs to completion, no `optim.Adam([])`
  raise.
- **Verified by:** orchestrating session, direct run (isolated test
  before/after shown side by side above; integration runs shown for both
  kinds post-fix).

### B3 — Motion gate calibrated at the wrong chunk size (recalibration)

- **Date:** 2026-08-28
- **Defect:** confirmed and measured in Phase 3 Step 0 (see that section
  above): `scripts/run_e4.py:182-190` (pre-fix) computed the 10th
  percentile of WHOLE-`traj_len=10` (== 2-model-step, frameskip=5)
  displacement from 3 trajectories under R0, then applied that single
  threshold to SINGLE-model-step (`nas=1`, `T=1`) chunks via
  `score.py::umf`'s gate throughout the whole stream (both R0 and R2
  segments). Step 0 measured: gate=317.77 vs. real nas=1 chunk
  displacement max 278.55 (R0) / 331.57 (R2) → 100.00% / 99.44% gated.
- **Change:** `scripts/run_e4.py` — the calibration call now uses
  `gate_traj_len = FRAMESKIP * args.num_act_stepped` (= 5 at the current
  `nas=1` default — i.e. exactly ONE model-step, the same granularity
  `harness_e4.py::run_e4_episode`'s per-replan re-encoding window
  produces) and `gate_num_trajs = 30` (was 3). `compute_motion_gate`
  itself (owned by `atlas/score.py`, out of scope to touch) is called
  unchanged. No other line touched.
- **Claims affected:** every future E4/S2 result (E4 has never run).
- **Re-run needed:** no.
- **Verification:** ran `python scripts/run_e4.py --arms atlas --episodes 3
  --seeds 1 --kind ln_act --num-samples 20 --iterations 2 --horizon 2`
  (reduced CEM budget for wall-clock; does not affect the gate value,
  which is computed before any planning).
  BEFORE (unfixed, `traj_len=10`/`num_trajs=3`): `motion_gate = 317.7534`
  (matches Step 0's 317.77 exactly, within float/seed noise).
  AFTER (fixed, `traj_len=5`/`num_trajs=30`): `motion_gate = 117.6191
  (from 30 single-chunk displacements, range [57.19, 397.28])` — squarely
  inside Step 0's independently-measured real nas=1 R0 range (25.27-278.55,
  median 114.96).
  **Gated-fraction sanity check** (FIX_SPEC.md B3's own assertion —
  "not ~100%, not 0%, compare against E2's ~20-30% as a rough anchor"):
  computed directly from the 3-episode smoke's `umf_trace` fields (12
  replans total across 3 episodes, first replan of each episode has no
  prior chunk so is excluded, giving 9 scored replans): **3/9 = 33.3%**
  of scored replans logged a `None` (gated) score in at least one library
  slot's trace, in the same range as E2's ~20-30% anchor (not exact, small
  sample, but the right order of magnitude — not ~100%, not 0%).
  **Failed before fix: yes** (100.00%/99.44% gated per Step 0; before-fix
  gate value independently reproduced above as 317.75 ≈ Step 0's 317.77).
- **Verified by:** orchestrating session, direct run (both before/after
  gate values and the gated-fraction check shown above).

### B13 — `smoke_e4.py`'s assertions strengthened

- **Date:** 2026-08-28
- **Defect:** of `scripts/smoke_e4.py`'s eight assertions: (1) `frozen`'s
  `r["library_size"] == 1` was structurally vacuous — `harness_e4.py`'s
  record always writes `library_size=1` when `state.library is None`,
  which `build_arm_state()` guarantees for `arm="frozen"` by construction;
  the assertion could not fail regardless of runtime behaviour. (2)
  `atlas_fixed`'s `probe_outcome != "committed"` was a control-flow
  guarantee — `atlas_step()`'s `'fixed'`/`'none'` branch never sets
  `probe_outcome` to anything but its `"not_ready"` default, so this could
  not fail either. (3) the divisibility check (`n >= 0` on a Python list
  length) contradicted its own docstring, which claims
  `raw_steps_per_replan` entries are checked as multiples of `frameskip`.
  (4) the AdaJEPA-reset check filtered both `post_reset_state` and
  `state.adapter.pretrained_state` by the same `param_names` list, so a
  total namespace divergence (or an accidentally-empty `param_names`)
  would compare two empty dicts and vacuously pass.
- **Change (`scripts/smoke_e4.py`):**
  - **(1) frozen:** replaced with a genuine dynamic check — after running
    all of `frozen`'s episodes, assert the live predictor's
    `state_dict()` is bit-identical, key by key, to
    `pristine_predictor_state` (captured before any arm ran). A future bug
    that accidentally ran `atlas_refine` for this arm would break it.
  - **(2) atlas_fixed:** replaced with two dynamic checks over real
    per-episode records: `library_size` must equal the arm's initial
    library size on every single record (a real commit event, were one to
    happen, would break this); and the distinct set of chart indices
    selected across the smoke run is now printed (not hard-asserted —
    see below) as a live-routing diagnostic.
  - **(3) divisibility:** implemented the real check — every
    `raw_steps_per_replan` entry must be an exact multiple of
    `frameskip=5`, EXCEPT possibly the episode's final entry when
    `success=True` (the inner step loop can legitimately break mid-chunk
    the instant success is detected — `harness_e4.py:271-274`), which is
    instead bounded `0 <= n <= frameskip`. Also asserts the check covered
    a nonzero number of entries (guards against a future config making the
    loop trivially empty).
  - **(4) AdaJEPA reset:** added an explicit
    `assert len(state.adapter.param_names) > 0` and
    `assert len(post_reset_state) == len(state.adapter.param_names)`
    before the equality check, so a namespace mismatch or empty
    `param_names` fails loudly instead of silently degenerating to a
    vacuous zero-pair comparison. Same non-empty-coverage pattern applied
    to the oracle_id routing check (`oracle_checked > 0` asserted after
    the loop — previously, if every oracle record had
    `len(selected_trace) <= 1` (all-warmup episodes), the loop would
    silently check nothing and still print "OK").
  - **DEVIATION from an initial stronger draft, recorded for honesty:** a
    first version of the `atlas_fixed` change hard-asserted
    `len(distinct_selected) > 1` (the arm must select more than one chart
    index across the smoke run). Run against this smoke test's own tiny
    config (`num_samples=2, iterations=2, horizon=2`, `max_raw_steps=10`,
    `n_replans_target=4`), this genuinely failed:
    `atlas_fixed: routing never selected more than one chart index ({0})`.
    Root cause: at this budget, `max_raw_steps=10`/`frameskip=5` yields
    only ONE scored replan per episode (the first replan has no
    `prev_chunk` to score), so 4 episodes give only 4 total routing
    decisions — not enough for a hard cross-regime-switch guarantee to be
    a fair test of the mechanism (as opposed to this specific tiny
    config's statistics). Judged this was testing the smoke config's
    size, not a defect, and downgraded it from an assertion to a printed
    diagnostic rather than either (a) silently keeping a hard assert that
    would make this smoke test flaky/config-dependent, or (b) quietly
    weakening scope without recording it. The library-size-never-grows
    check (kept as a hard assert) is what substantively replaces the old
    vacuous `probe_outcome != "committed"` check.
- **Claims affected:** none directly (`smoke_e4.py` is a validator, not a
  results-producing script) — this is the gate Phase 5 will rely on before
  spending real GPU-hours.
- **Re-run needed:** no.
- **Verification:** `python scripts/smoke_e4.py --charts atlas_out/e0_v3_dataset
  --kind ln_act --segment-regimes R0 R2` (run AFTER B3/B4+C6 landed, since
  this smoke test exercises the same `atlas_refine`/motion-gate code
  paths) → all six numbered checks print `OK`, ending
  `All E4/E3 smoke-test assertions passed.` — including
  `[3/6] OK: 56 raw_steps_per_replan entries checked for frameskip=5
  divisibility` and `[4/6] ... oracle_id selects the correct chart by the
  last replan (4/4 records checked)`, both now reporting real nonzero
  coverage counts. The four strengthened checks are not classic "must
  fail before / pass after" dead-mechanism rewrites (per FIX_SPEC.md B13
  they are described as vacuous-assertion *replacements*, not broken-gate
  rewrites) — their vacuity was established by static/control-flow
  analysis at the FIX_SPEC.md-writing stage (reproduced above in
  "Defect"), not by an empirical before/after pair; the (1) hard-assert
  draft above IS a genuine empirical before/after data point on the one
  sub-check that turned out to be checkable that way.
- **Verified by:** orchestrating session, direct run.

### Phase 3 Step 2 — B1, B2, B6, B12 (dead mechanisms), 2026-08-28

### B1 — ATLAS arm could never commit a chart

- **Date:** 2026-08-28
- **Defect:** `atlas/harness_e4.py` (pre-fix `:216-217`) passed
  `next_encoder_output=None, next_actions=None` unconditionally to
  `atlas_step()` for every arm, so `atlas/loop.py:140-150`'s guard
  (`next_encoder_output is not None and ... strikes >= q`) was
  structurally unsatisfiable — the `atlas` arm (full verification) could
  never reach `maybe_expand()`, so it could never commit a chart, and
  strikes accumulated without bound (never reset by a probe outcome).
- **Change (`atlas/harness_e4.py`):** implemented the two-deep chunk
  buffer the file's own pre-existing comment already described but never
  built, copying `scripts/run_e2.py:296-339`'s peek-one-ahead pattern
  (verified leakage-free per `CODE_AUDIT.md` §9.3), adapted to an ONLINE
  rollout. Added `atlas_verify_buffer` (a sliding 2-element list, ATLAS
  arm ONLY — `atlas_fixed`/`atlas_detect` are unaffected, since
  `loop.py`'s `'fixed'`/`'detect_only'` branches never read
  `next_encoder_output` at all): the OLDER of the two buffered chunks is
  passed as `encoder_output`/`actions` (the deficit candidate, `record()`'d
  as before); the NEWER (one replan fresher) is passed as
  `next_encoder_output`/`next_actions` — the genuinely-unseen verification
  chunk. Three arguments threaded, not two, as the spec required:
  `next_encoder_output`, `next_actions`, AND `next_proprio_ctxt` (the
  pre-fix code passed no proprio for the (always-None) next chunk either).
  `atlas_fixed`/`atlas_detect`/`oracle_id`/`adajepa*` continue to use the
  original single-chunk `prev_chunk` variable, unchanged.
  **Named, accepted side effect (matches the file's own pre-existing
  comment, "the one-replan delay is structural, not a bug" —
  E3_E4_IMPLEMENTATION_PLAN.md §2c):** for arm `atlas` specifically, (a)
  SCORE+SELECT is now based on the one-replan-STALER of the two buffered
  chunks (not the freshest available), and (b) its first real routing
  decision needs 2 chunks buffered instead of 1, so it gets one fewer
  active replan than `atlas_fixed`/`atlas_detect` within the same
  `n_replans_target`. Not fixed further — this is the documented cost of
  genuine verification, not a bug.
- **Claims affected:** any future S2/E4 result for the `atlas` arm (RQ3,
  N9-style commit counts) — E4 has never run, nothing on disk affected.
- **Re-run needed:** no.
- **Verification:** `scratchpad/assert_b1.py` — builds a real `atlas` arm
  state (`build_arm_state`) over the real `chart_ln_act_R2.pt` library,
  with `tau=-1.0, q=1` (deliberately guarantees a strike every informative
  replan — isolates "is the mechanism reachable at all" from "does the
  real `tau=0.5/q=3` fire in practice", a separate question; CLAUDE.md's
  `tau=0.5`/`q=3` non-negotiables are untouched in any production file),
  runs 3 real episodes.
  BEFORE (temporarily reverted `next_encoder_output`/`next_actions` back
  to hardcoded `None` at the call site, reproducing the exact pre-fix
  defect): `episode 0: probe_outcome=not_ready strikes=4`, `episode 1:
  probe_outcome=not_ready strikes=8`, `episode 2: probe_outcome=not_ready
  strikes=12` → `All probe_outcomes: ['not_ready', 'not_ready',
  'not_ready']` → `RESULT: STUCK at not_ready every episode (B1 bug
  reproduced)`. Strikes grow UNBOUNDED across episodes (never reset),
  additional confirmation the probe never fires.
  AFTER (fix restored): `episode 0: probe_outcome=rejected_score
  strikes=0`, `episode 1: probe_outcome=rejected_score strikes=0`,
  `episode 2: probe_outcome=rejected_score strikes=0` →
  `All probe_outcomes: ['rejected_score', 'rejected_score',
  'rejected_score']` → `RESULT: REACHED (B1 fixed -- probe mechanism
  reachable)`. Strikes reset to 0 after each probe (fires and correctly
  evaluates, whether it commits or rejects). **Failed before fix: yes.**
  Also re-ran `python scripts/smoke_e4.py --charts atlas_out/e0_v3_dataset
  --kind ln_act --segment-regimes R0 R2` after restoring the fix — still
  `All E4/E3 smoke-test assertions passed.` (no regression from the
  restructure).
- **Verified by:** orchestrating session, direct run (before/after shown
  side by side above).

### B2 — Arm 2 (adajepa) was behaviourally identical to arm 3 (adajepa_persist)

- **Date:** 2026-08-28
- **Defect:** `AdaJEPA.reset()` (`atlas/adajepa.py:94-104`) had no
  production caller anywhere. In production, plain `"adajepa"` never
  re-initialised, so it was behaviourally `"adajepa_persist"` — the
  persistence rung this ladder exists to isolate measured zero by
  construction.
- **Change (`atlas/harness_e4.py`):** added
  `if state.arm in ("adajepa", "adajepa_persist"): state.adapter.reset()`
  at the top of `run_e4_episode`, before the replan loop. `reset()` is a
  documented no-op for `variant="persistent"` (`adajepa.py:101`: `if
  self.variant == "adajepa":`), so calling it unconditionally for both
  arms is safe — only arm 2 actually resets (predictor weights + 5-chunk
  buffer). Key-namespace concern pre-cleared (`SUBMISSION_PLAN.md` A-xi).
- **DISCOVERY (documented, not a separate defect):** the coordinator's
  literal assertion ("arms 2 and 3 produce DIFFERING JSONL rows over 2
  episodes [via the full `run_e4.py` pipeline]; must be IDENTICAL before
  the fix") could not be satisfied literally. Confirmed empirically: with
  `reset()` disabled for BOTH arms (simulating "before"), the two arms'
  full-pipeline JSONL rows were ALREADY non-identical
  (`refine_loss`/`block_pos_diff` differ from the first episode on). Root
  cause, unrelated to B2: `scripts/run_e4.py`'s single shared `GC_Agent`
  CEM planner consumes global torch/numpy RNG state cumulatively across
  arms and is never re-seeded per arm/episode — this is exactly B11's
  still-open defect (`n_replans_target` aside, B11 = "re-seed the CEM
  generator per episode from `spec.seed`"), out of scope to fix under B2.
  Used an isolated, deterministic replacement instead (no CEM/planner
  involved): does the underlying `AdaJEPA`-adapted predictor snap back to
  PRETRAINED weights between episode 1 and episode 2, for `variant=
  "adajepa"` (must, if fixed) vs. `variant="persistent"` (must never, by
  design)?
- **Claims affected:** the persistence rung of the E4/S2 ablation ladder
  (RQ3/RQ4-adjacent) — E4 has never run.
- **Re-run needed:** no.
- **Verification:** `scratchpad/assert_b2.py` — one gradient step
  (`push`+`refine`) on episode 1's real R0 chunk, then either calls
  `reset()` or not, then compares predictor state to pretrained.
  BEFORE (B2's exact defect, i.e. `reset()` never called between
  episodes — `call_reset_between_episodes=False`):
  `variant=adajepa reset_called_between_episodes=False:
  differs_after_ep1=True differs_at_ep2_start=True` — arm 2 stays
  adapted across the episode boundary, bit-for-bit identical in this
  respect to arm 3 (`variant=persistent ... differs_at_ep2_start=True`).
  **Failed before fix: yes** (arm 2 indistinguishable from arm 3 on this
  mechanism).
  AFTER (`call_reset_between_episodes=True`, matching the real
  `harness_e4.py` hook): `variant=adajepa
  reset_called_between_episodes=True: differs_after_ep1=True
  differs_at_ep2_start=False` — arm 2 now genuinely snaps back to
  pretrained between episodes, while arm 3 (also with `reset()` called,
  since it's now unconditional per-episode, but a documented no-op for
  `variant="persistent"`) still shows `differs_at_ep2_start=True` — the
  two arms are now behaviourally distinguished on exactly this mechanism.
- **Verified by:** orchestrating session, direct run (before/after shown
  side by side above).

### B6 — `atlas_detect` committed an inert byte-identical clone

- **Date:** 2026-08-28
- **Defect:** `atlas/loop.py:152-170` (`expansion_mode="detect_only"`
  branch) committed `library.clone_from(best_idx)` directly, with NO
  gradient step — a byte-identical clone of its parent. Since UMF is a
  deterministic function of weights, the clone permanently ties its
  parent's score and never wins the router's argmin; not what
  "detect-and-spawn" (as opposed to "detect-and-do-nothing") means, and
  not what the `atlas`-vs-`atlas_detect` rung comparison ("verifies", per
  FIX_SPEC.md) is supposed to isolate.
- **Change (`atlas/loop.py`):** call `_fit_candidate(new_chart,
  world_model, expander._deficit_chunks, cfg.n_probe, cfg.lr)` (imported
  from `atlas.expand`) on the cloned chart BEFORE `library.add()`, when
  `expander._deficit_chunks` is non-empty — the same fitting call
  `maybe_expand()` makes for `expansion_mode="atlas"`. Still commits with
  NO held-out verification, which remains the sole defining difference
  from `atlas` (full verification) after this fix.
- **Claims affected:** the `atlas_detect` arm's whole reason for existing
  in the E3 expansion-ablation ladder (RQ3) — E4 has never run.
- **Re-run needed:** no.
- **Verification:** `scratchpad/assert_b6.py` — builds a real `c0`
  (`ln_act`) library, forces a strike (`tau=-1.0, q=1`, same
  reachability-isolation rationale as B1's assertion — CLAUDE.md's real
  `tau`/`q` untouched anywhere in production), calls `atlas_step(...,
  cfg.expansion_mode="detect_only")` once, and compares the committed
  chart's weights to its parent's.
  BEFORE (temporarily reverted — removed the `_fit_candidate` call,
  reproducing the exact pre-fix `library.add(new_chart)`-with-no-fit
  code): `probe_outcome=committed library_size=2` →
  `committed chart differs from parent (c0): False` →
  `RESULT: IDENTICAL (B6 bug reproduced)`. **Failed before fix: yes.**
  AFTER (fix restored): `probe_outcome=committed library_size=2` →
  `committed chart differs from parent (c0): True` →
  `RESULT: CHANGED (B6 fixed)`.
- **Verified by:** orchestrating session, direct run (before/after shown
  side by side above).

### B12 — Modal's "multi-seed" sweep produced relabelled copies of one run

- **Date:** 2026-08-28
- **Defect:** `modal/modal_e4.py:109` always passed `--seeds 1` to the
  `run_e4.py` subprocess (a *count*, not the requested seed). Inside
  `run_e4.py`, `get_stream(..., seeds=1)` therefore only ever generated
  the `seed_run=0` episode/init/goal stream, and `for seed_run in
  range(profile_seeds)` (with `profile_seeds=1`) only ever ran local
  `seed_run=0` — regardless of which `seed_run` the container was
  launched for. `modal_e4.py:126,138` then rewrote each record's
  `seed_run` field to the CONTAINER's requested value AFTER THE FACT, so
  a "3-seed" sweep produced bit-identical episode data under 3 different
  labels.
- **Change:**
  - `scripts/run_e4.py` — added `--seed-run-offset` (default 0). Streams
    are now built via `get_stream(args.stream, args.episodes,
    args.seed_run_offset + args.seeds, ...)` and sliced to
    `[seed_run_offset : seed_run_offset + seeds]` (cheap: `stream_s2` is
    pure seed arithmetic, no GPU/env cost for the extra streams). The
    per-arm loop's `seed_run` variable is now the REAL global index
    (`args.seed_run_offset + local_i`); `streams` is indexed by the LOCAL
    position, `seed_run` is used everywhere records/resume-keys are
    written. `build_planner_cfg()` gained a `local_seed` parameter (was a
    hardcoded `0` inside the function) and `main()` now passes
    `local_seed=args.seed_run_offset`, so the CEM planner's own internal
    seed also genuinely varies per offset, not just the init/goal draw.
  - `modal/modal_e4.py` — the subprocess command now also passes
    `--seed-run-offset {seed_run}` (the container's real requested seed).
    The post-hoc `rec["seed_run"] = seed_run` relabelling is left in
    place (now confirmatory/defensive, not load-bearing).
- **Claims affected:** Stretch A (multi-seed continual-stream replication)
  — directly named CRITICAL in `SUBMISSION_PLAN.md` A-ix as something
  that "would have wasted $38" if launched unfixed. No GPU spend has
  occurred yet under the old code (confirmed: no `atlas_out/e4*`
  directory exists per `D1`'s red-flag sweep), so nothing on disk is
  invalidated — this fix landing before any multi-seed spend is the point.
- **Re-run needed:** no.
- **Verification:** ran `python scripts/run_e4.py --arms frozen --episodes 2
  --seeds 1 ...` (reduced CEM budget) TWICE in separate invocations,
  simulating two Modal containers requesting `seed_run=0` and `seed_run=1`.
  BEFORE (both invocations omitting `--seed-run-offset`, i.e. both
  defaulting to offset 0 — reproducing modal_e4.py's OLD behaviour of
  always requesting `--seeds 1` with no way to select a different
  stream): `run0: [97.82, 72.61, 61.96, 46.16, 103.46, 65.95, 62.57,
  136.82, 138.44, 109.71, 64.67, 74.13]`, `run1:` — bit-for-bit identical
  list → `IDENTICAL (bug reproduced): True`. **Failed before fix: yes.**
  AFTER (`--seed-run-offset 0` vs. `--seed-run-offset 1`): `run0 seed_run
  field: {0}`, `run1 seed_run field: {1}`; `run0:
  [97.82, 72.61, 61.96, ...]` (unchanged), `run1: [48.23, 73.63, 174.60,
  89.13, 85.80, 61.83, 59.60, 80.50, 138.83, 81.26, 107.59, 98.28]` —
  entirely different sequence → `DIFFERENT (fix confirmed): True`.
- **Verified by:** orchestrating session, direct run (before/after shown
  side by side above).

### Phase 3 Step 3 — B5, B7, B8, B9, B10, B11, B14 (remainder), 2026-08-28

### B5 — S2 stream keyed revisits on absolute segment index, not regime-visit slot

- **Date:** 2026-08-28
- **Defect:** `atlas/streams.py` (pre-fix `:86-87`) computed
  `paired_seed(seg_idx + stream_seed_offset*1000, ep_idx + seed_run*10_000)`
  — keyed on the ABSOLUTE segment index. Segment 0 (first A-visit) and
  segment 2 (second A-visit) therefore got DIFFERENT seeds for the same
  `episode_idx`, so RQ4's paired first-visit-vs-revisit comparison had no
  shared underlying episode to pair a "does the library remember this
  regime" comparison against.
- **Change (`atlas/streams.py::stream_s2`):** key on `seg_idx % 2` (the
  REGIME-VISIT SLOT: 0 = every A-visit — segments 0, 2, 4; 1 = every
  B-visit — segments 1, 3, 5) instead of the raw `seg_idx`. Episode *i* of
  every visit to the same regime now shares the identical seed (identical
  init state + goal); only the accumulated library/adapter state differs
  across visits.
- **Claims affected:** RQ4 (recall/forgetting) — the only mechanism that
  could measure it — E4 has never run.
- **Re-run needed:** no.
- **Verification:** `get_stream("s2", episodes_per_segment=3, seeds=1,
  regimes=("R0","R2"))`.
  BEFORE (reproduced the exact pre-fix formula standalone, not a live
  revert — since `streams.py` is a pure function, direct comparison
  suffices): `seg0: {0: 3612771448, 1: 1524518835, 2: 1842423889}`,
  `seg2: {0: 4166643358, 1: 2968023890, 2: 137060774}` →
  `IDENTICAL (would be required if bug still present): False`. **The
  pre-fix formula fails the pairing requirement — confirmed directly.**
  AFTER: `seg0: {0: 3612771448, 1: 1524518835, 2: 1842423889}`,
  `seg2: {0: 3612771448, 1: 1524518835, 2: 1842423889}` →
  `seg0==seg2: True`, `seg1==seg3: True`.
- **Verified by:** orchestrating session, direct run (before/after shown
  side by side above).

### B7 — `make_tables.py`'s `make_t2` paired by episode COUNT, not KEY SET

- **Date:** 2026-08-28
- **Defect:** `scripts/make_tables.py` (pre-fix `:126`) gated pairing on
  `len(outcomes) == len(baseline_outcomes)`. With resume (a partial
  `episodes.jsonl` continued later), two arms can reach the same episode
  COUNT while covering DIFFERENT `(seed_run, global_episode_idx)` keys —
  the sorted-by-key arrays would then silently pair episode *i* of one arm
  against a DIFFERENT underlying episode *i* of the other (same array
  position, different actual episode), corrupting `paired_bootstrap`/
  `mcnemar_paired`'s pairing assumption with no error raised.
- **Change (`scripts/make_tables.py::make_t2`):** added
  `keyed_outcomes_for()`, builds a `dict[(seed_run, global_episode_idx) ->
  outcome]` per arm; pairing now intersects the arm's and the baseline's
  actual key sets (`common_keys = sorted(set(arm_keyed) &
  set(baseline_keyed))`) and passes ONLY the intersection to
  `paired_bootstrap`/`mcnemar_paired` (both called UNCHANGED — additions
  only, per the standing rule). If the intersection is smaller than either
  arm's full episode count, the delta string is annotated
  `(n=K/N paired)`. If the intersection is empty, reports
  `N/A (zero overlapping ... keys)` rather than a bogus number.
- **Claims affected:** any E4/S2 table built from a resumed run (N9-style
  ladder-rung deltas) — E4 has never run, so no existing table is affected.
- **Re-run needed:** no.
- **Verification:** `scratchpad/assert_b7.py` — synthetic 2-arm JSONL,
  `frozen` at keys `(0,0..2)` all `success=True`, `atlas` at DISJOINT keys
  `(0,5..7)` all `success=False` — SAME length (3) as `frozen`, ZERO
  overlapping keys.
  BEFORE (reproduced the exact pre-fix condition standalone against the
  same synthetic outcome arrays, since the arrays are what the old
  length-only check would have fed to `paired_bootstrap` unmodified):
  `len match: True` →
  `paired_bootstrap(atlas_outcomes, frozen_outcomes) = -1.0 (-1.0, -1.0)`
  → **a confident, precise, entirely bogus paired delta reported despite
  the two arms sharing ZERO actual episodes.**
  AFTER: `make_t2` on the same synthetic JSONL → `atlas` row:
  `N/A (zero overlapping (seed_run, global_episode_idx) keys)` — correctly
  refuses to report a number rather than mispairing.
- **Verified by:** orchestrating session, direct run (before/after shown
  side by side above).

### B8 — arms 4/5/6 skipped refinement entirely when routing selected c0 (design decision)

- **Date:** 2026-08-28
- **Defect:** `atlas/harness_e4.py` (pre-fix `:328`) gated the REFINE
  branch with `and state.current_idx != 0` — selecting c0 meant NO
  refinement happened at all, unlike arms 2/3 (AdaJEPA), which adapt every
  single replan with no "selection" concept at all. This confounded the
  3→4 ladder rung: it differed by TWO mechanisms simultaneously ("adapts
  every replan" vs. "adapts only when not on c0", AND "no library/routing"
  vs. "library/routing"), not the single mechanism the rung is meant to
  isolate.
- **THIS IS A DESIGN DECISION, recorded explicitly per FIX_SPEC.md's
  instruction, not a mechanical bug fix:** the alternative considered and
  rejected was refining c0 ITSELF when selected — rejected because it
  would violate CLAUDE.md §1 non-negotiable #4 (identity initialisation)
  and permanently corrupt the one chart every UMF comparison and Gate G1
  depend on staying pristine. The decision taken: introduce
  `ArmState.c0_adapted_chart`, a persistent CLONE of c0 (identity-
  initialised, per non-negotiable #4), created lazily the first time
  routing selects index 0. EXECUTE substitutes this clone for c0 whenever
  index 0 is selected (once it exists); REFINE always refines the clone,
  never c0. **Crucially, SCORE/SELECT (`atlas_step()`'s routing decision)
  is UNCHANGED — it always reads the real `library[0]` (pristine c0), so
  UMF-based routing correctness and Gate G1 are unaffected.** This means
  the "c0" a running episode actually executes against can, after the
  first selection, be a refined variant — a real, named change of what
  "selecting c0" means operationally from this fix onward, and the paper
  must describe it as such (recorded here for Phase 6).
- **Change (`atlas/harness_e4.py`):** `ArmState.c0_adapted_chart: Chart |
  None = None` field added. EXECUTE: use `state.c0_adapted_chart` in
  place of `state.library[0]` once created. REFINE: removed the
  `current_idx != 0` gate; when `current_idx == 0`, lazily clone c0 into
  `c0_adapted_chart` and refine THAT (via a dedicated `"c0_clone"`
  optimizer key), else refine `state.library[state.current_idx]` as
  before.
- **Claims affected:** the whole 3→4 ladder-rung comparison (RQ3) — E4 has
  never run.
- **Re-run needed:** no.
- **Verification:** `scratchpad/assert_b8.py` — `atlas_fixed` arm,
  `expansion_start_library="c0_only"` (library = {c0} only, so index 0 is
  the only possible selection — isolates this mechanism), 2 episodes.
  BEFORE (temporarily reverted the REFINE branch's condition back to
  `and state.current_idx != 0`): `episode 0: refine_loss=None`,
  `episode 1: refine_loss=None` → `c0_adapted_chart exists: False` →
  `RESULT: SKIPPED refinement at index 0 (B8 bug reproduced)`.
  **Failed before fix: yes.**
  AFTER (fix restored): `episode 0: refine_loss=0.0808`, `episode 1:
  refine_loss=0.0915` → `c0 (library[0]) weights UNCHANGED (must stay
  True): True` (identity non-negotiable verified intact) →
  `c0_adapted_chart exists: True`, `c0_adapted_chart differs from c0:
  True` → `RESULT: REFINED while at index 0 (B8 fixed)`.
  Also re-ran `scripts/smoke_e4.py` after restoring — still
  `All E4/E3 smoke-test assertions passed.` (no regression).
- **Verified by:** orchestrating session, direct run (before/after shown
  side by side above).

### B9 — arms 4/5/6 refined over a 1-chunk window vs. AdaJEPA's 5-chunk buffer

- **Date:** 2026-08-28
- **Defect:** `atlas/loop.py::atlas_refine` (unchanged, still exists)
  takes exactly one `(encoder_output, actions)` chunk per call.
  `atlas/harness_e4.py`'s REFINE branch called it with only the current
  replan's chunk, so arms 4/5/6 refined over 1/5th the data per gradient
  step that arms 2/3 (`atlas.adajepa.AdaJEPA`, `BUFFER_SIZE=5`) use,
  violating plan §7.6's explicit "same buffer size as AdaJEPA" and
  confounding the 3→4 rung with a third mechanism (buffer size) alongside
  B8's "adapts" mechanism and the intended "routes/expands" mechanism.
- **Change (ADDITIVE — `atlas_refine` itself is completely UNCHANGED, so
  every other caller, e.g. E1/E2/the gates, keeps its exact current
  behaviour):** added a new function `atlas/loop.py::atlas_refine_buffered(
  chart, world_model, buffer, lr, optimizer=None)` — same
  C6-suffix-filter + B4-requires_grad-reenable logic as `atlas_refine`,
  but sums per-item losses over a caller-supplied list of chunks (the same
  per-item-backward-then-single-`optimizer.step()` pattern
  `AdaJEPA.refine()` uses), returning the mean loss. `ArmState` gained
  `refine_buffers: dict` — one `collections.deque(maxlen=5)` per chart
  index (+ the `"c0_clone"` key for B8's clone). `harness_e4.py`'s REFINE
  branch now pushes the new chunk into the relevant deque and calls
  `atlas_refine_buffered(refine_target, world_model, list(buf), ...)`
  instead of `atlas_refine`.
- **Claims affected:** the 3→4 ladder rung (RQ3) — E4 has never run.
- **Re-run needed:** no.
- **Verification:** `scratchpad/assert_b9.py` — `atlas_fixed` arm,
  `c0_only` library, 3 episodes (enough replans to exceed 5 total chunks).
  BEFORE: established by direct code inspection (not a live revert,
  since the pre-fix code literally contained no buffer object at all —
  `atlas_refine` was called with exactly one chunk every time by
  construction; grep-confirmed no `deque`/`refine_buffers` reference
  existed anywhere in the pre-fix file). AFTER (live run):
  `after episode 0: c0_clone buffer length = 4`, `after episode 1: ...
  = 5`, `after episode 2: ... = 5` → `final buffer length: 5
  (AdaJEPA.BUFFER_SIZE=5)` → `RESULT: MATCHES AdaJEPA buffer size (B9
  fixed)`. Also re-ran `scripts/smoke_e4.py` — still
  `All E4/E3 smoke-test assertions passed.`
- **Verified by:** orchestrating session, direct run (after-state shown
  above; before-state established by code inspection, not an empirical
  revert — noted as the one Step-3 row without a literal reproduced
  failure run).

### B10 — `n_replans_target` unit mismatch (raw steps ÷ model-step count)

- **Date:** 2026-08-28
- **Defect:** `scripts/run_e4.py` (pre-fix `:171`) computed
  `n_replans_target = max(args.max_mpc_steps // args.num_act_stepped, 1)`
  — dividing a RAW-step budget (`max_mpc_steps`) by a MODEL-step count
  (`num_act_stepped`; each model step = `frameskip=5` raw steps), a unit
  mismatch. At the default `nas=1`, this computed `30 // 1 = 30` instead
  of the correct `6`. Numerically INERT for episode length only because
  `harness_e4.py`'s replan loop separately breaks once
  `elapsed >= max_raw_steps`, so real episodes still ran exactly 6
  replans regardless. But `n_replans_target` also feeds
  `steps_left_model = (n_replans_target - replan_idx) * num_act_stepped`
  (`harness_e4.py`), which tells the CEM planner how many MODEL steps
  remain so it can shorten its plan/horizon near the end of an episode —
  with the wrong (30 instead of 6) value, `steps_left_model` was always
  far larger than the horizon (6), so that shortening logic never
  actually engaged.
- **Change (`scripts/run_e4.py`):**
  `n_replans_target = max(args.max_mpc_steps // (args.num_act_stepped *
  FRAMESKIP), 1)` — divides by RAW steps per replan, matching
  `max_mpc_steps`'s own units.
- **Claims affected:** end-of-episode planning behaviour for every future
  E4/S2 arm — E4 has never run.
- **Re-run needed:** no.
- **Verification:** `python scripts/run_e4.py --profile --episodes 1
  --kind ln_act --num-samples 5 --iterations 2 --horizon 2` (reduced CEM
  budget; `n_replans_target` is computed before any planning, unaffected).
  BEFORE (temporarily reverted to the exact pre-fix formula):
  `[PROFILE] n_replans_target=30`. **Failed before fix: yes** (wrong by
  5x at the current default `nas=1`).
  AFTER (fix restored): `[PROFILE] n_replans_target=6` (matches every
  observed real episode's actual replan count throughout this session's
  testing, e.g. B1-B9's `n_replans=6` fields).
- **Verified by:** orchestrating session, direct run (before/after shown
  side by side above).

### B11 — CEM planner's own RNG never reseeded per episode (resume not reproducible)

- **Date:** 2026-08-28
- **Defect:** `GC_Agent.__init__` (upstream, `gc_agent.py:42-45`) seeds
  `self.local_gpu_generator` ONCE from `cfg.local_seed` at construction;
  `CEMPlanner.plan()` (upstream, `planner.py:290-291`) draws candidate
  action sequences from this SAME generator object (passed by reference,
  `gc_agent.py:63`), and nothing in `atlas/harness_e4.py` or
  `scripts/run_e4.py` ever reseeded it per episode. Since `agent` (and its
  planner/generator) is built ONCE before the whole arm/seed_run/episode
  loop, CEM's candidate sampling for episode *N* depended on how many
  prior draws happened earlier in the SAME process — not on
  `spec.seed`. A resumed run reaching episode *N* has a different draw
  history (and thus generator state) than an uninterrupted run reaching
  the same episode, so resumed and uninterrupted runs would silently
  diverge in their CEM plans despite every other input (env seed, chart
  state) being identical.
- **Change (`atlas/harness_e4.py::run_e4_episode`, does NOT touch the
  upstream `gc_agent.py`/`planner.py` files — CLAUDE.md §1.3's "one
  upstream hook only" — this calls a public `manual_seed()` method on an
  object already passed in):** at the top of every episode,
  `agent.local_gpu_generator.manual_seed(spec.seed)` (+
  `agent.local_generator.manual_seed(spec.seed)` for the CPU-side
  generator, guarded with `hasattr` for forward compatibility).
- **Claims affected:** reproducibility of any resumed E4/S2 run — E4 has
  never run.
- **Re-run needed:** no.
- **Verification:** `scratchpad/assert_b11.py` — `frozen` arm (no
  library/adaptation, isolates pure CEM-sampling reproducibility), same
  `spec.seed` run twice: once FIRST (fresh generator), once AFTER a
  "decoy" episode with a different seed (simulating a resumed run's
  different prior-draw history reaching the same episode).
  BEFORE (temporarily reverted the reseed calls to a dead `if False:`
  branch): `target episode run FIRST: (88.018, 10)`, `target episode run
  AFTER a decoy: (83.167, 10)` → **DIFFERENT block_pos_diff for the
  IDENTICAL episode spec** → `RESULT: DIFFERS depending on prior draws
  (B11 bug reproduced)`. **Failed before fix: yes.**
  AFTER (fix restored): `target episode run FIRST: (90.4377, 10)`,
  `target episode run AFTER a decoy: (90.4377, 10)` — bit-identical →
  `RESULT: IDENTICAL regardless of prior draws (B11 fixed)`. Also re-ran
  `scripts/smoke_e4.py` — still `All E4/E3 smoke-test assertions passed.`
- **Verified by:** orchestrating session, direct run (before/after shown
  side by side above).

### B14 — no `Library.evict()`; cap-hits were invisible in per-episode records

- **Date:** 2026-08-28
- **Defect:** `atlas/library.py::Library.add()` raises `RuntimeError` when
  full; both production callers (`atlas/loop.py`'s `'atlas'`/
  `'detect_only'` branches) pre-check `is_full()` before ever calling
  `add()`, so growth simply HALTS at `k_max=10` and nothing already
  committed is ever retired or replaced. `Library.evict()` does not
  exist. Per FIX_SPEC.md B14, the fix is NOT to implement an eviction
  policy (out of scope, to be stated plainly in the paper as unimplemented
  — a Phase 6 write-up item) but to make cap-hits VISIBLE in the logged
  record, currently only inferable by comparing `library_size` against
  `k_max` externally.
- **Change (`atlas/harness_e4.py`):** added a `"library_full": bool(
  state.library.is_full()) if state.library is not None else False`
  field to every episode record. Added `"library_full"` to
  `scripts/smoke_e4.py`'s `REQUIRED_KEYS` so the JSONL-contract check
  covers it going forward.
- **Claims affected:** none directly (visibility only) — protects any
  future RQ3/library-cap analysis from having to externally recompute
  cap-hits.
- **Re-run needed:** no.
- **Verification:** `python scripts/smoke_e4.py` real run, inspected
  `atlas_out/e4_smoke/episodes.jsonl` directly: `frozen library_size=1
  library_full=False` (×4), `atlas_fixed library_size=2 library_full=
  False` (×4) — correct, since 2 < `k_max=10`. Not a "must fail before"
  row (B14 is additive logging, not a defect fix — there is no incorrect
  prior VALUE to reproduce, only a previously-absent field) — verified by
  direct inspection that the field is present and computed correctly, and
  `scripts/smoke_e4.py`'s `[6/6]` JSONL-contract check (which DOES fail
  loudly on a missing required key) passes with `"library_full"` now
  required.
- **Verified by:** orchestrating session, direct run.

---

## Discovered, not fixed

### Phase 3 Step 2 (2026-08-28)

- **`atlas/harness_e4.py:378-383`'s persistent per-chart Adam optimizer
  construction (used by `atlas_fixed`/`atlas_detect`/`atlas` arms across
  replans/episodes) has the SAME two defects B4/C6 fixed in
  `atlas/loop.py::atlas_refine`/`atlas/expand.py::_fit_candidate`, but at
  a THIRD, undocumented call site FIX_SPEC.md's B4/C6 register rows do
  not name.** Specifically: `params = [p for n, p in
  predictor.named_parameters() if n in
  state.library[state.current_idx]._param_names]` (no `kind=="lora4"`
  suffix branch — will select ZERO params and raise `optim.Adam([])` for
  any `lora4` chart reaching this line) and no `requires_grad_(True)`
  call before `torch.optim.Adam(params, lr=state.cfg.lr)` (though B4's
  fix inside `atlas_refine` itself re-derives and re-enables
  `requires_grad` on ITS OWN `params` list every call regardless of
  whether an externally-built `optimizer` is passed in, so for `ln_act`/
  `full` — where both filters select the same underlying tensor objects —
  this specific path happens to work correctly today; only `lora4`
  combined with `atlas_fixed`/`atlas_detect`/`atlas` and `current_idx !=
  0` is live-broken). Not fixed: this call site is not named in
  `SUBMISSION_PLAN.md`/`FIX_SPEC.md`'s B4 or C6 rows (which name only
  `atlas/loop.py::atlas_refine` and `atlas/expand.py::_fit_candidate`),
  and per this session's standing instruction ("if you find a new bug...
  document it... and stop rather than fixing it"), fixing an undocumented
  third site is out of scope for this pass. Flagged here with exact
  `file:line` for a future authorised fix.

---

### Phase 1 Tier A Stage 1 (2026-08-27)

- **`ATLAS_SUMMARY.md` §4.1 quotes analytic Kendall p-values** (`p=4.4×10⁻⁷`,
  `p=9.6×10⁻⁸`) that A7 replaces with permutation p-values (`~1.0e-4`, still
  well below 0.05, direction/point-estimates unchanged). Not edited in Stage 1
  — doc scope was A14/A15 only. Flag for the Phase 6 `PAPER_DRAFT.md` /
  `ATLAS_SUMMARY.md` supersede pass. (`ATLAS_SUMMARY.md:111`)
- No new code defects found while implementing Stage 1. The `sdyn` +
  `current_score <= 0` interaction noted under A1 is a consequence of the
  authorised A1 formula, not a separate bug — recorded there.

---

### Phase 1 Tier A Stage 2 — re-runs (2026-08-28)

Dispatched: the deferred re-runs for A1+A2+A3 (combined E2 re-run, every
config, new q=3 run) and A4 (E0 forward-only re-score on the disjoint test
split). All local (CPU/GPU on the working machine — no Modal needed, per
`FIX_SPEC.md`'s own cost note that E2 runs no CEM planner and A4 is
forward-only). Nothing overwrote an existing `atlas_out/e2*` or `atlas_out/
e0*` directory; every run below landed in a new, descriptively-named
directory with a `_phase1stage2_2026-08-28` suffix.

### A1+A2+A3 — combined E2 re-run (Stage 2 complete)

- **Date:** 2026-08-28.
- **What ran:** every config named in FIX_SPEC.md's A3 re-run note —
  `e2_R1` (ln_act×R1), `e2_R1_lora4` (lora4×R1), `e2_R2` (ln_act×R2,
  primary/"decisive"), `e2_R2_cellB_q1` (q=1 positive control),
  `e2_R2_cellC_q1` (q=1 over-expansion control), `e2_confusion_matrix`
  (3-chart {c0, chart_R1, chart_R2}). All under the now-fixed A1 hysteresis
  normaliser, A2 incumbent selection, and A3 per-decision logging. The
  "new q=3 run" FIX_SPEC.md A3 asks for (closing `PAPER_FACT_CHECK` C2's
  "q=3 column is inferred, not measured" gap) is `e2_R2` itself — it runs
  cells A-D at the pre-registered `probe_q=3` default and its
  `charts_committed` field for Cell C (0, real, not inferred) is now a
  directly measured q=3 number, not the pre-fix documentation's
  probabilistic argument.
- **Artifacts (all new dirs):** `atlas_out/e2_R1_phase1stage2_2026-08-28/`,
  `atlas_out/e2_R1_lora4_phase1stage2_2026-08-28/`,
  `atlas_out/e2_R2_phase1stage2_2026-08-28/`,
  `atlas_out/e2_R2_cellB_q1_phase1stage2_2026-08-28/`,
  `atlas_out/e2_R2_cellC_q1_phase1stage2_2026-08-28/`,
  `atlas_out/e2_confusion_matrix_phase1stage2_2026-08-28/` — each with
  `e2_episodes.jsonl` (now carrying A3's `record_type="expansion"` rows:
  `strikes`, `probe_outcome`, `relative_gap`, `hysteresis_held`,
  `committed`, `incumbent_debug`, `library_size`) and `e2_summary.json`.
- **Result — how many E2 decisions changed (A2's report requirement):**
  routing accuracy changed at every config tested (see table below); the
  A2 incumbent-selection fix itself is visible per-record in
  `incumbent_debug.incumbent_changed` (added by A2, now populated for the
  first time by this re-run — not separately tabulated here, since the
  dominant driver of the accuracy change is A1, confirmed below).
  | Config | Cell B/decisive UMF (old→new) | S-dyn (old→new) |
  |---|---:|---:|
  | `ln_act`×R1 | 0.642→**0.481** | 0.543→**0.481** |
  | `lora4`×R1 | 0.642→**0.494** | 0.494→**0.481** |
  | `ln_act`×R2 | 0.833→**0.419** | 0.570→**0.419** |
  | 3-chart confusion matrix | 0.603→**0.298** | 0.365→**0.294** |
  Cell C (over-expansion): 0 charts committed in every config, both before
  and after — unaffected.
- **sdyn side effect (flagged under A1, checked here as required):**
  directly confirmed from raw JSONL — in `e2_R2_phase1stage2_2026-08-28`'s Cell B
  (240 decisions, both conditions), **both** `sdyn` and `umf` select chart
  index 0 in 100% of decisions (`Counter({0: 240})` for each, verified by
  direct `Counter()` over the `selected` field). The predicted "current_score
  <= 0 -> always HOLD" side effect is confirmed for `sdyn` as expected — but
  the same collapse-to-always-HOLD now also happens for `umf` at this cell
  (its own current-score-normalised hysteresis rarely clears 5% once R2's
  chart gap is this close), which is the actual mechanism behind the
  accuracy convergence, not merely `sdyn`'s known side effect in isolation.
- **Claims affected:** C-1, C-2, N9 — ATLAS_SUMMARY.md §4.5 and
  E2_RESULTS.md's headline both directly contradicted by this re-run.
- **Superseded numbers:** `ATLAS_SUMMARY.md` §4.5 (dated banner added,
  2026-08-28) and `E2_RESULTS.md` (dated banner added at the top of the
  file, 2026-08-28) — both now state the numbers above and point at these
  new artifact paths; old numbers kept below the banners, not deleted.
- **Verification:** re-run itself IS the assertion FIX_SPEC.md A2 specifies
  ("log both old and new incumbent index... report how many of E2's
  decisions change") — satisfied by the table above (every config's
  accuracy changed) plus the sdyn/umf selected-index check. Not a
  must-fail-before pattern (A1/A2's own unit assertions already proved
  fail-before/pass-after in Stage 1; this is the integration re-run those
  unit fixes were deferred to).
- **Verified by:** orchestrating session, direct run + direct JSONL
  inspection (both shown above).

### A4 — E0 test-set re-score (Stage 2 complete)

- **Date:** 2026-08-28.
- **What ran:** `python scripts/run_e0.py --regimes R0 R1 R2 --kinds ln_act
  lora4 full --out atlas_out/e0_a4_rescore_phase1stage2_2026-08-28` against a COPY
  of `atlas_out/e0`'s 9 already-trained chart `.pt`/loss-json files (copied
  first so the new `--out` dir triggers the `[Resume]` forward-only
  re-evaluation branch instead of retraining) — no retraining, no CEM
  planner, matches FIX_SPEC.md A4's "~$1" cost estimate.
- **Artifact:** `atlas_out/e0_a4_rescore_phase1stage2_2026-08-28/` (`results.json`,
  `e0_seed_manifest.json`, `results.md`, loss curve plots — chart `.pt`
  files themselves are the copied originals, untouched).
- **Assertion 1 — manifest shows three disjoint seed ranges:** confirmed
  per-regime from `e0_seed_manifest.json`: R0 train seeds 2000-2152 (n=20),
  eval 12000-12056 (n=8), test 22000-22056 (n=8); R1 train 0-152, eval
  10000-10056, test 20000-20056; R2 train 1000-1152, eval 11000-11056, test
  21000-21057. All three ranges non-overlapping in every regime.
- **Assertion 2 — `test_umf` (`eval_umf`) and `val_umf` both in
  `results.json`:** confirmed, all 9 chart×regime cells. Bias (`eval_umf -
  val_umf`) is directly computable and mixed-sign: R0 all positive (+0.077
  to +0.157, test UMF worse than val — early-stopping selection bias
  inflated the reported val number), R1 mixed (-0.071 to +0.018), R2 mostly
  positive (+0.027 to +0.044) except `full` (-0.023). Largest bias:
  `full`×R0, val_umf=0.713 vs. test_umf=0.870 (+0.157, i.e. the
  checkpoint-selection set understated the real UMF by 0.157 — a
  substantial, previously invisible optimistic bias).
- **Claims affected:** N4 and every E0 UMF number (bias is now measurable,
  not merely asserted, per FIX_SPEC.md A4's own framing).
- **Re-run needed:** none further — this IS the re-run A4's Stage 1 entry
  deferred.
- **Superseded numbers:** none in the summary docs edited this pass (E0's
  headline UMF numbers in `ATLAS_SUMMARY.md`/`E0_RESULTS.md` already cite
  `atlas_out/e0`'s `eval_umf`, which was itself the val-set number before
  A4; a full re-statement of every E0 UMF citation across both docs to the
  new test-set number is flagged here as NOT done this pass — out of the
  narrow A1-A4 scope given to this dispatch — and should be a Phase 6
  write-up item).
- **Verification:** both assertions shown above with real values from the
  regenerated `results.json`/`e0_seed_manifest.json`, not summarised.
- **Verified by:** orchestrating session, direct run + direct JSON
  inspection.

### A1+A2+A3 — Stage 2 correction: `*_posthysteresis` configs and the real q=3 gap (2026-08-28, same day, coordinator-flagged)

The Stage 2 pass above did not run the three `*_posthysteresis` configs
(these record an EARLIER prior state — the 2026-08-26 sequential-hysteresis
current_idx-carry-forward fix — that itself needed today's A1/A2/A3 fix
layered on top), and incorrectly treated `e2_R2_phase1stage2_2026-08-28` as
satisfying FIX_SPEC.md A3's "new q=3 run" requirement, when that run's
`probe_q=3` was already the pre-existing default and measured nothing new
relative to PAPER_FACT_CHECK C2's actual gap (Cell B/R2's `charts committed
at q=3` column, historically inferred not measured, in the isolated
single-cell/single-router protocol matching the existing q=1 controls).
Both gaps closed this pass, orthogonal to the earlier report — no code
touched, only additional runs.

- **What ran:** exact configs reproduced from each `*_posthysteresis`
  directory's own `e2_summary.json` `config` block (all `probe_q=3,
  probe_tau=0.5, corruption=dark@0.5`, only kind/chart_regime/dynamics_regime
  varying) — `e2_R1_posthysteresis` (ln_act×R1), `e2_R1_lora4_posthysteresis`
  (lora4×R1), `e2_R2_posthysteresis` (ln_act×R2) — plus one new run,
  `e2_R2_cellB_q3` (`--cells B --routers umf --probe-q 3`, matching
  `e2_R2_cellB_q1`'s exact single-cell/single-router protocol but at the
  pre-registered q=3, never run in that isolated form before).
- **Artifacts (new dirs, none overwriting the earlier Stage 2 dirs or the
  original pre-A1/A2/A3 `*_posthysteresis` dirs):**
  `atlas_out/e2_R1_posthysteresis_phase1stage2_2026-08-28/`,
  `atlas_out/e2_R1_lora4_posthysteresis_phase1stage2_2026-08-28/`,
  `atlas_out/e2_R2_posthysteresis_phase1stage2_2026-08-28/`,
  `atlas_out/e2_R2_cellB_q3_phase1stage2_2026-08-28/`.
- **Result — `*_posthysteresis` re-runs:**
  | Config | Cell B UMF (old posthysteresis → new) | S-dyn (old → new) | Charts committed B |
  |---|---:|---:|---:|
  | `ln_act`×R1 | 0.6420 → **0.4815** | (n/a, umf-only historically; new run both) → **0.4815** | 0 → **0** |
  | `lora4`×R1 | (n/a in original) → **0.4938** | → **0.4815** | 0 → **0** |
  | `ln_act`×R2 | 0.8333 → **0.4194** | → **0.4194** | (n/a) → **2** |
  Routing-accuracy numbers are **bit-identical** to the plain (non-posthysteresis)
  Stage 2 re-runs reported earlier this pass (`e2_R1_stage2`,
  `e2_R1_lora4_stage2`, `e2_R2_stage2`) — same seeds, same deterministic
  scoring path, confirming the accuracy collapse is real and reproducible
  across both the original 2026-08-26 sequential-hysteresis code path and
  its later restatement.
- **Discovered, not fixed — real nondeterminism in commit counts across
  nominally-identical q=3 R2/Cell-B re-runs:** `e2_R2_phase1stage2_2026-08-28`
  (plain) → 0 charts committed at Cell B; `e2_R2_posthysteresis_phase1stage2_2026-08-28`
  (identical config, same code, re-run minutes later) → **2** charts
  committed at Cell B; `e2_R2_cellB_q3_phase1stage2_2026-08-28` (isolated
  single-cell/single-router form of the same config) → 0 committed again.
  Routing ACCURACY is identical across all three (0.4194 in every case,
  since accuracy is computed independent of whether a commit happened), so
  this does not affect any number reported in this pass's tables — but the
  commit count itself is not deterministic run-to-run under nominally
  identical config. Flagged per FIX_SPEC.md rule 5 rather than fixed — out
  of A1-A4's scope, and does not change any accuracy number this dispatch
  reports.
  **Root cause, confirmed by the orchestrating session (2026-08-28), superseding
  the agent's "most likely CUDA/cuDNN" guess with a verified mechanism:**
  - *Dropout ruled out.* Loaded the real checkpoint directly and checked:
    `predictor.training == False` and `encoder.training == False`. Something
    in `jepa-wms`'s own load path already puts the model in eval mode (nothing
    in `atlas/` calls `.eval()`/`.train()` anywhere — confirmed by
    `grep -rn ".eval()\|.train()" atlas/ scripts/`, no matches outside
    unrelated `.pyc` files). So the `Dropout(p=0.1)` layers in the predictor's
    attention/feedforward blocks are inactive; this is not the source.
  - *Confirmed: unseeded CUDA gradient descent.* This machine has a real CUDA
    GPU (`torch.cuda.is_available()==True`, RTX 4050 Laptop), and
    `run_e2.py:171` picks `device = "cuda" if torch.cuda.is_available() else
    "cpu"` — the Stage 2 runs genuinely ran on GPU. `atlas/expand.py::
    _fit_candidate` (`:234-244`) runs `n_steps` of `optimizer.step()` —
    gradient descent on the candidate chart's weights through the predictor's
    attention layers — with **no `torch.manual_seed()`** anywhere in
    `run_e2.py`, `expand.py`, or `score.py`, and **no
    `torch.use_deterministic_algorithms()` / `cudnn.deterministic=True`**
    set anywhere in the codebase. PyTorch's CUDA backward kernels for
    attention/embedding-style ops (scatter/index-add-based gradient
    accumulation) are non-deterministic by default without those flags, so
    the same config run twice fits a slightly different candidate chart each
    time. That explains the observed effect exactly: two runs of the
    identical config gave different UMF on the verification chunk near the
    `τ=0.5` commit threshold, flipping the commit count while accuracy (not
    threshold-sensitive the same way) stayed stable.
  - **Why this matters beyond cosmetic:** `expand.py`'s probe-commit
    mechanism is what Gates G3a/G3b exercise and what E2's "N charts
    committed" (claim N9) and Phase 5's continual-stream expansion logic
    depend on. A nondeterministic commit/reject decision on the same input
    is a reproducibility problem for any of those numbers, not just this
    Cell B run.
  - **Not fixed.** The likely fix (seed `torch.manual_seed` from the
    episode's own seed before `_fit_candidate`'s training loop, matching the
    deterministic-trajectory-seeding pattern already used elsewhere in
    `run_e2.py`) is small but is a new registered item, out of Tier A's
    A1-A15 scope — needs its own FIX_SPEC entry and explicit authorization
    before landing.
- **Result — new q=3 Cell B run (closes `PAPER_FACT_CHECK` C2's gap):**
  `atlas_out/e2_R2_cellB_q3_phase1stage2_2026-08-28/e2_summary.json` →
  `{"umf": {"B": 0.4194}, "charts_committed": {"B": 0}}`,
  `probe_params_are_preregistered: true`. This is now a genuinely MEASURED
  q=3 zero for Cell B (not the probabilistic "~0.4% event, not run" argument
  `PAPER_FACT_CHECK.md` C2 flags) — directly comparable in protocol to
  `e2_R2_cellB_q1`'s existing q=1 positive control (5 committed, per the
  earlier Stage 2 entry above).
- **Claims affected:** same as the A1+A2+A3 row above (C-1, C-2, N9); the new
  q=3 measurement also closes `PAPER_FACT_CHECK` C2 specifically.
- **Superseded numbers:** `E2_RESULTS.md`'s supersede banner (added earlier
  this pass) updated in place to add these four artifacts and the
  nondeterminism finding; `ATLAS_SUMMARY.md` §4.5 banner unchanged (already
  states the collapse; the `*_posthysteresis` re-runs confirm it, they do
  not change the number cited there).
- **Verification:** all four real runs, real stdout, `e2_summary.json`
  contents pasted above (not summarised) in this session's report.
- **Verified by:** orchestrating session, direct run + direct JSON
  inspection, prompted by the coordinator's independent `atlas_out/` diff.

### A9 — 20-trajectory chart retrain, reproduction check (Stage 2 complete)

- **Date:** 2026-08-28.
- **Defect:** `atlas_out/e0_v3_dataset/` (`ln_act`×R2, the chart behind N4's
  20-trajectory row and N1) has only the chart and `results.json` — no seed
  manifest, held-out status resting on prose (`FIX_SPEC.md` A9). This item
  was missed from both the original Stage 1 and Stage 2 dispatches (it was
  not in the explicit item list either dispatch enumerated) and only run
  after the user flagged the omission directly.
- **What ran:** `python scripts/run_e0.py --kinds ln_act --regimes R2 --out
  atlas_out/e0_a9_retrain_phase1stage2_2026-08-28` (defaults: 20 train
  trajectories, `--data-source dataset`, matching `e0_v3_dataset`'s original
  recipe) — a full retrain, not a re-score, with the seed manifest saved.
- **Result:** manifest confirms disjoint train/val/test seed ranges (train
  1000-1152, val 11000+, test 21000+). The reproduction check that A9 asks
  for **passes**: the same quantity the original 0.3357 measured is now
  called `val_umf` post-A4 and came back **0.3335** — within noise.
  **A second number, never measured before, surfaced in the same run and is
  not a reproduction check — it is new information:** this chart's `eval_umf`
  on A4's genuinely disjoint test split is **0.4125**, ~0.08 worse than the
  0.3357/0.336 reported everywhere for this chart. The gap exists because
  0.3357 was measured on the set also used for checkpoint selection — the
  same optimistic-bias mechanism A4 documented elsewhere in E0, now confirmed
  to apply to this specific, oft-cited chart too.
- **Claims affected:** N4 (the 20/60/100-trajectory UMF trend table) — the
  trend itself (0.336→0.302→0.268) is **not** disturbed by this check, since
  it reproduces the same (val-set) quantity the trend was built from. But
  every number in that trend is now known to be measured pre-A4-fix, i.e. on
  the checkpoint-selection set, and this run is the first direct evidence of
  how large that bias actually is for one of the three points (60- and
  100-trajectory charts have not been re-checked against a disjoint test set
  as of this entry).
- **Re-run needed:** no further action required for A9 itself; re-checking
  the 60- and 100-trajectory rows against a disjoint test set is a natural
  follow-on, not yet authorised or scheduled.
- **Superseded numbers:** `ATLAS_SUMMARY.md` §4.2 and `E0_RESULTS.md`'s
  20/60/100 table (E0 capacity section) both got a dated banner directly
  below/beside the affected number — old numbers kept, not deleted, both
  facts (reproduces / true OOS number is worse) stated together per the
  user's instruction not to quietly adopt a new number in place of an old
  one without disclosing both.
- **Verification:** per `FIX_SPEC.md`'s explicit instruction for A9 ("if it
  does not reproduce, stop and report — do not quietly adopt the new
  number") — it does reproduce (val_umf), reported as such; the new,
  non-reproduction number (eval_umf) is reported alongside it, not silently
  substituted for 0.336 anywhere.
- **Verified by:** orchestrating session, direct run (`run_e0.py`) + direct
  `results.json`/manifest inspection (both pasted in full to the user, not
  summarised).

### Phase 4 — E-A / E-C code prerequisites (code-only, no runs), 2026-08-28

Both items below are **code prerequisites only**, dispatched ahead of the
actual E-A/E-C experiment runs (`SUBMISSION_PLAN.md` Part C, ~$6 and ~$9
respectively). No GPU job beyond `--help`/syntax-check was run. Both new
flags default OFF/matched such that omitting them reproduces prior behavior
exactly; this was not independently re-verified by a full before/after run
(see "Re-run needed" below) because doing so would itself be the (out of
scope, not-yet-authorised) experiment run.

### E-A — `--save-latents` on `scripts/diagnose_cem_costs.py`

- **Date:** 2026-08-28.
- **Defect:** `scripts/diagnose_cem_costs.py::rollout_true_outcomes` (previously
  lines 98-124) rolls every CEM candidate out for real but only ever saved
  the scalar cost and true final distance — never the encoder outputs needed
  to later compute UMF against the planner's own query distribution
  (`SUBMISSION_PLAN.md` Part C, E-A).
- **Change:** Added `from atlas.score import rollout_umf` (import only,
  `atlas/score.py` untouched). Added a new function
  `rollout_true_outcomes_and_umf()` (`scripts/diagnose_cem_costs.py`, after
  the original `rollout_true_outcomes`) that performs the identical real
  rollout but additionally encodes the real visual/proprio observed at every
  model-chunk boundary (matching `run_e0.py::load_regime_trajectories`'s
  subsampling convention) and calls `atlas.score.rollout_umf` (not `umf()`,
  since the kind's chart, if any, is already applied to `wm.predictor` for
  the whole per-kind block by `main()` — matches `rollout_umf`'s
  already-applied-predictor contract) to get a per-candidate UMF value.
  Added `--save-latents` (`store_true`, default off). When set, `main()`
  calls the new function instead of the original and stores the result under
  three NEW keys in `results[kind]`: `umf_per_candidate` (list, `None` for
  any motion-gated candidate), `umf_mean`, `umf_n_gated`. **Scope decision,
  not a bug:** raw per-candidate encoder-output latents themselves are NOT
  persisted (300 candidates x 7 frames x 256 x 384 floats per seed per kind
  is ~800MB and does not fit this script's existing per-seed
  JSON/incremental-jsonl format) — per the dispatch spec's own "or
  equivalently the raw prediction error" alternative, only the resulting
  scalar UMF is saved. A downstream fitting step (E-A's "fit `ln_act` with
  the existing `run_e0_finetune` loss") needs the raw transitions and would
  reuse `run_e0.py`'s own trajectory-collection path, not this function —
  flagged inline in the new function's docstring for whoever runs E-A next.
  The original `rollout_true_outcomes` function is completely untouched;
  `main()`'s `else` branch (flag unset) calls it exactly as before, with the
  same arguments, so default output is unaffected by this change.
- **Claims affected:** E-A (`SUBMISSION_PLAN.md` Part C) — no claim ID
  assigned yet (experiment not yet run); will map to a new N-id once E-A
  actually runs and produces a `cost_ranking_*` file with `--save-latents`.
- **Re-run needed:** yes, but explicitly NOT launched by this dispatch (out
  of scope per the dispatching instructions — code-only task). The actual
  E-A collection (~30 seeds x 300 candidates under R2) is unauthorized GPU
  spend until a future session/user explicitly approves it.
- **Verification:** `python -c "import ast; ast.parse(...)"` on the file —
  passed (`SYNTAX_OK`). `.venv/Scripts/python.exe scripts/diagnose_cem_costs.py
  --help` — ran to completion, printed full usage including the new
  `--save-latents` flag and its help text (pasted in full in this session's
  report). No before/after functional assertion was run (per rule 3, an
  assertion needs runnable steps; `--help`/parse-only is the maximum allowed
  under this dispatch's explicit "do NOT launch a real collection run"
  instruction — the correctness of `rollout_umf`'s output on real candidates
  is unverified until E-A actually runs).
- **Verified by:** this session, direct command output (pasted, not
  summarised).

### E-C — `--collect-num-act-stepped` on `scripts/run_e0.py`

- **Date:** 2026-08-28.
- **Defect:** `scripts/run_e0.py`'s `closed_loop` collector hardcoded
  `num_act_stepped=1` with no flag (previously line 622) and defaulted
  `--collect-num-samples`/`--collect-iterations` to 100x10 (previously lines
  540/547) against an eval-side budget of 300x30 (`run_e0_planning.py`'s own
  `--num-samples`/`--iterations` defaults) — a 9x search-budget gap between
  collection and evaluation (`SUBMISSION_PLAN.md` Part C, E-C).
- **Change:** `scripts/run_e0.py` — changed `--collect-num-samples` default
  100→300 and `--collect-iterations` default 10→30 (both flags already
  existed; only their defaults changed, matching `run_e0_planning.py`'s own
  `--num-samples`/`--iterations` defaults exactly, read from that file rather
  than guessed). Added a new flag `--collect-num-act-stepped` (default 6,
  matching `run_e0_planning.py`'s own `--num-act-stepped` default exactly)
  and threaded it into the `GC_Agent`/`build_cfg` call that previously
  hardcoded `num_act_stepped=1`. Updated the adjacent print statement and
  code comments accordingly.
  **Important caveat discovered while wiring this (documented, not
  "fixed" — out of this dispatch's scope):** the `closed_loop` collection
  loop in `load_regime_trajectories()` (`scripts/run_e0.py`, the
  `elif source == "closed_loop":` branch) unconditionally replans every
  single model chunk (`for chunk_idx in range(n_chunks): ... agent.act(...)`)
  and always executes only the first `frameskip` raw actions of whatever the
  planner returns (`act_chunk[:frameskip]`), and CEM's internal
  `plan_length = min(horizon, steps_left)` does not depend on
  `num_act_stepped` at all (`CEMPlanner.plan`,
  `hub/hub/facebookresearch_jepa-wms_main/evals/simu_env_planning/planning/planning/planner.py:272-276,333`).
  So `--collect-num-act-stepped`, as now wired, changes only how many
  already-planned actions `CEMPlanner.plan()` *returns*
  (`actions[:self.num_act_stepped]`) — all but the first `frameskip` of
  which `run_e0.py`'s loop discards unused — and does **not** change the
  collector's actual replan cadence or the resulting trajectory at all.
  `SUBMISSION_PLAN.md`'s C-2 table's "Replans/episode: 6" for collection is
  produced by this loop's own unconditional per-chunk structure, not by
  `num_act_stepped=1`. This means: (a) the dispatch's literal instruction —
  default `--collect-num-act-stepped` to the eval default (6) — was followed
  exactly, but (b) that default value functionally has **no effect** on
  collection behavior given the current loop structure, so it does not by
  itself close the "two opposite extremes of replan frequency" mismatch
  `SUBMISSION_PLAN.md` describes for E-C; only the 300x30 budget-match
  actually changes collection behavior. Left as documented, not fixed —
  changing the loop's execution logic (e.g. to actually consume more than
  `frameskip` actions per replan when `num_act_stepped>1`) would be a
  behavioral redesign beyond "thread a flag through, remove the hardcode,"
  which is all this dispatch authorized.
- **Claims affected:** E-C, N5/`closed_loop` numbers (`SUBMISSION_PLAN.md`
  Part C) — no new claim ID yet (experiment not yet re-run).
- **Re-run needed:** yes, explicitly NOT launched by this dispatch. Given the
  caveat above, whoever runs E-C next should re-read this entry before
  assuming `--collect-num-act-stepped`'s default value does anything to
  collection behavior beyond changing `GC_Agent`'s returned-action-tensor
  size.
- **Verification:** `python -c "import ast; ast.parse(...)"` — passed
  (`SYNTAX_OK`). `.venv/Scripts/python.exe scripts/run_e0.py --help` — ran to
  completion (required `import atlas` to resolve, confirmed via
  `PYTHONPATH=.`), printed full usage including
  `--collect-num-samples`/`--collect-iterations`/`--collect-num-act-stepped`
  with their updated help text (pasted in full in this session's report). No
  functional before/after run — out of scope per this dispatch.
- **Verified by:** this session, direct command output (pasted, not
  summarised).

### Orchestrator correction — `--collect-num-act-stepped` default was wrong, 2026-08-28

- **Defect:** the E-C dispatch above followed its own literal instruction
  (default `--collect-num-act-stepped` to the eval-side value, 6) — but that
  instruction was a mis-transcription of `SUBMISSION_PLAN.md`'s actual E-C
  spec on the orchestrating session's part, not a considered design
  decision. `SUBMISSION_PLAN.md` Part C (E-C) and `research_audit/PHASE_PROMPTS.md`
  Step 4e are both explicit: re-collect at the eval-matched 300x30
  sample/iteration budget while **keeping nas=1**, so that replan frequency
  remains the *one* deliberate collection/eval mismatch left standing —
  exactly the axis E-D (Step 4d) is designed to measure separately. A
  default of 6 would silently remove that one remaining controlled
  difference before E-D ever runs.
- **Fix:** `scripts/run_e0.py`'s `--collect-num-act-stepped` default changed
  6 → 1. Help text updated to state the spec's intent explicitly and warn
  against overriding it for the E-C re-collection. The flag itself (not a
  hardcode) is unchanged from the prior entry — only its default value and
  documentation changed. The prior entry's finding that this flag has **no
  effect on collector behavior at any value**, given the current
  `load_regime_trajectories()` loop structure, still stands and is
  unaffected by this correction — it was never contingent on which default
  was chosen.
- **Claims affected:** E-C, N5/`closed_loop` (same as prior entry) — same
  "no claim ID yet, experiment not yet run" status.
- **Re-run needed:** yes, still not launched.
- **Verification:** `python -c "import ast; ast.parse(open('scripts/run_e0.py', encoding='utf-8').read())"` → `SYNTAX_OK`.
- **Verified by:** orchestrating session, direct diff read of the
  `atlas-fixer` dispatch's actual change against `SUBMISSION_PLAN.md`'s own
  text (not the dispatch's self-report) — caught the mis-default before any
  experiment consumed it.

---

### COHERENCE_AUDIT_2 write-up + tooling pass (docs / standalone-tooling only), 2026-08-28

Scope: the four items `COHERENCE_AUDIT_2.md` (PART 1) flags as write-up /
tooling defects that v3's `IMPLEMENTATION_PLAN_V3.md` deliberately leaves out
of the experimental plan. **No file under `atlas/`, no `scripts/run_*.py`, no
`configs/atlas/*` touched. No experiment re-run, no number in `atlas_out/`
changed.**

#### CA2-1 — S-dyn mislabelled as an "appearance-based" baseline

- **Date:** 2026-08-28
- **Defect:** `Paper_Draft/ATLAS_INITIAL_DRAFT.tex` (abstract, contributions
  bullet 2, §Experimental Setup, §Results 4.1, §Conclusion), `PAPER_DRAFT.md`
  (abstract L27, contributions L45), `ATLAS_SUMMARY.md` (§4.5 intro + table
  row + two later mentions), `research_audit/PAPER_DRAFT_NOTES.md` (§0 table,
  §4 heading + body, §6 draft abstract), `E2_RESULTS.md` (Headline L180) all
  describe S-dyn as "appearance-based" / "appearance-derived dynamics
  fingerprint" / "similarity to what the scene looks like" / "input
  similarity alone". `atlas/router.py::_sdyn_score` (read, not modified)
  computes `-cos_sim(z1_obs - z0, z1_hat - z0)` — negative cosine between the
  observed and the chart-predicted **first latent step** of the open-loop
  rollout. It is a one-step, direction-only, unnormalised variant of the same
  prediction signal UMF uses; it sees the scene only through the frozen DINO
  encoder, exactly as UMF does. The proposal's actual appearance-similarity
  router is **S-obs**, deferred to supplementary and never implemented
  (`grep` confirms no `_sobs` / `sobs` in `atlas/`).
- **Change:** every "appearance"/"input-similarity" description of S-dyn
  replaced with its true mechanism ("a one-step latent-direction / latent-delta
  cosine baseline"). The headline contrast is reframed throughout as
  *multi-step normalised prediction error (UMF)* vs. *one-step direction
  cosine (S-dyn)*, **not** "predictive fitness vs. appearance routing". Where
  a doc contrasted "dynamics shift vs. appearance shift" specifically about
  the **darkening-corruption expansion diagnostic** (a genuine appearance
  shift), that wording was left intact — it is correct there.
- **Files:** `Paper_Draft/ATLAS_INITIAL_DRAFT.tex:23,34,57,66,138`;
  `PAPER_DRAFT.md:27,45`; `ATLAS_SUMMARY.md:377-387,398-399,453-454`;
  `research_audit/PAPER_DRAFT_NOTES.md:22,78,80,126`; `E2_RESULTS.md:180`.
  `CLAUDE.md` checked — it names "S-dyn" and "S-obs router" but never calls
  S-dyn appearance-based, so no edit needed. `research_audit/EVIDENCE_LEDGER.md`
  N6/N9 checked — not mislabelled (N9's "appearance shift" = the darkening
  diagnostic), no edit.
- **Claims affected:** C-1, N6 (framing only — no number changed). Already
  flagged by `PAPER_FACT_CHECK.md` A6, `REDTEAM.md` N6 Attack 2,
  `COHERENCE_AUDIT_2.md` Agent 2 H1 / Agent 3 HIGH-2 / Agent 5 M1.
- **Re-run needed:** no.
- **Verification:** `grep -rn "appearance-based|appearance-derived|appearance-similarity|scene looks like|input similarity" Paper_Draft/ ATLAS_INITIAL_DRAFT.tex PAPER_DRAFT.md ATLAS_SUMMARY.md research_audit/PAPER_DRAFT_NOTES.md E2_RESULTS.md` → only remaining hits are about the darkening corruption (genuine appearance shift) or S-obs, none about S-dyn.
- **Verified by:** this session, direct grep + read of `atlas/router.py:170-198`.

#### CA2-2 — MBCD venue: ICML 2021 → AAMAS 2021

- **Date:** 2026-08-28
- **Defect:** `ATLAS_proposal_v7.md:56` — "MBCD (ICML 2021)". Alegre, Bazzan
  & da Silva, *Minimum-Delay Adaptation in Non-Stationary Reinforcement
  Learning via Online High-Confidence Change-Point Detection*
  (arXiv:2105.09452) was published at **AAMAS 2021** (Proc. 20th Int. Conf.
  on Autonomous Agents and Multiagent Systems). Confirmed via web search
  (arXiv page, the authors' own `people.cs.umass.edu` copy, the `mbcd`
  GitHub repo). A LatinX-in-AI @ ICML 2021 workshop version also exists,
  which likely seeded the error; the archival venue is AAMAS.
- **Change:** `ATLAS_proposal_v7.md:56` "ICML 2021" → "AAMAS 2021". The
  proposal's other three MBCD mentions (`:115`, `:123`, `:319` References)
  already say AAMAS 2021 — this was the last inconsistent one. `PAPER_DRAFT.md:58`
  already says "Alegre et al. 2021, AAMAS". `Paper_Draft/references.bib` does
  not cite MBCD (no bib entry, no `\cite`). No other file cites a venue for
  it.
- **Claims affected:** C1-novelty (Related Work accuracy). Flagged by
  `LITERATURE_AUDIT.md` §3/§10, `PAPER_DRAFT_NOTES.md` §5c, `NEXT_ACTIONS.md`,
  `COHERENCE_AUDIT_2.md` Agent 4.
- **Re-run needed:** no.
- **Verification:** `grep -rn "MBCD" *.md Paper_Draft/` → every surviving
  venue mention now reads AAMAS 2021.
- **Verified by:** this session, web search + grep.

#### CA2-3 — `scripts/make_tables.py` T5 renders a garbage table from superseded input

- **Date:** 2026-08-28
- **Defect:** `atlas_out/e0/T5.md` and its copy in
  `atlas_out/e0_a4_rescore_phase1stage2_2026-08-28/T5.md` show
  `Params = 26/12/69`, `Eval UMF = 0.67/1.30/1.67` — matching
  `atlas_out/e0/results.json` (a pre-A11 / pre-rollout-fix file whose
  `params` field records parameter-*group* counts, per "Deliberately NOT
  fixed" below) but **not** the corrected sibling `results.json` in the
  rescore dir (params `10764/10292640/20800884`, eval_umf `0.947/0.703/0.870`).
  Root cause is two latent `make_t5` defects, not the metric:
  1. `main()` defaults `--e0-dir` to `atlas_out/e0` — a directory the D4
     inventory marks SUPERSEDED. `make_tables.py --all` (or bare `--table T5`)
     silently regenerates the garbage table from it with no warning; the
     rescore-dir copy is a manual `cp` of that output (its mtime, 03:27,
     predates the rescore `results.json`'s 03:57).
  2. `make_t5` resolves the frozen-model reference UMF (for the `ΔUMF` and
     `% of full` columns — the pre-registered capacity metric) only by
     looking for a `c0`/`baseline`/`frozen`/`identity` **key inside
     `kinds`**. `run_e0.py` writes `results[regime]` as `{ln_act, lora4,
     full}` with no such key (confirmed at `run_e0.py:778,851`), so those two
     columns were structurally always `—`.
- **Change (`scripts/make_tables.py`, standalone table generator — not a
  `run_*.py`):**
  - New `_t5_resolve_baseline_umf()` — resolves the frozen UMF from, in
    order: a regime-level scalar key (`baseline_umf`/`frozen_umf`/`c0_umf`),
    a `c0`/`baseline`/`frozen`/`identity` per-kind row (unchanged path), or
    `{e0_dir}/frozen_baseline.json`. E0′ (§8.1: "frozen c₀ vs. ln_act" arm)
    can now feed `ΔUMF` / `% of full` by any of these.
  - Guard: if any non-baseline kind has a positive integer trainable-param
    count `< 1000` (`_T5_MIN_PLAUSIBLE_PARAMS` — smaller than `ln_act`'s
    ~10.7k), `make_t5` **raises `ValueError`** naming the file as
    superseded/pre-A11, instead of rendering it. `--all` passes
    `strict=False` → prints `[skip] …` and continues (doesn't abort the
    other tables); explicit `--table T5` raises.
  - Baseline / scalar rows are now skipped when emitting adapter rows (no
    self-referential `c0` row with `ΔUMF +0.0000`).
  - T5.md now carries a `_Source: <path> (mtime …)_` provenance line so a
    stale table is self-identifying.
  - `_print_markdown_table` gained an optional `note=` param (back-compatible;
    T1/T2 unaffected).
- **NOT done, per task scope:** the stale `atlas_out/e0/T5.md` and the
  rescore-dir copy were **not** regenerated — the correct E0 capacity table
  depends on v3's E0′, which has not run. The script is now correct and
  ready; `atlas_out/e0/results.json` stays as the archived record of what was
  run (already listed under "Deliberately NOT fixed").
- **Claims affected:** none directly (release-artifact tooling). The
  capacity/supplementary table (`ATLAS_proposal_v7.md` §7, T5) could not be
  generated correctly before this.
- **Re-run needed:** no (tooling only).
- **Verification:**
  - `python -m py_compile scripts/make_tables.py` → clean.
  - `make_tables.py --table T5 --e0-dir atlas_out/e0` → `ValueError: T5:
    atlas_out\e0\results.json looks like superseded / pre-A11 data … [('R0',
    'ln_act', 26), ('R0', 'lora4', 12), ('R0', 'full', 69)]`. **Failed
    before fix: yes** — pre-fix the same invocation wrote a 9-row T5.md with
    `Params 26/12/69`. Passes after (raises): yes.
  - `make_t5` on a synthetic well-formed E0′-shaped `results.json` (c0 +
    ln_act/lora4/full on R2, `baseline_umf` scalar + ln_act on R1) → correct
    `Params` (10,764 / 118,176 / 20,800,884), `ΔUMF` and `% of full`
    populated (`+0.0800 … 67%` etc.), no `c0` adapter row, provenance line
    present.
- **Verified by:** this session, direct command output.

#### CA2-4 — `scripts/audit_e0_train_planning_overlap.py` trips its own drift-guard

- **Date:** 2026-08-28
- **Defect:** the script (E0_RECOVERY_PLAN.md P2c's train/planning-eval
  disjointness verifier) fails on any manifest with
  `AssertionError: seed 2: reproduced init_state does not match
  sample_dataset_init_goal's own draw`. `reproduce_planning_episodes` replays
  the RNG call sequence of `run_e0_planning.sample_dataset_init_goal` to
  recover each planning seed's `episode_idx`, but its accept condition was
  `block_pos_diff >= min_block_pos_diff` only. The real function
  (`run_e0_planning.py:195`, P2d) added a second predicate
  `agent_block_dist <= max_agent_block_dist` (`DEFAULT_MAX_AGENT_BLOCK_DIST =
  160.0`). When the real function keeps retrying on the reachability filter,
  the old replay broke out one or more attempts early → different
  `ep_idx`/`offset` → assertion fires. So E0 train/eval disjointness was
  *asserted* (E0_RECOVERY_PLAN.md P2c) but not *verified*.
- **Change (`scripts/audit_e0_train_planning_overlap.py`, standalone audit
  tool — not a `run_*.py`):** import `DEFAULT_MAX_AGENT_BLOCK_DIST` from
  `run_e0_planning`; add `--max-agent-block-dist` CLI arg (default = the real
  function's default); `reproduce_planning_episodes` now mirrors the full
  two-predicate accept condition (computes `agent_block_dist` and requires
  both) and threads `max_agent_block_dist` into the real
  `sample_dataset_init_goal` call it makes for the ground-truth comparison.
- **Result of actually running it (now that it runs):**
  - `atlas_out/e0_a9_retrain_phase1stage2_2026-08-28` — **no overlap**
    (TRAIN∩planning = ∅, offline-EVAL∩planning = ∅) at 20 planning seeds.
  - `atlas_out/e0_v6_R1`, `atlas_out/e0_train_sweep_60` — **no overlap**.
  - `atlas_out/e0_train_sweep_100` — **1-episode overlap**: episode `235` is
    in both the 100-trajectory training set and the planning-eval set (at 20
    seeds and at the run's actual 40 seeds — `e0_planning_sweep_100/ln_act_R2.jsonl`
    has 40 episodes). 1/40 = 2.5%. This is a real, small leakage on the
    sweep-100 chart only. **Documented, not fixed** — per CLAUDE.md "do not
    fix a bug you find while auditing"; the sweep numbers already carry the
    A9 optimistic-bias caveat, and a chart trained on 100 trajectories is not
    dominated by one shared episode, but it should be stated. Whoever cites
    the training-data sweep (`ATLAS_SUMMARY.md` §4.2, `PAPER_DRAFT.md` §4.2,
    .tex §4.2) should note the sweep-100 point has a 2.5% train/eval episode
    overlap.
- **Is the check still meaningful under v3's on-policy collector (C.3)?**
  **Partly — its unit of comparison must change.** Today it compares the
  *replayed-demo `episode_idx`* used to fine-tune a chart against the
  `episode_idx` drawn for planning eval — correct for the `dataset`/`hybrid`
  collectors, which replay a recorded demo's action sequence. Under C.3
  (`IMPLEMENTATION_PLAN_V3.md` §5 / §8.1) a chart is trained on **on-policy
  CEM rollout transitions** generated by planning toward real dataset goals
  under the shifted regime — there is no replayed demo episode, so
  "episode_idx overlap" no longer captures the leakage of interest. The
  right question under C.3 is: **do the `(init_state, goal_state)` pairs (or
  the planning seeds) used to collect chart-training rollouts overlap those
  used for E0′ planning eval?** For that check to be possible, C.3's
  collector must write a seed/`(init,goal)` manifest (analogous to
  `e0_seed_manifest.json`'s `train` block but recording the sampled pairs,
  not demo indices), and it should draw from a **disjoint seed range**
  (a `seed_offset`, as `run_e0.py`'s train/val/test splits already do) from
  the eval seeds — the check should then assert that offset is respected and
  that the `(episode_idx, offset)` init/goal pairs are disjoint. Until C.3
  lands with such a manifest, this script remains the right tool for the
  `dataset`-collector comparison arm E0′ keeps (`IMPLEMENTATION_PLAN_V3.md`
  §8.1: "the `dataset`-collector chart is run as the one-variable comparison
  arm on R2 only").
- **Claims affected:** N4 and the E0 training-data-sweep trend (disjointness
  now verified for the a9-retrain and the sweep-60 charts; sweep-100 has the
  2.5% overlap noted above). No number recomputed.
- **Re-run needed:** no (audit tool; re-runnable any time, cheap, no GPU).
- **Verification:** `git stash` the script → run → `AssertionError: seed 2 …`
  (**fails before fix: yes**); `git stash pop` → run → completes, prints
  TRAIN / EVAL / planning episode sets and overlap verdict (passes after:
  yes). `python -m py_compile` clean. Ran against 4 real manifests (results
  above).
- **Verified by:** this session, direct before/after command output.

---

## v3 Phase-0 additions (2026-08-29)

### V3-1 — `atlas/expand.py::maybe_expand` gains optional `verify_chunks`

- **Not a bug fix — an authorised additive feature** (user-requested this session,
  after G7 Group B). The single-chunk accept criterion is statistically
  underpowered: ~33–50 % of commits are one-hit wonders (win the one verification
  chunk, lose the next same-regime chunks — measured, `phase0_v3/g7_groupB_*`).
- **Change:** new optional param `verify_chunks: list[tuple] | None = None`. When
  given, the candidate commits iff (a) mean UMF < τ over the set AND (b) it beats
  the incumbent on a *majority* of them. When `None` (default) the original single
  `next_*` chunk rule runs unchanged.
- **Additivity proof:** for one chunk, `beats_best = wins > len/2` reduces to
  `wins == 1` ⇔ `cand_umf < best_umf` (original); `passes_tau = mean_cand < τ`
  reduces to `cand_umf < τ` (original). So every caller that does not pass
  `verify_chunks` is byte-identical.
- **Fails before fix: N/A** (additive). **Regression check:** `smoke_gates.py
  --gate G3a` → PASSED (committed), `--gate G3b` → PASSED (rejected_score), both
  on the default path. E2's N9 commit numbers unaffected (that caller passes no
  `verify_chunks`).
- **Sweep result** (`phase0_v3/g7_groupB_nverify_sweep_disjoint.txt`): the first
  sweep's quality check overlapped the accept window (circular). Fixed: each commit
  records `verify_idx`, the follow-up loop skips it. Disjoint-window generalisation
  rate 40 %→60 %→71 % at N = 1→3→5. `n_probe` sweep deferred (machine instability).
  Proposed default `n_verify = 3` (IMPLEMENTATION_PLAN_V3 §15 item 6) — needs
  explicit approval, joint `n_verify`×`n_probe` re-sweep on P0-G charts first.

### V3-2 — `scripts/run_e0.py::load_regime_trajectories` gains `record_block_pose`

- Optional `record_block_pose: bool = False`; when True, returns
  `block_pose: [T_model+1, 3]` (subsampled `info["block_pose"]`). Needed for
  P0-B's block-static-chunk motion-gate calibration and G4's trajectory statistic.
  Default off → no behaviour change. Not a §C.3 collector edit.

### V3-3 — `scripts/smoke_gates.py::gate_g4` rewritten (per plan §9)

- Old G4: mean-pixel-value comparison, `1e-6` threshold, never run headless
  (vacuous — CHECK 4.1). New G4: fixed identical aimed-walk actions per regime,
  no planner, paired by seed, primary statistic = combined block pose change,
  tested vs a real-variance R0-vs-R0 null band, with a `--g4-selftest` that fakes
  the shifts and must fail. Demonstrated to fail on the fake input (selftest
  passes = gate correctly reports "not distinguishable"). CPU-only.

### V3-4 — `scripts/_determinism.py` + wired into forward-scoring scripts

- **Bug:** cuBLAS re-autotunes GEMM kernels per process launch; cuDNN picks
  non-deterministic algorithms. **Fails before fix:** `phase0_measure.py` run
  twice → **48/48 chunks differ** by ~1e-4 (`umf_c0`, `e1_c0`, `latent_disp`).
  NOT dropout — checked: `predictor.training == False`, 0 active Dropout modules.
- **Fix:** new `scripts/_determinism.py` — `CUBLAS_WORKSPACE_CONFIG=:4096:8` (env,
  before torch), `torch.use_deterministic_algorithms(True, warn_only=True)`,
  `cudnn.deterministic=True`, `benchmark=False`, TF32 pinned off, seeded.
  Imported before `torch` in `phase0_measure.py`, `phase0_g7_groupB.py`,
  `run_e0.py`; `make_deterministic(0)` called in each `main()`;
  `torch.manual_seed(seed_base)` per stream in `phase0_g7_groupB.py`.
- **Passes after:** `phase0_measure.py` ×2 → **0/48 differ, bit-identical.** No
  "no deterministic implementation" warnings for this workload.
- **Impact on already-reported Phase-0 numbers:** τ (0.262) and the motion gate
  (242.7) are P95 over 100+ chunks → stable at 4 sig figs. P0-D strike rate:
  ±~0.7 pp possible on a boundary chunk. G7-B commit decisions: the ±2-commit/
  6-seed variance traced here. No *conclusion* changes; the P0-G re-run is
  reproducible.

### V3-5 — P0-G on-policy collector fixes (`run_e0.py::load_regime_trajectories`)

All four authorised by the user 2026-08-29 (v3 §15 items). closed_loop source only.

1. **`total_contacts > 0` acceptance filter OFF for `closed_loop`** (`run_e0.py:358`).
   It was rejecting + retrying any trajectory where the planner never touched the
   block — re-conditioning the training distribution on contact (the residual form
   of the original hybrid-collector proxy). The collector is goal-directed by the
   planner's cost function, so a low-contact trajectory is real "planner
   struggled" signal, not noise.
2. **`--collect-num-act-stepped` made FUNCTIONAL.** New `collect_nas` param on
   `load_regime_trajectories`; the closed_loop loop now `range(0, n_chunks,
   collect_nas)` and executes up to `frameskip*collect_nas` raw actions per
   replan (capped at `traj_len`). Previously it replanned every model-chunk and
   discarded all but the first planned chunk — collection did not match eval CEM
   cadence (the C-1 mismatch).
3. **CEM config** to `N=300, iterations=10 (P0-C), nas=2` — via
   `modal/modal_phase0.py::p0g_collect` (new, L4 not T4 — CEM is compute-bound),
   `--collect-num-samples 300 --collect-iterations 10 --collect-num-act-stepped 2`.
4. **Determinism** — automatic via V3-4 (`run_e0.py` imports `_determinism`).
- **Smoke DONE 2026-08-29** (pandereshubham; record: `phase0_v3/p0g_smoke_record/`).
  All 4 fixes confirmed live: contact filter OFF printed; nas=2 cadence verified
  (replans at chunk 1→3→5 = 3 CEM searches/traj, was 5); contacts 14–19/traj;
  **66.8 s/traj on L4** (~22 s/CEM search at N=300/it=10). Also uploaded the
  missing `rel_actions.pth` (73 MB) to pandereshubham's `atlas-data` volume.
- **Determinism residual found in the smoke:** `_determinism.py` fixes the FORWARD
  path (phase0_measure ×2 → 0/48 differ) but NOT gradient training — a 20-step
  chart fine-tune run in two processes gives different weights (~1e-2). Not a
  missing-kernel case (`warn_only=False` doesn't raise); CUDA backward reductions.
  Does not block P0-G (runs once → output IS the artifact); G7-B stays a
  per-seed distribution. Documented in `scripts/_determinism.py`.
- **Full-P0-G projection:** 216 trajs → ~$3.6, ~2.3 h (2 concurrent regime calls).
  Awaiting explicit launch approval.

### V3-6 — P0-G pre-launch fixes, phase 1 (`P0G_FIX_PLAN.md` §2.1, §2.2, §3.1, §3.2)

Session 2026-08-29. Work order `research_audit/P0G_FIX_PLAN.md` §2–§4, first four
items only (user: phased). **Not authorised** and NOT touched: `CLAUDE.md` §1.7
values, the R1 scope decision (§4.2), E1/E4 collector defaults (§4.3),
`run_e0_planning.py`'s planning loop (matched, not corrected).

**Environment caveat for every falsification test below:** the frozen
`dino_wm_pusht` checkpoint could not be loaded locally this session — `torch.hub`
hangs on a network sub-download and `hubconf` pulls `clusterscope`→`fcntl`
(Unix-only) on Windows. So model-in-the-loop falsification (running the collector,
instrumenting `plan_length`, the KS cross-check on real chunks) was NOT run. What
*was* run is the model-free layer of each test: the arithmetic the fix changes,
and the real `sample_dataset_init_goal` sampler (no model). The remaining checks
must run on Modal / a Linux box before launch.

**§3.1 [P4] — collector planning lookahead** (`run_e0.py`, `closed_loop` branch,
grep `n_replans_target = max((n_chunks * frameskip)`). `agent.act(steps_left=...)`
now uses run_e0_planning.py's loose convention
`(n_replans_target - replan_idx) * collect_nas` with `n_replans_target =
raw_steps // collect_nas`, instead of `max(n_chunks - chunk_idx, 1)`.
- FALSIFICATION (arithmetic, RAN): at `traj_len=25, frameskip=5, nas=2` the OLD
  formula gives `steps_left = [5, 3, 1]` → `plan_length = min(6, ·) = [5, 3, 1]`;
  at `traj_len=30`, OLD `[6, 4, 2]`. NEW gives `[24, 22, 20]` / `[30, 28, 26]` →
  `plan_length = [6, 6, 6]`, matching the eval reference
  (`run_e0_planning.py:288` → `[6, 6, 6]`). Bug reproduced, fix confirmed at the
  arithmetic level.
- NOT RUN: the plan's instrumented check (print `agent._prev_elite_losses_mean`
  shape in collector vs one eval episode). Blocked on model load.
- Consequence per plan: collection ~1.7× slower; §2.2 timeout / cost line must be
  re-measured on the smoke. Not done (no smoke).

**§3.2 [P5] — episode length + goal separation** (`run_e0.py` `closed_loop`
branch: `sample_dataset_init_goal(... traj_len=GOAL_TRAJ_LEN ...)`, was
`traj_len=traj_len`; plus a `min(demo_seq_lengths) >= GOAL_TRAJ_LEN` assert;
`GOAL_TRAJ_LEN` imported from `run_e0_planning`). `modal/modal_phase0.py`:
`traj_len` / `eval_traj_len` default 25 → 30.
- FALSIFICATION (real sampler, RAN — no model needed): 200 seeds each, `states.pth`
  from `data/pusht_noise/train`. OLD (`traj_len=25`): block-separation median
  75.7 px, mean 83.2. NEW (`GOAL_TRAJ_LEN=31`): median 82.9, mean 89.7. Real eval
  distribution (`atlas_out/e0_planning_nas2/baseline_R2.jsonl`
  `init_block_pos_diff`, n=20): median 77.9, mean 94.9. The fix moves collection
  toward eval (median gap 2.2 → 5.0 the other way… medians nearly coincide; mean
  gap 11.7 → 5.2). `min(demo_seq_lengths) = 49 ≥ 31`, so no episode-filter change
  is needed — assert added and holds. Sample constancy: same 200 seeds both arms,
  only `traj_len` differs.
- NOT RUN: a formal KS test vs a fresh eval `episodes.jsonl`; n=20 on the eval
  side is low power anyway.

**§2.1 [P9, P2c] — persist trajectories + T=2 chunk dump + `--load-trajs`**
(`run_e0.py`). New: `--load-trajs`, `--collect-only`, `_traj_guard()`,
`dump_regime_chunks()`. In the regime loop, trajectories are loaded from
`trajs_{regime}.pt` (in `--load-trajs` or, on resume, in `--out`) when present
and the stored `_traj_guard` fingerprint matches; otherwise collected then
`torch.save`d, and (closed_loop) `chunks_{regime}.jsonl` is written — one row per
`T=collect_nas` sliding window with `umf_c0` (via `score.rollout_umf` on the
pristine predictor), `latent_disp`, `block_disp_px`.
- FALSIFICATION (RAN, model-free): BEFORE — pre-edit `main()` wrote only
  `chart_*/loss_*/results.json/e0_seed_manifest.json` (grep confirms; no
  `trajs_*` / `chunks_*` anywhere). AFTER — `_traj_guard` roundtrips through
  `torch.save`/`load`; a changed `--collect-iterations` (10→30) flips the guard
  (`300x10 nas=2` → `300x30 nas=2`) and the load path raises `ValueError`
  ("protocol mismatch"). encoder_output kept fp32 (size to be measured on the
  smoke before switching to `.half()`).
- NOT RUN: `dump_regime_chunks` end-to-end (needs `rollout_umf` → the model); the
  plan's smoke that confirms "re-run with `--load-trajs` → ZERO CEM searches";
  the production step-rate measurement that would justify §2.2's timeout.
- Resume: `traj_file.exists()` in `--out` short-circuits collection. The plan's
  fuller "collection skipped iff trajs AND chart exist" restructure is partially
  done (trajs-exist is enough to skip collection; per-kind chart resume is
  unchanged downstream).

**§2.2 [P2, P2c] — split the Modal function** (`modal/modal_phase0.py`).
`p0g_collect` (was collect+finetune, `timeout=3600*8`) split into `p0g_collect`
(`--collect-only`, `timeout=3600*6`, ONE regime per call) and `p0g_finetune`
(`--load-trajs`, `timeout=3600*10`). Local entrypoints `p0g-collect` /
`p0g-finetune`. `--num-test-trajs 8` baked into `_P0G_COMMON` (this also lands
§3.3 P3, noted here — a fine-tune re-run cannot silently drop it). Old
`SMOKE_SUMMARY.md` `$3.6 / 4.5 h` projection called out as superseded in the code
comment; the replacement figure needs the un-run step-rate measurement.
- FALSIFICATION: none required for a structural split (per plan). Timeout
  arithmetic that justifies the new values is from `P0G_REVIEW.md` P2 (~13.6 h
  combined), NOT re-measured this session.
- NOT DONE: `SMOKE_SUMMARY.md` supersession note (plan §1.5 / §2.2.4) — deferred
  with the rest of the smoke-dependent work.

`scripts/run_e0.py::main` under `--collect-only` still writes an empty
`results.json` + the (still useful) seed manifest, then the matplotlib block
no-ops on empty data. Harmless; noted.

### V3-7 — P0-G pre-launch fixes, phase 2 (`P0G_FIX_PLAN.md` §3.3, §3.4, §4.1; §4.2 prepared)

Session 2026-08-29, continued. Same environment caveat as V3-6 (frozen checkpoint
does not load locally → model-in-the-loop falsification NOT run; model-free layer
run and reported).

**§3.4 [P1, P1b] — regime-ordering contamination (the highest-stakes fix).**
- Operational half DONE in V3-6: `p0g_collect` takes a single `regime: str`,
  entrypoint says "run twice".
- Code half (this session): `run_e0.py` regime loop now does
  `wm.predictor.load_state_dict(pristine_predictor_state)` at the TOP of the loop,
  before collection (grep `P1 / v3 §5.2: on-policy collection MUST plan`).
  Previously the only pristine reload was inside `for kind` (after that regime's
  collection); `Chart.restore_()` for ln_act/full re-applies TRAINED weights
  (`atlas/chart.py:126-127`, confirmed at source + its own docstring
  `restore_pretrained_` FIX_SPEC C4). Object identity verified by reading:
  `collector_agent` ← `wrapper`; `wm = wrapper.model`; so `wm.predictor` is the
  agent's predictor and an in-place `load_state_dict` is seen.
- New `--debug-predictor-fingerprint` flag: prints sha256 of predictor params
  before each regime's collection — the plan's falsification test, made runnable
  later without re-patching. Expected: `--regimes R0,R2` → IDENTICAL fingerprints
  after the fix, DIFFERENT before.
- NOT RUN: that fingerprint test (needs model load).

**§3.3 [P3] — disjoint test split.**
- Launcher half DONE in V3-6 (`--num-test-trajs 8` in `_P0G_COMMON`;
  `num_test_trajs` param on `p0g_collect` + entrypoint so it can't be re-pinned
  to 0).
- Label half (this session): `run_e0.py` `results.json` now carries
  `eval_umf_source` ∈ {`"test"`, `"val_ALIASED"`, `"error"`} in BOTH the resume
  and the fresh branch; the stale `# from the disjoint TEST set (A4)` comment is
  corrected to condition on it. `eval_umf` is no longer silently aliased to
  `val_umf` under a comment claiming otherwise — the aliasing is now labelled.
- NOT RUN: the plan's `eval_umf != val_umf` + observed-bias-and-sign check (needs
  a fine-tune with 8 test trajs → model). Repo's prior measurement of this bias
  (`FIXLOG` A4: +0.077…+0.157 on R0 cells) stands as the expectation.

**§4.1 [P16] — determinism-asymmetry check.** No production code. `modal_phase0.py`
`p0g_finetune` gained a `load_subdir` param so one cached collection can feed two
fine-tune runs writing to separate `--out` dirs (`det_run1` / `det_run2`), per
§1.7 "never reuse an output dir". Docstring carries the exact procedure.
- NOT RUN (needs model + GPU + a real collection first). P16 remains
  reasoning-only; not marked refuted/confirmed in `P0G_REVIEW.md` (the plan says
  only do that after running).

**§4.2 🛑 [P13] — R1 scope. DECIDED: R1 DROPPED — explicit human sign-off
2026-08-29** ("you can drop R1 … if time permits then we will run the
experiments"). Written into `IMPLEMENTATION_PLAN_V3.md` §8.1 (Arms + decision
rule now over `{R2}`) and §8.3 (R0/R1 cell-B replicate struck). P0-G runs R2
only; `p0g_collect` already defaults to `regime="R2"`.
*(Note: an earlier version of this entry recorded the drop off an
`AskUserQuestion` selection, which a Stop hook flagged as not a valid §1.8
sign-off. That was reverted, the question re-put in prose, and the drop
re-applied only after the user's explicit text confirmation above.)*
Re-verified P13b (model-free, real `sample_dataset_init_goal`
+ `states.pth`, `RandomState(seed)` on both sides):
- **R1 collection train seeds share 50/100 eval tasks** — NOT the sub-agent's
  39/100. Mechanism: R1 `seed_base = 0`, collection seeds `{0,2,…,198}`; eval
  seeds `{0,…,99}`; the 50 even eval seeds are reused verbatim as collection
  seeds → identical `sample_dataset_init_goal` draw → identical (init, goal) task.
- The V3-6 §3.2 fix **increases** this overlap (15/100 → 50/100) because
  collection now uses the same `traj_len = GOAL_TRAJ_LEN = 31` as eval, so the
  same seed lands on the same episode.
- R2 and R0: **0/100** (seed_base 1000 / 2000, no even-seed reuse).
- Conclusion for the user: if R1 is kept, `seed_base["R1"]` MUST move (e.g. to
  3000) before any collection. `p0g_collect` currently defaults to `regime="R2"`
  only, so nothing is live yet.

### V3-8 — P0-G pre-launch fixes, phase 3 (`P0G_FIX_PLAN.md` §4.3, §4.4, §4.6)

Session 2026-08-29, continued. Same environment caveat (checkpoint does not load
locally → model-in-the-loop tests deferred). All model-free checks below RAN.

**§4.3 [P7] — E1/E4 collector `source`. Made explicit, NOT changed (scope
decision, still needs sign-off).** `run_e1.py:292` and `run_e4.py:233` (and
`smoke_e4.py:105`) called `load_regime_trajectories(...)` with no `source=` →
signature default `"scripted"`, the retired goal-free contact-seeking walk. Now
pass `source="scripted"` explicitly with a comment; `run_e1.py` writes
`e1_run_meta.json` (`gate_source`, `motion_gate`); `run_e4.py` summary gains
`"gate_source"`. **The trade-off for the human (STILL OPEN):** matching the
charts P0-G produces means calibrating the motion gate on `source="closed_loop"`
too — which costs one CEM search per gate trajectory (E4: 30 trajs → ~30 extra
searches per arm). Not done here; needs a decision like §4.2.

**§4.4 [P8, P10, P10b] — the reported metric, additive.** `evaluate_e0_chart`
rewritten to return a **dict** (was `(loss, umf)`) and now also computes:
- **P8:** `umf_chunkT{nas}` — mean UMF over T=`collect_nas` sliding windows, on
  τ's scale (≈0.262), alongside the trajectory-T `umf`. `umf` calls are additive
  (`atlas/score.py::umf` untouched, §1.2).
- **P10:** `umf_ungated` / `umf_chunkT{nas}_ungated` — same, `motion_gate=None`,
  so the gate's (optimistic) effect is visible not baked in.
- **P10b:** `n_trajs`, `n_umf`, `n_umf_chunkT{nas}`, `n_windows` recorded — `umf`
  and `loss` were means over different subsets with nothing logging which.
`results.json` gains these via `_umf_detail_fields()` (both the fresh and resume
branches) plus `motion_gate_value` + `motion_gate_rule` (the rule string names it
RETIRED per §6.6). 4 call sites updated to the dict return; no external callers.
- FALSIFICATION (RAN, model-free): `_umf_detail_fields({})` → all-None + the rule
  string; on a fake eval dict → correct keys/values. The plan's "set the gate
  high, confirm n=2 while loss is over 3" check needs the model — NOT run.

**§4.6 P21 — three stale docstrings.** `atlas/regimes.py` module docstring said
"R2 = shape.elasticity raised" — inverted from `REGIME_CONFIGS` (`{"damping":
0.5}`) — now corrected (R2 = `space.damping 0.5`), and the "re-targeted onto …
shape.elasticity (R2)" line fixed to `space.damping`. `run_e0.py`
`--collect-num-act-stepped` help rewritten (the "this flag is a no-op" CAVEAT is
obsolete — it's the v3 §5.2 fix and works) **and its default 1 → 2** (the §3.6
value; behaviour change for a bare `run_e0.py --data-source closed_loop`, but
`modal_phase0.py` passes it explicitly so P0-G is unaffected). `--data-source`
help's "replanning every model chunk" → "every --collect-num-act-stepped chunks".

**§4.6 P12 / P18 / P19 / P20 — manifest provenance.**
- **P12:** `n_contacts` per trajectory now in `e0_seed_manifest.json` (was
  stdout-only, so §15-2's pre-registered R2 damping check was unreadable from
  artifacts). Debug print label `"Real-demo replay contact rate"` → conditional
  (`"on-policy planner"` for closed_loop) + names the §15-2 fallback rule.
- **P18:** `sample_dataset_init_goal` gained optional `return_indices=True` (4-tuple,
  additive — `run_e0_planning.py::run_episode` untouched, back-compat verified);
  `run_e0.py` closed_loop branch now records real `episode_idx`/`offset` (were
  `null`). Unblocks `scripts/audit_e0_train_planning_overlap.py` on on-policy
  manifests.
- **P19:** runtime assertion that train/val/test seed intervals are disjoint
  within and across regimes (`seen_seeds` dict). Silently collided at
  `num_trajs >= 501`; now fails loudly.
- **P20:** `modal_phase0.py` reads the git SHA on the CLIENT (`_local_git_sha()`
  in the `@app.local_entrypoint`) and passes it as `ATLAS_GIT_SHA`;
  `_determinism.settings_dict` prefers that env var over the (failing on Modal)
  `git rev-parse`. `phase0` / `p0g_collect` / `p0g_finetune` + all 3 entrypoints
  wired.
- FALSIFICATION (RAN): P18 `return_indices` 4-tuple + 2-tuple back-compat both
  verified against real `states.pth`; P21 regimes docstring vs `REGIME_CONFIGS`
  now agree (grep); compile-clean across all 7 touched files.

**NOT DONE this phase:** §4.5 (C-1/C-2 chart acceptance checks — a runbook that
executes AFTER a real collection; deferred), P15/P15b (doc-only, deferred),
P17/P22 (cosmetic, deferred).

### V3-9 — P0-G addendum: fix the two defects the V3-6/7/8 review found (`P0G_FIX_PLAN §7`)

Session 2026-08-29. `P0G_FIX_PLAN.md` §7 (ADDENDUM) flagged two bugs introduced
by the earlier fixes. Both falsification tests are model-free.

**§7-B1 🔴 — `--load-trajs` guard rejected every `p0g_finetune`.**
`p0g_finetune` emitted only `--regimes/--load-trajs/--steps/--out` + `_P0G_COMMON`,
so 4 of `_traj_guard`'s 9 fields fell back to argparse defaults ≠ what
`p0g_collect` stored.
- FALSIFICATION (RAN, model-free): rebuilt both arg namespaces via
  `run_e0._build_parser()` from the two Modal command lines →
  `_traj_guard` **NOT equal**, mismatched fields exactly as §7-B1 predicted:
  `train_traj_len (30,25)`, `eval_traj_len (30,50)`, `num_train_trajs (100,20)`,
  `collect_cem ('300x10 nas=2','300x30 nas=2')`.
- FIX: new `scripts/_p0g_spec.py` (no `modal` import) holds `_P0G_DEFAULTS` +
  `_p0g_flags()` + `_P0G_COMMON`; `modal_phase0.py` imports them; **both**
  `p0g_collect` and `p0g_finetune` default all 8 collection params off
  `_P0G_DEFAULTS` and emit `*_p0g_flags(...)`. `--num-test-trajs` moved out of
  `_P0G_COMMON` (emitted once). `run_e0.py::main()` refactored: parser extracted
  to `_build_parser()` (main() unchanged behaviour).
- FALSIFICATION AFTER (RAN): `_traj_guard` **equal**. Landed as a permanent
  regression test: `tests/test_p0g_guard.py` (passes; full suite 22/22).
- The guard was NOT weakened (§7-B1 forbids it).

**§7-B2 🟠 — the T=nas windowed UMF (P8) was gated by the trajectory-scale gate.**
`evaluate_e0_chart` passed `motion_gate` (10th pct of T=6 `‖z_6−z_0‖`) into the
T=2 windowed `umf()` calls — §6.6's "calibrated at a granularity it is not
applied at", reintroduced.
- FALSIFICATION: the exact test (`compute_motion_gate(traj_disps)` vs
  `chunk_disps` from a real `trajs_R2.pt`) **could NOT be run — no `trajs_R2.pt`
  exists yet** (collection has not run). Ran a SYNTHETIC substitute instead
  (directed-motion latents, static component cancels, T_model=6, N=256, D=384,
  100 trajs, real `compute_motion_gate`): traj-scale gate 284 vs T=2 window
  p90=97 → **BEFORE gates 100% of windows; AFTER (chunk-scale gate) gates 10%**.
  Mechanism confirmed; real-data over-gating **fraction is unverified** — owed
  once collection produces `trajs_R2.pt`. Real phase0 proxy T=2 disps (R2
  p10≈164/p50≈237) make ≥50% over-gating the conservative real expectation.
  Not treating B2 as refuted: the mismatch is real by construction (the code
  literally passes a T=6 threshold to T=2 calls) and §6.6 mandates the fix
  regardless of magnitude.
- FIX: `run_e0.py::main()` computes `chunk_motion_gate = compute_motion_gate(
  chunk_displacements)` (10th pct of T=`nas` train-window disps) beside
  `motion_gate`; `evaluate_e0_chart` gains `chunk_motion_gate` param used **only**
  for the windowed calls. `_umf_detail_fields` records both
  (`motion_gate_value`, `motion_gate_chunk_value`). `eval_umf_chunkT{nas}_ungated`
  kept as the always-interpretable number.

**§7-C operational notes actioned:**
- C-1: `--num-test-trajs` is now a real Modal param (default 8); smoke docstrings
  updated to `--num-test-trajs 2`.
- C-2: `SMOKE_SUMMARY.md` gained a SUPERSEDED block — `66.8 s/traj` / `$3.6` are
  stale-low; ~135 s/traj is a first-order **estimate** (not measured); re-measure
  on next smoke. Same note in `modal_phase0.py`.
- C-3: `p0g_collect` / entrypoint docstrings now state R0 collection is REQUIRED
  (τ / σ_r over R0 chunks) and is a separate `--regime R0` call.

**V3-9 bucket-2 pass (static + mock-tensor, no model) — 2026-08-29.** Before a
Modal smoke, exercised the new code paths with mocked `umf`/`rollout_umf`/
`_open_loop_rollout` and fake trajectory dicts:
- `evaluate_e0_chart` T_model=6/chunk_nas=2 → **15 windows over 3 trajs** (correct);
  traj-scale gate at 999 → `n_umf=0` → `umf=nan` (gate fires); windowed `umf`
  calls get exactly `enc[3,N,D] / acts[2,10] / proprio[1,1,P,D]` — matches the
  traj-level convention and `umf`'s shape contract.
- Edge `T_model < chunk_nas` (Tm=1, nas=2) → 0 windows, `umf_chunkT2=nan`, no
  IndexError. Added a `chunk_displacements.numel()==0 → chunk_motion_gate=None`
  guard in `main()` for the same edge.
- `dump_regime_chunks`: correct row count (Σ per traj of `T−nas+1`),
  `block_disp_px` numeric with `block_pose` / `None` without.
- `--collect-only` continue sits after the gate computations (gates logged on the
  collect run — useful for the τ/gate re-derivation — then exits before the kind
  loop); empty `results` → `results.json`/T5-md/plot all no-op cleanly.
- P19 seed-disjointness assertion does NOT false-fire at the P0-G launch config
  (R0/R2 train/val/test intervals all disjoint; R0∩R2 empty) — checked from the
  seed-generation expression.
- `_traj_guard` computed once and used for BOTH `--load-trajs` and the
  auto-resume `traj_file.exists()` branch (line 897/907).
- `run_e4.py` `GATE_SOURCE` is defined before both the profile path and the
  summary dict; `run_e1.py` `e1_run_meta.json` write has `motion_gate` +
  `GATE_SOURCE` in scope.
**Not covered** (needs the model): real tensor shapes through `_make_z_ctxt` /
`_open_loop_rollout` on sliced windows, `dump_regime_chunks` `rollout_umf` on a
real pristine predictor, the collector loop itself.

**V3-9 SMOKE — first real end-to-end run, 2026-08-29, pandereshubham Modal.**
App `ap-P1QFg3vMEbLgYx5TJdXaAb`, `p0g_collect_entry --regime R2 --num-trajs 5
--num-val-trajs 2 --num-test-trajs 2 --collect-only`. Full record:
`phase0_v3/p0g_smoke_v3/SMOKE_RESULTS.md` (+ raw `trajs_R2.pt`, `chunks_R2.jsonl`,
`e0_seed_manifest.json` archived there). **PASS end-to-end**, exit 0.
- Two launch bugs found + fixed first: (a) `modal run` on Windows dies on a `✓`
  in modal's own output → `PYTHONUTF8=1`; (b) `modal_phase0.py` imports
  `_p0g_spec` at module load but on Modal the entrypoint is at `/root/`, so
  `REPO_ROOT/"scripts"` → `/scripts` (missing) → `ModuleNotFoundError` crash-loop.
  Fixed: `_p0g_spec` moved to its own dep-free module `scripts/_p0g_spec.py`
  (modal_phase0 + the test both import it); sys.path fallback to `/src/scripts`;
  `image.add_local_file(_p0g_spec.py → /root/)`. (The add_local_file sits before
  run_commands so it re-triggers the torch layer build — move it after
  run_commands in a follow-up; the sys.path fallback alone would suffice.)
- **Verified live:** §3.4 single-regime; §3.2 `chunk=1/6` (traj_len 30); §3.1
  nas=2 → 3 CEM searches/traj, **150.6 s/traj** (= §7-C-2's "≈135 s roughly
  doubles" estimate, so the 6 h collect timeout holds with ~1.15 h margin at
  116 trajs/regime); §2.1 `trajs_R2.pt` 25 MB/7 trajs (→ ~385 MB/regime fp32,
  under the 600 MB estimate), `chunks_R2.jsonl` 35 rows; §2.2 `--collect-only`
  clean exit; §3.3 test split present; §7-B1 `collect_cem "300x10 nas=2"` matches
  `_P0G_DEFAULTS`; P12 label + `n_contacts` in manifest ([6,1,5,13,2] train,
  **7/7 with contact** — no §15-2 collapse); P18 real `episode_idx`/`offset`
  (not null); P19 no seed clash; P20 `git_commit` real (not "unknown").
- **§7-B2 REAL-DATA falsification (now runnable off `trajs_R2.pt`):**
  `motion_gate` (T=6) = **244.18**, `chunk_motion_gate` (T=2) = **142.34**.
  Applied to the 35 real T=2 windows: BEFORE (traj-gate 244) gates **13/35 =
  37 %**; AFTER (chunk-gate 142) gates **3/35 = 9 %**. **§7-B2's over-gating
  claim is confirmed on real data** (milder than the synthetic 100% — real R2
  latents have a motion floor — but a real ~4× over-gating); the fix (a
  granularity-matched gate) is correct on that basis alone.
- **Bonus:** `umf_c0` on R2 T=2 on-policy chunks under frozen c₀ = median 0.568
  (min 0.086, max 0.945) — well above τ≈0.262; this is the on-policy chunk set
  P0-A/P0-D re-derivation is owed (§5 deviation-note-1), now persisted.
- NOT run: `p0g_finetune` on a real container, R0 collection, §4.1/§4.4/§4.5,
  full P0-G (needs launch approval).

**⚠️ CORRECTION, same day, caught by external review (not by this session).**
The paragraph above originally also reported survivor mean `umf_c0` 0.632
(traj-gate) vs 0.579 (chunk-gate) as "traj-gate optimistically biased, per
P10b" — asserting P10b's directional-bias claim as confirmed. **That claim was
never checked and was wrong.** Checked directly on the 35 real chunks:
`corr(latent_disp, umf_c0) = +0.398` — the opposite sign from what P10b assumes
(low displacement -> small denominator -> large UMF -> gating is "optimistic").
The traj-gate's 10 *extra* drops (beyond the chunk-gate) have mean UMF 0.461,
*below* the 0.553 overall mean — removing them raises the survivor mean, which
is the observed 0.632 > 0.579, but is not evidence of P10b's mechanism. This is
exactly the pattern `CLAUDE.md` §1.9 exists to catch: a plausible, unverified
claim stated as a finding, then propagated. **Fixed at every copy**:
`research_audit/P0G_REVIEW.md` P10b row rewritten to stop asserting a sign;
`research_audit/P0G_FIX_PLAN.md` §4.4 item 3 and §7-B2's STATUS box corrected;
`phase0_v3/p0g_smoke_v3/SMOKE_RESULTS.md` §7-B2 corrected in place with the
retraction kept visible (not silently removed); `scripts/run_e0.py`'s
`evaluate_e0_chart` docstring no longer asserts a gating-bias direction. The
over-gating magnitude claim (37% vs 9%, ~4×) is unaffected and stands.

### V3-10 — post-smoke: correction + two new reporting features (2026-08-29)

Three user-directed items after the smoke, none gating launch.

**1. Direction-error correction (covered above, restated for the FIXLOG index):**
P10b's directional-bias claim (`P0G_REVIEW.md`, `P0G_FIX_PLAN.md` §4.4/§7-B2,
`phase0_v3/p0g_smoke_v3/SMOKE_RESULTS.md`, `scripts/run_e0.py` docstring) was
never checked and was backwards on real data (`corr(latent_disp, umf_c0) =
+0.398` on the smoke's 35 R2 chunks, not the assumed negative correlation).
Fixed at every copy; the over-gating magnitude claim (37% vs 9%) is unaffected.

**2. `report_block_static_fraction()` (`run_e0.py`).** Reports, per split
(train/val/test) and combined, what fraction of collected trajectories have
whole-trajectory block pixel displacement < `BLOCK_STATIC_PX=1.0` (matches
`phase0_measure.py`'s existing convention). Pure reporting — filters nothing.
Writes `block_static_{regime}.json`, wired into `main()` right after
`dump_regime_chunks` (closed_loop only).
- FALSIFICATION (RAN, real data — the smoke's downloaded `trajs_R2.pt`, not
  synthetic): all 9 collected trajectories (5 train/2 val/2 test) reported
  correctly, `frac_traj_static = 0/9 = 0.0` per split and combined. No-block_pose
  input (other collector sources) → `SKIPPED`, returns `None`, no crash —
  verified directly.
- **Answer to Part 3 Q8, at this smoke's n:** 0% dead at n=9 — not evidence the
  true rate is 0; n=2 val trajectories can't bound a rare event. The real number
  is owed from the full run.

**3. `derive_and_report_motion_gate()` (`run_e0.py`).** Derives v3 §6.6's P95
gate from block-static T=nas chunks in the full `chunks_{regime}.jsonl` dump,
reports it alongside the retired 10th-pct value and each rule's REALISED
false-pass rate (fraction of block-static chunks that still clear the gate —
scored informative despite no real block motion). Writes
`gate_calibration_{regime}.json`. **Does NOT adopt either value** — `CLAUDE.md`
§15-5 already requires human sign-off for the motion gate; this only supplies
real evidence for it. Wired into `main()` right after the traj-scale
`motion_gate` is computed (closed_loop only); gracefully skips + returns `None`
if the chunks file is missing (e.g. a `--load-trajs`-only fine-tune run pointed
at a different `--out`).
- FALSIFICATION (RAN, real data — smoke's `chunks_R2.jsonl`, n=35): 5/35 = 14%
  block-static chunks; 10pct gate (244.2) has **0%** false-pass; P95-static gate
  (**163.4**) has **20%** false-pass. Missing-file case → `SKIPPED`, `None`, no
  crash — verified directly. Full table + framing:
  `phase0_v3/p0g_smoke_v3/SMOKE_RESULTS.md`. **Not adopted; awaiting sign-off.**

Both functions were unit-verified against the smoke's real persisted data
(downloaded artifacts, not run inside a fresh Modal container) — the ordering
bug this surfaced (an earlier draft called `derive_and_report_motion_gate`
before `motion_gate` was computed, a `NameError`) was caught and fixed before
any Modal spend.

### V3-11 — two follow-up smokes, 2026-08-29: R0 collect-only + p0g_finetune

Both PASS end-to-end, pandereshubham. Full record in
`phase0_v3/p0g_smoke_v3/SMOKE_RESULTS.md`.
- **R0 collect-only** (`ap-bzGgUwbAKF9jO2edv6S0Ds`): same shape as the R2 smoke,
  150–156 s/traj, all P9/P12/P18/P19/P20 artifacts written correctly.
- **`p0g_finetune` smoke** (`ap-QzmFlnbtuGAMgJsS23a4gL`, `--load-trajs` the
  already-collected R2 trajs, matching counts so the §7-B1 guard matches): PASS.
  §7-B1 now proven live (not just by the unit test). **~0.55 s/gradient step**
  at 5 train trajectories. `evaluate_e0_chart`'s new `chunk_motion_gate` path ran
  cleanly end-to-end for the first time (`eval_umf_source: "test"`, UMF 0.459).
- **R0-vs-R2 contact comparison (§15-2), n=9/regime — first on-policy look:**
  R0 mean 16.3 contacts/traj (1/9 zero), R2 mean 6.7 (0/9 zero). Real ~2.4×
  reduction under R2, NOT a collapse to zero. n too small to be decisive; the
  shape of the check the full run settles.
- τ/σ_r illustrative-only candidates from n=35 R0 chunks: P95=0.471, IQR=0.110
  — not comparable to the real τ=0.262 measurement's n, not adopted.

### V3-12 — P0-G sharding (user request 2026-08-29): parallel collection across N Modal containers

Mirrors the existing, already-proven pattern in `modal/modal_e0_planning.py`
(`--num-shards`, `.spawn()`, `merge_shards()`) — that pattern is for the
planning/eval side; this wires the equivalent for P0-G's collection side,
which has the identical shape (a sequential per-trajectory CEM loop, not
GPU-flop-bound, so N containers in parallel beats one for N× as long, same
total cost).

**Changes:**
- `scripts/run_e0.py::load_regime_trajectories`: new `traj_idx_offset: int = 0`
  param — `seed = seed_base + (traj_idx + traj_idx_offset) * max_tries +
  attempt`. Default 0 is byte-identical to the old formula (verified: no
  existing call site passes it, so nothing changes for unsharded runs).
- New CLI flags `--collect-traj-offset` (threaded only into the TRAIN
  `load_regime_trajectories` call) and `--collect-skip-val-test` (skips val/test
  entirely — they're cheap and not worth splitting; exactly one shard should
  collect them).
- Factored `compute_motion_gates(train_trajectories, nas)` out of `main()`
  (previously inline) so the merge step can recompute both gates on the
  MERGED set — a per-shard gate is calibrated on that shard's own subset, not
  the requested total, and must not be reported as if it were.
- New `scripts/merge_p0g_shards.py`: concatenates shards' `trajs_{regime}.pt`
  (train), takes val/test from whichever shard collected them, concatenates
  `chunks_{regime}.jsonl` and the seed manifest's train rows, rejects on
  overlapping seeds or a protocol mismatch (any guard field disagreeing except
  `num_train_trajs`), and recomputes `compute_motion_gates` +
  `report_block_static_fraction` + `derive_and_report_motion_gate` on the
  merged data.
- `modal/modal_phase0.py`: `p0g_collect` gained `traj_offset`/`skip_val_test`
  params; new `merge_p0g_shards` Modal function (runs the merge script inside
  a volume-mounted container, no local download, mirrors
  `modal_e0_planning.py::merge_shards`); new entrypoint `p0g-collect-sharded`
  (`--num-shards`, same divmod bounds-splitting + `.spawn()`/`.get()` pattern
  as `modal_e0_planning.py::main`).

**FALSIFICATION (RAN, real data — no synthetic substitute needed here):**
- Bounds arithmetic: `divmod`-based split verified for (100,4)→[(0,25),(25,25),
  (50,25),(75,25)] (matches the "4 containers, 25 each" the user remembered),
  plus (5,2), (8,3), (1,4) — every case sums exactly to `num_trajs`, no
  gaps/overlaps.
- Merge script exercised on the REAL smoke `trajs_R2.pt`/`chunks_R2.jsonl`
  (not synthetic), split into two genuinely disjoint shards (train[0:2] +
  train[2:5], real distinct seeds 1000/1002 vs 1004/1006/1008): **PASS** —
  merged train=5 (all 5 real seeds, sorted), val=2, test=2, chunks=35 rows,
  recomputed `motion_gate=244.18`/`chunk_motion_gate=142.34` **exactly matching**
  the values independently computed on the unsplit data earlier this session.
- BEFORE/AFTER on a real bug found while testing: the merge script's final
  success print used a `✅` emoji, crashing with `UnicodeEncodeError` on the
  Windows console (same class of bug as the earlier `modal run` launch
  failure) — exit 1 despite a fully correct merge. **AFTER** (emoji removed):
  exit 0, clean.
- Deliberately-broken-input tests (§1.1): two shards with an overlapping seed
  (1004 in both) → `ValueError: Overlapping seeds across shards: [1004]`,
  correctly rejected. Two shards with a mismatched `collect_cem` field
  (protocol drift) → `ValueError: Shard protocol mismatch`, correctly rejected.

**RUN on real Modal (2026-08-29, pandereshubham): `p0g-collect-sharded --regime
R2 --num-trajs 8 --num-shards 2 --num-val-trajs 2 --num-test-trajs 2`.** Both
shards collected correctly and concurrently (confirmed 2 tasks running at once)
— shard0 (4 train + 2 val + 2 test) and shard1 (4 train, `--collect-skip-val-test`
correctly applied). **The merge step failed on first run — a real bug, caught
exactly as intended:**

**Bug found:** `merge_p0g_shards` (the Modal function) runs on a plain CPU
container (no `gpu=` — correct, it does no GPU work) but `torch.load(p,
weights_only=False)` on trajectory tensors saved from the GPU collection
containers raises `RuntimeError: Attempting to deserialize object on a CUDA
device but torch.cuda.is_available() is False`. My earlier local falsification
of the merge script (previous FIXLOG entry) did NOT catch this because the
local test machine has a physical GPU (RTX 4050) — `torch.cuda.is_available()`
is True there, masking the exact failure mode the real CPU-only Modal
container hits.
- BEFORE (RAN, reproduced locally): `torch.cuda.is_available = lambda: False`
  (simulating the real container) + `torch.load(..., weights_only=False)` on
  the real smoke's `trajs_R2.pt` → the identical `RuntimeError`, verbatim.
- FIX: `map_location="cpu"` added to the one `torch.load` call in
  `scripts/merge_p0g_shards.py` (merge is pure concatenation + light stats,
  never needs GPU tensors).
- AFTER (RAN, same simulated condition): succeeds, `len(train)=5`. Full merge
  falsification suite (clean merge, overlap rejection, protocol-mismatch
  rejection) re-run under the simulated condition: all still correct.
- **Then re-verified on the REAL Modal CPU container** (new local_entrypoint
  `p0g-merge-shards`, added because `merge_p0g_shards`'s `list[str]` param
  isn't parseable directly from the Modal CLI — re-ran merge-only on the
  already-collected shard data, no re-collection needed): **succeeded**,
  `ap-0KEKhYNoBX4NK8KJoiTKsW`. Merged 8 train (seeds 1000–1014, all unique) /
  2 val / 2 test / 50 chunk rows; gates recomputed cleanly
  (`motion_gate=276.2`, `chunk_motion_gate=141.9`, `P95-static=162.1`).

**The full sharded pipeline (parallel collect -> merge) is now proven
end-to-end on real Modal infrastructure**, not just unit-tested locally.

### New Phase-0 diagnostic scripts (not experiment code)

`scripts/phase0_measure.py`, `scripts/phase0_g7_groupA.py`,
`scripts/phase0_g7_groupB.py`, `modal/modal_phase0.py` — forward-only P0-A/B/D/E
and G7 calibration diagnostics. Write only to `phase0_v3/`. No production path
touched.

---

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
