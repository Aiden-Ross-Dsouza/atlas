# ATLAS — FIX SPEC

**Last updated: 2026-08-27 — created during Phase 0. Phase 1 and Phase 3 specs are
written; Phase 2's are summarised and expand on dispatch.**

## What this file is

The per-fix implementation specs handed to delegated sub-agents. Each entry gives the
**defect** at `file:line`, the **exact intended change**, and the **named assertion**
that proves the fix landed. Register IDs match `SUBMISSION_PLAN.md` Part B.

## Rules every sub-agent working from this file must follow

1. **Log every change in `research_audit/FIXLOG.md`** using the row template at the
   bottom of that file. A fix that is not logged did not happen.
2. **Never touch** `atlas/score.py::umf`, `atlas/stats.py`'s existing functions, or
   `scripts/run_e0_planning.py`'s planning loop. They produced the numbers on disk.
   Additions alongside them are fine; modifications are not.
3. **Do not delete a superseded number.** Supersede it in place in its owning results
   document with a dated banner. This repo's convention throughout.
4. **Run the assertion, paste the actual output** into your report. "Should pass" is not
   evidence. Where an entry says *must fail before*, run it on the unfixed code first and
   paste that output too — an assertion that passes both before and after is vacuous,
   which is exactly the defect that made gates G2 and G5 worthless here.
5. **Do not fix anything not in this file.** If you find something new, document it in
   `FIXLOG.md` under "Discovered, not fixed" and stop.
6. Work through changes one file at a time and re-run the relevant assertion before
   moving on. Do not batch ten edits and test at the end.

---

# PHASE 1 — Tier A: bugs that corrupt numbers already on disk

Owner agent: `atlas-fixer`. Budget ~$3.20. Every fix here is followed by a cheap re-run,
because the numbers it affects are already published in `PAPER_DRAFT.md`.

### A1 — Hysteresis margin is inert at K=2

- **Defect:** `atlas/router.py:94-101`. `relative_gap = (current − best) / spread` where
  `spread = max(valid) − min(valid)`. With exactly two charts, an incumbent that is not
  the argmin **is** the max, so `relative_gap = (max−min)/(max−min) = 1.0`, which always
  exceeds `m = 0.05`. The router is pure argmin; hysteresis never holds it.
- **Change:** normalise by the incumbent's own score instead of the batch spread:
  `relative_gap = (current − best) / current if current > 0 else 0.0`. Scale-free without
  being degenerate at K=2. Keep `m = 0.05` (a §1.7 non-negotiable — the *value* does not
  change, only the normaliser). Update the in-code comment to describe the new semantics
  and cite this spec.
- **Assertion:** a unit test with a 2-chart score dict where the incumbent is worse by
  1% of its own value must now **hold** (return the incumbent), and by 20% must
  **switch**. Under the old formula both switch. **Must fail before the fix.**
- **Re-run:** all E2 configs (see A3 for the combined re-run).

### A2 — `maybe_expand()` selects the incumbent on the verification chunk

- **Defect:** `atlas/expand.py:131-142`. `best_chart` is the argmin **over the held-out
  chunk**, then the candidate must beat it **on that same chunk** — a look-ahead
  advantage for the incumbent. Conservative, so it under-counts commits, but it is not
  the quantity N9 claims.
- **Change:** select the incumbent as the argmin over the **deficit chunks**
  (`self._deficit_chunks`), matching `expand.py:9-11`'s own docstring and proposal §2.
  Keep the verification comparison itself on the held-out chunk.
- **Assertion:** log both the old and new incumbent index on a run where they differ, and
  confirm the committed/rejected verdict is now computed against the deficit-chunk
  argmin. Report how many of E2's decisions change.

### A3 — Per-chunk expander state is never logged

- **Defect:** `atlas_out/e2_R2_cellB_q1/` has only a summary; the per-chunk decisions
  behind "3 charts committed" were never written, so N9 is stuck at L4.
- **Change:** in `scripts/run_e2.py`'s per-episode loop, add to each record:
  `strikes`, `probe_outcome`, `relative_gap`, `hysteresis_held` (bool), `committed`
  (bool), `library_size`.
- **Assertion:** the regenerated JSONL lets N9's commit count and the K=3 hysteresis
  binding rate be recomputed from raw records alone. This also closes
  `PAPER_FACT_CHECK` D4, currently flagged "unverified rather than clean".
- **Re-run (covers A1, A2, A3):** every E2 config — `e2_R1`, `e2_R1_lora4`, `e2_R2`,
  the three `*_posthysteresis` variants, `e2_confusion_matrix`, `e2_R2_cellB_q1`,
  `e2_R2_cellC_q1`, **plus a new q=3 run** (see A12/`PAPER_FACT_CHECK` C2). Write to
  **new directory names** — never reuse one. E2 runs no CEM planner (~0.2-0.4 s/episode),
  so this is ~$1 total.

### A4 — Reported `eval_umf` is measured on the checkpoint-selection set

- **Defect:** `scripts/run_e0.py:489-493` (the help text admits it),
  `atlas/harness.py:131-156`. The 8-trajectory set serves both early stopping (consulted
  up to 80× at `--eval-every 25 --patience 5`) and the final reported number. **This
  biases every UMF number in the paper**, not just N4's trend.
- **Change:** add `--num-test-trajs` (default 8) with a **new seed offset disjoint from
  both train and val** (val uses `seed_offset=10_000`; use `20_000`). Report `eval_umf`
  from the test set; keep reporting the val number separately as `val_umf` so the bias
  is measurable rather than merely asserted.
- **Assertion:** the seed manifest shows three disjoint seed ranges. `test_umf` and
  `val_umf` both present in `results.json`, and their difference is the bias estimate.
- **Re-run:** **no retraining.** Re-score the already-saved charts on the new test set —
  forward-only, no planner, ~$1.

### A5 — `analyze_n100.py` does not compute the statistics N1 cites it for

- **Defect:** the script never calls `paired_bootstrap` or `mcnemar_paired`, so N1's CI
  and McNemar *p* are not reproducible from `analysis_n100.json`.
- **Change:** add both, using `atlas/stats.py`'s existing functions unmodified.
  Regenerate the JSON.
- **Assertion:** the regenerated file contains CI `[-9.0, +7.0]` and `p = 1.000`.

### A6 — Oracle−random CI is structurally one-sided

- **Defect:** `d_i = oracle_i − random_i ≥ 0` by construction at every episode, so no
  resample can be negative. "CI excludes zero" is near-vacuous.
- **Change:** add `oracle_gap_permutation(per_chart_successes, n=10_000, seed=0)` to
  `atlas/stats.py` — **a new function; do not modify existing ones.** Permute chart
  labels within each episode to build the null for "is this library's oracle gap larger
  than chance". Return `(observed_gap, p_value, null_distribution_summary)`.
- **Assertion:** on a synthetic library where all charts are identical, the permutation
  *p* must be ≈1.0 and the observed gap ≈0. On the real N8 data, report both the gap and
  the permutation *p*.

### A7 — Partial-Kendall p-value is invalid

- **Defect:** Kendall's null does not account for the estimated OLS coefficients, so
  `partial_p` is anticonservative. Point estimates (−0.358, −0.374) are fine.
- **Change:** replace with a permutation test on the residuals in
  `scripts/analyze_n100.py`. Keep and continue reporting the point estimates.
- **Assertion:** the permutation *p* is reported alongside the (retained) coefficient,
  and is ≥ the old analytic *p*.

### A8 — `sr_by_bucket` silently drops episodes above 300px

- **Defect:** `pd.cut(bins=(0,80,120,300))` sends overflow to NaN, dropped by
  `groupby(observed=True)` without warning, so per-bucket *n* need not sum to 100.
- **Change:** open-ended top bucket; assert *n* sums to the episode count.
- **Assertion:** the assertion fires on the current data if any episode exceeds 300px.

### A9 — 20-trajectory chart has no seed manifest

- **Defect:** `atlas_out/e0_v3_dataset/` has only the chart and `results.json`. Its
  held-out status rests on prose. It anchors **N4** and is the chart behind **N1**.
- **Change:** none — `run_e0.py` already writes manifests.
- **Re-run:** retrain `ln_act` × R2 at 20 trajectories with the manifest saved, on T4
  (~$1). Write to a **new** directory; do not overwrite `e0_v3_dataset`.
- **Assertion:** the new `eval_umf` reproduces 0.336 within noise, and the manifest shows
  disjoint train/val/test seed ranges. **If it does not reproduce, stop and report — do
  not quietly adopt the new number.**

### A10 — R2 trio budgets unrecorded and unmatched

- **Defect:** `e0_v3_dataset`, `e0_v4_lora4`, `e0_v4_full` have no manifests;
  `harness.py:163-170` documents that `lora4` once OOM'd and was retrained at a smaller,
  confounding budget. `full`'s 0.728 is the shape of a budget confound.
- **Change — option (a), default:** report descriptively; drop any capacity *ordering*
  claim. **Option (b), only if Phase 1 finishes early:** retrain all three R2 kinds at
  one recorded matched budget now the OOM is fixed (~$3, T4).
- **Assertion (b only):** all three manifests show identical train/val/test seed sets and
  step budgets.

### A11 — Artifacts record the discredited LoRA parameter count

- **Defect:** `atlas_out/e0_v4_lora4/results.json` and `e0_v6_R1/results.json` record
  `params=10292640` while `PAPER_DRAFT.md` Appendix B says 118,176. `Chart.n_params()`
  (`atlas/chart.py:169-170`) sums `_params`, which at construction holds the 12 full base
  matrices; `lora_A`/`lora_B` are added only later by `update_from_predictor_`.
- **Change:** add `Chart.n_trainable_params()` returning the count that will actually
  receive gradients (for `lora4`, the `lora_A`/`lora_B` numel; otherwise `n_params()`).
  Record **both** in `results.json` as `params_stored` and `params_trainable`.
  Regenerate the two affected files with a dated supersede note.
- **Assertion:** `n_trainable_params()` returns 118,176 for `lora4` and 10,764 for
  `ln_act`. **Must differ from `n_params()` for `lora4` before the fix.**

### A12 — Dead CLI dispatches that silently no-op

- **Defect:** `scripts/make_tables.py:179` offers `--table T5` but `main()` (`:186-189`)
  dispatches only T1 and T2 — passing T5 produces **no error and no output**.
  `scripts/make_figures.py:114` accepts only `{F1, F2}` while `atlas/plots.py` defines
  `umf_traces`, `crosspolicy`, `umf_vs_sr` (S1/S2/S3) that nothing dispatches.
- **Change:** implement `make_t5` (the E0 capacity table: Adapter × {Params, KB, ΔUMF,
  Success, % of full}) and wire the S1/S2/S3 figure dispatches. Where a table or figure
  genuinely cannot be produced from available data, **raise a clear error rather than
  silently doing nothing.**
- **Assertion:** every value in each `choices` list either produces output or raises.

### A13 — `merge_planning_shards.py` does not check contiguity

- **Defect:** merging is otherwise clean (verified 2026-08-27: hard-fails on duplicates,
  sorts by episode, filters nothing, 0 content mismatches on both sweeps). But an
  upstream-missing episode would pass silently.
- **Change:** assert the merged episode indices are contiguous from `--episode-start`.
- **Assertion:** the check fires on a deliberately gapped pair of shards.

### A14, A15 — Documentation corrections

- **A14:** `ATLAS_SUMMARY.md` states the converged-CEM spread as "3.8–8.3px"
  unqualified; the real range over all six seed/kind cells is **3.77–27.15px**
  (`ln_act` seeds 1 and 2 are 17.9 and 27.2). `E0_RESULTS.md` hedges correctly; the
  summary dropped the hedge. Restate with the full range; remove "tight cluster".
- **A15:** the "pre-fix +55.6pp" Cell B figure has **no surviving raw records** — drop
  the before/after framing entirely and report only current numbers.
- Both are edits to `ATLAS_SUMMARY.md` with a dated banner. `PAPER_DRAFT.md` is Phase 6.

---

# PHASE 2 — Gates, latent hazards, unfinished audit items

Owner agent: `atlas-process-auditor`. Budget $0. Full specs expand on dispatch; the two
that matter most:

- **C1 — G2 asserts nothing.** `scripts/smoke_gates.py:203-217` is a literal
  `if ...: pass` printing PASSED unconditionally, on i.i.d. `randn` with no learnable
  structure. Rewrite on structured data (a learnable predictor-weight perturbation, as
  G3a already uses) with a real assertion, driven through `_open_loop_rollout` rather
  than the hand-rolled zeroed-proprio path. **The new gate must FAIL on a deliberately
  leaked chart** — demonstrate that, or the rewrite is worthless.
- **C2 — G5 is a tautology.** `smoke_gates.py:418-434` checks only that `paired_seed()`
  ignores its `arm` argument, which is true by inspection since `arm` is never
  referenced. Build two envs at the same seed; assert identical init states and goals.
  **Must FAIL on deliberately mismatched seeds.**

Also in Phase 2: C3 (extend G1 to refined charts and `kind="full"`), C4
(`restore_pretrained_()` + rename the misnamed `restore_`), C5 (wrapper-bypass assertion
for `VisualCorruption`), C6 (`lora4` online refinement raises on an empty param list),
C7 (`diagnose_cem_costs.py` writes only at the end — the documented cause of two lost
dose-response runs), C8 (remote vs local hub), C9 (unloaded configs), C10 (E1 harness,
only if revived), D1–D10 (the audit's own unfinished items, including the **never-done**
red-flag sweep and the **completely unverified** S-1…S-8 process claims).

---

# PHASE 3 — E4/continual path

Owner agent: `atlas-fixer`. Budget ~$0.50. **Nothing on disk is contaminated — E4 has
never run — so every fix here is free.** But launching without them produces a result
that reads as a finding and is a wiring artifact.

### B1 (CRITICAL) — ATLAS arm can never commit a chart

- **Defect:** `atlas/harness_e4.py:216-217` passes `next_encoder_output=None,
  next_actions=None` unconditionally, so `atlas/loop.py:140-150`'s guard is
  unsatisfiable and `maybe_expand()` is never called for arm 6. The comment at
  `harness_e4.py:198-203` describes a "two-deep chunk buffer" that was never built.
  **Under the default `expansion_start_library="full"`, correct ATLAS behaviour and this
  bug both produce "0 commits" — indistinguishable.**
- **Change:** implement the two-deep buffer. Hold chunk *k* as deficit data; pass chunk
  *k+1* plus `next_proprio_ctxt` as the verification chunk on the following replan.
  Copy the pattern from `scripts/run_e2.py:296-339`, verified leakage-free in
  `CODE_AUDIT.md` §9.3. Three arguments must be threaded, not two.
- **Assertion:** in a 3-episode smoke, the ATLAS arm reaches `maybe_expand()` and logs a
  `probe_outcome` other than `"not_ready"`. **Must fail before the fix** — confirm
  `probe_outcome` is `"not_ready"` on every episode of the unfixed code.

### B2 (CRITICAL) — Arm 2 is behaviourally identical to arm 3

- **Defect:** `AdaJEPA.reset()` (`atlas/adajepa.py:94-104`) has no caller outside
  `scripts/smoke_e4.py:142`. In production plain AdaJEPA never re-initialises, so it *is*
  Persistent-AdaJEPA and the ladder's persistence rung — the one labelled "*(ours)*" —
  measures zero by construction. The 5-transition buffer is also never cleared, so its
  window spans episode **and regime** boundaries.
- **Change:** call `state.adapter.reset()` at the top of `run_e4_episode` when
  `variant == "adajepa"`, and clear the buffer.
- **Pre-cleared:** the key-namespace concern is resolved — for `ln_act`,
  `pretrained_state` (from `state_dict()`) and `param_names` (from `named_parameters()`)
  come from the same plain `nn.Module` with no parametrization, so they coincide exactly
  and `load_state_dict(strict=False)` genuinely restores.
- **Assertion:** arms 2 and 3 produce **differing** JSONL rows over 2 episodes.
  **Must fail before the fix** — they must be identical on the unfixed code.

### B3 (CRITICAL, suspected) — Motion gate calibrated at the wrong chunk size

- **Defect:** `scripts/run_e4.py:182-190` takes the 10th percentile of **whole-trajectory**
  displacement (`traj_len=10`, from only **3** trajectories — a meaningless quantile) and
  applies it to **single-model-step** chunks at nas=1. If it gates everything, `umf()`
  returns `None` everywhere → the router never switches → `Expander.record()` returns
  early → no strike, no expansion in **any** arm → arms 4-7 collapse into arm 1 while
  writing a plausible success table.
- **Change, in this order:** (a) **measure first** — a short script comparing the
  distribution of 1-model-step latent displacements under R0/R2 against
  `compute_motion_gate`'s current output. **Report the numbers before changing
  anything.** (b) Recalibrate at the chunk size actually scored, over ≥30 trajectories.
- **Assertion:** gated fraction in a smoke run lands near E2's measured ~20-30%, not
  ~100% and not 0%.

### B4 (CRITICAL) — ATLAS arms crash unless an AdaJEPA arm ran first

- **Defect:** `scripts/run_e4.py:165-168` freezes all predictor params; only
  `AdaJEPA.__init__` re-enables them, and `load_state_dict` does not restore
  `requires_grad`. `atlas_refine()`/`_fit_candidate()` then build Adam over frozen
  tensors and `loss.backward()` raises. **Because `modal_e4.py:108` runs
  `run_e4.py --arms <single arm>` in a fresh subprocess per container, this is not an
  ordering hazard — it is guaranteed to fire on every ATLAS-arm container.**
- **Change:** re-enable `requires_grad` on the chart's own parameter surface inside
  `atlas_refine` and `_fit_candidate`, making each arm self-contained.
- **Assertion:** `python scripts/run_e4.py --arms atlas --episodes 2 --seeds 1` runs to
  completion standalone. **Must fail before the fix.**

### B12 (CRITICAL) — Modal's multi-seed sweep produces relabelled copies

- **Defect:** `modal/modal_e4.py:109` always passes `--seeds 1` (a *count*, not the
  requested seed). Inside `run_e4.py`, `for seed_run in range(profile_seeds)` therefore
  only ever runs local `seed_run=0`; `get_stream(...)` is built with `seeds=1` so
  `stream_s2` only generates its `seed_run=0` stream; `local_seed` is hardcoded to 0.
  `modal_e4.py:126,138` then **rewrites each record's `seed_run` field after the fact.**
  A "3-seed" sweep yields **bit-identical episode data under three labels**.
- **Change:** thread the requested seed through to stream construction and to
  `local_seed`. Add a `--seed-run-offset` or equivalent so containers genuinely differ.
- **Assertion:** two containers at `seed_run=0` and `seed_run=1` produce **different**
  `init_block_pos_diff` sequences. **Must fail before the fix.**

### B13 — `smoke_e4.py`'s assertions are substantially weaker than they look

- **Defect:** of eight assertions, two are **structurally vacuous** (`frozen`'s
  `library_size==1` tests a hardcoded literal since `build_arm_state` sets
  `library=None` for that arm; `atlas_fixed`'s `probe_outcome != "committed"` tests a
  control-flow guarantee since `expansion_mode="fixed"` reaches neither expansion
  branch); one is near-vacuous **and contradicts its own docstring** (the docstring
  claims `raw_steps_per_replan` entries are checked as multiples of `frameskip`, but the
  code checks only `n >= 0` on a Python list length, which cannot be negative); one has a
  latent vacuous mode (the AdaJEPA-reset check filters both sides by the same
  `param_names`, so a total namespace divergence would compare two empty dicts and pass).
- **Change:** implement the `frameskip` divisibility check the docstring promises; assert
  non-empty coverage in the reset and oracle checks; replace the two structural
  assertions with dynamic ones. **This must land before Phase 5 leans on this file.**

### B5–B11, B14 — the remainder

- **B5:** `atlas/streams.py:86-87` puts `segment_idx` in the seed key, so first visit and
  final revisit get different init/goal — **RQ4's paired Δ is unmeasurable as wired.**
  Key on regime-visit index instead. *Assertion:* segment 0 ep *i* and segment 2 ep *i*
  share `init_block_pos_diff`.
- **B6:** `atlas/loop.py:152-170` commits a byte-identical clone with no gradient step,
  so it ties its parent and loses the argmin tie-break — inert library mass, and not what
  the compared methods do. Call `_fit_candidate()` on the deficit chunks before
  `library.add()`. Makes the 5→6 rung differ by **exactly** "verifies".
  *Assertion:* a committed detect-only chart differs in weights from its parent.
  **Must fail before the fix.**
- **B7:** `make_tables.py:126` pairs by equal **length**, not equal key set. Intersect on
  `(arm, seed_run, global_episode_idx)` and assert.
- **B8:** arms 4/5/6 skip refinement when `current_idx == 0` while 2/3 always refine, so
  the 3→4 rung differs by two mechanisms. Refine a **clone of c₀** on first selection
  rather than skipping. Record the decision explicitly — this is a design call, not
  purely a bug.
- **B9:** arms 2/3 use a 5-transition buffer; arms 4/5/6 see one chunk, violating plan
  §7.6's explicit "same buffer size as AdaJEPA". Give `atlas_refine` the same 5-chunk
  buffer. The offline-pre-training difference is *specified* by §7.4 and stays — but must
  be named in the paper as part of what the 3→4 delta measures.
- **B10:** `run_e4.py:154` divides a raw-step budget by a chunk count. Inert at
  `horizon=6` but wrong; the planner never shortens its plan near episode end.
- **B11:** re-seed the CEM generator per episode from `spec.seed`, so resumed and
  uninterrupted runs are identical.
- **B14:** no `evict()` exists — `Library.add()` raises when full and both callers
  pre-check `is_full()`, so growth simply halts at `K_max=10` and nothing is retired.
  Log a `library_full` flag per episode so cap-hits are visible. State in the paper that
  eviction is unimplemented and out of scope.
