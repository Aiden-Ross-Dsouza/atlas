# ATLAS — Phase dispatch prompts

**Last updated: 2026-08-28.**

## What this file is

The exact, copy-pasteable prompts for executing each phase of
`SUBMISSION_PLAN.md`. `SUBMISSION_PLAN.md` Part D describes *what* each phase is;
this file is *how to dispatch it*.

**Start a fresh session for each phase.** Open a terminal, `cd` to the repo, run
`claude` — plain, **not** `--continue` or `--resume`. A fresh session auto-loads
only `CLAUDE.md`; `--continue` replays an entire prior conversation into context
and costs real money every turn. `CLAUDE.md` points a cold session at
`SUBMISSION_PLAN.md`, which points here.

**Run phases one at a time.** Several phases touch the same files
(`atlas/loop.py`, `atlas/expand.py`, `scripts/make_tables.py`,
`atlas/harness_e4.py`). Concurrent phases produce silent overwrites, not clean
conflicts.

## Status

| Phase | Agent | Cost | Status |
|---|---|---:|---|
| 0 Setup | — | $0 | **DONE** 2026-08-27 |
| 1 Tier A — Stage 1 (code) | `atlas-fixer` | $0 | **DONE** 2026-08-27, see `FIXLOG.md` |
| 1 Tier A — Stage 2 (re-runs) | `atlas-fixer` + orchestrator | ~$0 (ran locally, no Modal needed — see `FIXLOG.md`) | **DONE** 2026-08-28 — E2 re-run (A1+A2+A3, incl. all three `*_posthysteresis` variants and a genuinely new, isolated q=3 measurement — the first pass's substitution of `e2_R2` for "new q=3 run" was caught and corrected) and A4 re-score both done; A9 retrain done (reproduces: new `val_umf` 0.3335 vs. old 0.3357 within noise — but surfaced a previously unmeasured, worse `eval_umf`=0.4125 on the disjoint test split, disclosed alongside the old number, not substituted for it). All 11 result dirs renamed `atlas_out/*_phase1stage2_2026-08-28` for unambiguous phase attribution. Dated supersede banners added to `ATLAS_SUMMARY.md`/`E0_RESULTS.md`/`E2_RESULTS.md`; full trail in `FIXLOG.md`. `EVIDENCE_LEDGER.md` bookkeeping still outstanding. Also discovered, not fixed: `expand.py::_fit_candidate`'s chart-commit decision is non-deterministic run-to-run — confirmed root cause is unseeded CUDA gradient descent (dropout ruled out directly), logged in `FIXLOG.md`, no FIX_SPEC entry yet |
| 2 Gates + unfinished audit | `atlas-process-auditor` | $0 | **DONE** 2026-08-28 — G1-G6 rewritten/verified (G2 and G5 independently re-confirmed to fail on deliberately broken input, not just on the agent's report); D1-D4/D10 done; see `FIXLOG.md`, `RED_FLAG_SWEEP.md`, `E0_DIRECTORY_INVENTORY.md`. Known open item: `gate_g3a` is flaky/order-dependent when run after G1/G2 in `--all` — pre-existing, logged not fixed |
| 3 E4/continual fixes | `atlas-fixer` | ~$0.50 | not started |
| 4 Defence experiments | `atlas-runner` / `atlas-analyst` | ~$31 | not started |
| 5 Continual stream | `atlas-runner` | ~$19 | not started |
| 6 Write-up | `paper-drafter` → `paper-fact-checker` | $0 | not started |

## Recommended order

Phase 1 Stage 1 (done) → Phase 1 Stage 2 (done) → Phase 2 (done) → **Phase 3** →
Phase 4 → Phase 5 → Phase 6.

Phase 3 is next: it unblocks Phase 5 and its Step 0 is a cheap go/no-go
measurement that could change Phase 5's entire design.

---

# PHASE 1 — Stage 2 (the re-runs)

Stage 1 is complete. Stage 2 spends ~$3.20 regenerating the numbers Stage 1's code
changes invalidated. Dispatch `atlas-fixer`.

```
Read CLAUDE.md, SUBMISSION_PLAN.md, research_audit/FIX_SPEC.md (PHASE 1) and
research_audit/FIXLOG.md. Phase 1 Stage 1 is already applied — see FIXLOG.

Dispatch the `atlas-fixer` subagent with this task:

  Perform Phase 1 Stage 2: the re-runs deferred from Stage 1.

  1. A1+A2+A3 combined E2 re-run. Re-run EVERY E2 config under the corrected
     hysteresis normaliser, the corrected incumbent selection, and the new
     per-decision logging: e2_R1, e2_R1_lora4, e2_R2, the three
     *_posthysteresis variants, e2_confusion_matrix, e2_R2_cellB_q1,
     e2_R2_cellC_q1, PLUS a new q=3 run (PAPER_FACT_CHECK C2 — that column
     currently reports three zeros that were never measured).
     WRITE TO NEW DIRECTORY NAMES. Never reuse one.
     E2 runs no CEM planner (~0.2-0.4 s/episode), so this is ~$1.
     Report: how many routing decisions changed, how many expansion
     decisions changed, and the new N6/N7/N9 numbers.
     CHECK SPECIFICALLY: FIXLOG's A1 entry flags that sdyn scores can be
     <= 0, in which case the new normaliser's `else 0.0` branch means
     ALWAYS HOLD. Report sdyn's routing accuracy before and after, and say
     plainly whether this changed it.

  2. A4 re-score. Re-score the already-saved charts on the new
     --num-test-trajs split. Forward-only, no planner, no retraining. ~$1.
     Report eval_umf (test) and val_umf side by side for every chart; their
     difference is the bias estimate that has never been measured.

  3. A9 retrain. Retrain ln_act x R2 at 20 trajectories with the seed
     manifest saved, on T4. ~$1. NEW directory — do not overwrite
     e0_v3_dataset.
     If the new eval_umf does NOT reproduce 0.336 within noise, STOP and
     report. Do not quietly adopt the new number. That chart anchors both
     N4 and N1.

  4. Update research_audit/EVIDENCE_LEDGER.md with every regenerated number
     and its new evidence level. Add dated supersede banners to
     ATLAS_SUMMARY.md and E2_RESULTS.md for anything that moved.

  A10 option (b) — retraining the R2 trio at matched budget, ~$3 — is
  OPTIONAL. Do it only if 1-4 finish cleanly and I approve.

Re-run the key assertions yourself before accepting the report.
```

---

# PHASE 2 — Gates and unfinished audit items

In progress. If it stalls, the priority order is C1 and C2 (the dead gates) first,
then D1 (the never-done red-flag sweep) and D2 (S-1 through S-8, completely
unverified). C3/C5/C7/C8/C9 are lower value. **C6 has been moved to Phase 3** — see
below. **C10 is skipped** (Phase 5 routes around the E1 harness).

D2 is genuinely large — it verifies the claims that license trusting everything
else. If the agent is running long, that is the item to let finish rather than cut.

---

# PHASE 3 — E4/continual fixes

Unblocks Phase 5. Nothing on disk is contaminated (E4 has never run), so every fix
is free. Budget ~$0.50 for smoke runs.

```
Read CLAUDE.md, then SUBMISSION_PLAN.md, then research_audit/FIX_SPEC.md
(the "PHASE 3" section) and research_audit/FIXLOG.md. You are orchestrating
Phase 3 only.

Phase 3 fixes the E4/continual path. Launching Phase 5 without these fixes
produces a result that reads as a finding and is a wiring artifact.

Dispatch the `atlas-fixer` subagent (subagent_type: "atlas-fixer") with this
task:

  Apply B1-B14 from the "PHASE 3" section of research_audit/FIX_SPEC.md,
  PLUS item C6 (moved here from Phase 2 — see Step 1). Follow the six rules
  at the top of FIX_SPEC.md. Log every change in research_audit/FIXLOG.md.

  Work in this order and STOP after Step 0.

  STEP 0 — MEASURE, CHANGE NOTHING. This is a go/no-go.
    B3 asks whether the motion gate is calibrated at the wrong chunk size.
    scripts/run_e4.py:182-190 takes the 10th percentile of WHOLE-TRAJECTORY
    displacement (traj_len=10, from only 3 trajectories) and applies it to
    SINGLE-model-step chunks at nas=1. If it gates everything, umf() returns
    None everywhere, the router never switches, Expander.record() returns
    early, and arms 4-7 silently collapse into arm 1 while still writing a
    plausible success table.
    Write a short script measuring the actual distribution of 1-model-step
    latent displacements under R0 and R2, and compare against
    compute_motion_gate's current output for traj_len=10, num_trajs=3.
    REPORT THE NUMBERS. Do not change the gate yet. Then STOP.

  STEP 1 — unblock testing. Nothing else can be tested until B4 lands.
    B4 + C6 ARE ONE FIX — do them together, they modify the same lines.
      B4: atlas_refine and _fit_candidate build an Adam over parameters that
          scripts/run_e4.py:165-168 froze, and loss.backward() raises. This
          is CRITICAL, not conditional: modal_e4.py runs one arm per fresh
          subprocess, so EVERY ATLAS-arm container is guaranteed to crash,
          not just some orderings.
      C6: the same selection is also wrong for lora4 — after parametrization
          the base name leaves named_parameters(), so the list is EMPTY and
          Adam([]) raises. atlas/harness.py:118-125 already does this
          correctly for the offline path; copy that approach.
      Combined fix: make parameter selection correct for BOTH kinds in one
      change — enable requires_grad on the chart's own surface (including
      lora_A/lora_B for lora4), then select by requires_grad.
      Assertions: `--arms atlas --episodes 2 --seeds 1` runs standalone with
      kind=ln_act, AND the same with --kind lora4 does not raise on an empty
      optimizer list. BOTH MUST FAIL before the fix.
      Record in FIXLOG that C6 was moved from Phase 2 and why.

    B13 Strengthen scripts/smoke_e4.py BEFORE it is used as Phase 5's gate.
        Two of its eight assertions are structurally vacuous (frozen's
        library_size==1 tests a hardcoded literal; atlas_fixed's
        probe_outcome != "committed" tests a control-flow guarantee), one
        checks only n >= 0 on a list length while its docstring claims a
        frameskip-divisibility check that does not exist, and one is
        vacuous-if-namespace-diverges. Implement the divisibility check,
        assert non-empty coverage in the reset and oracle checks, and
        replace the two structural assertions with dynamic ones.

  STEP 2 — the dead mechanisms. Each MUST FAIL before the fix; run the
  assertion on unfixed code first and paste that output.
    B1  Implement the two-deep chunk buffer. harness_e4.py:216-217 passes
        next_encoder_output=None unconditionally, so loop.py:140-150's guard
        is unsatisfiable and the ATLAS arm can never commit. Copy the
        pattern from run_e2.py:296-339 (verified leakage-free). Three
        arguments must be threaded, not two.
        Assertion: ATLAS arm logs a probe_outcome other than "not_ready".
        Before the fix it must be "not_ready" on every episode.
    B2  Call adapter.reset() per episode for variant=="adajepa", and clear
        the buffer. The key-namespace concern is already resolved — for
        ln_act the two key sets coincide exactly, so the reset genuinely
        restores.
        Assertion: arms 2 and 3 produce DIFFERING JSONL rows over 2
        episodes. Before the fix they must be identical.
    B6  atlas_detect commits a byte-identical clone with no gradient step,
        so it ties its parent and never wins the argmin. Call
        _fit_candidate() on the deficit chunks before library.add().
        Assertion: a committed detect-only chart differs in weights from its
        parent. Before the fix it must be identical.
    B12 modal_e4.py:109 always passes --seeds 1, so seed_run only relabels
        output records — a multi-seed sweep yields bit-identical data under
        different labels. Thread the real seed through to stream
        construction and local_seed.
        Assertion: containers at seed_run=0 and seed_run=1 produce DIFFERENT
        init_block_pos_diff sequences. Before the fix they must be identical.

  STEP 3 — the remainder: B5, B7, B8, B9, B10, B11, B14.
    B5  is what makes RQ4 measurable at all — assert segment 0 ep i and
        segment 2 ep i share init_block_pos_diff.
    B8  is a DESIGN DECISION, not purely a bug. Refine a clone of c0 on
        first selection rather than skipping refinement, so "adapts" is held
        constant across the 3->4 rung. Record the decision and rationale
        explicitly in FIXLOG.

  Do not touch atlas/score.py::umf, atlas/stats.py's existing functions, or
  scripts/run_e0_planning.py's planning loop.

When the agent returns from Step 0, report the gate measurement to me before
authorising Steps 1-3. When it returns from Steps 1-3, do NOT accept its
report as evidence: re-run the before/after assertions for B1, B2, B4+C6, B6
and B12 yourself and read the actual output.
```

**Why Step 0 is a hard stop.** B3 is the only item whose answer could invalidate
Phase 5's design. If the gate swallows essentially every 1-model-step chunk, `nas=1`
is the wrong protocol and the fallback is `nas=2` (3 replans, ~$8 more, but a gate
that passes traffic). A five-minute measurement guarding a multi-hour commitment.

---

# PHASE 4 — Defence experiments

The experiments that defend the paper's claims against the two reviewer objections
it currently cannot answer. Ordered cheapest-first, so each buys information before
the next is committed.

**Note the code prerequisites.** E-A needs two changes that are *not* in the B/C
register: `--save-latents` on `diagnose_cem_costs.py`, and C7 (per-seed incremental
write, currently in Phase 2 — the documented cause of two lost dose-response runs).
Dispatch `atlas-fixer` for those before `atlas-runner` launches E-A.

```
Read CLAUDE.md, then SUBMISSION_PLAN.md — Part C in full (both scientific
questions and all five experiments E-A through E-E), plus Part E for budget.
You are orchestrating Phase 4. Total ~$31; run the sub-steps in the order
below and report after each.

STEP 4a — E-B, free, run FIRST as a go/no-go.
  Dispatch the `atlas-analyst` subagent:
    atlas_out/umf_locality.json already recomputes UMF restricted to the
    top-k most-MOVING DINOv2 tokens rather than the ~97% static white
    background that global UMF averages over. Reading it: baseline global
    0.367 / top16 0.238 / SR 45%; ln_act 0.336 / 0.204 / 50%; lora4 0.329 /
    0.168 (BEST moving-token UMF) / 40% (WORSE planning). So the localized
    metric ALSO fails to rank charts by planning competence. This is
    currently unreported and is a stronger result than the diagnostic's own
    hypothesis.
    Two extensions, both from data already on disk, ~$0:
      (i)  Correlate per-candidate UMF against per-candidate cost-rank error
           using the per-candidate arrays in atlas_out/cost_ranking_*. Does
           a candidate the chart predicts better actually get ranked better?
      (ii) Score UMF on the CEM candidate distribution rather than on demo
           replays, and ask whether THAT UMF ranks charts consistently with
           planning outcome.
    (ii) is the diagnostic-only version of E-A and is the go/no-go for it.
    Recompute everything from raw per-candidate arrays, never from a
    summary. Update EVIDENCE_LEDGER.md.

STEP 4b — E-E, ~$4. Defends the project's ONLY positive result.
  Dispatch the `atlas-runner` subagent:
    E2 scores routing "correctness" by REGIME LABEL, not by which chart
    plans better — in direct tension with the paper's own title, and with no
    planner in E2's loop at all. Given the N1/N2 dissociation, a
    "correctly" selected chart is not thereby shown to plan better.
    Take E2's already-collected trajectories and their seeds. For each seed,
    run the CEM planner under EACH chart, and answer directly: is the
    UMF-argmin chart also the better-planning chart?
    Reuse run_e0_planning.py's sample_dataset_init_goal and block_success()
    — do NOT use atlas/harness.py::run_e1_episode, whose random-goal
    sampling and wrong success criterion returned 0% for every router
    including the oracle.
    ~3 charts x 40 seeds x ~150 s at nas=6.
    Report: agreement rate between "UMF-selected" and "actually planned
    better", against chance; and the planning-SR gap between UMF-routing and
    random-routing over the same episodes.
    Either outcome is publishable — high agreement upgrades the routing
    result to what the paper needs; chance agreement is the dissociation
    appearing a third time, inside the positive result.

STEP 4c — E-A, ~$6. The action-distribution defence.
  FIRST dispatch `atlas-fixer` for the two code prerequisites:
    - Add --save-latents to scripts/diagnose_cem_costs.py so it stores the
      per-candidate encoder outputs alongside costs and true distances.
    - Apply C7 if Phase 2 has not: write output per seed, not once at the
      end. Two dose-response runs were already lost to this.
  THEN dispatch `atlas-runner`:
    Collect ~30 seeds x 300 candidates under R2 using the CEM candidate
    batch (the planner's own query distribution, rolled out for real).
    Fit ln_act on those with the existing run_e0_finetune loss.
    Evaluate on HELD-OUT seeds against three read-outs:
      1. UMF on held-out candidates
      2. cost-ranking rho — the quantity that decides what actually
         executes, and far more sensitive than binary SR (n=20 seeds
         suffices where SR needs n=100)
      3. planning SR at N=100, paired against the existing baseline
    Report all three. Do not interpret — just report.

STEP 4d — E-D, ~$12. Separates feedback from compute.
  Dispatch `atlas-runner`:
    N5's +10pp at nas=2 is confounded: plan_length stays pinned at
    horizon=6 regardless of steps_left, so each of nas=2's three replans
    runs a FULL 6-step search — 3x the compute over the same 30 raw steps.
    "Feedback helped" and "more search helped" are currently inseparable.
    Run three paired arms at N=40:
      (1) nas=6, iterations=30  — 1 replan, 1x compute  [already exists]
      (2) nas=6, iterations=90  — 1 replan, 3x compute  [NEW — the missing
          control nobody ran]
      (3) nas=2, iterations=30  — 3 replans, 3x compute [exists at N=20,
          extend to 40]
    If (2) ~ (1), the nas=2 gain is genuinely feedback.
    If (2) ~ (3), it was compute all along.

STEP 4e — E-C, ~$9. Retracts an unsupportable claim.
  Dispatch `atlas-fixer` for one change, then `atlas-runner`:
    Code: add --collect-num-act-stepped to run_e0.py (currently hardcoded
    to 1 with no flag), and default the collection CEM budget to the eval
    budget.
    Run: re-collect closed_loop data at 300x30 (matching eval) keeping
    nas=1, retrain ln_act, evaluate at N=100.
    This leaves exactly ONE deliberate mismatch — replan frequency — which
    E-D measures directly. The current closed_loop result was collected at
    100x10 with nas=1 and evaluated at 300x30 with nas=6: a 9x search-budget
    gap AND opposite extremes of replan frequency, so it cannot distinguish
    "closed-loop replay doesn't help" from "the collector was too weak".

RULES FOR EVERY STEP:
  - Always use --detach for Modal runs.
  - Download artifacts IMMEDIATELY and verify with ls -la and a record
    count. Three headline results were nearly lost because runs completed
    on a remote volume and were never pulled down.
  - Never reuse an output directory name.
  - Smoke before you spend: 2 episodes or --profile, inspect every logged
    field, THEN launch.
  - Report only numbers read from a downloaded raw file, never from a
    summary or a log line.
  - Update EVIDENCE_LEDGER.md after each step.

Report to me after each of 4a-4e before starting the next.
```

**Cut order if time-bound:** 4a and 4b are near-free and defend the positive result
— never cut them. 4c is the action-distribution defence and is the most likely to
turn the paper constructive rather than purely negative. **Drop 4d first, 4e
second.**

---

# PHASE 5 — The continual stream

See `SUBMISSION_PLAN.md` Part D, Phase 5 for the full design. **Requires Phase 3
complete.** Dispatch `atlas-runner`. ~$19, ~4 h wall sharded 7 ways.

Key points to carry into the prompt when you get there: 10 episodes/segment, 1 seed,
A,B,A,B over R0/R2, nas=1, iterations=10, 7 arms one container each. Pre-register
the read-out in `research_audit/E4_PREREGISTRATION.md` **before looking at results**.
Smoke all seven arms at 2 episodes/segment and inspect `gated`, `strikes`,
`probe_outcome`, `charts_committed_cumulative`, `selected_idx`, and that arms 2 and 3
differ — before launching.

**Honest sizing to state in the paper:** 40 paired episodes/arm supports "the
mechanism runs end to end and here is what it does" and a charts-committed count. It
does **not** power a success-rate comparison between adjacent ladder rungs.

---

# PHASE 6 — Write-up

`paper-drafter` then `paper-fact-checker`. Runs in parallel with everything else.
Scope: all `PAPER_FACT_CHECK.md` findings, `OPUS_REMAINING_TASKS.md` §B items 9-22,
the dissociation figure (`SUBMISSION_PLAN.md` Part A-iv — every input is already
local), stripping the nine internal-provenance HTML comments before any LaTeX
conversion, relabelling S-dyn as a **dynamics-fingerprint** baseline, stating N1 and
N3 as one mechanism seen as cause and consequence, fixing the MBCD venue (AAMAS
2021, not ICML), and adding Lambert 2020 / Grimm 2020 / Singh 2026 / Vakalis 2026 —
framing the dissociation as a **replication in a new substrate**, not a discovery.

**Exit gate:** `paper-fact-checker` returns zero section-A findings.
