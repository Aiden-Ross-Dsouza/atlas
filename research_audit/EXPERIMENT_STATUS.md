# ATLAS — Experiment & Component Status

**Last updated: 2026-08-27, pass 1 — `Implemented?` and `Blocking dependency` populated by reading `ATLAS_proposal_v7.md` and `ATLAS_implementation_plan_v2.md` against the actual repository tree. `Run?` populated from the presence or absence of raw output under `atlas_out/`. `Results verified?` left as `pending` everywhere; the results-auditor agent fills it in.**

---

## What this file is

The single place to look up, for any experiment or named sub-mechanism in
the ATLAS proposal, whether it (a) exists in code, (b) has actually been
executed, and (c) has had its numbers independently checked. Written during
a pre-submission audit for the NeurIPS 2026 Workshop on Continual World
Models (Idea Track, deadline 29 Aug 2026 AoE).

Repository root: `D:/Shubham/DeepLearning/Atlas/atlas/`.
Raw experiment output: `atlas_out/`.
Claim identifiers (C1, N1, RQ4, ...) refer to rows in
`research_audit/CLAIMS_MATRIX.md`.

**`Implemented?` means code exists and is reachable — evidence level L2.
It does not mean the code is correct (L3).** Those are different columns of
evidence and the distinction is the whole reason this file exists.

---

## Section 1 — The five named experiments

| Experiment | Implemented? | Run? | Results verified? | Blocking dependency | Notes |
|---|---|---|---|---|---|
| **E0** — adapter capacity (RQ0) | **yes** | **yes** | pending | none | `scripts/run_e0.py` (878 lines, training), `scripts/run_e0_planning.py` (601 lines, planning eval). Extensive output: `atlas_out/e0*`, `e0_v3_*` through `e0_v6_*`, `e0_planning_*`. The pre-registered decision rule (smallest kind reaching >=90% of `full` on both metrics in both regimes) became **inapplicable** when `full`'s gain went negative; the project then judged E0 against a 15pp bar that appears in neither design document (see `CLAIMS_MATRIX.md` S-6). Only `ln_act` x R2 has been evaluated at N=100. |
| **E1** — fitness routing (RQ1), declared "THE GATE" | **partial** | **no** | n/a | **REOPENED 2026-08-27 — see notes** | `scripts/run_e1.py` (351 lines) and `atlas/harness.py::run_e1_episode` exist and have passed a smoke test only. `atlas_out/e1_smoke/episodes.jsonl` is 1.8 KB and `atlas_out/e1_verify/episodes.jsonl` is 0.5 KB — smoke artifacts, not the specified 60 episodes x 3 seeds. The project originally closed E1 on the argument (`HANDOFF.md` §7.1) that the oracle-minus-random spread over the real library is only 2.5-3.3pp, below E1's own 10pp reporting threshold — **but that spread was computed with a fabricated data point** (audit found this 2026-08-27; the `chart_R1` row was a duplicated baseline array, not a real evaluation). The coordinating session ran the real evaluation: **oracle 60.0%, random 46.7%, spread 13.3pp, CI [3.3,25.0]** — clears the 10pp bar. The original argument for skipping E1 no longer holds; whether to actually run it is now an open decision. See `CLAIMS_MATRIX.md` row N8, `RESULTS_AUDIT.md` §11. **A second, still-unresolved blocker remains regardless:** at `num_act_stepped=6` one replan covers a whole 30-step episode, leaving no room for E1's specified "2 warmup replans then route" structure (implementation plan §7.0a flags this as needing "a real decision", never made) — this would need resolving before any real E1 stream run, independent of the denominator question above. |
| **E2** — appearance vs dynamics (RQ2) | **yes** | **yes** | pending | none | `scripts/run_e2.py` (500 lines). Output in `atlas_out/e2*` (9 directories). **Deviates from spec in two documented ways:** no CEM planner at all (routing accuracy computed over collected trajectories, making it ~$0.10/run instead of the planned ~6 GPU-h), and `dark` corruption instead of the specified `colour_change` (which was measured to alter only 5.6% of pixels on this environment). Both deviations are defensible and disclosed; the first means E2 measures the selector in isolation from planning. |
| **E3** — expansion ladder (RQ3) | **partial** | **no** | n/a | **Blocked on E4** — E3 runs inside the E4 stream by design (plan §7.4: arms 4→5→6 *are* E3) | The expansion machinery itself (`atlas/expand.py`, 10.5 KB) exists and has been observed to commit charts end-to-end **once**, in E2's `q=1` diagnostic (`atlas_out/e2_R2_cellB_q1`, 3 charts committed). That is a mechanism demonstration, not the RQ3 comparison. The actual RQ3 result — charts committed by ATLAS vs detect-only vs fixed-library against a true regime count of 2 — requires the E4 stream and does not exist. |
| **E4** — continual stream (RQ4) | **yes** | **NO** | n/a | Nothing blocks it technically. Blocked on: never having been executed, GPU budget approval, and a decision about whether the planner protocol makes the experiment meaningful | `scripts/run_e4.py` (323 lines, with resume support and a profile mode), `atlas/harness_e4.py` (16.6 KB), `modal/modal_e4.py` (179 lines), `scripts/smoke_e4.py` (221 lines). **`atlas_out/` contains no `e4` directory of any kind — zero episodes have ever been produced.** `CLAUDE.md` §0.1 claims `run_e4.py` raises `NotImplementedError`; that is stale, it no longer does. This is the experiment the paper is named for. |
| **E5** — cross-policy diagnostic (supplementary) | **no** | **no** | n/a | `scripts/run_e5.py:45` raises `NotImplementedError` | The underlying function `atlas/harness.py::build_cross_policy_matrix` reportedly exists, but the driver is a 56-line stub. Explicitly deprioritized by the project (`OPUS_REMAINING_TASKS.md` Section D). |

---

## Section 2 — Named sub-mechanisms of the method

The three rules that constitute ATLAS (proposal §1, §6).

| Mechanism | Implemented? | Run? | Results verified? | Blocking dependency | Notes |
|---|---|---|---|---|---|
| **SELECT** (argmin UMF over the library, hysteresis margin m=0.05) | **yes** | **yes**, but only offline | pending | none for the offline form; the in-the-loop form is blocked on E4 | `atlas/router.py` (9.6 KB), five routers: `umf`, `e1`, `sdyn`, `random`, `oracle_id`. Exercised for real only in E2, where **no planner is in the loop** — selection accuracy is scored against a regime label on pre-collected trajectories. Selection has never influenced a live planning decision in any completed experiment. |
| **REFINE** (one SGD step on the selected chart, strictly after scoring) | **yes** | **no** | n/a | Blocked on E4 | `atlas/adajepa.py` (5.9 KB), `atlas/loop.py::atlas_refine`. Gate G2 (prequential order) is claimed to pass on synthetic data. No production run has ever performed a refine step: E0 trains charts offline, E2 has no refinement, E1 never ran. |
| **EXPAND** (strike counter + fixability probe, commit only after verification on the next unseen chunk) | **yes** | **partially** | pending | Full RQ3 evaluation blocked on E4 | `atlas/expand.py`. Fired for real once, in E2's `q=1` diagnostic — 3 charts committed through `Expander.record()` → `library.clone_from()` → `_fit_candidate()` → `library.add()`. **Note the diagnostic used q=1, not the pre-registered q=3**; at q=3 nothing commits in any E2 cell, because three consecutive strikes at the observed 15.7% per-chunk rate happens ~0.4% of the time. |
| **atlas_step()** — the prequential controller that composes all three | **yes** | **NO** | n/a | Blocked on E4 | `atlas/loop.py` (9.9 KB). Per `CLAUDE.md` §0.1, no gate exercises `atlas_step()` itself — only `Expander` and `route()` are tested directly. **The full ATLAS method has never executed.** |
| **UMF scoring + informative-chunk gating (C3)** | **yes** | **yes** | pending | none | `atlas/score.py` (11.8 KB). Rewritten 2026-08-25 after a rollout bug that invalidated every prior number (5x wrong time base, zeroed proprio, wrong context window). Heavily exercised since. |
| **Chart / Library** (disjoint parameter sets, identity init, apply/restore) | **yes** | **yes** | pending | none | `atlas/chart.py` (12.7 KB), `atlas/library.py` (4.5 KB). Three kinds: `ln_act` (10,764 params), `lora4` (118,176 trainable), `full` (20,800,884). **Known landmine:** `chart.restore_()` does not restore pretrained weights for any kind except `lora4` — it re-applies the same chart (`HANDOFF.md` §4, citing `atlas/chart.py:107-127`), with 10 production call sites relying on the current behaviour. |
| **Regimes** (`PhysicsRegime` + visual corruption) | **yes** | **yes** | pending | none | `atlas/regimes.py` (12.7 KB). Two documented redefinitions: R1 mass→friction (mass is provably inert against a kinematic pusher), R2 elasticity→damping (the env hard-codes zero velocity damping, so restitution could not express itself). Current: **R1 = friction 2.0, R2 = damping 0.5**. |
| **Paired seeding** | **yes** | partially | pending | Full 360-episode design blocked on E4 | `atlas/streams.py` (4.4 KB), `paired_seed()`. Gate G5 claimed passing. The 360-paired-episodes-per-arm design (plan §8) that the statistics section depends on requires E4 and therefore does not exist. |

---

## Section 3 — The 7-arm ablation ladder ("the paper's central table", plan §7.4)

**Every arm below is implemented and has never produced a single episode.**
`scripts/run_e4.py --arms` accepts all seven; `atlas_out/` has no E4 output.

| # | Arm | Adapts | Persists | Library+routing | Expands | Verifies | Implemented? | Run? | Blocking dependency |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Frozen | | | | | | yes | no | E4 never launched |
| 2 | AdaJEPA | ✓ | | | | | yes | no | E4 never launched |
| 3 | Persistent-AdaJEPA *(ours)* | ✓ | ✓ | | | | yes | no | E4 never launched |
| 4 | ATLAS-fixed-library | ✓ | ✓ | ✓ | | | yes | no | E4 never launched; also needs E0 chart checkpoints as the fixed library |
| 5 | ATLAS-detect-only | ✓ | ✓ | ✓ | ✓ | | yes | no | E4 never launched |
| 6 | **ATLAS** | ✓ | ✓ | ✓ | ✓ | ✓ | yes | no | E4 never launched; `atlas_step()` has never executed |
| 7 | Oracle-ID | — | — | oracle | — | — | yes | no | E4 never launched |

The proposal's attribution argument (claim **L-1**) requires that adjacent
arms differ by exactly one mechanism. **This has never been verified against
the code** — it is the proposal-code-auditor agent's highest-value check,
because if arms 2 and 3 (or 4 and 5) differ by more than one thing, the
central table does not attribute anything.

---

## Section 4 — Gates (implementation plan §9)

| Gate | Checks | Implemented? | Run & passing? | Notes |
|---|---|---|---|---|
| **G1** identity | Identity chart leaves predictor output bit-identical; restore returns every tensor bit-identically | yes | claimed passing since 2026-08-26 | `HANDOFF.md` §7.2 states the *previous* G1 implementation "tested nothing it claimed" — it never applied the chart and never called the model. Any claim that G1 passed before 2026-08-26 was unfounded. `CLAUDE.md` §0.1 is stale here. |
| **G2** prequential | Over-refined chart must not auto-win on the next window | yes | claimed passing | On synthetic latents, not production data |
| **G3a** probe fires | Genuine new regime ⇒ chart commits | yes | claimed passing | Uses a perturbation of the predictor's own weights as the "new regime", not a real physics shift |
| **G3b** probe discriminates | Unfixable noise ⇒ nothing commits | yes | claimed passing | Deliberately the crudest possible unfixable case; does **not** test whether verification would reject a chart that predicts better but plans no better — which is precisely the failure mode E0 found |
| **G4** regimes real | Random-action rollouts differ visibly and statistically per regime | yes | **NOT RUN** — the only acknowledged skipped gate | Needs a live environment factory not yet wired into `smoke_gates.py::main()`. Regime reality is instead argued from the separate `REGIME_DESIGN_REVIEW.md` analysis. |
| **G5** pairing | Same seeds across arms ⇒ identical initial states and goals | yes | claimed passing | |
| **G6** denominator | Low-motion chunks return `None` | yes | claimed passing | |

---

## Section 5 — The headline gap

**No experiment in this project has ever tested continual learning.**

- Every planning number on disk (E0, all variants) comes from independent
  episodes in a **single fixed regime**, with the chart trained **offline
  beforehand**. There is no stream, no regime sequence, no revisit, no
  retention measurement, no recall measurement.
- E2 measures routing accuracy over **pre-collected trajectories with no
  planner**, scored against a regime label.
- The one experiment that would exercise adaptation, persistence, expansion,
  recall and forgetting — E4's S2 stream — has produced zero episodes.
- The prequential controller `atlas_step()` that constitutes the ATLAS
  method has never run.

The target venue is a workshop on **continual** world models. This is the
single most consequential fact in this file and it should be the first thing
any future session reads here.

See `CLAIMS_MATRIX.md` row **G-1**.
