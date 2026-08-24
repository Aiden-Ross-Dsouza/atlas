# E0 Handoff — Adapter Capacity Experiment

*Written 2026-08-24 for a fresh agent/session picking this up. Companion narrative writeup (same
content, human-readable): the "E0 Field Log" artifact published this session — ask the user for the
link if you need it; it isn't reproduced here. This file is the technical, actionable version.*

**Read `CLAUDE.md` §0.1 first** for the whole-project status. This file is E0-specific detail that
doesn't fit there.

## What E0 is

RQ0: can a lightweight adapter absorb a Push-T physics regime shift? Three adapter kinds (`ln_act`,
`lora4`, `full`) × two regimes (R1 = high friction, R2 = high elasticity), each scored two ways —
held-out prediction error (UMF) and real CEM planning success. Pre-registered rule (implementation
plan §7.1): the smallest kind reaching ≥90% of `full`'s gain on **both** metrics, in **both**
regimes, becomes the adapter kind used everywhere downstream (E1 onward).

## Current state, in one paragraph

**⚠️ 2026-08-25: the sentence below ("the UMF half is done and trustworthy") was WRONG — see
`CLAUDE.md` §0.1's CRITICAL entry and `E0_DIAGNOSIS_AND_PLAN.md`.** `atlas/score.py::_open_loop_rollout`
— shared by UMF scoring AND E0's own fine-tuning — did not unroll the checkpoint correctly (5×
wrong time base, zeroed proprio the checkpoint structurally requires, 1-frame instead of
`ctxt_window=2` context). Fixed and hardware-verified (`E0_IMPLEMENTATION_PLAN.md` T1–T5): real
30-raw-step trajectory now gives frozen-model identity-chart UMF = 0.227 (vs. pre-fix 24–52 online /
0.67–1.67 offline). **Every UMF number below and in `E0_RESULTS.md`'s original table is invalidated**
— E0's charts have not yet been retrained through the fix (T9, blocked on approval). The planner-config
paragraph below is still accurate as far as it goes, but has itself been further corrected since (T6
— see "RESOLVED (T6)" section below): the config now defaults to the SUBSTRATE's own validated spec
(`num_samples=300, horizon=6, num_act_stepped=6`), not the AdaJEPA-derived Sec7.0 numbers this
paragraph describes as "corrected."

Original paragraph, kept for the record: The UMF half is done and trustworthy (`atlas_out/e0/results.json`, `E0_RESULTS.md`'s original
table). The planning-success half went through five rounds of real bugs before producing anything
trustworthy — success metric, goal sampling, and (the big one) **the entire CEM planner
configuration was wrong**, pulled from jepa-wms's own shipped eval config instead of this project's
own §7.0/§7.6 planner budget. That's fixed now in `scripts/run_e0_planning.py`, but only n=1 has
been re-measured under the corrected config so far, and there's a real unresolved unit ambiguity
(see below) that affects whether the fix is even fully correct. **Do not trust any planning-success
number dated before 2026-08-24 afternoon** — check the config fields logged alongside it.

## Files

| File | Role |
|---|---|
| `scripts/run_e0.py` | Offline fine-tuning (UMF half). Stable, not touched this session except adding `--num-train-trajs` (see below). |
| `scripts/run_e0_planning.py` | Planning-success half. Rewritten multiple times this session — see bug log. |
| `scripts/diagnose_cem_costs.py` | One-off diagnostic: captures CEM's per-candidate costs to check whether a chart distorts the planner's ranking, not the routine eval path. |
| `modal/modal_e0_planning.py` | Modal wrapper for both `run_e0_planning.py` and `run_e0.py` (via `run_e0_train`), plus `diagnose_cem_costs`. Three named entrypoints — invoke as `modal run modal/modal_e0_planning.py::main`, `::run_e0_train`, or `::diagnose_cem_costs`, not the bare file (fails with "specify a function" once >1 entrypoint exists). |
| `E0_RESULTS.md` | Results log, chronological, most-recent-first. Read top-down; older sections are explicitly marked superseded. |
| `code-review.md` | Broader bug log for the whole repo, not just E0 (gitignored — ask the user for it if you don't have it). |

## The bug log (chronological, what to know before touching this again)

1. **Predictor-state contamination, missing adapter params, dead mass-regime, low contact rate** —
   all pre-existing before this session's later work, all fixed. See `code-review.md` Bugs #1/#3/#4/#6
   and `REGIME_DESIGN_REVIEW.md`/`ACTION_SAMPLING_REVIEW.md` if you need the detail. Not touched
   again this session; the UMF table stands.
2. **`run_e0_planning.py` Modal deployment bugs** — `ATLAS_HOME.resolve()` follows the volume mount
   into an inaccessible internal path (fixed: use the raw env var); `modal volume put`'s destination
   is volume-root-relative, not mount-relative (fixed: upload paths corrected, `code-review.md` Bug #8).
3. **Success metric floor effect (0% everywhere)** — two compounding bugs: the success check
   compared agent position too (meaningless for independently-random goals — fixed to block-only
   position/angle); the angle-wrap formula broke on multi-rotation differences (upstream goal
   generator uses an unbounded Gaussian — fixed with a proper fold-into-range formula). Root cause:
   independently-random goals were often 167–223px apart, unsolvable in one shot. Fixed by sampling
   real `(init, goal)` pairs from `data/pusht_noise/train/states.pth` (175MB, not the full 7GB
   dataset) instead of generating them independently. `code-review.md` Bug #9.
4. **The CEM config bug (the big one, my own mistake).** `run_e0_planning.py` used
   `num_samples=300, iterations=30, horizon=6, num_act_stepped=6` — copied from jepa-wms's own
   shipped eval YAML (`configs/evals/simu_env_planning/pt/dino-wm/pt_L2_cem_sourcedset_H6_nas6_*.yaml`)
   and mislabeled "the published spec" in this script's own comments, **never cross-checked against
   the implementation plan's own planner budget**. Found by auditing `scripts/run_e1.py` (written by
   a different agent/session in parallel) and noticing its config didn't match. Implementation plan
   §7.0 ("Calibrate the budget on day 2") and §7.6 ("Planner constant everywhere") both state, in
   these exact words: **CEM 200 samples × 10 opt steps, subplanner horizon 25, 5 executed actions per
   replan, ≤30 MPC steps** — for every arm, not just E1. Fixed: `run_e0_planning.py` defaults now
   `num_samples=200, iterations=10, horizon=25`.
5. **`steps_left` units bug**, exposed by fix #4. `agent.act(steps_left=...)` needs model-chunk
   units (matches `CEMPlanner.horizon`'s units), not raw environment steps — invisible under the old
   single-replan config (steps_left always far exceeded horizon regardless of units), would corrupt
   later replans under a real multi-replan config. Fixed by mirroring `atlas/harness.py::run_e1_episode`'s
   already-correct handling of this (same bug existed there originally too, already fixed by the
   other agent before I got to it).
6. **`num_act_stepped` — see the unresolved discrepancy below.** I set it to `1` based on direct
   empirical measurement in `run_e0_planning.py`'s episode-length convention. `run_e1.py` (written
   independently) uses `5`. Both are internally consistent within their own scripts but disagree
   with each other. **This is not resolved — read the next section before changing either script.**

## ✅ RESOLVED (2026-08-24, T6): what does "≤30 MPC steps" mean?

**Superseded by `E0_IMPLEMENTATION_PLAN.md` T6.** Both readings below turned out to be
downstream of the same mistake: neither `run_e0_planning.py` nor `run_e1.py` was using
plan §7.0's own AdaJEPA-derived budget correctly in the first place — §7.0's numbers
("CEM 200×10, horizon 25, 5 executed actions, ≤30 MPC steps") are **AdaJEPA's**
published hyperparameters, a **different substrate** (small ResNet + PLDM), not
`dino_wm_pusht`'s. Applied literally to this checkpoint, `horizon=25` means 125 raw
steps of lookahead for a task DINO-WM itself samples to be feasible within 25 — not a
faithful port, an unexamined transplant.

**Fix:** both scripts now default to `dino_wm_pusht`'s own validated config
(`vendor/jepa-wms/configs/evals/simu_env_planning/pt/dino-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml:200-205`):
`num_samples=300, iterations=30, num_elites=10, horizon=6, num_act_stepped=6,
var_scale=1.0` → **30 raw steps/episode, 1 replan**. This is a documented deviation
from plan §7.0, justified as substrate fidelity — see
`ATLAS_implementation_plan_v2.md` §7.0a. Verified: both scripts' `build_cfg`/
`build_planner_cfg` now resolve to byte-identical planner dicts.

**At `num_act_stepped=6`, one replan covers the whole 30-step episode — there is no
`1`-vs-`5` ambiguity left to resolve.** Reading A (below) and Reading B (below) are
both now moot: neither script uses `num_act_stepped=1` or `=5` any more.

**New open issue this creates for E1 specifically (not resolved):** E1's routing
design needs *multiple* replans per episode (`N_WARMUP_REPLANS` then routed ones) to
exercise routing at all, but nas=6/horizon=6 means one replan already covers a full
30-raw-step episode — leaving no room for E1's warmup-then-route structure within
`MAX_MPC_STEPS=30`. See the comment block above `CEM_NUM_SAMPLES` in
`scripts/run_e1.py` for the detail. This needs a real decision (e.g. scaling
`MAX_MPC_STEPS` to `N_desired_replans × num_act_stepped × frameskip` raw steps) before
the real 60×3 run — flagged, not silently resolved.

<details>
<summary>Original (superseded) discrepancy writeup, kept for the record</summary>

## ⚠️ Open, unresolved discrepancy: what does "≤30 MPC steps" mean?

Two independent, each internally-consistent readings of the same plan sentence ("CEM 200×10, horizon
25, **5 executed actions per replan, ≤30 MPC steps**") exist in the codebase right now, reaching
different conclusions:

**Reading A — `run_e0_planning.py` (this session, mine):** `max_steps=30` is literal raw
environment steps (one real episode = 30 raw `.step()` calls, matching how `max_steps`/`elapsed`
have always been counted in this script). To get "~6 replans per episode" out of 30 raw steps,
`num_act_stepped` must be **1** model-chunk (= `frameskip=5` raw actions) per replan — empirically
verified: `num_act_stepped=5` gives only ~2 replans/episode (25 raw actions/replan, 30÷25≈1.2), while
`num_act_stepped=1` gives exactly 6, confirmed by direct measurement in a local test run. Under this
reading, an E0 planning episode is **30 raw steps total**.

**Reading B — `run_e1.py` (a different agent, written independently, per `CLAUDE.md` §0.1's E1
entry):** `n_replans_target = max_mpc_steps // num_act_stepped = 30 // 5 = 6` — i.e. `max_mpc_steps`
and `num_act_stepped` are treated as the *same* units for this division (both effectively
model-chunk-equivalent), and each replan then executes `num_act_stepped(5) × frameskip(5) = 25` raw
actions. This also gives exactly 6 replans, but the total episode is **150 raw steps**, not 30.
`CLAUDE.md`'s own E1 status entry claims this was "empirically verified... on real hardware" via a
logged `raw_steps_per_replan=[25,25,25,25,25,25]`.

**Both readings independently produce "6 replans," matching the plan's prose, via different total
episode lengths and different `num_act_stepped` values.** This is a genuine ambiguity in what "MPC
step" means in the source document (one raw actuator command, vs. one replanning decision point) —
not something I can resolve by re-reading the plan harder, and I did not resolve it before writing
this handoff. **Do not silently pick one and proceed** — this determines whether E0's planning
episodes should be 30 or 150 raw steps long, which is a real, consequential difference (5×
compute/cost either way), and whether `run_e0_planning.py` and `run_e1.py` are even measuring
comparable things right now. Suggested next step: ask the user, or find an authoritative source
(AdaJEPA's own released code/paper, since the plan explicitly says these are "AdaJEPA's published
planning hyperparameters, which we adopt") for what "5 executed actions per replan" and "≤30 MPC
steps" concretely mean in raw-action terms.

</details>

## What's validated vs. not, right now

**⚠️ This table is itself now partly superseded by `E0_IMPLEMENTATION_PLAN.md` T1's rollout-bug fix
(2026-08-25) — see `CLAUDE.md` §0.1's CRITICAL entry before trusting any row below.** In particular
the "UMF table ... unaffected" row was WRONG: the UMF half shared the same broken
`_open_loop_rollout` as everything else and is invalidated too.

| Claim | Status |
|---|---|
| UMF table (3 kinds × 3 regimes) | ❌ **INVALIDATED** (not "unaffected" as this row previously claimed) — shared the same broken `_open_loop_rollout` as the planning half. Fixed in T1; charts themselves not yet retrained (T9, blocked on approval). |
| `replans=6` achievable under corrected config | Superseded — T6 restored the substrate's own config (`nas=6`, `horizon=6`), under which one replan covers the whole 30-step episode (see the "RESOLVED (T6)" section above), not 6. |
| Baseline vs. `ln_act`/R1 comparison, corrected config | ⚠️ n=1 only (episode 0), and measured under a planner config (Reading A) since superseded by T6 — needs re-measurement under the T6 config, and against a retrained (T9) chart. |
| CEM-cost ranking-distortion finding (`ρ≈0.089`, adapter's top pick ranked #110/300 by baseline) | ⚠️ Measured under the **wrong** CEM config *and* the broken rollout (`diagnose_cem_costs.py` run before either fix). Mechanism may still be real but is not re-validated against either correction. |
| G3a/G3b gates | ✅ Passing post-rollout-fix too (re-confirmed 2026-08-25), still explicitly scoped to mechanism-correctness only — do not cite as evidence ATLAS would catch a chart like the one above (see `scripts/smoke_gates.py::gate_g3a`/`gate_g3b` docstrings). |
| Richer-retrained `ln_act`/R1 chart (5 trajs × 30 steps vs. original 3×10) | ❌ **INVALIDATED** — trained through the same broken rollout as everything else. The "UMF got worse with more data" finding was itself likely an artifact of fitting a mis-specified target harder (see `E0_DIAGNOSIS_AND_PLAN.md`), not a real richer-data effect. Re-measure post-T9 if still relevant. |

## Operational notes (learned the hard way this session)

- **`modal run --detach` protects against network disconnects, not against the local launcher
  process being killed.** If the local `modal run --detach ...` process gets forcibly terminated
  (not just disconnected), the remote job can still eventually receive a cancellation — it doesn't
  show up immediately in `modal app list` (looks `ephemeral`/healthy), but manifests later as
  `[modal-client] Received a cancellation signal` mid-run. Lost two episodes this way. If a job needs
  to survive the session, prefer checking `modal app logs <app-id>` periodically over trusting
  `modal app list`'s state alone, and avoid killing/reaping the launcher process if you can help it.
- **`modal app logs <app-id>`** is the reliable way to reconnect to a detached job's live output
  after the original `modal run` process's stream disconnects — more robust than re-running `modal
  run` (which starts a new job) or trusting a stale local log file.
- Real per-episode wall time under the (now superseded, see "RESOLVED (T6)" above) `num_samples=200,
  iterations=10, horizon=25, num_act_stepped=1` config: **~42 minutes** on an L4. Budget accordingly
  — under the T6 config (`num_samples=300, horizon=6`) throughput is different again; see
  `scripts/profile_episode.py`'s output (T8, now implemented for real) for a current measurement
  rather than trusting this stale number.
- `data/pusht_noise/train/states.pth` (175MB) and `seq_lengths.pkl` must be on the Modal volume at
  the **volume-root-relative** path `/data/pusht_noise/train/...` (not `/atlas_root/data/...`) —
  same lesson as the hub-cache and chart uploads, see `code-review.md` Bug #8/#9.

## Immediate next steps, in order

**⚠️ Superseded by `E0_IMPLEMENTATION_PLAN.md` T1–T13 (2026-08-25) — the rollout bug found after this
list was written invalidates the UMF table this list's steps 2-5 assumed was solid ground. Read
`CLAUDE.md` §0.1's CRITICAL entry and `E0_IMPLEMENTATION_PLAN.md` before resuming any of this.**
Original list kept for the record:

1. ~~Resolve the `num_act_stepped`/episode-length ambiguity above~~ — done, see "RESOLVED (T6)" above.
2. Once resolved, get real n>1 data for baseline vs. `ln_act`/R1 under the settled config (currently
   n=1 each) — check whether the CEM-ranking-distortion finding re-appears with real statistics.
3. Extend to `lora4`/`full` and R2 once the R1 baseline comparison is trusted.
4. Check the richer-retrained chart's actual planning success (not just its worse UMF) — it's sitting
   trained and ready at `atlas_out/e0_richer/chart_ln_act_R1.pt`.
5. Only after 2–4: decide whether E0's pre-registered rule produces a valid winner, or whether "no
   valid winner, RQ0 failed under this protocol" is the honest reportable result (per `CLAUDE.md`
   §1.8 — don't manufacture a winner to keep moving).

**Current real next steps, in order (post T1–T8):**

1. 🛑 T9 (needs approval) — retrain E0's charts through the repaired pipeline, with more data
   (replay real demo trajectories under `PhysicsRegime`) and early stopping.
2. 🛑 T10 (needs approval) — the chart×regime planning matrix: E0's real Success column, E1's
   oracle/random denominators, and the C3 UMF-vs-success validation figure, all from one run. This is
   the project's decision point (RQ0) — run before E1.
3. Resolve E1's new episode-length-vs-warmup-replans issue (flagged in `scripts/run_e1.py`'s own
   comment block) before attempting T11.
4. 🛑 T11 (needs approval, only if T10 says the denominator exists) — E1's real 60×3 run.
