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

The UMF half is done and trustworthy (`atlas_out/e0/results.json`, `E0_RESULTS.md`'s original
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

## What's validated vs. not, right now

| Claim | Status |
|---|---|
| UMF table (3 kinds × 3 regimes) | ✅ Valid, unaffected by anything above |
| `replans=6` achievable under corrected config | ✅ Confirmed empirically (Reading A, `num_act_stepped=1`) |
| Baseline vs. `ln_act`/R1 comparison, corrected config | ⚠️ n=1 only (episode 0). Earlier dramatic "baseline wins clean, adapter fails identically" result (from the *wrong* config) did **not** replicate — both failed this instance, `ln_act` somewhat worse on both metrics. Not enough data to conclude anything statistically. |
| CEM-cost ranking-distortion finding (`ρ≈0.089`, adapter's top pick ranked #110/300 by baseline) | ⚠️ Measured under the **wrong** CEM config (`diagnose_cem_costs.py` run before the fix). Mechanism may still be real but is not re-validated. |
| G3a/G3b gates | ✅ Passing, but explicitly scoped to mechanism-correctness only — do not cite as evidence ATLAS would catch a chart like the one above (see `scripts/smoke_gates.py::gate_g3a`/`gate_g3b` docstrings). |
| Richer-retrained `ln_act`/R1 chart (5 trajs × 30 steps vs. original 3×10) | ⚠️ UMF got *worse* (0.68→1.11), not better. Planning success of this richer chart has not been measured at all — only the original chart has been re-tested under the corrected config so far. Chart lives at `atlas_out/e0_richer/chart_ln_act_R1.pt`, kept separate from the original `atlas_out/e0/chart_ln_act_R1.pt` on purpose. |

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
- Real per-episode wall time under the corrected config (`num_samples=200, iterations=10, horizon=25,
  num_act_stepped=1`, Reading A / 30 raw steps): **~42 minutes** on an L4. Budget accordingly —
  this is not a quick check anymore the way the old (wrong) config's ~7 min/episode was.
- `data/pusht_noise/train/states.pth` (175MB) and `seq_lengths.pkl` must be on the Modal volume at
  the **volume-root-relative** path `/data/pusht_noise/train/...` (not `/atlas_root/data/...`) —
  same lesson as the hub-cache and chart uploads, see `code-review.md` Bug #8/#9.

## Immediate next steps, in order

1. **Resolve the `num_act_stepped`/episode-length ambiguity above** — this blocks trusting any
   further planning-success numbers, in either `run_e0_planning.py` or `run_e1.py`.
2. Once resolved, get real n>1 data for baseline vs. `ln_act`/R1 under the settled config (currently
   n=1 each) — check whether the CEM-ranking-distortion finding re-appears with real statistics.
3. Extend to `lora4`/`full` and R2 once the R1 baseline comparison is trusted.
4. Check the richer-retrained chart's actual planning success (not just its worse UMF) — it's sitting
   trained and ready at `atlas_out/e0_richer/chart_ln_act_R1.pt`.
5. Only after 2–4: decide whether E0's pre-registered rule produces a valid winner, or whether "no
   valid winner, RQ0 failed under this protocol" is the honest reportable result (per `CLAUDE.md`
   §1.8 — don't manufacture a winner to keep moving).
