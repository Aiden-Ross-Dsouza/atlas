# E2 Results — Appearance vs Dynamics (routing accuracy)

*Ran 2026-08-26. Four runs, all on Modal L4, ~6 min each. Figure: `atlas_out/e2_R2/F2a.pdf`,
regenerable from logs alone via `scripts/make_e2_figure.py`.*

## 🔴 SUPERSEDED 2026-08-28 (Phase 1 Stage 2, FIX_SPEC.md A1/A2/A3) — the entire "headline" and "2026-08-26 update" sections below are stale

`atlas/router.py`'s hysteresis normaliser was inert at K=2 (`FIXLOG.md` A1:
`relative_gap = (max−min)/(max−min) = 1.0`, always `> m=0.05` — the router
was pure argmin, hysteresis never held) and `atlas/expand.py::maybe_expand()`
selected the incumbent on the verification chunk instead of the deficit
chunks (`FIXLOG.md` A2). Both fixed 2026-08-27; the entire E2 suite was
re-run under the fix, plus per-decision logging (`strikes`, `probe_outcome`,
`relative_gap`, `hysteresis_held`, `committed`, `library_size` — `FIXLOG.md`
A3) added to the JSONL. **Local, CPU/GPU-light re-run (no Modal needed — E2
runs no CEM planner), same protocol, `--corruption dark`:**

| Config | Cell | UMF acc. (old → new) | S-dyn acc. (old → new) | Charts committed |
|---|---|---:|---:|---:|
| `ln_act`×R1 | B | 0.642 → **0.481** | 0.543 → **0.481** | 0 → 0 |
| `lora4`×R1 | B | 0.642 → **0.494** | 0.494 → **0.481** | 0 → 0 |
| **`ln_act`×R2 (was "decisive")** | B | **0.833 → 0.419** | **0.570 → 0.419** | 0 → 0 |
| 3-chart confusion matrix | all | UMF 0.603, S-dyn 0.365 → **UMF 0.298, S-dyn 0.294** | | n/a |
| R2 Cell B, q=1 (positive control) | B | n/a | n/a | → **5 committed** (was measuring UMF acc only pre-fix) |
| R2 Cell C, q=1 (over-expansion) | C | 1.0 (unchanged) | n/a | 0 → 0 |

**The headline reverses, it does not merely shrink.** Post-fix, UMF and
S-dyn select the SAME chart on almost every decision at every config tested
— both dominated by "hold the incumbent unless the challenger beats it by
>=5% of the incumbent's own score", which under the new (correctly
scale-free) normaliser almost never fires except when UMF's own gap is
already large. The 3-chart confusion matrix's post-fix `sel=R0` column
dominates every row (see raw JSONL) exactly the way the *pre-fix* S-dyn
router did — UMF has converged onto S-dyn's old failure mode, not the other
way around. **"UMF-based selection discriminates a dynamics shift; S-dyn
does not" (this file's own headline, below) is no longer supported by any
number on disk.** Cell C (over-expansion, 0 commits everywhere) is the one
finding that survives unchanged.

Raw artifacts (all new directories, none overwriting the originals below):
`atlas_out/e2_R1_phase1stage2_2026-08-28/`, `atlas_out/e2_R1_lora4_phase1stage2_2026-08-28/`,
`atlas_out/e2_R2_phase1stage2_2026-08-28/`, `atlas_out/e2_R2_cellB_q1_phase1stage2_2026-08-28/`,
`atlas_out/e2_R2_cellC_q1_phase1stage2_2026-08-28/`, `atlas_out/e2_confusion_matrix_phase1stage2_2026-08-28/`
(each: `e2_episodes.jsonl` with the new A3 per-decision `record_type="expansion"`
records, `e2_summary.json`, `F2a.pdf`).

**Addendum, same day:** the three `*_posthysteresis` configs below (a
separate, EARLIER prior state from the 2026-08-26 sequential-hysteresis fix)
were also re-run under today's A1/A2/A3 fix, plus a genuinely new, isolated
q=3 Cell-B measurement closing `PAPER_FACT_CHECK` C2's "q=3 column is
inferred, not measured" gap:

| Config | Cell B UMF (old posthysteresis → new) | S-dyn (→ new) | Committed B |
|---|---:|---:|---:|
| `ln_act`×R1 posthysteresis | 0.6420 → **0.4815** | → **0.4815** | 0 → **0** |
| `lora4`×R1 posthysteresis | (n/a) → **0.4938** | → **0.4815** | 0 → **0** |
| `ln_act`×R2 posthysteresis | 0.8333 → **0.4194** | → **0.4194** | (n/a) → **2** |
| `ln_act`×R2 Cell B, q=3, isolated (new) | n/a | n/a | → **0** |

Routing accuracy is bit-identical between each `*_posthysteresis` re-run and
its plain-config counterpart above — the collapse is reproducible across
both the 2026-08-26 code path and its later restatement. **One finding not
reflected in the accuracy numbers:** `e2_R2_phase1stage2_2026-08-28` (plain) → 0
committed at Cell B, `e2_R2_posthysteresis_phase1stage2_2026-08-28` (identical
config, re-run minutes later) → 2 committed, `e2_R2_cellB_q3_phase1stage2_2026-08-28`
(isolated single-cell form) → 0 committed again — commit COUNT is not
deterministic run-to-run under nominally identical config (routing accuracy
is; commits are computed off a real gradient-descent probe fit, likely
CUDA/cuDNN nondeterminism), flagged in `FIXLOG.md` as discovered-not-fixed,
out of scope for this pass.

Additional artifacts: `atlas_out/e2_R1_posthysteresis_phase1stage2_2026-08-28/`,
`atlas_out/e2_R1_lora4_posthysteresis_phase1stage2_2026-08-28/`,
`atlas_out/e2_R2_posthysteresis_phase1stage2_2026-08-28/`,
`atlas_out/e2_R2_cellB_q3_phase1stage2_2026-08-28/`.

Everything below this banner (headline, 2026-08-26 update, per-condition
tables, "the mechanism" section) describes the PRE-A1/A2/A3 state and is
kept for the historical record — do not cite it as current.

---

## 🟢 UPDATE (2026-08-27): original 2×2 cells re-run under the hysteresis fix — Limitation #2 now FULLY closed with real numbers, not just superseded by the 3-chart matrix

The 2026-08-26 update above closed Limitation #2's *mechanism* (sequential
hysteresis fixed) via the 3-chart confusion matrix, but explicitly left the
*original* 2×2 cell numbers below (Cell B's "0.880 vs. 0.324" headline)
un-re-run under the fix — flagged then as "cheap to re-run if those exact
numbers are cited externally." Re-run locally (no GPU/CEM planner needed,
~$0.10 total per `run_e2.py`'s own cost estimate), same protocol as the
original three runs, `--corruption dark` (matching the original methodology,
not the script's own `colour_change` default):

| Config | Cell | UMF acc. (pre-fix → post-fix) | S-dyn acc. (pre-fix → post-fix) |
|---|---|---:|---:|
| `ln_act`×R1 | B | 0.512 → **0.642** | 0.381 → **0.543** |
| `lora4`×R1 | B | 0.619 → **0.642** | 0.494 → **0.494** |
| `ln_act`×R2 (decisive) | B | 0.880 → **0.833** | 0.324 → **0.570** |

**The headline direction survives — UMF still clearly beats S-dyn on the
decisive R2 Cell B — but the margin shrank substantially: +55.6pp → +26.3pp.**
S-dyn's accuracy rose the most (0.324→0.570): with hysteresis now correctly
carrying the router's own prior selection forward (rather than always
resetting to `current_idx=0`, which had been silently inflating "always
defaults to `c0`"-biased S-dyn's apparent correctness whenever `c0` happened
to be the right answer at that point in a sequence), S-dyn's naive
stickiness now sometimes coincides with the correct answer for the right
reason (post-switch persistence) rather than the wrong one (never having
switched at all). **`lora4`×R1's Cell B is now tied with `ln_act`×R1's UMF
accuracy (0.642 each)** — the R1/R2 separability story (§"The mechanism"
below) is otherwise unchanged: R1 chart pairs still route near chance,
R2 still routes decisively above it.

Cell C (over-expansion) still commits 0 charts in every one of the three
configs, confirming the fix didn't disturb that result. Full corruption
check confirms `dark`, not `colour_change`, was used this time (100% of
pixels changed, matching the original methodology — the very first attempt
at this re-run mistakenly used the script's own `colour_change` default and
was discarded before being recorded anywhere).

Artifacts: `atlas_out/e2_R1_posthysteresis/`, `atlas_out/e2_R1_lora4_posthysteresis/`,
`atlas_out/e2_R2_posthysteresis/` (each: `e2_episodes.jsonl`, `e2_summary.json`, `F2a.pdf`).

---

## 🟢 UPDATE (2026-08-26): 3-chart confusion matrix + sequential hysteresis — Limitations #2 and #4 below closed

Two of this file's own "read before citing" caveats are now addressed, both cheap (ran locally,
zero GPU cost — E2 needs no CEM planner, see the deviation note below).

**Sequential hysteresis (closes Limitations #2).** `route()` was being called with
`current_idx=0` on every decision — hysteresis therefore always favoured `c0`, inflating the
R0-condition accuracies and deflating the shifted-condition ones (documented, not yet fixed, in
the original Limitations list). Fixed in `scripts/run_e2.py`: each router's own previously-selected
chart is now carried forward as the next decision's `current_idx`, reset at the start of each
(cell, condition, seed) sequence. Verified locally on a smoke run before trusting it on real data:
episode 0 picks the wrong chart under an unfavourable initial `current_idx=0`, episode 1 correctly
reverts once the identity chart's advantage is clear — the mechanism behaves as a real deployment
would, not as a frozen always-c0-favoured comparison.

**3-chart confusion matrix (closes Limitations #4).** Library `{c0, chart_ln_act_R1, chart_ln_act_R2}`
(chance accuracy 1/3, not the 2×2 cells' 1/2), routed against uncorrupted trajectories from each of
R0/R1/R2 (dynamics-only, no appearance shift — this isolates the same C1 question the 2×2's Cell B
does, at one more chart). N=40 episodes × 3 seeds per true regime (120 decisions/regime), using the
R1 charts recovered this session from `atlas_out/e0_v6_R1/` (both confirmed to load and to have
trained under `{"friction": 2.0}`, the calibrated R1 regime) plus the existing `chart_ln_act_R2.pt`.
Sequential hysteresis (above) is already baked into this run — not a separate ablation.

| router | accuracy | vs. chance (0.333) |
|---|---:|---|
| **umf** | **0.603** | +27.0pp — real, ~2× discrimination |
| sdyn | 0.365 | +3.2pp — indistinguishable from chance |

Confusion matrices (rows=true regime, cols=selected chart), out of 83–104 ungated decisions per
row (some gated by the min-motion threshold, per G6):

```
umf              sel=R0  sel=R1  sel=R2       sdyn             sel=R0  sel=R1  sel=R2
true=R0             54      19      10        true=R0             61      20       2
true=R1             23      50      22        true=R1             62      26       7
true=R2              6      32      66        true=R2             65      23      16
```

**S-dyn's near-chance accuracy is not noise — the matrix shows exactly why.** It selects the
`sel=R0` column (61, 62, 65) at nearly the same rate regardless of the true regime — the identical
"always defaults to `c0`" bias the 2×2 Cell B result already found, now independently reproduced
at 3 charts instead of 2. UMF's diagonal is clearly dominant in every row (65%, 53%, 63%) despite
the harder 1-in-3 task. **Expanding the library from 2 charts to 3 did not erode UMF's advantage,
and it did not save S-dyn** — this is the direct answer to "only tested with 2 charts, chance=0.5."

Artifacts: `atlas_out/e2_confusion_matrix/e2_confusion_episodes.jsonl` (per-decision) +
`e2_confusion_summary.json` (matrices above, machine-readable). Charts consolidated at
`atlas_out/e2_charts/chart_ln_act_{R1,R2}.pt` for this run (both source charts remain the
canonical copies in their own directories — this is a convenience copy, not a new training run).

---

## Headline

**UMF-based selection discriminates dynamics specialists; S-dyn (a one-step latent-direction baseline, `atlas/router.py::_sdyn_score` — not an appearance-similarity router) does not.**
Routing accuracy is governed by a single quantity — whether the chart's UMF advantage clears the
pre-registered hysteresis margin (`m = 0.05`, CLAUDE.md §1.7).

This is the project's positive result, and it is **orthogonal to E0's negative one**: E2 validates
the *selector*, not the *library*. The charts it routes between still do not improve planning
success (`E0_RESULTS.md`). Correct routing here routes to a chart that does not help.

## Design and what it deviates from

| | |
|---|---|
| Cells (plan §6.3) | A same/same (R0 vs R0), B same/**DYN**, C **APP**/same, D **APP**/**DYN** |
| Library | `{c0 (identity), chart_{kind}_{regime}}` — 2 charts, so chance = 0.50 |
| Routers | `umf`, `sdyn` |
| Scale | 40 episodes × 3 seeds × 2 conditions per cell (~80 ungated decisions/condition) |
| Corruption | `dark` @ 0.5 |

**Deviation 1 — no CEM planner.** Episodes are collected trajectories, not planned episodes.
Routing accuracy is a property of UMF scoring on an observed chunk; the planner does not enter
the metric. This is what made E2 affordable (~$0.08/run vs plan §7.3's ~6 GPU-h estimate).

**Deviation 2 — `dark`, not `colour_change`.** Plan §6.3 names colour. Measured on this env,
`colour_change` alters only **5.6% of pixels** (Push-T renders are ~97% white, mean 248; an HSV
hue rotation is a no-op on desaturated pixels), which would let Cell C pass vacuously. `dark`
changes **100%** of pixels (mean |diff| 99.6) — the conservative direction, making Cell C harder
to pass. `run_e2.py` measures and records the magnitude on every run.

## §7.3's two carrying numbers

### Cell B (decisive) — PASSES

Condition B of Cell B is the only condition where the correct answer is the *chart* rather than
`c0`, so it is the discriminating measurement. Cell-level accuracy pools it with an R0 condition
and dilutes it; the per-condition numbers:

| run | shift | UMF gap under shift | `umf` acc | `sdyn` acc |
|---|---|---:|---:|---:|
| `ln_act` × R1 | friction 2.0 | +2.4% | 0.512 | 0.381 |
| `lora4` × R1 | friction 2.0 | +2.6% | 0.619 | 0.494 |
| **`ln_act` × R2** | **damping 0.5** | **+14.1%** | **0.880** | **0.324** |

**UMF 0.880 vs S-dyn 0.324 = +55.6pp.** §7.3's criterion ("UMF routing accuracy ≫ S-dyn's") is
met by a wide margin on R2.

### Cell C (over-expansion) — 0 committed, with a positive control

| condition | chunks with UMF > τ=0.5 | commits @ q=3 | commits @ q=1 |
|---|---:|---:|---:|
| R0, uncorrupted | 0.000–0.013 | 0 | — |
| **R0 + dark (Cell C)** | **0.000** | **0** | **0** |
| **R2 (Cell B)** | **0.157** | 0 | **3** |

A 100%-of-pixels appearance shift raises UMF above τ in **zero** chunks and commits nothing,
while a genuine dynamics shift exceeds τ in 15.7% and commits 3 charts at `q=1`. The expansion
verifier demonstrably *can* fire and correctly *does not* on appearance alone.

**The `q=1` runs are diagnostics, not results** — flagged `probe_params_are_preregistered:
false` in their own summaries. They exist because at the pre-registered `q=3`, three *consecutive*
strikes at a 15.7% per-chunk rate occur ~0.4% of the time, so nothing commits in **any** cell,
including B. Without the control, "Cell C committed 0" would have been indistinguishable from
"the Expander cannot fire," which is precisely the vacuous-verification failure mode
`smoke_gates.py::gate_g3b`'s docstring warns about. The reportable numbers stay at `q=3, τ=0.5`.

## The mechanism — why R1 failed and R2 worked

Same router, same code, same threshold; only chart separability changed.

| condition | mean UMF `c0` | mean UMF chart | gap | vs `m=0.05` |
|---|---:|---:|---:|---|
| R0 (no shift) | 0.123–0.141 | 0.157–0.190 | **+0.035…+0.060 favouring `c0`** | correct to keep `c0` |
| R1 | 0.2638 | 0.2574 | 0.0064 (**2.4%**) | **below margin — switch impossible** |
| R2 | 0.4188 | 0.3596 | 0.0592 (**14.1%**) | **above margin — switch fires** |

Under R1 a switch is *mathematically* excluded: hysteresis keeps the current chart unless a
competitor beats it by 5%, and no R1 chart of either kind gets close. Accuracy therefore sits at
chance regardless of adapter kind — `lora4` (2.6%, 0.619) behaves like `ln_act` (2.4%, 0.512).
Under R2 the gap triples past the margin and accuracy jumps to 0.880. This is the right panel of
F2a.

## S-dyn is the clean contrast

S-dyn's scores differ between the two charts by ~0.1% and it selects `c0` **68–79% of the time
regardless of regime**. It scores 0.92–0.98 on the same-dynamics cells purely because it always
picks `c0` and `c0` is always correct there — and collapses to 0.21–0.32 the moment the correct
answer changes. It has essentially no dynamics discrimination. That dissociation — UMF tracks
dynamics, S-dyn tracks nothing regime-dependent — is what Cell B was designed to expose.

## Limitations — read before citing any number here

1. **False positives on unchanged dynamics.** UMF keeps `c0` only 70–77% of the time under R0,
   i.e. a **23–30% false-switch rate**, despite the mean gap correctly favouring `c0`. This is
   per-chunk variance, not bias. A quarter-of-the-time wrong switch is a real weakness.
2. ~~**`current_idx=0` on every decision.**~~ **FIXED (2026-08-26), and the original 2×2 cells
   themselves re-run under the fix (2026-08-27) — see the top-of-file UPDATE section.** `route()`
   now carries each router's own previously-selected chart forward. The 2×2 numbers below are
   the PRE-FIX ones, kept for the historical record — cite the 2026-08-27 UPDATE section's numbers
   instead (Cell B: `ln_act`×R2 UMF 0.833 vs S-dyn 0.570, down from the pre-fix 0.880 vs 0.324 —
   the direction survives, the margin does not).
3. **27–30% of decisions were gated out** by the min-motion gate (`motion_gate` 312.34, 10th pct
   of R0 training displacement). Expected behaviour per G6, but each condition is ~80 decisions,
   not 120.
4. ~~**One chart per library.**~~ **ADDRESSED (2026-08-26), see the UPDATE section at the top of
   this file.** A 3-chart `{c0, chart_R1, chart_R2}` library (chance=1/3) confirms UMF's advantage
   survives the larger library and S-dyn's near-chance failure reproduces independently.
   `K_max` pressure (library eviction under the cap) is still untested.
5. **E2 does not rescue ATLAS end-to-end.** See headline.

## Artifacts

| run | dir | config |
|---|---|---|
| R1, `ln_act` | `atlas_out/e2_R1/` (= `atlas_out/e2/`) | q=3, pre-registered |
| R1, `lora4` | `atlas_out/e2_R1_lora4/` | q=3, pre-registered |
| **R2, `ln_act`** | `atlas_out/e2_R2/` | q=3, pre-registered — **primary** |
| R2, Cell C | `atlas_out/e2_R2_cellC_q1/` | q=1, **diagnostic** |
| R2, Cell B | `atlas_out/e2_R2_cellB_q1/` | q=1, **diagnostic** (positive control) |

Each holds `e2_episodes.jsonl` (per-decision: cell, condition, seed, episode, regime, corruption,
router, selected, correct, hit, gated, per-chart scores) and `e2_summary.json`.
`scripts/make_e2_figure.py` regenerates F2a from these alone.

## Code fixed to make this runnable

`atlas/regimes.py::VisualCorruption` had never been exercised against the real env and was broken
three ways: `observation()` called `obs.ndim` on PushTEnv's **dict** obs; `gym.ObservationWrapper.
reset()` passed PushTEnv's `(obs, state)` **tuple** into `observation()`; and `salt_pepper` used
an unseeded RNG, which would have broken G5's paired-seed guarantee.
