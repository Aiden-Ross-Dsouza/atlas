# FINAL FIVE-DAY PLAN — Aug 31 → Sept 5 AoE

---

# ⭐ REVISION 2 — Sept 1 → Sept 5 AoE (OPERATIVE PLAN)

**Written 2026-09-01, after independent verification of every Day-1 number from raw JSONL.**
Everything below the "═══ REVISION 1 ═══" separator is the original plan, **retained unchanged**
for provenance. Where the two disagree, **this section wins.**

**Why revised:** (a) Day 1 landed and was independently verified — two corrections are required
before anything is drafted; (b) the freeze moves from EOD Sept 1 → **EOD Sept 2**, because Modal
runs remotely while drafting proceeds locally; (c) three new experiments are added whose purpose
is to convert two *anecdotal* metric inversions into a *systematic, mechanistically modelled*
property. The paper claims no new architecture, so every claim must be carried by numbers,
statistics, and a mechanism derived and then checked in code.

---

## R2.0 — VERIFICATION VERDICT ON DAY 1 (main session, 2026-09-01, from raw JSONL)

**Passed — reproduces exactly:** protocol/config guards on all 20 cells (damping, it=10, N=300,
H=6, nas, settle-40, episode ranges); `n50_*` genuinely tasks 20–49 and disjoint; 0 pairing
mismatches everywhere; **1.C ladder at nas=6 every number exact**; **1.D all six rows exact**
(merged n=50 nas=2 Δ−40.2 [−58.1,−21.6] p=0.0001; nas=6 Δ−30.5 [−50.4,−10.3] p=0.0027);
**1.B nas=2 all-20 Δ+16.9 p=0.0441** and nas=6 Δ−3.2 p=0.84; **1.G.1 and 1.G.5 exact**; incident
containment clean (no `pandereshubham` partials leaked — every cell is exactly 20/30 episodes).

**Additionally verified (not claimed by the Day-1 log):** the ladder is a **clean single-axis
sweep**. `atlas/regimes.py::set_regime_config` does `REGIME_CONFIGS[name] = dict(cfg)` — full
replacement, not merge — so `--regime R2 --regime-config '{"damping":X}'` carries nothing else;
and `R0: {}` with the env default `space.damping = 0` (`pusht_env.py:687`) puts the damping-0
point genuinely on the same axis. **No hidden confound between the ladder endpoints.**

### 🛑 R2.0-a — CORRECTION 1 (material). `B2-transfer-01`'s clean-subset statistic is wrong.

`FINAL_FIVE_DAY_DAY1_RESULTS.md` §1.B and `EVIDENCE_LEDGER` row `B2-transfer-01` both state
*"neither-succeeded (n=16) Δ+20.1, p=0.025."* Recomputed both candidate definitions:

| subset definition | n | Δ | p |
|---|---:|---:|---:|
| neither **SETTLED**-succeeded ← what was used | 16 | +20.1 | 0.025 |
| neither **PASS-THROUGH**-succeeded ← §1.G.2's definition | **8** | **+24.4** | **0.109 n.s.** |

**The pass-through definition is the correct one, and this is not a matter of taste.** The
subset exists to remove the *unequal-compute* confound, and that confound is created by
pass-through success, because that is what `break`s the episode loop
(`run_e0_planning.py:329-333`). Settled success is computed post-hoc and terminates nothing, so
conditioning on it controls for nothing. Sanity check: at damping 0.5 the "neither settled"
subset is n=20 — the entire sample — i.e. not a subset at all.

**Required edit:** in both files, replace the clean-subset figures with **n=8, Δ+24.4, p=0.109
(n.s.)**, and state that the 1.B claim rests on the **all-20 test (p=0.044)** alone. The
*direction* is unaffected and consistent at both cadences; the second line of statistical
support is not. Log as a dated correction, do not silently overwrite (standing rule 5).

### 🛑 R2.0-b — CORRECTION 2 (minor, but must land before Figure 1 ships).

"H1 MONOTONE DIVERGENCE CONFIRMED" is true at **nas=6 only**:

```
nas=6 divergence:  0 → 30 → 50 → 50 → 55 → 55 pp     ✓ monotone
nas=2 divergence:  0 → 25 → 45 → 35 → 45 → 45 pp     ✗ dips at damping 0.2
nas=2 settled SR: .65 → .35 → .10 → .15 → .05 → .00  ✗ 2 eps → 3 eps
```

Nothing was hidden — the nas=2 numbers are printed in the Day-1 log — but the verdict header is
unqualified and the ledger's "nas=2 same shape" reports endpoints only. It is a one-episode
wobble at n=20, i.e. plainly sampling noise, **but a reviewer will read it off Figure 1.**

**Required edit:** state it as *"monotone at nas=6; monotone to within ±1 episode at nas=2."*
Plot both cadences in Figure 1 with the wobble visible rather than smoothing it.

---

## R2.1 — THE GOVERNING RULE FOR EVERYTHING ADDED BELOW

> **Every launch gets a pre-registered decision rule in `IMPLEMENTATION_PLAN_V3.md` §8 BEFORE it
> goes out, and every completed run is reported whatever it returns.**


**This costs nothing, and Day 1 already proved it:** 1.B returned a null (the adapter does not
transfer to damping 0.1), it became an honest scope condition in §4, §3 was untouched, and the
paper is stronger for it. **Every experiment below is designed so that both outcomes are
publishable.** If that property is ever unclear for a proposed run, then tell me why do you think so.

---

## R2.2 — WHAT TO RUN (ranked by value, all decision rules pre-registered here)

Standing config for every planning cell: `--iterations 10 --num-samples 300 --horizon 6
--settle-steps 40`, R2 `--regime-config '{"damping": 0.5}'` unless stated,
`--charts-root phase0_v3 --charts-subdir p0g_onpolicy --out-root phase0_v3`.

⚠ **`--episodes` is an EXCLUSIVE END INDEX, not a count** — verified at source
(`run_e0_planning.py:621`, `new_eps = range(episode_start, episodes)`). `--episode-start 50
--episodes 100` yields 50 episodes, tasks 50–99.

⚠ **Pin the Modal profile on every command** — `MODAL_PROFILE=aiden-dsouza-201323` as an env var,
never the global `~/.modal.toml` (a concurrent session flipped it mid-batch on Day 1 and leaked
8 apps to the wrong account).

### 🥇 A — Controller-family rank inversion (~$5, no code change) — HIGHEST VALUE

> **✅ DONE 2026-09-01 — then REWRITTEN the same day after external verification. Read the
> ledger row `N16-controller-family`, not this block, for what the result is.**
> `scripts/day2_controller_family.py`, fig `phase0_v3/day2_fig_iteration_ladder.png` (primary)
> + `day2_fig_controller_family.png` (appendix A1).
>
> **What the run actually established — PRIMARY: the CEM-iteration ladder** (frozen `c₀` only,
> nas=2, same 20 paired tasks, one varying knob): pass-through SR **0.35 / 0.40 / 0.45 / 0.40**
> (flat) while mean settled distance **36.1 / 100.2 / 137.3 / 158.5 px** (monotone, 4.4×) and
> block-moved-toward-goal **18/20 → 10 → 6 → 3**. Paired it1 vs it30 Δ**−122.4 px**, 19/20,
> Wilcoxon **p = 1.9e-6 one-sided** (3.8e-6 two-sided). No adapter, no group confound, no
> compute asymmetry. **This is §4's lead.**
>
> **SECONDARY — the cross-metric Kendall τ:** τ_b = **−0.512** (not −0.48; tie-corrected),
> perm p = 0.025, null band [−0.45, +0.45], 48/66 pairs inverted.
>
> **⚠️ The pre-registered branch fired, but its stated interpretation did not survive analysis.**
> The 12 controllers form **two disjoint 6-blocks on pass-through SR** (all frozen 0.35–0.55 >
> all chart 0.00–0.30), so the overall τ largely re-expresses `N15` rather than measuring an
> independent family. The genuine within-family signal is τ_b **−0.36 (frozen, p=0.33) / −0.41
> (chart, p=0.25)** — same direction, n=6, underpowered. **And on settled distance the two
> families interleave** — `it1 baseline` (36 px) beats every chart controller and `it3 baseline`
> (100 px) beats `nas6 chart` (111 px) — so the earlier claim here that frozen and chart orders
> are simple reverses of each other was **wrong** and is retracted. Chart range is **57–111 px**,
> not 57–78 (`nas6 chart` was dropped from the original range).
>
> `it30 baseline` worst on settled distance (159 px) — more CEM → harder shoves — stands.

**The problem it solves.** The paper currently has *two* metric inversions (adapter comparison,
cadence comparison), which reads as "we found two cases." This makes it a measured property of
the metric over a **family** of controllers.

**Why it is now cheap:** verified this session — `c2_dose_it{1,3}_*`, `c2_alpha0_*`,
`c2_nas6_*`, `c2_p0g_R2` all have **0 settle records** (they predate the flag). Re-running them
with `--settle-steps 40` costs ~$0.5/cell and needs **no code change** — `--iterations`,
`--num-act-stepped` and `--objective-alpha` are all live flags.

**Cells (8 new; it=10/nas=2 and nas=6 already on disk → 12 controllers total):**
`--iterations ∈ {1, 3, 30}` × `--kind ∈ {baseline, ln_act}` at nas=2, plus
`--objective-alpha 0` × both kinds at nas=2.

**Analysis:** rank all 12 controllers by (i) pass-through SR and (ii) mean settled distance;
report **Kendall τ between the two orderings**, with a permutation CI.

**Pre-registered decision rule — both branches publishable:**
- τ near 0 or negative → *"the two criteria induce systematically different orderings over a
  controller family"* — §4's claim upgrades from anecdote to measurement.
- τ near 1 → *"over a controller family the criteria largely agree; the inversions are confined
  to pairs that differ in outcome variance"* — a **sharper and more honest** finding that makes
  H7 (variance compression) the precise scope condition rather than a side note.

**Artifacts:** `phase0_v3/fam_it{1,3,30}_{baseline,ln_act}_nas2/`,
`phase0_v3/fam_alpha0_{baseline,ln_act}_nas2/`

### 🥇 B — n=100 on the headline (~$3, no code change)

> **✅ DONE 2026-09-01.** `scripts/day2_n100_analysis.py`, ledger `N15-n100`, results
> `FINAL_FIVE_DAY_DAY1_RESULTS.md` §B. **REPLICATES — all three disjoint task sets chart-closer.**
> 50–99 Δ−53.7 px p=0.0001; **merged paired n=100 Δ−47.0 [−61,−33] p<0.0001, 71/100.** Clean
> subset sig. in 2/3 sets + merged (p=0.0022). Pass-through inversion intact (McNemar p=0.0012).
> "n=20" is dead.

`--episode-start 50 --episodes 100` (⇒ 50 episodes, tasks 50–99), nas=2, both kinds, settle-40.
Takes the paper's main comparison to **n=100 paired** across three disjoint task sets.

**Pre-registered:** report all three task sets **separately and merged**, whatever they show. If
50–99 fails to replicate, §4 becomes *"significant in two of three task sets"* and the task-set
variance becomes a stated limitation. §3 is frozen-baseline-only and unaffected either way.

**Artifacts:** `phase0_v3/n100_{baseline,ln_act}_nas2_ep50-99/`

### 🥇 C — Analytical mechanism model ($0, local, do today) — WHAT MAKES IT A RIGOROUS PAPER

> **✅ DONE 2026-09-01.** `scripts/day2_coast_model.py`, ledger `N14-coast-model`, results
> `FINAL_FIVE_DAY_DAY1_RESULTS.md` §C, fig `phase0_v3/day2_fig_coast.png`.
> **Pre-registered outcome = agreement.** C-1: per-episode Pearson r 0.91–0.99 (median 0.97),
> pred/meas ratio 1.00 from the clean free-coast slope. C-2: the closed form predicts the
> settled-SR ladder to ±1 episode at all 10 points, both cadences, no free parameters. §5/H6
> is now a derivation.

pymunk integrates `v ← v · damping^Δt`, so the asymptotic coast has a **closed form**:

```
D  =  v₀ / ln(1/damping)
```

Two free, decisive checks against data already on disk:

1. **Validate the model per-episode.** Estimate entry velocity `v₀` from the early slope of
   `settled_trace` (steps 1→5) and total coast from steps 1→40; test `D ≈ v₀/ln(1/damping)`
   across all six damping values and both cadences.
2. **Predict the ladder.** A crossing survives the hold only if `D ≲ goal radius`. That yields a
   **predicted** survival-vs-damping curve to overlay on the measured 100% / 24% / 0%.

**Pre-registered:** report the fit whatever it is. Agreement → §5 stops being a table and becomes
*"here is why, derived from the simulator's integrator and confirmed at six damping values."*
Disagreement → the residual is itself informative (it would mean contact during the hold, or
rotational energy, matters) and is reported as such.

**This is the single highest value-per-dollar item in the plan**, and it is the direct answer to
"we are not proposing an architecture, so everything must be backed by logic correctly
transferred into code."

### 🥈 D — `lora4` second adapter class (~$2, no production-code change)

Train `lora4` via `modal_phase0.py::p0g_finetune` on the existing cached `trajs_R2.pt` at
**matched budget** to the `ln_act` recipe (same trajectories, steps, early stopping — this kills
the OPUS #12 training-set-size confound for the pair), then C-2 screen at nas=2 with settle-40.

**Pre-registered:** report the per-kind outcome-distribution table {ΔUMF, mean, sd, settled dist,
SR, bands} regardless of direction. Compression in 2/2 kinds → *"both adapter classes tested"*
(**not** "capacity-independent" — `full` is not run, say so). Compression in only one → a
capacity-dependence finding. **OOM is a genuine technical non-run** and is reported as absent.
Batch size down, never fewer trajectories.

### 🥈 E — Momentum negative control (~$2 + ~2h, additive code change) — Day 2

> **⛔ ABANDONED 2026-09-01 per this item's own 90-min gate.** The `--block-cog` flag was added
> (additive, falsification passed: BEFORE exit 2, AFTER byte-identical), but **the physics knob
> is inert** — `pymunk.Body.center_of_gravity` set at runtime does nothing without also
> recomputing `Body.moment` (verified: identical trajectories for cog ∈ {None,(0,110),(40,45)};
> `shape=` likewise inert). Same failure class as the R1 mass bug. Making it real needs
> `moment_for_poly` surgery + a G4 check — beyond the budgeted scope. Flag reverted
> (`FIXLOG V3-23`). **Item C (`N14-coast-model`) already establishes the momentum mechanism
> more strongly** — it derives the collapse from `space.damping` and predicts it to ±1 episode.

**A falsification test of the paper's own mechanism.** Verified this session:
`PushTEnv.__init__` exposes `block_cog` and `shape ∈ {T,I,L,Z,square,small_tee}`. Add an
**additive** `--block-cog` flag threading to the constructor at `run_e0_planning.py:505`. This
touches **env construction, not the planning loop**, so the protected-file rule is satisfied —
FALSIFICATION line first (BEFORE: flag unrecognised, exit 2; AFTER: default-flag 3-episode run
field-identical to an archived record).

Then run a **dynamics shift with no momentum**: `block_cog` offset at `damping = 0`. The world
model's predictions break, but the block still stops dead.

**Pre-registered prediction: divergence (pass-through SR − settled SR) ≈ 0.**
- Holds → the criterion fails **specifically under residual momentum**, not under dynamics shift
  in general. That isolates the mechanism and is a genuine negative control — the strongest form
  of evidence an evaluation paper can carry.
- Fails → a second, momentum-independent failure mode exists, and the paper gets bigger.

**Gate:** if the flag's falsification smoke is not clean within 90 minutes, **abandon E** and
spend the time writing. It is upside, not a dependency.

### ⛔ NOT RUN (recorded as decisions)

- **1.A** damping-0.1 native chart (~$15) — 1.B answered the question well enough; the
  severity-specific vs off-distribution distinction becomes one stated-open sentence in §8.
- **E4 / E1 / E2 / E5** — unchanged, see REVISION 1 and FIXLOG "E4 DEFERRAL".
- **A second environment** — a month of work. It is the 2027 archival paper.

---

## R2.3 — REVISED SCHEDULE

| day | work |
|---|---|
| **Sept 1 (today)** | Apply corrections R2.0-a and R2.0-b. Pre-register **A, B, D** in §8. Launch **A, B, D** detached with `MODAL_PROFILE` pinned. Do **C** locally while they run. |
| **Sept 2** | Analyse A/B/D; build the Kendall-τ table and the theory-vs-measurement figure. Land **E**'s flag + smoke, launch it. Run 2.3 screen-power table. **Begin writing §3 and §5.** 🧊 **EXPERIMENTS FREEZE EOD.** |
| **Sept 3** | Analyse E. **Full draft complete by EOD — this is the hard gate, not the experiments.** |
| **Sept 4** | Adversarial pass (unchanged from REVISION 1 §Day 4): every number against raw JSONL, every citation against its abstract, regenerate every figure from logs, red-team the six surfaces, CFP compliance, AI-agent disclosure. |
| **Sept 5** | Final read, submit with hours to spare. |

**The freeze moved but did not disappear.** Modal runs while you draft — but *analysis* and
*integration into the paper* compete with writing for the only genuinely scarce resource, which
is your attention. Anything not downloaded **and ledgered** by EOD Sept 2 does not go in.

**Day 4 remains untouchable.** This project has shipped a fabricated citation and a floor-effect
statistical error, each caught only because a review pass had room to run — and this session
caught a third (R2.0-a) that would otherwise have reached the draft as an L5 row.

---

## R2.4 — RED-TEAM SURFACES, UPDATED

| # | attack | answer after REVISION 2 |
|---|---|---|
| 1 | "Your regime is degenerate." | 6-point ladder, verified single-axis; §3 is frozen-baseline only |
| 2 | "n=20." | n=100 paired across three disjoint task sets (**B**) |
| 3 | "Your metric is arbitrary." | R0 control (32/32 survive) + settle-length sensitivity (a 5-step hold gives the identical verdict) |
| 4 | "One toy environment." | Stated in §1 as scope. **Partly answered by C** — the mechanism is derived from the integrator, not from Push-T |
| 5 | "This is just variance." | sd-ratio CI excludes 1 in every pool + residual-momentum table + **C's closed-form model** |
| 6 | "Unequal compute between arms." | §1.G.2 matched-replan subset + §1.G.3 disclosure + §1.G.5 reframing |
| 7 | **"You found two cases and called it a property."** | **A's Kendall τ over 12 controllers** |
| 8 | **"Does the metric break under any shift, or only yours?"** | **E's momentum negative control** |

---

## R2.5 — HONEST EXPECTATION, UPDATED

**Acceptance estimate: 65–75%** (was 50–65%). The movement comes from A (anecdote → measured
property), C (empirical → mechanistic, derived and then checked in code), E (a falsification test
of our own mechanism), and B (n=100).

**I do not think 80% is purchasable in five days, and the plan should not be built as if it
were.** The binding constraint is **one environment**, and that is a month of work, not five
days. C partially mitigates it — a mechanism derived from the physics integrator rather than from
Push-T generalises on the page in a way an extra Push-T cell never can — but it does not replace
a second environment.

*(Prior estimates recorded rather than averaged: an earlier session said 65–75% before Day 1;
this session said 50–65% pre-Day-1 and 65–75% after. Neither of us knows this workshop's bar.)*

---

---

## R2.6 — §7's CONSTRUCTIVE CONTRIBUTION, CORRECTED (2026-09-01)

**Supersedes the first `N13-screen-power` table.** The original computed *power* — how many
paired episodes are needed to detect a difference — using one-sided exact McNemar on
**pass-through success**, against the C-2 adapter labelled "known-bad."

**Its arithmetic was exact** (the C-2 row re-derived in closed form from the hypergeometric:
0.81 / 50.00 / 99.19 / 100.00% — matches to 2 dp). **Its construct was wrong:** §3 invalidates
threshold-crossing success, so §7 cannot recommend it as the acceptance gate; and on the paper's
own settle-validated metric the C-2 adapter is **better** (`N12-n50`: −40.2 px, n=50, p=0.0001),
so that column was a **false-alarm rate**, not a detection rate.

**The right question is discrimination, not power:** *does the screen flag the adapters that are
actually harmful and leave the others alone?* Two screen statistics × four cases whose ground
truth is fixed on the settle-validated metric. One-sided ("chart worse"), α=0.05, 4 000
subsamples, pairing asserted. `scripts/day2_screen_power.py` → `phase0_v3/day2_screen_power.json`.

| case (ground truth on settled distance) | pass-through McNemar<br>n=15 / n=20 | settled-dist Wilcoxon<br>n=15 / n=20 |
|---|---|---|
| **HELPFUL** ln_act@0.5 nas=2 — Δ**−59.8** px, must NOT flag | 79.9% / **100% ❌ false alarm** | 0.0% / **0.0% ✅** |
| **HELPFUL** ln_act@0.5 nas=6 — Δ−27.8 px, must NOT flag | 52.2% / **100% ❌** | 0.0% / **0.0% ✅** |
| **HARMFUL** ln_act@0.1 nas=2 — Δ**+16.9** px, SHOULD flag | 67.2% / 100% | 53.2% / **100% ✅** |
| ln_act@0.1 nas=6 — Δ−3.2 px (wash), must NOT flag | 44.2% / **100% ❌** | 0.0% / **0.0% ✅** |

> **The pass-through screen flags all four cases at 100% — three of them false alarms. Zero
> discriminative power. The settled-distance screen flags exactly the one harmful case and none
> of the other three: 4/4 correct.**

**Calibration** (exact paired null — random sign-flip per pair, which is the null a symmetric
paired test assumes; *not* frozen-vs-frozen, whose near-zero discordance means 0% reflects the
test having nothing to fire on):

| screen | n=5 | n=10 | n=15 | n=20 | |
|---|---:|---:|---:|---:|---|
| settled-dist Wilcoxon | 3.1% | 4.7% | 5.4% | 4.9% | ✅ ≈ α=0.05 |
| pass-through McNemar | 0.0% | 0.5% | 1.9% | 0.7% | ❌ badly under-calibrated |

The pass-through screen is simultaneously **conservative under the null and firing on everything
in practice** — the signature of a test whose discordance is driven by something other than the
quantity of interest.

**§7's sentence:** *a paired closed-loop screen of ~15–20 episodes is a usable acceptance gate
only when its statistic is the settle-validated outcome; the same probe built on
threshold-crossing success cannot distinguish a helpful adapter from a harmful one.*

**Keep the pass-through row in the paper** — relabelled as the false-alarm result. It is not an
embarrassment to be deleted; it *is* the finding, and it is the constructive half of the thesis.

### The pattern this makes three of

Arithmetically correct, wrong construct: (1) the settle-1 floor-effect error; (2)
`B2-transfer-01`'s settled-vs-pass-through subset (R2.0-a); (3) this. All three passed every
numerical check on this project's list — recompute from raw, assert pairing, verify config — and
failed on *what the quantity measures*. **Standing pre-report question, added to the Day-4
checklist: "if this number moved, what would have had to change in the world?"**



═══════════════════════════════════════════════════════════════════════════════
# REVISION 1 — retained unchanged below this line (Aug 31). Superseded where R2 disagrees.
═══════════════════════════════════════════════════════════════════════════════


**Status:** operative. Supersedes the experiment lists of both `FABLE5_SIX_DAY_PLAN.md` and the
intermediate five-day draft. **Retains all governance rules verbatim:** pre-registration
paragraph in `IMPLEMENTATION_PLAN_V3.md` §8 *before* any launch; every run `--detach`,
downloaded immediately, archived under `phase0_v3/`, ledgered the same day; protected files
(`atlas/score.py::umf`, `atlas/stats.py` existing functions, `run_e0_planning.py`'s planning
loop) get **additions only**, each with a FALSIFICATION line in `FIXLOG.md`; **no citation
enters the draft without someone having read its actual abstract.**

**Venue:** NeurIPS 2026 Workshop on *World Models in Physical AI* (Sydney). **8 pages max
excluding references**, NeurIPS template, double-blind, **non-archival**, OpenReview.
**Deadline Sept 5 2026 AoE.** Notification Sept 29.

**Paper spine:** `Paper_Draft/PAPER_ARCHITECTURE_v1.md` (thesis, H1–H9, evidence map, section
plan, retired-claims list). Every experiment below feeds a named hypothesis in that file.

**Budget:** ~$25–30. **Compute is not the binding constraint; wall clock is.**

---

## ✅ SESSION PROGRESS (2026-09-01) — Day 0 + Day 1 experiments

Results log: `research_audit/FINAL_FIVE_DAY_DAY1_RESULTS.md`. Ledger rows: `B3-dose-ladder`,
`B2-transfer-01`, `N12-n50`, `N13-screen-power`, `N14-coast-model`. GPU runs on Modal profile
`aiden-dsouza-201323` (~$5); items C / 1.G / 2.3 are $0 local.

| item | status |
|---|---|
| **R2.0-a** correction (B2-transfer-01 clean subset → neither-pass-through n=8 p=0.109 n.s.) | ✅ applied — results doc §1.B + ledger `B2-transfer-01` + both scripts; dated, not overwritten |
| **R2.0-b** correction (monotonicity: nas=6 monotone, nas=2 monotone-to-±1-episode) | ✅ applied — results doc §1.C + ledger `B3-dose-ladder`; Figure 1 plots both cadences |
| **2.3** screen table | ✅ done, then **CORRECTED same day** — the first version built the screen on pass-through success and called the *helpful* adapter "known-bad"; replaced by the 2x2 discrimination table (R2.6). Ledger `N13-screen-power` + its correction note |
| **C** analytical coast model ($0 local) | ✅ done — closed form `D=v₀/ln(1/damping)` from the pymunk integrator; C-1 per-episode r 0.91–0.99, ratio 1.00; C-2 predicts the settled-SR ladder to ±1 episode at all 10 points. §5/H6 is now a derivation. Ledger `N14-coast-model` |
| **B** n=100 headline (~$3, aiden) | ✅ done — tasks 50–99 replicate (Δ−53.7 p=0.0001); **merged paired n=100 Δ−47.0 [−61,−33] p<0.0001, 71/100**; all 3 disjoint sets chart-closer; pass-through inversion intact. Ledger `N15-n100` (supersedes `N12-n50` as §4 headline) |
| **A** controller-family (~$5, aiden) | ✅ done, then **REWRITTEN** (external verification: two-group artifact framed as family measurement + a dropped data point). **PRIMARY = the CEM-iteration ladder** (frozen only, one knob): pass-through SR flat 0.35→0.45, settled distance **4.4× monotone** (it1 36 → it30 159 px), it1 vs it30 **Δ−122 px, 19/20, p=1.9e-6** — the thesis with no adapter/confound. τ_b=−0.51 demoted to secondary (within-group n.s.). Ledger `N16-controller-family` |
| **E** momentum negative control | ⛔ abandoned per its own 90-min gate — `--block-cog` / `shape=` inert in this pymunk setup; `FIXLOG V3-23`. **Not needed:** C derives `D_∞ → 0` as damping → 0 and the R0 cells confirm it (**32/32 survive, coast 0.00 px**) — the momentum-free case is measured. One §8 sentence for the loss of a *non-zero-damping* momentum-free shift |
| **D** `lora4` second adapter class | ⬜ **DROPPED** — the adapter is no longer the §4 headline (H5′ / the iteration ladder is), so "one adapter class" is no longer the exposure. Not worth a day + its OOM history |
| **Day 0** pre-flight (0.1 pre-reg §8.7, 0.2 CLI verify, 0.3 config) | ✅ done |
| **1.A** damping-0.1 on-policy collection | ⬜ NOT run — needs go-ahead; 1.B weakened its rationale |
| **1.B** R2 adapter @ damping 0.1 | ✅ done — chart does NOT transfer; C-2 effect is severity-specific to 0.5 (nas=2 chart sig. worse, p=0.044) |
| **1.C** damping dose ladder (6 new cells) | ✅ done — **H1 divergence confirmed** (monotone at nas=6; monotone to ±1 episode at nas=2 per R2.0-b); Figure 1 built |
| **1.D** N=50 replication (tasks 20–49) | ✅ done — **REPLICATES both cadences**; merged n=50 nas=2 Δ−40.2 p=0.0001, nas=6 Δ−30.5 p=0.0027 |
| **1.E** no-early-stop | ⛔ CUT (unchanged) |
| **1.F** evening documentation (ledger + §8 addendum) | ✅ done (§8.7 addendum + 3 ledger rows) |
| **1.G.1** settle-length sensitivity | ✅ done — "40" is not tuned; R0 flat across all holds |
| **1.G.2** early-termination asymmetry | ✅ (carried from prior session, unchanged) |
| **1.G.3** H5 compute-confound reframe | ✅ (carried from prior session, unchanged) |
| **1.G.4** regenerate figures | 🟡 partial — Figure 1 (ladder) built; rest = Day 3 |
| **1.G.5** termination-timing table | ✅ done — reproduces the plan's table exactly from raw JSONL |

---

## THE ONE RULE

> **Experiments freeze at end of Day 2 (Sept 1).** Anything not downloaded *and* ledgered by
> then does not go in the paper. Days 3–5 are writing and checking only.

This project has already shipped a fabricated citation and a floor-effect statistical error,
each caught only because a review pass had time to run. Day 4 is what buys that. Do not spend it.

---

## DAY 0 — PRE-FLIGHT (do first, ~45 min, before any launch)

**0.1 Write the pre-registration paragraph** into `IMPLEMENTATION_PLAN_V3.md` §8 as a dated
addendum, covering every launch below with its decision rule. Governance requires this *before*
launch, not after.

**0.2 Verify the CLI against `--help`.** Confirmed present in `scripts/run_e0_planning.py`
(2026-08-31); the Modal wrapper `modal/modal_e0_planning.py` mirrors them with underscores:

| local flag | modal wrapper | note |
|---|---|---|
| `--kind {baseline,ln_act,lora4,full}` | `--kind` | |
| `--regime {R0,R1,R2}` + `--regime-config '{"damping":X}'` | `--regime`, `--regime-config` | |
| `--episodes N`, `--episode-start K` | `--episodes`, `--episode-start` | **`--episode-start` is the disjoint-task flag** |
| `--iterations`, `--num-act-stepped` | same | **defaults are 30 and 6 — pass explicitly every time** |
| `--charts-root`, `--charts-dir` | `--charts-root`, `--charts-subdir` | |
| `--out-root`, `--out-dir` | `--out-root`, `--out-subdir` | |
| `--settle-steps N` | `--settle-steps` | |
| — | `--num-shards` | parallel fan-out |

⚠ **Footgun T-14:** charts live at `--charts-root phase0_v3 --charts-subdir p0g_onpolicy`.
**NOT** `p0g_onpolicy_frozen_check` — that directory holds an identity chart and will silently
reproduce the baseline.

**0.3 Confirm the standing config** for every planning cell below: `it=10, N=300, H=6,
--settle-steps 40`, seeds via `--episode-start`/`--episodes`.

---

## DAY 1 — Mon Aug 31 (tonight) — LAUNCH EVERYTHING, IN LEAD-TIME ORDER

All launches are `--detach` and independent; Modal runs them as parallel containers. **Order
matters only because of queue congestion — launch by longest lead time first.**

### 1.A — damping=0.1 on-policy collection (LONGEST POLE — launch first) ~$15

`modal_phase0.py` P0-G collector at `damping=0.1`, same recipe as the R2 collection
(100 train / 8 val / 8 test trajectories, CEM `300×10 nas=2`, `_determinism.py` active,
`total_contacts > 0` filter **off**).

- **Feeds:** limitation #3 in the architecture doc — §4's adapter comparison is currently
  measured only where the frozen baseline is *directionally wrong* (it pushes the block away
  from goal in 14–16/20 episodes at damping 0.5).
- **Hard abort:** if the chart is not trained **and** screened by **EOD Day 2**, drop it. 1.B
  below is the guaranteed fallback.
- **Artifact:** `phase0_v3/p0g_onpolicy_dmp01/`

### 1.B — the R2 adapter evaluated at damping=0.1 (CHEAP FALLBACK FOR 1.A) ~$1

**New in this plan.** You do **not** need a retrained adapter to ask the core question. Screen
the **existing** `p0g_onpolicy` R2-trained adapter against frozen `c₀` at `damping=0.1`,
nas=2, 20 paired episodes, settle-40.

- **Why it exists:** 1.A is the single riskiest item in the plan (collection + training +
  screening, all before EOD Day 2). This costs $1, finishes in ~40 minutes, and answers *"does
  the adapter's less-destructive behaviour appear where the baseline actually works?"* tonight,
  guaranteed.
- **Caveat that travels with it:** the adapter is trained at damping 0.5 and tested at 0.1, so
  a null is ambiguous between "the effect is severity-specific" and "the adapter is
  off-distribution." State that. If 1.A lands, 1.A supersedes this and 1.B becomes a
  transfer-robustness footnote.
- **Artifact:** `phase0_v3/dmp01_transfer_{baseline,ln_act}_nas2/`

### 1.C — damping dose ladder (HIGHEST PAPER VALUE) ~$5

Frozen `c₀` only. `damping ∈ {0.05, 0.2, 0.3}` — 0, 0.1, 0.5 already on disk — both cadences
nas ∈ {2,6}, seeds 0–19, settle-40. **6 new cells.**

- **Feeds H1.** Converts §3 from a three-point contrast into a dose-response curve, and makes
  **Figure 1 the lead figure**. This is what defeats *"you picked a degenerate regime."*
- **Pre-registered decision rule:** the claim is **monotone divergence** — pass-through SR falls
  more slowly than settled SR as damping rises. If the two fall *together* at some intermediate
  damping, report it: "the criterion is valid in a band and fails outside it" is a **more
  precise** finding, not a worse one. **Do not drop points.**
- **Also feeds H6.** Recompute coast at each new damping — the existing three points
  (0.0 → 8.7 → 72–74 px) already make residual momentum a dose-response that *mechanically
  explains* H1 rather than merely correlating with it. Six points make it a curve.
- **Artifact:** `phase0_v3/ladder_dmp{005,02,03}_baseline_nas{2,6}/`

### 1.D — N=50 replication on disjoint tasks ~$4

`--episode-start 20 --episodes 30`. Tasks 20–49, disjoint from every existing cell. Frozen +
adapted, nas=2 **and** nas=6, settle-40. 4 arms × 30 episodes.

- **Feeds H4.** Kills *"n=20, same 20 tasks reused everywhere."* Merged with 0–19 gives paired
  n=50.
- **Pre-registered replication target: settled distance at nas=2 (chart < frozen)** — **not**
  the retired pass-through headline.
- **Decision rule:** if the direction replicates on 20–49, merge and report n=50, stating the
  two-launch structure. If it does not, **report both sets separately** and §4's claim weakens
  to *"significant in one of two task sets."* **Do not average it away.** §3 (H1–H3) is
  frozen-baseline-only and unaffected either way, so this cannot sink the paper.
- **Artifact:** `phase0_v3/n50_{baseline,ln_act}_nas{2,6}_ep20-49/`

### 1.E — ~~no-early-stop control~~ **CUT 2026-08-31. Do not run. Do not re-propose.**

An earlier revision of this plan proposed an additive `--no-early-stop` flag so both arms would
execute all 30 raw steps, to remove the differential-compute confound from §4. **It was wrong,
for three reasons, and the reasoning is recorded here so a later session does not resurrect it:**

1. **Redundant.** 1.G.2's neither-succeeded subset already delivers matched compute (3.00 vs
   3.00 replans) and the effect holds (p=0.0244). The question was answered before the run was
   proposed.
2. **It would measure a protocol nobody deploys.** The paper's claim is about the substrate's
   *own shipped* evaluator, which terminates on the first crossing. Disabling that measures a
   variant with no standing in the literature and invites *"you changed the protocol until your
   result appeared"* — the exact attack §3 exists to survive.
3. **Early termination is the phenomenon, not a confound on it.** See 1.G.5 — the criterion
   halts control at the moment the block is transiting fastest, and halts the harder-shoving arm
   earliest. Removing it would delete the mechanism in order to measure it.

It also cost a change to the protected planning loop on the day with the least slack, for a
number already on disk.

### 1.F — Evening documentation

Ledger rows for every launch. A dated §8 addendum in `IMPLEMENTATION_PLAN_V3.md` recording this
plan as superseding the six-day plan, and why.

### 1.G — Free analyses, tonight while runs are detached (~2h, $0)

These need no GPU and three of them are load-bearing.

**1.G.1 Settle-length sensitivity** — settled SR and settled distance at hold ∈ {1, 5, 15, 30,
40} for every arm, from the existing `settled_trace`. Pre-empts *"40 is a tuned number."*
Feeds §3 and red-team surface #3.

**1.G.2 The early-termination asymmetry in H4 — RESOLVE AND REPORT.**
*Verified in the main session 2026-08-31:* episodes break on pass-through success, so the two
arms do **not** receive equal planning budget — at nas=2 the frozen arm averages **2.25**
replans and the adapter **2.85**, because the frozen arm is stopped earlier by the criterion.
The clean subset removes this entirely:

| population | n | chart | frozen | Δ | better | Wilcoxon p | replans |
|---|---:|---:|---:|---:|---:|---:|---|
| all tasks | 20 | 77.6 px | 137.3 px | **−59.8** | 17/20 | **0.0002** | 2.85 vs 2.25 ✗ |
| **neither succeeded** | 11 | 91.9 px | 139.8 px | **−47.9** | 9/11 | **0.0244** | **3.00 vs 3.00 ✓** |

**The effect survives at perfectly matched compute.** Report **both** rows, disclose the
asymmetry, and note that the subset conditions on a joint outcome symmetric across arms. **This
is the answer** — no additional run is needed or wanted (see 1.E, cut).

Note the frozen arm's full-budget behaviour is *already* observable here: on the 11 tasks where
it never crossed, it ran all 30 raw steps and still ended at **139.8 px**. It does not recover
given the full budget. That is the question a no-early-stop run would have asked.

**1.G.3 The H5 compute confound — REFRAME BEFORE DRAFTING.**
*Verified at source 2026-08-31:* `planner.py:159` sets `plan_length = min(horizon, steps_left)`,
but `run_e0_planning.py:279` computes `n_replans_target = max_steps // num_act_stepped` in
**mismatched units** (raw steps ÷ model steps), so `steps_left` is always ≫ `horizon` and
`plan_length` is pinned at **6 in both cadences**. Measured replans: nas=2 → **2.25–2.85** CEM
searches; nas=6 → **1.00**. So nas=2 receives ~**3× the CEM search compute** at the same
lookahead. This is `EVIDENCE_LEDGER` Section 4's N5 caveat, confirmed.

> **Consequence — phrase H5 as an inversion, not a causal control claim.**
> ✅ *"The criterion inverts the ranking of two planning configurations."*
> ❌ *"Closed-loop MPC is better and the metric hides it."*
>
> The **inversion survives the confound** — both metrics score the identical runs, so a compute
> difference cannot flip a sign between them. What the confound blocks is the causal attribution
> to feedback. Disclose the 3× compute in §5 and §8. Do **not** write
> `DAY1_CADENCE_METRIC_ANALYSIS.md`'s "contradicts control theory" framing without it.

**1.G.4 Regenerate figures** — Fig 2 (drift curves from `settled_trace`, one line per arm), Fig
3 (SR-vs-radius from `c2_threshold_sweep.json`), ECDF. Re-derive the ⬜ items in the
architecture doc: sd-ratio CIs, the C-1 ranking numbers, the contact-collapse figures.

**1.G.5 Termination timing — a NEW result, and it belongs in §5 beside H6.**
*Recomputed from `success_at_step` in the main session 2026-08-31:*

| arm | n succ | fires at step (of 30) | budget left unused |
|---|---:|---:|---:|
| frozen, R2, nas=2 | 9 | **8.6** | **21.4** |
| chart, R2, nas=2 | 2 | 8.5 | 21.5 |
| frozen, R2, nas=6 | 11 | 12.3 | 17.7 |
| chart, R2, nas=6 | 5 | **21.0** | 9.0 |
| frozen, R0, nas=2 | 13 | 15.1 | 14.9 |

**The criterion does not merely mis-score the outcome — it terminates control at the moment the
block is transiting fastest, and it terminates the harder-shoving arm earliest** (a harder shove
reaches the goal window sooner). The frozen arm stops controlling at step 8.6 of 30 with 21.4
steps of budget unused, and the block then coasts +74 px. This is the same mechanism as H6's
residual-momentum table, seen from the control side rather than the physics side, and it is
*why* the two are causally linked rather than merely correlated.

Write it as a finding in §5. **Do not treat it as a confound to be engineered away** — that was
1.E's error.

---

## DAY 2 — Tue Sept 1 — ANALYSE, ONE CONTINGENCY, FREEZE AT EOD

**2.1** Analyse 1.C and 1.D the moment they land. Build **Figure 1** (the ladder) and the n=50
tables with paired-bootstrap CIs. Assert pairing (0 mismatches on `init_block_pos_diff` at 1e-6)
on every new cell before computing anything.

**2.2** Train + screen the damping-0.1 adapter if 1.A's collection completed. **Abort per 1.A
if not** — 1.B already covers the question.

**2.3 Screen power table (free, subsampling).** From all paired closed-loop data on disk:
detection rate of a paired closed-loop probe at n ∈ {5, 10, 15, 20} against the known-bad
configuration, and false-block rate on frozen-vs-frozen pairs. Feeds §7 — this is what converts
a diagnostic into a tool a reviewer can use.

**2.4 OPTIONAL — `lora4` second adapter class (~$3).** Only if 1.A–1.D are **all** green and
downloaded **by midday**. Matched budget to the `ln_act` recipe (kills the OPUS #12
training-set-size confound for this pair); C-2 screen at nas=2 with settle-40. Buys "both
adapter classes tested compress" in §6/§8. **It has an OOM history. If anything else is
outstanding, skip it.**

**2.5 🧊 FREEZE AT EOD.** Every number destined for the paper has an `EVIDENCE_LEDGER` row with
its raw-file path. No further launches, for any reason.

---

## DAY 3 — Wed Sept 2 — FULL DRAFT

Write into the NeurIPS template against `Paper_Draft/PAPER_ARCHITECTURE_v1.md`. **Writing order
is not reading order:**

1. **§3** (H1–H3) and **§5** (H6) first — the two strongest, fully-measured sections, with
   Figures 1 and 2.
2. **§4** (H4, H5, H7) — with 1.G.2's two-row table and 1.G.3's reframing.
3. **§2** setup, then **§6** (H8, H9).
4. **§8 limitations** — write it long, then trim. All eight items in the architecture doc §6
   already have numbers.
5. **§1 intro last**, once you know what the paper actually says.
6. **Abstract very last.**

**Every number goes in with its ledger tag in an HTML comment.** Strip at camera-ready.

**🛑 Before you write §4 and §6, open `PAPER_ARCHITECTURE_v1.md` §5 and grep the draft for all
nine retired claims.** Four separate documents in this repo assert *"the adapter improves
prediction and collapses control."* It is dead and it will try to migrate.

---

## DAY 4 — Thu Sept 3 — ADVERSARIAL PASS (never drop this day)

**4.1** Fact-check every number against the **raw JSONL** — not against `FIXLOG`, not against
`EVIDENCE_LEDGER`, not against this file.

**4.2** Fact-check every citation by reading its actual abstract. One fabricated attribution to
a real paper has already occurred in this project (arXiv:2608.18227).

**4.3** Regenerate every figure and table **from logs alone**. Anything that does not regenerate
is fixed or cut.

**4.4 Red-team the six attack surfaces** — verify the draft actually answers each:

| # | attack | answer |
|---|---|---|
| 1 | "Your regime is degenerate." | §3's ladder spans non-degenerate regimes; §3 is frozen-baseline only |
| 2 | "n=20." | n=50 on disjoint tasks (1.D) |
| 3 | "Your metric is arbitrary." | validated at damping 0 (32/32 survive) + settle-length sensitivity (1.G.1) |
| 4 | "One toy environment." | stated in §1 as scope, not buried in §8 |
| 5 | "This is just variance." | sd-ratio CI excludes 1 in every pool + the residual-momentum table — a physical mechanism |
| 6 | **"Your two arms didn't get equal compute."** | 1.G.2's matched-replan subset (3.00 vs 3.00, p=0.0244) + 1.G.3's disclosure. And 1.G.5 reframes the unequal budget as the finding rather than a defect — **this surface is new; the earlier plan did not have it** |

**4.5** Read the CFP line by line: page limit, template version, anonymisation, the **AI-agent
disclosure** requirement, the reviewing commitment. Write the disclosure from the
FIXLOG/EVIDENCE_LEDGER audit trail — it will read as rigor, not confession.

---

## DAY 5 — Fri Sept 4 → Sept 5 AoE — FINAL READ, SUBMIT

Buffer. One full read aloud. Submit with hours to spare — OpenReview at a deadline is not where
you want to discover a template problem. **No new content.**

---

## DROP ORDER IF A DAY SLIPS

Drop from the top:

1. **2.4** `lora4` (already conditional)
2. **1.A** damping-0.1 collection (1.B is the $1 fallback)
3. **1.D**'s nas=6 arms (keep the nas=2 pair — nas=2 is the pre-registered target)
4. **1.C**'s `damping = 0.05` and `0.3` points (keep `0.2` — four points still beats three)

*(1.E is not in this list because it is cut outright, not deprioritised.)*

**Never drop:** the ladder's `0.2` point · settle-length sensitivity (1.G.1) · the H4 matched
subset (1.G.2) · the H5 reframing (1.G.3) · the termination-timing table (1.G.5) · the
drift-curve figure · **Day 4 in its entirety.**

---

## RISK REGISTER

| risk | mitigation |
|---|---|
| 1.A collection overruns EOD Day 2 | 1.B ($1, done tonight) is the guaranteed fallback; hard abort is pre-registered |
| 1.D fails to replicate | Pre-registered: report separately, weaken §4 to "one of two task sets." §3 unaffected — cannot sink the paper |
| Someone re-proposes a no-early-stop run | Cut with reasons recorded in 1.E. The early stop is the phenomenon (1.G.5), and 1.G.2 already supplies matched compute |
| Modal queue congestion | Launch order 1.A→1.D is the priority order; 1.C and 1.D are the ones that must land |
| A retired claim survives into the draft | Architecture doc §5 grep, Day 3, before §4 and §6 are written |
| `lora4` OOM | Batch size down, never fewer trajectories. Skip on any doubt |
| Writing overruns into Day 4 | THE ONE RULE. Day 4 is not a buffer for Day 3 |

---

## EXPLICITLY NOT IN THIS PLAN

Recorded so absence reads as decision, not oversight:

- **E1, E2, E3+E4, E5.** E4 specifically is **deferred, not cancelled** — see FIXLOG "E4
  DEFERRAL" (2026-08-31) and `EVIDENCE_LEDGER` Section 6. Its original kill premise
  (`FABLE5_VALIDATION.md` §6) is **retired**; the deferral now rests on wall clock, the `nas=1`
  → UMF≡e1 collapse (FIXLOG V3-22), E4's use of the invalidated pass-through metric, and scope.
- The Phase-0 freeze pipeline: G7 asserting stage, σ_r forward pass, `n_probe` sweep, `K_max`/`m`
  sign-offs, any τ re-derivation.
- The τ-vs-chart gate analysis — it served an ATLAS routing story the paper no longer tells. One
  neutral sentence in §6 covers it (the gate *does* commit the adapter at pinned τ=0.5, and the
  adapter is not harmful, so the observation is no longer damning).
- **R1 anything** (prediction-level shift only per G4; signed-off dropped).
- The remaining mechanism hypotheses — predicted-displacement dispersion, action-sensitivity.
  The residual-momentum result (H6) is a better-measured mechanism than either would produce.
- **Any change to `atlas/score.py::umf`, `atlas/stats.py`'s existing functions, or the planning
  loop — including the `--no-early-stop` flag proposed and cut in 1.E.** No code change to a
  protected file is in scope for this plan at all.
- A second environment. That is the 2027 archival paper, with E4 and damping-0.1 as primary.

---

## THE HONEST EXPECTATION

This plan produces a defensible **7–8 page** workshop submission: a measured,
mechanistically-explained, dose-response-supported criterion-validity result on one substrate,
with two independently inverted conclusions and a constructive screen.

**Write to content, not to the limit.** 8 pages is a maximum. A tight 7 beats a padded 8, and
padding is more punishable at a workshop than brevity.

**Acceptance estimate: 50–65%.** Non-archival workshop status genuinely raises accept rates.
Dragging it down: single substrate, single shift family, and a core finding a hostile reviewer
can frame as an eval-harness bug — which is exactly what the reachability/stabilizability
framing (architecture doc §1) and H5's second inversion exist to defeat. *(A prior session
estimated 65–75%. The disagreement is recorded rather than averaged. Neither of us knows this
workshop's bar.)*

**The submission is non-archival.** It is a poster, reviewer feedback, and a timestamped
artifact you can extend into an archival submission in 2027 with a second environment, E4 at
`nas=2` with the settle metric ported, and `damping=0.1` as the primary regime. Decide with
that on the table.
