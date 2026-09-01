# FINAL FIVE-DAY PLAN — Day 1 results log

**Date:** 2026-08-31 → 2026-09-01. Executes `research_audit/FINAL_FIVE_DAY_PLAN.md` Day 1.
Every number recomputed from raw per-episode JSONL this session (`CLAUDE.md` §1.9).

## WHERE EVERYTHING FROM THIS SESSION LIVES (for a reviewer)

This file is the **session narrative + full tables**. The canonical, single-source-of-truth
numbers are also in:

| what | where |
|---|---|
| **Pre-registration** (decision rules, written *before* launch) | `IMPLEMENTATION_PLAN_V3.md` §8.7 |
| **Canonical numbers** (arithmetic record) | `EVIDENCE_LEDGER.md` §1 rows **`B3-dose-ladder`**, **`B2-transfer-01`**, **`N12-n50`**, **`N13-screen-power`**, **`N14-coast-model`**, **`N15-n100`** (supersedes `N12`), **`N16-controller-family`** |
| **Two post-hoc corrections** (independently caught, applied dated) | R2.0-a (B2 clean subset) + R2.0-b (monotonicity) — see `FINAL_FIVE_DAY_PLAN.md` REVISION 2, and the 🛑 blocks in §1.B / §1.C below |
| Plan progress checklist | `FINAL_FIVE_DAY_PLAN.md` "✅ SESSION PROGRESS" table near the top |
| Analysis scripts (read-only, no production path) | `scripts/day1_free_analyses.py`, `day1_ladder_analysis.py`, `day1_n50_analysis.py`, `day1_fig_ladder.py`, `day2_screen_power.py`, `day2_coast_model.py`, `day2_n100_analysis.py`, `day2_controller_family.py` |
| Analysis JSON outputs | `phase0_v3/day1_*.json`, `day2_screen_power.json`, `day2_coast_model.json`, `day2_n100_analysis.json`, `day2_controller_family.json` |
| Figures | `phase0_v3/day1_fig_ladder.{png,pdf}` (Fig 1, dose ladder), `day2_fig_coast.{png,pdf}` (coast model), `day2_fig_controller_family.{png,pdf}` (12-controller τ) |
| Raw per-episode JSONL (the actual evidence) | `phase0_v3/ladder_dmp{005,02,03}_baseline_nas{2,6}/`, `phase0_v3/dmp01_transfer_ln_act_nas{2,6}/`, `phase0_v3/n50_{baseline,ln_act}_nas{2,6}_ep20-49/` |
| Migration doc (new Modal account) | `modal/MIGRATE_NEW_ACCOUNT.md` |
| Modal incident (wrong-account leak) | this file, "⚠️ INCIDENT" below; memory `project_modal_account_pinning` |

No production code was modified this session. No new FIXLOG entry (the `--settle-steps` flag
it uses was added in a prior session, V3-20/21).

---

## ✅ RESOLVED — Modal migration to `aiden-dsouza-201323` complete; 1.C launched

**The "spend limit" error was transient** (retried clean). The real blocker was an **empty
`atlas-data` volume** on the new account (dir stubs only) — a container ran and crashed on
the missing jepa-wms hub clone.

**Migration done** per `MODAL/migrate_new_account.md` (~287 MB, not the 11 GB first estimated —
the LPIPS decoder 3.4 GB / `dino_wm_pusht` 263 MB / tokens.pth 4.8 GB all auto-download or are
unused). Uploaded: `hub/hub/facebookresearch_{jepa-wms,dinov2}_main`, Push-T demo data
(`states.pth` 184 MB, `rel_actions.pth` 74 MB, `seq_lengths.pkl`, `val/`), R2 + R0 charts.
⚠ Git Bash mangled the first upload to a literal `C:/Program Files/Git/...` path on the volume
— redone with `MSYS_NO_PATHCONV=1`, bogus path removed.

**Migration smoke PASSED** (`_migration_smoke`, 1 ep, tiny CEM, exit 0, 2 s/ep, 1.72 GB GPU) —
volume verified, deleted after.

**1.C launched 2026-09-01 ~00:40 IST** — 6 cells, each its own `--detach` ephemeral app
(⚠ launching multiple `modal run --detach` from one parent script tears down all but the
last — each must be its own top-level process):

| cell | app | nas | damping |
|---|---|---|---|
| `ladder_dmp005_baseline_nas2` | ap-TgjJ6QwGHwAGwBKaMLajWz | 2 | 0.05 |
| `ladder_dmp005_baseline_nas6` | ap-zEX7NwND40Sm8XhQrY9aPA | 6 | 0.05 |
| `ladder_dmp02_baseline_nas2`  | ap-JdhbOfeOcs7e9VzMDzjKMU | 2 | 0.2 |
| `ladder_dmp02_baseline_nas6`  | ap-xOiqRam2JgOf8YDqqSl3MW | 6 | 0.2 |
| `ladder_dmp03_baseline_nas2`  | ap-eAwUudOiOc3UUaFWrUgFGN | 2 | 0.3 |
| `ladder_dmp03_baseline_nas6`  | ap-xyywd2nENZKUnYEKzhhNF7 | 6 | 0.3 |

All `--kind baseline --regime R2`, it=10, N=300, H=6, `--settle-steps 40`, seeds 0–19,
`--charts-root phase0_v3 --charts-subdir p0g_onpolicy --out-root phase0_v3`.

### ⚠️ INCIDENT — first launch batch leaked to the wrong account, killed by its budget cap

Modal's active account is a single global `~/.modal.toml` field. A concurrent Claude session
flipped it to `pandereshubham` mid-work. Each `modal run` starts a fresh shell that re-reads
that file, so the 1.C/1.B batch (00:40–00:50 IST) launched on **`pandereshubham`**, not
`aiden-dsouza-201323`. pandereshubham then hit its $30 workspace budget (101%, ~$30.33 —
**~$0.33 over**) and Modal paused the containers after **63 / 160 episodes**.

- Partial JSONLs pulled to `phase0_v3/_salvage_pandereshubham/` (2/8/4/11/4/19/2/8 episodes
  per cell) — **kept for the record, NOT used** (mixing two accounts' containers in one cell
  is the cross-image confound this project has been bitten by; plan rule 4).
- All 8 pandereshubham apps stopped (confirmed 0 alive).
- **Relaunched all 8 clean on `aiden-dsouza-201323` at ~01:20 IST** with
  `MODAL_PROFILE=aiden-dsouza-201323` forced as an env var on every command (overrides the
  global file — immune to further flips). URLs verified `.../aiden-dsouza-201323/...` for all 8;
  8 apps × 1 task confirmed running; pandereshubham 0 alive. `aiden` volume was already
  migrated + smoke-tested, so no re-upload.

ETA ~50 min (nas=2 cells) / ~17 min (nas=6).

**Still pending:** 1.B (chart arm, damping 0.1 — needs `chart_ln_act_R2.pt`, already on volume),
1.D (N=50 rep, ep 20–49), 1.A (damping-0.1 collection).

---

## (historical) initial blocker note

The first attempt hit `Workspace ... has exceeded its spend limit` — transient; see above.

- **`aiden-dsouza-201323`** (the new account, active profile, token set 2026-08-31): the
  first `modal run` returned **`Workspace ac-0cCJRoHEprkOxC2lhCwFKl has exceeded its spend
  limit`** during image build. No container started; app `ap-5ibpwrTRDjl2kdiFh4dUSh`
  (0 tasks) was stopped immediately — **$0 spent.**
- **`aidendsouzavnit`** (the account all prior Phase-0 work ran on): not launched against —
  spend status unknown, and the plan / user direction is to use the new account.

**Nothing downstream of a launch was done.** Waiting on the user to raise the spend limit on
`aiden-dsouza-201323` (Modal dashboard → Settings → Usage / Spend limit) or authorise the
old account.

**When unblocked, launch order (pre-registered §8.7), all `--detach`, `export PYTHONUTF8=1`:**

1. **1.B** (~$1, ~40 min, guaranteed value) — chart arm only, damping 0.1, nas 2 **and** 6:
   ```
   modal run --detach modal/modal_e0_planning.py::main --kind ln_act --regime R2 \
     --regime-config '{"damping": 0.1}' --episodes 20 --num-samples 300 --iterations 10 \
     --horizon 6 --num-act-stepped {2,6} --charts-root phase0_v3 --charts-subdir p0g_onpolicy \
     --out-root phase0_v3 --out-subdir dmp01_transfer_ln_act_nas{2,6} --settle-steps 40
   ```
   Frozen arm already on disk (`c2_settle2_dmp01_baseline_nas{2,6}`, verified below).
2. **1.C** (~$5) — dose ladder, `--kind baseline`, `--regime-config '{"damping": D}'` for
   D ∈ {0.05, 0.2, 0.3}, nas ∈ {2,6}, `--episodes 20`, `--settle-steps 40`. 6 cells,
   `--out-subdir ladder_dmp{005,02,03}_baseline_nas{2,6}`.
3. **1.D** (~$4) — `--episode-start 20 --episodes 50` (⚠ `--episodes` is an END index),
   `--kind {baseline,ln_act}`, R2 damping 0.5, nas ∈ {2,6}, settle-40.
   `--out-subdir n50_{baseline,ln_act}_nas{2,6}_ep20-49`.
4. **1.A** (~$15) — `modal_phase0.py` P0-G collector at damping 0.1. Hard abort if not
   trained+screened by EOD Day 2; 1.B is the fallback.

---

## Day 0 pre-flight — DONE

- **CLI verified against `--help`** (`scripts/run_e0_planning.py`, 2026-08-31): `--settle-steps`,
  `--regime-config`, `--episode-start`, `--num-act-stepped`, `--iterations` all present.
  Modal wrapper `modal/modal_e0_planning.py::main` mirrors them; the shard path (`num_shards>1`)
  threads `settle_steps` correctly (line 243).
- **Pre-registration** written into `IMPLEMENTATION_PLAN_V3.md` §8.7 before any launch.
- **Charts confirmed** at `phase0_v3/p0g_onpolicy/chart_ln_act_R2.pt` (NOT
  `p0g_onpolicy_frozen_check`, trap T-14).
- **Existing damping-0.1 frozen baseline re-verified from raw JSONL** (this is 1.B's frozen arm):

  | arm | protocol (from summary JSON) | pass-through SR | settled SR (hold 40) |
  |---|---|---:|---:|
  | `c2_settle2_dmp01_baseline_nas2` | R2 damping 0.1, it=10, N=300, nas=2, settle-40 | 11/20 (0.55) | 2/20 (0.10) |
  | `c2_settle2_dmp01_baseline_nas6` | same, nas=6 | 14/20 (0.70) | 4/20 (0.20) |

  Matches `DAY1_CADENCE_METRIC_ANALYSIS.md` §4 and `EVIDENCE_LEDGER` row `B2-damping-sweep`.

---

## 1.G.1 — settle-length sensitivity  ✅ RAN (free, local, `scripts/day1_free_analyses.py`)

Settled SR and **median** settled block-distance at hold ∈ {1, 5, 15, 30, 40} raw steps,
from the `settled_trace` checkpoints already in every `c2_settle2_*` episode record.
20 episodes/arm, seeds 0–19. Output: `phase0_v3/day1_free_analyses.json`.

| arm | pass-through SR | hold 1 | hold 5 | hold 15 | hold 30 | hold 40 |
|---|---:|---|---|---|---|---|
| **R0** baseline nas=2 | 0.65 | .65 / 17 px | .65 / 17 | .65 / 17 | .65 / 17 | **.65 / 17** |
| **R0** baseline nas=6 | 0.95 | .95 / 14 | .95 / 14 | .95 / 14 | .95 / 14 | **.95 / 14** |
| damping 0.1 baseline nas=2 | 0.55 | .55 / 19 | .35 / 32 | .10 / 32 | .10 / 32 | **.10 / 32** |
| damping 0.1 baseline nas=6 | 0.70 | .70 / 15 | .40 / 30 | .20 / 34 | .20 / 34 | **.20 / 34** |
| **R2 (0.5)** baseline nas=2 | 0.45 | .40 / 40 | **.00** / 64 | .00 / 101 | .00 / 120 | .00 / 123 |
| **R2 (0.5)** baseline nas=6 | 0.55 | .55 / 18 | .25 / 63 | **.00** / 99 | .00 / 117 | .00 / 126 |
| **R2 (0.5)** ln_act nas=2 | 0.10 | .10 / 68 | .10 / 68 | .05 / 68 | .00 / 67 | .00 / **67** |
| **R2 (0.5)** ln_act nas=6 | 0.25 | .30 / 30 | .15 / 57 | .05 / 106 | .05 / 116 | .05 / 118 |

**Findings (feeds §3 / red-team surface #3 "40 is a tuned number"):**

1. **"40" is not doing the work.** Under R2 (0.5) the frozen arm's settled SR is already **0
   by a 5-step hold** at nas=2 and by a 15-step hold at nas=6. Any hold ≥ 5 gives the same
   qualitative verdict. The choice of 40 is defensible as "long enough for the block to
   asymptote," not as a tuned threshold — a 5-step hold would have supported the identical claim.
2. **The settle-check is inert where successes are real.** R0's settled SR and settled distance
   are **byte-flat across hold 1 → 40** (0.65/0.95; 14–17 px). The R0 block stops dead
   (`damping ≈ 0`); the hold length is irrelevant. This is the R0 control, seen as a curve.
3. **damping 0.1 genuinely settles** — settled SR stabilises by hold ≈ 15 (0.10 / 0.20) and is
   flat 15 → 40, at ~32–34 px. A functional-but-degraded regime, distinct from both R0 (flat
   from step 1) and R2 0.5 (never stops).
4. **The chart's R2 nas=2 block is stopped.** Its settled distance is flat across every hold
   (68 → 67 px, hold 1 → 40) while the frozen R2 nas=2 block travels 40 → 123 px over the same
   window. The residual-momentum result (H6 / `DAY1_CADENCE` §2) re-expressed as a hold-length
   curve: the chart @ nas=2 is the only arm whose block does not keep moving during the hold.

---

## 1.G.5 — termination-timing table  ✅ RAN (free, local) — reproduces the plan's table exactly

For pass-through successes: the raw step (of 30) the per-step criterion fires at, and the
unused budget. Recomputed from `success_at_step` / `passthrough_success`.

| arm | n succ | fire step (mean) | budget unused | fire steps |
|---|---:|---:|---:|---|
| frozen, R2 (0.5), nas=2 | 9 | **8.56** | **21.44** | 4,4,5,6,8,10,11,12,17 |
| chart, R2 (0.5), nas=2 | 2 | 8.5 | 21.5 | 5,12 |
| frozen, R2 (0.5), nas=6 | 11 | 12.27 | 17.73 | 2,5,7,7,8,9,11,17,18,25,26 |
| chart, R2 (0.5), nas=6 | 5 | **21.0** | **9.0** | 4,17,27,27,30 |
| frozen, R0, nas=2 | 13 | 15.08 | 14.92 | — |
| frozen, R0, nas=6 | 19 | 17.89 | 12.11 | — |
| frozen, damping 0.1, nas=2 | 11 | 11.73 | 18.27 | — |
| frozen, damping 0.1, nas=6 | 14 | 15.43 | 14.57 | — |

**Verification:** `FINAL_FIVE_DAY_PLAN.md` §1.G.5 quotes frozen R2 nas=2 "8.6 / 21.4", frozen
R2 nas=6 "12.3 / 17.7", chart R2 nas=6 "21.0 / 9.0", frozen R0 nas=2 "15.1 / 14.9" — **all
four reproduce to 2 dp** from raw JSONL. The finding stands: the criterion halts control while
the block is still transiting, and halts the harder-shoving (frozen) arm earliest — frozen R2
leaves ~21 raw steps of glide budget unused, the chart ~9. Write as a §5 finding beside H6.

---

## 1.C — damping dose ladder  ✅ RAN (aiden-dsouza-201323, all 8 cells, 20 eps each)

Frozen `c₀`, R2, it=10, N=300, H=6, `--settle-steps 40`, seeds 0–19. damping 0 = the R0
settle arms; 0.1 / 0.5 already on disk; **0.05 / 0.2 / 0.3 new this session.** All 12
cells recomputed from raw JSONL (`scripts/day1_ladder_analysis.py` →
`phase0_v3/day1_ladder_analysis.json`). Archived under `phase0_v3/ladder_dmp{005,02,03}_baseline_nas{2,6}/`.

### The curve (nas=6 shown; nas=2 same shape)

| damping | pass-through SR | **settled SR** | divergence (pt−st) | settled dist median | coast median (H6) | contacts/ep |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (R0) | 0.95 | 0.95 | 0 pp | 14 px | 0.0 | 15.1 |
| 0.05 | 0.70 | 0.40 | 30 pp | 30 px | 2.7 | 8.3 |
| 0.1  | 0.70 | 0.20 | 50 pp | 34 px | 9.7 | 7.8 |
| 0.2  | 0.65 | 0.15 | 50 pp | 52 px | 15.0 | 7.1 |
| 0.3  | 0.55 | 0.00 | 55 pp | 70 px | 27.2 | 6.3 |
| 0.5  | 0.55 | 0.00 | 55 pp | 126 px | 62.9 | 6.0 |

nas=2: pass-through 0.65 → 0.60 → 0.55 → 0.50 → 0.50 → 0.45; settled 0.65 → 0.35 → 0.10 →
0.15 → 0.05 → 0.00; coast 0.0 → −0.1 → 0.8 → 6.4 → 14.2 → 54.8.

### Decision rule outcome (pre-registered §8.7 / plan 1.C) — **H1 divergence CONFIRMED (monotone at nas=6; monotone to ±1 episode at nas=2)**

> **Correction 2026-09-01 (REVISION-2 R2.0-b):** "monotone divergence" is exact at **nas=6**
> (`0 → 30 → 50 → 50 → 55 → 55 pp`). At **nas=2** the divergence dips once —
> `0 → 25 → 45 → 35 → 45 → 45 pp` — because settled SR ticks 0.10 → 0.15 (2/20 → 3/20) between
> damping 0.1 and 0.2. A one-episode wobble at n=20 = sampling noise, but Figure 1 plots both
> cadences with it visible rather than smoothed. The **continuous** metrics (settled distance
> median, coast) are monotone at *both* cadences.

- **Pass-through SR falls slowly and plateaus** (0.95 → 0.55 at nas=6, all of the drop by
  damping 0.2). **Settled SR falls fast and hits zero** (0.95 → 0 by damping 0.3). Divergence
  (pass-through − settled): monotone non-decreasing at nas=6 (0 → 55 pp); monotone-to-±1-episode
  at nas=2.
- The two SRs **do not fall together** at any intermediate damping → the criterion-validity
  finding holds across the entire non-degenerate range, not just at 0.5. This defeats
  *"you picked a degenerate regime."*
- **Coast / residual momentum (H6) is itself a clean dose-response**: 0.0 → 2.7 → 9.7 → 15.0 →
  27.2 → 62.9 px (nas=6). Six points; mechanically explains the SR divergence rather than
  merely correlating — as damping rises the block retains more post-push momentum, so a
  pass-through crossing increasingly fails to become a resting state.
- **No points dropped.** All six damping values reported.

## 1.B — R2 adapter (trained @ damping 0.5) screened at damping 0.1  ✅ RAN

Chart `p0g_onpolicy/chart_ln_act_R2.pt` vs frozen `c₀` at damping 0.1, paired on seeds 0–19
(0 pairing mismatches on `init_block_pos_diff`), settle-40. Frozen arm = existing
`c2_settle2_dmp01_baseline_nas{2,6}`.

> **🛑 CORRECTION 2026-09-01 (REVISION-2 R2.0-a; independently re-derived here).** The
> clean-subset row below originally read *"neither-settled-succeeded, n=16, Δ+20.1, p=0.025."*
> That used the wrong subset definition. The subset exists to remove the unequal-compute
> confound, and that confound is created by **pass-through** success (it `break`s the episode
> loop, `run_e0_planning.py:329-333`); settled success terminates nothing. The correct clean
> subset is **neither-pass-through-succeeded: n=8, Δ+24.4 px, p=0.109 (n.s.).** The 1.B claim
> therefore rests on the **all-20 test (p=0.044)** alone — the direction is consistent at both
> cadences, the second line of statistical support is not. Not silently overwritten (rule 5).

| cadence | chart pass-SR | frozen pass-SR | chart settled dist | frozen settled dist | paired Δ (all 20) | chart better | Wilcoxon p | clean subset (neither pass-through) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **nas=2** | 0.15 | 0.55 | 58.1 px | 41.2 px | **+16.9** | 7/20 | **0.044** | n=8: Δ+24.4 / **p=0.109 n.s.** |
| nas=6 | 0.35 | 0.70 | 44.6 px | 47.8 px | −3.2 | 8/20 | 0.84 n.s. | n=5: Δ−15.2 / p=0.81 n.s. |

**The chart does NOT transfer to the milder shift — the C-2 "less destructive" finding is
specific to damping 0.5.** At damping 0.5 the chart ends the block *significantly closer*
(V3-21: nas=2 Δ−59.8 px, p=0.0002). At damping 0.1 that advantage is **gone**: nas=6 is a
wash on settled distance, and **nas=2 reverses — the chart is worse on the all-20 test**
(Δ+16.9, p=0.044; the clean subset is n.s. at n=8), while frozen `c₀` at 0.1 is a functional
planner (pass-through
0.55, moves block toward goal). The pre-registered caveat fires and resolves toward
**"the adapter is off-distribution at 0.1"** *and* **"the effect is severity-specific"** —
both hold. Paper limitation: the C-2 result characterises the adapter only where the frozen
baseline is directionally broken (damping 0.5); at a shift the frozen planner can still handle,
the adapter provides no benefit and at closed-loop cadence actively hurts. 1.A (a
damping-0.1-native chart) is the test that could separate these; if it does not run, 1.B
stands as the transfer result.

---

## 1.D — N=50 replication on disjoint tasks 20–49  ✅ RAN → **REPLICATES at both cadences**

4 arms × 30 episodes, `--episode-start 20 --episodes 50` (tasks 20–49, disjoint from every
prior cell), frozen + `ln_act` chart, R2 damping 0.5, it=10/N=300/H=6, settle-40, on
`aiden-dsouza-201323`. Archived `phase0_v3/n50_{baseline,ln_act}_nas{2,6}_ep20-49/`.
0 pairing mismatches on `init_block_pos_diff`, every cell. Analysis:
`scripts/day1_n50_analysis.py` → `phase0_v3/day1_n50_analysis.json`.

**Primary metric (pre-registered §8.7): paired settled block-distance, chart − frozen.**

| cadence | task set | n | chart / frozen mean | paired Δ | Δ CI95 | chart better | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|
| **nas=2** | 0–19 *(V3-21)* | 20 | 77.6 / 137.3 | −59.8 | [−84.9, −35.8] | 17/20 | 0.0002 |
| **nas=2** | **20–49 (new)** | 30 | 87.6 / 114.8 | **−27.2** | [−50.9, −1.9] | 19/30 | **0.029** |
| **nas=2** | **merged n=50** | 50 | 83.6 / 123.8 | **−40.2** | **[−58.3, −21.7]** | 36/50 | **0.0001** |
| nas=6 | 0–19 *(V3-21)* | 20 | 111.1 / 138.9 | −27.8 | [−58.3, +4.5] | 14/20 | 0.064 n.s. |
| nas=6 | **20–49 (new)** | 30 | 108.3 / 140.6 | **−32.3** | [−58.6, −6.2] | 20/30 | **0.019** |
| nas=6 | **merged n=50** | 50 | 109.4 / 139.9 | **−30.5** | **[−50.6, −10.2]** | 34/50 | **0.0027** |

**Verdict (pre-registered decision rule):** the direction replicates on the fresh task set at
**both** cadences (Δ < 0, CI excludes 0) → **merge and report paired n=50**, disclosing the
two-launch structure. No STOP fired.

Two consequences beyond replication:
1. **Kills "n=20, same 20 tasks reused everywhere"** (red-team surface #2). The chart-closer
   effect holds on 30 tasks it was never evaluated on.
2. **Resolves the V3-21 nas=6 caveat.** V3-21 reported nas=6 as *not significant* ("the
   substrate's headline cadence") at n=20. With n=50 it **is** significant (Δ−30.5 px,
   p=0.003) — the n=20 null was underpowered, not a real cadence difference. The paper can
   now state the less-destructive effect at both cadences.

Effect magnitude is ~half on 20–49 (Δ−27 vs −60 at nas=2) — task-set variance is real; the
merged CI [−58, −22] is the honest interval. Pass-through SR still favours frozen on both sets
(chart 0.20 vs frozen 0.40, merged) — the metric-inversion finding is intact.

**Clean subset (neither-pass-through-succeeded, the unequal-compute-free cut):** merged n=50
nas=2 n=28 Δ−29.0 px **p=0.028** (holds); nas=6 n=23 Δ−23.0 px **p=0.111 (n.s.)**. So the
headline all-50 test is significant at both cadences; the confound-free subset is significant
at nas=2 (the pre-registered primary) and directional-but-n.s. at nas=6. Report both.

---

## 2.3 — screen-power table (free, subsampling)  ✅ RAN → feeds §7

**Probe:** run `n` paired closed-loop episodes (adapter vs frozen `c₀`, same seeds), one-sided
exact McNemar on pass-through task success (H1: adapter has fewer successes), α=0.05. This is
the "small paired closed-loop eval" §7 proposes as the practical screen for a
UMF-approved-but-control-harmful adapter. `scripts/day2_screen_power.py` →
`phase0_v3/day2_screen_power.json`. B=20 000 subsamples per cell.

| pair | full-n probe p | n=5 | n=10 | n=15 | n=20 |
|---|---:|---:|---:|---:|---:|
| **detection** — C-2 headline (chart 1/20 vs frozen 10/20) | 0.002 | 0.8% | **50%** | **99%** | **100%** |
| **detection** — C-2 relaunch (chart 2/20 vs frozen 9/20) | 0.008 | 0.2% | 18% | 79% | **100%** |
| **false-block** — frozen `c₀` vs frozen `c₀` (two launches, diff dates) | 0.875 | 0% | 0% | 0% | **0%** |

**Reading (for §7):** **~15–20 paired closed-loop episodes** catch the known-bad adapter with
79–100% power, while the false-positive rate on a genuine frozen-vs-frozen pair is **0 / 20 000
at every n**. A paired closed-loop screen at n≈15–20 is cheap, well-powered, and essentially
never fires on a null — exactly the tool the paper argues should replace/augment open-loop UMF
acceptance. (Probe is on the *deployed* pass-through metric — the one where the harm the C-2
diagnosis found actually shows; on settled distance the same adapter is not harmful, which is
the whole point of the metric-validity finding.)

---

## E — momentum negative control  ⛔ ABANDONED per its own 90-min gate — the shift mechanism is inert

**Goal:** a dynamics shift with **no residual momentum** — `PushTEnv(block_cog=...)` (T-block
centre-of-gravity offset) at `damping=0` — to test whether the pass-through/settled divergence
is *specific* to momentum. Pre-registered prediction: divergence ≈ 0.

**What was done:** added an additive `--block-cog "x,y"` flag to `run_e0_planning.py` +
`modal_e0_planning.py` (threads to `PushTEnv.__init__(block_cog=...)`; default None = env
unchanged). **Falsification, run:**
- BEFORE: `--block-cog "0,100"` → `error: unrecognized arguments`, **exit 2**. ✓
- AFTER (byte-identity): a no-flag run's per-episode record key-set is **identical** to archived
  `c2_settle2_R0_baseline_nas2` (0 new keys, 0 missing); `block_cog` field appears only when the
  flag is passed. ✓

**Why abandoned — the physics parameter itself does nothing.** Direct check (reset env with
`block_cog` ∈ {None, (0,110), (40,45)}, apply *identical* push actions for 25 steps): all three
produce **byte-identical block trajectories** (end pose 349.0, 231.0, −2.54 in every case).
Same for `shape ∈ {I,L,Z,square,small_tee}` — no trajectory change. pymunk's
`Body.center_of_gravity` setter is inert unless `Body.moment` is also recomputed about the new
CoG (confirmed: `cog=(0,110)` **with** `body.moment` bumped *does* move the block). Making it a
real, validated shift needs `moment_for_poly(..., offset=−cog)` surgery on the env + a
G4-style reality check — **more than the "~2 h additive" the plan budgeted, and E is explicitly
"upside, not a dependency."**

**This is the same failure class as the R1 mass bug** (`REGIME_DESIGN_REVIEW.md`): a physics
knob that is algebraically inert in this pymunk setup. **The `--block-cog` code change was
reverted** (`git checkout`; verified compiles). Logged in `FIXLOG.md` as a defect-of-record
(no ATLAS code changed).

**Consequence for the paper — E's negative control already exists, for free.** Item C's closed
form gives `D_∞ = v₀/ln(1/damping) → 0` as `damping → 0` (`ln(1/d) → ∞`). The **R0 cells
(damping = 0) confirm it exactly**: of **32/32** pass-through successes across `c2_settle2_R0_*`
(nas 2 + 6), **32 survive the 40-step settle**, and the coast over every episode's hold is
**0.00 px** (median, mean, and max). So the momentum-free case *is* measured — a dynamics
regime with no residual momentum shows **zero** pass-through/settled divergence — and the
mechanism is nailed analytically by C rather than by a bespoke `block_cog` run. One sentence in
§8 covers the loss of a *non-zero-damping shift* without momentum.

---

## A — controller family  ✅ RAN → **the CEM-iteration ladder is the clean result; the τ is secondary**

> **REWRITTEN 2026-09-01 after external verification.** The first write-up (a) framed the
> 12-controller Kendall τ as an independent "family measurement" when the 12 form two disjoint
> blocks on pass-through SR and the τ mostly re-expresses `N15`; and (b) stated the chart
> settled-distance range as "57–78 px, one exception" by **silently dropping `nas6 chart` at
> 111 px**. Both fixed here; numbers re-derived from raw JSONL.

8 new planning cells on `aiden-dsouza-201323` (no code change — `--iterations`,
`--objective-alpha` are live flags), R2 damping 0.5, nas=2, N=300, settle-40, seeds 0–19:
`--iterations ∈ {1,3,30} × --kind ∈ {baseline,ln_act}` + `--objective-alpha 0 × both kinds`.
Plus `it=10/nas=2` and `it=10/nas=6` on disk → 12 controllers. Analysis
`scripts/day2_controller_family.py` → JSON + `day2_fig_controller_family.png`.

### PRIMARY — the CEM-iteration ladder (frozen `c₀` only, nas=2, the *same* 20 paired tasks)

| CEM iterations | pass-through SR | mean settled dist | median progress (init−settled) | moved toward goal | contacts/ep |
|---:|---:|---:|---:|---:|---:|
| **1** | 0.35 | **36.1 px** | **+52 px** | **18/20** | 5.50 |
| 3 | 0.40 | 100.2 px | −3 px | 10/20 | 4.85 |
| 10 (deployed) | 0.45 | 137.3 px | −63 px | 6/20 | 4.00 |
| 30 | 0.40 | **158.5 px** | −66 px | **3/20** | 8.10 |

Paired **it1 vs it30**: settled Δ **−122.4 px, it1 closer 19/20, Wilcoxon p = 1.9e-6** (one-sided),
0 pairing mismatches.

**Pass-through SR is flat** (0.35 → 0.40 → 0.45 → 0.40, ±1 episode, non-monotone) **while settled
distance degrades 4.4× monotonically** and the block goes from moving *toward* the goal in 18/20
to *away* in 17/20. Figure: `phase0_v3/day2_fig_iteration_ladder.png` (§4 lead figure —
replaces the two-cluster τ scatter).

**It is not "less contact is better."** `it=1` has **5.5 contacts/ep — more than it=3 (4.85)
or it=10 (4.0)** — and the best outcome; `it=30` has **the most contacts (8.1) and the most
damage** (−66 px progress, block away 17/20). The driver is **optimisation pressure acting
through momentum**: harder-optimised plans produce harder, more committed shoves, which is
exactly what the coast model (§C) formalises. it=1 controls; it just doesn't optimise.

**This is the paper's thesis with no adapter, no group confound, no cross-cadence compute
asymmetry, one varying parameter: optimising the world model's own CEM objective harder
monotonically destroys real control, and the deployed threshold metric registers nothing.**
Stronger than the τ, and than the adapter comparison.

**Honest cost, to state in §4:** the best controller in the whole study is the frozen model at
`it=1`. §4 must say the adapter beats the **deployed** config (`it=10`), not the unoptimised one.

### SECONDARY — cross-metric Kendall τ over all 12 controllers

| | value |
|---|---|
| Kendall τ_b (pass-through-SR quality order vs settled-distance quality order) | **−0.512** |
| permutation p (H₀: τ = 0) | 0.025 |
| null 95% band | [−0.45, +0.45] |
| pairwise inversions | 48 / 66 |
| within frozen (n=6) | τ_b −0.36, p = 0.33 |
| within chart (n=6) | τ_b −0.41, p = 0.25 |

**Caveat:** the 12 controllers form **two disjoint 6-blocks** on pass-through SR — every frozen
`c₀` (0.35–0.55) outranks every `ln_act` chart (0.00–0.30) — so the overall τ largely
re-expresses `N15`. The **within-group** τ is the genuine family signal: same direction in both
groups, n=6 each, n.s.

**By settled distance the two families interleave:**
`it1 base 36 < {it1,it3,it30,α0,it10 chart: 57–78} < it3 base 100 < nas6 chart 111 <
{it10,nas6,α0,it30 base: 137–159}`. Chart range is **57–111 px** (`nas6 chart` = 111.1), and
`it3 baseline` and `nas6 chart` each cross into the other family's block. `--objective-alpha 0`
≈ `it=10` on every metric — the proprio cost term is not load-bearing.

**Provenance note:** `objective_alpha` is written to **no artifact** (not summary, not
per-episode) — same defect flagged for `p0c_it30`. Empirical bit-identity check: `fam_alpha0_*`
vs `c2_settle2_*_nas2` (it=10) agree on **9/20 (baseline), 12/20 (chart)** episodes — vs ~19/20
expected from launch noise alone — so the `--objective-alpha 0` flag did take effect.

---

## B — n=100 headline across three disjoint task sets  ✅ RAN → **REPLICATES; all three sets**

`ln_act` chart vs frozen `c₀`, R2 damping 0.5, nas=2, it=10/N=300/H=6, settle-40, on
`aiden-dsouza-201323`. New cells `phase0_v3/n100_{baseline,ln_act}_nas2_ep50-99/` (tasks 50–99,
disjoint from 0–19 and 20–49). 0 pairing mismatches every set. Pre-registered
`IMPLEMENTATION_PLAN_V3.md` §8.8. Analysis `scripts/day2_n100_analysis.py` →
`phase0_v3/day2_n100_analysis.json`.

**Primary metric: paired settled block-distance, Δ = chart − frozen (negative ⇒ chart closer).**

| task set | n | chart / frozen mean | **paired Δ** | Δ CI95 | chart closer | Wilcoxon p | clean subset (neither pass-through) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–19 | 20 | 77.6 / 137.3 | −59.8 | [−84.9, −35.8] | 17/20 | 0.0002 | n=11, Δ−47.9, p=0.024 |
| 20–49 | 30 | 87.6 / 114.8 | −27.2 | [−50.9, −1.9] | 19/30 | 0.029 | n=17, Δ−16.8, p=0.33 n.s. |
| **50–99 (new)** | 50 | 78.4 / 132.1 | **−53.7** | [−75.3, −31.9] | 35/50 | **0.0001** | n=28, Δ−40.9, **p=0.030** |
| **merged n=100** | 100 | 81.0 / 128.0 | **−47.0** | **[−61.2, −32.8]** | **71/100** | **p<0.0001** | n=56, Δ−34.9, **p=0.0022** |

**Verdict (pre-registered §8.8):** 50–99 replicates the direction — in fact more strongly than
20–49 (Δ−53.7 vs −27.2). **All three disjoint task sets: chart closer.** Merge and report
**paired n=100: Δ−47.0 px [−61, −33], p<0.0001, chart closer 71/100.** The confound-free clean
subset is significant in 2 of 3 sets (0–19, 50–99) and at merged n=100 (Δ−34.9, p=0.0022);
20–49's clean subset is n.s. at n=17.

**Pass-through SR still favours frozen** at every set (merged 0.39 vs 0.23, McNemar p=0.0012) —
the metric inversion is intact and now established at n=100. **"n=20" is dead.** §4's headline
is a 100-episode, three-task-set paired result.

---

## C — analytical coast (residual-momentum) model  ✅ RAN → H6 becomes a derivation, not a table

**The physics, verified at source.** `pusht_env.py`: `sim_hz=100`, `control_hz=10` → 1 raw env
step = 0.1 s (10 pymunk substeps of 0.01 s). `space.damping = self.damping` (`:436-437`), and
pymunk applies `v ← v·damping^dt` per substep. A body coasting from speed `v₀` therefore travels
a **closed-form** distance:

```
D(T) = v₀ · (1 − damping^T) / ln(1/damping)        D_∞ = v₀ / ln(1/damping)
```

`scripts/day2_coast_model.py` → `phase0_v3/day2_coast_model.json`, fig `day2_fig_coast.{png,pdf}`.
Read-only, no production path. `v₀` estimated from the `settled_trace` slope; measured coast =
`trace[40] − trace[1]`. All 10 ladder cells (damping {.05,.1,.2,.3,.5} × nas {2,6}), n=20 each.

### C-1 — per-episode coast vs closed form

> **Use the disjoint fit (below) in the paper.** The originally-headlined version fits `v₀` on
> hold-steps 1→5 and predicts the 1→40 coast — the fit window is ~24 % of the predicted
> quantity at damping 0.5 (circular). Removing the overlap makes the fit *better*.

| fit | window | pooled n | Pearson r | pred/meas median |
|---|---|---:|---:|---:|
| **DISJOINT (paper)** | `v₀` on steps 5→15, predict finite 15→40 coast — no shared data | **200** | **0.981** | **0.99** |
| per-cell disjoint | — | 20 each | **0.942–0.999** | 0.96–1.00 |
| overlapping (superseded) | `v₀` on 1→5, predict 1→40 | 200 | 0.91–0.99 | 0.96–1.00 |

**The closed form is exact, and cleaner once the circularity is removed.** Disjoint per-episode
predicted vs measured coast: pooled r = **0.981**, ratio **0.99**, per-cell r 0.942–0.999 across
two orders of magnitude of coast.

### C-2 — the model predicts the ladder

Predicted settled SR among pass-through crossings = fraction where
`trace + free-coast ≤ 20 px` (and rotation already < 20°). Measured vs predicted:

| damping | nas=2 measured / predicted | nas=6 measured / predicted |
|---:|---:|---:|
| 0.05 | 0.35 / **0.35** | 0.40 / **0.40** |
| 0.1  | 0.10 / **0.10** | 0.20 / 0.25 |
| 0.2  | 0.15 / **0.15** | 0.15 / **0.15** |
| 0.3  | 0.05 / 0.10 | 0.00 / **0.00** |
| 0.5  | 0.00 / **0.00** | 0.00 / **0.00** |

**8 of 10 points match to the episode; 2 are off by a single episode (1/20).** The entire
settled-SR collapse across the damping ladder, at both cadences, is reproduced by the simulator's
own velocity integrator with `v₀` read from the trace — no free parameters.

### Decision-rule outcome (pre-registered, plan R2.2 item C)

**Agreement.** §5 (H6) stops being a correlation table and becomes: *"the residual momentum that
defeats the pass-through metric is the closed-form free-coast integral of the environment's
`space.damping`; disjoint per-episode fit (`v₀` on hold-steps 5→15, predicting the 15→40 coast —
no shared data) gives pooled Pearson r 0.981, ratio 0.99, n=200, per-cell r 0.942–0.999; and
the model predicts the settled-SR ladder to within ±1 episode at all ten points across both
cadences."* The overlapping-window fit (`v₀` on 1→5, ~24 % of the predicted quantity) is kept
only as the superseded version.

---

## 1.G.2 / 1.G.3 — already resolved in the plan; not re-run

`FINAL_FIVE_DAY_PLAN.md` §1.G.2 (matched-compute subset, 3.00 vs 3.00 replans, p=0.0244) and
§1.G.3 (`plan_length` pinned at 6 in both cadences → nas=2 gets ~3× the CEM search) were both
verified at source in the main session 2026-08-31 and carry decision text in the plan. No
new computation needed; carried into the draft as written.

---

## Report — three buckets

**Ran and passed:**
- **1.C damping dose ladder** (6 new GPU cells + 4 on disk) — H1 monotone divergence confirmed;
  H6 coast is a 6-point dose-response. `phase0_v3/day1_ladder_analysis.json`.
- **1.B damping-0.1 transfer** (2 GPU cells) — chart does not transfer; C-2 effect is
  severity-specific to damping 0.5. nas=2 chart significantly *worse* at 0.1 (p=0.044).
- 1.G.1 settle-length sensitivity — new table, 4 findings, `phase0_v3/day1_free_analyses.json`.
- 1.G.5 termination timing — reproduces the plan's table exactly from raw JSONL.
- Day 0 pre-flight: CLI verified, §8.7 pre-registration written, migration to
  `aiden-dsouza-201323` completed + smoke-tested.

**Ran and failed / recovered:**
- First 1.C/1.B batch leaked to `pandereshubham` (concurrent session flipped the global Modal
  profile), paused at 63/160 episodes by that account's $30 cap (~$0.33 over). Partials
  salvaged, not used. Relaunched clean on `aiden-dsouza-201323` with `MODAL_PROFILE` pinned.

- **1.D N=50 replication** (4 GPU cells, tasks 20–49) — **REPLICATES both cadences**; merged
  n=50 nas=2 Δ−40.2 px p=0.0001, nas=6 Δ−30.5 px p=0.0027 (resolves V3-21's nas=6 null as
  underpowered). `phase0_v3/day1_n50_analysis.json`.

**Did not run:**
- **1.A** (damping-0.1 on-policy collection, ~$15) — needs explicit go-ahead. 1.B already
  showed the *existing* chart does not transfer to damping 0.1 (nas=2 significantly worse),
  so 1.A's remaining value is narrower: "can a 0.1-*native* chart recover the less-destructive
  behaviour a functional baseline?" — a nice-to-have, not load-bearing for the current paper.
- 1.G.4 full figure regeneration — Figure 1 (ladder) built; the rest is Day 3.

## Day-1 bottom line

Four results landed, all ledgered (L5), plus two independently-caught corrections applied:
- **H1 dose-response confirmed** — pass-through SR plateaus, settled SR → 0; divergence monotone
  at nas=6, monotone-to-±1-episode at nas=2 (R2.0-b); coast is a 6-point dose-response (H6).
  Figure 1 (both cadences).
- **n=100 headline (item B)** — three disjoint task sets (0–19, 20–49, 50–99), **all
  chart-closer**; merged paired **n=100 Δ−47.0 px [−61,−33], p<0.0001, 71/100**; pass-through
  inversion intact (one-sided McNemar p=0.0012, two-sided 0.0025). "n=20" is dead.
- **Controller family (item A)** — the **CEM-iteration ladder** (frozen only, one knob): pass-through
  SR flat 0.35→0.45, settled distance degrades **4.4× monotonically** (it1 36 → it30 159 px),
  paired it1 vs it30 **Δ−122 px, 19/20, p=1.9e-6**. The thesis with no adapter and no confound.
  Cross-metric τ_b = −0.51 is secondary (two-group artifact; within-group τ n.s. at n=6).
- **C-2 "less destructive" effect is severity-specific** to damping 0.5 — the chart does not
  transfer to 0.1; nas=2 worse on the all-20 test (p=0.044), clean subset n.s. (R2.0-a).
- **Screen (2.3, corrected):** the paired closed-loop screen discriminates helpful from harmful
  adapters **only** when its statistic is settled distance (4/4 correct, calibrated); on
  pass-through it flags all four cases (3 false alarms). Feeds §7.
- **Coast model (C):** H6 residual momentum is the closed-form free-coast integral of
  `space.damping` — **disjoint** per-episode fit (no circularity) pooled r **0.981**, ratio
  0.99, n=200; predicts the settled-SR ladder to ±1 episode at all 10 points. §5 is a derivation.

Spend: ~$5 on `aiden-dsouza-201323` (14 GPU cells). ~$0.33 stranded on `pandereshubham` (incident).

## EVIDENCE_LEDGER rows — ADDED this session (`EVIDENCE_LEDGER.md` §1)

- **`B3-dose-ladder`** (L5) — H1 dose-response, frozen c₀ settle-40, damping ∈ {0,.05,.1,.2,.3,.5}
  × nas ∈ {2,6}, n=20/cell. Full numbers in the ledger row.
- **`B2-transfer-01`** (L5) — chart(ln_act @0.5) vs frozen c₀ @ damping 0.1, paired n=20.
  nas=2 settled-dist Δ+16.9 px (chart worse), p=0.044.
- **`N12-n50`** (L5) — N=50 replication on disjoint tasks 20–49; merged n=50 nas=2 Δ−40.2 px
  p=0.0001, nas=6 Δ−30.5 px p=0.0027.
