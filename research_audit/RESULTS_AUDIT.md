# ATLAS — RESULTS_AUDIT

**Last updated: 2026-08-27, pass 2 — resumes exactly where pass 1's "What I did not get to" left off. Covers claim (j) [highest priority, N8's E1-closing oracle−random spread], claim (i) [N9, E2 Cell C tau-crossing fraction], the Step 3 directional-prediction check on E0's RQ0 rule, a partial Step 6 self-report sweep, and the CODE_AUDIT.md §6.1 hysteresis-inertness cross-check against N6/N7. Same methodology as pass 1: independently recomputed from raw files on disk using `D:/Shubham/DeepLearning/Atlas/atlas/.venv/Scripts/python`, `atlas/stats.py`'s own `paired_bootstrap` reused (not reimplemented), nothing in `atlas_out/` or any `.py` file touched. Pass 2 found one claim (N9's "0.000 chunks exceed tau" in Cell C) that does not hold exactly against the raw per-episode records — see §2(i) below — and one mechanism-attribution problem (N7's "post-hysteresis" framing) that CODE_AUDIT.md's own finding makes hard to sustain — see the new §7 below. Pass 1's material (claims (a)-(h), inventory, red flags) is preserved unchanged below; new content is appended, not merged in.**

This file is part of the ATLAS pre-submission research audit. See
`research_audit/CLAIMS_MATRIX.md` for the claim register (N1-N9, C1-C4, etc.)
this file's claim-labels refer to, `research_audit/EXPERIMENT_STATUS.md` for
what is implemented versus run, and `.claude/skills/research-audit/SKILL.md`
for the L0-L7 evidence-level rules that govern what may be written here.

Evidence levels used below: **L4** = raw per-unit records exist on disk.
**L5** = the reported summary statistic has been independently recomputed
from those raw records in this pass and matches. **L6** = the recomputed
result actually supports the claim's direction and magnitude at the stated
sample size under its stated criterion (a stronger bar than L5 — a number can
match exactly, L5, while the underlying sample is too small or too biased to
support the conclusion drawn from it, so L6 is judged separately in prose).

---

## 1. Inventory of `atlas_out/`

Full recursive listing (186 files, path | size bytes | mtime | line count for
`.jsonl` files) was captured in this pass. Summary by directory:

**Directories with raw per-episode JSONL records (L4 evidence available):**

| Directory | File(s) | Records | mtime (latest) |
|---|---|---|---|
| `e0_planning_n100` | `baseline_R2.jsonl`, `ln_act_R2.jsonl` | 100 + 100 | 2026-08-26 |
| `e0_planning_nas2` | `baseline_R2.jsonl`, `ln_act_R2.jsonl` | 20 + 20 | 2026-08-26 |
| `e0_planning_sweep_60` | `ln_act_R2.jsonl` (+2 shards) | 40 | 2026-08-26 |
| `e0_planning_sweep_100` | `ln_act_R2.jsonl` (+2 shards) | 40 | 2026-08-26 |
| `e0_planning` | `baseline_R1.jsonl` | 1 (+ `cem_diagnostics/*.json`, not per-episode) | 2026-08-24 |
| `e0_calib_fric2` | `baseline_R1.jsonl` | 15 | 2026-08-26 |
| `e0_v3_baseline_R0` | `baseline_R0.jsonl` | 20 | 2026-08-25 |
| `e0_v3_planning_dataset_baseline` | `baseline_R2.jsonl` | 20 | 2026-08-25 |
| `e0_v3_planning_dataset_ln_act` | `ln_act_R2.jsonl` | 20 | 2026-08-25 |
| `e0_v3_planning_hybrid_ln_act` | `ln_act_R2.jsonl` | 20 | 2026-08-25 |
| `e0_v4_planning_full` | `full_R2.jsonl` | 20 | 2026-08-25 |
| `e0_v4_planning_lora4` | `lora4_R2.jsonl` | 20 | 2026-08-25 |
| `e0_v5_planning_closed_loop` | `ln_act_R2.jsonl` | 20 | 2026-08-26 |
| `e1_smoke` | `episodes.jsonl` | 5 | 2026-08-24 |
| `e1_verify` | `episodes.jsonl`, `T1.md` | 1 | 2026-08-24 |
| `e2`, `e2_R1`, `e2_R2` | `e2_episodes.jsonl` | 1944 each | 2026-08-26 |
| `e2_R1_lora4` | `e2_episodes.jsonl` | 1944 | 2026-08-26 |
| `e2_R1_posthysteresis`, `e2_R1_lora4_posthysteresis`, `e2_R2_posthysteresis` | `e2_episodes.jsonl` | 1944 each | 2026-08-27 |
| `e2_R2_cellC_q1` | `e2_episodes.jsonl` | 246 | 2026-08-26 |
| `e2_confusion_matrix` | `e2_confusion_episodes.jsonl` | 720 | 2026-08-26 |
| `e2_smoke` | `e2_episodes.jsonl` | 6 | 2026-08-26 |
| `cost_ranking_R0`, `cost_ranking_R2`, `cost_ranking_R2_v2`, `cost_ranking_R2_converged` | `.json` (not `.jsonl`, but contain full per-candidate `costs`/`true_dist`/`contacts` arrays nested under `per_seed[i].results[kind]` — this counts as raw per-unit data) | 300 candidates x 10 seeds (R0, R2, R2_v2), x3 seeds x2 kinds (R2_converged) | 2026-08-26/27 |

**Directories/files with ONLY pre-aggregated summaries, no raw per-episode
records (L4 not available; anything computed from these is a claim, not
independently verifiable by this audit):**

- `atlas_out/e2_R2_cellB_q1/e2_summary.json` — **no `e2_episodes.jsonl` in this
  directory.** This is the diagnostic run cited for "3 charts committed in
  Cell B at q=1" (claim (i) in the audit task, not reached this pass — see
  bottom). Only the summary survives; the per-chunk decisions that produced
  the 3-commit count cannot be independently re-derived from this directory.
- `atlas_out/analysis_n100.json`, `atlas_out/umf_locality.json`,
  `atlas_out/e0_v6_R1_results.json` — derived analysis outputs, not raw.
- All `atlas_out/e0*/results.json` / `results.md` / `loss_*.json` /
  `val_loss_*.json` — training-loss summaries, not per-episode records (E0's
  per-episode evidence lives in the `e0_planning*`/`e0_v*_planning*`
  directories listed above, not in `e0*` training directories themselves).
- `.pt` chart checkpoint files — binary weights, not evidence about results.

**Confirmed absent (zero raw records of any kind):**

- **No `e3`, `e4`, or `e5` directory of any kind exists under `atlas_out/`.**
  `find atlas_out -maxdepth 1 -type d` in this pass returned 40 directories;
  none matches `e3*`, `e4*`, or `e5*`. This directly confirms
  `EXPERIMENT_STATUS.md`'s claim that E3/E4/E5 have never produced output.
- `e1_smoke/episodes.jsonl` (5 lines) and `e1_verify/episodes.jsonl` (1 line)
  are confirmed smoke-scale only — nowhere close to E1's specified 60
  episodes x 3 seeds (180 episodes). No other `e1*` directory exists.

---

## 2. Headline-number verification

Each row: claim as given in the audit task -> what was recomputed -> verdict.
Unless noted, pairing was checked directly by comparing `init_block_pos_diff`
(and where relevant `init_block_angle_diff`, `init_agent_block_dist`) between
arms at matching `episode` index, to float precision (threshold 1e-6).

### (a) E0 planning N=100 paired, R2, baseline vs ln_act — **MATCHES** (L5, and L6 for the null-result direction)

Source: `atlas_out/e0_planning_n100/baseline_R2.jsonl` (100 records) and
`ln_act_R2.jsonl` (100 records).

- Baseline: 44/100 = 44.0%. Chart: 43/100 = 43.0%. **Matches claimed 44.0%/43.0%.**
- Delta (chart − baseline) = **−1.0pp**. Matches.
- Paired bootstrap 95% CI (10,000 resamples, seed 0, via `atlas.stats.paired_bootstrap`) = **(−0.09, +0.07)** = [−9.0, +7.0] pp. **Matches exactly.**
- McNemar exact p (via `atlas.stats.mcnemar_paired`) = **1.0**. Matches.
- Discordant pairs: baseline-success-only = 9, chart-success-only = 8 (17 total discordant of 100).
- **Pairing check: 0/100 episodes mismatched** on `init_block_pos_diff`,
  `init_block_angle_diff`, `init_agent_block_dist` between the two arms at
  matching episode index. The pairing is genuine — this is a true paired
  design, not an unpaired comparison mislabeled as paired. This was the
  single highest-risk item in the whole audit and it checks out clean.

### (b) Within-arm Kendall tau(UMF, success) — **MATCHES** (L5), with a caveat worth flagging (see below)

Source: same two files as (a).

- Baseline: tau = **−0.4055** (recomputed), p = 2.45e-6 (< 1e-4 as claimed), **n=92**. Matches.
- Chart (`ln_act`): tau = **−0.4489**, p = 1.33e-7, **n=94**. Matches.
- Partial correlation (OLS-residualize `umf_mean` and `success` on
  `init_block_pos_diff` + `total_contacts`, then Kendall tau on residuals,
  recomputed independently with `statsmodels.api.OLS` + `scipy.stats.kendalltau`,
  not by importing the project's `analyze_n100.py` function):
  baseline partial tau = **−0.3579**, p = 4.36e-7 (claimed −0.358, 4.4e-7 — matches);
  chart partial tau = **−0.3736**, p = 9.62e-8 (claimed −0.374, 9.6e-8 — matches).

**Why n=92/94, not 100 (root cause found and confirmed):** in both files,
episodes with `"umf_per_replan": [null]` have `umf_mean = null`. This is 8
episodes in baseline (episodes 6,7,31,42,43,46,86,98) and 6 in chart
(7,31,41,43,86,89). This is the project's G6 gate behavior working as
designed: `atlas/score.py`'s motion gate returns `None` for low-motion
chunks to avoid a near-zero-denominator UMF blowup, and since
`num_act_stepped=6` means one replan covers the *entire* 30-step episode,
a single low-motion replan window nulls out the *whole* episode's UMF, not
just one chunk of it.

**Bias check on the drop (recomputed, not previously reported anywhere I
found): the dropped episodes are NOT a random subset.** Baseline: dropped
episodes have 100% success rate (8/8), mean `init_block_pos_diff` = 50.5px,
mean `total_contacts` = 2.8; kept episodes have 39.1% success rate, mean
`init_block_pos_diff` = 94.2px, mean `total_contacts` = 5.2. Chart: dropped
6/6 = 100% success, mean init diff 52.8px; kept 39.4% success, mean init
diff 93.1px. **The excluded episodes are exactly the easy, small-displacement,
low-motion, always-successful ones — small `init_block_pos_diff` produces
little block motion, which triggers the same low-motion gate that nulls
UMF.** This does not make the reported tau wrong (the exclusion criterion is
mechanical/pre-specified, not post-hoc cherry-picking, and it's the same G6
gate used everywhere else in the project), but it means the −0.4/−0.45
correlation is measured over the *harder* 92-94% of episodes, not the full
sample, and an honest write-up should say so. This is a documentation gap
(claim N2 in `CLAIMS_MATRIX.md` does not currently mention it), not a code
bug — filing it here rather than implying `CODE_AUDIT.md` territory, since
G6's behavior is intentional and documented in `CLAUDE.md` §4.

### (c) Training-size sweep (UMF and planning success at 20/60/100 trajectories) — **MATCHES** (L5)

Sources: `atlas_out/e0_v3_dataset/results.json` (20-traj UMF),
`atlas_out/e0_train_sweep_60/results.json` and `e0_train_sweep_100/results.json`
(60/100-traj UMF); `atlas_out/e0_planning_n100/baseline_R2.jsonl` (paired
baseline, first 40 episodes) vs `e0_planning_sweep_60/ln_act_R2.jsonl` and
`e0_planning_sweep_100/ln_act_R2.jsonl` (40 episodes each) for planning success.

- UMF: 20-traj = **0.33570** (from `e0_v3_dataset/results.json`, `eval_umf` field — this file's timestamp, 2026-08-26, is *after* the 2026-08-25 rollout fix, so it is not one of the invalidated pre-fix numbers). 60-traj = **0.30229**. 100-traj = **0.26776**. Matches claimed 0.336/0.302/0.268.
- **Held-out check, 60/100-traj arms: CONFIRMED genuinely held out.** Both
  `e0_train_sweep_60/e0_seed_manifest.json` and `e0_train_sweep_100/e0_seed_manifest.json`
  record explicit train/eval seed lists: train seeds run 1000-1472 (60-traj)
  or 1000-1792 (100-traj); eval seeds are the *same* 8-seed block, 11000-11056,
  in both files — a disjoint seed range from any training seed, and identical
  across both sweep sizes (so the UMF-vs-training-size comparison is evaluated
  on the same held-out episodes at every size, isolating the effect of
  training-set size alone).
- **Held-out check, 20-traj arm (`e0_v3_dataset`): CANNOT VERIFY independently.**
  This directory has **no `e0_seed_manifest.json`** — only `chart_ln_act_R2.pt`
  and `results.json`. `E0_RESULTS.md` line 318 itself flags this run's
  early-stopping step as "not recorded (no saved loss curve)," consistent
  with thinner record-keeping for this earlier run. I could not independently
  confirm the 20-traj eval set is disjoint from its training set the way I
  could for 60/100; I am relying on `E0_RESULTS.md`'s prose description
  ("same recipe as the existing 20-trajectory chart") rather than raw seed
  records. This is a real gap, not a red flag of wrongdoing — just missing
  provenance for one of the three sweep points.
- Planning success, recomputed via `atlas.stats.paired_bootstrap` +
  `mcnemar_paired` against the first 40 episodes of
  `e0_planning_n100/baseline_R2.jsonl` (pairing confirmed 0 mismatches both times):
  - 60-traj: SR = 16/40 = **40.0%**, delta vs baseline40 (16/40=40.0%) = **0.0pp**, CI **(−12.5, +12.5)**, McNemar p=1.0, discordant 3:3. Matches claimed 40.0%, CI [−12.5,+12.5].
  - 100-traj: SR = 17/40 = **42.5%**, delta = **+2.5pp**, CI **(−12.5, +17.5)**, McNemar p=1.0, discordant baseline-only=4, chart-only=5. Matches claimed 42.5%, CI [−12.5,+17.5].
  - N=100 point (43.0%, CI [−9,+7]) already verified in (a) above.
- **Direction and magnitude, L6-level assessment:** UMF falls monotonically
  and substantially (0.336→0.302→0.268, a ~20% relative drop across 5x the
  data) while every planning-success CI spans zero at every size. The
  dissociation claim (N4) is well-supported by this recomputation — this is
  the cleanest of the claims checked in this pass.

### (d) Cost-ranking diagnostic — **MATCHES** (L5) across every sub-number, including a sign-convention check

Sources: `atlas_out/cost_ranking_R0/cost_ranking_R0_seeds0-1-2-3-4-5-6-7-8-9.json`
(10 seeds x 300 candidates, baseline only) and
`atlas_out/cost_ranking_R2_v2/cost_ranking_R2_seeds0-1-2-3-4-5-6-7-8-9.json`
(10 seeds x 300 candidates, baseline + `ln_act`) — both contain full raw
per-candidate `costs`/`true_dist`/`contacts` arrays, recomputed from those
arrays directly, not from the `pooled`/`mean_of_per_seed_rhos` fields already
present in the files (those fields were used only as a cross-check and
matched my independent recomputation exactly).

- Per-seed mean Spearman rho: R0 baseline = **+0.5012**, 95% CI **[+0.2765,+0.7258]**
  (file's own precomputed CI matches: [0.2765,0.7258]). Matches claimed +0.501, [+0.277,+0.726].
- R2 baseline = **−0.0720**, CI **[−0.2428,+0.0989]**. Matches claimed −0.072, [−0.243,+0.099]. (Note: `cost_ranking_R2` (non-`_v2`) lacks a saved CI; `cost_ranking_R2_v2` is the complete artifact and is what the claim's CI traces to.)
- Pooled rho (n=3000, all candidates/seeds mixed): R0 = **0.7406** (claimed 0.741), R2 baseline = **0.2059** (claimed 0.206). Matches.
- **Regret** (true_dist of the argmin-cost candidate minus the batch's best true_dist, averaged over 10 seeds, recomputed candidate-by-candidate from the raw arrays): R0 baseline = **8.47px**, R2 baseline = **88.13px**, R2 `ln_act` = **92.25px**. Matches claimed 8.5/88.1/92.3px.
- **Top-10-by-cost vs batch mean:** recomputing `batch_mean − top10_mean` (this sign convention, not `top10_mean − batch_mean`, is what the claimed numbers use — verified by sign-matching after computing both ways): R0 = **+28.16px**, R2 baseline = **−15.74px**, R2 `ln_act` = **−8.46px**. Matches claimed +28.2/−15.7/−8.5px. (Positive = top-10-by-cost candidates are *closer* to goal than the batch average, i.e. cost ranking is doing useful work; negative = the cost-ranked "best" candidates are actually *worse* than a random draw from the batch — this is what happens under R2.)
- **Contact fraction:** 80.03% for all three (R0 baseline, R2 baseline, R2 `ln_act`), matching claimed 80.0% across all three, computed from the raw `contacts` arrays.
- n=10 seeds, 300 candidates/seed confirmed (`n_candidates: 300` field, 10 entries in `per_seed`, `pooled_n: 3000`).

### (e) Converged-CEM check — **MATCHES** for the baseline claim (L5/L6); one imprecision found in how the std-range is stated (self-report issue, see §6)

Source: `atlas_out/cost_ranking_R2_converged/cost_ranking_R2_seeds0-1-2_iterlast.json`
(3 seeds, `iterations: 30`, `capture_iteration: "last"`, 300 candidates/seed/kind, baseline + `ln_act`).

- Pooled rho, baseline: recomputed **−0.1787**, p=6.69e-8. Matches claimed −0.179, p≈7e-8.
- Per-seed median `true_dist` (baseline), recomputed directly from the raw
  `true_dist` arrays, vs. each seed's own `init_block_pos_diff` (from
  `init_state`/`goal_state`): seed 0: init 91.5px → median 119.7px (worse).
  Seed 1: init 45.8px → median 62.6px (worse). **Seed 2: init 158.2px →
  median 200.8px (worse)** — exactly matches the example given in the claim.
  **All 3/3 seeds land farther from goal than the episode started, confirmed.**
- Std of the 300 final baseline candidates' `true_dist`, recomputed: seed 0
  = 7.01px, seed 1 = 8.32px, seed 2 = 4.74px → range **[4.74, 8.32]**.
- **The claimed "3.8-8.3px" range is not the baseline-only range** (baseline's
  own range is 4.74-8.32). Recomputing std for all 6 (seed, kind) cells:
  baseline = {7.01, 8.32, 4.74}, `ln_act` = **{3.77, 17.86, 27.15}**. The claimed
  lower bound (3.8) comes from `ln_act` seed 0 (3.77), and the upper bound
  (8.3) from baseline seed 1 (8.32) — i.e. the range as stated mixes both
  kinds and **silently omits `ln_act`'s own seed-1 (17.86px) and seed-2
  (27.15px) values, which are 2-6x wider than the stated range.**
  `E0_RESULTS.md` itself hedges this correctly ("3.8-8.3px for *most*
  seed/kind pairs" — line 84) but `ATLAS_SUMMARY.md` drops the hedge and
  states the range as if unqualified (line 241-242: "std of the 300 final
  candidates' true outcome is 3.8-8.3px — a tight cluster, not noise").
  This is flagged in detail under §6 (self-report gap) below since it is a
  drift between two of the project's own documents, not a code bug.
- `ln_act` per-seed converged medians, recomputed: seed 0 = 89.8px (≈init),
  seed 1 = 54.4px (worse than init 45.8), seed 2 = 231.7px (worse than init,
  worse than baseline's own 200.8) — all match `E0_RESULTS.md`'s table exactly.

### (f) nas=2 closed-loop, N=20 — **MATCHES** (L5)

Source: `atlas_out/e0_planning_nas2/baseline_R2.jsonl` and `ln_act_R2.jsonl` (20+20 records; `replans: 3` confirmed in every record, consistent with the nas=2 closed-loop protocol).

- Baseline 8/20 = **40.0%**, chart 10/20 = **50.0%**, delta **+10.0pp**.
- Paired bootstrap CI = **(−0.10, +0.30)** = [−10.0,+30.0]pp. Matches.
- McNemar p = **0.625**. Matches.
- Discordant pairs: baseline-only = 1, chart-only = 3 (3:1 favoring chart). Matches.
- Pairing: 0/20 mismatches on `init_block_pos_diff`.
- **L6 assessment, stated plainly per the audit's own instruction: with 20
  paired episodes and only 4 discordant, this comparison contains exactly 4
  bits of information about which arm is better — the other 16 episodes
  agree regardless of arm and contribute nothing to the McNemar statistic.**
  A 3-vs-1 split among 4 coin flips is not distinguishable from chance
  (p=0.625 confirms this) and cannot support a directional claim at any
  reasonable confidence. The "flips positive" framing (N5) is a correct
  description of the point estimate but the claim's own CI and discordant
  count already show it is not evidence of anything at this N.

### (g) E2 3-chart confusion matrix — **MATCHES exactly** (L5), recomputed from raw records, not from the summary

Source: `atlas_out/e2_confusion_matrix/e2_confusion_episodes.jsonl` (720
records: 360 per router x {umf, sdyn}; each record has `regime`, `router`,
`selected`, `gated`).

- Recomputed confusion matrix (rows=true regime R0/R1/R2, cols=selected chart index) for `umf`, excluding `gated=true` records: **[[54,19,10],[23,50,22],[6,32,66]]**. Exact match to claimed matrix.
- `sdyn`: **[[61,20,2],[62,26,7],[65,23,16]]**. Exact match.
- Row sums: **[83, 95, 104]** — matches claimed "83-104 ungated decisions per row" exactly. (Of 360 records/router, 78 are gated, leaving 282 = 83+95+104 ungated, consistent with `gated_fraction` reported elsewhere in the project ≈ 0.217-0.3 depending on cell.)
- Accuracy = trace/total: `umf` = **0.60284** (claimed 0.603), `sdyn` = **0.36525** (claimed 0.365), chance = 1/3 = **0.3333** (claimed 0.333). All match.
- The `sdyn` confusion matrix's structure (column 0 dominates every row: 61, 62, 65) confirms the claim that S-dyn defaults to selecting `c0` (index 0) regardless of true regime — recomputed directly from the matrix, not asserted from the summary.

### (h) E2 post-hysteresis Cell B and R1 rows — **MOSTLY MATCHES**, with one number (the pre-fix baseline) **CANNOT VERIFY**

Sources: `atlas_out/e2_R2_posthysteresis/e2_summary.json`,
`e2_R1_posthysteresis/e2_summary.json`,
`e2_R1_lora4_posthysteresis/e2_summary.json` (each has a `routing_accuracy`
dict keyed by cell letter A-D; "Cell B" is the decisive cell per
`E2_RESULTS.md`). I did not re-derive these three summary numbers from the
per-episode `e2_episodes.jsonl` in this pass (ran out of budget before doing
so for `h`; the confusion-matrix recomputation in (g) above, which used the
equivalent raw per-episode data for a related run, gives me reasonable
confidence the summary-generation code is not broken, but this specific
number should be independently re-derived from `e2_R2_posthysteresis/e2_episodes.jsonl`
in a follow-up pass rather than trusted on this basis alone — flagging as
**L4, not L5,** for this specific sub-claim).

- `ln_act` x R2, post-fix: `e2_R2_posthysteresis` Cell B: umf = **0.8333**, sdyn = **0.5699**, delta = **+26.3pp**. Matches claimed 0.833/0.570/+26.3pp.
- `ln_act` x R1: `e2_R1_posthysteresis` Cell B: umf = **0.6420**, sdyn = **0.5432**. Matches claimed 0.642/0.543.
- `lora4` x R1: `e2_R1_lora4_posthysteresis` Cell B: umf = **0.6420**, sdyn = **0.4938**. Claimed 0.642/0.494 — **0.4938 rounds to 0.494, matches.**
- **The "pre-fix 0.880 vs 0.324 (+55.6pp)" number: CANNOT VERIFY from raw
  records.** I found a directory `atlas_out/e2_R2` (mtime 2026-08-26, i.e.
  *before* `e2_R2_posthysteresis`'s 2026-08-27) that looks like it should be
  the "pre-fix" run, but its own Cell B numbers are umf=**0.8280**,
  sdyn=**0.5753** — neither of which is 0.880/0.324. Per `E2_RESULTS.md`
  (lines 6-55), there were **two separate hysteresis-related fixes**, not
  one: an original run (producing the claimed 0.880/0.324, described in the
  document's older "Cell B (decisive) — PASSES" section), a "sequential
  hysteresis" fix on 2026-08-26 (which is what produced the `e2_R2`
  directory's 0.828/0.575 — an intermediate value, not the final one), and
  a further fix producing `e2_R2_posthysteresis`'s 0.833/0.570. **No
  directory in `atlas_out/` contains raw episode records reproducing
  0.880/0.324** — the original run's raw output appears to have been
  overwritten in place by the later reruns reusing the same output
  directory name. The 0.880/0.324 number is currently **L0** (asserted in
  `E2_RESULTS.md`/`ATLAS_SUMMARY.md` prose only) rather than L4/L5 — it
  cannot presently be checked, and the "+55.6pp margin, now roughly halved"
  framing rests on a number with no surviving raw backing. This is a
  provenance gap worth fixing going forward (keep dated output directories,
  never reuse a directory name for a rerun) but is not evidence the number
  is wrong — just that it cannot currently be audited.

---

## 3. Directional predictions — NOT REACHED this pass (see bottom)

## 4. Red flags found so far

- **Non-random exclusion in the UMF-success correlation (n=92/94), detailed
  in (b) above.** Structural (G6 gate), not a bug, but the excluded episodes
  are systematically the easy/successful/low-motion ones — worth stating
  explicitly wherever N2 is reported.
- **No mismatched headline number was found in any of (a) through (g), or in
  the three sub-numbers of (h) that could be checked.** Every number that
  could be recomputed from raw records in this pass matched the claim,
  including exact confusion-matrix cell counts, exact discordant-pair
  counts, and CI bounds to 3 significant figures. This is itself worth
  recording plainly: across roughly 60 individually-checked numeric claims
  spanning claims (a)-(h), zero were found to be fabricated or arithmetically
  wrong. The two issues found (the b-caveat and the e/h provenance gaps) are
  about *incomplete disclosure and lost provenance*, not incorrect arithmetic.
- **`atlas_out/e2_R2_cellB_q1/` has no `e2_episodes.jsonl`** — only a summary
  survives for the run that is cited as evidence the EXPAND mechanism fired
  (3 charts committed). This is L4-incomplete: the summary's "3 charts
  committed" figure cannot be independently re-derived from per-chunk
  decisions in this directory. (Not reached in depth this pass — flagged for
  follow-up under claim (i), not started.)
- **Directory-name reuse across reruns destroys provenance**, demonstrated
  concretely in (h) above: `e2_R2`/`e2_R2_posthysteresis` etc. are not
  reused-in-place (each rerun got a new directory name, which is why (h)'s
  post-fix numbers were verifiable) — but the *original* pre-any-fix Cell B
  run apparently was overwritten before being given its own preserved
  directory, which is why 0.880/0.324 is now unverifiable. This suggests the
  project's output-naming discipline improved partway through (compare: the
  `e0_pre_regime_fix_2026-08-22` directory *was* preserved with a dated name
  — so the practice exists, just wasn't applied to every run).
- Pairing was checked directly (not assumed) in (a), (c), and (f), and was
  clean (0 mismatches) every time — no evidence of a paired-in-name-only
  comparison anywhere checked in this pass.

## 5. What does not exist — CONFIRMED

- **E3, E4, E5: zero raw records of any kind.** `find atlas_out -maxdepth 1 -type d`
  lists 40 directories; none begins `e3`, `e4`, or `e5`. This directly confirms
  `EXPERIMENT_STATUS.md`'s claim and `CLAIMS_MATRIX.md` row G-1: nothing in
  this project has ever exercised the continual stream, the 7-arm ladder, or
  cross-policy diagnostics.
- **E1: smoke-scale only.** `e1_smoke/episodes.jsonl` = 5 lines,
  `e1_verify/episodes.jsonl` = 1 line. The specified 60 episodes x 3 seeds
  (180 episodes) does not exist anywhere on disk. Confirmed by direct file
  inspection, not inferred from documentation.

## 6. Self-report gap

One clear instance found and fully substantiated in this pass (detailed in
(e) above): **`ATLAS_SUMMARY.md` (lines 241-242) states the converged-CEM
candidate-spread range ("3.8-8.3px") as an unqualified fact about "the 300
final candidates," while its own source document, `E0_RESULTS.md` (line 84),
correctly hedges the identical number as holding "for most seed/kind pairs"
— and the raw data (recomputed in this pass) shows two of the six
seed/kind cells (`ln_act` seeds 1 and 2, R2) have stds of 17.9px and 27.2px,
2-6x outside the stated range.** This is a real instance of a summary
document silently dropping a hedge present in its own cited source, which is
exactly the failure mode this audit exists to catch — flagging it prominently
as requested. It does not overturn the underlying claim (baseline really
does converge confidently to a worse-than-start plan in 3/3 seeds, verified
independently above) but it does mean `ln_act`'s own convergence behavior
under R2 is less well-behaved / more variable than the summary implies, and
that nuance is currently invisible to a reader of `ATLAS_SUMMARY.md` alone.

A second, related gap (detailed in (h) above): the "+55.6pp pre-fix" Cell B
number that `E2_RESULTS.md` and `ATLAS_SUMMARY.md` both cite as the
historical baseline the post-fix result "roughly halved" from no longer has
surviving raw records anywhere in `atlas_out/` — it is currently an L0
assertion, not an L4/L5 one, though I found no reason to think it's
fabricated (an intermediate, still-elevated value of 0.828/0.575 does exist
on disk one fix-step later, which is at least directionally consistent).

I did not reach `HANDOFF.md` or `OPUS_REMAINING_TASKS.md` in this pass (see
below), so I cannot yet say whether either of those documents has a further
self-report gap of this kind — that check is unstarted, not clean.

---

## 7. Pass 2 (2026-08-27) — claim (j): N8, the E1-closing oracle−random spread

**Source located.** `HANDOFF.md` §7.1 (lines 211-229) cites "the existing 20
paired R2 planning episodes (`e0_v3_planning_dataset_baseline` vs
`e0_v3_planning_dataset_ln_act`)" — the same two 20-episode files already
pairing-verified clean in claim (f)'s sibling directories in pass 1 (0
mismatches on `init_block_pos_diff`). No separate script produces this table
anywhere in `scripts/` or `atlas/` (grepped both directories for
`SR_oracle`/`SR_random`/`chart_R1` — nothing matches outside `run_e2.py`);
the computation in `HANDOFF.md` §7.1 is the only record of it, done ad hoc
and, per the file's own §7 preamble, "recorded... nowhere else yet."

### (j-1) The `{c0, chart_R2}` row — MATCHES exactly (L5)

Recomputed directly: `sb` = `atlas_out/e0_v3_planning_dataset_baseline/baseline_R2.jsonl`
success array (mean 0.450, 9/20), `sc` = `atlas_out/e0_v3_planning_dataset_ln_act/ln_act_R2.jsonl`
success array (mean 0.500, 10/20). Oracle = per-episode `max(sb,sc)`, mean =
**50.0%**. Random (uniform draw over 2 charts) = per-episode `(sb+sc)/2`,
mean = **47.5%**. Spread = **+2.5pp**, matches claimed exactly. Bootstrap CI
via `atlas.stats.paired_bootstrap(oracle, random, n=10000, seed=0)` =
**(0.0, +7.5)pp** — matches claimed `[0.0, +7.5]` exactly, same seed
convention as pass 1's other bootstrap checks.

### (j-2) The `{c0, chart_R1, chart_R2}` row — number MATCHES, but `chart_R1` is a proxy, not chart_R1's real performance on R2 episodes. This should be disclosed and is not.

**No file anywhere in `atlas_out/` contains `chart_R1` (a chart trained on
regime R1) evaluated on the R2 planning episodes.** `find atlas_out
-maxdepth 1 -type d` (pass 1's own listing, re-checked) has no directory
combining an R1-trained chart with R2-regime planning episodes at N=20 or
any other size — `e0_calib_fric2` has a `chart` x R1 result, but on R1
episodes, not R2. I tested the hypothesis that the third library slot was
filled by **duplicating the baseline (`c0`) success array** as a stand-in
for `chart_R1` (a defensible modeling choice — a friction-tuned chart run
against a damping-shifted regime it was never fit to should behave close to
frozen — but a choice, not a measurement) and it reproduces the claimed
numbers **exactly**:

- Oracle over `{sb, sc, sb}` = `max(sb, sc, sb)` = same as the 2-chart
  oracle = **50.0%**. Matches claimed 50.0%.
- Random over 3 slots = mean of `(sb+sc+sb)/3` = **46.67%** → rounds to
  claimed **46.7%**. Matches.
- Spread = **+3.33pp** → rounds to claimed **+3.3pp**. Matches.
- Bootstrap CI (same method) = **(0.0, +10.0)pp**, matches claimed
  `[0.0, +10.0]` exactly.

This is an exact 4-for-4 match (oracle, random, spread, CI) under the
c0-duplicate hypothesis, which is strong evidence that is in fact how the
number was produced. **`HANDOFF.md` §7.1 presents this as a 3-chart library
result without disclosing that one of the three "charts" is not real
`chart_R1` planning data — it is the baseline arm counted twice.** This
doesn't make the number wrong under the stated (undisclosed) assumption, and
the assumption itself is a defensible upper-bound-ish placeholder for an
untested chart. But as written, a reader of `HANDOFF.md` §7.1 or
`CLAIMS_MATRIX.md` N8 would reasonably believe `chart_R1` was actually run
on these 20 R2 episodes and contributed real, independent data to the
3.3pp figure. It did not. This should be corrected in any paper draft that
cites the 3-chart row.

### (j-3) Why both CIs' lower bound is exactly 0.0 — a genuine structural floor, not a bug, and a much sparser result than "spread = 2.5-3.3pp" alone suggests

The task asked whether the exact-0.0 lower bounds (flagged in `CLAIMS_MATRIX.md`
N8) are suspicious. **They are not a bug — they are mathematically forced**,
and the reason is worth stating precisely because it also reveals how thin
this result is:

By construction, oracle success at any episode is `max` over the library's
per-chart successes, so oracle ≥ every individual chart, hence oracle ≥ the
mean-over-library (= random). So the per-episode difference
`d_i = oracle_i − random_i` satisfies `d_i ≥ 0` **at every single episode,
for any library, always** — there is no way to construct a negative `d_i`.
A bootstrap resample is a resample of these non-negative `d_i` values, so
every possible resample mean is also ≥ 0, and the 2.5th-percentile of 10,000
such non-negative resample means lands at exactly 0.0 whenever a
non-trivial fraction of episodes have `d_i = 0` exactly (which happens
whenever every chart in the library agrees at that episode, success or
failure). This is a structural floor effect of the oracle-vs-random
construction itself, not an artifact of `atlas.stats.paired_bootstrap` or of
small N — it would occur at any sample size with the same qualitative
episode-agreement pattern.

**Recomputed how sparse the underlying signal actually is: only 1 of the 20
episodes has `d_i > 0` at all.** Direct comparison of
`baseline_R2.jsonl` vs `ln_act_R2.jsonl` success bits: 19/20 episodes have
identical outcomes between the two arms (both succeed, or both fail); the
single exception is **episode 17**, where baseline fails and the chart
succeeds. Every reported number in this row — the 2.5pp spread, the 46.7%
random rate, the CI — is a restatement of that one episode. The "spread"
being below the project's own 10pp reporting threshold (`atlas/stats.py:35`,
`min_spread=0.10`) is not just a borderline miss; with 19/20 episodes tied,
the E1-closing argument in `HANDOFF.md` §7.1 rests on the outcome of a
single episode out of the whole dataset the closure decision is based on.
This strengthens rather than weakens N8's conclusion (E1's denominator
really is that thin) but the "2.5-3.3pp" framing on its own does not
communicate how little independent data supports it — worth stating
explicitly if N8 is cited in a paper.

**Verdict on (j): L5 for the arithmetic (all four headline numbers and both
CIs reproduce exactly), L0→flagged for the `chart_R1`-proxy disclosure gap
(not previously documented anywhere), and a new finding (episode-17
singularity) not previously stated in any project document.**

---

## 8. Pass 2 (2026-08-27) — claim (i): N9, E2 Cell C tau-crossing fraction and commit counts

Sources: `atlas_out/e2_R2_cellC_q1/e2_episodes.jsonl` (246 records: 120
condition A [`R0`, `none`] + 120 condition B [`R0`, `dark` — this is the
actual "Cell C" row] + 6 stray null-regime records), cross-checked against
the larger `atlas_out/e2_R2/e2_episodes.jsonl` (1944 records, includes a
`cell="C"` subset of 243 records per condition at higher N). Each record has
a `scores: [c0_umf, chart_umf]` pair (null/null when `gated=true`, per the
G6 motion gate).

### `atlas_out/e2_R2_cellB_q1/` — confirmed, again, no raw episodes file exists

Re-confirmed directly: this directory contains only `e2_summary.json`
(`charts_committed: {"B": 3}`, `routing_accuracy.umf.B: 0.828`) — no
`e2_episodes.jsonl`. The "3 charts committed in Cell B at q=1" figure
remains **L4-incomplete**, exactly as pass 1 flagged: it cannot be
independently re-derived from per-chunk decisions in this directory, because
those per-chunk decisions were never saved. This is not new information but
is reconfirmed rather than assumed.

### `e2_R2_cellC_q1` — commits = 0 confirmed at L4 (summary matches config), but the "0.000 chunks exceed τ" framing does NOT hold exactly against the raw per-chunk scores

`e2_summary.json` for this directory: `"charts_committed": {"C": 0}`,
`"routing_accuracy": {"umf": {"C": 0.7425}}`, `probe_q: 1`. The zero-commit
figure is consistent with what a fresh recount of the raw file would need to
show for the mechanism to have stayed quiet, and I have no raw commit-log to
contradict it (`expand.py`'s internal strike/probe bookkeeping isn't stored
per-chunk in this JSONL, only `scores`/`selected`/`gated`/`hit`, so the
commit count itself sits at the same L4 ceiling as Cell B — the summary
value is the only surviving record).

**What I could check directly: whether any individual ungated chunk's UMF
actually exceeds τ=0.5 under the Cell C (R0 + `dark`) condition.**
`E2_RESULTS.md` line 153 states this fraction is **0.000** for Cell C, used
to support the sentence "a 100%-of-pixels appearance shift raises UMF above
τ in zero chunks." Recomputed directly from the 120 condition-B (`R0`,
`dark`) records in `e2_R2_cellC_q1/e2_episodes.jsonl` (78 ungated after the
G6 filter): **c0's score (`scores[0]`) exceeds 0.5 in 1/78 ungated chunks
(1.28%)**; the chart's score (`scores[1]`) exceeds 0.5 in a different 1/78.
Cross-checked against the independent, larger `cell="C"`/`condition="B"`
subset of the main `e2_R2/e2_episodes.jsonl` (243 records, 156 ungated):
**2/156 (1.28%) ungated chunks exceed τ on at least one chart** — the same
rate, on a ~2x larger sample, confirming this is not a single-file fluke.
The two offending records there:

```
{seed:1, episode:21, scores:[0.2735, 0.6196], selected:0, correct:0, hit:1}
{seed:2, episode:19, scores:[0.5108, 0.4387], selected:1, correct:0, hit:0}
```

**So the claim "0.000 chunks exceed τ" in Cell C is not exact — the true
rate is ≈1.3%, not 0%, confirmed independently in two different raw files at
two different sample sizes.** This does not overturn the zero-commits
result (a single isolated tau-crossing chunk, non-consecutive with any
other, plausibly still fails the `q=1` strike-then-probe pipeline's
fixability check rather than never reaching it at all — I did not trace
`expand.py`'s exact strike-accumulation logic against these two specific
episodes to confirm this, so this explanation is plausible but **not
verified**, flagged as such). What the current wording gets wrong is the
mechanistic story: `E2_RESULTS.md`'s framing ("raises UMF above τ in zero
chunks") implies the probe/strike machinery is never even triggered under
appearance-only corruption. The raw data shows it *is* occasionally
triggered (≈1.3% of chunks) and the system still correctly declines to
commit — which, if the fixability-probe explanation above is right, is
actually a *stronger* demonstration that verification works (it fired and
correctly rejected) than the "never even fires" story the write-up
currently tells. Either way, "0.000" as a factual data point is wrong by
this audit's independent recount; "0 commits" (the number that actually
matters for RQ2's `charts committed = 0` criterion) is unaffected by this
finding and remains the best-supported summary number available (L4, not
independently re-derivable to L5 due to the missing per-chunk commit log).

**For contrast, the R0-uncorrupted condition A (120 records, 89 ungated) and
the Cell B / R2 condition (243 records, 216 ungated) were also recomputed:**
condition A gives 0/89 (0.0%) on `scores[1]` and 2/89 (2.2%) on `scores[0]`
— consistent with the table's stated "0.000-0.013" **range** for this row
(a range across the two chart slots, not a single number, which the table
already discloses correctly, unlike the Cell C row). Cell B (`R2`, the
decisive comparison) gives `scores[0]` (c0's own UMF, the relevant one for
whether the current default chart looks inadequate) > τ in **33/216 =
15.28%** of ungated chunks — closely matching the claimed **0.157 (15.7%)**,
within rounding/sample-composition tolerance of a slightly different N
(216 here vs whatever `e2_R2_cellB_q1`'s own now-lost raw file would have
given). This row is corroborated, not contradicted.

**Verdict on (i): L4 for the commit counts (both cells; Cell B's raw episode
file is confirmed absent, a pre-existing L4-incomplete gap, not new). The
"0.000" tau-crossing figure specifically for Cell C is contradicted by
direct recount at L5 — true value ≈1.28% (2/156 and 1/78 in two independent
files) — and this should be corrected wherever N9/RQ2 is written up, even
though the substantively important "0 commits" figure is not affected by
this specific correction.**

---

## 9. Pass 2 (2026-08-27) — Step 3: E0's pre-registered RQ0 rule vs. actual data

`ATLAS_proposal_v7.md` §7 / `ATLAS_implementation_plan_v2.md` §7.1
pre-register RQ0's decision rule as: use the smallest adapter class reaching
**≥90% of the full predictor's gain**, in both UMF reduction and planning
success, in both regimes (`CLAIMS_MATRIX.md` RQ0 row). This was **not**
independently re-derived from raw data in this pass beyond what pass 1
already established — E0_RESULTS.md's own text (self-checked directly,
not taken on faith) states the rule plainly became inapplicable:

- `E0_RESULTS.md:1191-1193`: "The pre-registered rule ('smallest kind
  reaching ≥90% of full's gain, in both regimes')... no positive 'full's
  gain' to measure 90% of in R1 or R2." I did not re-derive `full`'s planning
  numbers myself in this pass, but pass 1's claim (a)/(c)/(f) recomputations
  already establish the general pattern this rests on: every planning-success
  CI in the project spans zero (N1, N4, N5), so a "gain" for `full` in either
  regime failing to be reliably positive is consistent with everything
  already independently verified. **This is an instance of the project's own
  self-report being accurate, not a gap** — E0_RESULTS.md discloses the rule
  became undefined rather than silently substituting a different bar without
  saying so. (The *substitute* bar — an ad hoc 15pp threshold — is a
  separate, already-flagged issue: `OPUS_REMAINING_TASKS.md` #22 / S-6 in
  `CLAIMS_MATRIX.md`, not re-checked further in this pass.)

RQ2's Cell C "charts committed = 0" directional prediction: **holds**, per
§8 above — both `e2_R2_cellC_q1` and the larger `e2_R2` Cell C subset are
consistent with 0 commits at `q=1` (summary-level, L4). The τ-crossing-rate
correction in §8 does not change this verdict.

I did not reach a systematic check of E4's expected charts-committed pattern
(RQ3, "≈2 charts, the true regime count for S2") — this remains open, since
**no E4 output exists at all** (pass 1 §5, reconfirmed: no `e3*`/`e4*`/`e5*`
directory under `atlas_out/`), so there is nothing to check RQ3's numeric
prediction against; it is correctly PENDING/NOT RUN, not a discrepancy.

---

## 10. Pass 2 (2026-08-27) — CODE_AUDIT.md §6.1 hysteresis inertness: does it change how N6/N7 should be read?

**Yes, for N7; not materially, for N6.** `CODE_AUDIT.md` §6.1 (`router.py:94-101`)
establishes that the spread-normalised hysteresis margin `m=0.05` is
**mathematically inert for any 2-chart library**: if the current chart is
not the argmin, `relative_gap` is forced to exactly `1.0` (current chart is
by definition the max of a 2-element set when it isn't the min), which
always exceeds `m=0.05`, so the router always switches — hysteresis never
holds the router on its current chart in this configuration. I re-derived
this algebraically from the quoted code (`router.py:94-101`, reproduced in
`CODE_AUDIT.md`) rather than re-running it, and the logic is airtight: for
`K=2`, `relative_gap ∈ {0.0, 1.0}` exactly, with no intermediate values
possible, so the `< m` comparison can never produce a "hold" outcome except
when the current chart is already winning (a case where hysteresis was
never needed).

**N7** (`atlas_out/e2_R1_posthysteresis`, `e2_R2_posthysteresis`,
`e2_R1_lora4_posthysteresis`) are all 2-chart libraries (`{c0, chart}`,
confirmed by pass 1's claim (h) reading the same directories' `Cell B`
rows). By the above, **the hysteresis-margin fix could not have changed a
single routing decision in any of these three runs** — the mechanism was
provably a no-op there both before and after the margin-formula fix.
`E2_RESULTS.md`'s "pre-fix +55.6pp → post-fix +26.3pp, margin roughly
halved" framing (and `HANDOFF.md`'s implicit attribution of the change to
"the hysteresis fix") therefore cannot be explained by the margin change
itself for these specific runs. Pass 1's claim (h) already found the
pre-fix 0.880/0.324 raw data is unrecoverable and that **two separate
fixes** happened ("sequential hysteresis" — an architectural fix to how
`current_idx` is threaded/persisted across chunks, distinct from the
margin-formula change CODE_AUDIT.md's §6.1 examines) plus a further
unspecified fix producing the final posthysteresis numbers. Combining that
with this pass's algebraic result: **whatever actually moved N7's numbers
between the pre-fix run and `*_posthysteresis`, it was not the margin
threshold itself** (that part is proven inert at K=2) — it must have been
the "sequential" fix or something else. The label "posthysteresis" attached
to these three directories is therefore misleading as a causal story, even
though the numbers in them (already verified exactly in pass 1 §2(h)) are
not themselves in question. **This should be corrected in any write-up
that credits the margin-hysteresis fix specifically for N7's improvement.**

**N6** (`atlas_out/e2_confusion_matrix`, 3-chart library) is less directly
undermined: at `K=3`, `relative_gap` can take intermediate values (not just
{0,1}), so the margin mechanism is "non-trivial but still very permissive"
per `CODE_AUDIT.md` §6.1's own characterization — it is not proven inert
the way K=2 is. I did not independently re-derive how often the K=3
hysteresis condition actually holds in `e2_confusion_matrix`'s 720 records
in this pass (would require re-simulating per-chunk `relative_gap` values
chunk-by-chunk with a persisted `current_idx`, which the flat `scores`
field alone doesn't directly give without also tracking the router's state
across consecutive chunks within an episode — not done here, flagged as a
possible follow-up). N6's headline (60.3% vs chance 33% vs S-dyn 36.5%,
verified exactly in pass 1 §2(g)) is not contradicted by anything found in
this pass, but should not be read as having been re-validated against the
hysteresis-inertness question either — it is simply less exposed to the
specific K=2 argument that undermines N7.

---

## What I did not get to (updated after pass 2, 2026-08-27)

Pass 1 verified claims (a)-(h) (h partially — see above). **Pass 2 verified
claim (j) fully (§7), claim (i) fully (§8), the E0 RQ0/E2-Cell-C directional
checks from Step 3 (§9), and the CODE_AUDIT.md hysteresis cross-check
against N6/N7 (§10).** Remaining open items, genuinely unchecked:

1. **Step 3 — remaining directional predictions.** E1's GO/PIVOT threshold
   and RQ4/E4's expected charts-committed pattern (≈2, the true regime
   count for S2) were not checked — the latter is unfalsifiable right now
   since no E4 output exists at all (confirmed absent, pass 1 §5 and pass 2
   §9), so there is nothing to compare against; this is a correctly-PENDING
   item, not a discrepancy.
2. **Step 4 — red flags, systematic sweep.** Still not done as a dedicated
   pass over every directory (exact-zero variance across seeds, the
   360-paired-episodes-per-arm spec check against E4, further NaN sweeps
   beyond the umf_mean nulls in (b) and the τ-crossing recount in §8).
   Covered only opportunistically as a byproduct of claims (a)-(j).
3. **Step 6 — self-report gap, full sweep. Still incomplete.** Pass 2 read
   `HANDOFF.md` §7.1-§7.5 closely (for claim (j) and the gate-status/E0-power
   sections) and grepped `OPUS_REMAINING_TASKS.md` for done/confirmed
   markers, but did **not** do a full close read of either file end-to-end,
   nor of `E2_RESULTS.md` beyond the Cell B/C sections already used for
   claims (h)/(i)/(j). Specifically still unchecked: `OPUS_REMAINING_TASKS.md`
   item 1 ("knock-aways confirmed at N=20... never re-verified at N=100" —
   the item is marked DONE but its own text flags an internal gap that was
   not independently checked here); `CLAIMS_MATRIX.md` Section C's S-1
   through S-8 (rollout-bug-fix claim, full gate-passing status, the "E0
   closed" framing, S-8's two-parallel-sessions claim) remain **completely
   unverified by this results-focused audit** — better suited to
   `CODE_AUDIT.md` or a dedicated process-claims pass.
4. **Full inventory table for the `e0*` training-only directories** — still
   not individually characterized beyond pass 1's §1 summary (dozens of
   `e0_*_smoke`/`e0_pre_regime_fix_2026-08-22`/`e0_contact_check*`/etc.
   directories, believed training-loss-only/low-priority but not confirmed
   one by one).
5. **New from pass 2, not yet followed up:** whether `expand.py`'s
   strike/probe logic actually explains why the 2 tau-crossing chunks found
   in §8 didn't produce a Cell C commit (plausible via the fixability probe
   rejecting them, but not traced through the code in this pass — would
   need a `CODE_AUDIT.md`-style read of `expand.py`, not a results
   recomputation, so flagged for a different agent). Also unresolved: an
   exact re-simulation of the K=3 hysteresis condition in
   `e2_confusion_matrix` (§10) to see how often it actually holds, versus
   the algebraic K=2 inertness proof which is complete.

**Recommended next step for whoever resumes this:** the highest-leverage
remaining item is probably the `HANDOFF.md`/`OPUS_REMAINING_TASKS.md` full
close-read (item 3 above) — both files make several "done"/"confirmed"
assertions this audit has still only sampled, not swept.

---

## 11. Pass 3 (2026-08-27) — claim (j)/N8 SUPERSEDED: the missing `chart_R1`×R2 evaluation was actually run

Pass 1 (§7 above) found the 3-chart oracle-vs-random spread's `chart_R1` row
was a duplicated baseline array, not a real measurement. Rather than leaving
this as a documented gap, the coordinating session ran the missing
evaluation for real: `atlas_out/e0_chart_r1_on_r2/ln_act_R2.jsonl`, 20
episodes, `chart_ln_act_R1.pt` (from `e0_v6_R1`, the corrected/T9 training
pipeline — **not** the same run as `chart_R2`, which comes from
`e0_v3_dataset`; disclosed as a caveat, not a matched triple) applied to R2
planning episodes at the same seeds/init-goal pairs as the existing
`e0_v3_planning_dataset_{baseline,ln_act}` rows (confirmed by re-using
`run_e0_planning.py`'s deterministic per-episode-index seeding, identical to
how pass 1 verified pairing for claims (a)/(f)/(j)).

**Real per-episode results, n=20, verified directly from the downloaded
JSONL (not a summary file):**

```
episode:    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
baseline:   0  1  0  0  1  0  1  1  1  1  0  1  0  1  0  0  1  0  0  0
chart_R2:   0  1  0  0  1  0  1  1  1  1  0  1  0  1  0  0  1  1  0  0
chart_R1:   0  0  0  0  1  0  1  1  0  1  1  1  0  1  0  0  1  0  0  1
```

baseline SR = 45.0% (9/20), `chart_R2` SR = 50.0% (10/20), `chart_R1` SR =
45.0% (9/20).

Oracle (per-episode max of the three) = 12/20 = **60.0%**. Random (mean of
the three arms) = 28/60 = **46.67%**. **Spread = 13.33pp.** Bootstrap 95% CI
on the per-episode oracle-minus-random difference (10,000 resamples,
`np.random.RandomState(0)`, matching pass 1's bootstrap methodology): **[3.3,
25.0]**. This CI does **not** contain zero, and 13.3pp **clears** the
project's own `min_spread=0.10` (10pp) reporting threshold
(`atlas/stats.py:35`) — the opposite of pass 1's fabricated-data result
(2.5-3.3pp, CI touching 0.0).

**Where the real signal comes from, checked directly (not assumed):**
episodes 10 and 19 are where `chart_R1` is the *unique* winner — baseline
and `chart_R2` both fail, `chart_R1` succeeds. This is genuine discordant
information a duplicated-baseline row could never produce (a duplicate of
baseline is definitionally never a unique winner over baseline). Unlike pass
1's finding that the fabricated version's entire spread traced to one
episode, this real version has real, distributed discordance across
multiple episodes and multiple arms.

**Verdict: N8/RQ1's closure argument ("no routing algorithm can manufacture
benefit the library doesn't contain, so E1 isn't worth running") no longer
holds on real data.** A genuine, CI-excludes-zero denominator exists in this
3-chart library. This does not by itself prove a real E1 routing evaluation
would succeed — it only establishes that the argument used to avoid running
one is no longer sound. Whether to actually run E1 is a decision for the
team, not something this recomputation resolves on its own. See
`research_audit/CLAIMS_MATRIX.md` row N8/RQ1 for the corresponding
evidence-level update, and `research_audit/REDTEAM.md`'s N8/RQ1 sections for
the adversarial analysis this finding reopens (written against the old,
fabricated number — its attacks on the *arithmetic* no longer apply, but its
discussion of what a real E1 run would need is still relevant).

**Evidence level: L5** (independently recomputed from raw per-episode
records the coordinating session generated and downloaded directly from the
Modal volume, using this file's own established bootstrap methodology).

---

## 12. Pass 3 (2026-08-27) — N3/N3b re-measured at n=20 seeds/regime (double the original sample)

The original N3/N3b mechanism result (R0 vs R2 planner cost-ranking) was
measured at n=10 seeds/regime. The coordinating session ran 10 additional,
non-overlapping seeds (10-19) per regime on Modal (L4 GPU,
`scripts/diagnose_cem_costs.py` via `modal_e0_planning.py::diagnose_cem_costs`,
identical config to the original: `num_samples=300`, `capture_iteration=first`,
`charts_dir=e0_v3_dataset` for R2). Combined with the original seeds 0-9,
this gives n=20/regime, independently recomputed here from the three raw
per-seed JSON files (not from any summary), same methodology as claims
(d)/(e) above.

**R0 (no-shift control), baseline kind, n=20:**
mean per-seed Spearman rho = **0.532**, sd=0.329, 95% CI (normal approx) =
**[0.388, 0.676]**. Original n=10: 0.501, CI [0.277,0.726]. Point estimate
essentially unchanged; CI width roughly halved (0.45 -> 0.29).

**R2 (shifted regime), n=20:**
baseline mean per-seed rho = **0.0014**, CI [-0.132, 0.134]. `ln_act` chart
mean per-seed rho = **0.014**, CI [-0.115, 0.143]. Original n=10: baseline
-0.072 (CI [-0.243,0.099]), chart -0.051. The point estimate moved from
mildly negative to essentially exactly zero, and the CI is now tightly
centered on zero rather than merely including it near one edge.

**Assessment: this is a genuine strengthening, not a wash.** The core claim
("the planner ranks well absent a shift, and at chance level under this
specific shift, using the identical model and planner in both cases") is
unchanged, but is now supported by twice the data with visibly tighter
intervals in both regimes, and the R2 result specifically reads as *more*
convincingly "chance-level" (CI symmetric around zero) than the original
("slightly negative, could plausibly be zero or worse"). No red flags:
seeds are confirmed non-overlapping (0-19 covered exactly once each across
the three files), same `num_samples=300` and `capture_iteration=first`
config in all three files (checked directly, not assumed), same chart file
(`e0_v3_dataset`) used for the R2 chart arm throughout.

**Evidence level: L5** (recomputed directly from three raw per-seed JSON
files per regime, normal-approximation CI on the per-seed rho distribution,
matching this file's own established methodology for claims (d)/(e)).
