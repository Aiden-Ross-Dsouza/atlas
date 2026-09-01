# ATLAS — EVIDENCE LEDGER

**Last updated: 2026-08-27 — file created during Phase 0. Rows are seeded from
verification done during planning; the `Recomputed` column is completed as Phase 1
proceeds.**

## What this file is

The single source of truth mapping **each claim the paper makes** to **the raw file on
disk that backs it** and **the independently recomputed value**. Written for a future
Claude Code session with zero memory of the conversation that produced it.

**When any other document in this repository disagrees with this one about a number,
this one wins.** That rule exists because the repo holds ~470 KB of AI-authored markdown
against ~190 KB of Python, and a claim repeated across six documents reads as
corroborated when it has only ever been asserted once and copied forward.
`ATLAS_SUMMARY.md`, `E0_RESULTS.md`, `E2_RESULTS.md` and `HANDOFF.md` remain the
narrative record; this file is the arithmetic record.

## Evidence levels

Per `.claude/skills/research-audit/SKILL.md`: **L0** asserted · **L1** specified ·
**L2** code exists · **L3** code runs correctly · **L4** raw results exist ·
**L5** statistic independently recomputed · **L6** claim actually supported ·
**L7** survives adversarial attack.

**A claim may not be quoted in the paper above the level this table gives it.**

## Standing rules for maintaining this file

1. A row is L5 only if this session or a later one **re-derived the number from the raw
   per-unit records** — not from a summary JSON, and not from a prior document's prose.
2. If the backing file does not exist locally, the row is **L0/L1 regardless of how
   confidently any document states it.** Not hypothetical: three headline results were
   archived only after their artifacts were found missing locally on 2026-08-27.
3. Every new experiment writes a seed manifest and per-episode records **from the
   start**, and downloads its artifacts immediately. That archive gap must not recur.
4. **Never reuse an output directory name for a re-run.** The pre-fix E2 Cell B numbers
   (row N7-pre) are permanently unverifiable because a re-run overwrote them in place.
5. **Whenever a "corrected" number is reported against a "before" number, state explicitly
   whether the seed set / sample was held constant.** Twice in G7 alone a "corrected vs
   before" comparison silently changed two things at once (contamination-guard + `m`
   comparison; window-fix + an added seed). One sentence — "same N seeds, same accept
   decisions, only X changed" — catches it before it needs a second pass.

---

## Section 1 — Claims with local raw backing

| Claim | Statement | Backing file(s) | Recomputed | Level |
|---|---|---|---|---|
| **N1** | baseline 44.0% vs `ln_act` 43.0%, N=100 paired, CI [-9.0,+7.0], McNemar p=1.000 | `atlas_out/e0_planning_n100/{baseline,ln_act}_R2.jsonl` | 44/100, 43/100; pairing 0/100 mismatches | L5 |
| **N2** | within-arm Kendall tau -0.406 (baseline, n=92), -0.449 (chart, n=94); partial -0.358 / -0.374 | same as N1 | matches | L5 |
| **N3** | R0 rho 0.532 CI [0.388,0.676]; R2 rho 0.001 CI [-0.132,0.134], n=20/regime | `atlas_out/cost_ranking_R0/`, `cost_ranking_R2_v2/` (seeds 0-9 + 10-19) | **pending** — seeds 10-19 archived 2026-08-27; only 0-9 recomputed before that | L4 → L5 pending |
| **N3-dose** | rho falls 0.532 / 0.295 / 0.169 / 0.078 / 0.001 across damping 0 → 0.5, n=20/point | `atlas_out/cost_ranking_dose_{0125,025,0375}/` | **pending** — archived 2026-08-27 | L4 → L5 pending |
| **N3b** | converged CEM under R2 lands farther from goal than episode start, 3/3 seeds | `atlas_out/cost_ranking_R2_converged/` | matches — **but see Section 3 #2** | L5 |
| **N4** | UMF 0.336 / 0.302 / 0.268 at 20/60/100 training trajectories; every planning CI spans zero | `atlas_out/e0_v3_dataset/`, `e0_train_sweep_{60,100}/`, `e0_planning_sweep_{60,100}/` | UMF matches; SR 40.0% / 42.5%; **pairing 0 mismatches across 120 field comparisons per sweep** (verified 2026-08-27) | L5 |
| **N5** | nas=2 closed-loop, N=20: 40.0% vs 50.0%, +10.0pp, CI [-10,+30], p=0.625 | `atlas_out/e0_planning_nas2/` | matches; 4 discordant pairs of 20 | L5 |
| **N6** | 3-chart routing: UMF 60.3% vs S-dyn 36.5% vs chance 33% | `atlas_out/e2_confusion_matrix/e2_confusion_episodes.jsonl` | confusion matrices match exactly | L5 |
| **N7** | Cell B post-fix: UMF 0.833 vs S-dyn 0.570 | `atlas_out/e2_R2_posthysteresis/` | summary matches; **not yet re-derived from the 1944-record episodes JSONL** | L4 |
| **N8** | 3-chart oracle 60.0% vs random 46.7%, spread 13.3pp | `atlas_out/e0_chart_r1_on_r2/ln_act_R2.jsonl` + both `e0_v3_planning_dataset_*` arms | **pending** — archived 2026-08-27 | L4 → L5 pending |
| **N9** | 3 charts committed under dynamics shift, 0 under appearance shift | `atlas_out/e2_R2_cellB_q1/e2_summary.json` (summary only) | commit count **not re-derivable** — per-chunk log never written. Tau-crossing rate recounted as ~1.28%, not the stated 0.000 | L4 |
| **N10** | R1 regime: baseline 70.0% vs chart 60.0%, N=40, CI [-27.5,+7.5] | `atlas_out/e1_baseline_vs_chart_R1/{baseline,ln_act}_R1.jsonl` | matches | L5 |
| **C2-head** | C-2 headline: `ln_act`×R2 on-policy chart vs frozen c₀, it=10/nas=2, N=20 paired, `damping=0.5`: **chart 1/20 vs frozen 10/20**, ΔSR −0.45 CI [−0.65,−0.25], McNemar p=0.0039; discordant 0 (chart-only) / 9 (frozen-only). | `phase0_v3/c2_p0g_R2/ln_act_R2.jsonl` (chart) + `phase0_v3/p0c/p0c_it10_baseline_R2.jsonl` (frozen) | recomputed 2026-08-30 (`scripts/c2_threshold_sweep.py`): chart 1/20, frozen 10/20, pairing exact (init fields 1e-6); reproduces `c2_screen_summary.json`. **Arms are separate launches** — see C2-settle non-determinism note; aggregate + effect size robust, per-episode pairing at nas=2 is not (item 1.3 pending). | L5 (aggregate) / L4 (per-episode pairing) |
| **C2-var** | C-2 outcome-distribution: chart/frozen final-block-distance **sd-ratio** CI (paired bootstrap, n=20000) **excludes 1** in every pool — headline (it=10) 0.46 [0.28,0.76]; nas=6 0.51 [0.32,0.75]; nas=2-pool (n=80, it∈{1,3,10}+α0) 0.64 [0.48,0.80]. Means equal (~58 px both). SR-vs-position-radius curves cross at ~30 px at nas=6; at nas=2 the chart never reaches frozen in [10,60] px. | `phase0_v3/c2_{p0g_R2,nas6_*,dose_it{1,3}_*,alpha0_*}/`, `phase0_v3/c2_threshold_sweep.json` | recomputed this session from raw JSONLs; all 5 cell counts reproduce `FABLE5_VALIDATION.md` §1.1 exactly | L5 |
| **C2-settle** | **1.1 settle-check — pass-through success is not a valid task metric under R2.** Holding position 15 raw steps after each pass-through success: settled SR = **0/20 in all 4 arms** (baseline/ln_act × nas2/nas6); 0 of 27 pass-through successes survive; min settled dist 23.4 px. Pre-registered decision rule (§8.6) **fires** (frozen settled SR ≥2 below pass-through). **CORRECTED 2026-08-31 (Opus review, re-derived here):** the follow-on "0/20 vs 0/20 → no dissociation" claim was a **floor-effect error** (both arms zero at 20 px ⇒ that comparison has no power). On settled *distance* the dissociation **reverses** — chart ends the block closer: nas=6 all-20 chart 64.0 vs frozen 101.5 px, Δ−37.5, 15/20, Wilcoxon p=0.011; nas=6 neither-succeeded subset (no drift-substitution confound, n=9) chart 69.2 vs frozen 131.5, 8/9, **p=0.0078**. nas=2 weaker (all-20 p=0.033 on cross-launch pairing this session found unreliable; clean subset n.s., p=0.28). Reading: under R2 neither arm solves the task, but the adapter is less destructive and the threshold metric rewards the opposite. **RE-RUN 1.1-R (2026-08-31, FIXLOG V3-21):** clean measurement — `--settle-steps 40` on *every* episode + `settled_trace`. Falsification PASSED (11/11 nas=6 successes reproduce archived `settled_block_pos_diff` exactly at trace step 15). R0 control PASSED (baseline R0 nas=6: pass-through 19/20, settled 19/20 — real successes survive). Settled block-distance paired Δ: **nas=2 chart 77.6 vs frozen 137.3 px, Δ−59.8 [−84.9,−36.0], 17/20, p=0.0002** (neither-succeeded n=11: Δ−47.9, 9/11, p=0.024); **nas=6 chart 111.1 vs frozen 138.9, Δ−27.8 [−58.5,+4.5], 14/20, p=0.064 n.s.** (subset p=0.074 n.s.). **This re-inverts the settle1 read**: nas=2 significant, nas=6 (the substrate's headline cadence) not — `settled_trace` shows the nas=6 chart block is still gliding at episode end (parallel drift curves), while the nas=2 chart block genuinely stops. Progress (init−settled): nas=2 frozen median **−63 px** (shoves block away, 14/20), nas=2 chart **0 px** (near-inert, +12 px on the clean subset). Defensible claim: **the adapter is *less destructive*; the pass-through metric rewards aggressive shoves that transit the goal for one step. Not "better control"; not significant at nas=6.** | `phase0_v3/c2_settle{,2}_*/`, `c2_settle_analysis.json`, `c2_settle2_analysis.json` | recomputed from raw JSONL; settle branch verified vs `pusht_env.py` (zero action = hold; `step()` doesn't reset `space.damping`); Opus's settle1 stats reproduce exactly; two this-session nas=2 launches agree 0–1/20 per episode (cross-date `p0c` disagreement is image/silicon, not within-session). **Caveats:** settle count not swept beyond the trace checkpoints; nas=6 n.s. | L5 (nas=2 settled-distance Δ, R0 control, falsification) / L4 (nas=6, "less destructive" interpretation) |
| **C2-anchor** | **Frozen R0 anchor at the headline protocol (it=10, nas=2, settle-40).** Frozen baseline: R0 **13/20 pass-through, 13/20 settled** (all 13 successes survive the hold, settled dist 4.7–17 px) vs R2 9/20 pass-through, **0/20 settled**. **Regime deficit at protocol: pass-through 65%→45% (−20pp); settled 65%→0% (−65pp).** The frozen planner genuinely solves 65% of R0 tasks and 0% of R2 tasks. R0 cadence effect large (nas=2 65% vs nas=6 95%); R2's small (45 vs 55). Settle-check metric validated at nas=2, not only nas=6. | `phase0_v3/c2_settle2_R0_baseline_{nas2,nas6}/`, `phase0_v3/c2_settle2_{baseline}_nas{2,6}/` (R2) | recomputed this session from raw JSONL | L5 |
| **C1-rank** | **C-1 cost-ranking / elite-set (R2-chart vs frozen c₀, both on R2, 20 seeds × 300 CEM candidates).** iter-0: chart elite-10 true-dist **67.56 px** vs frozen **89.36 px**; mean per-seed Spearman ρ(cost, true_dist) **+0.2759** (chart) vs **+0.0014** (frozen). Converged (iter-last): chart elite-10 **56.82 px** vs frozen **123.82 px**, chart better in **19/20** seeds; mean per-seed ρ +0.191 vs +0.007. The chart improves CEM candidate ranking at every level — the "mean statistic improves" half of the C-2 dissociation. | `phase0_v3/cost_ranking_p0g_R2/*.json` (iter0), `phase0_v3/cost_ranking_p0g_R2_iterlast/*.json` (converged); `charts_dir=/atlas_root/phase0_v3/p0g_onpolicy` (on-policy P0-G chart) | recomputed this session from the raw per-candidate `costs`/`true_dist` arrays; reproduces `FABLE5_VALIDATION.md` §1.1 exactly | L5 |
| **C2-route** | **1.4 R0 UMF crosscheck — the UMF router is a well-behaved regime gate.** R2-trained P0-G chart vs frozen c₀, paired per T=2 window: on **R0** (unshifted) chart UMF 0.904 vs c₀ 0.254, Δ+0.650 (+256%), chart beats c₀ in **0/580** windows ⇒ argmin-UMF router never applies the chart on R0. On **R2** chart 0.486 vs 0.627, Δ−0.141, 474/580 (82%) ⇒ router applies it. The false-positive-route concern (`FABLE5_VALIDATION.md` §3.3) is answered negatively: argmin-UMF would not switch to the chart on R0. **INTERPRETATION RESOLVED (1.9 R0-chart control, 2026-08-31):** an R0-fit adapter (identical recipe) improves R0 UMF by 8.5% pooled (test −4.1%; eval_loss 0.059 vs the R2 chart's 0.472 — R0 ≈ c₀'s pretraining distribution, little headroom). So the R2-chart's 3.6× worse R0 UMF is **active negative transfer**, not absence of a training signal — it makes R0 UMF far worse than the ~0.25 no-effect baseline. **Writable paper sentence:** the R2 adapter genuinely reduces R2 prediction error 23%, the router correctly selects it on R2 / rejects it on R0; the routing mechanism is sound and the C-2 failure is specifically *in-distribution prediction error ≠ control competence*. | `phase0_v3/r0_umf_crosscheck.json` (R2-chart on R0), `phase0_v3/r0_chart_offline_umf.json` (R0-chart on R0), `phase0_v3/c2_widened_offline_umf.json` (R2-chart on R2) | run this session, `scripts/c2_widen_offline_umf.py` unmodified; R0 `trajs_R0.pt` re-pulled from volume (local copy corrupt), loads clean. Forward-only, local. | L5 |
| **B2-damping-sweep** | **`damping=0.5` is over-severe; `0.1` is a graded shift.** Frozen `c₀`, it=10, settle-40, seeds 0–19. Settled SR: R0 95% → damping 0.1 ~20% (2/20 nas2, 4/20 nas6) → damping 0.5 **0%**. At 0.1 the planner moves the block **toward** goal in 16/20 (both cadences), settled dist 41–48 px, block comes to rest (coast ≈ 0). At 0.5 it moves the block **away** in 14–16/20, settled dist ~138 px, 55–63 px residual momentum. **Qualitative break, not a gradient** — at 0.5 the substrate planner is directionally wrong. Every R2 result on disk is measured at 0.5. Whether the prediction/control dissociation holds against a *functional* baseline (0.1) is untested (chart never trained/evaluated there). §1.10 decision rule → case (c) intermediate; scientific choice escalated to user. | `phase0_v3/c2_settle2_dmp01_baseline_nas{2,6}/`, `phase0_v3/c2_settle2_baseline_nas{2,6}/` (0.5), `phase0_v3/c2_settle2_R0_baseline_nas{2,6}/` | recomputed this session from raw JSONL | L5 |
| **B2-contact** | On-policy P0-G contact rate: **R2 4.50/traj (median 4.0)** vs **R0 17.07 (median 17.0)** — a −74% collapse, present on-policy (not just the replay-confounded 38.5→13.3). §15-2's pre-registered `damping=0.1` fallback is measured-triggered; deliberately not applied (FABLE5 retired the experiments it was tuning for). Paper limitations must disclose. | `phase0_v3/p0g_onpolicy/trajs_R2.pt`, `phase0_v3/_r0_redownload/trajs_R0.pt` (`n_contacts` field, n=100 train each) | recomputed this session direct from the traj blobs | L5 (numbers) / decision recorded in `IMPLEMENTATION_PLAN_V3.md` §4 |
| **B3-dose-ladder** | **H1 dose-response (FINAL_FIVE_DAY Day 1.C, 2026-09-01).** Frozen `c₀`, R2, it=10, N=300, H=6, settle-40, seeds 0–19, n=20/cell. damping ∈ {0(R0), .05, .1, .2, .3, .5} × nas ∈ {2,6}. **nas=6:** pass-through SR 0.95→0.70→0.70→0.65→0.55→0.55; **settled SR 0.95→0.40→0.20→0.15→0.00→0.00**; divergence (pt−st) 0→30→50→50→55→55 pp (**monotone**); median coast 0.0→2.7→9.7→15.0→27.2→62.9 px (monotone — H6). **nas=2:** settled SR 0.65→0.35→0.10→**0.15**→0.05→0.00, divergence 0→25→45→**35**→45→45 (dips once at damping 0.2, 2/20→3/20 — sampling noise at n=20); settled-dist median + coast both monotone. **Corrected 2026-09-01 (R2.0-b):** verdict is "divergence monotone at nas=6; monotone to ±1 episode at nas=2; continuous metrics monotone at both." Pre-registered decision rule (SRs never fall together, do not drop points): **CONFIRMED**. | `phase0_v3/ladder_dmp{005,02,03}_baseline_nas{2,6}/`, `c2_settle2_dmp01_baseline_nas{2,6}/`, `c2_settle2_baseline_nas{2,6}/`, `c2_settle2_R0_baseline_nas{2,6}/`; `scripts/day1_ladder_analysis.py` → `phase0_v3/day1_ladder_analysis.json`; fig `phase0_v3/day1_fig_ladder.{png,pdf}` | recomputed from raw JSONL this session (0.05/0.2/0.3 cells new on aiden-dsouza-201323; rest re-derived from disk) | L5 |
| **B2-transfer-01** | **The C-2 "less destructive" adapter effect is severity-specific to damping 0.5 (Day 1.B, 2026-09-01).** `p0g_onpolicy` `ln_act` chart (trained @ damping 0.5) vs frozen `c₀`, both @ **damping 0.1**, paired seeds 0–19 (0 mismatches on `init_block_pos_diff`), settle-40. **nas=2: chart worse on the all-20 test** — settled block-dist 58.1 vs 41.2 px, paired Δ **+16.9**, Wilcoxon **p=0.044**. **Clean subset (neither pass-through succeeded — the correct unequal-compute-free cut; corrected 2026-09-01 R2.0-a from an earlier settled-success-based n=16/p=0.025): n=8, Δ+24.4, p=0.109 (n.s.).** The claim rests on the all-20 test alone. Pass-through SR 0.15 (chart) vs 0.55 (frozen). **nas=6: wash** — settled dist 44.6 vs 47.8, Δ−3.2, p=0.84 n.s. (clean subset n=5, Δ−15.2, n.s.). Contrast: @ damping 0.5 the same chart ends the block Δ−59.8 px closer (p=0.0002, V3-21). Interpretation: adapter is off-distribution at 0.1 AND the effect is severity-specific — both. A 0.1-native chart (Day 1.A) is the separating test; if not run, this is the transfer result. | `phase0_v3/dmp01_transfer_ln_act_nas{2,6}/`, `phase0_v3/c2_settle2_dmp01_baseline_nas{2,6}/`; `scripts/day1_ladder_analysis.py` | recomputed from raw JSONL this session | L5 |
| **N12-n50** | **N=50 replication of the settled-distance effect on disjoint tasks (Day 1.D, 2026-09-01).** `ln_act` chart vs frozen `c₀`, R2 damping 0.5, it=10, settle-40; tasks 20–49 (n=30, new) merged with 0–19 (n=20, V3-21). 0 pairing mismatches. **nas=2:** 20–49 Δ−27.2 px [−50.9,−1.9] p=0.029; **merged n=50 Δ−40.2 [−58.3,−21.7] p=0.0001** (36/50 chart closer). **nas=6:** 20–49 Δ−32.3 p=0.019; **merged n=50 Δ−30.5 [−50.6,−10.2] p=0.0027** — resolves V3-21's nas=6 "n.s." as underpowered (n=20), not a cadence effect. Clean subset (neither pass-through succeeded, R2.0-a convention): merged nas=2 n=28 Δ−29.0 **p=0.028** (holds); nas=6 n=23 Δ−23.0 **p=0.111 n.s.** — the all-50 headline is significant both cadences, the confound-free subset only at nas=2 (the pre-registered primary). Pre-registered replication rule → direction replicates both cadences → merged n=50 reported. Kills "n=20, same tasks". Pass-through SR still favours frozen (0.20 vs 0.40 merged) — metric inversion intact. | `phase0_v3/n50_{baseline,ln_act}_nas{2,6}_ep20-49/`, `c2_settle2_{baseline,ln_act}_nas{2,6}/`; `scripts/day1_n50_analysis.py` → `phase0_v3/day1_n50_analysis.json` | recomputed from raw JSONL this session; bootstrap CI n=20000 | L5 |
| **N13-screen-power** | **Acceptance-screen DISCRIMINATION (plan 2.3; SUPERSEDED-AND-REPLACED 2026-09-01 same day — see the correction note below this table) — feeds §7.** The question is not "how many episodes detect a difference" but **"does the screen flag the adapters that are actually harmful and leave the others alone."** 2x2: two candidate screen statistics x four adapter/cadence cases whose ground truth is fixed on the settle-validated metric. Both screens one-sided ("chart worse"), alpha=0.05, 4 000 subsamples, pairing asserted (0 mismatches). **Flag rate at n=20 (n=15 in brackets):** (a) HELPFUL ln_act@0.5 nas=2 (settled D **-59.8 px**, chart better -> must NOT flag): pass-through **100% [79.9%] FALSE ALARM** / settled-dist **0.0% [0.0%] correct**; (b) HELPFUL ln_act@0.5 nas=6 (D -27.8, better): pass-through **100% [52.2%] FALSE ALARM** / settled **0.0% correct**; (c) HARMFUL ln_act@0.1 nas=2 (D **+16.9 px**, chart worse -> SHOULD flag): pass-through 100% [67.2%] / settled **100% [53.2%] correct detection**; (d) ln_act@0.1 nas=6 (D -3.2, wash/better): pass-through **100% [44.2%] FALSE ALARM** / settled **0.0% correct**. **Result: the pass-through screen flags all 4 cases at 100% (3 of them false alarms) — zero discriminative power. The settled-distance screen flags exactly the 1 harmful case and none of the 3 others — 4/4 correct.** **Calibration (exact paired null, random sign-flip per pair, the null a symmetric paired test assumes):** settled-dist Wilcoxon **3.1/4.7/5.4/4.9% at n=5/10/15/20 ~ alpha=0.05, correctly calibrated**; pass-through McNemar **0.0/0.5/1.9/0.7%, badly under-calibrated** (discrete exact test, too few discordants to reach alpha). So the pass-through screen is simultaneously conservative under the null AND fires on everything in practice — the signature of a test whose discordance is driven by something other than the quantity of interest. **Paper sentence:** a paired closed-loop screen of ~15-20 episodes is a usable acceptance gate **only when its statistic is the settle-validated outcome**; the same probe built on threshold-crossing success cannot tell a helpful adapter from a harmful one. | `phase0_v3/{c2_settle2_ln_act_nas{2,6},c2_settle2_baseline_nas{2,6},dmp01_transfer_ln_act_nas{2,6},c2_settle2_dmp01_baseline_nas{2,6}}/`; `scripts/day2_screen_power.py` -> `phase0_v3/day2_screen_power.json` | recomputed from raw JSONL this session; the n=5/10/15/20 McNemar row of the superseded version was additionally verified in closed form against the hypergeometric (0.81/50.00/99.19/100.00%) — its **arithmetic was exact; only its construct was wrong** | L5 |
| **N14-coast-model** | **H6 residual momentum is the closed-form free-coast integral of `space.damping` (plan R2.2 item C, 2026-09-01).** Physics verified at source (`pusht_env.py`: sim_hz=100, control_hz=10 → 0.1 s/raw-step; `v ← v·damping^dt` per substep). Closed form: `D_∞ = v₀/ln(1/damping)`. Tested per-episode on all 10 ladder cells (damping {.05,.1,.2,.3,.5} × nas {2,6}, n=20), `v₀` from `settled_trace` slope, measured coast = trace[40]−trace[1]. **C-1 — PRIMARY, fully-disjoint fit** (external verification 2026-09-01): `v₀` fitted on hold-steps **5→15**, predicting the finite **15→40** coast — fit and predicted windows share **no data**. **Pooled n=200: Pearson r = 0.981, pred/meas median = 0.99**; per-cell r **0.942–0.999**. The model is *cleaner* once the circularity is removed. **Superseded overlapping version** (`per_cell`/`scatter` in the JSON): `v₀` from steps 1→5 predicting the 1→40 coast — the fit window is ~24% of the predicted quantity at damping 0.5; r 0.91–0.99, ratio 0.96–1.00; kept for the record, not for the paper. **C-2:** predicted settled-SR (crossing survives iff trace + free-coast ≤ 20 px & angle < 20°, `v₀` from the 5→15 slope) matches measured to **±1 episode at all 10 points** both cadences (8/10 exact). No free parameters. Pre-registered outcome: **agreement** → §5/H6 is a derivation, not a correlation table. **Momentum negative control (subsumes retired item E):** the closed form gives `D_∞ → 0` as `damping → 0`; the **R0 cells confirm it** — **32/32** pass-through successes across `c2_settle2_R0_*` (nas 2+6) survive the 40-step settle, coast **0.00 px** (median/mean/max). A dynamics regime with no residual momentum ⇒ zero pass-through/settled divergence. | `phase0_v3/{ladder_dmp*,c2_settle2_*,c2_settle2_R0_*}` `settled_trace`; `scripts/day2_coast_model.py` → `phase0_v3/day2_coast_model.json`; fig `phase0_v3/day2_fig_coast.{png,pdf}` | computed this session from raw JSONL; physics constants read from `pusht_env.py` | L5 |
| **N15-n100** | **n=100 headline, three disjoint task sets (plan R2.2 item B, 2026-09-01) — supersedes `N12-n50` as the §4 headline.** `ln_act` chart vs frozen `c₀`, R2 damping 0.5, nas=2, settle-40; tasks 0–19 (n=20) + 20–49 (n=30) + **50–99 (n=50, new)**. 0 pairing mismatches. Paired settled block-distance Δ (chart−frozen): **0–19 −59.8 p=0.0002; 20–49 −27.2 p=0.029; 50–99 −53.7 [−75.3,−31.9] p=0.0001; merged n=100 Δ−47.0 [−61.2,−32.8] p<0.0001, chart closer 71/100.** Clean subset (neither pass-through, R2.0-a): sig. in 0–19 (p=0.024) and 50–99 (p=0.030) and merged n=100 (n=56, Δ−34.9, p=0.0022); 20–49 n.s. (n=17). Pre-registered replication rule (§8.8): **all 3 sets chart-closer → merge, report paired n=100.** Pass-through SR still favours frozen every set (merged 0.39 vs 0.23, **one-sided** McNemar p=0.0012; two-sided p=0.0025) — inversion intact at n=100. | `phase0_v3/{c2_settle2_*_nas2,n50_*_nas2_ep20-49,n100_*_nas2_ep50-99}/`; `scripts/day2_n100_analysis.py` → `phase0_v3/day2_n100_analysis.json` | recomputed from raw JSONL this session; bootstrap CI n=20000 | L5 |
| **N16-controller-family** | **REWRITTEN 2026-09-01 (external verification: the original framed a two-group artifact as an independent family measurement and mis-stated a range by dropping a data point).** **PRIMARY — the CEM-iteration ladder (frozen `c₀` only, nas=2, damping 0.5, settle-40, the *same* 20 paired tasks, 0 pairing mismatches):** it ∈ {1,3,10,30} → pass-through SR **0.35 / 0.40 / 0.45 / 0.40** (flat, ±1 episode, non-monotone) while mean settled block-distance **36.1 / 100.2 / 137.3 / 158.5 px** (monotone, **4.4×**), median progress (init−settled) **+52 / −3 / −63 / −66 px**, block moved toward goal **18/20 → 10 → 6 → 3**. Paired it1 vs it30: settled Δ **−122.4 px, it1 closer 19/20, Wilcoxon p=1.9e-6** (one-sided). it=1 is **not** inert — 5.5 contacts/ep (more than it=3 or it=10); it controls, it does not optimise. **Reading: optimising the world model's own CEM objective harder monotonically destroys real control while the deployed threshold metric registers nothing** — the thesis with no adapter, no group confound, no compute asymmetry, one varying parameter. **Honest cost:** the best controller in the study is the frozen model at it=1; §4 must state the adapter beats the *deployed* config (it=10), not the unoptimised one. **SECONDARY — cross-metric Kendall τ over all 12 controllers:** τ_b = **−0.512** (perm p=0.025, null band [−0.45,+0.45], 48/66 pairs inverted). **Caveat:** the 12 form two disjoint 6-blocks on pass-through SR (all frozen 0.35–0.55 > all chart 0.00–0.30), so the overall τ largely re-expresses `N15`. The genuine within-family signal is **τ_b −0.36 (frozen only, p=0.33) / −0.41 (chart only, p=0.25)** — same direction, n=6, underpowered. By settled distance the two families **interleave**: `it1 base 36 < {it1,it3,it30,α0,it10 chart 57–78} < it3 base 100 < nas6 chart 111 < {it10,nas6,α0,it30 base 137–159}` — **chart range is 57–111 px** (not 57–78; `nas6 chart` at 111 was dropped in the original), and `it3 baseline` and `nas6 chart` each cross into the other family's block. `α=0` ≈ `it=10` on every metric. **Provenance note:** `objective_alpha` is written to NO artifact (not summary, not per-episode) — same defect flagged for `p0c_it30`. Bit-identity check: `fam_alpha0_*` vs `c2_settle2_*_nas2` at it=10 agree on **9/20 (baseline) / 12/20 (chart)** episodes, vs ~19/20 expected from launch noise alone → the `--objective-alpha 0` flag *did* take effect (8–11/20 episodes changed). | `phase0_v3/{fam_it{1,3,30}_*_nas2,fam_alpha0_*_nas2,c2_settle2_*_nas{2,6}}/`; `scripts/day2_controller_family.py` → `phase0_v3/day2_controller_family.json`; fig `phase0_v3/day2_fig_controller_family.{png,pdf}` | recomputed from raw JSONL this session; τ_b tie-corrected on raw value arrays; permutation CI n=20000; paired ladder Wilcoxon one-sided | L5 |
| **N11** | E-B (i): localized (top-k moving-token) UMF also fails to rank charts by planning competence (global/top16/SR): baseline 0.367/0.238/45%, `ln_act` 0.336/0.204/50%, `lora4` 0.329/**0.168 (best)**/40% (worse). **New this pass:** within-episode CEM cost-vs-true-distance rank correlation is ~0 and *chart-invariant* — mean per-seed Spearman rho (raw candidate batch, 20 seeds×300 candidates, R2): baseline 0.0014±0.296, `ln_act` 0.0140±0.287 (indistinguishable, both ≈ noise); mean per-candidate `\|rank(cost)-rank(true_dist))\|` 99.76 vs 99.35 of 300 (≈ the ~100 expected under two independent random permutations). Pooled-across-seed rho (≈0.25-0.27, both kinds) is driven by across-episode goal-difficulty variance, not real per-candidate discrimination — do not quote the pooled number as within-episode signal. Same null pattern at the CEM's converged/last-iteration candidate batch (3 seeds): baseline -0.006±0.149, `ln_act` -0.006±0.091. | `atlas_out/umf_locality.json`; `atlas_out/cost_ranking_R2_v2/*.json` (20 seeds, raw `costs`/`true_dist`/`contacts` arrays); `atlas_out/cost_ranking_R2_converged/*.json` (3 seeds) | recomputed independently from raw per-candidate arrays (not the stored summary `spearman_rho`, which was cross-checked and matches to 1e-9); script archived at Phase 4 Step 4a | L5 |


### Correction note — `N13-screen-power`, 2026-09-01 (same day as first entry)

The first version of this row reported a **power** table: one-sided exact McNemar on
**pass-through success**, "detection" of the C-2 adapter (1/20 vs 10/20) at 50/99/100% for
n=10/15/20, and 0/20 000 false-block on frozen-vs-frozen.

**Its arithmetic was exact** — the C-2 row was re-derived in closed form from the
hypergeometric this session (0.81 / 50.00 / 99.19 / 100.00%) and matches to 2 dp.
**Its construct was wrong, in two ways:**

1. **It built the recommended screen on the criterion the paper invalidates.** §3 spends the
   paper demolishing threshold-crossing success; §7 cannot then recommend it as the acceptance
   gate.
2. **It labelled the C-2 adapter "known-bad".** On the paper's own settle-validated metric that
   adapter is **better** (`N12-n50`: settled distance -40.2 px, n=50, p=0.0001). So the
   "detection" column was measuring a **false-alarm rate**, not detection.

Additionally the frozen-vs-frozen null was uninformative: two runs of an identical config have
near-zero discordance by construction, so 0% reflects the test having nothing to fire on rather
than correct calibration. Replaced by a sign-flip permutation null, which is the exact null a
symmetric paired test assumes.

**The pass-through row is retained in the replacement, relabelled as the false-alarm result — it
is not deleted, because it *is* the finding.** This is the third instance in this project of an
**arithmetically correct, wrong-construct** number (the settle-1 floor-effect error; the
`B2-transfer-01` settled-vs-pass-through subset; this). All three passed every numerical check
and failed on "what does this quantity actually measure." Standing pre-report question, added
here: **"if this number moved, what would have had to change in the world?"**

## Section 2 — Claims with NO local raw backing

| Claim | Status |
|---|---|
| **N7-pre** | The "pre-fix +55.6pp" Cell B figure (UMF 0.880 vs S-dyn 0.324) has **no surviving raw records anywhere** — a re-run overwrote the directory in place. **L0.** Drop the before/after framing entirely; report only current numbers. |

## Section 3 — Artifact/document disagreements found 2026-08-27

| # | Disagreement | Resolution |
|---|---|---|
| 1 | `ATLAS_SUMMARY.md` §3 says the LoRA count was corrected to 118,176 and that "every comparison in this project now uses 118,176". Both `atlas_out/e0_v4_lora4/results.json` and `e0_v6_R1/results.json` record `params=10292640`. | **The artifact is wrong, not the paper.** `Chart.n_params()` sums `_params`, which at construction holds the 12 full base matrices; `lora_A`/`lora_B` are added only later by `update_from_predictor_`. Fix A11: record the trainable count, regenerate with a dated supersede note. |
| 2 | `ATLAS_SUMMARY.md` states the converged-CEM candidate spread as "3.8–8.3px" unqualified. Real range across all six seed/kind cells is **3.77–27.15px** (`ln_act` seeds 1 and 2 are 17.9 and 27.2). `E0_RESULTS.md` correctly hedges "for most seed/kind pairs"; the summary dropped the hedge. | Restate with the full range. Main text must not say "tight cluster". Fix A14. |
| 3 | `ACTION_SAMPLING_REVIEW.md` frames the aimed-walk collector as reliable "by construction". `run_e0.py:326-341`'s own inline math gives **~43% single-attempt contact**; the ~100% headline comes from the retry-until-contact loop (`max_tries=8`, `1−(1−0.43)^8 ≈ 0.989`). | The doc overstates. Consequence beyond wording: training trajectories are **rejection-sampled on contact**, so the training distribution is conditioned on contact while CEM's candidate distribution (~80% contact) is not — a fourth axis of train/deploy mismatch. Disclose; record the realised single-attempt rate in the manifest. |
| 5 | **(2026-08-31)** `scripts/run_e4.py:80` sets `CEM_NUM_ACT_STEPPED = 1`, citing `E0_RECOVERY_PLAN.md` P5. `IMPLEMENTATION_PLAN_V3.md` §3.2 excludes `nas=1` on scientific grounds and §3.3 records it as "Rejected — option D". | **The code is stale, not the plan.** Verified at source this session: `score.py:78` `T = actions.shape[0]`; at `T=1` umf's denominator (`score.py:107`) is observations-only, so identical across charts, and its numerator equals `router.py:167`'s `_e1_score` — hence `argmin_c UMF = argmin_c e1` exactly. E4's routing arm and its own ablation baseline would be the same router. Logged as **FIXLOG V3-22**, not fixed (E4 deferred). Fix when revived: `--num-act-stepped 2` (3 replans, the §3.1 minimum; gate auto-recalibrates via B3; costs half of `nas=1`). |
| 4 | `CLAUDE.md` §0.1 and several docs assert "all headless gates pass". | G2 contains **no assertion at all** (a literal `if ...: pass`) and G5 is a tautology that cannot fail. Both "pass" only in the sense of running to completion. Fixes C1, C2. |

## Section 4 — Numbers that are structurally not what they appear

Each caveat must travel with its number wherever the number is quoted.

| Claim | Caveat |
|---|---|
| **N8** oracle−random CI | `d_i = oracle_i − random_i ≥ 0` **by construction at every episode**, so no bootstrap resample can be negative and the interval can never contain zero. "CI excludes zero" carries almost no inferential weight. Report as an effect-size range plus a permutation test (fix A6), never as a hypothesis test. |
| **N2** within-arm tau | Computed on n=92/94 of 100. The excluded episodes are **not random** — they are the easy, small-displacement, always-successful ones, nulled by the motion gate. The correlation is measured over the harder ~93%. |
| **N4** UMF trend | All three points share one fixed 8-trajectory set serving **both** early-stopping checkpoint selection (consulted up to 80×) and the reported `eval_umf`. Optimistically biased until fix A4 lands. |
| **N7 / N6** hysteresis | The spread-normalised margin is **algebraically inert at K=2** — the incumbent, when not the argmin, is by construction the max of a 2-element set, so the relative gap is exactly 1.0 and always clears m=0.05. All N7 runs are 2-chart, so whatever moved those numbers, it was not the margin. K=3 (N6) is non-trivial but permissive, and has never been simulated. |
| **N5** +10pp | Confounded with **3× the CEM search compute**, because `plan_length` stays pinned at `horizon=6` regardless of `steps_left`. The within-nas=2 paired comparison is fair; the nas=6-vs-nas=2 narrative is not. |
| **N1** and **N3** | **Not independent corroboration.** Under the one-shot open-loop protocol a broken cost ranking *is* the entire episode outcome — one mechanism observed as cause and as consequence. |
| Every planning number | Produced at `nas=6`: **exactly one CEM search per 30-step episode**, executed before the agent observes any consequence of its own actions (`replans==1` confirmed for all 100 episodes of both N1 arms). Scope every planning claim to one-shot open-loop planning. |
| Every UMF number | The chart behind them was trained on **R0 expert demonstrations replayed open-loop under R2**, rejection-sampled on contact — not on the distribution CEM queries. |
| **N11 / N3** | The near-zero within-episode cost-rank correlation (N11) is the *per-candidate* face of the same collapse N3 reports as a *per-seed aggregate* (R2 rho≈0.001, CI spans zero) — not independent corroboration, one mechanism seen at two granularities. `lora4` was NOT re-measured for N11 (no `lora4` kind in any `cost_ranking_*` file on disk) — the E-B moving-token dissociation for `lora4` (best top-16 UMF, worst SR) rests on `umf_locality.json` alone, not on a cost-ranking recompute. |

---

## Section 5 — Phase 0 of IMPLEMENTATION_PLAN_V3 §11.1 (measurement gates; started 2026-08-28)

Orchestration note: all Phase-0 Modal outputs are namespaced `p0*` on the `atlas-data`
volume and mirrored locally under `phase0_v3/`, kept out of the crowded `atlas_out/`.
Nothing in §8 (E0′/E1/E2/E3/E4) is authorized until a human reads P0-C and P0-F/G7.

Modal account: all Phase-0 jobs run on the **`aidendsouzavnit`** profile per user
direction (2026-08-28). Local GPU (RTX 4050, 6 GB) is used for smoke tests only;
`.venv` has `torch 2.11.0+cu128`.

| Gate | Feeds (§) | Status | Measured value | Cost |
|---|---|---|---|---|
| P0-A τ | §6.1 | **MEASURED (provisional, dataset-replay proxy)** — see values block below | **τ = 0.262** (P95 UMF(c₀), n=137 R0 informative chunks) | done, ~$0.4 |
| P0-B motion gate | §6.6 | **MEASURED (provisional)** | **242.7** (P95 latent ‖z_T−z_0‖_F, n=389 block-static chunks) | done |
| P0-D strike rate + q | §6.2 | **MEASURED (provisional)** | R0 **5.1%** / R1 34.0% / R2 67.5%; **q = 3** (derived, matches default) | done |
| P0-E σ_r (IQR, R0 informative set) | §6.3 | **MEASURED (provisional)** | umf **0.0462** · e1 **2133.5** · sdyn **0.0409** | done |
| P0-C `iterations` gate | §3.5 planner budget → whole §11.2 table | **COMPLETE.** baseline / R2 / nas=2 / N=300 / H=6, n=20 paired (seed = episode index). it=30 from `e0_planning_nas2`; it=15 → `p0c_it15/`; it=10 → `p0c_it10/`. All local at `phase0_v3/p0c/`. | **it=30: 40% (8/20)** · **it=15: 45% (9/20)**, Δ+0.05 CI [−0.10,+0.20], McNemar p=1.000 · **it=10: 50% (10/20)**, Δ+0.10 CI [−0.10,+0.30], McNemar p=0.625. Both cut-iterations SRs lie inside the it=30 CI → **`iterations = 10` ADOPTED (not final — E0′ at N=100 is the decisive check; revert per §14 if it degrades).** Discordant-episode audit (`phase0_v3/p0c/discordant_analysis.txt`): 4/20 discordant, 1 flips toward more iters, `--iterations` flag verified honoured in raw `planner_diagnostics` (15 vs 10 CEM iters/replan) → the direction is noise, not a wiring bug. | ~$1.3 spent |
| P0-G `onpolicy` collection | §5 chart data | blocked on P0-C (iterations sets per-traj cost) | pending | ~$15 (≤$8 at it=15) |
| P0-F G4 rewrite | §9 | **DONE — rewritten + run locally (free).** Fixed identical aimed-walk actions per regime (no planner), paired by seed, primary statistic = combined block pose change (translation + 30 px/rad·rotation), tested vs a real-variance R0-vs-R0-different-seeds null band; self-test fakes the shifts and must report "not distinguishable". n=160, plus a contact-duration sweep 40→200 steps in fixed-target and block-tracking modes. | **R2: REAL** at every duration (Δpose +32–41 px, KS p=2e-11). **R1: NOT DISTINGUISHABLE** — Δpose flat at **+8–9 px across 40→200 steps** (never exits the ±13 px null band); block-tracking mode's null band explodes to ±50 px so it settles nothing. The compounding-with-duration hypothesis is **not supported**. Self-test passes. Raw: `phase0_v3/g4_duration_sweep.txt`, `phase0_v3/p0c/discordant_analysis.txt`. | $0 (local) |
| P0-F G6 rewrite | §9 | scoped: **informativeness-filter test, NOT explosion guard** (explosion checked, does not occur — see below). block-static→None, block-moved→score, gate calibrated at `frameskip×nas`. Not started (local, free). | $0 |
| P0-F G7 — **Groups A + B (calibration-stage diagnostics) DONE**, $0, local 4050. A: `scripts/phase0_g7_groupA.py` → `phase0_v3/g7_groupA_calibration.txt`. B: `scripts/phase0_g7_groupB.py` → `phase0_v3/g7_groupB_calibration.txt` | §7.1 | **A** = motion-gate calibration table + within-trajectory strike-counter liveness via real `Expander.record()`. **B** = real `{c0,R1,R2}` library (from `e2_charts/`) through a real A/B/A/B stream (288 chunks) driven by the real `route()` + `Expander` + `atlas_refine()` in production order (SCORE→SELECT→EXPAND→REFINE; EXECUTE/CEM omitted — doesn't feed these mechanisms). 8-config grid over (gate P50/P75 × τ 0.262/0.5 × m 0/0.05). Verified in-script against `atlas/` source. **NOT final G7** — library is the old-collector `e2_charts`, not P0-G on-policy; freezes/asserts nothing. | see findings block below |

**Two deviations from §11.1 as written, both logged here:**
1. **Ordering.** P0-A/B/D/E are listed first but §11.1 also says their chunk source is
   P0-G's `onpolicy` collector, which runs last. There is no on-disk R0 nas=2 UMF(c₀)
   set (`e0_v3_baseline_R0` predates `--log-umf`; `e0_planning_nas2` is R2 only). Per
   user direction P0-A/B/D/E run **first**, so the chunk source is **dataset-replay**
   under each regime (forward-only, no CEM) as a **proxy**. These are provisional
   numbers — to be re-derived against P0-G `onpolicy` chunks once P0-G runs, and both
   values reported (the plan's own "report both / for continuity" rule, §6.6).
2. `scripts/run_e0.py::load_regime_trajectories` gained an optional
   `record_block_pose=False` param (returns `info["block_pose"]` subsampled) — needed
   for P0-B's block-static-chunk test. Pure add-on, default off, no behaviour change.
   Not one of the §C.3 collector edits (those still need approval, §15-5).

### Phase-0 measured values (provisional — dataset-replay proxy, 2026-08-28)

Raw: `phase0_v3/phase0_chunks.jsonl` (1440 chunks = 3 regimes × 80 trajs × 6 T=2 windows),
`phase0_v3/phase0_summary.json`. Job: `modal/modal_phase0.py` on aiden, app
`ap-5yW7XOAH0wQWAtmDEWMiHF`. Chunks are **dataset-replay** (real Push-T demos replayed
under each regime, forward-only, no CEM) — a proxy for §6's `onpolicy` source, which is P0-G.

| Symbol | §1.7 pinned | Phase-0 measured | Note |
|---|---|---|---|
| **τ** | 0.5 | **0.262** | P95 UMF(c₀), R0 informative (latent disp > motion gate), n=137. Measured value is *below* the pinned default → old 0.5 was lenient. **Changing τ needs user approval (§15-5).** |
| **motion gate** | "10th pct train displ" | **242.7** | P95 latent ‖z_T−z_0‖_F over the 389 chunks with block pixel displacement < 1 px. Gates 71% / 68% / 52% of R0 / R1 / R2 chunks — high; flag for G7 liveness. |
| **q** | 3 | **3** | Derived: p_R0 = 0.0511, `0.0511² · 1100 = 2.87 ≥ 1`, `0.0511³ · 1100 = 0.15 < 1`. The round default is the principled choice (§6.2 predicted this). |
| **per-chunk strike rate** | — | R0 **0.051** / R1 0.340 / R2 0.675 | This is `P(UMF(c₀) > τ)` on a single informative chunk — the `p` that feeds the q derivation `p^q·N<1`. It is **NOT** "the strike *counter* fires X%": the counter needs `q=3` *consecutive* strikes on the library's best-chart UMF and resets to 0 on any non-strike (`atlas/expand.py:83–90`). Counter firing rate is a separate G7 measurement on an ordered stream. R0 ≈ 5% by construction (τ = P95 under R0). |
| **σ_umf** | — | **0.0462** | IQR of UMF(c₀) over R0 informative set — hysteresis normaliser constant (§6.3). |
| **σ_e1** | — | **2133.5** | e1 lives ~5 orders of magnitude above umf/sdyn — confirms why a per-router fixed scale is needed (CLAUDE.md §1.7 ⚠). |
| **σ_sdyn** | — | **0.0409** | |

**G7 Group A findings (2026-08-28, `phase0_v3/g7_groupA.txt`) — within-trajectory
(6-chunk, single-regime) probe; a lower bound on stream liveness, full A/B/A/B version
is Group B.**

*Motion-gate calibration (PART 1):* the gated chunks are **not degenerate** at any
percentile — gated-chunk UMF median (0.19–0.27) ≈ kept-chunk UMF median (0.18–0.22); the
gate removes genuine low-block-motion chunks (gated block-disp median 0–5 px vs kept
17–42 px), not garbage. Survival: P50 gate → 75% kept (block-disp med 23 px), P75 → 58%,
**P95 → only 36%**. No cliff — the percentile just trades data volume against chunk
motion. A P95 gate discards a lot of usable data for marginal gain.

*Strike-counter liveness (PART 2), real `Expander.record()`, q=3:*

| τ / gate | R0 arms (/80 traj) | R1 arms | R2 arms |
|---|---|---|---|
| **0.50** / P50 | 0 | **1** | 10 |
| 0.50 / P90–P95 | 0 | **0** | **0** |
| **0.262** / P50 | 0 | 7 | 48 |
| 0.262 / P75 | 0 | 6 | 33 |
| 0.262 / none (no gate) | **5** (spurious) | 13 | 63 |

- **At τ = 0.5 (the §1.7-pinned value) the expansion mechanism is near-dead**: R1 arms in
  1/80 trajectories, R2 in ≤10/80, and a P90+ motion gate zeroes it entirely.
- **At τ ≈ 0.26 (P0-A's measured value) it is alive** for both shifted regimes with **0
  R0 false-arms** — provided a motion gate of ≥ P50 is present (without any gate, R0
  spuriously arms 5×, which is exactly why τ is defined over *informative* R0 chunks).
- **τ and the motion gate must be frozen together.** The healthy non-degenerate corner is
  **τ ≈ 0.26 + gate ≈ P50–P75**. The current `CLAUDE.md` §1.7 pair (τ = 0.5, high gate)
  sits in the dead zone — changing it needs explicit approval (§15-5) but now has hard
  evidence behind it.

**Group B (real library + real A/B/A/B stream + real router/Expander/refine, 288 chunks,
2026-08-29) — confirms Group A on a proper cross-regime stream:**

| config | motion gate | router switch | probe armed / commit | max lib |
|---|---|---|---|---|
| P50 τ=0.26 m=0 | 25% | 41% (3/3 charts) | **13 / 0** | 3/5 |
| P50 τ=0.26 m=0.05 | 25% | 42% (4 charts) | **12 / 2** | **5/5 (cap bound)** |
| P50 **τ=0.5** m=0 | 25% | 43% | **1 / 0** | 3/5 |
| P50 **τ=0.5** m=0.05 | 25% | 37% | **0 / 0** | 3/5 |
| P75 τ=0.26 m=0 | 37% | 48% | 7 / 1 | 4/5 |
| P75 **τ=0.5** m=0/0.05 | 37% | 38–47% | **0 / 0** | 3/5 |

- **Expansion probe: DEAD at τ=0.5** (armed 0–1 in 288 chunks), **ALIVE at τ≈0.26** (armed
  6–13, with real commits) — Group A and Group B agree.
- **Motion gate**: 25% (P50) / 37% (P75), both non-degenerate.
- **Router switch rate 37–48% regardless of `m`** — the FIX_SPEC-A1 incumbent-normalised
  hysteresis is *weakly* active at K=3 (not inert as at K=2), but changes the switch rate by
  only ~5 pp. The switch rate is arguably high (chattery) at every setting — a point for the
  §6.3 `m`/normaliser decision, which already proposes `m=0` headline + sensitivity rows.
- **K_max = 5 is NOT purely decorative**: at the aggressive corner (P50, τ=0.26, m=0.05) the
  probe committed 2 charts (3→5) and the cap then bound (2 `rejected_full`). At every other
  config it never binds.

**Group B run 2 (instrumented, 2026-08-29) — `phase0_v3/g7_groupB_calibration.txt` +
`g7_groupB_kmax10.txt`:**

- **τ=0.5 probe dead confirmed a 3rd time** (armed 0–1 / 288 chunks at every gate×m).
  τ≈0.26: armed 6–13.
- **Verification-chunk contamination is minor.** Of armed probes, the verify chunk was
  same-trajectory ~70–80% of the time (`10same/3cross`, `8same/4cross`, …) — the earlier
  cross-trajectory concern barely affects the numbers.
- **`rej_score` dominates; the probe arms but rarely commits** (2–3 commits / 288 chunks
  even at the best config). **Caveat — Group B under-tests commit liveness**: it pre-loads
  `{c0, chart_R1, chart_R2}`, so when an R2 segment strikes, the 20-step candidate is
  competing against an *already-R2-trained* incumbent and usually loses (correct
  behaviour). The real S2 ladder starts from `{c0}` only, where B=R2 is genuinely novel
  and commits *should* happen. Commit-liveness must be re-measured on a `{c0}`-start stream.
- **K_max sweep:** at cap=10 the library grows to **6** (committed 3: 3→6) then stops on
  its own — not a runaway. So K_max=5 clips this stream by ~1. `K_max = 6–8` is a cleaner
  choice than 5; the plan's "2·regimes+1 = 5" is a hair tight (and this count is inflated
  by the pre-loaded charts anyway).
- **End-to-end confirmed:** in the cap=10 run the router selected a *newly-committed* chart
  **67 / 288 times** — probe commit → router actually uses the new chart → the full
  SCORE→SELECT→EXPAND loop works, not just each piece alone.
- **Router switch rate 35–58%**, `m` damps it inconsistently (P75: 57%→35%; P50: ~no
  change). Still chattery — supports v3 §6.3 (m=0 headline + sensitivity row).

**Group B `{c0}`-cold-start + contamination-guarded (2026-08-29) — supersedes the
warm-start numbers above.** A real bug was found and fixed: `atlas_refine` leaves the
predictor holding the last-refined chart's weights (`loop.py:294–302`), so the earlier
multi-config grid snapshotted a *contaminated* `c0` for configs 2–8. Fix: `deepcopy`
pristine `state_dict` once, `load_state_dict` before each config; a per-config `c0`
checksum now prints CLEAN. Production (`run_e4.py`, one arm per subprocess) was never
affected. Retraction: the earlier "`m=0.05` kills expansion (0 commits, library frozen)"
was a **contamination artefact** — with the guard, `m` does not materially affect commits
(a `--no-refine` ablation confirms: m=0 and m=0.05 identical). `m`'s real effect is ~10–15 pp
of switch-rate damping — still → `m = 0` headline, but for the honest reason.

- **Commit count is draw-dependent** (`{c0}` start, m=0, P50, τ=0.26, 288-chunk stream, real
  `Expander`): 7-seed distribution `[0,1,1,2,2,2,4]`, mean 1.7, **1 zero in 7**. At a
  576-chunk stream: `[3,4,5]`, **0 zeros** — the zero was a short-stream artefact, not a
  broken mechanism. Probe *arms* reliably every draw (13–21×).
- **Commit QUALITY — the load-bearing finding.** For each commit, the committed candidate
  and the incumbent it beat are snapshotted and re-scored on the next same-regime chunks
  (excluding the chunks used to accept it — see the window note). The current **single-chunk
  accept gate produces a large fraction of one-hit wonders** — commits that won the
  verification chunk then lose to the incumbent right after. It is statistically
  underpowered, same species of problem as n=20 CIs.
- **Three methodology corrections during this measurement** (all instances of standing
  rule 5, all logged rather than smoothed):
  1. The first quality check's follow-up window **overlapped** the accept window (circular).
     Fixed: each commit records `verify_idx`; the follow-up loop skips those chunks.
  2. Commit decisions are **not deterministic run-to-run** → ~±2 commits / 6 seeds, so no
     *cross-run* comparison is clean even same-seed. My "same seeds → same commits" claim
     was wrong.
  3. **The cause is CUDA non-determinism, not Dropout.** Checked directly: `predictor`
     loads in `.eval()` mode, 0 Dropout modules active. But `phase0_measure.py` run twice
     gives **all 48/48 chunks differing by ~1e-4** — cuBLAS re-autotunes GEMM kernels per
     process launch, cuDNN picks non-deterministic algorithms. **Every Phase-0 number has
     ~1e-4 cross-process noise**, not just G7-B. Impact: τ and the motion gate (P95 over
     100+ chunks) are stable at 4 sig figs; P0-D strike rate could move ±~0.7 pp on a
     boundary chunk; G7-B commit decisions (borderline `cand_umf < best_umf`) are where it
     actually bit. **Fixed** by `scripts/_determinism.py` (`CUBLAS_WORKSPACE_CONFIG`,
     `use_deterministic_algorithms`, `cudnn.deterministic`, seeded) — `phase0_measure.py`
     re-run twice with it → **0/48 chunks differ, bit-identical.** Added to
     `phase0_measure.py` and `phase0_g7_groupB.py`; **required for the P0-G collector.**
- **Clean numbers — `verify_chunks` (additive; default off; E2/N9 unaffected; G3a/G3b still
  pass). Same commit set classified BOTH ways in one run** (the only confound-free
  comparison):

  | `n_verify` | commits | generalise, DISJOINT (correct) | generalise, OVERLAP (buggy) |
  |---|---|---|---|
  | 1 (current gate) | 12 | 5/12 = **42 %** | 5/12 = **42 %** |
  | 3 | 8 | 6/8 = **75 %** | 5/8 = 62 % |
  | 5 | 5 | 4/5 = **80 %** | 5/5 = 100 % |

  - **The window bug's real effect is at high `n_verify`** (nv=5: fake 100 % → real 80 %),
    where half the follow-up window *was* the accept chunks. At nv=1 it does nothing (1
    excluded chunk out of a 10-chunk window). The earlier "40 % at nv=1" ≈ this run's 42 %;
    the difference from the "67 %" circular run was **the added seed + fit noise, not the
    circularity**.
  - **`n_verify = 1` → ~42 % generalise** (the live gate is poor). **`n_verify = 3` → ~75 %**
    (real, substantial gain). **`n_verify = 5` → ~80 %** (marginal over 3, costs 3/8 commits).
  - Mean UMF gap over follow-ups +0.026–0.031 across all runs — the library nets positive
    even with the flukes.
  - **`n_verify = 3` is the working proposal, NOT final.** The P0-G re-run must: fix
    `torch.manual_seed` per stream + disable predictor Dropout during scoring/fit (kills the
    ±2-commit run-to-run noise); use a fixed seed list and ≥576-chunk streams; sweep
    `n_verify` × `n_probe` jointly on the on-policy charts. `n_probe` sweep was attempted
    locally but the machine killed the job (~04:20, not a code fault) — deferred there.
- **Does expansion HELP net? Yes, modestly** (5 seeds, `{c0}` start, frozen-c0 baseline):
  R2 selected-UMF **frozen 0.415 → refine 0.393 → full 0.380** (expansion adds ~3 % over
  refinement, 8 % over frozen). And **refinement *hurts* R0** (0.129 → 0.168, forgetting),
  while **`full` forgets less than `refine`** (0.163 vs 0.168) — dedicated R2 charts absorb
  the R2 refinement pressure, keeping `c0` closer to R0. C2 shows a measurable benefit on
  both sides of the plasticity/stability tradeoff. Small effect; E3/E4's ladder + a planner
  is the real adjudication.

**What this feeds:** the τ + motion-gate + `m` + `K_max` + accept-criterion freeze
(§15-5 / §6). Combined Group A+B evidence →
**τ ≈ 0.26 · gate P50 · m = 0 · K_max ≈ 5–6 · `n_verify` ≈ 3 (new)**. Final asserting G7
re-runs one frozen config against P0-G's on-policy charts + a ≥576-chunk stream.

---

**G6 — "UMF blow-up near zero motion" checked empirically, does NOT occur (2026-08-28).**
Over all **1440** `phase0_chunks.jsonl` chunks (3 regimes, T=2 production config, motion
gate OFF): `UMF(c₀)` min 0.045 / median 0.197 / **max 2.05**; 8 chunks (0.6%) exceed 1.0,
**0 exceed 5**; the hard `displacement==0` guard fired **0/1440**. The 389 block-static
(<1 px) chunks have the *lowest* UMF (median 0.126), not the highest — numerator and
denominator shrink together on quiet chunks, and the denominator is over the full
256×384 latent (agent motion + background keep it ≥ ~17). The historical "UMF 24–52"
was the pre-T1 rollout bug, not a zero-motion artefact.
**Consequence for G6:** drop the explosion-guard framing (obsolete post-rollout-fix);
keep `displacement==0` as untested defensive code; G6's real job is to verify the
motion gate works as an **informativeness filter** (block-static → None; block-moved →
real score; gate calibrated at `frameskip × num_act_stepped`).
**Sharper open question this surfaces:** the P0-B gate (242.7) would drop 52–71% of real
chunks, and those chunks are *not* degenerate — UMF 0.1–0.3, same range as everything
kept. So 242.7 is plausibly discarding usable, informative data that E1/E3/E4 need.
**Resolution path:** G7 sweeps the gate percentile (P50/P75/P90/P95) with the kept-vs-
dropped UMF distribution at each, and the threshold is picked empirically from that
sweep (same "derive it, don't assume it" rule as τ, q, iterations) — not treated as a
separate detour.

**G4 ↔ P0-D reconciliation (R1):** G4 says a fixed push ends only ~8 px further under
R1 (within noise). P0-D says the frozen model mispredicts 34% of R1 chunks (vs 5% R0).
These agree: stratifying `phase0_v3/phase0_chunks.jsonl`, R1 strike chunks have *higher*
block displacement than R1 non-strike chunks (51 vs 37 px) — the misprediction is real
signal on high-motion chunks, not small-denominator UMF inflation. So **R1 is a
prediction-level regime shift (model-correctable) but only a marginal trajectory-level
one.** R2 is unambiguous at both levels. Paper must scope R1 accordingly and not lump it
with R2. See `PAPER_DRAFT_NOTES.md` §8.2.

**Caveats travelling with these numbers:**
- Provisional: dataset-replay ≠ `onpolicy`. Re-derive against P0-G chunks; report both.
- The motion gate gating 52–71% of chunks is the §6.6 / B3 risk zone. Not necessarily
  wrong (agent-only motion genuinely moves the latent by up to ~243), but P0-F's G7
  liveness sweep must confirm enough informative chunks survive in a real stream.
- τ is measured over n=137; a wider `onpolicy` collection will tighten it.


---

## Section 6 — E4 / S2 status (2026-08-31)

**No E4 episode exists.** `atlas_out/` contains no `e4` directory (re-confirmed
2026-08-31). Every E4/S2 claim in this repository is **L2 (code exists)** at most —
`run_e4.py` + `harness_e4.py` + `streams.py` + `modal_e4.py` are complete and have been
smoke-run end to end (12 episodes, FIXLOG B3/B4), but nothing has been measured.

**The stated reason for E4's retirement is retired.** `FABLE5_VALIDATION.md` §6 killed
E3+E4 because "the library's only non-identity chart destroys control" — which rests on
the C-2 pass-through headline. The Day-1 settle-check (C2-settle, L5) reversed that: on
the settle-validated metric the chart is significantly *better* (nas=2 Δ−59.8 px,
p=0.0002). The E4 decision was not reopened afterward. **Do not cite
`FABLE5_VALIDATION.md` §6's reason as the reason.** The deferral now stands on wall
clock, the `nas=1` collapse (Section 3 row 5 / FIXLOG V3-22), E4's use of the
invalidated pass-through metric, and scope (the draft is an evaluation paper).

Full reasoning, including what a revived E4 would need: **FIXLOG "E4 DEFERRAL"**
(2026-08-31).
