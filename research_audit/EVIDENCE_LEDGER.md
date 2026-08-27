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
