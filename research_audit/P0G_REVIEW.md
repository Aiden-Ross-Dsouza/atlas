# P0-G PRE-LAUNCH REVIEW — does the on-policy chart collection actually make sense?

**Date:** 2026-08-29 · **Scope:** `modal/modal_phase0.py::p0g_collect` and everything it invokes.
**Nothing was launched. No code was changed.** Review only.

**Method.** Two independent passes over the source (not the summary docs) by the main session, plus
a `scientific-redteam` dispatch and a scoped `phd-skills:xray` dispatch (code↔spec alignment,
evaluation integrity), folded into this one file. Per `CLAUDE.md` §1.9 every finding carries an
explicit *what was done to verify it*. **Every sub-agent finding written below was re-verified at
source by the main session before being entered here**; where re-verification failed, that is
stated and the finding is labelled **UNVERIFIED**.

---

## VERDICT

> ### **DO NOT LAUNCH AS CONFIGURED.**
>
> Not because the idea is wrong. On-policy collection is the right fix and it does genuinely close
> three of the four charges in `COHERENCE_AUDIT_2.md` CHECK 2 (§A). It must not launch because:
>
> 1. **P1** — a verified ordering bug means the **second regime's data is collected by an
>    already-adapted predictor**, silently violating §5.2's "against the frozen `c₀` predictor",
>    the single property the whole design rests on. *Zero-code fix: one call per regime.*
> 2. **P2** — the run **cannot finish inside its own 8 h Modal timeout** (~13.6 h needed), and a
>    timeout kill discards up to 4 h of unpersisted collection.
> 3. **P3** — the number that decides whether the charts are any good is **computed on the set that
>    selected them**, and this repo has already measured that bias at **+0.077 to +0.157** UMF.
> 4. **P4 / P5** — the collector is **not at the evaluation planner config** it claims to be at, on
>    three axes (lookahead, episode length, goal separation).
>
> All four are cheap. Below them sits one problem that is *not* cheap and is the real scientific
> question — **§B**, the R2 objective collapse. It does not block the run, but it changes what the
> run is allowed to claim and makes the planning check in **§C** non-optional.

---

## PART 0 — Does P0-G make sense at all? The step-by-step

The question asked was not "are there bugs" but "does this whole thing make sense". In order:

**Step 1 — Is the problem P0-G exists to solve real?** **Yes.** `COHERENCE_AUDIT_2.md` CHECK 2 found
the `scripted`/`hybrid` collectors optimised *"walk to a random point within 40 px of the block"*
with acceptance `total_contacts > 0` — no goal anywhere in the loop. `EVIDENCE_LEDGER.md` §4 says
the same of the headline charts: *"trained on R0 expert demonstrations replayed open-loop under R2,
rejection-sampled on contact — not on the distribution CEM queries."* A real defect, worth fixing.

**Step 2 — Does the `closed_loop` collector fix it?** **Substantially yes** — three of four charges
genuinely closed, verified item by item in **§A**. Credit this properly in the write-up.

**Step 3 — Is the collected data therefore "goal-directed"?** **In form yes; in effect no, under
R2.** This is the crux. §5.2 argues *"Goal-directed by identity, not by approximation… There is no
substitute objective to drift away from."* But the objective's *effect* under R2 has already been
measured by this project at L5, and it is null (N3, N11, N3b). The objective did not need to be
*substituted*; under R2 it **degenerated**. Full argument in **§B**.

**Step 4 — Is training a chart on this data still the right move?** **Defensibly yes**, on §5.2's
argument (c) alone — the states a broken planner actually visits are the states the chart will be
scored and refined on. That is the DAgger-round-1 rationale and it is legitimate. But it is now an
*empirical bet*, not a design guarantee, and must be written up that way.

**Step 5 — Is "accept the chart on eval UMF" a valid way to close that bet?** **No.** N1 (44/100 vs
43/100, McNemar p=1.000, while UMF fell monotonically) and `lora4`×R1 (better UMF, **4/10** vs 8/10
planning) say latent prediction error does not track planning competence here. **§C** specifies the
smallest checks that close this, with costs and numeric red flags.

**Step 6 — Is "train on 25-step sampled rollouts, judge success by whether the T-block reaches the
target" a coherent pairing?** Coherent *in principle* — the chart is trained to predict and judged
on whether better prediction buys better control; that dissociation is what this paper is about.
**Not coherent as wired**, on three axes: training rollouts are 25 raw steps vs 30 at eval, at
effective lookahead 5/3/1 vs 6/6/6, and the acceptance UMF is a **5**-model-step rollout while every
Phase-0 threshold (τ, motion gate, σ_r) and the deployed loop score **2**-step chunks. (**P4, P5,
P8**.)

**Bottom line:** the right experiment, aimed at a real defect, with a stated justification that is
partly unsupported and an implementation that deviates from its own design in four places. Fix the
implementation, downgrade the justification to what the evidence supports, add §C, and it is worth
running.

---

## FINDINGS

Severity: **BLOCKS** = do not launch · **TRUST** = fix before any chart from this run is used
downstream · **HUMAN** = needs an explicit recorded decision · **COSMETIC**.

| ID | Sev | Finding | Evidence | Verification |
|---|---|---|---|---|
| **P1** | **BLOCKS** | With `--regimes R0,R2` in one process, **R2's trajectories are planned by a predictor still holding R0's fine-tuned `ln_act` weights.** The collector is not frozen `c₀` for any regime after the first. | `run_e0.py:688` regime loop → `:700/704/711` collection → `:752` kind loop → `:757` `load_state_dict(pristine_predictor_state)`. The pristine reload is *inside* the kind loop, i.e. **after** that regime's collection. `chart.py:126-127`: for `kind != "lora4"`, `restore_()` is `self.apply_(predictor)` — it re-applies the chart's **trained** params. `chart.py:186-204` documents this explicitly (FIX_SPEC C4). | Read the four line numbers in execution order; then read `Chart.restore_` at source rather than trusting `run_e0.py:643`'s comment about it. Both agree. Surfaced by the red-team agent, **independently confirmed by the code-spec agent (H2) and re-verified by the main session.** |
| **P1b** | **BLOCKS** | `run_e0.py:655-658`'s comment — *"Built from the PRISTINE predictor… Collecting under an already-adapted model would be a second DAgger round, not this experiment"* — is true only for the first regime. A comment asserts a property the code does not hold. | same as P1 | same as P1 |
| **P1c** | **TRUST** | Consequence: the smoke's **R2 eval UMF 0.396 was produced from R0-adapted collection**, and the R0-vs-R2 contact comparison that §15-2's pre-registered damping check depends on is confounded (R0 collected pristine, R2 collected adapted). | `SMOKE_SUMMARY.md`: the smoke ran `--regimes R0,R2` in one call. | Read the smoke command; applied P1. |
| **P2** | **BLOCKS** | **The run cannot finish inside its own timeout, and the fine-tune cost line is ~19× low.** One "step" is a full pass over **every** training trajectory, not a minibatch. | `harness.py:159` `for step in pbar:` → `:170` `for traj in trajectories:` → `:182` `(loss/len(trajectories)).backward()` → `:185` `optimizer.step()` outside the traj loop. `modal_phase0.py:63` `timeout=3600*8`. `SMOKE_SUMMARY.md`: "fine-tune ~$0.4 … total ~$3.6, ~4.5 h". | Read the loop nesting. Rate from `smoke_R0R2_modal.log`: `60/60 [00:25<00:00, 2.31step/s]` at **5** train trajs → **0.0866 s/traj-pass**. At 100 trajs: 8.66 s/step × 2000 = **4.81 h/chart** × 2 regimes = **9.62 h**, + 4.01 h collection (216 × 66.78 s, matching SMOKE_SUMMARY's own 4.0 h) = **13.63 h vs an 8 h timeout**. ≈ **$11** at the $0.799/h the summary's own $3.2/4.0 h implies; `EVIDENCE_LEDGER.md` §5 independently budgets P0-G at "~$15", not $3.6. **Break-even: both regimes must early-stop by ~step 830.** Earliest possible stop is step 150 (`patience=5 × eval_every=25`) — which is the only scenario where "$0.4" is right, and nothing supports it (see P2b). Computed independently by the main session and the eval-integrity agent; figures agree. **Stated assumption:** linear scaling 5→100 trajs — justified by the sequential per-trajectory `backward()`, **not measured**. Even at an implausible 2× speedup the total is 8.8 h, still over. |
| **P2b** | **TRUST** | **No convergence evidence exists at any production budget**, and the only signal points at overfitting. | `smoke_R0R2_modal.log:99-103`: train `0.437265 → 0.206620` (−53%) while val `0.613223 → 0.558034 → 0.543752` (−11%), val still improving, `(best)` on every check, **patience never touched**. Repo-wide, **no `val_loss_*.json` artifact exists on disk**. | Read the log lines and grepped for the artifact. The smoke cannot bound where early stopping fires — which is exactly the unknown P2's cost hinges on. A 2.63× train/val ratio at step 60 on 5 trajectories is an overfitting signature. |
| **P2c** | **TRUST** | **A timeout kill loses everything, and a resume re-collects.** | Charts are written only at the end of `run_e0_finetune` (`harness.py:243`); the seed manifest only after *all* regimes (`run_e0.py:895-896`); collected trajectories are never serialised. The per-`(kind, regime)` resume check at `run_e0.py:762` sits **after** collection at `:700-714`. | Read the write sites and the resume branch. A resumed run re-collects R0's 108 trajectories (~2 h, ~$1.6) purely to skip a cached chart. |
| **P3** | **BLOCKS** | **The reported chart-quality number is computed on the early-stopping selection set.** `--num-test-trajs 0` disables FIX_SPEC A4's disjoint test split, and `results.json`'s field comment still claims otherwise. | `modal_phase0.py:86` `"--num-test-trajs", "0"` — hard-coded in the launcher; `p0g_entry` exposes no override. `run_e0.py:845-848` `if test_trajectories: … else: eval_loss, eval_umf = val_loss, val_umf`. `run_e0.py:868-869` still reads `"eval_umf": …  # from the disjoint TEST set (A4)` and `"val_umf": …  # bias = eval_umf - val_umf` — under this config both hold the same float, so the recorded bias is **identically 0.0 by construction**. | Confirmed empirically in the smoke log: `Step 60/60 - Val Loss: 0.543752 (best)` and `Eval Loss = 0.543752` — identical to six digits. **Magnitude is already measured in this repo:** `FIXLOG.md:1721-1727` records `eval_umf − val_umf` across all 9 chart×regime cells as mixed-sign, "R0 all positive (+0.077…)", with `full`×R0 at **val 0.713 vs test 0.870 = +0.157**. `EVIDENCE_LEDGER.md` §4 already flags this exact bias for N4. **Decision-relevant:** the smoke's R2 UMF of 0.396 plus a +0.08–0.16 correction straddles the proposed τ ≈ 0.262 comparison. Fix: `--num-test-trajs 8` ≈ 16 trajectories ≈ 18 min ≈ **$0.25**. |
| **P4** | **BLOCKS** | **The collector's planner runs at a shorter lookahead than every CEM search at evaluation.** Collection plans **5 → 3 → 1** model-steps; eval plans **6 → 6 → 6**. | Collection `run_e0.py:307` `steps_left=max(n_chunks - chunk_idx, 1)`, `n_chunks = 25//5 = 5`, `chunk_idx ∈ {0,2,4}` → 5, 3, 1. Eval `run_e0_planning.py:288` `steps_left_model = (n_replans_target - replan_idx)·nas` = 30, 28, 26, deliberately loose (`:257` calls it "a LOOSE upper bound"). Planner `…/planning/planning/planner.py:159,275,401,573` `plan_length = min(self.horizon, steps_left)`. | Read all three sites plus the planner. Arithmetic: **100% of collected raw steps come from `plan_length < 6`; 60% from `plan_length ≤ 3`; the final 20% from a single-model-step myopic plan.** `EVIDENCE_LEDGER.md` §4 (N5) independently confirms the project knows `steps_left` truncates `plan_length` — *"`plan_length` stays pinned at `horizon=6` regardless of `steps_left`"*, true of the **eval** path, which is exactly why the collector's tight bound is a deviation. Chunk indices confirmed in `timing_R0_modal.log` (`chunk=1/5 → 3/5 → 5/5`). **This is the same collection≠eval protocol class (C-1) P0-G was built to fix, reappearing in a different variable.** |
| **P4b** | UNVERIFIED | The red-team agent reported per-search wall times of 39 s / 22 s / ~4 s as empirical confirmation of the shrinking `plan_length`. | — | **Could not be reproduced** from `timing_R0_modal.log` with two independent extractions of the tqdm timing fields. **P4 stands on the code and arithmetic, which were verified directly.** Do not quote the wall-time triple. |
| **P5** | **BLOCKS** | **Episode and goal geometry deviate from §3.6's pinned protocol.** Collection: goal **24** demo-steps away, **25**-step budget. Eval: **30** away, **30**-step budget. §5.2 requires "same sampler, same filters as evaluation". | `modal_phase0.py:66-67` `traj_len=25, eval_traj_len=25`; `run_e0.py:230-234` forwards `traj_len=traj_len` into `sample_dataset_init_goal` (`run_e0_planning.py:192-193`). Eval: `run_e0_planning.py:72` `GOAL_TRAJ_LEN = FRAMESKIP*6+1 = 31`, `:393` `--max-steps` default 30. §3.6 pins `MAX_MPC_STEPS = 30`, `GOAL_TRAJ_LEN = 31`. | Read all call sites. **The filters *do* match** (`min_block_pos_diff = 40.0`, `max_agent_block_dist = 160.0`, identical on both sides) — only the lengths differ. The code-spec agent replayed both samplers on the real `states.pth` over 100 seeds each: collection pairs mean block displacement **79.0 px** (median 70.4) vs eval **90.7 px** (median 77.5); 0/100 filter fallbacks either side. *(That replay is the agent's; the main session verified the call-site parameterisation but not the 79.0/90.7 figures — treat the magnitudes as the agent's measurement.)* `traj_len=30` is legal (`30 % frameskip == 0`), so this is a one-token fix. |
| **P6** | **TRUST** | **Charts are admitted on eval UMF alone**, with no planning check anywhere in P0-G, against this project's own documented dissociation. | `modal_phase0.py:77-89` invokes `run_e0.py` only — no `run_e0_planning.py` call. `EVIDENCE_LEDGER.md:52` (N1, L5): 44/100 vs 43/100, McNemar p=1.000. N4: UMF 0.336→0.302→0.268 across 20/60/100 trajectories, every planning CI spanning zero. N11 (L5): localized top-16 UMF ranks `lora4` **best** (0.168) while its SR is **worst** (40%). `CLAUDE.md` §0.1: `lora4`×R1, UMF 0.242, **4/10** vs 8/10 baseline. | Read the Modal command and the ledger rows. **See §C** for the specified checks, costs and red-flag numbers. |
| **P7** | **TRUST** | **The contact-conditioning bias is removed from chart training and fully retained everywhere the charts are judged.** Worse than previously recorded: **E1 and E4 calibrate their motion gate on `scripted`** — the goal-free contact-seeking random walk §5.2 declares *retired*. | `run_e0.py:78` signature default is `source: DataSource = "scripted"`. `run_e1.py:292` and `run_e4.py:233` call `load_regime_trajectories(...)` **with no `source=`** → `scripted`. `run_e2.py:184,213,272,454` and `phase0_measure.py:69` pass `source="dataset"` → still contact-rejection-sampled at `run_e0.py:364` with `max_tries=8`. | Grepped every `load_regime_trajectories` call site in `scripts/` and read the signature default and the acceptance line. **Consequence:** τ = 0.262, motion gate = 242.7, strike rates 0.051/0.340/0.675 and σ_r were **all** measured on contact-filtered replay chunks (`phase0_measure.py:69`), and E2's scoring chunks are contact-filtered too. The red-team agent reported this as "all `source='dataset'`"; **the E1/E4 case is `scripted`, which is strictly worse** — those two are the flagship continual experiments. |
| **P8** | **TRUST** | **P0-G's acceptance UMF is a 5-model-step rollout; every Phase-0 threshold and the deployed loop score 2-step chunks.** | §3.6 pins "T = 2 model-steps per scored chunk". `modal_phase0.py:66-67` → `eval_traj_len=25` → `actions` is `[5,10]` → `evaluate_e0_chart`/`umf` unroll 5 model steps; `harness.py:177-182` backprops the same full 5-step unroll. τ = 0.262 and gate 242.7 were measured on **T=2** chunks (`EVIDENCE_LEDGER.md` §5: "1440 chunks = 3 regimes × 80 trajs × 6 **T=2** windows"). | Read the shape chain and the ledger. Open-loop error compounds with horizon, so "Eval UMF 0.396" is **not on τ's scale** and cannot be compared to it. Independently found by both xray agents (M3 / MEDIUM-7). |
| **P9** | **TRUST** | **P0-G persists no trajectories and no per-chunk scores**, so §5 deviation-note 1's promised re-derivation of τ / gate / strike rate / σ_r "against P0-G's `onpolicy` chunks" requires paying for collection twice. | `run_e0.py:760, 880, 896, 913` write only `chart_*.pt`, `loss_*.json`, `val_loss_*.json`, `results.json`, `e0_seed_manifest.json`, `results.md`. `load_regime_trajectories` returns in-memory tensors that are discarded. §5.2 says "**record** the executed `(observation, action, proprio)` chunks". | Read the write path end to end; grepped `run_e0.py` for any per-chunk JSONL emission — none. Found independently by the red-team agent (PG-6) and the code-spec agent (M4). |
| **P10** | **TRUST** | **The motion gate P0-G applies is the retired rule, at the wrong granularity, and its value is pinned nowhere.** Five mutually incompatible numbers are in circulation. | `run_e0.py:745-750` builds one `‖z_{-1}−z_0‖_F` **per whole trajectory** (T=5) and calls `compute_motion_gate`, whose default is `percentile=10.0` (`score.py:237-251`). §6.6 **replaces** this with P95 over **block-static chunks at `T = num_act_stepped = 2`**, and explicitly demands the calibration length derive from `frameskip × num_act_stepped` *"so the gate can never again be calibrated at a granularity it is not applied at."* Circulating values: **242.7** (P0-B, T=2), **295.08** (this collector — `timing_R0_modal.log`: `motion_gate (10th pct of train displacement) = 295.0807`), **317.77 → 117.62** (§8.4's B3), **P50/P75** (G7 A/B), plus whatever the full run computes. | Read `compute_motion_gate`, `run_e0.py:743-749`, the log line, `EVIDENCE_LEDGER.md` §5 and §8.4. Neither the percentile, the population, nor the length matches §6.6, and the B3 invariant is encoded nowhere. Because the value is recomputed per run from that run's own training set it is also **not comparable between R0 and R2**. Note `run_e4.py:230-232` gets the *granularity* right (`frameskip × nas`) and the *source* wrong (P7); P0-G is the mirror image. Neither matches §6.6. |
| **P10b** | **TRUST** | **The gate silently changes the denominator of the reported mean, and `n` is never recorded.** | `run_e0.py:466-472`: `losses` gets one entry per trajectory; `umf_scores` only per *ungated* trajectory — so `eval_loss` and `eval_umf` in the same `results.json` row can be means over different subsets, unmarked. `len(umf_scores)` is never printed or written (`:865-874`). Contrast `phase0_v3/p0c/p0c_it10_summary.json`, which **does** carry `"umf_episodes_with_value": 18` beside `"umf_mean_of_means"`. | Read the block against `score.py:86-89`. At 8 val trajectories a 10th-percentile gate is expected to drop ~0–1, so true `n` may be 7 or 8 and nothing records which. **Directional bias:** the gate drops *low*-displacement trajectories, which have the smallest UMF denominator and hence the largest UMF — so the gated mean is systematically **optimistic**, the same sign as P3, compounding it. If everything is gated, `nan` is formatted straight into the T5 table (`:909`) with no warning. |
| **P11** | **TRUST** | **Collection never terminates on success; evaluation does.** Training data contains post-goal behaviour the deployment protocol never produces. | `run_e0.py:317-323`: `obs, reward, done, info = env.step(act)` — `done`/`reward` never read; the loop always runs the full `traj_len`. `run_e0_planning.py:318-321`: `if final_check["success"]: success = True; break`. | Read both loops. Early termination confirmed in real data (`atlas_out/e0_planning_nas2/baseline_R2.jsonl`, episodes with `"steps" < 30`). The red-team agent's supporting claim that R2 successes complete in ~8.4 raw steps was **not re-derived** — treat the *magnitude* as UNVERIFIED, the *asymmetry* as confirmed. |
| **P12** | **HUMAN** | **The pre-registered R2 contact-collapse check is produced but gates nothing and is never persisted.** | §4 / §15-2: the on-policy contact count is *the* check on whether the 38.5→13.3 replay collapse was a collector artifact; `damping=0.1` is the pre-registered fallback. `run_e0.py:736-739` computes and **prints** it — but `run_e0.py:715-724` writes only `seed`/`episode_idx`/`offset` to the manifest, so `n_contacts` survives **only in captured stdout**. `p0g_collect` runs collection and fine-tuning as one un-gated command. | Read the manifest construction and the debug print. The print is also mislabelled `"Real-demo replay contact rate"` for `closed_loop`. By the time a human can read the number the charts are trained at `damping=0.5`; if it fails, the pre-registered response invalidates them. Per P1c the R0-vs-R2 comparison is confounded anyway in a combined call. |
| **P13** | **HUMAN** | **Dropping R1 forecloses two experiments the live plan still specifies.** | §8.1: *"Arms: frozen `c₀` vs. `ln_act` chart, **on `R1` and `R2`**, N = 100 paired episodes each. **Charts: trained on `onpolicy` data (C.3)**"*; decision rule *"…CI excludes zero **in at least one regime**"*. §8.3: *"B decisive (R0/R2, **plus R0/R1 as replicate**)"*. §4: *"R1 is E0's second capacity regime and E2's cell-B replicate."* `modal_phase0.py:65` `regimes = "R0,R2"`. | Read §4/§8.1/§8.3 against the launcher default. See **§D**. **Countervailing fact (why HUMAN, not BLOCKS):** P0-F/G4 measured R1 as prediction-level only (Δpose +8–9 px, inside the ±13 px null band, flat 40→200 steps), so an R1 *planning* arm may be uninformative a priori. Good argument for dropping R1 — but it must be recorded against §8.1/§8.3 before launch. Cost to include: +108 trajectories ≈ 2.0 h collection + ~4.8 h fine-tune. |
| **P13b** | **HUMAN** | **If R1 is added, its collection seeds overlap the planning evaluator's.** | `run_e0.py:184` `seed_base = {"R0":2000, "R1":0, "R2":1000}`; `run_e0_planning.py:424,556` uses `seed == episode index`, i.e. 0…N−1. | The code-spec agent replayed both samplers against the real `data/pusht_noise/train/states.pth`: R2 collection seeds vs eval seeds 0–99 share **0** demo episodes; **R1 collection seeds share 39 of 100.** *(Agent's replay; the main session verified the two seed conventions at source but did not re-run the replay — treat "39/100" as the agent's measurement.)* Not live at `R0,R2`. Also `scripts/audit_e0_train_planning_overlap.py`, which exists to check exactly this, cannot run on on-policy manifests (see P18). |
| **P14** | **HUMAN** | **§5.2's stated justification for the collector is not supported for R2.** | See **§B**. `EVIDENCE_LEDGER.md` N3, N3-dose, N11, N3b. | Read the ledger rows; corroborated independently against `phase0_v3/p0c/` — see §B. |
| **P15** | **TRUST** | **`_determinism.py`'s SDPA pinning is structurally inert on this model — root cause identified.** The predictor's attention re-enables every backend per call, overriding the global flags. | `_determinism.py:52-56` sets `enable_flash_sdp(False)`, `enable_mem_efficient_sdp(False)`, `enable_math_sdp(True)`, claiming at `:48-53` to "Force the math backend … for every scaled-dot-product attention call". But `hub/…/app/plan_common/models/vit.py:163` wraps the `F.scaled_dot_product_attention` call in **`with sdpa_kernel(ALL_SDPA_BACKENDS):`**, where `ALL_SDPA_BACKENDS` (`vit.py:24-29`) lists MATH, EFFICIENT, FLASH and CUDNN. `run_e0.py:637-639` sets `m.use_sdpa = True` on every predictor module, *forcing* that branch (`vit.py:158`). | **The import/call chain itself is correct** — `run_e0.py:30-33` imports `_determinism` before `torch`; `:604` calls `make_deterministic(0)` before `torch.hub.load` at `:615`; `modal_phase0.py:77-89` subprocesses `run_e0.py`. **So Part 1 item 3 checks out.** The failure is override, not ordering. Confirmed by the surviving warning in `timing_R0_modal.log`, **during the `ln_act_R0` fine-tune backward**: `UserWarning: Memory Efficient attention defaults to a non-deterministic algorithm … (attention_backward.cu:752)`. Root cause found by the eval-integrity agent; **`vit.py:24,163` re-read and confirmed by the main session.** The module docstring's conclusion ("cannot be removed without patching the vendored jepa-wms attention") is **correct**; its code comment at `:48-53` asserting the fix is applied is **false on this path**. |
| **P15b** | **TRUST** | **`run_e0.py:640` undoes the TF32 pin `_determinism.py` sets at `run_e0.py:604`.** | `_determinism.py:62-64` pins `matmul.allow_tf32 = False` "so a driver/hardware change does not silently reintroduce matmul variance". `run_e0.py:640` then calls `torch.set_float32_matmul_precision("high")` — the documented way to *allow* TF32 for fp32 matmuls. | Read the call ordering directly: `:604` `make_deterministic(0)` → `:640` `set_float32_matmul_precision("high")` → `:887` `settings_dict(0)`. Because `settings_dict` runs last, the manifest will honestly record `matmul_allow_tf32: true` — **contradicting the module that set it False**. Not a run-to-run determinism break (TF32 is deterministic), but it voids the pin's stated purpose. Found by the eval-integrity agent; ordering re-verified here. Whether `set_float32_matmul_precision` flips the flag on this exact torch build was **not executed** — UNVERIFIED by execution. |
| **P16** | **HUMAN** | **The backward-pass residual is not symmetric across R0 and R2**, because it acts through early-stopping checkpoint *selection*, not through the weights directly. | `harness.py:222-233`: the returned chart is the best-val snapshot chosen by comparing noisy val losses on 8 trajectories with `patience=5`. Effective validation `n` is **8** (40 chunks, but 5 chunks per trajectory come from one compounding autoregressive unroll off a single `z_ctxt`, `harness.py:145-147`, so they are not independent). | **Reasoned from the code, not measured — labelled as reasoning, not a result.** Exposure is regime-dependent because R0 and R2 have different loss scales (smoke: R0 eval UMF 0.171 vs R2 0.396) and therefore different val-curve flatness; late in training per-check improvements are ~0.01 (P2b), plausibly the same order as the noise of an 8-unroll mean. **Cheap check, ~6 min:** run one fine-tune twice at `--steps 200 --num-train-trajs 20` per regime and compare `stopped_early_at_step` and `eval_umf` across launches. This directly answers Part 1 item 4 and **has not been done.** |
| **P17** | COSMETIC | An `ln_act` chart is trained for **R0** — ~4.8 h of fine-tune budget for a chart no experiment in §8 uses. | `modal_phase0.py:79-81` `--kinds ln_act --regimes R0,R2`. §8's libraries are `{c₀, chart_R1, chart_R2}`. | Read the launcher and §8. R0 *collection* is needed (τ and σ_r are defined over R0 chunks); the R0 *fine-tune* is not. Either skip it, or keep it deliberately as a null control (a chart trained on the regime `c₀` already fits should show ~no UMF gain — genuinely useful, but say so). |
| **P18** | COSMETIC | For `closed_loop` the manifest records `episode_idx: null`, so demo-episode disjointness between train and val is **unauditable**. Quantified: risk is low. | `run_e0.py:225` `episode_idx: int \| None = None`, assigned only in the `dataset`/`hybrid` branch (`:240`), never in `closed_loop`. `SMOKE_SUMMARY.md` confirms `episode_idx: null`. Train and val both draw `data_split="train"` (`p0g_collect` passes no `--data-split`). | Read the branch, then read the pool size directly from `data/pusht_noise/train/seq_lengths.pkl`: **18,685** episodes, **all** with `seq_length ≥ 31`. P(any of 8 val draws collides with one of 100 train episodes) = 1 − (1 − 100/18685)^8 ≈ **4.2%**; expected shared episodes = 0.043. A collision still yields only a shared *(init, goal) task*, not shared trajectory data — actions are generated live by CEM and offsets would differ. **Auditability problem, not a demonstrated leak.** ~3 lines to close permanently. |
| **P19** | COSMETIC | Seed-level disjointness is **clean at the launch config**, with a latent trap at scale. | `run_e0.py:207` `seed = seed_base + traj_idx·max_tries + attempt`; `:184` `seed_base = {"R0":2000,"R1":0,"R2":1000} + seed_offset`; `--collect-max-tries` default **2** (`:595`), not passed by the launcher. | Computed the intervals: R0 train **[2000,2199]**, R0 val **[12000,12015]**, R2 train **[1000,1199]**, R2 val **[11000,11015]**. **No pair overlaps, within or across regimes.** Only even seeds are ever drawn (contact filter off ⇒ `attempt=0` always, `run_e0.py:364`), so `--collect-max-tries` is dead code for this source while still doubling the reserved span. **Latent collision:** R1→R2 and R2→R0 collide when `max_tries × num_trajs > 1000` → `num_trajs ≥ 501` at `max_tries=2`, or **`num_trajs ≥ 126`** at the non-`closed_loop` default `max_tries=8`. No runtime disjointness assertion exists. |
| **P20** | COSMETIC | **The seed manifest will record `git_commit: "unknown"`** on Modal, so the charts every downstream experiment depends on carry no code-version provenance. | `_determinism.py:76-82` shells `git rev-parse HEAD` / `git status --porcelain` inside a bare `except Exception` returning `"unknown"`/`None`. `modal_phase0.py:26` `ignore=[".venv", ".git", "data", "hub", …]` — there is no `.git` in `/src`. | Read both. Found by the eval-integrity agent; the ignore list re-read and confirmed here. This is the one field that would let a later anomaly be attributed to a specific commit. |
| **P21** | COSMETIC | **Stale documentation on the headline v3 §5.2 fix and on the regimes.** | `run_e0.py:583-594` still states, as a "CAVEAT … (not fixed here)", that `--collect-num-act-stepped` *"does **NOT** change replan cadence or the resulting trajectory at all"* — and that it must stay at 1, contradicting §3.6's `nas=2`; the default is still 1. `run_e0.py:551-553`'s `--data-source` help says the same. `atlas/regimes.py:7` says *"R2 high restitution — `shape.elasticity` raised (was: `space.damping`)"* — **exactly inverted** from `REGIME_CONFIGS` 60 lines below (`regimes.py:76`: `"R2": {"damping": 0.5}`) and from §4. | Read the help strings against the loop they describe (`run_e0.py:301,314-316`) and the docstring against `REGIME_CONFIGS`. The flag **is** functional (`chunk=1/5→3/5→5/5` in the timing log). Same "comment asserts what the code no longer does" class as P1b and P15. |
| **P22** | COSMETIC | `evaluate_e0_chart` runs the open-loop rollout **twice** per trajectory. | `run_e0.py:456-457` applies the chart and calls `_open_loop_rollout` for `loss`; `:466` then calls `umf(...)`, which at `score.py:95-99` applies the chart and rolls out again on the same inputs. | Read both. Pure duplicate work under `no_grad`; apply/restore is balanced via `try/finally` in both places, so correctness is unaffected. Not on the critical path given P2. |

**Also checked and found correct** (worth recording so they are not re-audited): the determinism
import/call **ordering**; the per-`kind` pristine reset within a regime (`run_e0.py:757`, snapshot at
`:648`); physics re-application ordering in the `closed_loop` branch (goal render cannot leave the
env unshifted; `space.damping` is correctly set *after* `reset`, `regimes.py:124-133`); the
`collect_nas` chunk arithmetic (**exactly 25 raw steps, no off-by-one**, matching `keep_idx`); and
the **CEM budget match** — `p0g_collect` passes `300 / 10 / nas=2` and `phase0_v3/p0c/p0c_it10_summary.json`
records the adopted eval protocol as `num_samples 300, iterations 10, horizon 6, num_act_stepped 2,
min_block_pos_diff 40.0, max_agent_block_dist 160.0`. **Identical.** Credit this.

---

## §A — Did P0-G actually fix `COHERENCE_AUDIT_2.md` CHECK 2? (verified, item by item)

| CHECK 2 charge | Status | Evidence |
|---|---|---|
| `scripted`/`hybrid` optimise "walk to a random point near the block", no goal anywhere | **FIXED for chart training** | `run_e0.py:227-234`: draws a real `sample_dataset_init_goal` pair and calls `agent.set_goal(...)`; actions come from the real CEM `agent.act()`. **But see P7 — E1/E4 still call the collector with the `scripted` default.** |
| Acceptance `total_contacts > 0` conditions the training distribution on contact | **FIXED** | `run_e0.py:364` short-circuits for `closed_loop`, always accepting attempt 0. Confirmed in the timing log: `contact filter OFF (v3 §5.2)`. **But see P7 — retained everywhere the charts are judged.** |
| `--collect-num-act-stepped` documented as a **no-op** | **FIXED in code** | `run_e0.py:301` `for chunk_idx in range(0, n_chunks, collect_nas)`, `:314-316` `n_exec = min(frameskip·collect_nas, …)`. Confirmed empirically: `chunk=1/5 → 3/5 → 5/5` = 3 CEM searches, not 5. **But the help text still says it is a no-op — P21.** |
| Collected at 100×10 CEM against a 300×30 eval budget (a 9× gap) | **FIXED on `num_samples`, `iterations` and cadence** | Collection `300 × 10 × nas=2` matches P0-C's adopted eval protocol exactly. **But the gap moved axes — P4/P5: lookahead, episode length and goal separation now mismatch.** |

**Verdict: three of four charges genuinely closed, the fourth partially.** State this as progress in
the write-up. The residual is that "matches the eval protocol" is now true of the search budget and
replan cadence, and false of lookahead, episode length and goal separation.

---

## §B — THE CENTRAL SCIENTIFIC PROBLEM: under R2 the planner's objective is measured to be degenerate

§5.2 offers two justifications for the collector:

> **(a)** *"Goal-directed by identity, not by approximation. The generating objective is the
> planner's own L2-latent goal cost. There is no substitute objective to drift away from."*
> **(b)** *"Regime-exercising structurally. Reducing that cost requires displacing the block…"*

Both are statements about the objective's **form**. The project has already measured its **effect**,
and under R2 the effect is null:

| Row | Level | Measurement |
|---|---|---|
| **N3** | L4→L5 | Within-episode CEM cost-vs-true-goal-distance Spearman ρ: **R0 = 0.532** [0.388, 0.676] · **R2 = 0.001** [−0.132, +0.134], n = 20/regime |
| **N3-dose** | L4→L5 | ρ falls **0.532 / 0.295 / 0.169 / 0.078 / 0.001** monotonically across damping 0 → 0.5 |
| **N11** | **L5**, recomputed from raw per-candidate arrays | ρ ≈ 0 and **chart-invariant**: baseline 0.0014 ± 0.296, `ln_act` 0.0140 ± 0.287 over 20 seeds × 300 candidates; mean \|rank(cost) − rank(true_dist)\| = **99.76 of 300**, ≈ the ~100 expected from two independent random permutations |
| **N3b** | **L5** | Converged CEM under R2 lands **farther from the goal than the episode start**, 3/3 seeds |

**Independent corroboration derived here from the raw P0-C files**, which no document connects to
this: success rate *falls as CEM search effort rises*, on the same 20 paired seeds.

| iterations | SR | source |
|---|---|---|
| 10 | **10/20 = 0.50** | `phase0_v3/p0c/p0c_it10_baseline_R2.jsonl` — 20 rows, counted directly |
| 15 | **9/20 = 0.45** | `phase0_v3/p0c/p0c_it15_baseline_R2.jsonl` — 20 rows, counted directly |
| 30 | **8/20 = 0.40** | `atlas_out/e0_planning_nas2/baseline_R2.jsonl` — 20 rows, episodes 0–19, counted directly |

*Sample-constancy disclosure per `CLAUDE.md` §5:* all three arms are n = 20 on episodes 0–19 with
`seed == episode index` (`run_e0_planning.py:424`), same `regime_config {"damping": 0.5}`, same
`nas=2`, same `num_samples=300`, same `horizon=6` — verified by reading the episode indices and
config blocks out of all three JSONL/summary files. Only `iterations` differs. **Note also that
`phase0_v3/p0c/p0c_it30_baseline_R2.jsonl` is a 2-episode stub** — the real it=30 arm lives at
`atlas_out/e0_planning_nas2/`, which `EVIDENCE_LEDGER.md` §5 states but the `p0c/` directory does
not signpost. A reader following the directory finds the stub.

Each pairwise difference is individually within noise (`EVIDENCE_LEDGER.md` §5 records McNemar
p = 1.000 and p = 0.625). But the **direction is monotone and is exactly what N3/N11/N3b predict:
optimising harder against a cost decorrelated from the true objective makes outcomes worse.**

### What this does and does not mean

**It does not mean the data is garbage, and it does not mean the collector is self-confirming.** The
strongest defence survives intact and should be stated plainly: **the prediction targets come from
the physics engine, not from the model.** The planner picks actions; `PushTEnv` under
`PhysicsRegime` decides what happens. A state where `c₀` is wrong therefore enters the dataset *with
the correct target attached*. A planner with no uncertainty estimate cannot steer around its own
blind spot — it walks into it. **So the circularity attack in Part 2 item 5 fails, and it should be
recorded as failing.**

**What it does mean** is that §5.2's stated warrant is the wrong one:

- **(a) does not survive.** The objective did not need to be *substituted*; under R2 it
  **degenerated**. "Goal-directed by identity" is true of the definition and false of the behaviour
  — and that difference is the whole point of CHECK 2.
- **(b) does not survive.** *"Reducing that cost requires displacing the block [toward the goal]"* is
  precisely the proposition N11 measures at ρ ≈ 0 and N3b falsifies outright (converged CEM moves
  *away*).
- **(c) survives, and is now the only warrant.** "It is the deployment distribution" is true and
  sufficient to justify collecting this way. But (c) says the data is *on-distribution*; it says
  nothing about whether it is *informative*. That is now an empirical bet, and §C is how it is
  settled.

**The real residual objection is coverage, not circularity.** Collected actions are `argmin` over a
cost computed through the *unshifted* model, i.e. "R0-optimal sequences executed under R2 physics" —
a one-dimensional slice. Install the chart and the planner picks different actions and visits states
the chart never saw. §5.2 states this caveat and names E5's cross-policy matrix as its measurement —
but **E5 is item 1 on §11.3's pre-registered cut ladder**, so the caveat's only instrument is the
first thing budget pressure deletes.

**Second-order consequence nobody has stated.** Because the R2 action distribution is `argmin` over
a ρ≈0 cost, the collected R2 actions are closer to *structured noise executed under shifted physics*
than to *expert-quality goal-seeking under shifted physics*. Combined with **P11** (collection never
stops at success, so part of every 25-step R2 trajectory is post-goal jitter) and **P4** (the last
20% is planned at one-chunk lookahead), the **composition** of the on-policy set differs materially
from what §5.2 describes. P4 and P11 are free to fix; the ρ collapse is not, and must be disclosed.

**Required write-up change:** retract §5.2 bullets (a) and (b) for R2, keep (c), and state the
N3/N11 collapse explicitly. Per `CLAUDE.md` §1.8 this is a result to report, not a framing to
preserve.

---

## §C — The acceptance criterion, and the smallest checks that fix it

**Is eval UMF admissible as the gate on these charts? No.** N1 (44/100 vs 43/100, p = 1.000, UMF
falling monotonically), N4 (UMF 0.336→0.302→0.268 across a 5× data range, every planning CI spanning
zero), N11 (`lora4` has the *best* localized UMF and the *worst* SR), and `lora4`×R1 (UMF 0.242,
**4/10** vs an 8/10 baseline) together say the metric has demonstrated ~zero predictive value for
planning competence in this substrate **and has already failed to flag a chart that actively
destroyed planning.** Compounded by **P3** (the acceptance number *is* the selection number, with a
measured +0.077–0.157 bias), **P8** (wrong horizon scale) and **P10/P10b** (retired gate rule,
unrecorded `n`, optimistic direction).

Run **both** checks below. They answer different questions and neither is expensive.

### C-1 — the mechanism check (forward-only, ≈ $0, run first)

Recompute the within-episode CEM cost-vs-true-distance Spearman ρ **with the P0-G R2 chart
applied**, reusing the existing machinery (`scripts/diagnose_cem_costs.py`,
`scripts/analyze_cost_ranking.py`) over `atlas_out/cost_ranking_R2_v2/`'s 20 seeds × 300 candidates.

- **Both comparators are already on disk:** baseline ρ = 0.0014 ± 0.296; dataset-trained `ln_act`
  ρ = 0.0140 ± 0.287 (N11).
- **RED FLAG:** the P0-G chart's ρ remains indistinguishable from baseline (mean inside ±0.05 with a
  CI spanning zero). If the chart restores **no** cost-ranking signal under R2, it cannot improve
  planning under R2 whatever its UMF says — and E0′'s R2 arm is predictable before it is paid for.
- This is the **more diagnostic** of the two and it is forward-only. Run it first.

### C-2 — the catastrophe screen (≈ 36 min, ≈ $0.50 — the comparator already exists)

Run the P0-G `ln_act` chart through real CEM-planned R2 episodes at exactly the P0-C config —
`nas=2, N=300, iterations=10, horizon=6, max_steps=30`, `--regime-config '{"damping": 0.5}'`,
episodes/seeds **0–19** — via `scripts/run_e0_planning.py --kind ln_act`. **No new code.**

- **The paired baseline arm is already on disk:** `phase0_v3/p0c/p0c_it10_baseline_R2.jsonl`,
  n = 20, seeds 0–19, **SR 10/20**, `mean_wall_time_s = 108.88`. Only the chart arm costs anything:
  20 × 108.9 s ≈ **36 min ≈ $0.50** on L4.
- **Statistics:** `mcnemar_paired` + `paired_bootstrap` from `atlas/stats.py`, unmodified. Report the
  discordant-pair split, not just two rates.
- **RED FLAG — block the charts if either fires:**
  1. **Chart SR ≤ 5/20** (≥ 25 pp below the 10/20 baseline). This is the `lora4`×R1 signature
     (4/10 vs 8/10) and it is detectable at n = 20.
  2. **The mechanism statistic does not move:** knock-aways (episodes with final `block_pos_diff` >
     `init_block_pos_diff`) do not decrease **and** mean final block distance does not decrease.
     Both are already logged per episode by `run_e0_planning.py`. R2 was chosen because its failure
     mode is *overshoot*; a chart that lowers UMF while leaving overshoot untouched has, by this
     project's own stated theory, learned nothing that matters.
- **Honest power statement, which must travel with the result:** n = 20 paired detects roughly
  25–30 pp. Given N1's well-powered null the expected effect is ≈ 0, so **this is a catastrophe
  screen, not an efficacy test.** Passing means "not broken", never "works". Do not let a null here
  be written up as support.

*(The red-team agent independently proposed the same check at n = 30, ≈ $1.80–$3. n = 20 is the
efficient choice because the baseline arm at n = 20 is already paid for; going to n = 30 requires
extending the baseline too.)*

---

## §D — Scope: what dropping R1 actually forecloses

1. **E0′ as pre-registered is not runnable on on-policy charts.** §8.1 specifies R1 **and** R2, and
   its decision rule reads *"CI excludes zero in at least one regime"* — which becomes vacuous with
   one regime. That is a pre-registration change governed by `CLAUDE.md` §1.8.
2. **E2 cell B loses its R0/R1 replicate**, or runs it on dataset-collector charts — a two-variable
   change (regime **and** collector) of exactly the kind `OPUS_REMAINING_TASKS.md` #11 condemns in
   `hybrid`.
3. **The two-mechanism capacity argument in §4 is dropped, not deferred.** The paper can no longer
   say adapters were screened against both a tangential-contact shift and a post-contact-glide shift
   on comparable data.
4. **The S2 stream becomes strictly K = 2** (R0↔R2). `K_max` can never bind, RQ4's persistent-library
   recall is a two-element library, and §15-7's no-new-code K≥3 alternative
   (`R0 / damping-0.25 / damping-0.5`) needs a `damping=0.25` chart this collection does not
   produce. Add it now (+~2 h) or abandon it explicitly.

**The defensible answer is probably "drop R1"**, because G4 already downgraded it to prediction-level
only. **But that decision must be written into §8.1/§8.3 before launch.** A reviewer who sees a
pre-registered two-regime design reported as one, with no recorded decision, reads it as exactly the
goalpost-moving `CLAUDE.md` §1.8 forbids.

---

## §E — `damping = 0.5`: is the original calibration criterion still the right one?

**No, and a better justification is already available.** `E0_RECOVERY_PLAN.md` §0.4 selected
`damping = 0.5` on a **UMF-ratio-vs-R0** criterion (2.47×) measured under open-loop expert replay.
Two problems carrying that forward: (i) UMF ratio is the very quantity §C shows does not track what
charts are now supposed to earn admission for; (ii) it was measured under the collector P0-G exists
to replace.

The criterion that *does* hold up under on-policy collection is: **does the frozen planner exhibit a
learnable, directional deficit here?** On that criterion R2 is well justified independently — the
documented failure mode is systematic **overshoot** with a named mechanism (`c₀` was trained at
`damping = 0`, so commanded pushes are calibrated for a block that stops dead). That is a stronger
and more citable justification than "2.47×"; lead with it.

**Does P0-G resolve the contact-collapse confound §15-2 flags, or assume it away?** It *produces* the
measurement (`run_e0.py:736-739`) but **gates nothing on it and never persists it** (P12) — and per
**P1c** the R0-vs-R2 contact comparison from a single `--regimes R0,R2` call is confounded anyway.
So as configured: **assumed away, not resolved.**

*One partial mitigation, flagged by the red-team agent and not re-derived here:* it reports
`E0_RECOVERY_PLAN.md` as already containing a live-planner contact measurement of **12.5 → 6.9
(−45%)**, against replay's 38.5 → 13.3 (−65%) — i.e. real but milder with a planner in the loop, not
a collapse to zero. **UNVERIFIED by the main session**; check it before relying on it. If it holds,
it weakens this concern considerably but does not remove the need for the probe below, because
nobody has measured contacts at the P0-G config specifically (`nas=2, it=10`, 25 raw steps) and the
smoke ran R2 without reporting it.

**Fix, ordered before the money:** a 10-trajectory two-regime probe at the production config
(`--num-trajs 10 --num-val-trajs 0 --steps 1`, run as **two separate single-regime calls** per P1),
≈ 22 min ≈ **$0.30**. Read the R0 and R2 contact counts, apply §15-2's rule, then commit the R2
budget. Persist `n_contacts` into the manifest while there.

**One disclosure a reviewer will demand regardless:** `space.damping` is a **global world**
parameter, not an object property, and the checkpoint saw `damping = 0` in all 18,685 training
demos. This is an **extrapolation** shift, not an interpolation one. That makes it a *hard* shift —
which is fine — but a null must then be written as *"a 10.7k-parameter adapter could not absorb an
out-of-support global dynamics change"*, not as *"adapters do not help"*.

---

## §F — Answers to the specific questions asked, in order

| # | Question | Answer |
|---|---|---|
| 1 | What does the collector's CEM actually optimise? Any residual filter? Does `nas` really control replan cadence, or is `chunk=1/5→3/5→5/5` a logging artifact? | **Real goal**, via `sample_dataset_init_goal` + `agent.set_goal` — no proxy (§A). **No residual filter** — `run_e0.py:364` short-circuits for `closed_loop`. **`nas` is genuinely functional and the log line is real, not an artifact** (`run_e0.py:301,314-316`; confirmed against the timing log). One nuance: the red-team agent traced the objective to `planning/objectives.py:129-149` as terminal L2 in DINOv2 latent space to the goal *image* **plus `alpha = 0.1 ×` a proprio (agent-position) term**, while `block_success` deliberately discards the agent term as "pure noise" — making "goal-directed by identity" a mild overclaim even before §B. **This decomposition was not re-read by the main session — UNVERIFIED; verify before quoting.** |
| 2 | Train/eval seed disjointness for the full 216-trajectory run | **Clean at the launch config** — intervals computed from the generation expression, not inferred from a manifest (**P19**). Latent collision at `num_trajs ≥ 501` (or ≥ 126 if `--collect-max-tries` reverts to 8). Episode-level disjointness is unauditable at ~4.2% risk (**P18**); if R1 is ever added its seeds overlap the evaluator's (**P13b**). |
| 3 | Is `_determinism.py` on the exact path the full run uses? | **Yes** — `modal_phase0.py:77-89` → `run_e0.py:30-33` (imported before `torch`) → `:604` `make_deterministic(0)`. Ordering is correct. **But its SDPA mitigation is structurally inert** because the predictor wraps every SDPA call in `sdpa_kernel(ALL_SDPA_BACKENDS)` (**P15**), and `:640` undoes its TF32 pin (**P15b**). |
| 4 | Could the backward residual affect R0 and R2 differentially? | **Yes, plausibly — through early-stopping checkpoint selection, not through the weights.** Reasoned from `harness.py:222-233` and the effective validation `n` of 8; **not measured**. A ~6-minute check is specified in **P16** and has not been run. |
| 5 | Is closed-loop CEM the right way to get "on-policy" data? Is it circular? | **The circularity attack fails** — targets come from the physics engine, so the data contains `c₀`'s errors with correct labels. **The real objection is single-round DAgger coverage**, whose only instrument (E5) is item 1 on the cut ladder. Full argument in **§B**. |
| 6 | Is eval UMF a coherent acceptance criterion? If not, specify the check. | **No** — **P6**, and **§C** specifies two checks, both costed, both with numeric red flags. |
| 7 | Does R0+R2-only foreclose anything? | **Yes, two experiments and the K≥3 stream** — **P13**, **§D**. |
| 8 | Is 100 train + 8 val enough? | Effective validation `n` is **8** (40 chunks, but 5 per trajectory come from one compounding unroll, so not independent). That same set drives up to 80 early-stopping decisions **and** the reported number (**P3**), making the recorded bias identically zero. The only convergence evidence anywhere is the smoke — train −53% vs val −11% at step 60 on 5 trajectories, val still falling, patience never touched — and **no production-budget loss curve exists on disk**. So `--steps 2000` is unjustified in *both* directions. Restore the test split and archive `val_loss_*.json`. |
| 9 | Is contact-filter-off consistent downstream? | **No — it is fully reintroduced, and worse than previously recorded**: E1 and E4 calibrate on **`scripted`**, the retired goal-free collector; E2 and `phase0_measure` on contact-filtered `dataset`. **P7.** |
| 10 | Is the `damping = 0.5` calibration criterion still right? | **No** — **§E**. And P0-G assumes the confound away rather than resolving it. |

---

## WHAT TO DO, IN ORDER

**Before any GPU spend:**

1. **P1 — fix the regime-ordering contamination.** Zero-code option: **two separate single-regime
   calls** (`--regimes R0`, then `--regimes R2`) instead of `p0g_collect`'s `"R0,R2"` default. Code
   option: reload `pristine_predictor_state` immediately before collection in each regime. *(These
   two modes are not otherwise equivalent — CEM generator state also carries across regimes in a
   combined call.)*
2. **P3 — set `--num-test-trajs 8`.** ≈ $0.25. Without it the acceptance number is the selection
   number, with a measured +0.077–0.157 bias.
3. **P4 / P5 — align collection to the eval protocol**, or drop the claim that it is aligned. Pass a
   loose `steps_left` (matching `run_e0_planning.py:288`'s convention) and set `traj_len = 30` with
   goal separation 31. If not fixed, **state all three deviations explicitly** in §5.2 and the paper.
4. **P2 / P2c — fix the timeout and the cost line.** Raise `timeout` past ~14 h, or split collection
   and fine-tune into separate Modal functions (which also fixes the resume trap), or run the two
   regimes as concurrent calls — which P1's fix requires anyway. Correct `SMOKE_SUMMARY.md`'s $3.6
   to ~$11.
5. **P13 — record the R1 decision** against §8.1/§8.3.
6. **P16 — run the 6-minute determinism-asymmetry check.** Free, and currently an unexamined
   assumption feeding every chart.

**Then, as the first spend:**

7. **§E's 10-trajectory contact probe** (≈ 22 min, ≈ $0.30). Read R0 vs R2 contact counts, apply
   §15-2's pre-registered rule, *then* commit the R2 budget.

**Then collect and train. Then, before any chart touches E0′/E1/E2/E3+E4:**

8. **§C-1** (forward-only, ≈ $0) and **§C-2** (≈ 36 min, ≈ $0.50). **No chart enters a downstream
   experiment on eval UMF alone.**

**Also fix, cheaply, alongside:**

9. **P9 / P8** — persist the collected trajectories or a T=2 per-chunk score dump. Without it P0-G
   buys charts but *not* the on-policy chunk set that τ, the motion gate, the strike rate and σ_r are
   all owed a re-derivation against, and re-collecting costs the whole budget again.
10. **P12** — persist `n_contacts` into the manifest; fix the `"Real-demo replay"` label.
11. **P10 / P10b** — pin one motion-gate rule at one granularity; record `len(umf_scores)` in
    `results.json` the way `p0c_it10_summary.json` already records `umf_episodes_with_value`.
12. **P7** — fix E1's and E4's `source=` default before either runs. Not a P0-G blocker, but it
    silently reverses P0-G's central contribution downstream.
13. **P20, P21** — archive the git SHA some other way on Modal; fix the three stale docstrings that
    describe behaviour the code no longer has.

---

## Standing caveats on this document

- Every number above was read from a real file in this repository. Where a claim originated with a
  sub-agent and was re-derived here, the verification column says so; where it could not be
  re-derived — **P4b**, **P5**'s 79.0/90.7 px figures, **P11**'s 8.4-step magnitude, **P13b**'s
  39/100, **§E**'s 12.5→6.9 contact figures, **§F-1**'s objective decomposition — it is labelled
  **UNVERIFIED** and must not be quoted as established.
- **P2**, **P15b** and **P16** rest on stated reasoning plus measured data points, not on completed
  experiments. Each names the specific check that would settle it.
- No sample-constancy claim is made without stating the sample; see the disclosure block in **§B**
  for the P0-C comparison, per `CLAUDE.md` §5.
- Three of the four items on `CLAUDE.md`'s own pre-P0-G checklist are addressed here (item 2's
  determinism spot-check → **P15**; item 3's line-by-line collector read → **§A**, **P4**, **P5**;
  item 4's corrected cost estimate → **P2**). **Item 1 — the retroactive classification of every
  standing Phase-0 conclusion as verified-before-reported / verified-only-after-challenge /
  never-challenged — has no artifact anywhere in the repo and remains outstanding.**
