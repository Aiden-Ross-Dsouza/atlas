# P0-G FIX PLAN — handoff spec for the session that implements the pre-launch fixes

**Companion to `research_audit/P0G_REVIEW.md` (2026-08-29).** That file is the *findings*; this file
is the *work order*. Finding IDs (P1, P2, …) refer to that file's table. Read it first — every fix
below assumes you know what it is fixing and why.

**Status (updated 2026-08-29):** §2–§4 implemented in code by the V3-6/V3-7/V3-8 sessions; §7's two
defects fixed by V3-9 (§7-B1 verified model-free + regression test; §7-B2 fixed, mechanism verified
synthetically, real-data severity owed once collection produces `trajs_R2.pt`). See the per-section
STATUS boxes. **Still before launch:** a real Modal smoke (every model-in-the-loop falsification +
the timing/cost re-measure is still owed — see the STATUS boxes' "NOT RUN" lines), §4.3/§4.5's open
items, and human launch approval. P0-G has not launched.

---

## 0. THE PROMPT (paste this to start the session)

> Read, in this order: `CLAUDE.md`, `research_audit/IMPLEMENTATION_PLAN_V3.md` §3.6/§5/§6/§8/§15,
> `research_audit/EVIDENCE_LEDGER.md` §4 and §5, then `research_audit/P0G_REVIEW.md` in full, then
> `research_audit/P0G_FIX_PLAN.md` (this file).
>
> Your job is to implement the fixes in `P0G_FIX_PLAN.md` §2–§4, in the order given, so that P0-G
> can be launched safely. You are **not** authorised to launch P0-G itself, to commit, or to push.
> You are **not** authorised to change any value in `CLAUDE.md` §1.7, to decide the R1 scope
> question (§4.2 below), or to change E1/E4 collector behaviour (§4.3 below) — those three need
> explicit human sign-off and you must stop and ask.
>
> Follow `P0G_FIX_PLAN.md` §1's discipline for every single fix: **demonstrate the bug first, then
> fix it, then demonstrate the fix, then log it in `research_audit/FIXLOG.md` with the before/after
> you actually observed.** A fix with no recorded failing "before" is not accepted. Per
> `CLAUDE.md` §1.9, do not report any result you have not run.
>
> Report back as three explicit buckets: what you ran and it passed, what you ran and it failed,
> what you did not run.

---

## 1. NON-NEGOTIABLE DISCIPLINE FOR THIS WORK

Read this section twice. Most of the damage this repo has recorded came from skipping it.

### 1.1 Prove the bug before you fix it

For every fix below there is a **"FALSIFICATION TEST"** line. Run it *before* the change and confirm
it fails. Run it *after* and confirm it passes. If you cannot make the test fail before the fix, you
have not understood the bug — stop and say so rather than applying a change that demonstrates
nothing. This is the same rule `atlas-process-auditor` operates under and it exists because this
repo has shipped "fixes" for bugs that were never present.

### 1.2 Files that are additions-only — DO NOT MODIFY EXISTING LOGIC

Per `CLAUDE.md` (the 2026-08-27 authorisation and its limits):

- **`atlas/score.py::umf`** — off limits. You may *call* it with different inputs; you may not change it.
- **`atlas/stats.py`'s existing functions** — off limits.
- **`scripts/run_e0_planning.py::run_episode`'s planning loop** — **off limits, and this one is a
  trap.** §3.1 below asks you to make the *collector* match this loop. You will notice that this
  loop's `steps_left` is itself a 5× unit inflation (it passes 30 where the true remaining model-chunk
  count is 6). **Do not "fix" it.** It is the reference protocol that produced every planning number
  on disk, including the 44/100 and 8/20 baselines. Changing it invalidates all of them. Match it;
  do not correct it.

### 1.3 Everything else you touch must be logged

`research_audit/FIXLOG.md`, one entry per change: the defect, `file:line`, the change, which claim it
affects, the observed before/after from your falsification test, and whether a re-run is needed.
Nothing is fixed silently.

### 1.4 Sample-constancy disclosure

`CLAUDE.md` §5: whenever you report a "corrected" number against a "before" number, state explicitly
whether the seed set and `n` were held constant. Two separate G7 corrections were themselves
confounded by an unstated changed sample. One sentence prevents it.

### 1.5 The smoke numbers become obsolete

§3.1 and §3.2 change the collection protocol. **After those land, the smoke's `R0 UMF 0.171` /
`R2 UMF 0.396` are numbers from a different protocol.** Do not compare new output to them. Do not
carry them forward. Note the supersession in `phase0_v3/p0g_smoke_record/SMOKE_SUMMARY.md` rather
than editing the numbers out.

### 1.6 Line numbers here are approximate — locate by content

The line numbers in this file and in `P0G_REVIEW.md` were read on 2026-08-29 against a dirty working
tree. **Locate every edit site by grepping for the quoted code**, not by jumping to a line number.

### 1.7 Operational

- Modal runs: **always `--detach`.**
- **Archive every downloaded result into `atlas_out/` or `phase0_v3/` permanently before computing
  anything from it. Never `rm` a downloaded temp directory.** The repo has one permanently
  unverifiable result (`N7-pre`) because a re-run overwrote a directory in place.
- **Never reuse an output directory name for a re-run.**
- Do not `git commit` or `git push`.

---

## 2. STAGE 1 — do this first, it de-risks everything else

### 2.1 [P9] Persist collected trajectories and emit a T=2 chunk dump

> **STATUS 2026-08-29 (V3-6):** DONE in code — `--load-trajs`, `--collect-only`,
> `_traj_guard()`, `dump_regime_chunks()` in `run_e0.py`; persist + chunk dump
> wired into the regime loop; resume short-circuits collection when
> `trajs_{regime}.pt` exists. Guard roundtrip + mismatch-`ValueError` verified
> (model-free). NOT RUN: `dump_regime_chunks` end-to-end, the "`--load-trajs` →
> 0 CEM searches" smoke, the production step-rate measurement (all need the
> frozen checkpoint, which will not load locally this session).

**Why first:** collection is 4 h and ~$3 of the budget; fine-tuning is the part you will need to
re-run repeatedly while testing §2.2, §3.3 and §4.1. Right now `load_regime_trajectories` returns
in-memory tensors that are thrown away, so *every* fine-tune experiment re-pays for collection. Land
this and the rest of the work becomes nearly free. It also fixes the §5 deviation-note-1 promise
that P0-G would let τ / motion gate / strike rate / σ_r be re-derived on on-policy chunks.

**Change, in `scripts/run_e0.py::main()`:**

1. After the three `load_regime_trajectories` calls, save the trajectory list per regime to
   `args.out / f"trajs_{regime}.pt"` — `encoder_output`, `actions`, `proprio`, `seed`,
   `n_contacts`, and `block_pose` if present. Consider `.half()` on `encoder_output` to halve size
   (see the size note below); if you do, cast back to `float32` on load and **log that you did**.
2. Add `--load-trajs <dir>` which, when set, loads these instead of collecting. Guard it: assert the
   loaded manifest's `collect_cem`, `traj_len` and `regime_config` match the current args, and
   **refuse to run** on mismatch rather than silently training on a different protocol's data.
3. Emit `args.out / f"chunks_{regime}.jsonl"`: for each trajectory, every **T = `collect_nas`**
   sliding window (`nas=2` ⇒ windows `[0:3]`, `[1:4]`, …) with its per-window UMF under `c₀`, its
   latent displacement `‖z_T − z_0‖_F`, and its block displacement in px. **This is the artifact
   §6.1/§6.6 need to re-derive τ and the motion gate on-policy** (see §4.4).

**Size check before you commit to fp32:** `encoder_output` is `[T+1, 256, 384]` float32 ≈ 2.75 MB
per 25-step trajectory; ×108 trajectories ×2 regimes ≈ **~600 MB**. Fine for a Modal volume, but
measure it on the smoke rather than assuming — and it grows ~20% when §3.2 raises `traj_len` to 30.

**Move the resume check.** `run_e0.py`'s per-`(kind, regime)` resume branch (grep
`"⏩ [Resume]"`) currently sits *after* collection, so a resumed run re-collects ~2 h of trajectories
purely to skip a cached chart (P2c). With `--load-trajs` in place, restructure so collection is
skipped when both the trajectory file and the chart exist.

**FALSIFICATION TEST:**
```
# BEFORE: confirm nothing is persisted
python scripts/run_e0.py --kinds ln_act --regimes R0 --data-source closed_loop \
  --num-train-trajs 2 --num-val-trajs 1 --num-test-trajs 0 --steps 2 \
  --collect-num-samples 8 --collect-iterations 2 --collect-num-act-stepped 2 \
  --train-traj-len 25 --eval-traj-len 25 --out /tmp/p0g_t1
ls /tmp/p0g_t1     # must show NO trajs_*.pt and NO chunks_*.jsonl  ← the bug
# AFTER: same command must produce both; then re-run with --load-trajs /tmp/p0g_t1
#        and confirm ZERO CEM searches happen (no "collect_closed_loop" tqdm bar at all)
```
Run this locally on the 4050 (tiny CEM budget, CPU/GPU either way) — it costs nothing.

**Then, immediately, use it:** with trajectories cached, measure the real fine-tune step rate at a
production trajectory count. This is the one assumption `P0G_REVIEW.md` §P2 flags as UNVERIFIED:

```
# collect 100 trajs ONCE (~1.9 h, ~$1.6 — this is real spend, get approval), then:
python scripts/run_e0.py ... --load-trajs <dir> --steps 5      # read the tqdm step/s
```
Extrapolate to 2000 steps and record the number in FIXLOG. If it confirms ~8.6 s/step, §2.2's
timeout arithmetic holds. **If it does not, say so — do not quietly adopt whichever number is more
convenient.**

---

### 2.2 [P2, P2c] Fix the Modal timeout and the cost line

> **STATUS 2026-08-29 (V3-6):** DONE — `p0g_collect` split into `p0g_collect`
> (`--collect-only`, `timeout=3600*6`, one regime/call) + `p0g_finetune`
> (`--load-trajs`, `timeout=3600*10`); entrypoints `p0g-collect` / `p0g-finetune`.
> Old `$3.6 / 4.5 h` projection flagged as superseded in the code comment. NOT
> DONE: the replacement cost figure (needs §2.1's un-run step-rate measurement);
> `SMOKE_SUMMARY.md` supersession note (deferred with the smoke-dependent work).

**The problem:** `modal/modal_phase0.py::p0g_collect` has `timeout=3600 * 8`; the realistic
single-container total is ~13.6 h.

**Change:**
1. **Split `p0g_collect` into two Modal functions** — `p0g_collect` (collection only, writes
   `trajs_*.pt` + `chunks_*.jsonl`, needs §2.1) and `p0g_finetune` (loads via `--load-trajs`). This
   is strictly better than raising the timeout: a fine-tune failure no longer destroys the
   collection, and the fine-tune becomes independently re-runnable, which §4.1's determinism test
   needs.
2. Give each a timeout with real margin (collection `3600*6`, fine-tune `3600*10` per regime).
3. **Run one regime per call regardless** — §3.4 requires it for a different reason.
4. Correct `SMOKE_SUMMARY.md`'s projection from `$3.6 / 4.5 h` to the measured figure from §2.1's
   step-rate run, with a dated note. **Do not delete the old projection** — mark it superseded.

**FALSIFICATION TEST:** none needed for the split itself (it is structural), but you **must** record
in FIXLOG the measured step-rate that justifies the new timeout, per §2.1.

---

## 3. STAGE 2 — the four launch blockers

### 3.1 [P4] Make the collector's planning lookahead match evaluation

> **STATUS 2026-08-29 (V3-6):** DONE in code — `steps_left` now uses
> `run_e0_planning.py`'s loose convention (`(n_replans_target - replan_idx) *
> collect_nas`, `n_replans_target = raw_steps // collect_nas`). Arithmetic
> falsification RAN: OLD `[5,3,1]`/`[6,4,2]` → `plan_length` truncated; NEW
> `[30,28,26]` → `plan_length [6,6,6]` = eval. NOT RUN: the instrumented
> `plan_length`-shape check in-process; re-measured per-traj timing (feeds §2.2).

**The bug.** `scripts/run_e0.py`, in the `closed_loop` branch — grep
`steps_left=max(n_chunks - chunk_idx, 1)`. With `traj_len=25, frameskip=5, nas=2` this passes
`steps_left` = 5, 3, 1. The vendored planner does
`plan_length = min(self.horizon, steps_left)` (grep `plan_length = min` in
`hub/hub/facebookresearch_jepa-wms_main/evals/simu_env_planning/planning/planning/planner.py`), so
the three CEM searches plan **5, 3 and 1** model-steps ahead. Evaluation plans **6, 6, 6**.

**The fix.** Match `run_e0_planning.py::run_episode`'s convention. That loop computes
`n_replans_target = max_steps // num_act_stepped` where `max_steps` is in **raw** steps, then
`steps_left_model = (n_replans_target - replan_idx) * num_act_stepped`. The collector's equivalent,
expressed in the collector's own variables:

```python
n_replans_target = max((n_chunks * frameskip) // collect_nas, 1)
steps_left = max((n_replans_target - (chunk_idx // collect_nas)) * collect_nas, 1)
```

**Read §1.2 again before you write this.** The eval convention is a 5× unit inflation. You are
reproducing it deliberately so that collection and evaluation are byte-for-byte comparable. Put that
in the comment, cite `EVIDENCE_LEDGER.md` §4's N5 row, and **do not touch the eval side.**

**FALSIFICATION TEST — this one is important and easy to get wrong.** Do not test by eyeballing the
tqdm bar. Instrument directly:

```
# Temporarily log the planner's actual plan_length per search. The cleanest hook is
# run_e0_planning.py's existing --log-planner-diagnostics pattern: agent._prev_elite_losses_mean
# has shape [iterations, plan_length, ...]. Print its shape at each agent.act() in BOTH
# the collector and one eval episode, at identical config.
# BEFORE: collector prints plan_length 5, 3, 1 ; eval prints 6, 6, 6   ← the bug
# AFTER:  collector prints 6, 6, 6 ; eval unchanged at 6, 6, 6
```
Run at a tiny CEM budget (`--collect-num-samples 8 --collect-iterations 2`) — the *shape* is what
matters, not the search quality. Remove the instrumentation before you finish, and record both
printouts verbatim in FIXLOG.

**Cost consequence:** collection gets ~1.7× slower (searches at plan_length 6 instead of 5/3/1).
Re-measure the per-trajectory time and update §2.2's timeout and the cost line. Do not carry
forward the 66.8 s/traj figure — it was measured under the truncated horizon.

---

### 3.2 [P5] Match the episode length and goal separation

> **STATUS 2026-08-29 (V3-6):** DONE in code — `run_e0.py` closed_loop branch
> passes `traj_len=GOAL_TRAJ_LEN` (31) to `sample_dataset_init_goal` (was
> `traj_len`), + `min(seq_length) >= GOAL_TRAJ_LEN` assert; `modal_phase0.py`
> `traj_len`/`eval_traj_len` 25 → 30. Real-sampler falsification RAN (no model):
> OLD block-sep median/mean 75.7/83.2 px, NEW 82.9/89.7, eval ref 77.9/94.9 →
> fix moves collection onto the eval distribution. `min(seq_length)=49`, no
> filter change needed. NOT RUN: formal KS vs a fresh eval `episodes.jsonl`.

**The bug.** `modal/modal_phase0.py::p0g_collect` defaults `traj_len=25, eval_traj_len=25`. §3.6
pins `MAX_MPC_STEPS = 30` and `GOAL_TRAJ_LEN = 31`. Worse, `run_e0.py` forwards its `traj_len`
straight into `sample_dataset_init_goal` (grep `traj_len=traj_len,` inside the `closed_loop`
branch), so the goal is drawn **24** demo-steps from the init instead of **30**.

**The fix — two independent quantities, currently conflated:**
1. `traj_len` 25 → **30** in `p0g_collect` (both `traj_len` and `eval_traj_len`). 30 is divisible by
   `frameskip=5`, so `run_e0.py`'s divisibility assertion still passes. This gives 6 model chunks.
2. **Decouple the goal separation from the rollout length.** The call into `sample_dataset_init_goal`
   must pass **31** (`run_e0_planning.GOAL_TRAJ_LEN`), not `traj_len`. Import the constant rather
   than hardcoding `31`, so the two can never drift again.

**Careful — three traps here:**
- `sample_dataset_init_goal` with `traj_len=31` needs `seq_length >= 31`. Verified: all 18,685
  episodes in `data/pusht_noise/train/seq_lengths.pkl` qualify (`min = 49`). No filter change needed,
  but **assert it** rather than assuming.
- `run_e0.py`'s own `valid_eps` filter uses `seq_length >= traj_len + 1`. At `traj_len=30` that is
  31 — still all episodes. Fine, but check it, and note that this filter is **unused** on the
  `closed_loop` path (the sampler has its own).
- **Memory.** 6 model chunks instead of 5 means ~20% more activation memory in the training
  backward. `lora4` has OOM'd on this exact axis before (see `E0_RECOVERY_PLAN.md` P2a). P0-G only
  runs `ln_act` so it should be fine, but confirm peak GPU memory on the smoke before the full run.

**FALSIFICATION TEST:**
```
# BEFORE / AFTER: run the collector at --num-train-trajs 3 with the tiny CEM budget and print,
# per trajectory: len(raw_actions) and norm(goal_state[2:4] - init_state[2:4]).
# BEFORE: 25 raw steps, goals drawn 24 demo-steps apart
# AFTER:  30 raw steps, goals drawn 30 demo-steps apart
# Cross-check the AFTER goal-separation distribution against an eval run's
# init_block_pos_diff column in a real episodes.jsonl — the two distributions
# should now be statistically indistinguishable. Report a KS test or the two medians.
```
That cross-check is the real test. Matching the *parameter* is not the same as matching the
*distribution*; verify the distribution.

---

### 3.3 [P3] Restore the disjoint test split

> **STATUS 2026-08-29 (V3-6 + V3-7):** DONE in code — `--num-test-trajs 8` baked
> into `modal_phase0.py::_P0G_COMMON` + exposed as `num_test_trajs` param
> (can't be re-pinned to 0); `run_e0.py` `results.json` now records
> `eval_umf_source` ∈ {`test`, `val_ALIASED`, `error`} and the stale
> "from the disjoint TEST set (A4)" comment is conditioned on it. NOT RUN: the
> `eval_umf != val_umf` + observed-bias-sign check (needs model). Prior repo
> measurement (+0.077…+0.157 on R0) stands as expectation.

**The bug.** `p0g_collect` passes `"--num-test-trajs", "0"`, so `run_e0.py` falls into
`eval_loss, eval_umf = val_loss, val_umf` — the reported acceptance number is the early-stopping
selection number, and `results.json`'s recorded bias `eval_umf - val_umf` is **identically 0.0 by
construction**. This repo has already measured that bias at **+0.077 to +0.157** UMF
(`FIXLOG.md` A4's assertion block, `full`×R0: val 0.713 vs test 0.870).

**The fix.** `--num-test-trajs 8` in `p0g_collect`, and expose it as a parameter on `p0g_entry` so
it can never again be silently pinned to zero. Cost: 8 extra CEM trajectories per regime, ~9 min,
~$0.15 each.

**Also fix the labels** while you are in there: `run_e0.py`'s `results.json` field comments still say
`# from the disjoint TEST set (A4)`. Make them conditional or make the code refuse to emit
`eval_umf` at all when `test_trajectories` is empty — silently aliasing it to `val_umf` under a
comment that claims otherwise is how this got missed.

**FALSIFICATION TEST:**
```
# BEFORE: results.json shows eval_umf == val_umf exactly, and the smoke log shows
#         "Val Loss: X (best)" and "Eval Loss = X" identical to 6 dp.   ← the bug
# AFTER:  eval_umf != val_umf; record the observed bias and its SIGN in FIXLOG.
```
**Do not treat a positive bias as a problem to fix.** It is the number A4 exists to measure. Report
it.

---

### 3.4 [P1, P1b] Fix the regime-ordering contamination — **the most important fix here**

> **STATUS 2026-08-29 (V3-6 + V3-7):** DONE (both halves). Operational:
> `p0g_collect` is single-regime (`regime: str`), entrypoint says run twice.
> Code: `run_e0.py` reloads `pristine_predictor_state` at the TOP of the regime
> loop, before collection; `Chart.restore_` re-apply behaviour + object identity
> (agent → wrapper → `wm.predictor`) confirmed at source. New
> `--debug-predictor-fingerprint` flag = the plan's falsification test, runnable
> later. NOT RUN: the R0-vs-R2 fingerprint check itself (needs model).

**The bug.** In `run_e0.py::main()` the order is: `for regime in args.regimes:` → three
`load_regime_trajectories` calls (which run the CEM collector) → `for kind in args.kinds:` →
`wm.predictor.load_state_dict(pristine_predictor_state)`. The pristine reload is *inside* the kind
loop, i.e. **after** that regime's collection. And `run_e0_finetune` leaves the predictor dirty:
`Chart.restore_()` for `kind != "lora4"` is literally `self.apply_(predictor)` — it re-applies the
chart's **trained** weights (documented at `atlas/chart.py`, grep `restore_pretrained_` and read its
docstring). So with `--regimes R0,R2`, R2's on-policy trajectories are planned by an **R0-adapted**
predictor, violating §5.2's "against the frozen `c₀` predictor" and `run_e0.py`'s own comment
claiming exactly that property.

**The fix — do BOTH, they are not redundant:**
1. **Code:** add `wm.predictor.load_state_dict(pristine_predictor_state)` immediately *before* the
   collection block inside the regime loop, with a comment naming this finding. One line. Note the
   `collector_agent` holds a reference to `wrapper`, whose `.model.predictor` is the same object, so
   an in-place `load_state_dict` is correctly seen by the agent — verify that rather than assuming.
2. **Operational:** run **one regime per Modal call** anyway. §2.2 requires this for the timeout, and
   it also isolates the CEM generator state, which likewise carries across regimes in a combined
   call. The code fix alone does not address that.

**FALSIFICATION TEST — do this one properly, it is the highest-stakes fix in the file:**
```
# Instrument: right before the collection block, print a stable fingerprint of the predictor:
#   print(regime, hashlib.sha256(
#       b"".join(p.detach().cpu().numpy().tobytes() for p in wm.predictor.parameters())
#   ).hexdigest()[:16], flush=True)
# Run: --kinds ln_act --regimes R0,R2 --num-train-trajs 1 --num-val-trajs 1 --steps 3
#      with the tiny CEM budget.
# BEFORE: the R0 fingerprint and the R2 fingerprint DIFFER.        ← the bug, proven
# AFTER:  they are IDENTICAL.
```
If the fingerprints already match before your fix, **stop** — either the bug is not present on your
tree or your instrumentation is in the wrong place. Do not proceed on the assumption that it is
there.

**One more thing to check while you are here:** confirm the fingerprint is also identical at the
*start* of each kind's fine-tune (the existing `:757` reload). That path was audited as correct; a
second pair of eyes on it costs nothing.

---

## 4. STAGE 3 — before any chart is trusted downstream

### 4.1 [P16] The determinism-asymmetry check — free once §2.1 lands

> **STATUS 2026-08-29 (V3-7):** PREPARED, NOT RUN. `modal_phase0.py::p0g_finetune`
> gained `load_subdir` so one cached collection feeds two runs → separate
> `det_run1`/`det_run2` out dirs (never reuse a dir, §1.7); procedure in the
> docstring. Needs a real collection + model + GPU first. P16 stays
> reasoning-only — not marked refuted/confirmed in `P0G_REVIEW.md`.

`P0G_REVIEW.md` P16 argues, **from reasoning and not from measurement**, that the ~1e-2 backward
residual acts through early-stopping *checkpoint selection* on an 8-trajectory validation set, and
so is not symmetric across R0 and R2. Nobody has tested it.

```
# With trajectories cached (§2.1), run the SAME fine-tune twice in two separate processes:
python scripts/run_e0.py --load-trajs <dir> --regimes R2 --kinds ln_act \
       --steps 200 --num-train-trajs 20 --out <dir>/det_run1
#   ... and again to <dir>/det_run2   (NEVER reuse the output dir — §1.7)
# Compare val_loss_ln_act_R2.json's "stopped_early_at_step" and results.json's "eval_umf".
```
Repeat for R0. **Report the two deltas side by side.** If R2's `eval_umf` moves materially more than
R0's across launches, P16 is confirmed and the charts carry a regime-dependent selection noise that
must be disclosed. If not, P16 is refuted — **say so explicitly and mark it refuted in
`P0G_REVIEW.md`.** A refuted finding is a result.

*Sample-constancy note for this one (§1.4): the only thing differing between run1 and run2 is the
process launch. Same trajectories, same seeds, same steps. Say that in the report.*

---

### 4.2 🛑 [P13] The R1 scope decision — **STOP AND ASK, do not decide this**

> **STATUS 2026-08-29 (V3-7):** **DECIDED — R1 DROPPED** (explicit human text
> sign-off; an earlier `AskUserQuestion`-based record was reverted after a Stop
> hook flagged it, then re-applied on the user's written confirmation). Written
> into IMPLEMENTATION_PLAN_V3 §8.1 (Arms + decision rule → `{R2}`) and §8.3 (cell
> B replicate struck). Basis: G4 = prediction-level only; R1 collection ∩ eval =
> 50/100 tasks (re-verified model-free; §3.2 fix raised it 15→50). P0-G runs R2
> only — `p0g_collect` already defaults to `regime="R2"`.

§8.1 pre-registers E0′ on **R1 and R2** with on-policy charts; §8.3 wants an R0/R1 replicate for
cell B; `p0g_collect` defaults to `R0,R2`. The defensible answer is probably "drop R1" because
P0-F/G4 found R1 to be prediction-level only — but that is a **pre-registration change** governed by
`CLAUDE.md` §1.8 and it belongs to a human.

Prepare the decision, do not make it. Present: (a) the cost of including R1 (+108 trajectories ≈
2.0 h collection + ~4.8 h fine-tune, and note both figures move under §3.1/§3.2); (b) the G4 evidence
that an R1 planning arm may be uninformative a priori; (c) the fact that **if R1 is added, its
collection seeds overlap the planning evaluator's** — `seed_base["R1"] = 0` and
`run_e0_planning.py` uses `seed == episode index` (P13b; a sub-agent's replay put the demo-episode
overlap at 39/100, **which was not independently re-verified — re-verify it before presenting it**).
If R1 is approved, `seed_base["R1"]` must move first.

Whatever is decided, **write it into §8.1 and §8.3** with a date. Do not leave it as a launcher
default.

---

### 4.3 🛑 [P7] E1/E4's collector default — **STOP AND ASK**

> **STATUS 2026-08-29 (V3-8):** the no-approval work is DONE — `source="scripted"`
> now explicit at `run_e1.py`, `run_e4.py`, `smoke_e4.py`; `run_e1.py` writes
> `e1_run_meta.json` (gate_source), `run_e4.py` summary gains `gate_source`. The
> default is **NOT changed**. STILL OPEN for the human: matching P0-G's charts
> means `source="closed_loop"` for gate calibration = +1 CEM search per gate
> trajectory (~30/arm for E4). Same class of decision as §4.2.

`scripts/run_e1.py` and `scripts/run_e4.py` call `load_regime_trajectories(...)` for their motion-gate
calibration **with no `source=` argument**, and the signature default is `"scripted"` — the goal-free
contact-seeking random walk that §5.2 declares *retired*. `run_e2.py` and `phase0_measure.py` use
`"dataset"`, which is still contact-rejection-sampled. So P0-G removes the contact bias from chart
*training* and every place the charts are *judged* keeps it.

**Do not silently change these defaults.** Changing E1/E4's gate source changes downstream numbers
and is a scope decision. What you *should* do now, without approval:

1. Make the `source` argument **explicit** at all four call sites (passing the value they already
   use), so no future reader has to know the signature default.
2. Log the source into each script's output manifest.
3. Write the trade-off up for the human: matching the charts means `source="closed_loop"` for the
   gate calibration, which costs a CEM search per gate trajectory.

---

### 4.4 [P8, P10, P10b] The reported metric — make it comparable and auditable

> **STATUS 2026-08-29 (V3-8):** DONE in code (additive, `score.py::umf`
> untouched). `evaluate_e0_chart` returns a dict; `results.json` now carries
> `eval_umf_chunkT{nas}` (τ-scale, P8), `eval_umf_ungated` +
> `eval_umf_chunkT{nas}_ungated` (P10), `eval_n_trajs/n_umf/n_umf_chunkT{nas}/
> n_windows` (P10b), `motion_gate_value` + `motion_gate_rule` (names it RETIRED
> per §6.6). Model-free unit-checked. NOT RUN: the "gate high → n=2 while loss
> over 3" check (needs model).

Three separate problems with what P0-G reports, all fixable without touching `umf`:

1. **Wrong horizon scale (P8).** `eval_umf` is a **5**-model-step (soon 6) open-loop rollout; τ =
   0.262 and the motion gate 242.7 were measured on **T=2** chunks. The reported number is not on τ's
   scale. **Fix:** in `evaluate_e0_chart`, *additionally* compute a T=`collect_nas` windowed UMF by
   slicing `encoder_output`/`actions` into 2-step windows and calling the existing `umf` on each.
   Report both, clearly named (`eval_umf_trajT` and `eval_umf_chunkT2`). This is purely additive —
   you are calling `umf`, not changing it (§1.2).
2. **Retired gate rule (P10).** `run_e0.py` computes `compute_motion_gate(train_displacements)` at
   the default `percentile=10.0` over one displacement per **whole trajectory**. §6.6 replaces this
   with P95 over **block-static chunks at T = `num_act_stepped`**. **Do not try to re-derive §6.6's
   gate inside P0-G** — that is what §2.1's `chunks_*.jsonl` is for, and it is a separate Phase-0
   task. For now: log the gate value and the rule that produced it into `results.json`, and
   additionally report the **ungated** mean UMF so the gate's effect is visible rather than baked in.
3. **Unrecorded `n` (P10b).** `evaluate_e0_chart` appends to `umf_scores` only when `umf` returns
   non-`None`, so `eval_loss` and `eval_umf` can be means over different subsets and nothing records
   which. **Fix:** record `len(umf_scores)` alongside — `phase0_v3/p0c/p0c_it10_summary.json` already
   does exactly this as `"umf_episodes_with_value": 18`; copy that convention. Note the gate drops
   *low*-displacement trajectories, which have the smallest UMF denominator and hence the largest
   UMF, so the gated mean is **systematically optimistic** — same direction as P3, compounding it.

**FALSIFICATION TEST:** on a 3-trajectory smoke where you set the gate artificially high enough to
drop one trajectory, confirm `results.json` records `n = 2` while `eval_loss` is still a mean over 3.
That is the bug made visible. Then confirm the fix records both `n`s.

---

### 4.5 [§C of P0G_REVIEW] The chart acceptance checks — **run these before any chart goes downstream**

This is the point of the whole review. Charts must not enter E0′/E1/E2/E3+E4 on eval UMF alone.

**C-1 — mechanism check (forward-only, ≈$0, run first).** Recompute the within-episode CEM
cost-vs-true-distance Spearman ρ **with the P0-G R2 chart applied**, using the existing
`scripts/diagnose_cem_costs.py` / `scripts/analyze_cost_ranking.py` over
`atlas_out/cost_ranking_R2_v2/`'s 20 seeds × 300 candidates.
- Comparators already on disk: baseline ρ = 0.0014 ± 0.296; dataset-trained `ln_act` ρ = 0.0140 ±
  0.287 (`EVIDENCE_LEDGER.md` N11).
- **RED FLAG:** the P0-G chart's ρ stays indistinguishable from baseline (mean inside ±0.05, CI
  spanning zero) ⇒ the chart restores no cost-ranking signal under R2 and cannot improve planning
  there whatever its UMF says.
- **Compute ρ per-seed and report the mean ± sd. Do not report the pooled-across-seed ρ** —
  `EVIDENCE_LEDGER.md` N11 states explicitly that the pooled value (≈0.25–0.27) is driven by
  across-episode goal-difficulty variance, not within-episode discrimination. Quoting the pooled
  number as evidence of signal is a named trap in this repo.

**C-2 — catastrophe screen (≈36 min, ≈$0.50).** Run the P0-G `ln_act` chart through real CEM-planned
R2 episodes at exactly the P0-C config — `nas=2, N=300, iterations=10, horizon=6, max_steps=30`,
`--regime-config '{"damping": 0.5}'`, episodes **0–19** — via `scripts/run_e0_planning.py --kind
ln_act`. No new code.
- **The paired baseline arm already exists:** `phase0_v3/p0c/p0c_it10_baseline_R2.jsonl`, n=20,
  seeds 0–19, **SR 10/20**, `mean_wall_time_s = 108.88`. Only the chart arm costs anything.
- Statistics: `mcnemar_paired` + `paired_bootstrap` from `atlas/stats.py`, **unmodified** (§1.2).
  Report the discordant-pair split, not just two rates.
- **RED FLAG — block the charts if either fires:** (1) chart SR ≤ **5/20**, i.e. ≥25 pp below
  baseline — the `lora4`×R1 signature (4/10 vs 8/10); (2) knock-aways (final `block_pos_diff` >
  `init_block_pos_diff`) do **not** decrease **and** mean final block distance does **not** decrease.
- **The power statement must travel with the result, in the paper and in every summary:** n=20 paired
  detects ~25–30 pp. Given N1's well-powered null (44/100 vs 43/100) the expected effect is ≈0. **This
  is a catastrophe screen, not an efficacy test. A null here means "not broken", never "works".** Do
  not let a null be written up as support — that would be exactly the `CLAUDE.md` §1.8 violation this
  whole audit exists to prevent.

---

### 4.6 The cheap items — do them, they take minutes

> **STATUS 2026-08-29 (V3-8):** **P12** (n_contacts → manifest + label fix),
> **P18** (`return_indices` → real episode_idx/offset for closed_loop; additive),
> **P19** (train/val/test seed-disjointness assertion, within + across regimes),
> **P20** (git SHA read client-side, passed as `ATLAS_GIT_SHA`), **P21** (3 stale
> docstrings: regimes.py R2, `--collect-num-act-stepped`, `--data-source`; the
> nas default also 1→2) — ALL DONE, compile-clean, model-free-checked.
> **P15/P15b** (doc-only), **P17**, **P22** (cosmetic) — DEFERRED.

| ID | Fix |
|---|---|
| **P12** | Persist `n_contacts` per trajectory into `e0_seed_manifest.json` (currently stdout-only, so §15-2's pre-registered damping check is unreadable from artifacts). Fix the `"Real-demo replay contact rate"` label, which is wrong for `closed_loop`. |
| **P18** | Record the accepted `ep_idx` and `offset` from `sample_dataset_init_goal` into the trajectory dict so `episode_idx` is not `null` for `closed_loop`. ~3 lines. Closes the auditability gap permanently and unblocks `scripts/audit_e0_train_planning_overlap.py`, which currently cannot run on on-policy manifests. |
| **P19** | Add a runtime assertion that the train/val/test seed intervals are disjoint within and across regimes. Currently clean at the launch config but collides at `num_trajs ≥ 501` (or ≥126 at the non-`closed_loop` `max_tries=8` default), with no check anywhere. |
| **P20** | The manifest will record `git_commit: "unknown"` on Modal because `modal_phase0.py`'s image `ignore` list excludes `.git`. Fix by reading the SHA **locally** in the `@app.local_entrypoint` and passing it as a parameter into the remote function. |
| **P21** | Three stale docstrings that state the opposite of what the code does: `run_e0.py`'s `--collect-num-act-stepped` help (still says the flag is a no-op — it is the headline v3 §5.2 fix and it works); the same file's `--data-source` help; and `atlas/regimes.py`'s module docstring, which says R2 is `shape.elasticity` while `REGIME_CONFIGS` 60 lines below says `{"damping": 0.5}`. |
| **P22** | `evaluate_e0_chart` runs `_open_loop_rollout` twice per trajectory (once for `loss`, once inside `umf`). Correctness is fine (apply/restore is `try/finally`-balanced both times); it is pure duplicated work. Low priority — fix only if it is free. |
| **P15, P15b** | **Documentation, not code.** `_determinism.py`'s SDPA lines are structurally inert because the predictor wraps every SDPA call in `sdpa_kernel(ALL_SDPA_BACKENDS)` (`hub/.../app/plan_common/models/vit.py`, grep `ALL_SDPA_BACKENDS`). **Do not patch the vendored jepa-wms** — `CLAUDE.md` §1.3 allows exactly one upstream hook and this is not it. Correct the module's own comment to say the mitigation does not take on this model and why. Separately, `run_e0.py`'s `torch.set_float32_matmul_precision("high")` undoes the module's TF32 pin; TF32 is deterministic so this is not a reproducibility break — **fix the docstring's claim, keep the speed.** |

---

## 5. LAUNCH SEQUENCE, once the above is done

1. **§4.2 and §4.3 answered by a human**, in writing, in the relevant plan sections.
2. **The contact probe, ordered before the money** — 10 trajectories per regime at the production
   config, run as **two separate single-regime calls**, ≈22 min ≈$0.30. Read the R0 and R2 contact
   counts and apply §15-2's pre-registered rule. If R2's on-policy contact count collapses toward
   zero, `damping = 0.1` is the pre-registered fallback and **the R2 budget must not be committed at
   0.5**. This measurement is worthless from a combined `R0,R2` call (§3.4).
3. **Collection**, one regime per call, `--detach`, artifacts downloaded and archived immediately.
4. **Fine-tune** from cached trajectories.
5. **§4.1** determinism check, **§4.5 C-1**, then **§4.5 C-2**.
6. Only then does any chart go near E0′/E1/E2/E3+E4.

---

## 6. THE FIVE WAYS THIS GOES WRONG — read before you start

1. **You "fix" `run_e0_planning.py`'s `steps_left`.** It looks like a unit bug. It is. It is also the
   reference protocol behind every planning number on disk. Touch it and 44/100, 8/20, 10/20 and
   every E0/E2 number become incomparable. **Match it, do not correct it.** (§1.2, §3.1)
2. **You apply a fix without first making its test fail.** Half the "fixes" in this repo's history
   addressed bugs that were not there, or missed the bug that was. Every fix in §2–§4 has a
   FALSIFICATION TEST line. Run the BEFORE. (§1.1)
3. **You compare new numbers to the smoke's 0.171 / 0.396.** §3.1 and §3.2 change the collection
   protocol; those numbers are from a different experiment. Also, per §3.4, the smoke's R2 number was
   produced from R0-contaminated collection in the first place. (§1.5)
4. **You report a corrected number without saying what else changed.** §3.1, §3.2 and §3.3 all move
   at once. If you report a UMF delta after all three, state that three variables changed — do not
   attribute it to one. (§1.4, `CLAUDE.md` §5)
5. **You treat §4.5 C-2's null as a positive result.** It is powered to detect ~25–30 pp and the
   expected effect is ≈0. Passing it means the chart is not catastrophic. It does not mean the chart
   works, and writing it up that way is the exact failure mode `CLAUDE.md` §1.8 forbids.

**And one standing rule that sits above all of it:** `CLAUDE.md` §1.9 — before reporting any
quantitative result as a finding, you must have rerun it and diffed, or tested the specific mechanism
directly, or labelled it **UNVERIFIED**. That applies hardest to results that confirm what you
expected. Most of the wrong numbers this project has produced were plausible, not surprising.

---

## 7. ADDENDUM (2026-08-29) — two defects introduced by the V3-6/7/8 fixes

Added after a line-by-line review of the V3-6/V3-7/V3-8 diffs. **These are not findings from
`P0G_REVIEW.md`; they are new bugs created while fixing it.** §1's discipline applies unchanged.

**First, what is correct** — verified at source, not from the STATUS boxes, and listed so it is not
re-audited or "re-fixed": §3.1's `steps_left` is exactly right (`n_replans_target = (6×5)//2 = 15` →
`steps_left` 30, 28, 26, byte-identical to `run_e0_planning.py`'s eval loop, which was correctly left
untouched); §3.4's pristine reload is at the top of the regime loop, before collection, with a correct
object-identity note; §3.2 imports `GOAL_TRAJ_LEN` rather than hardcoding 31 and adds the
`min(seq_length)` assert; `dump_regime_chunks` correctly uses the chart-free `rollout_umf` at a call
site where the predictor genuinely is pristine; `sample_dataset_init_goal`'s `return_indices` is
additive with `default=False` so the eval path is untouched; `atlas/score.py` is unmodified
(`git status` confirms); `--collect-only` exits cleanly, so an empty `results` dict cannot raise and
skip `atlas_volume.commit()`. Everything compiles.

---

### 7-B1 🔴 BLOCKS THE RUN — the `--load-trajs` guard rejects every fine-tune invocation

> **STATUS 2026-08-29 (V3-9):** FIXED exactly as specified. `scripts/_p0g_spec.py`
> (`_P0G_DEFAULTS` + `_p0g_flags` + `_P0G_COMMON`, no `modal` import); both
> `p0g_collect` and `p0g_finetune` default off `_P0G_DEFAULTS` and emit
> `_p0g_flags`. `run_e0._build_parser()` extracted. FALSIFICATION ran model-free:
> BEFORE `_traj_guard` unequal on the 4 predicted fields; AFTER equal. Permanent
> regression test `tests/test_p0g_guard.py` (passes; suite 22/22). Guard not
> weakened.

**The defect.** `run_e0.py::_traj_guard()` compares nine protocol fields and raises `ValueError` on
any mismatch (grep `--load-trajs protocol mismatch`). But `modal_phase0.py::p0g_finetune` emits only
`--regimes`, `--load-trajs`, `--steps`, `--out` plus `_P0G_COMMON`. Four collection-defining flags
therefore fall back to **argparse defaults**, which are not what `p0g_collect` used:

| guard field | stored by `p0g_collect` | current at `p0g_finetune` | source of the current value |
|---|---|---|---|
| `train_traj_len` | **30** | **25** | `--train-traj-len` default |
| `eval_traj_len` | **30** | **50** | `--eval-traj-len` default |
| `num_train_trajs` | **100** | **20** | `--num-train-trajs` default |
| `collect_cem` | **`300x10 nas=2`** | **`300x30 nas=2`** | `--collect-iterations` default 30 |

**Consequence:** `p0g_finetune` fails 100% of the time. That kills §2.2's collect/finetune split,
§4.1's determinism-asymmetry check (`load_subdir` → `det_run1` hits the same guard), and P2c's resume
path (the `elif traj_file.exists()` branch fires identically). The entire de-risking layer is
non-functional.

**The fix — make drift structurally impossible, not just correct today.** In `modal/modal_phase0.py`:

```python
# Every value that feeds run_e0.py::_traj_guard lives here ONCE. Both
# p0g_collect and p0g_finetune read their defaults from this dict, so the two
# can never drift into a spurious "protocol mismatch" again (§7-B1).
_P0G_DEFAULTS = dict(traj_len=30, eval_traj_len=30, num_trajs=100,
                     num_val_trajs=8, num_test_trajs=8,
                     num_samples=300, iterations=10, nas=2)

# NOTE: --num-test-trajs moves OUT of _P0G_COMMON and into _p0g_flags below,
# so it is emitted exactly once.
_P0G_COMMON = ["--kinds", "ln_act", "--data-source", "closed_loop"]


def _p0g_flags(traj_len: int, eval_traj_len: int, num_trajs: int, num_val_trajs: int,
               num_test_trajs: int, num_samples: int, iterations: int, nas: int) -> list[str]:
    """The guard-relevant flags. BOTH p0g_collect and p0g_finetune must emit
    these identically — otherwise --load-trajs raises a protocol mismatch that
    is really just argparse defaults (§7-B1)."""
    return ["--num-train-trajs", str(num_trajs),
            "--train-traj-len", str(traj_len),
            "--num-val-trajs", str(num_val_trajs),
            "--num-test-trajs", str(num_test_trajs),   # P3: disjoint test split
            "--eval-traj-len", str(eval_traj_len),
            "--collect-num-samples", str(num_samples),
            "--collect-iterations", str(iterations),
            "--collect-num-act-stepped", str(nas)]
```

Then give **`p0g_finetune` the same eight parameters**, defaulted from `_P0G_DEFAULTS`, and emit
`*_p0g_flags(...)` in its command; do the same in `p0g_collect`; and thread all eight through both
`@app.local_entrypoint` wrappers so a CLI override reaches the remote function. Defaulting both
signatures off `_P0G_DEFAULTS` is the part that prevents this recurring — a helper alone still lets
the two signatures drift.

**Do not "fix" this by weakening the guard.** Dropping the count fields (they are derivable from
`len(blob["train"])`) would make this specific failure go away while removing a real protection
against training on a different protocol's data. Thread the parameters.

**FALSIFICATION TEST — model-free, no GPU, run it before and after:**
```python
# Rebuild both argument namespaces exactly as modal_phase0.py invokes run_e0.py
# (parse the two cmd lists through run_e0.py's own parser), then:
#   assert _traj_guard(collect_args, "R2") == _traj_guard(finetune_args, "R2")
# BEFORE: fails, printing 4 mismatched fields
#         {'train_traj_len': (30, 25), 'eval_traj_len': (30, 50),
#          'num_train_trajs': (100, 20), 'collect_cem': ('300x10 nas=2', '300x30 nas=2')}
# AFTER:  passes.
```
**Land this test in `tests/` rather than deleting it.** It is a permanent regression guard on a
failure mode that is invisible until a 4-hour collection has already been paid for.

---

### 7-B2 🟠 FIX BEFORE TRUSTING CHARTS — the τ-scale chunk UMF is gated at the wrong granularity

> **STATUS 2026-08-29 (V3-9):** FIXED. `run_e0.py::main()` computes
> `chunk_motion_gate` (10th pct of T=nas train-window disps); `evaluate_e0_chart`
> uses it for the windowed calls only; `_umf_detail_fields` records both gates;
> `eval_umf_chunkT{nas}_ungated` kept. FALSIFICATION: the exact real-data test
> **could not run — no `trajs_R2.pt` exists yet.** SYNTHETIC substitute (real
> `compute_motion_gate`, directed-motion latents) confirmed the mechanism:
> BEFORE traj-gate gates 100% of windows, AFTER chunk-gate gates 10%. **Real
> over-gating fraction is UNVERIFIED — owed once collection produces
> `trajs_R2.pt`.** Not refuted: the mismatch is real by construction and §6.6
> mandates the fix regardless of magnitude.

**The defect.** §4.4 added T=`nas` sliding-window UMF so the reported number is comparable to
τ ≈ 0.262. But `evaluate_e0_chart` passes the **trajectory-scale** `motion_gate` into those window
calls (grep `w_g = umf(`), while `motion_gate` is computed as the 10th percentile of
*whole-trajectory* latent displacement — now T=6 over 30 raw steps (grep
`motion_gate (10th pct of train displacement)`). A 2-step window's displacement is much smaller than
a 6-step trajectory's, so the trajectory-scale threshold rejects nearly every window:
`eval_umf_chunkT2` will come back `nan`, or as a small and upward-biased subset.

This is exactly the failure §6.6 names — *"calibrated at a granularity it is not applied at"* —
reintroduced by the fix intended to address it. **Note this is partly a defect in §4.4's own wording,
which said "slice into 2-step windows and call `umf`" without specifying which gate the windowed call
should use. The implementing session followed the spec; the spec was ambiguous.**

**The fix.** Derive a *second* gate at the window scale from the same training trajectories, and use
it for the windowed calls only. In `run_e0.py::main()`, beside the existing `motion_gate`:

```python
# §7-B2: the T=nas windowed UMF must be gated at the window scale. The
# trajectory-scale gate above is a 10th percentile of T=6 displacement and
# rejects almost every 2-step window. Same RETIRED 10th-percentile rule
# (v3 §6.6's block-static P95 replacement is a separate Phase-0 task, fed by
# chunks_{regime}.jsonl) — but at the granularity it is actually applied at.
nas = args.collect_num_act_stepped
chunk_disps = torch.tensor([
    (t["encoder_output"][i + nas] - t["encoder_output"][i]).norm(p="fro").item()
    for t in train_trajectories
    for i in range(0, t["actions"].shape[0] - nas + 1)
])
chunk_motion_gate = compute_motion_gate(chunk_disps)
```

Pass `chunk_motion_gate` into `evaluate_e0_chart` as a separate argument used **only** for the
windowed `umf` calls; keep the trajectory gate on the trajectory-level call. Record **both** values
and both rules in `_umf_detail_fields` (`motion_gate_value`, `motion_gate_chunk_value`), and keep
reporting `eval_umf_chunkT{nas}_ungated` — that is the number that stays interpretable regardless.

**FALSIFICATION TEST — model-free once §2.1 has produced one `trajs_{regime}.pt`** (the encoder
outputs are persisted, so no checkpoint is needed):
```python
blob = torch.load("…/trajs_R2.pt", weights_only=False)
traj_disps  = [ (t["encoder_output"][-1] - t["encoder_output"][0]).norm(p="fro").item()
                for t in blob["train"] ]
chunk_disps = [ (t["encoder_output"][i+2] - t["encoder_output"][i]).norm(p="fro").item()
                for t in blob["train"] for i in range(t["actions"].shape[0] - 1) ]
# BEFORE: compute_motion_gate(traj_disps) sits far above the ~90th percentile of
#         chunk_disps  ->  the trajectory gate rejects nearly every window.  ← the bug
# AFTER:  compute_motion_gate(chunk_disps) gates ~10% of windows, as intended.
```
Report both thresholds and the realised gated fraction in FIXLOG. **If the BEFORE check shows the
trajectory gate does *not* over-gate windows, say so and treat B2 as refuted** — do not apply a fix
that demonstrates nothing (§1.1).

---

### 7-C Three operational notes for the smoke (not defects)

> **STATUS 2026-08-29 (V3-9):** all three actioned. C-1: `--num-test-trajs` is a
> real Modal param (default 8), smoke docstrings say `--num-test-trajs 2`. C-2:
> `SMOKE_SUMMARY.md` + `modal_phase0.py` carry a SUPERSEDED note — 66.8 s/traj /
> $3.6 stale-low, ~135 s/traj is an **estimate** pending the next smoke. C-3:
> `p0g_collect` + entrypoint docstrings state R0 collection is REQUIRED (τ/σ_r)
> and is a separate call. Timings NOT re-measured (needs a run).

1. **The smoke is not a smoke.** `_P0G_COMMON` bakes in `--num-test-trajs 8`, so
   `p0g_collect --num-trajs 5 --num-val-trajs 2` still collects **15** trajectories (~30 min).
   Pass `--num-test-trajs 2` for smoke runs. (The B1 fix makes this a real parameter.)
2. **Every timing figure in this file and in `SMOKE_SUMMARY.md` is now stale in one direction.**
   §3.1 raises the summed plan_length per trajectory from 5+3+1 = 9 to 6+6+6 = 18, so CEM compute per
   trajectory roughly **doubles**: ~66.8 s → ~135 s, giving ~4.3 h collection and ~5.8 h fine-tune per
   regime against the 6 h / 10 h timeouts. Both fit, with less margin than they appear to.
   **This is a first-order estimate from the plan_length ratio, explicitly NOT a measurement** — take
   the real per-trajectory time off the smoke and update §2.2, `SMOKE_SUMMARY.md` and the cost line
   from it, per §2.1's un-run step-rate task.
3. **R0 collection is now a separate manual invocation** (`p0g_collect` defaults to `regime="R2"`
   after the §4.2 R1 decision). τ is defined as P95 of `UMF(c₀)` over **R0** chunks and σ_r over the
   R0 informative set (§6.1, §6.3), so **both** `--regime R0` and `--regime R2` must be run. Nothing
   in the code guards against forgetting the R0 call.
