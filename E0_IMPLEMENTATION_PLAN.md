# ATLAS — Implementation Plan: repair the UMF rollout, then E0 + E1

*Written 2026-08-24. Executable task list for an implementing agent.*
*Companion document: `E0_DIAGNOSIS_AND_PLAN.md` — read that first for **why**. This file is **what to do**.*

---

## 0. Before you touch anything

**Read, in this order:** `CLAUDE.md` (the operating contract — §1 non-negotiables and §7
working rules bind you), `E0_DIAGNOSIS_AND_PLAN.md` (the diagnosis), then this file.

**Rules that override your defaults:**

- **Never `git commit` or `git push`.** Never run `reset --hard`, `clean -fd`, `checkout --`,
  `rebase`, or `commit --amend`. `status`/`diff`/`log`/`show` are free.
- **Never claim a gate or test passed without pasting the actual output.** "This should
  pass" is not evidence.
- **Do not change any hyperparameter in `CLAUDE.md` §1.7** (τ=0.5, q=3, m=0.05, n_probe=20,
  K_max=10, chart lr=5e-4). If a task below seems to require it, stop and ask.
- **Tasks marked 🛑 STOP require explicit user approval before proceeding.** These are the
  expensive GPU runs. Do the code, report, and wait.
- Work in the order given. T1 blocks everything; T2–T5 block T9 onward.
- Report each task as: what changed, which files, what you ran, what the output was, what
  you did *not* run.

**Capture a baseline first** so regressions are detectable:

```bash
git status
git stash list
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest tests/ -x -q 2>&1 | tail -30
```

Record the current test outcome verbatim in your first report — some tests encode the buggy
behaviour (see T1's note on `tests/test_score.py`) and are *expected* to fail after T1.

---

## The single root cause you are fixing

`atlas/score.py::_open_loop_rollout` is used by **both** UMF scoring (`score.py:78`) **and**
E0 chart fine-tuning (`atlas/harness.py:87,117`). It does not unroll this checkpoint the way
the checkpoint unrolls. Four defects:

1. **Time base wrong by 5×.** `VideoWM.encode_act` expects 10-dim *model* actions
   (`frameskip=5 × action_dim=2`) and reshapes blindly (`video_wm.py:212-213`). Callers pass
   raw `[T_raw, 2]`, so `[1,10,2] → [1,2,10]`, but the loop runs `for t in range(T_raw)`.
   For `t ≥ T_raw/5` the action slice is **empty**, `torch.cat` is a no-op, and the last
   action feature is re-fed. On the 50-step eval, **40 of 50 steps run on a stale action**.
   Targets are consecutive *raw* frames while each model step advances **5** raw frames.
2. **Proprio hard-zeroed** (`score.py:135`) — Push-T's proprio is the pusher's x,y, exactly
   what actions control.
3. **Context never accumulates** — always `z_cur[:, -1:]` (`score.py:148-151`) instead of
   `[:, -ctxt_window:]` with `ctxt_window=2`.
4. **A correct implementation already ships and is unused**: `EncPredWM.unroll()`
   (`vendor/jepa-wms/app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py:289-353`),
   which `GC_Agent` plans through (`gc_agent.py:175`).

---

## T1 — Rewrite `_open_loop_rollout` on `EncPredWM.unroll` 🔴 blocks everything

**File:** `atlas/score.py`

### Facts you can rely on (verified against the vendored source)

- `torch.hub.load(..., "dino_wm_pusht")` returns `(model, preprocessor)` where `model` is an
  **`EncPredWM`** (`hubconf.py:165`), with `ctxt_window=2` and `proprio_mode="predict_proprio"`
  (from `wrapper_kwargs` in the eval YAML, which sets only `ctxt_window: 2`; `proprio_mode`
  falls through to its default at `vit_enc_preds.py:265`). The inner `VideoWM` is
  `model.model`. **ATLAS currently keeps only the inner model and throws the wrapper away**
  (`run_e0.py:243`, `run_e1.py:198`: `wm = model.model if hasattr(model, "model") else model`).
- `EncPredWM.unroll(z_ctxt, act_suffix)`:
  - `z_ctxt`: TensorDict with `"visual"` `[B, tau, V, H, W, D]` and `"proprio"`
    `[B, tau, proprio_tokens, D]`.
  - `act_suffix`: `[T, B, A]` with `A = model action dim = 10`.
  - returns a TensorDict whose `"visual"` is `[T + tau, B, V, H, W, D]` — **context frames
    are prepended**, so the predictions are `out["visual"][tau:]`.
  - It is **not** decorated `@torch.no_grad()`, so gradients flow — required for E0 training.
- `EncPredWM.encode(obs)` **is** `@torch.no_grad()` (`vit_enc_preds.py:355`). That is fine:
  encodings are precomputed once, outside the training loop.
- `GC_Agent` starts from a **single** encoded frame (`gc_agent.py:175`, and
  `task_specification.num_frames: 1`), and `unroll` grows the context to `ctxt_window`
  naturally via `vid_feats[:, -self.ctxt_window:]`. **So starting from `tau=1` is correct and
  matches the planner exactly** — the bug was never the starting context, it was that the
  context never *grew*.

### What to write

Replace the body of `_open_loop_rollout` with a thin adapter over `EncPredWM.unroll`. Target
signature:

```python
def _open_loop_rollout(enc_pred_wm, z_ctxt, actions) -> torch.Tensor:
    """
    enc_pred_wm : EncPredWM  (the object torch.hub.load returns — NOT .model)
    z_ctxt      : TensorDict with "visual" [1, 1, V, H, W, D] and "proprio" [1, 1, P_tok, D]
                  (the encoded FIRST frame of the chunk)
    actions     : [T_model, 10]  model-chunk actions, normalized
    returns     : [T_model, N, D]  predicted visual latents, N = grid*grid
    """
```

Implementation notes:

- Reshape `actions` to `unroll`'s `[T, B=1, A]` layout.
- Call `enc_pred_wm.unroll(z_ctxt, act_suffix=...)`, take `out["visual"][1:]` (drop the one
  context frame), and flatten `V,H,W → N` to match the existing `[T, N, D]` contract that
  `umf()`'s numerator and `compute_trajectory_loss` expect.
- **Add a hard guard** rather than silently reshaping: assert
  `actions.shape[-1] == enc_pred_wm.model.action_dim` (10) and raise a clear error naming
  the raw-vs-model time base if it is 2. This is the guard that would have caught the
  original bug.
- Delete the dummy-zero proprio block entirely (`score.py:132-136`).

Then update `umf()` in the same file:

- Its `world_model` parameter becomes the `EncPredWM` wrapper. `chart.apply_`/`restore_`
  still target the predictor — reach it as `enc_pred_wm.model.predictor`.
- Update the docstring's shape contract: `encoder_output` is `[T_model + 1, N, D]` sampled
  **every `frameskip` raw frames**, `actions` is `[T_model, 10]`.
- Keep the numerator/denominator algebra **exactly as is** — it already matches
  `Σ‖ẑ_k−z_k‖² / Σ‖z_k−z_0‖²` correctly and is not part of this bug.

### Acceptance check for T1

Write a throwaway script under the scratchpad (not in `scripts/`) that:

1. Loads the checkpoint from the local hub (copy the `HUB_PATH` pattern from
   `scripts/run_e0.py:238-243` — use `source="local"`).
2. Rolls out a real 30-raw-step Push-T trajectory through the repaired function.
3. Asserts **all three**:
   - it returns exactly `30 / frameskip = 6` predictions, not 30;
   - every unroll step consumes a **distinct** action feature (the old bug's signature is a
     repeated one — assert the encoded action features are not all equal after index 1);
   - **frozen-model UMF on a well-moving chunk is below 1.0.** UMF > 1 means "worse than
     predicting stasis". The current broken state gives 24–52 online
     (`atlas_out/e1_smoke/episodes.jsonl`) and 0.67–1.67 offline.

**Paste the actual numbers.** If UMF is still ≥ 1 on a high-motion chunk, stop and report —
something beyond this task is wrong.

> `tests/test_score.py:12-19` mocks `wm.action_dim = 2` and `wm.encode_act.side_effect =
> lambda a: a`, i.e. it hard-codes away exactly the reshape that misbehaves. It will fail
> after this change. **Update the mock to the real 10-dim contract** — do not weaken the new
> guard to keep the old test green.

---

## T2 — Thread the `EncPredWM` wrapper through every caller

**Files:** `scripts/run_e0.py`, `scripts/run_e1.py`, `atlas/harness.py`, `atlas/router.py`

Every site that currently does `wm = model.model if hasattr(model, "model") else model` and
then passes `wm` into scoring must pass the **wrapper** instead. Keep the inner model where
it is genuinely needed (freezing encoder/predictor params, `predictor.load_state_dict`,
`Chart(predictor, kind)`).

| Site | Current | Change |
|---|---|---|
| `scripts/run_e0.py:238-247` | keeps `wm` only | keep both; pass the wrapper to scoring/finetune |
| `scripts/run_e0.py:155-197` `evaluate_e0_chart` | `world_model.predictor`, `_open_loop_rollout(world_model, ...)` | wrapper in, `.model.predictor` for chart apply/restore |
| `atlas/harness.py:61-152` `run_e0_finetune` | `predictor = world_model.predictor` | wrapper in; `enc_pred_wm.model.predictor` |
| `atlas/harness.py:184-312` `run_e1_episode` | `predictor = world_model.predictor` | same |
| `atlas/router.py:109-164` `_e1_score`, `_sdyn_score` | `world_model.predictor` | same |
| `scripts/run_e1.py:198` | `wm = model.model` | pass `model` (wrapper) to `run_e1_episode` |

`atlas/router.py::route`'s parameter is already named `world_model` — keep the name, change
what is passed. Do **not** rename it; `atlas/loop.py:99` already mismatches it (see T12).

**Acceptance:** `python -c "import atlas.score, atlas.router, atlas.harness"` imports clean,
and `scripts/smoke_e1.py` still runs end-to-end at tiny CEM settings. Paste the output.

---

## T3 — Chunk-aligned trajectory generation in E0

**File:** `scripts/run_e0.py::load_regime_trajectories` (`:32-152`)

Three changes:

1. **Emit model-chunk actions.** Keep stepping the env at raw granularity (the aimed-walk
   sampler and `ACTION_GAIN=0.25` calibration stay exactly as they are — they are correct and
   were validated separately, see `ACTION_SAMPLING_REVIEW.md`). But reshape the collected
   raw actions to `[T_raw // 5, 10]` before normalizing, so the returned `actions` are model
   chunks. Require `traj_len % 5 == 0` and raise clearly if not.
2. **Subsample frames to the model time base.** Keep only every 5th raw frame (plus frame 0)
   so `encoder_output` is `[T_raw//5 + 1, N, D]` — one encoded frame per model step, matching
   the actions.
3. **Capture and return proprio.** Collect `obs["proprio"]` for each kept frame. Encode via
   the wrapper's `EncPredWM.encode()` so proprio is normalized and embedded the same way the
   planner does it — rather than the current `preprocessor.transform_obs_visual` +
   `wm.encode_obs` path, which has no proprio at all.

   For the exact obs-dict layout `encode()` expects, copy `atlas/harness.py:157-165`
   (`_make_obs_td`) — that is the same construction `GC_Agent.act` feeds straight into
   `model.encode`. **Verify the tensor layout empirically** (print shapes on one trajectory)
   rather than assuming; `encode()` does its own `/255.0` and `preprocessor.transform`, so it
   wants raw images, unlike the current call.

Also fix an inconsistency while you are here: `run_e0.py:82` builds
`PushTEnv(render_size=224)` while `run_e1.py:203` builds it with `with_velocity=True`, and
the eval YAML specifies `env.with_velocity: true`. **Make E0 match** (`with_velocity=True`).
Note in your report that this changes the proprio dimension and therefore is a real
behavioural change, not a cosmetic one.

**Acceptance:** generate one R1 trajectory and print `encoder_output.shape`,
`actions.shape`, and the proprio shape. Assert `encoder_output.shape[0] == actions.shape[0] + 1`
and `actions.shape[1] == 10`.

---

## T4 — Chunk-aligned chunk construction in the E1 episode loop

**File:** `atlas/harness.py::run_e1_episode` (`:284-294`)

Same three changes as T3, applied to the executed-chunk encoding that feeds the router:
subsample `imgs` to every `frameskip`-th frame, chunk `step_actions` to `[n/5, 10]`, and
carry real proprio into the encode. The `frameskip` parameter (`harness.py:195`) is currently
accepted and **never used in the body** — this is where it should be used.

**Acceptance:** run `scripts/smoke_e1.py` and paste one `umf_trace` row. Values must be
**O(1)**, not the 24–52 currently in `atlas_out/e1_smoke/episodes.jsonl`.

---

## T5 — Wire the motion gate (gate G6)

**Files:** `scripts/run_e0.py:191`, `scripts/run_e1.py:216`

`atlas/score.py::compute_motion_gate` (`:160`) is called **nowhere outside `tests/`**. Both
production call sites pass `motion_gate=None`, so the informative-chunk gate that G6 exists
to test has never run.

Compute it once from the training trajectories' Frobenius displacements
(`(z[-1] - z[0]).norm(p="fro")` per chunk, 10th percentile — the definition is already in
`compute_motion_gate`) and thread it into both `umf()` calls. In `run_e1.py`, replace the
`motion_gate = None  # TODO` block outright.

**Acceptance:** run gate G6 and paste the output:
```bash
python scripts/smoke_gates.py --gate g6
```
A low-motion chunk must return `None` — no score, no strike, no probe.

---

## T6 — Restore the substrate's own planner config

**Files:** `scripts/run_e0_planning.py:245-247`, `scripts/run_e1.py:70-77`, and the Modal
defaults in `modal/modal_e0_planning.py:73`

Set **both** scripts to the config `dino_wm_pusht` was actually validated under, from
`vendor/jepa-wms/configs/evals/simu_env_planning/pt/dino-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml:200-205`:

```
num_samples=300, iterations=30, num_elites=10, horizon=6,
num_act_stepped=6, var_scale=1.0, frameskip=5   →  30 raw steps/episode, 1 replan
```

This is the user's explicit decision, made against evidence: it is the config DINO-WM reports
~90% Push-T SR under, and the one where the frozen baseline solved 2/3 episodes, versus 0/1
under the plan-§7.0 config. **Plan §7.0's numbers come from AdaJEPA** (Table 4 — a different
substrate); applied here, `horizon=25` means 125 raw steps of lookahead for a task DINO-WM
samples to be feasible within 25.

Two consequences to record in the code comments and in your report:

- This **resolves** the open `num_act_stepped` 1-vs-5 ambiguity in `E0_HANDOFF.md`. At
  `nas=6`, one replan covers the whole 30-step episode — there is no ambiguity left to
  resolve. Update `E0_HANDOFF.md`'s "Open, unresolved discrepancy" section to say so.
- It is a **documented deviation from plan §7.0**, justified as substrate fidelity. Add it to
  `ATLAS_implementation_plan_v2.md` as a §7.0a correction, in the same style as §6.1a.

Also fix `scripts/run_e0_planning.py:100-105,171-176`: it bypasses `PhysicsRegime.reset()`
and compensates with two manual `regime._apply_physics()` calls. Route it through the wrapper
so the invariant holds in one place.

**Acceptance:** print the resolved planner cfg from both scripts and paste it. Do not run
episodes yet.

---

## T7 — Throughput: make an episode affordable

Measured: ~42 min/episode on an L4, which is ~10% of the card's fp32 peak — a memory-bound
signature, not a fundamental cost. Apply in this order; **1–2 change numerics slightly, 3–5
are bit-exact.**

1. **Enable SDPA in the predictor.** `use_sdpa` is absent from the eval YAML, so it defaults
   `False` (`app/vjepa_wm/utils.py:586`), and `app/plan_common/models/vit.py:174-185` falls
   back to manual attention that materialises `[num_samples, 16, 512, 512]` fp32 **three
   times per layer** (`dots`, non-in-place `masked_fill`, `softmax`) × 6 layers. Set it
   post-load on the loaded model rather than editing the YAML:
   ```python
   for m in wm.predictor.modules():
       if hasattr(m, "use_sdpa"):
           m.use_sdpa = True
   ```
   `vit.py:158-172` shows the SDPA branch is the documented equivalent path. This is both the
   speed fix and the memory fix that keeps `num_samples=300` inside the A5000's 24 GB.
2. **bf16 autocast** around the planning rollout — the checkpoint was *trained* in bf16
   (`configs/vjepa_wm/pt_sweep/...:92: dtype: bfloat16`). At minimum set
   `torch.set_float32_matmul_precision("high")`.
3. **Delete the unconditional batch-1 unroll** at
   `vendor/jepa-wms/evals/simu_env_planning/planning/planning/planner.py:323`
   (`predicted_best_encs = self.unroll(z_init, act_suffix=mean.unsqueeze(1))`). It runs every
   CEM iteration regardless of `decode_each_iteration` (which ATLAS sets `False`), doubles the
   `forward_pred` call count, and **nothing in ATLAS reads the result**. Gate it on
   `decode_each_iteration` rather than removing the line outright.
4. **Ring-buffer the horizon-loop `torch.cat`** at `vit_enc_preds.py:344-346`. Only
   `[:, -ctxt_window:]` is ever consumed, but the full history is re-allocated every step —
   O(H²) copies and it is what makes VRAM scale with horizon.
5. **Slice the objective before subtracting** at
   `evals/simu_env_planning/planning/planning/objectives.py:130-149` when
   `sum_all_diffs=False`: it builds the full `[T, B, ...]` difference then keeps only
   `diff[-1]`.

Items 3–5 touch **vendored** code. `CLAUDE.md` §1.3 allows exactly one upstream hook, so this
is a deviation: keep all vendored edits in **one contiguous, clearly-commented patch**, list
them in your report, and add them to the release checklist. If that feels like too much
scope, do 1–2 only and report the measured speedup — they are the bulk of the win and touch
no vendored logic.

**Acceptance:** before/after wall-clock for one planning episode at the T6 config, plus peak
VRAM. Paste both.

---

## T8 — Make `scripts/profile_episode.py` real

It currently raises `NotImplementedError` (`:43-52`) — and would `AttributeError` *before*
reaching it, because `:28-31` loads from the **remote** hub and `:38-41` does
`model.encoder` / `model.predictor`, which `EncPredWM` does not expose (the working scripts
correctly do `wm = model.model` first). **This stub is the origin of the "~30 s/episode"
figure that the entire 37-GPU-h budget in plan §12 rests on. It has never run.**

Implement it against the T6 config using the pattern `scripts/run_e0_planning.py` already
establishes (bypass `eval.py`/`PlanEvaluator`, reuse `GC_Agent`/`CEMPlanner` directly). Report
sec/episode, peak VRAM, and predictor forwards per replan.

**Acceptance:** `python scripts/profile_episode.py --episodes 3` runs and prints real
numbers. Paste them. Recompute the E1 budget from the measured value.

---

## T9 🛑 STOP — Retrain the charts

**Requires approval. Report T1–T8 results and wait.**

All 9 charts in `atlas_out/e0/*.pt` were trained through the broken rollout and are
invalidated. Re-run `{ln_act, lora4, full} × {R1, R2}` through the repaired pipeline.

**Write to a new directory** (`atlas_out/e0_v2/`). Do not overwrite `atlas_out/e0/` — it is
the record of what the bug produced.

Two recipe changes, both justified by the overfitting evidence (`full` reaches train loss
0.0015 on 30 transitions over 2000 full-batch steps with no early stopping):

- **More, better-distributed data.** `data/pusht_noise/train/` is fully present locally —
  18,685 episodes with `rel_actions.pth`, `abs_actions.pth`, `states.pth`, and pre-computed
  `tokens.pth`. Replay real demo action sequences under `PhysicsRegime` (`reset_to_state` +
  `PhysicsRegime` already support this) to get regime-shifted trajectories on the
  distribution the checkpoint was trained on. The upstream loader for these files already
  exists at `vendor/jepa-wms/app/plan_common/datasets/pusht_dset.py:45-50`. Note these are
  R0-physics demos, so they must be **re-simulated** under the regime, not used as-is.
  Fall back to the existing scripted sampler if the replay path proves slow — but say which
  you used.
- **Early stopping.** Hold out a validation split and stop on it instead of always taking all
  2000 steps (`harness.py:108-124`).

**Acceptance:** new T5 table with real parameter counts (see T12), plus baseline (identity
chart) UMF per regime for comparison. Every eval UMF should now be **< 1.0** for a competent
chart. Report honestly if it is not — per `CLAUDE.md` §1.8, a failed criterion is a result.

---

## T10 🛑 STOP — The chart × regime planning matrix (highest-value run)

**Requires approval.** New script: `scripts/run_e0_matrix.py`.

One run that delivers **three** things at once: E0's missing Success column, E1's
oracle/random denominators, and the C3 UMF-vs-success validation figure.

For each chart in `{c₀, ln_act_R1, ln_act_R2, …}` × regime `{R1, R2}`, run N paired episodes
with that chart **held fixed for the whole episode**, logging success, `block_pos_diff`,
`block_angle_diff`, and UMF. Reuse `scripts/run_e0_planning.py`'s episode loop, its fixed
`block_success()` metric, and its dataset-sampled `(init, goal)` pairs — all three were
correctly fixed already and are not part of this bug.

**This is the project's decision point. Run it before E1, not after.**

- If no regime-adapted chart beats `c₀` **in its own regime**, RQ0 has failed under this
  protocol. Then `SR_oracle − SR_random < 10 pp` and `atlas/stats.py::normalised_recovery`
  returns `None` **by design** (`stats.py:35`) — E1 cannot produce a reportable number no
  matter how much compute it is given. Finding that out here costs hours instead of the whole
  remaining budget.
- If some chart does beat `c₀`, this run *is* the E0 Success column, and E1 is worth running.

**Start narrow:** `ln_act` × R1 vs `c₀` × R1 at N≈20 paired seeds. Report before fanning out.

Also run the **frozen-baseline sanity check** the plan mandated for day 2 and never got: the
frozen model at R0 under the T6 config should land near DINO-WM's published ~90% SR. **If it
does not, stop** — something beyond this plan is still wrong.

---

## T11 🛑 STOP — E1, descoped

**Requires approval, and only if T10 says the denominator exists.**

```bash
python scripts/run_e1.py --charts atlas_out/e0_v2 --kind <winner> \
    --routers umf e1 sdyn random oracle_id --episodes <from T8 budget> --seeds 3 \
    --out atlas_out/e1
```

`oracle_id` **and** `random` must both be in `--routers` or T1 is all `nan` — that is exactly
why `atlas_out/e1_verify/T1.md` currently reads `nan`.

Report the pre-registered criterion honestly, including "denominator below 10 pp, not
reportable" if that is the outcome.

---

## T12 — Correctness fixes to land alongside

Small, independent, and safe to do while the GPU tasks are gated. None require approval.

| # | Fix | Location |
|---|---|---|
| 1 | `atlas_step()` passes `predictor=` but `route()`'s parameter is `world_model` → `TypeError` at bind time, for **every** router including `random`/`oracle_id`. `atlas_step()` is therefore currently uncallable. | `atlas/loop.py:99` vs `atlas/router.py:31` |
| 2 | Same family: `maybe_expand(library, predictor, ...)` binds positionally but then does `world_model.predictor` / `.grid_size` / `.encode_act` on a `ViTPredictor` → `AttributeError`. | `atlas/loop.py:130` vs `atlas/expand.py:88` |
| 3 | `_fit_candidate`'s wandb block references undefined `step` (loop var is `_`) and `kind` → `NameError`, which the surrounding `except ImportError` does **not** catch. Fires whenever wandb is installed with an active run. | `atlas/expand.py:214-224` |
| 4 | `atlas_refine()` calls the bare `predictor(z_cur, a_t)`, which `score.py:100-105` documents as an invalid signature for this ViTPredictor. | `atlas/loop.py:192` |
| 5 | Hysteresis `m=0.05` is applied to three scores on wildly different scales — UMF ~O(1), `e1` ~1e4 (so `m` is a no-op), `sdyn` ∈[−1,1] (so `m` dominates and blocked a correct switch in the smoke run). Normalise the scores, or make `m` per-router. **Do not change the value 0.05** (§1.7). | `atlas/router.py:74-79` |
| 6 | `random` router uses the unseeded global `random` module → the arm is not reproducible. Thread the episode seed. | `atlas/router.py:167-169` |
| 7 | `compute_t1`'s docstring promises paired-bootstrap CIs; the body computes none, though `paired_bootstrap`/`success_rate_ci` are imported and unused. Plan §8 requires **Δ with CI, never two bare means**. | `scripts/run_e1.py:131`, `:43` |
| 8 | `episodes.jsonl` opens in append mode — re-running silently concatenates old and new runs into one file. | `atlas/harness.py:360` |
| 9 | T5's "Params" column reports **tensor counts** (26/12/69), not parameters. Real: `ln_act` 10,764 · `lora4` 118,176 trainable (10,292,640 stored) · `full` 20,800,884. Use `chart.n_params()`, and report `lora4`'s trainable count separately from its stored count. | `scripts/run_e0.py:386` |
| 10 | Docs describe `ln_act` as "LN affine + action encoder". It is **LN-only** — the action encoder is a *sibling* of the predictor (`video_wm.py:82-83`), so `Chart(predictor, kind)` can never reach it. The "~10.4k matches the plan" validation was a coincidence: LN alone is 10,764. Correct the three places that still assert it. | `atlas/chart.py:5`, `ATLAS_implementation_plan_v2.md:125`, `scripts/dump_params.py:28` |
| 11 | `smoke_gates.py::gate_g4` uses a gymnasium-style API (`env.reset(seed=...)`, 5-tuple `step()`) against the legacy-`gym` `PushTEnv` — it has never actually run. Fix it or keep it explicitly marked skipped; do not let it report as passing. | `scripts/smoke_gates.py` |

For #10 there is a further open question worth flagging but **not** fixing here: the released
`dino_wm_pusht.pth.tar` has only `['encoder', 'predictor']` keys, and
`load_checkpoint_state_dict` guards on `checkpoint.get("action_encoder") is not None`
(`app/vjepa_wm/utils.py:412`) — which would mean the action-conditioning layer is randomly
initialised on every load. **Verify this on a loaded model before acting on it.** If it holds,
report it; it is a substrate-level finding, not an ATLAS bug.

---

## T13 — Update the record

Once T1–T8 land, update these so the next session is not misled:

- **`CLAUDE.md` §0.1** — rewrite the E0/E1 status paragraphs. State plainly that the pre-fix
  UMF table, chart ranking, and CEM cost diagnostic are invalidated.
- **`E0_RESULTS.md`** — add a new top section marking everything below it superseded, with
  the reason.
- **`code-review.md`** — Bug #7 contains a **retraction that was itself wrong** ("encode_act
  reshapes by total element count, so it transparently chunks correctly… retract that
  concern"). The values do chunk correctly; the retraction missed that the rollout's loop
  count and its targets stayed on the raw time base. Reverse it and log the real bug.
- **`E0_HANDOFF.md`** — close out the `num_act_stepped` discrepancy per T6.

---

## Final gate

Before declaring the code work complete:

```bash
python scripts/smoke_gates.py --all
```

**G1 (identity)** must be re-run specifically — T2's wrapper threading touches the chart
apply/restore path, which is exactly what G1 exists to catch. Paste the full output. If G4
is still skipped, say so explicitly rather than letting `--all` read as a clean pass.
