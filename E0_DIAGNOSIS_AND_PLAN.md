# ATLAS — E0 diagnosis and the plan to E0 + E1

*Written 2026-08-24. Status: proposed, not yet approved or implemented.*

## Context

E0 has produced a set of observations that look contradictory: adapters that *improve*
offline UMF *hurt* real CEM planning; the frozen and adapted models' CEM cost rankings
are uncorrelated (ρ≈0.089); retraining with 5× more data made UMF *worse* (0.68→1.11);
the "CEM config fix" made the frozen **baseline** worse (2/3 → 0/1); and UMF measured on
real planner-executed chunks is **24–52** where E0's own eval reports **0.67–1.67**
(`atlas_out/e1_smoke/episodes.jsonl`, `atlas_out/e1_verify/episodes.jsonl` vs
`atlas_out/e0/results.json`).

These are not independent puzzles. They share one cause: **the function that rolls the
world model forward for UMF — and that E0 trains through — does not match how this
checkpoint actually unrolls.** Fixing that is the whole critical path. Everything else in
this plan is downstream of it.

Secondary but binding: the planner budget in plan §7.0 was taken from AdaJEPA (a different
substrate) and is ~6× more expensive than the config this checkpoint was validated under,
while `scripts/profile_episode.py` — the plan's own budget safety-valve — is a
`NotImplementedError` stub, so the "~30 s/episode" figure the entire compute budget rests
on was never measured.

Deadline 29 Aug AoE (~5 days). Decisions already taken: standardise on the substrate's own
planner config; build the paper around **E0 + E1 only**; budget is ~$120 Modal credit plus
a local A5000 (24 GB).

---

## 1. Root cause: `atlas/score.py::_open_loop_rollout`

This one function is used by **both** UMF scoring (`score.py:78`) **and** E0 fine-tuning
(`harness.py:87,117`). It has four independent defects.

### 1a. Time base is wrong by 5× — the decisive one

`VideoWM.encode_act` expects `(B, num_frames, frameskip × action_dim)`, i.e. **10-dim
model actions**, and reshapes blindly (`vendor/jepa-wms/app/vjepa_wm/video_wm.py:212-213`):

```python
B, T, D = a.shape
a = a.reshape(B, -1, self.action_dim)     # self.action_dim == 10
```

`run_e0.py:137` and `harness.py:288-290` both pass **raw 2-dim per-env-step actions**
`[T_raw, 2]`. So `[1, 10, 2]` reshapes to `[1, 2, 10]` — 2 model actions — but
`_open_loop_rollout` then loops `for t in range(T)` with `T = actions.shape[0] = 10`
(`score.py:118,138`). For `t ≥ 2` the slice `act_feats_all[:, t:t+1]` is **empty**,
`torch.cat` with it is a no-op, and `act_buf[:, -1:]` re-feeds the *same stale action*
for every remaining step.

- 10-step training trajectory: **8 of 10 steps run on a repeated stale action.**
- 50-step eval trajectory: **40 of 50 steps** do.

Simultaneously, targets are `enc_out[1:]` — *consecutive raw frames* — while one predictor
step advances **5** raw frames. So the loss and the UMF numerator compare a 5-frames-ahead
prediction against a 1-frame-ahead target, for every step.

> `code-review.md` Bug #7 retracts this concern ("encode_act reshapes by total element
> count, so it transparently chunks correctly"). The *values* do chunk correctly; the
> retraction misses that `_open_loop_rollout`'s loop count and its targets are still on the
> raw time base. The concern was right; the retraction should be reversed.

### 1b. Proprio is hard-zeroed

`score.py:135` fabricates `torch.zeros(1, 1, grid*grid, prop_dim)` and never updates it.
Push-T supplies real proprio (the pusher's x,y — `envs/pusht_gym_wrap.py:86,94`), which is
exactly the quantity actions control. The checkpoint's own `unroll` carries proprio forward
via `compute_new_pose` (integrating the action) or `predict_proprio`
(`vit_enc_preds.py:337-345`).

### 1c. Context window is 1 frame, should be 2

`score.py:148-151` passes `z_cur[:, -1:]`. The shipped Push-T config specifies
`ctxt_window: 2` (`pt_L2_cem_sourcedset_H6_nas6_ctxt2_*.yaml:182`) and was trained with
`num_hist: 3` (`:149`). The canonical `unroll` slices `[:, -self.ctxt_window:]`.

### 1d. A correct implementation already ships and is unused

`EncPredWM.unroll(z_ctxt, act_suffix)` (`vit_enc_preds.py:289-353`) and
`EncPredWM.encode(obs)` (`:356`) are exactly this operation, done right — sliding
`ctxt_window`, real proprio propagation, correct action chunking. `GC_Agent` plans through
them (`gc_agent.py:175,141`). ATLAS holds the *inner* `VideoWM` (`run_e1.py:198`:
`wm = model.model`) and hand-rolls `forward_pred` instead, bypassing the wrapper that owns
`ctxt_window` and `proprio_mode`.

### Why this explains every observation

| Observation | Explanation |
|---|---|
| Charts improve UMF but hurt CEM planning | UMF and chart training share the broken rollout; CEM uses the correct one. The charts learned to compensate for stale actions, zero proprio and a 5× time-base error — a correction that is actively wrong at planning time |
| CEM cost ranking uncorrelated (ρ≈0.089) | The chart is not a physics correction at all, so it reorders counterfactual costs arbitrarily |
| More data made UMF worse (0.68→1.11) | Fitting a mis-specified target harder does not help |
| `full` (20.8M) always worst | Most capacity to absorb the corruption |
| UMF 24–52 online vs 0.67–1.67 offline | Online chunks are 25 raw steps (5 model chunks); offline eval was 50 raw steps. The corruption's severity scales with how much of the rollout runs on stale actions |
| `full` train loss → 0.0015 | 2000 full-batch steps on 30 transitions, no early stopping |

### On the "scripted actions vs planner actions" concern

The instinct is right and points at a real, citable phenomenon — objective mismatch
([Lambert et al., L4DC 2020](https://arxiv.org/abs/2002.04523)): prediction likelihood is
not correlated with control performance, and a planner doing argmin over ~2000
counterfactual rollouts actively seeks out wherever the model is wrongly optimistic.

But it is **not** the primary cause here; the rollout bug is, and it is far larger. Two
things cut against the distribution story as the explanation: the base `dino_wm_pusht`
checkpoint is itself trained off-policy (noise-perturbed human demos, DINO-WM App. A.1) and
plans at ~90% SR; and E0 trains on 30 transitions for 2000 full-batch steps with no early
stopping. **Coverage and overfitting are the real risks, not off-policy-ness per se.**
Addressed in P4 below.

---

## 2. What survives, what does not

**Invalidated** (all depend on `_open_loop_rollout`): E0's UMF table
(`atlas_out/e0/results.json`), the `ln_act > lora4 > full` ranking, all 9 charts in
`atlas_out/e0/*.pt`, the CEM cost-ranking diagnostic, every E1 smoke/verify UMF number,
the "richer retraining made it worse" result.

**Survives**: the regime design (R1 friction 0.8 / R2 elasticity 0.9, `regimes.py:55-59`)
and its G4 validation; the contact-rate and action-scale fixes in `load_regime_trajectories`;
the predictor-contamination fix; the `Chart` parameter-selection fixes; the success-metric
and dataset-goal fixes in `run_e0_planning.py`; the seed manifest and pairing machinery;
the Modal deployment path.

---

## 3. The work, in order

### P0 — Rewrite the rollout on the checkpoint's own API *(blocks everything)*

**`atlas/score.py::_open_loop_rollout`** — reimplement in terms of `EncPredWM.unroll()`
rather than hand-driving `VideoWM.forward_pred`. This requires threading the **wrapper**
(`EncPredWM`, i.e. the object `torch.hub.load` returns) rather than `model.model`. Callers
to update: `run_e0.py:246-295`, `run_e1.py:198`, `harness.py:87,117`, `router.py:109-164`.

Contract change to make explicit in the docstring and enforce with a shape assert:
`actions` is `[T_model, 10]` and `encoder_output` is `[T_model + 1, N, D]` sampled **every
`frameskip` raw frames**. Add a guard that rejects a raw-time-base call rather than
silently reshaping.

### P1 — Chunk-aligned data pipeline

- **`scripts/run_e0.py::load_regime_trajectories`** (`:134-142`): keep every raw action but
  emit them chunked `[T_raw/5, 10]`, and subsample `imgs` to every 5th raw frame so
  `encoder_output` is `[T_raw/5 + 1, N, D]`. Capture `obs["proprio"]` per kept frame and
  return it. Construct `PushTEnv(..., with_velocity=True)` to match `run_e1.py:203`
  (currently inconsistent, `run_e0.py:82`).
- **`atlas/harness.py::run_e1_episode`** (`:284-294`): same chunking for the executed chunk,
  and pass real proprio into the encode.

### P2 — Wire the motion gate

`compute_motion_gate` (`score.py:160`) is called nowhere outside tests. Compute it once
over the training displacements and pass it in `run_e0.py:191` and `run_e1.py:216`
(currently `motion_gate = None`). This is gate **G6**, and the 24–52 UMF values are exactly
the blow-up it exists to prevent.

### P3 — Planner config and throughput

Standardise on the substrate's validated Push-T config: `num_samples=300, iterations=30,
horizon=6, num_act_stepped=6, ctxt_window=2`, 30 raw steps/episode — the config
`dino_wm_pusht` reports ~90% SR under, and the one where the frozen baseline solved 2/3.
Update `run_e0_planning.py:245-247` and `run_e1.py:70-77` together, and record the deviation
from plan §7.0 as a substrate-fidelity correction. This also dissolves the
`num_act_stepped` 1-vs-5 ambiguity in `E0_HANDOFF.md`: at `nas=6`, one replan covers the
whole 30-step episode, exactly as DINO-WM does it.

Throughput fixes, in payoff order (3–5 are bit-exact; 1–2 change numerics slightly):

1. **Enable SDPA in the predictor.** `use_sdpa` is absent from the eval YAML so it defaults
   `False` (`utils.py:586`), and `vit.py:174-185` falls back to manual attention that
   materialises `[300,16,512,512]` fp32 three times per layer. Post-load
   `for m in wm.predictor.modules(): setattr(m, "use_sdpa", True)` where the attribute
   exists. Biggest single win, and it is also the memory fix that keeps 300 samples inside
   the A5000's 24 GB.
2. **bf16 autocast** around the planning rollout (the checkpoint was *trained* in bf16),
   or at minimum `torch.set_float32_matmul_precision("high")`.
3. Delete the unconditional batch-1 `predicted_best_encs` unroll (`planner.py:323`) —
   doubles `forward_pred` calls, and nothing in ATLAS reads the result.
4. Replace the horizon-loop `torch.cat` with a fixed `ctxt_window` ring buffer
   (`vit_enc_preds.py:344-346`).
5. Slice the objective to the last timestep before subtracting when `sum_all_diffs=False`
   (`objectives.py:130-149`) — 5.3 GB transient, 26/27 discarded.

3–5 touch vendored code; keep them in a single reviewable patch and note it in the release
checklist alongside the existing "one upstream hook" rule.

**Then implement `scripts/profile_episode.py` for real** (it currently raises before it can
even load — it uses the remote hub and `model.encoder`, which `EncPredWM` does not expose)
and measure sec/episode before committing budget.

### P4 — Retrain charts, re-run E0's UMF half

Re-run `{ln_act, lora4, full} × {R1, R2}` (R0 optional) through the repaired pipeline. Two
changes to the training recipe, both justified by the overfitting evidence:

- **More data.** `data/pusht_noise/train/` is fully present locally — 18,685 episodes with
  `rel_actions.pth`, `states.pth`, and pre-computed `tokens.pth`. Replay real demo action
  sequences under `PhysicsRegime` (`reset_to_state` + `PhysicsRegime` already support this)
  to get regime-shifted trajectories on the distribution the checkpoint was trained on. This
  directly addresses the coverage concern, and it is what `ACTION_SAMPLING_REVIEW.md`
  identified as DINO-WM's own practice. Fall back to the scripted sampler only if the replay
  path proves slow.
- **Stop overfitting.** Hold out a validation split and early-stop on it instead of always
  taking all 2000 full-batch Adam steps (`harness.py:108-124`).

### P5 — The chart × regime planning matrix *(the highest-value run)*

One run that simultaneously delivers **E0's missing Success column**, **E1's oracle/random
denominators**, and **the C3 validation figure**. For each chart in `{c₀, ln_act_R1,
ln_act_R2, …}` × regime `{R1, R2}`, run N paired episodes with that chart held fixed for the
whole episode, logging success and UMF.

This is the decision point the whole project hinges on, and it must run before E1:

- If no regime-adapted chart beats `c₀` **in its own regime**, then RQ0 has failed under
  this protocol, `SR_oracle − SR_random < 10 pp`, and `normalised_recovery` returns `None`
  **by design** (`stats.py:35`) — E1 cannot produce a reportable number no matter how much
  compute it gets. Knowing this costs a few hours, not the whole remaining budget.
- If some chart does beat `c₀`, this run *is* the E0 Success column and E1 is worth running.
- Either way, the UMF-vs-success scatter with Kendall τ (supplementary figure S3, proposal
  §8) falls out for free, and it answers JEPA-WM's open appendix question directly.

Start with `ln_act` × R1 vs `c₀` × R1 at N≈20 paired seeds before fanning out.

### P6 — E1, descoped

Run `--routers umf e1 sdyn random oracle_id` at whatever episode count P3's measured
sec/episode affords. `oracle_id` and `random` are **required** in `--routers` or T1 is all
`nan` — that is why `atlas_out/e1_verify/T1.md` reads `nan`. Report the pre-registered
criterion honestly, including "denominator below 10 pp, not reportable" if that is the
outcome.

### P7 — Small correctness fixes to land alongside

| Fix | Location |
|---|---|
| `atlas_step()` passes `predictor=` where `route()` takes `world_model` → `TypeError`; same at `maybe_expand` | `loop.py:99,130` vs `router.py:31`, `expand.py:88` |
| `_fit_candidate` wandb block references undefined `step`/`kind` → `NameError` not caught by its `except ImportError` | `expand.py:214-224` |
| Hysteresis `m=0.05` is applied to three scores on wildly different scales (UMF ~O(1–50), `e1` ~1e4, `sdyn` ∈[−1,1]) — a no-op for `e1`, dominant for `sdyn`. Normalise, or set per-router. | `router.py:74-79` |
| `random` router uses the unseeded global `random` module → not reproducible | `router.py:167-169` |
| `compute_t1` docstring promises paired-bootstrap CIs; body computes none, though `paired_bootstrap`/`success_rate_ci` are imported | `run_e1.py:131`, `:43` |
| `episodes.jsonl` opens in append mode — re-runs concatenate silently | `harness.py:360` |
| T5's "Params" column reports **tensor counts** (26/12/69), not parameters (10,764 / 118,176 trainable / 20,800,884) | `run_e0.py:386` |
| `run_e0_planning.py` bypasses `PhysicsRegime.reset()` and compensates with manual `_apply_physics()` calls | `run_e0_planning.py:100-105,171-176` |
| Docs still describe `ln_act` as "LN affine + action encoder"; it is LN-only and the action encoder is structurally unreachable. The ~10.4k count that "validated" it is a coincidence (LN alone = 10,764). | `chart.py:5`, `implementation_plan §5.1`, `dump_params.py:28` |

---

## 4. Verification

Run the relevant gate after each step, and show the output — not "should pass".

1. **P0 unit check.** Encode a real 30-raw-step Push-T trajectory. Assert
   `_open_loop_rollout` produces exactly `T_raw/5` predictions, that every step consumes a
   *distinct* action feature (the current bug's signature is a repeated one), and that
   frozen-model UMF on a well-moving chunk lands **below 1.0**. UMF > 1 means "worse than
   predicting stasis" — 24–52 is the current, obviously-broken state.
2. **G1 identity** — `{c₀}` only ⇒ bit-identical trajectory to frozen. Must be re-run after
   the wrapper-threading change in P0.
3. **G6 denominator** — now that P2 wires the gate, low-motion chunks must return `None`.
4. **G4 regimes real** — unchanged by this work, but `smoke_gates.py::gate_g4` uses a
   gymnasium-style API against a legacy-`gym` env and has never actually run; fix or keep it
   honestly marked skipped.
5. **G5 pairing** — confirm every router sees identical init/goal per episode index.
6. **Frozen-baseline sanity** — the plan's day-2 check that was never done: frozen model at
   R0 under the restored config should land near DINO-WM's published ~90% SR. If it does
   not, stop; something else is still wrong.
7. **`smoke_gates.py --all`** before declaring P0–P3 complete.

## 5. Budget

With the A5000 local and the substrate config restored, throughput stops being the binding
constraint. Measure first (P3), but the expectation is ~1.5 min/episode with SDPA+bf16
versus the ~42 min currently observed — putting P5 (~160 episodes) in the range of a few
local hours and a descoped E1 within a night or two, with the $120 Modal credit held in
reserve for parallel fan-out rather than being the primary path.

## 6. Evidence appendix — key file:line references

| Claim | Evidence |
|---|---|
| `encode_act` reshapes to 10-dim model actions | `vendor/jepa-wms/app/vjepa_wm/video_wm.py:199-213` |
| `model_action_dim = action_dim × tubelet × frameskip // action_skip` = 10 | `vit_enc_preds.py:84-85` |
| E0 passes raw `[T,2]` actions | `scripts/run_e0.py:137`; `atlas/harness.py:288-290` |
| Rollout loops over raw T with 1-frame context, zero proprio | `atlas/score.py:118,135,138,148-151` |
| E0 fine-tuning goes through the same rollout | `atlas/harness.py:87,117` |
| Canonical rollout with `ctxt_window` + proprio propagation | `vit_enc_preds.py:289-353` |
| `ctxt_window: 2`, `num_hist: 3`, `frameskip: 5`, CEM 300×30, `horizon: 6`, `nas: 6` | `configs/evals/simu_env_planning/pt/dino-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml:142,149,182,200-205` |
| `use_sdpa` defaults False; manual attention materialises the score matrix | `app/vjepa_wm/utils.py:586`; `app/plan_common/models/vit.py:174-185` |
| Redundant batch-1 unroll every CEM iteration | `evals/simu_env_planning/planning/planning/planner.py:323` |
| `motion_gate` never called outside tests | `scripts/run_e0.py:191`; `scripts/run_e1.py:216` |
| `atlas_step()` → `route()` kwarg mismatch | `atlas/loop.py:99` vs `atlas/router.py:31` |
| `profile_episode.py` is a stub, and would `AttributeError` first | `scripts/profile_episode.py:28-52` |
| Full offline dataset with actions is on disk | `data/pusht_noise/train/{rel_actions,abs_actions,states,tokens}.pth` |
| Online UMF 24–52; offline 0.67–1.67 | `atlas_out/e1_smoke/episodes.jsonl`; `atlas_out/e0/results.json` |

**External sources.** [Objective Mismatch in Model-based RL, Lambert et al., L4DC
2020](https://arxiv.org/abs/2002.04523) · [DINO-WM, arXiv:2411.04983](https://arxiv.org/pdf/2411.04983)
· [AdaJEPA, arXiv:2606.32026](https://arxiv.org/html/2606.32026v1) (Table 4: CEM 200 samples,
10 opt steps, subplanner horizon 25, 5 executed actions, frameskip 5, max 20 MPC steps).
