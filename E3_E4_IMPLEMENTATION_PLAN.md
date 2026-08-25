# E3 + E4 Implementation Plan — phased

> Written 2026-08-25. Executable by a second agent working **in parallel** with the agent
> that owns E0/E1 — see §0.3 for the file-ownership split that keeps the two apart.
> Every file this plan touches lives under `D:\Shubham\DeepLearning\Atlas\atlas\`.

---

## Context

E0 is a documented negative result (`ln_act` recovers 1 of 10 recoverable episodes on
R2/damping-0.5 → **normalised recovery 10%**), and E1 is gated behind P4/P5. Another agent
owns that critical path. Meanwhile **E3 and E4 are the paper's central table (T2) and money
plot (F1)** and are 100% unimplemented — `scripts/run_e4.py` raises `NotImplementedError`.
They are also the longest-lead item in the project.

This plan lets a **second agent build all of E3+E4 in parallel**, touching almost no file
the E0/E1 agent needs, so that the moment E0 yields a usable chart the stream can launch
with zero further engineering.

E3 is not a separate experiment: **arms 4/5/6 of the single E4 run *are* E3.** One run
produces T2, F1 and F2b.

**Binding constraints** (`CLAUDE.md` §1): single substrate; frozen DINOv2 encoder; adapters
only; identity-initialised charts; apply/restore, never model copies; **strict prequential
order — SCORE before REFINE**; fixed hyperparameters τ=0.5 q=3 m=0.05 n_probe=20 K_max=10
lr=5e-4. Never `git commit`/`push`. Never claim a gate passed without pasting its output.

---

## Phase map

| Phase | What | Gated? | Rough size |
|---|---|---|---|
| **0** | Read-in + budget reality check | — | 30 min |
| **1** | Unblock the three latent defects in `atlas/` | — | half day |
| **2** | Build the E4 episode runner | — | 1 day |
| **3** | Build `scripts/run_e4.py` | — | half day |
| **4** | Smoke test all 7 arms end-to-end | — | half day |
| **5** | 🛑 Profile → report → agree the budget cuts | **STOP** | 1 h + user |
| **6** | Modal runner + the real run | 🛑 after profile | 1 h + ~35 GPU-h |
| **7** | T2, F1, F2b | — | half day |
| **8** | Housekeeping (stale configs) — do any time | — | 20 min |
| **A** | Appendix: E2, only after Phase 4 works | — | half day |

Phases 1a/1b/1c are independent of each other and can be done in any order.
Phase 8 is independent of everything.

---

# Phase 0 — Read-in and the budget reality check

## 0.1 Ground truth (verified by reading code, not docs)

Docs in this repo are partly stale — `configs/regimes/pusht.yaml` still declares R1 as
`mass_scale: 0.2`, a regime that is *physically dead*. Trust the code.

| Fact | Where |
|---|---|
| `model, prep = torch.hub.load(HUB_PATH, "dino_wm_pusht", source="local", ...)`; `model` is the **`EncPredWM` wrapper**, `wm = model.model` is the inner `VideoWM`, predictor is `wm.predictor` | `scripts/run_e0_planning.py:365-385`, `scripts/run_e0.py:462-483` |
| Every rollout/scoring path needs the **wrapper** and **real proprio** — `proprio=None` is a channel-width `RuntimeError`, not a graceful path | `atlas/score.py:55-65, 130-190` |
| `_open_loop_rollout` **raises** if `actions.shape[-1] != 10` (the raw-vs-model time-base guard) | `atlas/score.py:164-172` |
| `REGIME_CONFIGS = {"R0": {}, "R1": {"friction": 2.0}, "R2": {"damping": 0.5}}` — calibrated defaults, no flag needed | `atlas/regimes.py:55-66` |
| `PhysicsRegime` is legacy-`gym`; `reset()` returns a 2-tuple and applies physics *after* the inner reset | `atlas/regimes.py:76-124` |
| Corrected episode primitives: `sample_dataset_init_goal()` (`:154`, difficulty + reachability filtered) and `block_success()` (`:207`, block-only, correctly wrapped angle) | `scripts/run_e0_planning.py` |
| `run_e1_episode()` is the working replan loop to generalise from — especially the chunk re-encoding block | `atlas/harness.py:284-441`, esp. `:401-423` |
| `route(...)` **does** accept `proprio_ctxt=` and `rng=` | `atlas/router.py:28-42` |
| `Chart.restore_()` for `ln_act`/`full` is just `apply_()` — it restores *this* chart, **not** a pristine baseline | `atlas/chart.py:107-127` |
| E0's on-disk layout is `chart_{kind}_{regime}.pt`, **not** `Library.save()`'s format; `load_library_from_e0()` bridges it | `scripts/run_e1.py:115-140` |
| `make_t2` / `make_f1` / `make_f2` already exist and define the JSONL field contract | `scripts/make_tables.py:71`, `scripts/make_figures.py:29,59` |
| Modal pattern: local-dir image → `/src`, volume at `/atlas_root`, remote fn = `reload()` → `subprocess.run(cmd, cwd="/src")` → `commit()` | `modal/modal_e0_planning.py` |
| `Library` has **no** `evict()` / `utilisation()`; `add()` raises when full (K_max=10) | `atlas/library.py:58-74` |

## 0.2 🔴 The budget does not close — read before writing code

Plan §7.4's "~21 GPU-h at 30 s/episode" is stale by an order of magnitude.

Measured, from `atlas_out/e0_v3_*/*_summary.json` (L4, `num_samples=300, iterations=30,
horizon=6, num_act_stepped=6`): **~146–151 s/episode, one replan per episode.**

E4 needs **multiple replans per episode** — routing, refinement and `Expander`'s
"verify on the NEXT chunk" (`atlas/expand.py:91`) all require it. At `num_act_stepped=1`
one replan executes `1 × frameskip(5) = 5` raw actions, so `MAX_MPC_STEPS=30` → **6 replans**.
CEM cost per replan is unchanged:

```
~150 s/replan × 6 replans            ≈  900 s/episode
7 arms × 6 segments × 20 ep × 3 seeds =  2 520 episodes
2 520 × 900 s                        ≈  630 GPU-h ≈ $500 on L4, 26 days on one GPU
```

**Infeasible. Do not silently absorb it.** Cut ladder, applied uniformly across every arm
and reported in the paper (plan §7.0's own instruction):

| # | Cut | Effect | Cost |
|---|---|---|---|
| 1 | `--iterations 30 → 10` | ~3× → ~300 s/ep | Must validate: frozen@R0, N=10, at `iterations=10`; SR must stay ≈85%+. If it collapses, revert and cut elsewhere. |
| 2 | `--episodes 20 → 10` per segment | 2× | 10×6×1 = 60 paired episodes/arm |
| 3 | `--seeds 3 → 1` | 3× | No cross-seed CI band on F1 |
| 4 | One Modal container per `(arm, seed_run)` | 7–21× **wall clock** | Same GPU-h, buys wall time. Arms are fully independent. |

**Recommended target:** `iterations=10, episodes=10, seeds=1, 7 arms` → 420 episodes ×
~300 s ≈ **35 GPU-h ≈ $28**, ~5 h wall with 7 parallel containers.

## 0.3 File ownership — conflict-safe split

| File | Action | Owner |
|---|---|---|
| `atlas/adajepa.py` | **rewrite** `refine()`/`push()`/`reset()` | E3/E4 — E0/E1 never imports it |
| `atlas/loop.py` | **fix** proprio/rng threading | E3/E4 — nothing else calls it |
| `atlas/streams.py` | **extend** with a `regimes` parameter | E3/E4 (additive) |
| `atlas/harness_e4.py` | **NEW** | E3/E4 |
| `scripts/run_e4.py` | **replace** the stub | E3/E4 |
| `scripts/smoke_e4.py` | **NEW** | E3/E4 |
| `modal/modal_e4.py` | **NEW** | E3/E4 |
| `scripts/make_tables.py` | **extend `make_t2` only** | E3/E4 |
| `configs/regimes/pusht.yaml`, `configs/atlas/default.yaml`, `e4.yaml` | **fix stale values** | E3/E4 |
| **`atlas/harness.py`** | **DO NOT TOUCH** | E0/E1 (P5 edits `run_e1_episode`) |
| **`scripts/run_e1.py`, `run_e0*.py`, `atlas/regimes.py`, `atlas/score.py`** | **import only** | E0/E1 |

`run_e4_episode()` goes in a **new `atlas/harness_e4.py`**, not `harness.py`. This is a
deliberate deviation from `CLAUDE.md` §3's module-ownership rule, taken *only* to keep two
agents off one file. Put a comment at the top of the new module saying so, and that it
should be folded into `harness.py` once P5 has landed.

---

# Phase 1 — Unblock the three latent defects

Three real defects make arms 2–6 non-functional today. None of these files is touched by
E0/E1 work. **1a, 1b, 1c are independent — do them in any order.**

## 1a — `atlas/adajepa.py` (unblocks arms 2 and 3)

**The defect:** `refine()` at `:108` calls `self.predictor(z_cur, a_t)`, the invalid
`ViTPredictor` call signature already fixed away in `loop.py` and `expand.py`. It also
adapts *all* `requires_grad` predictor params (20.8M) while ATLAS arms adapt 10.7k —
breaking plan §7.6's "same loss, lr, optimiser, buffer size; only the library/routing/
expansion differ", and with it the ladder's one-mechanism-per-rung property.

```python
# atlas/adajepa.py
class AdaJEPA:
    BUFFER_SIZE = 5

    def __init__(self, world_model, param_names, variant="adajepa", lr=5e-4):
        # world_model: the EncPredWM WRAPPER (not .model) -- _open_loop_rollout needs it.
        # param_names: exactly Chart(predictor, kind)._param_names for the E0 winner kind,
        #              so arms 2/3 adapt the SAME surface arms 4/5/6 do.
        self.world_model = world_model
        self.predictor = world_model.model.predictor
        self.param_names = list(param_names)
        self.variant, self.lr = variant, lr
        self.pretrained_state = {k: v.detach().clone()
                                 for k, v in self.predictor.state_dict().items()
                                 if k in self.param_names}
        self._buffer: deque[tuple[Tensor, Tensor, Tensor | None]] = deque(maxlen=self.BUFFER_SIZE)
        self._params = [p for n, p in self.predictor.named_parameters() if n in self.param_names]
        for p in self._params:
            p.requires_grad_(True)
        self._optimizer = optim.Adam(self._params, lr=lr)

    def reset(self):
        if self.variant == "adajepa":                      # 'persistent' is a deliberate no-op
            self.predictor.load_state_dict(self.pretrained_state, strict=False)
            self._buffer.clear()
            self._optimizer = optim.Adam(self._params, lr=self.lr)

    def push(self, encoder_output, actions, proprio_ctxt=None):
        self._buffer.append((encoder_output.detach(), actions.detach(),
                             None if proprio_ctxt is None else proprio_ctxt.detach()))

    def refine(self) -> float:
        from atlas.score import _open_loop_rollout, _make_z_ctxt
        if not self._buffer:
            return 0.0
        self._optimizer.zero_grad()
        total = 0.0
        for enc_out, actions, proprio_ctxt in self._buffer:
            z_ctxt = _make_z_ctxt(self.world_model, enc_out[0], proprio_ctxt)
            z_preds = _open_loop_rollout(self.world_model, z_ctxt, actions)
            loss = (z_preds - enc_out[1:]).pow(2).mean(dim=-1).mean()
            (loss / len(self._buffer)).backward()   # per-item backward = O(1) memory,
            total += loss.item()                    # same as harness.py's P2a fix
        self._optimizer.step()
        return total / len(self._buffer)
```

Keep `BUFFER_SIZE = 5`, `lr = 5e-4`, exactly one gradient step per replan, encoder frozen,
and the existing wandb block.

**Verify (paste output):** a scratch script that loads the checkpoint, builds one synthetic
chunk (reuse `_make_synthetic_proprio_ctxt`, `scripts/smoke_gates.py:34`), calls `push()`
then `refine()` twice. Both losses must be finite and must change.

## 1b — `atlas/loop.py` (unblocks arms 4, 5, 6)

**Defect 1** (`:99-134`): `atlas_step()` never forwards `proprio_ctxt` or `rng` to
`route()` / `expander.record()` / `expander.maybe_expand()`, though all three accept them.
Add keyword params `proprio_ctxt=None`, `next_proprio_ctxt=None`, `rng=None` and forward:

```python
selected_idx, route_info = route(
    kind=cfg.router, library=library, world_model=world_model,
    encoder_output=encoder_output, actions=actions, current_idx=current_idx,
    motion_gate=cfg.motion_gate, hysteresis=cfg.hysteresis,
    regime_label=regime_label, label_to_chart=cfg.label_to_chart,
    proprio_ctxt=proprio_ctxt, rng=rng,                                    # ← added
)
...
expander.record(best_umf, encoder_output, actions, proprio_ctxt)           # ← added
probe_outcome = expander.maybe_expand(
    library, world_model, next_encoder_output, next_actions,
    cfg.motion_gate, next_proprio_ctxt,                                    # ← added
)
```

**Defect 2** (`:199`): `atlas_refine()` passes a bare `[N,D]` tensor into
`_open_loop_rollout` without `_make_z_ctxt`, dropping proprio. Fix:

```python
from atlas.score import _open_loop_rollout, _make_z_ctxt
z_ctxt = _make_z_ctxt(world_model, encoder_output[0], proprio_ctxt)
z_preds = _open_loop_rollout(world_model, z_ctxt, actions)
```

**Defect 3:** `atlas_refine` builds a fresh `Adam` on every call, discarding moment state
across replans — not AdaJEPA's setup. Add an optional `optimizer=None` param; the caller
(`ArmState`) owns one `Adam` per chart index and passes it in.

Do **not** change `expansion_mode` semantics, τ/q/m/n_probe, or the strike logic.

**Verify:** `python scripts/smoke_gates.py --gate G3a`, `--gate G3b`, `--gate G6`
(they exercise `Expander` directly). Paste all three.

## 1c — `atlas/streams.py` (parameterise the stream regimes)

Per the user's decision: **CLI flag, default `R0`/`R2`.** R2/damping-0.5 is where every
trustworthy E0 number lives (50pp headroom, mechanistic overshoot failure mode); R1/
friction-2.0 has only 20pp and no chart trained under the calibrated value.
`paired_seed` is untouched — gate G5 depends on it.

```python
def stream_s2(episodes_per_segment=20, seeds=3, stream_seed_offset=0,
              regimes: tuple[str, str] = ("R0", "R2")):
    """S2 = A,B,A,B,A,B over (regime_A, regime_B).

    Plan §6.1/§7.4 says R0/R1. Default changed to R0/R2 (damping 0.5): E0's
    calibration (E0_RECOVERY_PLAN §0.5) measured R2 at 40pp degradation vs R1's
    20pp, and every trained/evaluated chart in this project targets R2.
    Overridable via --segment-regimes; the resolved pair is recorded in the run's
    summary JSON. Documented deviation, in the style of plan §7.0a.
    """
    seq = [regimes[i % 2] for i in range(6)]
    ...

def get_stream(name, episodes_per_segment=20, seeds=3, regimes=("R0", "R2")): ...
```

Also add `global_episode_idx = segment_idx * episodes_per_segment + episode_idx` to
`EpisodeSpec` — `make_f1`/`make_f2` sort on it.

**Verify:** `python -m pytest tests/test_streams.py -q` and
`python scripts/smoke_gates.py --gate G5`. Paste both, plus the printed S2 regime sequence
(`R0,R2,R0,R2,R0,R2`).

---

# Phase 2 — The E4 episode runner (`atlas/harness_e4.py`)

## 2a — `ArmState`: the state that persists *across* episodes

This is what makes E4 continual rather than 2 520 independent episodes.

```python
@dataclass
class ArmState:
    arm: str
    library: Library | None          # None for frozen / adajepa / adajepa_persist
    expander: Expander | None
    adapter: AdaJEPA | None          # arms 2, 3 only
    cfg: ATLASConfig
    label_to_chart: dict[int, int] | None
    current_idx: int = 0             # persists across episodes -- this IS recall
    optimizers: dict[int, torch.optim.Adam] = field(default_factory=dict)
    charts_committed_cumulative: int = 0
    probes_rejected_cumulative: int = 0
```

Per-arm behaviour — implement exactly this, no more:

| # | arm | library | router | refine each replan | expansion_mode | reset per episode |
|---|---|---|---|---|---|---|
| 1 | `frozen` | — | — | — | — | — |
| 2 | `adajepa` | — | — | `adapter.refine()` | — | `adapter.reset()` |
| 3 | `adajepa_persist` | — | — | `adapter.refine()` | — | none |
| 4 | `atlas_fixed` | `{c0, chart_B}` | `umf` | `atlas_refine(c*)` | `fixed` | none |
| 5 | `atlas_detect` | `{c0, chart_B}` | `umf` | `atlas_refine(c*)` | `detect_only` | none |
| 6 | `atlas` | `{c0, chart_B}` | `umf` | `atlas_refine(c*)` | `atlas` | none |
| 7 | `oracle_id` | `{c0, chart_B}` | `oracle_id` | — | — | none |

Rules to enforce in code, each with an assertion:

- **`library[0]` (c0) is never refined** — `if selected_idx != 0: atlas_refine(...)`.
- **Strict prequential order:** score the chunk executed by the *previous* replan → select
  → plan → execute → *then* refine. Never score a chunk a chart just trained on.
- **`oracle_id`'s `label_to_chart`** for S2 = `{REGIME_LABELS["R0"]: 0, REGIME_LABELS[B]: 1}`
  — **c0 is the correct R0 chart** (the identity chart on the unshifted regime *is* the
  oracle answer). Comment this; it is a design decision, not an accident.
- **Arms mutate the predictor differently** (AdaJEPA in-place vs `Chart.apply_`/`restore_`),
  so **arms run sequentially with a pristine predictor reload between them**. Copy
  `scripts/run_e0.py:491`'s `pristine_predictor_state = copy.deepcopy(wm.predictor.state_dict())`
  and `load_state_dict` it at the start of every arm *and* every seed_run.

## 2b — Starting library for the expansion arms: flag it, don't guess

Two defensible readings; the choice changes the whole E3 result.

- **(i) monotone ladder (default):** arms 4/5/6 all start from `{c0, chart_B}` = 2 charts =
  the true regime count. Correct ATLAS behaviour is then **0 commits**; detect-only
  over-commits. Preserves "each rung adds exactly one mechanism."
- **(ii) discovery:** arms 5/6 start from `{c0}` only and must *find* the second regime.
  Maps more directly onto T2's "ATLAS ≈ 2 committed vs. detect-only > 2", but breaks
  monotonicity against arm 4.

Implement **(i) as default**, expose `--expansion-start-library {full,c0_only}`, and
**ask the user which to report before the real run.** Both are cheap once the code exists.

## 2c — `run_e4_episode()`

```python
def run_e4_episode(
    state: ArmState,
    agent,                                # GC_Agent, pre-configured
    world_model,                          # EncPredWM wrapper
    base_env,                             # raw PushTEnv(render_size=224, with_velocity=True)
    regimes: dict[str, PhysicsRegime],    # {"R0": PhysicsRegime(base_env,"R0"), ...}
    spec: EpisodeSpec,
    dataset_states, dataset_seq_lengths,  # run_e0_planning.load_dataset_states()
    n_replans_target: int, frameskip: int, num_act_stepped: int,
    max_raw_steps: int, motion_gate: float | None,
    out_dir: Path, seed_run: int,
) -> dict[str, Any]:
```

Body — reuse, do not reinvent:

1. `regime = regimes[spec.regime]`; `regime_label = REGIME_LABELS[spec.regime]`;
   `router_rng = random.Random(spec.seed)`.

2. **Init/goal — E0's corrected sampler, not E1's:**
   ```python
   from scripts.run_e0_planning import (sample_dataset_init_goal, block_success,
                                        prepare_with_visual, make_obs_td)
   rs = np.random.RandomState(spec.seed)
   init_state, goal_state = sample_dataset_init_goal(dataset_states, dataset_seq_lengths, rs)
   ```
   `spec.seed` comes from `paired_seed(segment_idx, episode_idx)` and is **arm-independent**
   — that is gate G5, and the whole basis of T2's paired statistics.

3. `goal_obs, _ = prepare_with_visual(base_env, regime, spec.seed, goal_state)` →
   `agent.set_goal(make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))`; then the
   same for `init_state`.

4. **Replan loop**, `for replan_idx in range(n_replans_target)`, breaking on
   `elapsed >= max_raw_steps`:

   - **SCORE + SELECT + EXPAND** (arms 4/5/6/7, only when `prev_chunk` exists):
     ```python
     step_info = atlas_step(
         library=state.library, expander=state.expander, world_model=world_model,
         encoder_output=prev_enc, actions=prev_act, current_idx=state.current_idx,
         cfg=state.cfg, regime_label=regime_label,
         next_encoder_output=None, next_actions=None,   # filled on the NEXT iteration
         proprio_ctxt=prev_proprio, rng=router_rng,
     )
     state.current_idx = step_info.selected_idx
     ```
     **The one-replan delay is structural, not a bug:** `Expander.maybe_expand` verifies on
     the *next unseen chunk*. Keep a two-deep chunk buffer — chunk `k` is deficit data,
     chunk `k+1` is the held-out verification set.

   - **EXECUTE:** `chart.apply_(predictor)` →
     `agent.act(obs_td, steps_left=max((n_replans_target - replan_idx) * num_act_stepped, 1))`
     → `finally: chart.restore_(predictor)`. `steps_left` is in **model-chunk units**
     (`harness.py:364-368`) — do not multiply by frameskip.

   - **Step:** `rearrange(action.cpu(), "t (f d) -> (t f) d", d=2)` →
     `agent.preprocessor.denormalize_actions(...).numpy()` → `base_env.step(a)` per raw
     action, accumulating `info["n_contacts"]` and checking
     `block_success(goal_state, info["state"])` each step.

   - **Re-encode the executed chunk — copy `atlas/harness.py:401-423` verbatim.**
     (frame subsampling to `keep_idx = range(0, n_raw+1, frameskip)`;
     `world_model.encode({...})`; `enc["visual"].squeeze(0).squeeze(1).flatten(1,2)`;
     action normalise + reshape to `[T_model, frameskip*2]`;
     `proprio_ctxt = enc["proprio"][:, 0:1]`.) This block is subtle and already correct;
     re-deriving it is exactly how the 5× time-base bug got in last time.

   - **REFINE — last, always:**
     - arms 2/3: `state.adapter.push(enc_out, act_model, proprio_ctxt); state.adapter.refine()`
     - arms 4/5/6: `if state.current_idx != 0: atlas_refine(state.library[state.current_idx], world_model, enc_out, act_model, proprio_ctxt=proprio_ctxt, optimizer=state.optimizers.setdefault(...))`
     - arms 1/7: nothing

5. Emit the record (§2d), write it with `harness.log_episode(out_dir, record)`
   (`atlas/harness.py:486`) — read-only use of `harness.py`, no edit.

## 2d — Per-episode JSONL contract

`make_t2`, `make_f1` and `make_f2` already read specific keys. Emit **all** of these or the
downstream scripts silently produce empty tables:

```python
{
  # required by make_t2 / make_f1 / make_f2
  "arm": str, "success": bool, "segment_idx": int, "global_episode_idx": int,
  "probe_outcome": str,             # committed | rejected_score | rejected_full | not_ready
  "library_size": int,
  "charts_committed_cumulative": int, "probes_rejected_cumulative": int,
  # pairing + parity with E1
  "seed_run": int, "episode_idx": int, "regime": str, "regime_label": int, "seed": int,
  "selected_trace": list[int], "umf_trace": list[list[float | None]], "strikes": int,
  "elapsed_raw_steps": int, "n_replans": int, "raw_steps_per_replan": list[int],
  # E0 parity -- difficulty + the pre-registered mechanism metric
  "init_block_pos_diff": float, "init_block_angle_diff": float,
  "init_agent_block_dist": float, "total_contacts": int,
  "block_pos_diff": float, "block_angle_diff": float,
  "refine_loss": float | None, "wall_time": float,
}
```

**Knock-away** (`block_pos_diff > init_block_pos_diff`) is E0's pre-registered mechanism
metric for damping (`E0_RECOVERY_PLAN` §0.5). S2's B segment *is* damping, so report
knock-away count and mean damage per arm in T2 alongside SR. Derive it at table time from
the two fields above — no extra logging needed.

---

# Phase 3 — `scripts/run_e4.py`

Replace the stub body. Structure on `scripts/run_e1.py:213-347`, with these deltas.

**Constants**, with the deviation recorded inline:
```python
FRAMESKIP = 5
CEM_NUM_SAMPLES = 300
CEM_ITERATIONS = 30          # cut to 10 per Phase 0.2's ladder, after profiling
CEM_HORIZON = 6
CEM_NUM_ACT_STEPPED = 1      # E4 DEVIATION: nas=6 gives ONE replan per 30-raw-step
                             # episode, which structurally cannot exercise routing,
                             # refinement, or Expander's next-chunk verification.
                             # nas=1 -> 5 raw actions/replan -> 6 replans in 30 raw steps.
                             # Same reasoning as E0_RECOVERY_PLAN P5's fix for E1;
                             # document as a second deviation in the style of
                             # ATLAS_implementation_plan_v2.md §7.0a.
MAX_MPC_STEPS = 30
HYSTERESIS = 0.05
REGIME_LABELS = {"R0": 0, "R1": 1, "R2": 2}
```

**CLI:** `--stream s2`, `--arms`, `--episodes`, `--seeds`, `--segment-regimes R0 R2`,
`--charts` (dir of `chart_{kind}_{regime}.pt`), `--kind {ln_act,lora4,full}`,
`--expansion-start-library {full,c0_only}`, all CEM args, `--max-mpc-steps`, `--out`,
`--profile`.

**Hub path:** use `run_e0_planning.py:74-82`'s form —
`os.environ.get("ATLAS_HOME", str(atlas.ATLAS_HOME))`, **not** `atlas.ATLAS_HOME` directly
(`.resolve()` breaks on the Modal volume mount) and **not** the stub's current remote
`torch.hub.load("facebookresearch/jepa-wms", ...)`.

**Model load:** copy `run_e1.py:251-268` exactly, including `use_sdpa = True` on every
predictor module and `torch.set_float32_matmul_precision("high")` (T7 throughput).

**Motion gate:** compute **once, from the A-segment regime (R0)**, and hold it fixed for
the whole stream. A per-regime gate would shift the informative-chunk definition underneath
the strike counter — exactly what gate G6 exists to prevent. Reuse `run_e1.py:279-288`
verbatim (`load_regime_trajectories(model, prep, "R0", num_trajs=3, traj_len=10,
device=device, seed_offset=20_000)` → `compute_motion_gate`). Log the value into the
summary JSON.

**Envs:** one `base_env = PushTEnv(render_size=224, with_velocity=True)` and
`regimes = {r: PhysicsRegime(base_env, r) for r in set(segment_regimes)}`, built once —
`PhysicsRegime` is stateless besides `_cfg` and applies physics inside `reset()`.

**Loop — arms outermost:**
```python
for arm in args.arms:
    for seed_run in range(args.seeds):
        wm.predictor.load_state_dict(pristine_predictor_state)   # clean slate
        state = build_arm_state(arm, ...)                        # fresh library/expander/adapter
        for spec in streams[seed_run]:                           # 6 segments x N, in order
            if arm == "adajepa":
                state.adapter.reset()
            record = run_e4_episode(state, ..., spec=spec, seed_run=seed_run)
```
Delete a stale `out/episodes.jsonl` once before the loop (`log_episode` appends —
`run_e1.py:308-310`).

**Resume support:** copy `run_e0_planning.py:426-474`'s pattern (read existing JSONL, skip
completed `(arm, seed_run, global_episode_idx)` triples). A 35 GPU-h run *will* be
interrupted; without this it restarts from zero.

**Summary JSON** (`out/e4_summary.json`): arms, stream, segment_regimes, episodes, seeds,
every CEM value, `expansion_start_library`, `motion_gate`, the resolved `REGIME_CONFIGS`
for both segment regimes, `chart_kind`, `charts_dir`, per-arm SR, mean wall time, peak GPU
memory. The resolved regime configs matter — a chart trained under one physics and evaluated
under another is a silent, invalidating mismatch nothing else would catch.

**`--profile` mode:** run one arm (`atlas`) for `--episodes` episodes on segment 0 only;
print sec/episode, sec/replan, peak VRAM, and the extrapolated total GPU-h for the full
grid; then exit **without** running anything else.

---

# Phase 4 — Smoke test (`scripts/smoke_e4.py`)

Model on `scripts/smoke_e1.py`. Tiny CEM (`num_samples=2, iterations=2, horizon=2` —
jepa-wms's own `quick_debug` scale), 2 episodes per segment, 2 segments, **all 7 arms**.
Must assert and print:

- [ ] every arm completes without exception (this is what catches Phase 1's fixes)
- [ ] identical `init_block_pos_diff` per `global_episode_idx` across **all 7 arms** (gate G5)
- [ ] `raw_steps_per_replan == [5]*6` — proves `nas=1` took effect
- [ ] `umf_trace` finite and O(1) on informative chunks, `None` on gated ones
- [ ] `oracle_id` selects index 0 in R0 segments, index 1 in B segments
- [ ] `frozen` never changes `library_size`; `atlas_fixed` never commits
- [ ] `atlas_detect` commits when strikes reach q=3; `atlas` commits only after a probe passes
- [ ] arm 2 (`adajepa`) predictor state is bit-identical at each episode start; arm 3 is not
- [ ] the JSONL contains every key in §2d

**Milestone gate:** `python scripts/smoke_gates.py --all`.
**G1 caveat, do not paper over it:** `gate_g1` has the legacy-gym API bug (`env.reset(seed=)`
+ 5-tuple `step()`) *and* its design never applies a chart or calls the model — it only
re-runs an env rollout twice and checks determinism. `--all` passing therefore does **not**
verify chart apply/restore. Report G1 as skipped/vacuous, never as a pass.

---

# Phase 5 — 🛑 Profile, report, agree the cuts

```bash
python scripts/run_e4.py --profile --episodes 3 --num-samples 16 --iterations 3
```

Then at realistic scale on Modal (`--profile --episodes 2` at full CEM) to get the honest
sec/episode. **Report to the user and stop:**

1. measured sec/episode and sec/replan, and the extrapolated total GPU-h for the full grid;
2. the Phase 0.2 cut ladder decision (`iterations`, `episodes`, `seeds`);
3. validation that `iterations=10` does not collapse the frozen baseline — frozen@R0, N=10,
   SR must stay ≈85%+ (compare against the measured 95.0% at `iterations=30`);
4. the `--expansion-start-library` reading (§2b);
5. which E0 chart is `chart_B`. Nothing is decided until P4 reports; placeholder is
   `atlas_out/e0_v3_dataset/chart_ln_act_R2.pt` — **note it lives on the Modal volume
   `atlas-data`, not on local disk.**

Do not launch anything larger than the smoke test before this is answered.

---

# Phase 6 — Modal runner and the real run

## 6a — `modal/modal_e4.py`

Copy `modal/modal_e0_planning.py` structurally: same volume (`atlas-data`), same
`/atlas_root` mount, same `/src` local-dir image, same
`uv pip install -e vendor/jepa-wms` → torch cu121 → `-e .` chain, same `.env({...})` block,
`gpu="L4"`, `timeout=3600*6`, remote fn = `atlas_volume.reload()` →
`subprocess.run(cmd, check=True, cwd="/src")` → `atlas_volume.commit()`.

Expose **`arm` and `seed_run` as single-valued args** so the grid fans out across
containers — this is Phase 0.2's cut #4 and the only reason the run finishes in a day.
Modal can't pass lists from the CLI: use comma-separated strings + `.split(",")`, as
`run_e0_train` does at `modal_e0_planning.py:218-219`.

Each container writes `episodes_{arm}_{seed_run}.jsonl` into the same `out_subdir`; add a
small `--merge` mode that concatenates them into `episodes.jsonl` for the table/figure
scripts.

## 6b — Launch

```bash
for arm in frozen adajepa adajepa_persist atlas_fixed atlas_detect atlas oracle_id; do
  modal run --detach modal/modal_e4.py --arm $arm --seed-run 0 \
      --episodes 10 --iterations 10 --segment-regimes "R0,R2" \
      --charts-subdir e0_v3_dataset --kind ln_act --out-subdir e4_v1
done
```

**All Modal runs use `--detach`** (standing instruction). Known trap: `--detach` survives a
network blip but **not** the local launcher being killed — reconnect with
`modal app logs <app-id>`.

---

# Phase 7 — T2, F1, F2b

## 7a — `make_t2`

Extend **`scripts/make_tables.py::make_t2` only** (leave `make_t1` alone — E0/E1 territory).
Two columns the plan's T2 requires are missing; `mcnemar_paired` is imported but unused.

```
| Arm | SR overall [CI] | SR first visit A | SR final revisit A | paired Δ vs frozen [CI] |
  McNemar p | Charts committed | Probes rejected | Knock-aways | Mean damage |
```

- Pair on `(seed_run, global_episode_idx)` — sort records exactly as
  `run_e1.py::compute_t1:161-163` does. Never an unpaired test (`CLAUDE.md` §5).
- `first visit A` = `segment_idx == 0`; `final revisit A` = `segment_idx == 4`.
- Use `stats.success_rate_ci`, `stats.paired_bootstrap`, `stats.mcnemar_paired` as-is.
- Fix the existing fragility near `make_tables.py:~100`: `charts_committed` /
  `probes_rejected` are read from `eps[-1:]` only, which breaks if arms interleave in the
  JSONL. Take `max(...)` over that arm's records instead.

## 7b — Figures

F1 and F2b need no new plotting code — `make_figures.make_f1`/`make_f2` already call
`plots.money_plot` / `plots.two_panel`. But `make_f1` **hardcodes** `eps_per_seg = 20`,
6 segments and `["R0","R1",...]`; parameterise those from the run's summary JSON so a
10-episode R0/R2 run plots correctly.

## 7c — 🛑 Before reporting T2 as a result

E4 inherits E0's negative finding. If `ln_act` recovers ~10% of the gap offline, the ATLAS
arm may not separate from frozen. **That is a result** (`CLAUDE.md` §1.8), and the ladder's
*shape* — detect-only over-committing, Persistent-AdaJEPA degrading on revisit — can still
carry T2. Do not tune anything to rescue it.

---

# Phase 8 — Housekeeping (independent, any time)

- `configs/regimes/pusht.yaml` still declares R1 = `mass_scale: 0.2` and R2 = `damping: 0.3`
  — both superseded (mass is physically dead against a kinematic pusher). Live truth is
  `atlas/regimes.py:55-66`: R1 `friction: 2.0`, R2 `damping: 0.5`. Nothing loads this file
  today, so this is purely removing a live foot-gun.
- `configs/atlas/default.yaml`'s planner block is still AdaJEPA's (`cem_samples 200,
  horizon 25`); correct it to the substrate config (`300/30/6`) plus E4's `nas=1`.
- `configs/atlas/e4.yaml`'s budget-cut ladder is written against those stale numbers —
  replace it with Phase 0.2's ladder.

---

# Verification summary

Run the gate for whatever you touched; `--all` at milestones. Gate names are **uppercase**.

| After | Command | Must show |
|---|---|---|
| 1a | the 2-step `AdaJEPA.refine()` scratch script | two finite, changing losses |
| 1b | `smoke_gates.py --gate G3a`, `--gate G3b`, `--gate G6` | all PASS |
| 1c | `pytest tests/test_streams.py -q` + `--gate G5` | pass; S2 = `R0,R2,R0,R2,R0,R2` |
| 2–3 | `run_e4.py --profile --episodes 3 --num-samples 16 --iterations 3` | sec/episode + extrapolated GPU-h |
| 4 | `python scripts/smoke_e4.py` | every checkbox in Phase 4, printed |
| 4 | `python scripts/smoke_gates.py --all` | all pass — **G1 reported as skipped/vacuous** |
| 7 | `make_tables.py --table T2`, `make_figures.py --fig F1` on smoke output | T2.md with all columns; F1.pdf renders |

**Report every task in four buckets:** what changed · which files · what you ran and its
exact output · what you did *not* run.

**Placeholder inputs to substitute later** (per "treat as some path for now"):
`--charts atlas_out/e0_v3_dataset`, `--kind ln_act`, `--segment-regimes R0 R2`. All three
are CLI flags — swapping them is a command-line change, not a code change.

---

# Appendix A — E2 (2×2 appearance vs dynamics), ~90% reuse

Only worth starting after Phase 4 works. Three deltas from E4.

**A1 — `VisualCorruption` is broken against the real env.** `atlas/regimes.py:194` calls
`obs.ndim` on what PushTEnv actually returns — a dict `{"visual","proprio"}`. It has never
been run against Push-T.

```python
def observation(self, obs):
    if self.kind == "none":
        return obs
    if isinstance(obs, dict):
        obs = dict(obs)
        obs["visual"] = _corrupt(obs["visual"], self.kind, self.severity)
        return obs
    return obs if obs.ndim == 1 else _corrupt(obs, self.kind, self.severity)
```

Also give `_corrupt`'s `salt_pepper` branch a seeded RNG — it calls
`np.random.default_rng()` unseeded, which breaks paired reproducibility. And apply the
corruption to the **goal** observation too, or the planner chases a goal image from a
different visual distribution than its observations.

**A2 — E2 measures routing accuracy, not success rate.** Per episode, log the router's
selected index and the ground-truth correct index (`label_to_chart[regime_label]`);
accuracy = fraction correct. `run_e2.py` runs `{umf, sdyn}` × cells A–D, 40 episodes ×
3 seeds — reuse `run_e4_episode` with `expansion_mode="fixed"` and no refinement, making it
a pure routing measurement.

**A3 — Cell C is an assertion, not a number.** With `expansion_mode="atlas"`, Cell C
(different look, same physics) must yield `charts_committed == 0`. Assert and log it.

`plots.two_panel`'s F2a panel already consumes `routing_accuracy: {router: {cell: acc}}`,
and `make_figures.make_f2` already reads `atlas_out/e2/episodes.jsonl`.
