"""
harness_e4.py — E4/E3 continual-stream episode runner.

NEW FILE, deliberately outside atlas/harness.py (CLAUDE.md §3's module-
ownership rule would put this in harness.py) — E3_E4_IMPLEMENTATION_PLAN.md
§0.3's file-ownership split keeps the E3/E4 agent off the E0/E1 agent's
harness.py entirely (P5 there edits run_e1_episode concurrently). This
module should be folded into harness.py once P5 has landed.

Runs one continual-stream episode for one of the 7 E4/E3 arms:
  1. frozen            — plan-only, no adaptation
  2. adajepa            — AdaJEPA, reset per episode
  3. adajepa_persist    — AdaJEPA, no reset (cross-episode retention)
  4. atlas_fixed        — atlas_step()/atlas_refine(), expansion_mode='fixed'
  5. atlas_detect       — expansion_mode='detect_only'
  6. atlas              — expansion_mode='atlas' (full verification)
  7. oracle_id          — oracle_id router, no refinement

ArmState persists ACROSS episodes within one (arm, seed_run) run — this is
what makes E4 continual rather than 2 520 independent episodes: current_idx,
the library, the expander's strike counter, and each chart's per-index Adam
optimizer state all carry over from one episode to the next.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from einops import rearrange

from atlas.adajepa import AdaJEPA
from atlas.chart import Chart
from atlas.expand import Expander, ExpansionConfig
from atlas.library import Library
from atlas.loop import ATLASConfig, atlas_step, atlas_refine_buffered
from atlas.streams import EpisodeSpec
from atlas.harness import log_episode

ArmName = Literal[
    "frozen", "adajepa", "adajepa_persist",
    "atlas_fixed", "atlas_detect", "atlas", "oracle_id",
]


@dataclass
class ArmState:
    """State that persists ACROSS episodes for one (arm, seed_run) run."""
    arm: ArmName
    library: Library | None          # None for frozen / adajepa / adajepa_persist
    expander: Expander | None
    adapter: AdaJEPA | None          # arms 2, 3 only
    cfg: ATLASConfig
    label_to_chart: dict[int, int] | None
    current_idx: int = 0             # persists across episodes -- this IS recall
    optimizers: dict[int, torch.optim.Adam] = field(default_factory=dict)
    charts_committed_cumulative: int = 0
    probes_rejected_cumulative: int = 0
    # FIX_SPEC.md B8 (design decision, recorded in FIXLOG.md): a persistent
    # CLONE of c0, lazily created the first time routing selects index 0,
    # for arms 4/5/6 only. c0 ITSELF (library[0]) is NEVER refined --
    # CLAUDE.md §1 non-negotiable #4 (identity initialisation) and Gate G1
    # both depend on c0 staying pristine, and routing/UMF SCORE always
    # evaluates the real library (unaffected by this field). This clone
    # exists purely so EXECUTE+REFINE can adapt continuously once c0 is
    # selected, the same way AdaJEPA (arms 2/3) adapts from the first
    # replan regardless of any "selection" -- see the REFINE section below.
    c0_adapted_chart: Chart | None = None
    # FIX_SPEC.md B9: one 5-chunk sliding buffer per chart index (+ the
    # "c0_clone" key for c0's adapted clone), matching AdaJEPA's
    # BUFFER_SIZE=5 so arms 4/5/6 refine over the same amount of recent
    # data per step as arms 2/3 (plan §7.6). Keyed the same way as
    # `optimizers` (int chart index, or the literal string "c0_clone").
    refine_buffers: dict = field(default_factory=dict)


def build_arm_state(
    arm: ArmName,
    predictor,
    world_model,
    kind: str,
    chart_b_path: Path,
    cfg: ATLASConfig,
    expansion_start_library: Literal["full", "c0_only"] = "full",
) -> ArmState:
    """
    Construct a fresh ArmState for `arm` against the CURRENT (pristine)
    predictor state -- caller must have already reloaded the pristine
    predictor state_dict before calling this (see run_e4.py's per-(arm,
    seed_run) loop).

    Args:
        chart_b_path:  Path to the E0-trained chart for regime B (the
                       stream's shifted regime) -- placeholder input, see
                       plan §"Placeholder inputs to substitute later".
        expansion_start_library: 'full' (default, plan §2b(i)) starts arms
                       4/5/6 from {c0, chart_B} -- the monotone-ladder
                       reading, where correct ATLAS behaviour is 0 commits.
                       'c0_only' (plan §2b(ii)) starts them from {c0} only,
                       requiring genuine discovery of the second regime.

    run_e4_episode() encodes regime_label as R0=0, stream's shifted regime
    (segment B)=1 -- so oracle_id's label_to_chart is always {0: 0, 1: idx_b},
    where idx_b is chart_B's index in the library (plan §2c: "c0 is the
    correct R0 chart -- the identity chart on the unshifted regime IS the
    oracle answer", a design decision, not an accident).
    """
    if arm == "frozen":
        return ArmState(arm=arm, library=None, expander=None, adapter=None,
                         cfg=cfg, label_to_chart=None)

    if arm in ("adajepa", "adajepa_persist"):
        variant = "adajepa" if arm == "adajepa" else "persistent"
        param_names = Chart(predictor, kind)._param_names
        adapter = AdaJEPA(world_model, param_names, variant=variant, lr=cfg.lr)
        return ArmState(arm=arm, library=None, expander=None, adapter=adapter,
                         cfg=cfg, label_to_chart=None)

    if arm == "oracle_id":
        c0 = Chart(predictor, kind)
        library = Library(c0, max_size=cfg.k_max)
        chart_b = Chart.load(chart_b_path, predictor)
        idx_b = library.add(chart_b)
        oracle_map = {0: 0, 1: idx_b}
        return ArmState(arm=arm, library=library, expander=None, adapter=None,
                         cfg=cfg, label_to_chart=oracle_map)

    # atlas_fixed / atlas_detect / atlas
    expansion_mode = {"atlas_fixed": "fixed", "atlas_detect": "detect_only", "atlas": "atlas"}[arm]
    arm_cfg = ATLASConfig(
        router=cfg.router, tau=cfg.tau, q=cfg.q, hysteresis=cfg.hysteresis, lr=cfg.lr,
        n_probe=cfg.n_probe, motion_gate=cfg.motion_gate, k_max=cfg.k_max,
        label_to_chart=None, expansion_mode=expansion_mode,
    )
    c0 = Chart(predictor, kind)
    library = Library(c0, max_size=cfg.k_max)
    if expansion_start_library == "full":
        chart_b = Chart.load(chart_b_path, predictor)
        library.add(chart_b)
    expander = Expander(ExpansionConfig(tau=cfg.tau, q=cfg.q, n_probe=cfg.n_probe, lr=cfg.lr, kind=kind))
    return ArmState(arm=arm, library=library, expander=expander, adapter=None,
                     cfg=arm_cfg, label_to_chart=None)


def _make_obs_td(visual_hw3_uint8, proprio_vec, device: str):
    from tensordict import TensorDict
    visual = torch.from_numpy(visual_hw3_uint8.copy()).permute(2, 0, 1).float().unsqueeze(0)
    proprio = torch.from_numpy(np.asarray(proprio_vec, dtype=np.float32)).unsqueeze(0)
    return TensorDict({"visual": visual, "proprio": proprio}, batch_size=[]).to(device)


def run_e4_episode(
    state: ArmState,
    agent,                                # GC_Agent, pre-configured
    world_model,                          # EncPredWM wrapper
    base_env,                             # raw PushTEnv(render_size=224, with_velocity=True)
    regimes: dict[str, Any],              # {"R0": PhysicsRegime(base_env,"R0"), ...}
    spec: EpisodeSpec,
    dataset_states, dataset_seq_lengths,  # run_e0_planning.load_dataset_states()
    n_replans_target: int, frameskip: int, num_act_stepped: int,
    max_raw_steps: int, motion_gate: float | None,
    out_dir: Path, seed_run: int,
    router_rng_seed: int | None = None,
) -> dict[str, Any]:
    """
    Run one E4/E3 episode for arm `state.arm`, mutating `state` in place
    (current_idx, library, expander strike counter, AdaJEPA buffer/weights
    all persist for the NEXT call -- this is the continual-stream mechanism).

    Reuses E0's corrected init/goal sampler and success metric (block-only,
    not the agent-position term goal_utils.eval_state() would apply to an
    unrelated random goal -- see run_e0_planning.py's block_success()
    docstring), NOT E1's PushTWrapper.sample_random_init_goal_states (which
    draws a synthetic random goal, not a real recorded one).
    """
    from scripts.run_e0_planning import (
        sample_dataset_init_goal, block_success, prepare_with_visual, make_obs_td,
    )

    device = agent.device
    # FIX_SPEC.md B11: agent.local_gpu_generator (shared BY REFERENCE with
    # agent.planner's own CEMPlanner.local_generator -- gc_agent.py:63,
    # the generator CEMPlanner.plan() actually draws candidate actions
    # from, planner.py:290-291) is seeded ONCE at GC_Agent construction
    # from cfg.local_seed and never reseeded per episode. So CEM's
    # candidate sampling for episode N depends on how many prior draws
    # happened earlier in THIS PROCESS, not on spec.seed -- a resumed
    # run's generator state at episode N differs from an uninterrupted
    # run's state at the same episode (different draw history), so resume
    # is not reproducible even though every other input (env seed, chart
    # state) is. Reseed from spec.seed at the start of every episode so
    # resumed and uninterrupted runs draw identically. Only touches this
    # file (atlas/harness_e4.py) -- calls a public method on an object
    # already passed in, does not modify the upstream gc_agent.py/
    # planner.py files (CLAUDE.md §1.3's "one upstream hook only").
    if hasattr(agent, "local_gpu_generator"):
        agent.local_gpu_generator.manual_seed(spec.seed)
    if hasattr(agent, "local_generator"):
        agent.local_generator.manual_seed(spec.seed)

    predictor = world_model.model.predictor
    regime = regimes[spec.regime]
    from atlas.regimes import assert_no_bypassed_corruption
    assert_no_bypassed_corruption(regime)  # FIX_SPEC.md C5 -- see docstring
    router_rng = None
    if router_rng_seed is not None:
        import random as _random
        router_rng = _random.Random(router_rng_seed)

    rs = np.random.RandomState(spec.seed)
    init_state, goal_state = sample_dataset_init_goal(dataset_states, dataset_seq_lengths, rs)
    init_block_pos_diff = float(np.linalg.norm(goal_state[2:4] - init_state[2:4]))
    init_block_angle_diff = float(np.abs((goal_state[4] - init_state[4] + np.pi) % (2 * np.pi) - np.pi))
    init_agent_block_dist = float(np.linalg.norm(init_state[0:2] - init_state[2:4]))

    goal_obs, _ = prepare_with_visual(base_env, regime, spec.seed, goal_state)
    agent.set_goal(make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))
    obs, _ = prepare_with_visual(base_env, regime, spec.seed, init_state)

    regime_label = 0 if spec.regime == "R0" else 1  # R0=0, stream's shifted regime (B)=1

    # FIX_SPEC.md B2: AdaJEPA.reset() had no production caller anywhere --
    # in production, plain "adajepa" never re-initialised, making it
    # behaviourally identical to "adajepa_persist" (the persistence rung
    # this ladder exists to isolate would measure zero by construction).
    # reset() is a no-op for variant="persistent" (see adajepa.py's own
    # docstring), so this is safe to call unconditionally for both arms 2
    # and 3 -- but only arm 2 ("adajepa") actually resets, restoring
    # pretrained predictor weights and clearing its 5-transition buffer at
    # the START of every episode. Key-namespace concern pre-cleared
    # (SUBMISSION_PLAN.md A-xi): for ln_act, pretrained_state/param_names
    # come from the same plain nn.Module with no parametrization, so
    # load_state_dict(strict=False) genuinely restores.
    if state.arm in ("adajepa", "adajepa_persist"):
        state.adapter.reset()

    elapsed = 0
    success = False
    total_contacts = 0
    selected_trace: list[int] = []
    umf_trace: list[list[float | None]] = []
    raw_steps_per_replan: list[int] = []
    probe_outcome = "not_ready"
    refine_loss: float | None = None
    final_check = {"block_pos_diff": init_block_pos_diff, "block_angle_diff": init_block_angle_diff}

    # Single-chunk buffer for arms that don't need a held-out verification
    # chunk (atlas_fixed/atlas_detect/oracle_id/adajepa*): unchanged from
    # before.
    prev_chunk: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None

    # FIX_SPEC.md B1: two-deep chunk buffer, ATLAS ARM ONLY
    # (expansion_mode='atlas', full verification). Previously atlas_step()
    # was called with next_encoder_output=next_actions=None
    # UNCONDITIONALLY (harness_e4.py:216-217 pre-fix), so
    # atlas/loop.py:140-150's guard (`next_encoder_output is not None and
    # ... strikes >= q`) was structurally unsatisfiable -- the 'atlas' arm
    # could never reach maybe_expand() at all, so it could never commit.
    # 'atlas_fixed' (expansion_mode='fixed') and 'atlas_detect'
    # (expansion_mode='detect_only') never read next_encoder_output in the
    # first place (atlas/loop.py's 'fixed'/'detect_only' branches don't
    # touch it), so they are UNAFFECTED by this and keep using prev_chunk
    # directly, undelayed.
    #
    # For 'atlas' specifically: hold a sliding 2-chunk window
    # [chunk_{i-1}, chunk_i]. chunk_{i-1} (older) is the deficit-candidate
    # passed as encoder_output/actions (record()'d as usual); chunk_i (one
    # replan FRESHER) is passed as next_encoder_output/next_actions -- the
    # "next unseen chunk" maybe_expand() verifies a just-armed probe on.
    # Copies scripts/run_e2.py:296-339's peek-one-ahead pattern (verified
    # leakage-free, CODE_AUDIT.md §9.3), adapted to an ONLINE rollout where
    # the "next" chunk does not exist until the FOLLOWING replan produces
    # it. E3_E4_IMPLEMENTATION_PLAN.md §2c: "the one-replan delay is
    # structural, not a bug" -- SELECT for arm='atlas' is therefore based
    # on the one-replan-STALER of the two buffered chunks, at the cost of
    # one extra warmup replan before its first real routing decision
    # (needs 2 chunks buffered, not 1), vs. atlas_fixed/atlas_detect.
    atlas_verify_buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []

    t_start = time.time()
    for replan_idx in range(n_replans_target):
        if elapsed >= max_raw_steps:
            break

        # ── SCORE + SELECT + EXPAND (arms 4/5/6/7 only, once a chunk exists) ──
        if state.arm == "atlas":
            if len(atlas_verify_buffer) == 2:
                (d_enc, d_acts, d_proprio), (v_enc, v_acts, v_proprio) = atlas_verify_buffer
                step_info = atlas_step(
                    library=state.library, expander=state.expander, world_model=world_model,
                    encoder_output=d_enc, actions=d_acts, current_idx=state.current_idx,
                    cfg=state.cfg, regime_label=regime_label,
                    next_encoder_output=v_enc, next_actions=v_acts,
                    proprio_ctxt=d_proprio, next_proprio_ctxt=v_proprio, rng=router_rng,
                )
                state.current_idx = step_info.selected_idx
                umf_trace.append(step_info.scores)
                if step_info.probe_outcome != "not_ready":
                    probe_outcome = step_info.probe_outcome
                    if probe_outcome == "committed":
                        state.charts_committed_cumulative += 1
                    elif probe_outcome.startswith("rejected"):
                        state.probes_rejected_cumulative += 1
            else:
                umf_trace.append([None] * (len(state.library) if state.library is not None else 1))
        elif prev_chunk is not None:
            enc_out, acts, proprio_ctxt = prev_chunk
            if state.arm in ("atlas_fixed", "atlas_detect"):
                step_info = atlas_step(
                    library=state.library, expander=state.expander, world_model=world_model,
                    encoder_output=enc_out, actions=acts, current_idx=state.current_idx,
                    cfg=state.cfg, regime_label=regime_label,
                    next_encoder_output=None, next_actions=None,
                    proprio_ctxt=proprio_ctxt, rng=router_rng,
                )
                state.current_idx = step_info.selected_idx
                umf_trace.append(step_info.scores)
                if step_info.probe_outcome != "not_ready":
                    probe_outcome = step_info.probe_outcome
                    if probe_outcome == "committed":
                        state.charts_committed_cumulative += 1
                    elif probe_outcome.startswith("rejected"):
                        state.probes_rejected_cumulative += 1
            elif state.arm == "oracle_id":
                from atlas.router import route
                state.current_idx, route_info = route(
                    kind="oracle_id", library=state.library, world_model=world_model,
                    encoder_output=enc_out, actions=acts, current_idx=state.current_idx,
                    regime_label=regime_label, label_to_chart=state.label_to_chart,
                )
                umf_trace.append(route_info["scores"])
            else:
                umf_trace.append([None])
        else:
            umf_trace.append([None] * (len(state.library) if state.library is not None else 1))
        selected_trace.append(state.current_idx)

        # ── EXECUTE ──────────────────────────────────────────────────────────
        # FIX_SPEC.md B8: once a c0-adapted clone exists for this arm (see
        # ArmState.c0_adapted_chart), USE IT instead of pristine c0 whenever
        # routing selects index 0 -- planning/execution should reflect
        # whatever adaptation has accumulated, exactly as it would for any
        # other library chart. c0 itself is never substituted for routing/
        # SCORE (atlas_step() above always reads the real library).
        if state.library is not None and state.current_idx == 0 and state.c0_adapted_chart is not None:
            chart = state.c0_adapted_chart
        elif state.library is not None:
            chart = state.library[state.current_idx]
        else:
            chart = None
        if chart is not None:
            chart.apply_(predictor)
        try:
            obs_td = _make_obs_td(obs["visual"], obs["proprio"], device)
            steps_left_model = (n_replans_target - replan_idx) * num_act_stepped
            action = agent.act(obs_td, steps_left=max(steps_left_model, 1))
        finally:
            if chart is not None:
                chart.restore_(predictor)

        raw_actions = rearrange(action.cpu(), "t (f d) -> (t f) d", d=2)
        raw_actions = agent.preprocessor.denormalize_actions(raw_actions).numpy()

        imgs = [obs["visual"]]
        proprios = [obs["proprio"]]
        step_actions = []
        for a in raw_actions:
            if elapsed >= max_raw_steps:
                break
            obs, reward, done, info = base_env.step(a)
            imgs.append(obs["visual"])
            proprios.append(obs["proprio"])
            step_actions.append(a)
            elapsed += 1
            total_contacts += info["n_contacts"]
            final_check = block_success(goal_state, info["state"])
            if final_check["success"]:
                success = True
                break
        raw_steps_per_replan.append(len(step_actions))

        # ── Re-encode the executed chunk (verbatim pattern from
        # atlas/harness.py::run_e1_episode:401-423 — subtle time-base handling,
        # re-derive-risk flagged in the plan) ──────────────────────────────────
        n_raw = (len(step_actions) // frameskip) * frameskip
        if n_raw == 0:
            prev_chunk = None if not success else prev_chunk
            # atlas_verify_buffer intentionally left as-is: no new chunk was
            # produced this replan (e.g. success mid-chunk before a full
            # frameskip block), so there is nothing to slide in.
        else:
            keep_idx = list(range(0, n_raw + 1, frameskip))
            imgs_sub = np.stack([imgs[i] for i in keep_idx], axis=0)
            proprios_sub = np.stack([proprios[i] for i in keep_idx], axis=0)
            visual_t = torch.from_numpy(imgs_sub.copy()).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
            proprio_t = torch.from_numpy(proprios_sub.astype(np.float32)).unsqueeze(0).to(device)
            # FIX_SPEC.md D10 (SUBMISSION_PLAN.md Part A-vi) -- see
            # atlas/harness.py's identical assertion for the full rationale.
            assert visual_t.shape[0] == 1 and proprio_t.shape[0] == 1, (
                f"D10: expected a single-episode batch (dim0=1), got "
                f"visual={visual_t.shape}, proprio={proprio_t.shape}."
            )
            with torch.no_grad():
                enc = world_model.encode({"visual": visual_t, "proprio": proprio_t})
                enc_out = enc["visual"].squeeze(0).squeeze(1).flatten(1, 2)
                proprio_enc = enc["proprio"]

            acts_np = np.stack(step_actions[:n_raw], axis=0)
            act_norm = agent.preprocessor.normalize_actions(
                torch.from_numpy(acts_np).float().unsqueeze(0)
            ).squeeze(0)
            act_model = act_norm.reshape(n_raw // frameskip, frameskip * 2).to(device)

            new_chunk = (enc_out, act_model, proprio_enc[:, 0:1])

            # FIX_SPEC.md B1: slide the 2-chunk ATLAS-arm verification
            # window -- drop the oldest, append the newest. No-op (an
            # empty list append) for every other arm, kept unconditional
            # for simplicity since it costs nothing when unused.
            atlas_verify_buffer.append(new_chunk)
            if len(atlas_verify_buffer) > 2:
                atlas_verify_buffer.pop(0)

            # ── REFINE — last, always (never before scoring: strict
            # prequential order, CLAUDE.md §1.6) ────────────────────────────
            if state.arm in ("adajepa", "adajepa_persist"):
                state.adapter.push(enc_out, act_model, proprio_enc[:, 0:1])
                refine_loss = state.adapter.refine()
            elif state.arm in ("atlas_fixed", "atlas_detect", "atlas"):
                # FIX_SPEC.md B8 (design decision -- see FIXLOG.md B8 for
                # the full rationale): previously this branch was gated
                # `and state.current_idx != 0`, so selecting c0 meant NO
                # refinement happened at all -- unlike arms 2/3 (AdaJEPA),
                # which adapt every single replan regardless of any
                # "selection" concept. That made the 3->4 ladder rung
                # differ by TWO mechanisms at once (adapts vs. doesn't,
                # AND library-routing vs. not), confounding what the rung
                # is supposed to isolate. Now: when index 0 is selected,
                # refine a persistent CLONE of c0 (never c0 itself -- see
                # ArmState.c0_adapted_chart's docstring for why) instead of
                # skipping, so "adapts every replan" is held constant
                # across arms 2-6 and the rung differs by exactly
                # "routes/expands" as intended.
                if state.current_idx == 0:
                    if state.c0_adapted_chart is None:
                        state.c0_adapted_chart = state.library[0].clone()  # identity init (non-negotiable #4)
                    refine_target = state.c0_adapted_chart
                    opt_key = "c0_clone"
                    refine_param_names = refine_target._param_names
                else:
                    refine_target = state.library[state.current_idx]
                    opt_key = state.current_idx
                    refine_param_names = refine_target._param_names
                opt = state.optimizers.get(opt_key)
                if opt is None:
                    params = [p for n, p in predictor.named_parameters()
                              if n in refine_param_names]
                    opt = torch.optim.Adam(params, lr=state.cfg.lr)
                    state.optimizers[opt_key] = opt
                # FIX_SPEC.md B9: push into this chart's own 5-chunk
                # sliding buffer (same BUFFER_SIZE as
                # atlas.adajepa.AdaJEPA), then refine over the WHOLE
                # buffer via atlas_refine_buffered (not atlas_refine's
                # single-chunk path), so arms 4/5/6 see the same amount of
                # recent data per refinement step as arms 2/3.
                buf = state.refine_buffers.setdefault(opt_key, deque(maxlen=5))
                buf.append((enc_out, act_model, proprio_enc[:, 0:1]))
                refine_loss = atlas_refine_buffered(
                    refine_target, world_model, list(buf),
                    lr=state.cfg.lr, optimizer=opt,
                )
            # arms frozen / oracle_id: no refinement.

            prev_chunk = new_chunk

        if success:
            break

    record = {
        "arm": state.arm,
        "success": success,
        "segment_idx": spec.segment_idx,
        "global_episode_idx": spec.global_episode_idx,
        "probe_outcome": probe_outcome,
        "library_size": len(state.library) if state.library is not None else 1,
        # FIX_SPEC.md B14: Library has no evict() -- add() raises when
        # full, and both production callers (loop.py's 'atlas'/
        # 'detect_only' branches) pre-check is_full() before ever calling
        # add(), so growth simply HALTS at k_max and nothing already
        # committed is ever retired/replaced. There is no code fix for
        # "no eviction policy" beyond documenting it (out of scope for
        # this pass -- a paper-level statement, per FIX_SPEC.md); this
        # field makes cap-hits VISIBLE in the per-episode record instead
        # of only inferable by comparing library_size against k_max
        # externally.
        "library_full": bool(state.library.is_full()) if state.library is not None else False,
        "charts_committed_cumulative": state.charts_committed_cumulative,
        "probes_rejected_cumulative": state.probes_rejected_cumulative,
        "seed_run": seed_run,
        "episode_idx": spec.episode_idx,
        "regime": spec.regime,
        "regime_label": regime_label,
        "seed": spec.seed,
        "selected_trace": selected_trace,
        "umf_trace": umf_trace,
        "strikes": state.expander._strikes if state.expander is not None else 0,
        "elapsed_raw_steps": elapsed,
        "n_replans": len(selected_trace),
        "raw_steps_per_replan": raw_steps_per_replan,
        "init_block_pos_diff": init_block_pos_diff,
        "init_block_angle_diff": init_block_angle_diff,
        "init_agent_block_dist": init_agent_block_dist,
        "total_contacts": total_contacts,
        "block_pos_diff": final_check["block_pos_diff"],
        "block_angle_diff": final_check["block_angle_diff"],
        "refine_loss": refine_loss,
        "wall_time": time.time() - t_start,
    }
    log_episode(out_dir, record)
    return record
