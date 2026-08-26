"""
scripts/run_e0.py — E0: Adapter capacity experiment.

Fine-tunes {ln_act, lora4, full} × {R1, R2} = 6 charts offline.
Evaluates UMF reduction and planning success in-regime.

Usage:
    python scripts/run_e0.py
    python scripts/run_e0.py --kinds ln_act lora4 --regimes R1 --steps 500  # quick test

Output:
    atlas_out/e0/chart_{kind}_{regime}.pt
    atlas_out/e0/loss_{kind}_{regime}.json
    atlas_out/e0/results.json   (UMF and success per kind × regime)
    atlas_out/e0/results.md     (T5 supplementary table)
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import pickle
import time
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm
import atlas
from atlas.chart import Chart, ChartKind
from atlas.harness import run_e0_finetune, log_episode
from atlas.regimes import REGIME_CONFIGS, set_regime_config
from atlas.score import compute_motion_gate

DataSource = Literal["scripted", "dataset", "hybrid", "closed_loop"]


@functools.lru_cache(maxsize=None)
def _load_pusht_demo_dataset(split: str = "train"):
    """
    Loads (and caches, per-process) the real Push-T demonstration dataset used
    by T9's replay path. Cached because load_regime_trajectories() is called
    multiple times per regime (train + eval, and again by scripts/run_e1.py
    for its motion-gate sample) -- without caching, each call would re-read
    ~260MB (states.pth + rel_actions.pth) from disk.

    Only states.pth/rel_actions.pth/seq_lengths.pkl are loaded -- NOT
    tokens.pth (4.8GB; precomputed encodings we don't want, since we need
    live proprio/visual through world_model.encode(), see module docstring)
    or abs_actions.pth (env.step() with relative=True, the PushTEnv default,
    consumes rel_actions -- confirmed against pusht_dset.py:36-52's identical
    choice).

    Returns (states [N,246,5] float64, rel_actions [N,246,2] float64,
    seq_lengths: list[int]). states/actions are NOT dataset-normalized (no
    mean/std applied) -- these are the raw recorded values, matching what
    PushTEnv.reset_to_state / env.step() expect (env.step() itself does
    action*action_scale internally, action_scale=100 by default -- see
    _load_pusht_demo_dataset's caller for the /action_scale conversion,
    empirically confirmed exact under R0: replaying a real episode's actions
    from its own recorded initial state reproduces the recorded states to
    ~1e-5 (float32/64 rounding only), before any PhysicsRegime shift).
    """
    d = atlas.DATA_DIR / "pusht_noise" / split
    states = torch.load(d / "states.pth", weights_only=False).numpy()
    rel_actions = torch.load(d / "rel_actions.pth", weights_only=False).numpy()
    with open(d / "seq_lengths.pkl", "rb") as f:
        seq_lengths = pickle.load(f)
    return states, rel_actions, seq_lengths


def load_regime_trajectories(world_model, preprocessor, regime: str, num_trajs: int = 5, traj_len: int = 50, device: str = "cpu", max_tries: int = 8, seed_offset: int = 0, frameskip: int = 5, source: DataSource = "scripted", data_split: str = "train", agent=None, min_block_pos_diff: float = 40.0, max_agent_block_dist: float | None = None, corruption: str = "none", corruption_severity: float = 0.5) -> list[dict]:
    """
    Collects real trajectories from PushTEnv under the specified regime and
    encodes them through the frozen vision backbone.

    Physics regime is applied via atlas.regimes.PhysicsRegime — R0=default,
    R1=high friction, R2=high restitution. R1/R2 are NOT mass/damping-based:
    see ATLAS_implementation_plan_v2.md §6.1a and REGIME_DESIGN_REVIEW.md —
    Push-T's kinematic pusher makes mass-based shifts physically impossible to
    detect at any scale, so R1/R2 were re-targeted onto shape.friction and
    shape.elasticity, which are not subject to the same cancellation.

    Actions are NOT sampled IID per step. A single random target near the
    block is chosen once per trajectory, and the agent aims at it every step
    (small Gaussian noise added for diversity) — pure per-step IID random
    actions (the previous scheme) were empirically found to produce
    agent-block contact in only ~13-17% of rollouts, since a memoryless random
    walk rarely displaces net position over a short horizon. This mirrors how
    every public Push-T lineage (IBC, Diffusion Policy, DINO-WM) generates
    data — none uses fresh per-step random actions either; see
    ACTION_SAMPLING_REVIEW.md. Each trajectory is retried with a new seed (up
    to max_tries) if it produces zero total contact, as a hard guarantee on
    top of the aimed sampling (which only raises the expected contact rate,
    it doesn't guarantee it for every seed/spawn configuration).

    Frame/action chunking (E0_IMPLEMENTATION_PLAN.md T3): raw env steps are
    still what gets simulated (frameskip has no effect on physics), but the
    RETURNED actions/frames are chunked/subsampled to the MODEL time base --
    `_open_loop_rollout` unrolls one model step per `frameskip` raw steps, so
    training/eval data must match that or every step after the first runs on
    a stale, wrong-time-base target (see E0_DIAGNOSIS_AND_PLAN.md's root-cause
    writeup). `world_model.encode()` (not the old
    preprocessor.transform_obs_visual + encode_obs path) is used so real
    proprio is captured and normalized/embedded the same way the planner does
    it -- this checkpoint's predictor concatenates proprio into the token
    channel width (VideoWM.concat_obs_act(), dim=3) and errors without it, so
    it is not an optional enrichment here.

    source="dataset" (E0_IMPLEMENTATION_PLAN.md T9) replaces the scripted
    aimed-walk actions above with REAL recorded action sequences from
    data/pusht_noise/{data_split}/ (18,685 real Push-T demonstrations, the
    same distribution dino_wm_pusht's checkpoint was trained on): a real
    episode + offset is chosen (per-trajectory, per-attempt, seeded exactly
    like the scripted path's target/noise draws), the env is reset to that
    episode's REAL recorded state via reset_to_state (not a fresh random
    reset), and the episode's REAL recorded raw actions are replayed step by
    step. The recording was made under R0 physics; replaying those same
    actions under PhysicsRegime's R1/R2 genuinely diverges from the original
    trajectory (confirmed: replaying under R0 itself reproduces the recorded
    states to ~1e-5, i.e. float rounding only -- so any larger divergence
    under R1/R2 is the regime shift, not a bug in the replay). The
    accept/retry-on-zero-contact structure (max_tries) is unchanged between
    both sources.

    Args:
        world_model: EncPredWM instance (the object torch.hub.load returns —
                     NOT .model).
        source:      'scripted' (default, unchanged) or 'dataset' (T9 replay).
        data_split:  Which data/pusht_noise/{split}/ to draw real episodes
                     from when source='dataset'. Unused for 'scripted'.

    Returns a list of dicts:
        {'encoder_output': [T_model+1, N, D],
         'actions':        [T_model, 10]  (model-chunk, normalized),
         'proprio':        [T_model+1, P_tok, D_p]  (encoded, full sequence),
         'seed':           int,   # accepted seed (both sources)
         'episode_idx':    int | None,  # source='dataset' only: which real episode
         'offset':         int | None}  # source='dataset' only: start offset within it
    """
    import sys
    import numpy as np
    from pathlib import Path

    # Ensure hub code is in sys.path
    hub_path = str(Path(__file__).parent.parent / "hub" / "hub" / "facebookresearch_jepa-wms_main")
    if hub_path not in sys.path:
        sys.path.insert(0, hub_path)

    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
    from atlas.regimes import PhysicsRegime

    if source == "closed_loop":
        # Reused rather than reimplemented: run_e0_planning.py already owns the
        # exact (init, goal) sampler, env-reset path and obs-TensorDict layout
        # the planner is validated against. Importing the sibling script keeps
        # collection and evaluation on literally the same code (a divergence
        # here would train on one task distribution and evaluate on another).
        from einops import rearrange
        from run_e0_planning import (DEFAULT_MAX_AGENT_BLOCK_DIST, make_obs_td,
                                      prepare_with_visual, sample_dataset_init_goal)
        if agent is None:
            raise ValueError("source='closed_loop' requires a GC_Agent (pass agent=...)")
        if max_agent_block_dist is None:
            max_agent_block_dist = DEFAULT_MAX_AGENT_BLOCK_DIST

    if traj_len % frameskip != 0:
        raise ValueError(
            f"traj_len={traj_len} must be divisible by frameskip={frameskip} so "
            f"frame subsampling and action chunking land exactly on model steps."
        )

    # Disjoint seed ranges per regime (unchanged spirit from the original code,
    # extended to cover R0 too since PhysicsRegime now supports all three).
    # seed_offset additionally separates independent calls for the same regime
    # (e.g. train vs. eval trajectories) so they never draw the same seeds --
    # without it, two calls both starting traj_idx at 0 would silently overlap.
    seed_base = {"R0": 2000, "R1": 0, "R2": 1000}.get(regime, 0) + seed_offset

    if source in ("dataset", "hybrid", "closed_loop"):
        # hybrid (P2b) needs demo_states for real init sampling but not
        # demo_rel_actions (actions are generated live, not replayed) --
        # loaded anyway since _load_pusht_demo_dataset returns both from the
        # same file and is cached, so there is no extra cost.
        demo_states, demo_rel_actions, demo_seq_lengths = _load_pusht_demo_dataset(data_split)
        # Need traj_len real actions [offset, offset+traj_len) plus the state
        # AFTER the last one (offset+traj_len) as a valid (non-padding) row --
        # so seq_length must cover index offset+traj_len, i.e. >= traj_len+1.
        valid_eps = [i for i, l in enumerate(demo_seq_lengths) if l >= traj_len + 1]
        if not valid_eps:
            raise ValueError(
                f"No episodes in data/pusht_noise/{data_split} have seq_length >= "
                f"traj_len+1={traj_len + 1} (max seq_length in this split is "
                f"{max(demo_seq_lengths)}) -- reduce traj_len."
            )

    trajectories = []
    traj_pbar = tqdm(range(num_trajs), desc=f"collect_{source}_{regime}", unit="traj")
    for traj_idx in traj_pbar:
        for attempt in range(max_tries):
            seed = seed_base + traj_idx * max_tries + attempt
            rs = np.random.RandomState(seed)
            # with_velocity=True: matches scripts/run_e1.py and the shipped
            # eval YAML (env.with_velocity: true) -- the checkpoint's
            # preprocessor.proprio_std is sized for the 4-dim (x,y,vx,vy)
            # proprio this produces, not the 2-dim default. Confirmed
            # empirically (RuntimeError on proprio_std shape mismatch without
            # it) -- see E0_IMPLEMENTATION_PLAN.md T3.
            base_env = PushTEnv(render_size=224, with_velocity=True)
            env = PhysicsRegime(base_env, regime)
            if corruption != "none":
                # E2 only: appearance shifts on top of (or instead of) a physics
                # shift. Wraps OUTSIDE PhysicsRegime so physics is untouched --
                # the corruption only rewrites obs["visual"] on its way out.
                # Seeded per trajectory so paired arms see identical noise.
                from atlas.regimes import VisualCorruption
                env = VisualCorruption(env, corruption, corruption_severity, seed=seed)

            episode_idx: int | None = None
            offset: int | None = None
            if source == "closed_loop":
                # Goal must be prepared BEFORE the init reset: prepare_with_visual
                # resets the env to whatever state it renders, so rendering the
                # goal last would leave the env sitting on the goal.
                init_state7, goal_state = sample_dataset_init_goal(
                    demo_states, demo_seq_lengths, rs, traj_len=traj_len,
                    min_block_pos_diff=min_block_pos_diff,
                    max_agent_block_dist=max_agent_block_dist)
                goal_obs, _ = prepare_with_visual(base_env, env, seed, goal_state)
                agent.set_goal(make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))
                base_env.seed(seed)
                base_env.reset_to_state = init_state7
            elif source in ("dataset", "hybrid"):
                episode_idx = int(valid_eps[rs.randint(len(valid_eps))])
                max_offset = demo_seq_lengths[episode_idx] - traj_len - 1
                offset = int(rs.randint(max_offset + 1)) if max_offset > 0 else 0
                init_state5 = demo_states[episode_idx, offset]  # [ax, ay, Tx, Ty, angle] -- no velocity
                # Pad to with_velocity=True's 7-dim reset state -- PushTEnv's
                # own random-reset branch hardcodes agent velocity to 0
                # regardless of the sampled state, so zero-padding here
                # matches the existing convention exactly (also done
                # identically in run_e0_planning.py::sample_dataset_init_goal).
                init_state7 = np.concatenate([init_state5, [0.0, 0.0]])
                # reset_to_state/seed are set on the INNER env, not the
                # PhysicsRegime wrapper -- gym.Wrapper proxies attribute READS
                # but not WRITES (see atlas/harness.py::_prepare_env's
                # docstring, and run_e0_planning.py::prepare_with_visual,
                # which this mirrors).
                base_env.seed(seed)
                base_env.reset_to_state = init_state7
            else:
                env.seed(seed)

            obs, state = env.reset()
            imgs = [obs["visual"]]  # RGB array [224, 224, 3]
            proprios = [obs["proprio"]]
            raw_actions = []
            total_contacts = 0

            if source == "dataset":
                action_scale = env.action_scale  # read-proxied through PhysicsRegime; env.action_scale==100 default
                for t in range(traj_len):
                    # Real recorded action, converted back to the raw
                    # [-1,1]-ish input env.step() expects -- env.step()
                    # itself does action*action_scale internally (relative=True
                    # default: += agent.position). Confirmed empirically:
                    # replaying an episode's own actions from its own recorded
                    # initial state under R0 reproduces the recorded states to
                    # ~1e-5 (see module docstring).
                    act = demo_rel_actions[episode_idx, offset + t] / action_scale
                    obs, reward, done, info = env.step(act)
                    total_contacts += info["n_contacts"]
                    imgs.append(obs["visual"])
                    proprios.append(obs["proprio"])
                    raw_actions.append(act)
            elif source == "closed_loop":
                # ON-POLICY data (the P4 follow-up): actions come from the CEM
                # planner replanning against the LIVE regime-shifted state, so
                # the trajectory contains the model's own overshoot AND the
                # correction it then attempts. Replay ('dataset') and the
                # scripted-reactive 'hybrid' collector both lack the latter --
                # the diagnosed reason every P4 chart failed (E0_RESULTS.md).
                #
                # The agent MUST be built with num_act_stepped=1: at the eval
                # config's nas=6 a single plan covers the whole trajectory
                # open-loop, which would reproduce exactly the non-reactive data
                # this collector exists to replace.
                n_chunks = traj_len // frameskip
                for chunk_idx in range(n_chunks):
                    # Each chunk is a full CEM search (collect_num_samples x
                    # collect_iterations candidates) -- by far the slowest step
                    # in this branch, and otherwise invisible until the whole
                    # trajectory finishes. Surface it on the outer traj_pbar
                    # rather than a nested bar, since num_trajs/n_chunks is
                    # normally small and a second bar would just add noise.
                    traj_pbar.set_postfix(chunk=f"{chunk_idx + 1}/{n_chunks}",
                                          contacts=total_contacts, attempt=attempt + 1)
                    obs_td = make_obs_td(obs["visual"], obs["proprio"], device)
                    act_chunk = agent.act(obs_td, steps_left=max(n_chunks - chunk_idx, 1))
                    # [1, frameskip*2] -> [frameskip, 2] raw env actions, matching
                    # run_e0_planning.py::run_episode's own conversion exactly.
                    act_chunk = rearrange(act_chunk.cpu(), "t (f d) -> (t f) d", d=2)
                    act_chunk = agent.preprocessor.denormalize_actions(act_chunk).numpy()
                    for act in act_chunk[:frameskip]:
                        obs, reward, done, info = env.step(act)
                        total_contacts += info["n_contacts"]
                        imgs.append(obs["visual"])
                        proprios.append(obs["proprio"])
                        raw_actions.append(act)
            else:
                block_xy = state[2:4]
                # One persistent random target near the block, sampled once per
                # trajectory — NOT resampled every step, which is what keeps the
                # walk "aimed" instead of cancelling out like the old scheme.
                target = block_xy + rs.uniform(-40.0, 40.0, size=(2,))
                agent_xy = obs["proprio"][:2]

                for _ in range(traj_len):
                    # Undo the env's own act*action_scale (+= agent.position when
                    # relative=True, the default) to get a raw action that aims
                    # the agent at `target` this step.
                    direction = (target - agent_xy) / env.action_scale
                    noise = rs.normal(0.0, 0.15, size=(2,))
                    # ACTION_GAIN: without this, "aim hard at the target every
                    # step" produces raw actions with std ~[0.46, 0.43] (often
                    # saturating at the +-1 clip bound while the agent is still
                    # far away) -- but the checkpoint's own preprocessor.action_std
                    # is ~[0.20, 0.20] (real demonstration data takes gentler,
                    # smaller per-step actions), so un-scaled actions land ~2.2x
                    # outside the distribution the model was calibrated on, which
                    # after normalize_actions stretches to std ~2.1-2.3 and values
                    # up to +-5 std. Confirmed via direct audit (code-review.md
                    # Bug #6d). ACTION_GAIN=0.25 was tuned empirically (clipping
                    # is nonlinear, so std doesn't scale linearly with gain) to
                    # bring raw action std to ~[0.215, 0.204], matching the
                    # checkpoint's ~[0.202, 0.200] almost exactly. This alone
                    # drops single-attempt contact rate to ~43%, but the
                    # rejection-sampling retry above (max_tries=8) still reaches
                    # ~99% expected overall success (1-(1-0.43)^8).
                    ACTION_GAIN = 0.25
                    act = np.clip((direction + noise) * ACTION_GAIN, -1.0, 1.0)
                    obs, reward, done, info = env.step(act)
                    agent_xy = obs["proprio"][:2]
                    total_contacts += info["n_contacts"]
                    imgs.append(obs["visual"])
                    proprios.append(obs["proprio"])
                    raw_actions.append(act)

            if total_contacts > 0 or attempt == max_tries - 1:
                break  # accept: real contact happened, or retries exhausted

        imgs_np = np.stack(imgs, axis=0)            # [T_raw+1, 224, 224, 3]
        proprios_np = np.stack(proprios, axis=0)    # [T_raw+1, proprio_dim]
        acts_np = np.stack(raw_actions, axis=0)     # [T_raw, 2]

        # Subsample frames to the model time base: keep every `frameskip`-th
        # raw frame (plus frame 0) -> T_raw/frameskip + 1 frames, matching the
        # chunked actions below one-to-one.
        keep_idx = list(range(0, traj_len + 1, frameskip))
        imgs_sub = imgs_np[keep_idx]                # [T_model+1, 224, 224, 3]
        proprios_sub = proprios_np[keep_idx]        # [T_model+1, proprio_dim]

        # Encode visual+proprio TOGETHER via the wrapper's own encode() -- the
        # same obs-dict layout GC_Agent.act() feeds it (raw uint8 visual, raw
        # proprio; encode() does its own /255 + preprocessor.transform +
        # normalize_proprios).
        visual_t = torch.from_numpy(imgs_sub.copy()).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
        proprio_t = torch.from_numpy(proprios_sub.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            enc = world_model.encode({"visual": visual_t, "proprio": proprio_t})
            enc_out = enc["visual"].squeeze(0).squeeze(1).flatten(1, 2)  # [T_model+1, 256, 384]
            proprio_enc = enc["proprio"].squeeze(0)                     # [T_model+1, P_tok, D_p]

        # Chunk raw actions [T_raw, 2] -> model actions [T_raw/frameskip, 10].
        act_tensor_raw = torch.from_numpy(acts_np).float()
        act_norm = preprocessor.normalize_actions(act_tensor_raw.unsqueeze(0)).squeeze(0)  # [T_raw, 2]
        act_model = act_norm.reshape(traj_len // frameskip, frameskip * 2).to(device)      # [T_model, 10]

        trajectories.append({
            "encoder_output": enc_out,
            "actions": act_model,
            "proprio": proprio_enc,
            "seed": seed,  # the ACCEPTED seed (whichever attempt broke the retry loop) --
                            # written to e0_seed_manifest.json so E1/downstream experiments
                            # can be audited for zero overlap with E0's train/eval seeds.
            "episode_idx": episode_idx,  # source='dataset' only; None for 'scripted'
            "offset": offset,            # source='dataset' only; None for 'scripted'
            "n_contacts": total_contacts,  # informs the contact-rate check in main()
        })

    return trajectories


def evaluate_e0_chart(world_model, chart: Chart, val_trajectories: list[dict],
                       motion_gate: float | None = None) -> tuple[float, float]:
    """
    Evaluates a fine-tuned chart on held-out validation trajectories.

    Args:
        world_model: EncPredWM wrapper (the object torch.hub.load returns —
                     NOT .model). _open_loop_rollout/umf need the wrapper for
                     the canonical unroll(); chart apply/restore and
                     cfgs_loss-based loss reach the inner VideoWM via
                     world_model.model.
        motion_gate: Informative-chunk threshold (gate G6) — see
                     atlas.score.compute_motion_gate. None = skip gate.

    Returns (avg_eval_loss, avg_eval_umf).
    """
    import numpy as np
    from atlas.harness import compute_trajectory_loss
    from atlas.score import _open_loop_rollout, _make_z_ctxt, umf

    losses = []
    umf_scores = []

    with torch.no_grad():
        for traj in val_trajectories:
            enc_out = traj["encoder_output"]
            actions = traj["actions"]
            z_vis = enc_out[0]
            proprio_ctxt = traj["proprio"][0:1].unsqueeze(0)  # [1, 1, P_tok, D_p]
            z_ctxt = _make_z_ctxt(world_model, z_vis, proprio_ctxt)

            # Apply chart specifically for the open-loop rollout, then restore.
            # apply_() is INSIDE the try too (not just the rollout) -- if
            # apply_() itself raises partway through (e.g. a lora4 chart with
            # a corrupted _param_names list), restore_() must still run or the
            # predictor is left with a permanent, never-cleaned-up partial
            # parametrization that corrupts every subsequent chart/fine-tune
            # sharing this predictor object. See code-review.md Bug #6f.
            try:
                chart.apply_(world_model.model.predictor)
                z_preds = _open_loop_rollout(world_model, z_ctxt, actions)
            finally:
                chart.restore_(world_model.model.predictor)

            loss = compute_trajectory_loss(world_model.model, z_preds, enc_out[1:])
            losses.append(loss.item())

            # umf internally handles applying and restoring the chart,
            # expecting the predictor to start in baseline state.
            score = umf(chart, world_model, enc_out, actions, motion_gate=motion_gate,
                        proprio_ctxt=proprio_ctxt)
            if score is not None:
                umf_scores.append(score)

    avg_eval_loss = float(np.mean(losses)) if losses else float("nan")
    avg_eval_umf = float(np.mean(umf_scores)) if umf_scores else float("nan")
    return avg_eval_loss, avg_eval_umf


def main() -> None:
    parser = argparse.ArgumentParser(description="E0: Adapter capacity fine-tune.")
    parser.add_argument("--kinds", nargs="+", default=["ln_act", "lora4", "full"],
                        choices=["ln_act", "lora4", "full"])
    parser.add_argument("--regimes", nargs="+", default=["R1", "R2"])
    parser.add_argument("--regime-config", type=str, default=None,
                         help="JSON string overriding physics params for EVERY regime in "
                              "--regimes (P1 Step 2/P3), e.g. '{\"friction\": 2.0}'. Applied via "
                              "atlas.regimes.set_regime_config before trajectories are loaded, so "
                              "training and (this script's own) eval use the same calibrated "
                              "physics. All real invocations pass a single --regimes value; "
                              "applying one config to multiple regimes at once is not a supported "
                              "use case and is not validated. Logged into e0_seed_manifest.json "
                              "per regime so a chart's training physics stays attributable -- "
                              "run_e0_planning.py's own --regime-config must match this at eval "
                              "time or the comparison is a silent, invalidating mismatch.")
    parser.add_argument("--steps", type=int, default=2000,
                         help="MAXIMUM gradient steps -- early stopping (see --patience) can "
                              "stop sooner once validation loss stops improving.")
    parser.add_argument("--num-train-trajs", type=int, default=20,
                         help="Number of TRAINING trajectories. run_e0_finetune() loops over "
                              "every trajectory on every step, so compute (and, since all "
                              "trajectories' graphs are held simultaneously before one "
                              "accumulated backward, GPU memory) scales ~linearly with this. "
                              "Bumped from the original 3 now that --data-source=dataset (T9) "
                              "gives real, diverse trajectories instead of one scripted policy "
                              "-- intended for Modal (24GB), not the 6GB local card.")
    parser.add_argument("--train-traj-len", type=int, default=25,
                         help="Steps per TRAINING trajectory (must be a multiple of frameskip=5). "
                              "run_e0_finetune() backprops through the full open-loop unroll, so "
                              "GPU memory scales ~linearly with this too (measured ~0.27GB/step "
                              "for kind=full on this predictor) -- combined with the "
                              "--num-train-trajs bump above, this is sized for Modal's 24GB L4, "
                              "not local. See code-review.md Bug #6e.")
    parser.add_argument("--num-val-trajs", type=int, default=8,
                         help="Number of held-out validation trajectories, used both for early "
                              "stopping during training and (necessarily reusing the same set --  "
                              "this is not a 3-way train/val/test split) for the final reported "
                              "eval_loss/eval_umf. Bumped from the original hardcoded 2.")
    parser.add_argument("--eval-traj-len", type=int, default=50,
                         help="Steps per EVAL (held-out) trajectory. Runs under torch.no_grad "
                              "so length is cheap here -- needs to be long because 10 was too "
                              "short for properly-calibrated (gentle) actions to accumulate "
                              "enough displacement for UMF to be well-behaved. See "
                              "code-review.md Bug #6d. DINO-WM's own training trajectories "
                              "are ~100 steps.")
    parser.add_argument("--eval-every", type=int, default=25,
                         help="Gradient steps between early-stopping validation checks "
                              "(E0_IMPLEMENTATION_PLAN.md T9).")
    parser.add_argument("--patience", type=int, default=5,
                         help="Stop after this many consecutive validation checks with no "
                              "improvement (E0_IMPLEMENTATION_PLAN.md T9) -- the overfitting fix: "
                              "`full` previously reached train loss 0.0015 over 2000 steps on 30 "
                              "transitions with no validation signal at all.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e0")
    parser.add_argument("--data-source", choices=["scripted", "dataset", "hybrid", "closed_loop"], default="dataset",
                         help="'dataset' (default, T9): replay real Push-T demo action "
                              "sequences from data/pusht_noise/{--data-split}/ under the "
                              "shifted regime -- OPEN-LOOP: actions never react to the "
                              "shifted physics. 'hybrid' (P2b): real dataset init state, but "
                              "actions are the 'scripted' aimed-walk policy driven by the LIVE "
                              "post-shift agent position each step -- CLOSED-LOOP, real/diverse "
                              "start states. 'scripted': the original synthetic aimed-walk "
                              "sampler with a random reset (documented fallback per "
                              "E0_IMPLEMENTATION_PLAN.md T9 -- use if the replay path proves "
                              "too slow). 'closed_loop': ON-POLICY -- real (init, goal) pair "
                              "from run_e0_planning.py's own sampler, actions produced by the "
                              "CEM planner replanning every model chunk against the live "
                              "shifted state. The only source whose trajectories contain the "
                              "model's own overshoot AND its attempted correction; 'hybrid' is "
                              "reactive but its corrections come from a scripted policy, not "
                              "from the model being adapted. Costs one CEM search per chunk -- "
                              "see --collect-num-samples.")
    parser.add_argument("--data-split", type=str, default="train",
                         help="data/pusht_noise/{split}/ to draw real episodes from "
                              "(--data-source=dataset, hybrid or closed_loop only).")
    parser.add_argument("--collect-num-samples", type=int, default=100,
                         help="CEM population for --data-source=closed_loop COLLECTION only "
                              "(eval keeps the substrate's validated 300). Collection needs "
                              "trajectories that react to the shift, not optimal ones, and "
                              "cost is linear here: 300x30 would be ~9x this, putting a "
                              "28-trajectory collection into GPU-hours. Deviation from the "
                              "validated planner config -- record it with any result.")
    parser.add_argument("--collect-iterations", type=int, default=10,
                         help="CEM iterations for closed_loop collection (see "
                              "--collect-num-samples for the cost rationale).")
    parser.add_argument("--collect-max-tries", type=int, default=2,
                         help="Contact-rejection retries per trajectory for closed_loop. Much "
                              "lower than the default 8: each retry is a full planned episode, "
                              "and a goal-directed planner should make contact far more often "
                              "than the scripted sampler it was tuned for.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="atlas-e0", help="WandB project name")
    args = parser.parse_args()

    wandb_group = f"e0_experiment_{int(time.time())}"

    print("Loading dino_wm_pusht from local hub...")
    # Raw ATLAS_HOME env var, not Path(__file__)-relative: on Modal the code
    # (/src) and the volume-mounted hub cache (/atlas_root/hub) live at
    # different paths -- see run_e0_planning.py's HUB_PATH for the same fix.
    import os
    _atlas_home = os.environ.get("ATLAS_HOME", str(atlas.ATLAS_HOME))
    hub_path = str(Path(_atlas_home) / "hub" / "hub" / "facebookresearch_jepa-wms_main")
    model, prep = torch.hub.load(
        hub_path, "dino_wm_pusht",
        source="local",
        force_reload=False, trust_repo=True,
    )
    # `model` is the EncPredWM wrapper (owns ctxt_window/proprio_mode and the
    # canonical unroll() this checkpoint was validated with -- see
    # E0_IMPLEMENTATION_PLAN.md T1/T2). `wm` is the inner VideoWM, needed for
    # predictor state-dict ops, Chart construction, and encoder freezing --
    # wrapper.to(device) moves it too, since it's a registered submodule.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = model.to(device)
    wm = wrapper.model if hasattr(wrapper, "model") else wrapper
    for p in wm.encoder.parameters():
        p.requires_grad_(False)

    # Pristine snapshot of the predictor, taken once before any fine-tuning.
    # run_e0_finetune() mutates wm.predictor in-place for ln_act/full charts and
    # never restores it (Chart.restore_ just re-applies the already-updated chart
    # params, not the original checkpoint weights) — without reloading this
    # snapshot before every fine-tune, later charts would train on top of earlier
    # charts' weights instead of the pretrained baseline. See code-review.md #1.
    pristine_predictor_state = copy.deepcopy(wm.predictor.state_dict())

    if args.regime_config is not None:
        resolved_cfg = json.loads(args.regime_config)
        for regime in args.regimes:
            set_regime_config(regime, resolved_cfg)

    # Built from the PRISTINE predictor, before any chart is applied: the point
    # of on-policy collection is data showing what the FROZEN model gets wrong,
    # which is what the chart is then fit to correct. Collecting under an
    # already-adapted model would be a second DAgger round, not this experiment.
    collector_agent = None
    if args.data_source == "closed_loop":
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from run_e0_planning import build_cfg
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        # num_act_stepped=1 is the whole point: it forces a replan every model
        # chunk, which is what makes the collected trajectory reactive.
        collector_agent = GC_Agent(
            build_cfg(args.collect_num_samples, args.collect_iterations, horizon=6,
                      num_act_stepped=1),
            wrapper, dset=None, preprocessor=prep)
        collector_agent.device = device
        print(f"closed_loop collection: CEM {args.collect_num_samples}x"
              f"{args.collect_iterations}, num_act_stepped=1 (replan every chunk)", flush=True)

    # [Debug print statement] Print setup info
    print(f"\nE0: {len(args.kinds)} kinds × {len(args.regimes)} regimes "
          f"= {len(args.kinds) * len(args.regimes)} fine-tunes", flush=True)
    print(f"Output: {args.out}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    seed_manifest: dict = {}  # regime -> {"train": [...], "eval": [...]} accepted seeds --
                              # written to e0_seed_manifest.json so downstream experiments
                              # (E1) can be audited for zero seed overlap with E0.

    for regime in args.regimes:
        results[regime] = {}
        # [Debug print statement] Print regime start
        print(f"\n── Regime {regime} ─────────────────────────────────────────────", flush=True)
        print(f"  Loading trajectories for regime {regime}...", flush=True)
        # Separate lengths: training backprops through the full unroll (memory
        # scales with length, kept short); eval runs under no_grad (cheap, kept
        # long since UMF needs real accumulated displacement). See --train-traj-len
        # / --eval-traj-len help text and code-review.md Bug #6d/#6e.
        collect_kw = ({"agent": collector_agent, "max_tries": args.collect_max_tries}
                      if args.data_source == "closed_loop" else {})
        train_trajectories = load_regime_trajectories(
            wrapper, prep, regime, num_trajs=args.num_train_trajs, traj_len=args.train_traj_len,
            device=device, seed_offset=0, source=args.data_source, data_split=args.data_split,
            **collect_kw)
        val_trajectories = load_regime_trajectories(
            wrapper, prep, regime, num_trajs=args.num_val_trajs, traj_len=args.eval_traj_len, device=device,
            seed_offset=10_000, source=args.data_source, data_split=args.data_split,
            **collect_kw)
        seed_manifest[regime] = {
            "source": args.data_source,
            "regime_config": dict(REGIME_CONFIGS.get(regime, {})),
            "train": [{"seed": t["seed"], "episode_idx": t["episode_idx"], "offset": t["offset"]}
                      for t in train_trajectories],
            "eval": [{"seed": t["seed"], "episode_idx": t["episode_idx"], "offset": t["offset"]}
                     for t in val_trajectories],
        }
        # [Debug print statement] Print trajectories loaded
        print(f"  [Debug] Loaded {len(train_trajectories)} train & {len(val_trajectories)} eval trajectories for {regime}", flush=True)

        if args.data_source in ("dataset", "hybrid", "closed_loop"):
            # T9 acceptance check: real demo episodes should generally involve
            # agent-block contact (they're expert task completions), but verify
            # rather than assume -- especially post-regime-shift, where the
            # replayed actions no longer perfectly track the (now-different)
            # physics. n_contacts is per-trajectory total contact events across
            # traj_len steps, recorded by load_regime_trajectories() above.
            all_trajs = train_trajectories + val_trajectories
            n_with_contact = sum(1 for t in all_trajs if t["n_contacts"] > 0)
            print(f"  [Debug] Real-demo replay contact rate for {regime}: "
                  f"{n_with_contact}/{len(all_trajs)} trajectories had >=1 contact "
                  f"(n_contacts per traj: {[t['n_contacts'] for t in all_trajs]})", flush=True)

        # Gate G6: informative-chunk threshold, computed once from this
        # regime's own training displacements (10th percentile) -- wired into
        # every umf() call below instead of the motion_gate=None the eval
        # path has always used (E0_IMPLEMENTATION_PLAN.md T5).
        train_displacements = torch.tensor([
            (t["encoder_output"][-1] - t["encoder_output"][0]).norm(p="fro").item()
            for t in train_trajectories
        ])
        motion_gate = compute_motion_gate(train_displacements)
        print(f"  [Debug] motion_gate (10th pct of train displacement) = {motion_gate:.4f}", flush=True)

        for kind in args.kinds:
            # Reset the predictor to its pristine pretrained state before every
            # kind's fine-tune/eval. Without this, later charts train on top of
            # earlier charts' weights instead of the checkpoint baseline — see
            # code-review.md #1.
            wm.predictor.load_state_dict(pristine_predictor_state)

            loss_file = args.out / f"loss_{kind}_{regime}.json"
            chart_file = args.out / f"chart_{kind}_{regime}.pt"

            if loss_file.exists() and chart_file.exists():
                # [Resume support] Skip fine-tuning if results already exist on disk / volume
                print(f"  ⏩ [Resume] {kind}_{regime} already completed. Loading cached result...", flush=True)
                losses = json.loads(loss_file.read_text())
                final_loss = losses[-1] if losses else float("nan")
                try:
                    chart = Chart.load(chart_file, wm.predictor)
                    n_params = chart.n_params()  # real parameter count, not a tensor count (T12 #9)
                    eval_loss, eval_umf = evaluate_e0_chart(wrapper, chart, val_trajectories, motion_gate)
                except Exception as e:
                    # Print the real error instead of silently returning NaN --
                    # a swallowed exception here previously masked a real bug
                    # (Chart.load()'s corrupted _param_names for lora4, see
                    # code-review.md Bug #6f) that also left the shared
                    # predictor in a corrupted state for every subsequent
                    # kind/regime, only surfacing as a confusing crash much
                    # later. Reset the predictor defensively so a failure here
                    # can't cascade the same way even if a new bug appears.
                    print(f"    ⚠️ [Resume] Failed to re-evaluate cached {kind}_{regime}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    wm.predictor.load_state_dict(pristine_predictor_state)
                    n_params = 0
                    eval_loss, eval_umf = float("nan"), float("nan")
                results[regime][kind] = {
                    "train_loss": final_loss,
                    "eval_loss": eval_loss,
                    "eval_umf": eval_umf,
                    "params": n_params,
                    "status": "completed (cached)",
                }
                print(f"    Done {kind}_{regime} (Cached): Train Loss = {final_loss:.6f} | Eval Loss = {eval_loss:.6f} | Eval UMF = {eval_umf:.4f}", flush=True)
                continue

            # Initialize separate WandB run for this fine-tuning session under the shared group
            if args.wandb:
                try:
                    import wandb
                    wandb.init(
                        project=args.wandb_project,
                        group=wandb_group,
                        name=f"{kind}_{regime}",
                        reinit=True,
                        config={
                            "kind": kind,
                            "regime": regime,
                            "steps": args.steps,
                            "lr": args.lr,
                        },
                    )
                    wandb.define_metric("step")
                    wandb.define_metric("loss", step_metric="step")
                    wandb.define_metric(f"loss_{kind}_{regime}", step_metric="step")
                except Exception as e:
                    print(f"⚠️ WandB init failed for {kind}_{regime}: {e}", flush=True)

            # [Debug print statement] Print fine-tuning start
            print(f"  Fine-tuning {kind} on {regime} (up to {args.steps} steps, "
                  f"early stop after {args.patience} checks w/o improvement every "
                  f"{args.eval_every} steps)...", flush=True)
            chart = run_e0_finetune(
                world_model=wrapper,
                trajectories=train_trajectories,
                kind=kind,
                regime=regime,
                n_steps=args.steps,
                lr=args.lr,
                out_dir=args.out,
                val_trajectories=val_trajectories,
                eval_every=args.eval_every,
                patience=args.patience,
            )

            # Held-out Evaluation on Validation Trajectories
            eval_loss, eval_umf = evaluate_e0_chart(wrapper, chart, val_trajectories, motion_gate)

            if args.wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"eval_loss": eval_loss, "eval_umf": eval_umf})
                        wandb.finish()
                except Exception:
                    pass
            
            # Compute evaluation metrics
            losses = json.loads(loss_file.read_text())
            final_loss = losses[-1] if losses else float("nan")
            results[regime][kind] = {
                "train_loss": final_loss,
                "eval_loss": eval_loss,
                "eval_umf": eval_umf,
                "params": chart.n_params(),  # real parameter count, not a tensor count (T12 #9)
                "status": "completed",
            }
            # [Debug print statement] Print fine-tuning & eval completed
            print(f"    Done {kind}_{regime}: Train Loss = {final_loss:.6f} | Eval Loss = {eval_loss:.6f} | Eval UMF = {eval_umf:.4f}", flush=True)

    # Save summary results
    results_json = args.out / "results.json"
    results_json.write_text(json.dumps(results, indent=2))

    # Seed manifest: audit trail proving E1 (or any downstream experiment) uses
    # seeds disjoint from every trajectory that went into fine-tuning/evaluating
    # these charts. E1's own seeds (atlas.streams.paired_seed, SHA256-derived)
    # live in a completely different space from these small deterministic
    # integers, but this manifest makes that an auditable fact, not a claim.
    seed_manifest_json = args.out / "e0_seed_manifest.json"
    seed_manifest_json.write_text(json.dumps(seed_manifest, indent=2))
    print(f"  - Seed manifest : {seed_manifest_json}")

    # Generate Markdown Table T5
    md_lines = [
        "# Table T5: E0 Adapter Capacity Benchmarks",
        "",
        "| Regime | Adapter Kind | Target Params | Train Loss | Eval Loss | Eval UMF | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for regime, kinds in results.items():
        for kind, metrics in kinds.items():
            md_lines.append(
                f"| {regime} | `{kind}` | {metrics['params']:,} | {metrics['train_loss']:.6f} | {metrics['eval_loss']:.6f} | {metrics['eval_umf']:.4f} | {metrics['status']} |"
            )
    
    results_md = args.out / "results.md"
    results_md.write_text("\n".join(md_lines))

    # Generate local loss curves plot from saved JSON files
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4), dpi=300)
        colors = {
            "ln_act": "#1f77b4",
            "lora4": "#ff7f0e",
            "full": "#2ca02c",
        }
        linestyles = {
            "R1": "-",
            "R2": "--",
        }

        for regime in args.regimes:
            for kind in args.kinds:
                loss_file = args.out / f"loss_{kind}_{regime}.json"
                if loss_file.exists():
                    losses = json.loads(loss_file.read_text())
                    steps = list(range(1, len(losses) + 1))
                    plt.plot(
                        steps,
                        losses,
                        label=f"{kind} ({regime})",
                        color=colors.get(kind, None),
                        linestyle=linestyles.get(regime, "-"),
                        alpha=0.85,
                        linewidth=1.5,
                    )

        plt.xlabel("Gradient Step (1..2000)")
        plt.ylabel("Prediction Loss")
        plt.title("E0 Adapter Capacity: Fine-Tuning Loss Curves")
        plt.legend(loc="upper right", frameon=True)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()

        plot_png = args.out / "e0_loss_curves.png"
        plot_pdf = args.out / "e0_loss_curves.pdf"
        plt.savefig(plot_png)
        plt.savefig(plot_pdf)
        plt.close()
        print(f"  - Loss Plot PNG: {plot_png}")
        print(f"  - Loss Plot PDF: {plot_pdf}")
    except Exception as e:
        print(f"⚠️ Loss plot generation skipped: {e}")

    print(f"\n✅ E0 Experiment complete! Results saved to {args.out}")
    print(f"  - Summary JSON : {results_json}")
    print(f"  - Table T5 MD  : {results_md}")


if __name__ == "__main__":
    main()

