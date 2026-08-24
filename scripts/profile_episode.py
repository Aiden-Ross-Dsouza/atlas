"""
scripts/profile_episode.py — Day-2 compute budget calibration.

Runs N real Push-T planning episodes under the T6-restored substrate CEM
config (E0_IMPLEMENTATION_PLAN.md T6: num_samples=300, iterations=30,
horizon=6, num_act_stepped=6) and prints:
  - Seconds per episode
  - Peak VRAM (GB)
  - Predictor forwards per replan

Reuses scripts/run_e0_planning.py's already-working pattern (bypass
eval.py/PlanEvaluator, reuse GC_Agent/CEMPlanner directly, local hub load)
rather than re-deriving it -- this was the origin of the previously-unverified
"~30s/episode" figure the plan's whole compute budget (§12) rests on; this
script had never actually run before (loaded from the REMOTE hub, and did
model.encoder/model.predictor directly, which the EncPredWM wrapper torch.hub
returns does not expose -- would AttributeError before reaching the
NotImplementedError below it).

Usage:
    python scripts/profile_episode.py --episodes 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import atlas
from scripts.run_e0_planning import (
    HUB_PATH,
    build_cfg,
    load_dataset_states,
    prepare_with_visual,
    make_obs_td,
    block_success,
)

from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv  # noqa: E402
from evals.simu_env_planning.envs.pusht_gym_wrap import PushTWrapper  # noqa: E402
from evals.simu_env_planning.planning.gc_agent import GC_Agent  # noqa: E402
from atlas.regimes import PhysicsRegime  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a Push-T planning episode.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-act-stepped", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()

    print("Loading dino_wm_pusht from local hub...")
    model, prep = torch.hub.load(
        HUB_PATH, "dino_wm_pusht", source="local", force_reload=False, trust_repo=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA device found. Profiling on CPU will not reflect GPU timing.")
    wm = model.model if hasattr(model, "model") else model
    wm = wm.to(device)
    model = model.to(device)
    for p in wm.encoder.parameters():
        p.requires_grad_(False)
    for p in wm.predictor.parameters():
        p.requires_grad_(False)

    # T7 throughput fixes -- see run_e0_planning.py's identical block.
    for m in wm.predictor.modules():
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = True
    torch.set_float32_matmul_precision("high")

    # Count predictor forwards via a forward hook (not a manual call-site
    # count) so it reflects what actually happens inside CEMPlanner.plan(),
    # including any internal decode_each_iteration-gated calls.
    forward_count = 0

    def _count_forward(module, inputs, output):
        nonlocal forward_count
        forward_count += 1

    handle = wm.predictor.register_forward_hook(_count_forward)

    cfg = build_cfg(args.num_samples, args.iterations, args.horizon, args.num_act_stepped)
    agent = GC_Agent(cfg, model, dset=None, preprocessor=prep)
    agent.device = device

    base_env = PushTEnv(render_size=224, with_velocity=True)
    regime = PhysicsRegime(base_env, "R0")
    states, seq_lengths = load_dataset_states()

    print(f"\nProfiling {args.episodes} frozen episodes on {device} "
          f"(num_samples={args.num_samples}, iterations={args.iterations}, "
          f"horizon={args.horizon}, num_act_stepped={args.num_act_stepped})...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ep_times = []
    ep_replans = []
    ep_forward_counts = []
    for ep in range(args.episodes):
        rs = np.random.RandomState(ep)
        from scripts.run_e0_planning import sample_dataset_init_goal
        init_state, goal_state = sample_dataset_init_goal(states, seq_lengths, rs)

        goal_obs, _ = prepare_with_visual(base_env, regime, ep, goal_state)
        agent.set_goal(make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))
        obs, _ = prepare_with_visual(base_env, regime, ep, init_state)

        n_replans_target = max(args.max_steps // args.num_act_stepped, 1)
        elapsed = 0
        replans = 0
        forward_count_before = forward_count
        t_start = time.time()
        for replan_idx in range(n_replans_target):
            if elapsed >= args.max_steps:
                break
            obs_td = make_obs_td(obs["visual"], obs["proprio"], device)
            steps_left_model = (n_replans_target - replan_idx) * args.num_act_stepped
            action = agent.act(obs_td, steps_left=max(steps_left_model, 1))
            replans += 1

            from einops import rearrange
            raw_actions = rearrange(action.cpu(), "t (f d) -> (t f) d", d=2)
            raw_actions = agent.preprocessor.denormalize_actions(raw_actions).numpy()
            for a in raw_actions:
                if elapsed >= args.max_steps:
                    break
                obs, reward, done, info = base_env.step(a)
                elapsed += 1
                if block_success(goal_state, info["state"])["success"]:
                    break
        ep_time = time.time() - t_start
        ep_times.append(ep_time)
        ep_replans.append(replans)
        ep_forward_counts.append(forward_count - forward_count_before)
        print(f"  episode {ep}: {ep_time:.1f}s, {replans} replan(s), "
              f"{forward_count - forward_count_before} predictor forwards")

    handle.remove()

    mean_time = sum(ep_times) / len(ep_times)
    mean_replans = sum(ep_replans) / len(ep_replans)
    mean_forwards = sum(ep_forward_counts) / len(ep_forward_counts)
    forwards_per_replan = mean_forwards / mean_replans if mean_replans else float("nan")
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    print(f"\nMean sec/episode: {mean_time:.2f}")
    print(f"Mean replans/episode: {mean_replans:.2f}")
    print(f"Mean predictor forwards/episode: {mean_forwards:.1f}")
    print(f"Predictor forwards/replan: {forwards_per_replan:.1f}")
    if peak_mem_gb is not None:
        print(f"Peak GPU memory: {peak_mem_gb:.2f} GB")

    print("\nRecomputed E4 budget from this measurement:")
    print("GPU-h = (sec/ep x episodes x segments x seeds x arms) / 3600")
    for label, episodes, segments, seeds, arms in [
        ("E4 (20ep x 6seg x 3seeds x 7arms)", 20, 6, 3, 7),
    ]:
        gpu_h = (mean_time * episodes * segments * seeds * arms) / 3600
        print(f"  {label}: {gpu_h:.1f} GPU-h")


if __name__ == "__main__":
    main()
