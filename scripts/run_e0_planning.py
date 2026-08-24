"""
scripts/run_e0_planning.py — E0's missing "planning success" half (Table T5's
Success column). Runs N CEM-driven episodes for one fixed, pre-trained E0
chart and reports success rate.

Does NOT use jepa-wms's own eval.py/PlanEvaluator — that's a heavyweight
distributed/dataset-dependent framework with confirmed Push-T bugs (see
code-review.md Bug #7: PlanEvaluator.eval() reads a Metaworld-only attribute
that doesn't exist on PushTEnv; TensorWrapper.step() assumes an info["success"]
key Push-T never sets). Reuses only GC_Agent/CEMPlanner (generic, no
task-specific code) plus PushTWrapper's two Push-T utility methods
(sample_random_init_goal_states, eval_state), with its own minimal episode loop.

STATUS: correctness-tested at reduced CEM settings only (small num_samples/
iterations). NOT run at the published spec (num_samples=300, iterations=30) --
that OOMs immediately on a 6GB GPU. See code-review.md Bug #7 for the memory
numbers and the Modal recommendation.

Usage:
    python scripts/run_e0_planning.py --kind ln_act --regime R1 --episodes 1 \\
        --num-samples 16 --iterations 5 --horizon 3   # local smoke test
    python scripts/run_e0_planning.py --kind ln_act --regime R1 --episodes 20  # full spec, needs Modal
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import atlas
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

# One model step's action_dim=10 is FRAMESKIP=5 raw 2-dim actions concatenated
# (data.custom.frameskip=5 in the shipped config). Must unchunk before
# denormalize_actions/env.step(), which expect the raw 2-dim space.
FRAMESKIP = 5

# frameskip * goal_H + 1 (plan_evaluator.py's own goal_source=dset formula) --
# goal_H=6 matches the shipped config's planner.horizon/num_act_stepped.
GOAL_TRAJ_LEN = FRAMESKIP * 6 + 1

# Raw ATLAS_HOME env var, NOT atlas.ATLAS_HOME: that's .resolve()'d, which on
# Modal follows the volume mount (/atlas_root) down to its internal storage
# path (/__modal/volumes/vo-xxx/...) -- a path that isn't itself mounted
# anywhere, so torch.hub's file access on the resolved path 404s even though
# the file is really there under /atlas_root.
_atlas_home = os.environ.get("ATLAS_HOME", str(atlas.ATLAS_HOME))
HUB_PATH = str(Path(_atlas_home) / "hub" / "hub" / "facebookresearch_jepa-wms_main")
if HUB_PATH not in sys.path:
    sys.path.insert(0, HUB_PATH)

from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv  # noqa: E402
from evals.simu_env_planning.envs.pusht_gym_wrap import PushTWrapper  # noqa: E402
from evals.simu_env_planning.planning.gc_agent import GC_Agent  # noqa: E402

from atlas.chart import Chart  # noqa: E402
from atlas.regimes import PhysicsRegime  # noqa: E402


def build_cfg(num_samples: int, iterations: int, horizon: int, num_act_stepped: int) -> OmegaConf:
    return OmegaConf.create({
        "local_seed": 0,
        "task_specification": {"obs": "rgb_state"},
        "planner": {
            "planner_name": "cem",
            "iterations": iterations,
            "num_samples": num_samples,
            "num_elites": min(10, num_samples),
            "horizon": horizon,
            "var_scale": 1.0,
            "num_act_stepped": num_act_stepped,
            "repeat_actskip": False,
            "decode_each_iteration": False,
            "distribute_planner": False,
            "planning_objective": {"objective_type": "L2", "sum_all_diffs": False, "alpha": 0.1},
        },
    })


def prepare_with_visual(base_env: PushTEnv, seed: int, state: np.ndarray):
    # PushTWrapper.prepare() discards obs["visual"] (expects a PixelWrapper to
    # re-render it); we skip PixelWrapper, so reset the raw env directly.
    base_env.seed(seed)
    base_env.reset_to_state = state
    return base_env.reset()


def make_obs_td(visual_hw3_uint8: np.ndarray, proprio_vec: np.ndarray, device: str) -> TensorDict:
    # No leading batch dim: GC_Agent.set_goal()/.act() add it themselves.
    visual = torch.from_numpy(visual_hw3_uint8.copy()).permute(2, 0, 1).float().unsqueeze(0)
    proprio = torch.from_numpy(np.asarray(proprio_vec, dtype=np.float32)).unsqueeze(0)
    return TensorDict({"visual": visual, "proprio": proprio}, batch_size=[]).to(device)


def load_dataset_states(split: str = "train") -> tuple[np.ndarray, list[int]]:
    # Raw recorded [agent_x, agent_y, T_x, T_y, angle] trajectories -- only
    # states.pth is needed (not visual/action/token files) to sample reachable
    # (init, goal) pairs; PushTEnv regenerates observations from state directly.
    d = atlas.DATA_DIR / "pusht_noise" / split
    states = torch.load(d / "states.pth", weights_only=False).numpy()
    with open(d / "seq_lengths.pkl", "rb") as f:
        seq_lengths = pickle.load(f)
    return states, seq_lengths


def sample_dataset_init_goal(states: np.ndarray, seq_lengths: list[int], rs: np.random.RandomState,
                              traj_len: int = GOAL_TRAJ_LEN) -> tuple[np.ndarray, np.ndarray]:
    valid_eps = [i for i, l in enumerate(seq_lengths) if l >= traj_len]
    ep_idx = valid_eps[rs.randint(len(valid_eps))]
    max_offset = seq_lengths[ep_idx] - traj_len
    offset = rs.randint(max_offset + 1) if max_offset > 0 else 0
    init_state = states[ep_idx, offset]
    goal_state = states[ep_idx, offset + traj_len - 1]
    # dataset states are 5-dim (no velocity); pad to match with_velocity=True's
    # 7-dim format -- generate_state() itself hardcodes agent velocity to 0
    # regardless, so zero-padding matches the existing convention exactly.
    init_state = np.concatenate([init_state, [0.0, 0.0]])
    goal_state = np.concatenate([goal_state, [0.0, 0.0]])
    return init_state, goal_state


def block_success(goal_state: np.ndarray, cur_state: np.ndarray) -> dict:
    # wrapper.eval_state() compares state[:4] = [agent_x, agent_y, T_x, T_y] together,
    # which is only meaningful when goal/init come from a correlated real trajectory
    # (goal_source=dset upstream). Our goals are independently random (no dataset
    # dependency by design), so the agent-position term is pure noise -- success
    # would require the pusher to land within 20px of an unrelated random point.
    # Push-T's actual objective (and IBC/Diffusion Policy's own metric) is block
    # position/orientation only -- state indices [T_x, T_y, angle] = [2, 3, 4].
    pos_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
    # PushTWrapper.generate_state() draws the goal angle as randn()*2pi - pi --
    # an unbounded Gaussian, not a wrapped angle -- so a raw difference can span
    # several full rotations. np.minimum(d, 2pi-d) (the original eval_state()
    # formula) only wraps correctly within one rotation; for larger raw
    # differences it produces garbage (even negative) results. Wrap properly
    # via the standard atan2-free formula: fold into (-pi, pi] first.
    raw_diff = goal_state[4] - cur_state[4]
    angle_diff = np.abs((raw_diff + np.pi) % (2 * np.pi) - np.pi)
    success = pos_diff < 20 and angle_diff < np.pi / 9
    return {"success": success, "block_pos_diff": float(pos_diff), "block_angle_diff": float(angle_diff)}


def run_episode(agent: GC_Agent, base_env: PushTEnv, wrapper: PushTWrapper, regime: PhysicsRegime,
                 seed: int, max_steps: int, states: np.ndarray, seq_lengths: list[int],
                 log_planner_diagnostics: bool = False) -> dict:
    device = agent.device

    rs = np.random.RandomState(seed)
    init_state, goal_state = sample_dataset_init_goal(states, seq_lengths, rs)

    goal_obs, _ = prepare_with_visual(base_env, seed, goal_state)
    regime._apply_physics()  # must run after reset() rebuilds the pymunk space
    agent.set_goal(make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))

    obs, _ = prepare_with_visual(base_env, seed, init_state)
    regime._apply_physics()

    elapsed = 0
    success = False
    replans = 0
    final_check = {"block_pos_diff": None, "block_angle_diff": None}
    planner_diagnostics = []  # per-replan CEM elite-cost convergence, if requested
    t_start = time.time()
    while elapsed < max_steps and not success:
        obs_td = make_obs_td(obs["visual"], obs["proprio"], device)
        action = agent.act(obs_td, steps_left=max(max_steps - elapsed, 1))
        replans += 1

        if log_planner_diagnostics:
            # Free -- GC_Agent.plan() already computes and stores these; no extra
            # planner compute. Per-CEM-iteration mean/std of the elite candidates'
            # cost, e.g. does the search converge to a low-cost action, or stay
            # high/noisy throughout -- a cheap signal on whether an adapter is
            # confusing CEM's cost ranking, before capturing full per-candidate costs.
            planner_diagnostics.append({
                "elite_losses_mean": agent._prev_elite_losses_mean.squeeze(-1).tolist(),
                "elite_losses_std": agent._prev_elite_losses_std.squeeze(-1).tolist(),
            })

        action = rearrange(action.cpu(), "t (f d) -> (t f) d", d=2)
        action = agent.preprocessor.denormalize_actions(action).numpy()

        for a in action:
            if elapsed >= max_steps:
                break
            obs, reward, done, info = base_env.step(a)
            elapsed += 1
            final_check = block_success(goal_state, info["state"])
            if final_check["success"]:
                success = True
                break
    result = {"success": success, "steps": elapsed, "replans": replans, "wall_time": time.time() - t_start,
              **{k: v for k, v in final_check.items() if k != "success"}}
    if log_planner_diagnostics:
        result["planner_diagnostics"] = planner_diagnostics
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="E0: planning success (CEM-driven episodes).")
    parser.add_argument("--kind", required=True, choices=["baseline", "ln_act", "lora4", "full"],
                         help="'baseline' = frozen pretrained predictor, no chart applied.")
    parser.add_argument("--regime", required=True, choices=["R0", "R1", "R2"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300,
                         help="Published spec is 300; OOMs on a 6GB GPU. Use e.g. 16 for a local test.")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-act-stepped", type=int, default=6)
    parser.add_argument("--charts-dir", type=Path, default=atlas.OUT_DIR / "e0")
    parser.add_argument("--out-dir", type=Path, default=atlas.OUT_DIR / "e0_planning",
                         help="Where to write per-episode JSONL + summary JSON.")
    parser.add_argument("--log-planner-diagnostics", action="store_true",
                         help="Log CEM's per-iteration elite-cost mean/std (free -- already computed).")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dino_wm_pusht from local hub...")
    model, prep = torch.hub.load(
        HUB_PATH, "dino_wm_pusht", source="local", force_reload=False, trust_repo=True,
    )
    wm = model.model if hasattr(model, "model") else model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = wm.to(device)
    model = model.to(device)
    for p in wm.encoder.parameters():
        p.requires_grad_(False)

    if args.kind == "baseline":
        print("kind=baseline -- no chart applied, using frozen pretrained predictor as-is.")
    else:
        chart_path = args.charts_dir / f"chart_{args.kind}_{args.regime}.pt"
        print(f"Applying chart: {chart_path}")
        chart = Chart.load(str(chart_path), wm.predictor)
        chart.apply_(wm.predictor)

    cfg = build_cfg(args.num_samples, args.iterations, args.horizon, args.num_act_stepped)
    agent = GC_Agent(cfg, model, dset=None, preprocessor=prep)
    agent.device = device

    base_env = PushTEnv(render_size=224, with_velocity=True)
    wrapper = PushTWrapper(base_env)
    regime = PhysicsRegime(wrapper, args.regime)
    states, seq_lengths = load_dataset_states()

    print(f"Running {args.episodes} episode(s): kind={args.kind} regime={args.regime} "
          f"num_samples={args.num_samples} iterations={args.iterations}")
    if args.num_samples < 300:
        print("NOTE: reduced num_samples -- not the published spec (300). See module docstring.")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    jsonl_path = args.out_dir / f"{args.kind}_{args.regime}.jsonl"
    results = []
    pbar = tqdm(range(args.episodes), desc=f"{args.kind}_{args.regime}", unit="ep")
    with open(jsonl_path, "w") as f:
        for ep in pbar:
            result = run_episode(agent, base_env, wrapper, regime, seed=ep, max_steps=args.max_steps,
                                  states=states, seq_lengths=seq_lengths,
                                  log_planner_diagnostics=args.log_planner_diagnostics)
            results.append(result)
            f.write(json.dumps({"episode": ep, "kind": args.kind, "regime": args.regime, **result}) + "\n")
            f.flush()
            pbar.set_postfix(success=result["success"], steps=result["steps"])

    success_rate = sum(r["success"] for r in results) / len(results)
    mean_time = sum(r["wall_time"] for r in results) / len(results)
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    print(f"\nSuccess rate: {success_rate:.2f} ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"Mean wall time per episode: {mean_time:.1f}s")
    if peak_mem_gb is not None:
        print(f"Peak GPU memory: {peak_mem_gb:.2f} GB")

    summary_path = args.out_dir / f"{args.kind}_{args.regime}_summary.json"
    summary_path.write_text(json.dumps({
        "kind": args.kind, "regime": args.regime, "episodes": args.episodes,
        "num_samples": args.num_samples, "iterations": args.iterations, "horizon": args.horizon,
        "success_rate": success_rate, "mean_wall_time_s": mean_time, "peak_gpu_memory_gb": peak_mem_gb,
    }, indent=2))


if __name__ == "__main__":
    main()
