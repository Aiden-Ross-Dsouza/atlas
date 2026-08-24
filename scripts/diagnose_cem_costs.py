"""
scripts/diagnose_cem_costs.py — one-off diagnostic: does an E0 chart distort
CEM's cost landscape, or just make the whole search converge worse?

Context: run_e0_planning.py's graduated screen found the frozen baseline solves
episode 0 (seed=0) at R1, but every trained adapter fails it identically. This
captures CEM's actual per-candidate costs at iteration 0 (same RNG seed as any
other run -> same 300 sampled candidate action sequences, since planner.plan()
always starts from mean=0/std=var_scale) and at the final iteration, via a
runtime monkey-patch of CEMPlanner.cost_function (no jepa-wms files touched).

Usage:
    python scripts/diagnose_cem_costs.py --kind baseline --regime R1
    python scripts/diagnose_cem_costs.py --kind ln_act --regime R1
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from run_e0_planning import (
    HUB_PATH, build_cfg, load_dataset_states, make_obs_td, prepare_with_visual,
    sample_dataset_init_goal,
)

import atlas
from atlas.chart import Chart
from atlas.regimes import PhysicsRegime

import sys
if HUB_PATH not in sys.path:
    sys.path.insert(0, HUB_PATH)
from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv  # noqa: E402
from evals.simu_env_planning.envs.pusht_gym_wrap import PushTWrapper  # noqa: E402
from evals.simu_env_planning.planning.gc_agent import GC_Agent  # noqa: E402


def instrument_cost_function(planner, capture: list) -> None:
    original = planner.cost_function
    call_count = [0]

    def wrapped(actions, z_init):
        cost = original(actions, z_init)
        call_count[0] += 1
        if call_count[0] == 1 or call_count[0] == planner.iterations:
            capture.append({
                "iteration": call_count[0] - 1,
                "actions": actions.detach().cpu().numpy().tolist(),
                "costs": cost.detach().cpu().numpy().flatten().tolist(),
            })
        return cost

    planner.cost_function = wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CEM cost landscape under a chart vs baseline.")
    parser.add_argument("--kind", required=True, choices=["baseline", "ln_act", "lora4", "full"])
    parser.add_argument("--regime", required=True, choices=["R0", "R1", "R2"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-act-stepped", type=int, default=6)
    parser.add_argument("--charts-dir", default=str(atlas.OUT_DIR / "e0"))
    parser.add_argument("--out-dir", default=str(atlas.OUT_DIR / "e0_planning" / "cem_diagnostics"))
    args = parser.parse_args()

    from pathlib import Path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, prep = torch.hub.load(HUB_PATH, "dino_wm_pusht", source="local",
                                  force_reload=False, trust_repo=True)
    wm = model.model if hasattr(model, "model") else model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm, model = wm.to(device), model.to(device)
    for p in wm.encoder.parameters():
        p.requires_grad_(False)

    if args.kind != "baseline":
        chart_path = Path(args.charts_dir) / f"chart_{args.kind}_{args.regime}.pt"
        chart = Chart.load(str(chart_path), wm.predictor)
        chart.apply_(wm.predictor)

    cfg = build_cfg(args.num_samples, args.iterations, args.horizon, args.num_act_stepped)
    agent = GC_Agent(cfg, model, dset=None, preprocessor=prep)
    agent.device = device

    capture: list = []
    instrument_cost_function(agent.planner, capture)

    base_env = PushTEnv(render_size=224, with_velocity=True)
    wrapper = PushTWrapper(base_env)
    regime = PhysicsRegime(wrapper, args.regime)
    states, seq_lengths = load_dataset_states()

    rs = np.random.RandomState(args.seed)
    init_state, goal_state = sample_dataset_init_goal(states, seq_lengths, rs)

    goal_obs, _ = prepare_with_visual(base_env, args.seed, goal_state)
    regime._apply_physics()
    agent.set_goal(make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))

    obs, _ = prepare_with_visual(base_env, args.seed, init_state)
    regime._apply_physics()

    obs_td = make_obs_td(obs["visual"], obs["proprio"], device)
    action = agent.act(obs_td, steps_left=args.max_steps)
    print(f"kind={args.kind} regime={args.regime} seed={args.seed}: "
          f"iter0 min_cost={min(capture[0]['costs']):.4f} "
          f"final min_cost={min(capture[-1]['costs']):.4f} "
          f"chosen_action_mean_cost={capture[-1]['costs'][0]:.4f}")

    out_path = out_dir / f"{args.kind}_{args.regime}_seed{args.seed}.json"
    out_path.write_text(json.dumps({
        "kind": args.kind, "regime": args.regime, "seed": args.seed,
        "init_state": init_state.tolist(), "goal_state": goal_state.tolist(),
        "captures": capture,
    }))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
