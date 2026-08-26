"""
scripts/diagnose_cem_costs.py -- cost-ranking diagnostic (C3 validation):
does a chart's UMF improvement track a BETTER CEM cost ranking, or can UMF
improve while the planner's actual candidate ranking stays uninformative
(or gets worse)?

For ONE fixed (init_state, goal_state) pair: captures CEM's iteration-0
per-candidate costs under EACH requested kind (baseline + charts) -- every
kind sees the SAME K=num_samples candidate action sequences, since CEM's
iteration-0 draw is independent of the model (local_seed=0, mean=0/
std=var_scale, fixed in build_cfg()) -- then rolls out ALL K candidates for
REAL in the env (cheap CPU pymunk physics; the only GPU cost is the one CEM
search per kind already needed to capture costs) to get each candidate's
TRUE final block distance to goal -- an outcome no model, chart or baseline,
ever sees. Reports Spearman rho(planner_cost, true_final_dist) per kind:
both are "lower = better", so a well-calibrated cost ranking gives rho near
+1; near 0 or negative means the kind's cost function doesn't track real
outcomes even if its UMF (open-loop prediction error) looks good -- the
direct mechanism for why UMF and planning success can dissociate.

Rewritten for the post-T1/T6 pipeline (E0_IMPLEMENTATION_PLAN.md). The
original version of this file predated the rollout-bug fix and used
prepare_with_visual()'s old 3-arg signature and PhysicsRegime wrapping
PushTWrapper (both since changed in run_e0_planning.py, T6) -- replaced
outright rather than patched, since it would not run correctly as-is.

Usage:
    python scripts/diagnose_cem_costs.py --kinds baseline ln_act --regime R2 --seed 0
    # local smoke test (reduced CEM settings):
    python scripts/diagnose_cem_costs.py --kinds baseline --regime R2 --seed 0 \\
        --num-samples 16 --iterations 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from scipy.stats import spearmanr

import atlas
from atlas.chart import Chart
from atlas.regimes import PhysicsRegime

from run_e0_planning import (  # noqa: E402
    FRAMESKIP, HUB_PATH, block_success, build_cfg, load_dataset_states,
    make_obs_td, prepare_with_visual, sample_dataset_init_goal,
)

if HUB_PATH not in sys.path:
    sys.path.insert(0, HUB_PATH)
from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv  # noqa: E402
from evals.simu_env_planning.planning.gc_agent import GC_Agent  # noqa: E402


def instrument_cost_function(planner, capture: list) -> None:
    """Captures the FULL iteration-0 candidate batch (actions + costs) -- same
    candidates for every kind by construction (CEM's first draw doesn't depend
    on the model), so later iterations are irrelevant to this diagnostic and
    are not captured."""
    original = planner.cost_function
    call_count = [0]

    def wrapped(actions, z_init):
        cost = original(actions, z_init)
        call_count[0] += 1
        if call_count[0] == 1:
            capture.append(actions.detach().cpu().clone())  # [horizon, num_samples, A]
            capture.append(cost.detach().cpu().clone().flatten())  # [num_samples]
        return cost

    planner.cost_function = wrapped


def rollout_true_outcomes(base_env, regime, seed: int, init_state: np.ndarray,
                           goal_state: np.ndarray, model_actions: torch.Tensor,
                           agent: GC_Agent) -> np.ndarray:
    """model_actions: [horizon, num_samples, action_dim] (model-chunk,
    normalized -- CEM's own units). Resets to the identical init_state before
    EVERY candidate so each rollout is independent, then executes that
    candidate's raw actions for real. Returns per-candidate true final
    block_pos_diff (px, lower = better), shape [num_samples]."""
    num_samples = model_actions.shape[1]
    distances = np.empty(num_samples, dtype=np.float64)
    for i in range(num_samples):
        cand = model_actions[:, i, :]  # [horizon, A]
        raw = rearrange(cand, "t (f d) -> (t f) d", d=2)
        raw = agent.preprocessor.denormalize_actions(raw).numpy()
        prepare_with_visual(base_env, regime, seed, init_state)  # reset -- same start every candidate
        final_state = None
        for a in raw:
            _, _, _, info = base_env.step(a)
            final_state = info["state"]
        distances[i] = block_success(goal_state, final_state)["block_pos_diff"]
    return distances


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cost-ranking diagnostic: Spearman rho(planner cost, true outcome) per kind.")
    parser.add_argument("--kinds", nargs="+", required=True,
                         choices=["baseline", "ln_act", "lora4", "full"])
    parser.add_argument("--regime", required=True, choices=["R0", "R1", "R2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                         help="One (state,goal) pair per seed -- loops all kinds x all seeds, "
                              "reusing the loaded model (only the predictor weights get swapped "
                              "between kinds, via a pristine state-dict snapshot).")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-act-stepped", type=int, default=6)
    parser.add_argument("--charts-dir", type=Path, default=atlas.OUT_DIR / "e0")
    parser.add_argument("--out-dir", type=Path, default=atlas.OUT_DIR / "cost_ranking")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dino_wm_pusht from local hub...")
    model, prep = torch.hub.load(HUB_PATH, "dino_wm_pusht", source="local",
                                  force_reload=False, trust_repo=True)
    wm = model.model if hasattr(model, "model") else model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm, model = wm.to(device), model.to(device)
    for p in wm.encoder.parameters():
        p.requires_grad_(False)
    for m in wm.predictor.modules():  # T7 throughput fix -- see run_e0_planning.py
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = True
    torch.set_float32_matmul_precision("high")

    # Snapshot BEFORE any chart is applied -- the safe way back to pristine
    # weights between kinds (chart.restore_() does NOT do this for most
    # kinds; see HANDOFF.md's chart.restore_() gotcha).
    pristine_predictor_state = {k: v.clone() for k, v in wm.predictor.state_dict().items()}

    base_env = PushTEnv(render_size=224, with_velocity=True)
    regime = PhysicsRegime(base_env, args.regime)  # wraps base_env directly (T6) -- NOT PushTWrapper
    states, seq_lengths = load_dataset_states()
    cfg = build_cfg(args.num_samples, args.iterations, args.horizon, args.num_act_stepped)

    per_seed: list[dict] = []
    pooled_costs: dict[str, list[float]] = {k: [] for k in args.kinds}
    pooled_dist: dict[str, list[float]] = {k: [] for k in args.kinds}

    for seed in args.seeds:
        rs = np.random.RandomState(seed)
        init_state, goal_state = sample_dataset_init_goal(states, seq_lengths, rs)
        print(f"\n== seed={seed} regime={args.regime} "
              f"init_block_pos_diff={np.linalg.norm(goal_state[2:4] - init_state[2:4]):.1f}px ==")

        results: dict = {}
        shared_actions: torch.Tensor | None = None
        for kind in args.kinds:
            wm.predictor.load_state_dict(pristine_predictor_state)
            if kind != "baseline":
                chart_path = args.charts_dir / f"chart_{kind}_{args.regime}.pt"
                chart = Chart.load(str(chart_path), wm.predictor)
                chart.apply_(wm.predictor)

            agent = GC_Agent(cfg, model, dset=None, preprocessor=prep)
            agent.device = device

            capture: list = []
            instrument_cost_function(agent.planner, capture)

            goal_obs, _ = prepare_with_visual(base_env, regime, seed, goal_state)
            agent.set_goal(make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))
            obs, _ = prepare_with_visual(base_env, regime, seed, init_state)
            obs_td = make_obs_td(obs["visual"], obs["proprio"], device)
            # steps_left in MODEL-chunk units (run_e0_planning.py's convention);
            # a single replan's worth (n_replans_target=1) is just num_act_stepped.
            agent.act(obs_td, steps_left=args.num_act_stepped)

            actions, costs = capture  # [horizon, num_samples, A], [num_samples]
            if shared_actions is None:
                shared_actions = actions
            else:
                max_diff = (actions - shared_actions).abs().max().item()
                if max_diff > 1e-5:
                    print(f"  [WARN] seed={seed} kind={kind}'s iteration-0 candidates differ from "
                          f"the first kind's by {max_diff:.6g} -- the same-input guarantee this "
                          f"diagnostic relies on doesn't hold here for this seed.")

            true_dist = rollout_true_outcomes(base_env, regime, seed, init_state, goal_state,
                                               actions, agent)
            rho, pval = spearmanr(costs.numpy(), true_dist)
            print(f"  kind={kind}: rho={rho:.3f} (p={pval:.4g}), n={len(true_dist)}, "
                  f"mean_true_dist={true_dist.mean():.1f}px, "
                  f"best_by_cost_true_dist={true_dist[int(costs.argmin())]:.1f}px")
            results[kind] = {
                "spearman_rho": float(rho), "spearman_p": float(pval),
                "n_candidates": int(len(true_dist)),
                "mean_true_dist": float(true_dist.mean()),
                "best_by_cost_true_dist": float(true_dist[int(costs.argmin())]),
            }
            pooled_costs[kind].extend(costs.numpy().tolist())
            pooled_dist[kind].extend(true_dist.tolist())

        per_seed.append({
            "seed": seed, "init_state": init_state.tolist(), "goal_state": goal_state.tolist(),
            "results": results,
        })

    print(f"\n== Pooled across {len(args.seeds)} seed(s), "
          f"{args.num_samples} candidates/seed ==")
    pooled_summary: dict = {}
    for kind in args.kinds:
        rho, pval = spearmanr(pooled_costs[kind], pooled_dist[kind])
        mean_of_seed_rhos = float(np.mean([s["results"][kind]["spearman_rho"] for s in per_seed]))
        print(f"  kind={kind}: pooled rho={rho:.3f} (p={pval:.4g}, n={len(pooled_dist[kind])}) | "
              f"mean of per-seed rhos={mean_of_seed_rhos:.3f}")
        pooled_summary[kind] = {
            "pooled_spearman_rho": float(rho), "pooled_spearman_p": float(pval),
            "pooled_n": len(pooled_dist[kind]),
            "mean_of_per_seed_rhos": mean_of_seed_rhos,
        }

    seeds_str = "-".join(str(s) for s in args.seeds)
    out_path = args.out_dir / f"cost_ranking_{args.regime}_seeds{seeds_str}.json"
    out_path.write_text(json.dumps({
        "regime": args.regime, "seeds": args.seeds, "kinds": args.kinds,
        "num_samples": args.num_samples, "iterations": args.iterations,
        "per_seed": per_seed, "pooled": pooled_summary,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
