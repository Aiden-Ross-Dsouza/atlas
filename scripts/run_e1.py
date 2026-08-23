"""
scripts/run_e1.py — E1: Fitness routing experiment (THE GATE).

Per-episode: n_warmup_replans replans under c₀, then route using the specified
router for the rest. Regime is FIXED for the whole run (--regime) — E1 does not
switch regimes mid-episode (that's E3/E4's stream, a different experiment).
Same start states and goals across ALL routers via atlas.streams.paired_seed
(same seg_idx=0 for every episode index — a single-regime run, not a stream).

Does NOT use jepa-wms's own eval.py/PlanEvaluator — confirmed broken for
Push-T (see code-review.md Bug #7: PlanEvaluator.eval() reads a Metaworld-only
env attribute; TensorWrapper.step() assumes an info["success"] key Push-T
never sets). Reuses GC_Agent/CEMPlanner directly (generic, not Push-T-specific)
plus PushTWrapper's two stateless utility methods, following the same pattern
scripts/run_e0_planning.py already validated end-to-end.

Usage:
    python scripts/run_e1.py --charts atlas_out/e0 --kind ln_act --regime R1 \\
        --routers umf sdyn random oracle_id --episodes 2 --seeds 1   # smoke-scale
    python scripts/run_e1.py --charts atlas_out/e0 --kind ln_act --regime R1 \\
        --episodes 60 --seeds 3                                      # full spec

Output:
    atlas_out/e1/episodes.jsonl   (per-episode JSONL log)
    atlas_out/e1/T1.md            (Table T1: routing comparison)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import atlas
from atlas.chart import Chart
from atlas.harness import run_e1_episode
from atlas.library import Library
from atlas.stats import normalised_recovery, paired_bootstrap, success_rate_ci
from atlas.streams import paired_seed

REPO_ROOT = Path(__file__).parent.parent
HUB_PATH = str(REPO_ROOT / "hub" / "hub" / "facebookresearch_jepa-wms_main")
if HUB_PATH not in sys.path:
    sys.path.insert(0, HUB_PATH)

from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv  # noqa: E402
from evals.simu_env_planning.envs.pusht_gym_wrap import PushTWrapper  # noqa: E402
from evals.simu_env_planning.planning.gc_agent import GC_Agent  # noqa: E402

from atlas.regimes import PhysicsRegime  # noqa: E402

# Model action chunking: one model-level action = FRAMESKIP raw env actions
# concatenated (data.custom.frameskip=5 in the shipped dino_wm_pusht config,
# confirmed empirically in scripts/run_e0_planning.py's Bug #4). CEMPlanner's
# `horizon`/`num_act_stepped` are in model-chunk units (same units `horizon`
# unambiguously uses, since it directly bounds the number of predictor
# forward_pred() calls) -- so the plan's "horizon 25, 5 executed actions/replan,
# <=30 MPC steps" (implementation plan Sec7.0 / README) are read here as
# model-chunk units throughout: 30/5 = 6 replans/episode, each executing
# 5*FRAMESKIP = 25 raw env actions. This is the reading most consistent with
# horizon's units, but the source docs never spell out the frameskip interaction
# explicitly -- ASSUMPTION, not a settled fact. Verify replan/episode-length
# counts against a real run's logged `n_replans`/`elapsed_raw_steps` before
# trusting this for the reported T1 numbers.
FRAMESKIP = 5
CEM_NUM_SAMPLES = 200
CEM_ITERATIONS = 10
CEM_HORIZON = 25
CEM_NUM_ACT_STEPPED = 5
MAX_MPC_STEPS = 30
N_WARMUP_REPLANS = 2
HYSTERESIS = 0.05

REGIME_LABELS = {"R0": 0, "R1": 1, "R2": 2}


def build_planner_cfg(num_samples: int, iterations: int, horizon: int, num_act_stepped: int) -> OmegaConf:
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


def load_library_from_e0(
    charts_dir: Path, kind: str, regimes: list[str], predictor
) -> tuple[Library, dict[int, int]]:
    """
    Build a Library directly from E0's on-disk layout (chart_{kind}_{regime}.pt
    files in charts_dir) — E0 does NOT write Library.save()'s format (no
    library_meta.pt, different filenames), so Library.load() cannot be used
    against E0 output directly. c0 (index 0) is a fresh identity chart built
    from the live (frozen) predictor; each regime's fine-tuned chart is loaded
    and added in order. Returns (library, label_to_chart) for the oracle_id
    router, keyed by the same integer regime labels used throughout this script.
    """
    c0 = Chart(predictor, kind)
    library = Library(c0, max_size=max(10, len(regimes) + 1))
    label_to_chart: dict[int, int] = {}
    for regime in regimes:
        chart_path = charts_dir / f"chart_{kind}_{regime}.pt"
        if not chart_path.exists():
            raise FileNotFoundError(
                f"E0 chart not found: {chart_path}. Run scripts/run_e0.py first, "
                f"or point --charts at a directory containing chart_{{kind}}_{{regime}}.pt files."
            )
        chart = Chart.load(chart_path, predictor)
        idx = library.add(chart)
        label_to_chart[REGIME_LABELS[regime]] = idx
    return library, label_to_chart


def compute_t1(records: list[dict], routers: list[str]) -> str:
    """Build Table T1: router x {SR, oracle gap, normalised recovery}, paired bootstrap CIs."""
    by_router: dict[str, list[dict]] = {r: [] for r in routers}
    for rec in records:
        by_router[rec["router"]].append(rec)

    # Pair by (seed_run, ep_idx) — every router shares the same paired_seed episode order.
    def outcomes(router: str) -> np.ndarray:
        rows = sorted(by_router[router], key=lambda r: (r["seed_run"], r["ep_idx"]))
        return np.array([1.0 if r["success"] else 0.0 for r in rows])

    sr = {r: outcomes(r) for r in routers}
    sr_mean = {r: float(sr[r].mean()) if len(sr[r]) else float("nan") for r in routers}

    sr_oracle = sr_mean.get("oracle_id", float("nan"))
    sr_random = sr_mean.get("random", float("nan"))

    lines = [
        "# Table T1: E1 Fitness Routing",
        "",
        "| Router | SR | Oracle gap | Normalised recovery |",
        "|---|---|---|---|",
    ]
    for router in routers:
        gap = sr_oracle - sr_mean[router] if router not in ("oracle_id",) else 0.0
        rec_score = normalised_recovery(sr_mean[router], sr_oracle, sr_random)
        rec_str = f"{rec_score:.3f}" if rec_score is not None else "N/A (spread < 10pp)"
        lines.append(f"| {router} | {sr_mean[router]:.3f} | {gap:.3f} | {rec_str} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="E1: Fitness routing evaluation.")
    parser.add_argument("--charts", type=Path, required=True,
                        help="Directory with chart_{kind}_{regime}.pt files from E0.")
    parser.add_argument("--kind", default="ln_act", choices=["ln_act", "lora4", "full"],
                        help="Which E0 adapter kind to load charts for.")
    parser.add_argument("--routers", nargs="+",
                        default=["umf", "e1", "sdyn", "random", "oracle_id"],
                        choices=["umf", "e1", "sdyn", "random", "oracle_id"])
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--regime", default="R1", choices=["R0", "R1", "R2"],
                        help="Fixed regime for this E1 run (routing target). "
                             "No mid-episode switching -- see module docstring.")
    parser.add_argument("--library-regimes", nargs="+", default=["R1", "R2"],
                        help="Regimes whose E0 charts populate the library, in "
                             "addition to a fresh c0.")
    parser.add_argument("--num-samples", type=int, default=CEM_NUM_SAMPLES)
    parser.add_argument("--iterations", type=int, default=CEM_ITERATIONS)
    parser.add_argument("--horizon", type=int, default=CEM_HORIZON)
    parser.add_argument("--num-act-stepped", type=int, default=CEM_NUM_ACT_STEPPED)
    parser.add_argument("--max-mpc-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e1")
    args = parser.parse_args()

    n_replans_target = max(args.max_mpc_steps // args.num_act_stepped, 1)
    if args.max_mpc_steps % args.num_act_stepped != 0:
        print(f"WARNING: --max-mpc-steps ({args.max_mpc_steps}) is not a multiple of "
              f"--num-act-stepped ({args.num_act_stepped}); last replan will be shorter.")
    if n_replans_target <= N_WARMUP_REPLANS:
        raise ValueError(
            f"n_replans_target={n_replans_target} <= N_WARMUP_REPLANS={N_WARMUP_REPLANS}: "
            "no replans would be left for routing. Increase --max-mpc-steps or decrease "
            "--num-act-stepped."
        )
    print(f"n_replans_target={n_replans_target} ({N_WARMUP_REPLANS} warmup, "
          f"{n_replans_target - N_WARMUP_REPLANS} routed)")

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
    for p in wm.predictor.parameters():
        p.requires_grad_(False)

    print(f"Loading {args.kind} charts from {args.charts} for regimes {args.library_regimes}...")
    library, label_to_chart = load_library_from_e0(args.charts, args.kind, args.library_regimes, wm.predictor)
    print(f"  {library}, label_to_chart={label_to_chart}")

    motion_gate = None  # TODO: compute via atlas.score.compute_motion_gate() over a
                         # fresh sample of training displacements -- E0's own eval
                         # skips this too (documented gap, see E0_RESULTS.md); worth
                         # fixing here rather than repeating it, before the full run.

    cfg = build_planner_cfg(args.num_samples, args.iterations, args.horizon, args.num_act_stepped)
    agent = GC_Agent(cfg, model, dset=None, preprocessor=prep)
    agent.device = device

    base_env = PushTEnv(render_size=224, with_velocity=True)
    regime_wrapper = PhysicsRegime(base_env, args.regime)
    goal_utils = PushTWrapper(base_env)  # only .sample_random_init_goal_states/.eval_state used

    print(f"\nE1: {len(args.routers)} routers x {args.episodes} eps x {args.seeds} seeds, "
          f"regime={args.regime}, kind={args.kind}")
    print(f"Output: {args.out}")

    args.out.mkdir(parents=True, exist_ok=True)
    all_records = []
    for router in args.routers:
        for seed_run in range(args.seeds):
            for ep_idx in range(args.episodes):
                seed = paired_seed(0, ep_idx + seed_run * 10_000)
                record = run_e1_episode(
                    library=library,
                    agent=agent,
                    world_model=wm,
                    base_env=base_env,
                    regime=regime_wrapper,
                    goal_utils=goal_utils,
                    router=router,
                    episode_seed=seed,
                    n_warmup_replans=N_WARMUP_REPLANS,
                    n_replans_target=n_replans_target,
                    frameskip=FRAMESKIP,
                    num_act_stepped=args.num_act_stepped,
                    motion_gate=motion_gate,
                    hysteresis=HYSTERESIS,
                    out_dir=args.out,
                    episode_id=f"{router}_{seed_run}_{ep_idx}",
                    regime_label=REGIME_LABELS[args.regime],
                    label_to_chart=label_to_chart,
                )
                record["seed_run"] = seed_run
                record["ep_idx"] = ep_idx
                all_records.append(record)
                print(f"  [{router}] seed={seed_run} ep={ep_idx}: "
                      f"success={record['success']} replans={record['n_replans']} "
                      f"steps={record['elapsed_raw_steps']}")

    t1_md = compute_t1(all_records, args.routers)
    (args.out / "T1.md").write_text(t1_md)
    print(f"\n{t1_md}")
    print(f"\nEpisodes JSONL: {args.out / 'episodes.jsonl'}")
    print(f"T1 table: {args.out / 'T1.md'}")


if __name__ == "__main__":
    main()
