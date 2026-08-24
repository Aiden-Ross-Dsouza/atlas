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
from atlas.score import compute_motion_gate
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
# concatenated (data.custom.frameskip=5 in the shipped dino_wm_pusht config).
# CEMPlanner's `horizon`/`num_act_stepped` are in model-chunk units.
#
# E0_IMPLEMENTATION_PLAN.md T6: restored to the SUBSTRATE's own validated
# Push-T config (num_samples=300, iterations=30, horizon=6, num_act_stepped=6
# -- the config dino_wm_pusht reports ~90% SR under), replacing both the
# AdaJEPA-derived Sec7.0 reading this file used before (200/10/25/5, a
# different substrate's numbers) and run_e0_planning.py's separate
# num_act_stepped=1 reading -- T6 resolves that discrepancy for good.
#
# OPEN ISSUE this file's own multi-replan design creates, NOT resolved by T6:
# at nas=6, num_act_stepped(6)*FRAMESKIP(5) = 30 raw actions per replan, i.e.
# ONE replan already covers a full 30-raw-step episode (matching
# run_e0_planning.py's "1 replan" reading exactly) -- but E1 NEEDS multiple
# replans per episode (N_WARMUP_REPLANS then routed ones) to exercise routing
# at all. MAX_MPC_STEPS=30 with nas=6 therefore does NOT give room for
# N_WARMUP_REPLANS(2) + >=1 routed replan within one 30-raw-step episode --
# n_replans_target = MAX_MPC_STEPS // CEM_NUM_ACT_STEPPED = 30 // 6 = 5 model
# "MPC steps" worth of replans is itself a unit mismatch (MAX_MPC_STEPS was
# never defined as raw vs. model-chunk units for E1 specifically). This needs
# a real decision (e.g. MAX_MPC_STEPS scaled to N_desired_replans *
# num_act_stepped * FRAMESKIP raw steps) before the real 60x3 T11 run -- flag
# to the user rather than silently picking a value; not a code-correctness
# bug T6 authorizes fixing unilaterally.
FRAMESKIP = 5
CEM_NUM_SAMPLES = 300
CEM_ITERATIONS = 30
CEM_HORIZON = 6
CEM_NUM_ACT_STEPPED = 6
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
    """Build Table T1: router x {SR, oracle gap, normalised recovery}, paired bootstrap CIs.

    CIs (E0_IMPLEMENTATION_PLAN.md T12 #7 -- this docstring previously promised
    them but the body computed none): SR uses stats.success_rate_ci; oracle gap
    uses stats.paired_bootstrap on the SAME (seed_run, ep_idx)-paired episode
    order every router shares (plan Sec8: paired, never unpaired). Normalised
    recovery has no closed-form CI, so it's bootstrapped directly: resample
    episode indices jointly across router/oracle/random each iteration and
    recompute the ratio, using stats.normalised_recovery's own min_spread=0.10
    gate per resample (a resample can independently fall below the spread
    floor even when the point estimate doesn't).
    """
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

    def fmt_ci(lo: float, hi: float) -> str:
        return f"[{lo:.3f}, {hi:.3f}]"

    def bootstrap_normalised_recovery_ci(fit: np.ndarray, oracle: np.ndarray, random_: np.ndarray,
                                          n: int = 10_000, seed: int = 0) -> tuple[float, float] | None:
        if len(fit) != len(oracle) or len(fit) != len(random_) or len(fit) == 0:
            return None
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(fit), (n, len(fit)))
        fit_r, oracle_r, random_r = fit[idx].mean(1), oracle[idx].mean(1), random_[idx].mean(1)
        spread = oracle_r - random_r
        valid = spread >= 0.10
        if not valid.any():
            return None
        ratios = (fit_r[valid] - random_r[valid]) / spread[valid]
        return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))

    lines = [
        "# Table T1: E1 Fitness Routing",
        "",
        "| Router | SR [95% CI] | Oracle gap [95% CI] | Normalised recovery [95% CI] |",
        "|---|---|---|---|",
    ]
    for router in routers:
        sr_lo, sr_hi = success_rate_ci(sr[router])[1] if len(sr[router]) else (float("nan"), float("nan"))
        if router == "oracle_id":
            gap_str = "0.000 (by definition)"
        else:
            gap_mean, (gap_lo, gap_hi) = paired_bootstrap(sr["oracle_id"], sr[router])
            gap_str = f"{gap_mean:.3f} {fmt_ci(gap_lo, gap_hi)}"

        rec_score = normalised_recovery(sr_mean[router], sr_oracle, sr_random)
        if rec_score is None:
            rec_str = "N/A (spread < 10pp)"
        else:
            rec_ci = bootstrap_normalised_recovery_ci(sr[router], sr["oracle_id"], sr["random"])
            rec_str = f"{rec_score:.3f} {fmt_ci(*rec_ci)}" if rec_ci is not None else f"{rec_score:.3f} (CI: N/A)"

        lines.append(f"| {router} | {sr_mean[router]:.3f} {fmt_ci(sr_lo, sr_hi)} | {gap_str} | {rec_str} |")
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

    # T7 throughput fixes -- see run_e0_planning.py's identical block for detail.
    for m in wm.predictor.modules():
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = True
    torch.set_float32_matmul_precision("high")

    print(f"Loading {args.kind} charts from {args.charts} for regimes {args.library_regimes}...")
    library, label_to_chart = load_library_from_e0(args.charts, args.kind, args.library_regimes, wm.predictor)
    print(f"  {library}, label_to_chart={label_to_chart}")

    # Gate G6: informative-chunk threshold, computed from a fresh sample of
    # this regime's training displacements (10th percentile) -- reuses
    # run_e0.py's own trajectory sampler so the definition matches E0's
    # (E0_IMPLEMENTATION_PLAN.md T5; E0's own eval had the same gap, fixed
    # alongside this).
    print(f"Computing motion_gate from a fresh {args.regime} trajectory sample...")
    from scripts.run_e0 import load_regime_trajectories
    gate_trajectories = load_regime_trajectories(
        model, prep, args.regime, num_trajs=3, traj_len=10, device=device, seed_offset=20_000)
    gate_displacements = torch.tensor([
        (t["encoder_output"][-1] - t["encoder_output"][0]).norm(p="fro").item()
        for t in gate_trajectories
    ])
    motion_gate = compute_motion_gate(gate_displacements)
    print(f"  motion_gate = {motion_gate:.4f}")

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
    # harness.log_episode() opens in append mode by design (accumulates
    # records across this run's episode loop) -- so a stale file from a
    # PREVIOUS run must be cleared here, once, before the loop starts, or
    # re-running silently concatenates old and new runs into one file
    # (E0_IMPLEMENTATION_PLAN.md T12 #8).
    episodes_jsonl = args.out / "episodes.jsonl"
    if episodes_jsonl.exists():
        episodes_jsonl.unlink()
    all_records = []
    for router in args.routers:
        for seed_run in range(args.seeds):
            for ep_idx in range(args.episodes):
                seed = paired_seed(0, ep_idx + seed_run * 10_000)
                record = run_e1_episode(
                    library=library,
                    agent=agent,
                    world_model=model,
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
