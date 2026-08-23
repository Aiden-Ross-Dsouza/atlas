"""
scripts/smoke_e1.py — E1 smoke test. Run this BEFORE spending real GPU-hours on
the full 60-episode/3-seed run. Uses a tiny CEM budget (matches jepa-wms's own
`quick_debug` mode: num_samples=2, iterations=2) and 1-2 episodes per router.

Asserts, not just prints:
  1. Every router returns a valid chart index for a real chunk (catches the
     now-fixed _e1_score return-value bug directly, and any regression of it).
  2. Pairing (G5-style): two different routers given the same seed see
     identical initial state and goal state.
  3. episodes.jsonl is written with one record per (router, episode) and is
     valid JSON.
  4. Warmup replans never route away from c0 (selected_trace[:N_WARMUP] == [0]*N).
  5. load_library_from_e0() round-trips against a real E0 output directory.

Usage:
    python scripts/smoke_e1.py --charts atlas_out/e0 --kind ln_act --regime R1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so `scripts.run_e1` is importable regardless of CWD
HUB_PATH = str(REPO_ROOT / "hub" / "hub" / "facebookresearch_jepa-wms_main")
if HUB_PATH not in sys.path:
    sys.path.insert(0, HUB_PATH)

import atlas  # noqa: E402
from atlas.harness import run_e1_episode  # noqa: E402
from atlas.regimes import PhysicsRegime  # noqa: E402
from scripts.run_e1 import (  # noqa: E402
    FRAMESKIP,
    N_WARMUP_REPLANS,
    REGIME_LABELS,
    build_planner_cfg,
    load_library_from_e0,
)

from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv  # noqa: E402
from evals.simu_env_planning.envs.pusht_gym_wrap import PushTWrapper  # noqa: E402
from evals.simu_env_planning.planning.gc_agent import GC_Agent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 smoke test.")
    parser.add_argument("--charts", type=Path, required=True)
    parser.add_argument("--kind", default="ln_act", choices=["ln_act", "lora4", "full"])
    parser.add_argument("--regime", default="R1", choices=["R0", "R1", "R2"])
    parser.add_argument("--library-regimes", nargs="+", default=["R1", "R2"])
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e1_smoke")
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)  # fresh JSONL each run, so assertion 3 isn't fooled by stale data

    n_replans_target = 4  # small but > N_WARMUP_REPLANS(2), matches quick_debug-scale budget
    assert n_replans_target > N_WARMUP_REPLANS, "smoke config must leave room for routed replans"

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

    print(f"[1/5] Loading {args.kind} charts from {args.charts}...")
    library, label_to_chart = load_library_from_e0(args.charts, args.kind, args.library_regimes, wm.predictor)
    assert len(library) == len(args.library_regimes) + 1, (
        f"expected {len(args.library_regimes) + 1} charts (c0 + {args.library_regimes}), got {len(library)}"
    )
    print(f"  OK: {library}, label_to_chart={label_to_chart}")

    # Tiny CEM budget -- matches jepa-wms's own quick_debug (num_samples=2, iterations=2).
    cfg = build_planner_cfg(num_samples=2, iterations=2, horizon=4, num_act_stepped=1)
    agent = GC_Agent(cfg, model, dset=None, preprocessor=prep)
    agent.device = device

    base_env = PushTEnv(render_size=224, with_velocity=True)
    regime_wrapper = PhysicsRegime(base_env, args.regime)
    goal_utils = PushTWrapper(base_env)

    print("[2/5] Checking pairing: two routers, same seed -> identical init/goal states...")
    seed = 12345
    init_a, goal_a = goal_utils.sample_random_init_goal_states(seed)
    init_b, goal_b = goal_utils.sample_random_init_goal_states(seed)
    assert (init_a == init_b).all() and (goal_a == goal_b).all(), (
        "sample_random_init_goal_states(seed) is not deterministic -- pairing (G5) would be broken"
    )
    print("  OK: identical init/goal states for the same seed.")

    print("[3/5] Running 1 episode per router with tiny CEM budget...")
    routers = ["umf", "e1", "sdyn", "random", "oracle_id"]
    records = []
    for router in routers:
        record = run_e1_episode(
            library=library, agent=agent, world_model=wm,
            base_env=base_env, regime=regime_wrapper, goal_utils=goal_utils,
            router=router, episode_seed=seed,
            n_warmup_replans=N_WARMUP_REPLANS, n_replans_target=n_replans_target,
            frameskip=FRAMESKIP, num_act_stepped=1, motion_gate=None, hysteresis=0.05,
            out_dir=args.out, episode_id=f"smoke_{router}",
            regime_label=REGIME_LABELS[args.regime], label_to_chart=label_to_chart,
        )
        records.append(record)
        selected = record["selected_trace"]
        assert all(0 <= idx < len(library) for idx in selected), (
            f"[{router}] selected_trace has an out-of-range chart index: {selected}"
        )
        n_warmup_actual = min(N_WARMUP_REPLANS, len(selected))
        assert selected[:n_warmup_actual] == [0] * n_warmup_actual, (
            f"[{router}] warmup replans routed away from c0: {selected[:n_warmup_actual]}"
        )
        print(f"  OK [{router}]: selected_trace={selected} success={record['success']}")

    print("[4/5] Checking episodes.jsonl...")
    jsonl_path = args.out / "episodes.jsonl"
    assert jsonl_path.exists(), f"{jsonl_path} was not written"
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == len(routers), f"expected {len(routers)} JSONL lines, got {len(lines)}"
    for line in lines:
        json.loads(line)  # raises if invalid
    print(f"  OK: {len(lines)} valid JSONL records at {jsonl_path}")

    print("[5/5] All E1 smoke-test assertions passed.")


if __name__ == "__main__":
    main()
