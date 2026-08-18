"""
scripts/run_e1.py — E1: Fitness routing experiment (THE GATE).

Per-episode: 2 warmup replans under c₀, then route using the specified router
for the rest. Same start states and goals across ALL routers (paired seeding, G5).

Usage:
    python scripts/run_e1.py --charts atlas_out/e0/ln_act --routers umf sdyn random oracle_id
    python scripts/run_e1.py --episodes 60 --seeds 3

Output:
    atlas_out/e1/episodes.jsonl   (per-episode JSONL log)
    atlas_out/e1/T1.md            (Table T1: routing comparison)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import atlas
from atlas.library import Library
from atlas.chart import Chart
from atlas.stats import normalised_recovery, paired_bootstrap, success_rate_ci


def main() -> None:
    parser = argparse.ArgumentParser(description="E1: Fitness routing evaluation.")
    parser.add_argument("--charts", type=Path, required=True,
                        help="Directory with chart .pt files from E0.")
    parser.add_argument("--routers", nargs="+",
                        default=["umf", "e1", "sdyn", "random", "oracle_id"],
                        choices=["umf", "e1", "sdyn", "random", "oracle_id"])
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--regime", default="R1")
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e1")
    args = parser.parse_args()

    print("Loading dino_wm_pusht...")
    model, prep = torch.hub.load(
        "facebookresearch/jepa-wms", "dino_wm_pusht",
        force_reload=False, trust_repo=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.predictor.parameters():
        p.requires_grad_(False)

    # Load library from E0 output.
    print(f"Loading charts from {args.charts}...")
    library = Library.load(args.charts, model.predictor)
    print(f"  Library: {library}")

    print(f"\nE1: {len(args.routers)} routers × {args.episodes} eps × {args.seeds} seeds")
    print(f"Output: {args.out}")

    raise NotImplementedError(
        "run_e1.py: integrate the jepa-wms CEM planning eval loop here. "
        "Per-episode structure:\n"
        "  1. Reset env with paired_seed(seg, ep_idx)\n"
        "  2. Run 2 warmup replans under c₀ (record UMF but do not route)\n"
        "  3. Route: call atlas.router.route(kind, library, ...)\n"
        "  4. Apply selected chart, run CEM, execute 5 actions\n"
        "  5. Record success (reached goal within max_mpc_steps)\n"
        "  6. Log to episodes.jsonl via harness.log_episode()\n\n"
        "After all episodes: compute T1 using stats.normalised_recovery() "
        "and stats.paired_bootstrap().\n"
        "Pass criterion: normalised_recovery(UMF) >= 0.80 when oracle-random >= 10 pp."
    )


if __name__ == "__main__":
    main()
