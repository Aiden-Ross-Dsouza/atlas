"""
scripts/profile_episode.py — Day-2 compute budget calibration.

Runs a single episode with the frozen model and prints:
  - Seconds per episode
  - Peak VRAM (MB)
  - Predictor forwards per replan

Usage:
    python scripts/profile_episode.py --episodes 3
"""

from __future__ import annotations

import argparse
import time

import torch
import atlas


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a Push-T planning episode.")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    print("Loading dino_wm_pusht...")
    model, prep = torch.hub.load(
        "facebookresearch/jepa-wms", "dino_wm_pusht",
        force_reload=False, trust_repo=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA device found. Profiling on CPU will not reflect GPU timing.")
    model = model.to(device)

    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.predictor.parameters():
        p.requires_grad_(False)

    print(f"\nProfiling {args.episodes} frozen episodes on {device}...")
    print("(This requires the jepa-wms CEM planning eval to be integrated.)")
    print("See implementation plan §7.0 for the integration point.")
    raise NotImplementedError(
        "profile_episode.py requires integration with the jepa-wms CEM planning loop. "
        "Wire the eval loop from evals/simu_env_planning/ and re-run.\n"
        "Formula: GPU-h = (sec/ep × episodes × segments × seeds × arms) / 3600\n"
        "At 30 s/ep: E4 = 20×6×3×7 = 2520 eps ≈ 21 GPU-h.\n"
        "If >40 s/ep, apply budget cuts from config e4.yaml."
    )


if __name__ == "__main__":
    main()
