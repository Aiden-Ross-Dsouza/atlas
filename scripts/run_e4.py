"""
scripts/run_e4.py — E4 + E3: Continual stream (S2) with 7-arm ablation ladder.

E4 (RQ4): persistent chart library delivers recall on A→B→A.
E3 (RQ3): expansion arms 4/5/6 form the expansion ablation inside the same run.

Arms (in order of the ablation ladder):
  1. frozen
  2. adajepa
  3. adajepa_persist
  4. atlas_fixed      (expansion_mode: fixed)
  5. atlas_detect     (expansion_mode: detect_only)
  6. atlas            (expansion_mode: atlas, full verification)
  7. oracle_id

Usage:
    python scripts/run_e4.py --arms frozen adajepa atlas --episodes 20 --seeds 3
    python scripts/run_e4.py --profile --episodes 3  # budget calibration
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import torch
import atlas
from atlas.streams import get_stream, EpisodeSpec
from atlas.loop import ATLASConfig

ArmName = Literal[
    "frozen", "adajepa", "adajepa_persist",
    "atlas_fixed", "atlas_detect", "atlas", "oracle_id"
]

ALL_ARMS: list[ArmName] = [
    "frozen", "adajepa", "adajepa_persist",
    "atlas_fixed", "atlas_detect", "atlas", "oracle_id",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="E4+E3: Continual stream S2.")
    parser.add_argument("--stream", default="s2", choices=["s2"])
    parser.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e4")
    parser.add_argument("--profile", action="store_true",
                        help="Profile mode: run one arm for --episodes, print timing, exit.")
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

    # Generate paired episode specs.
    streams = get_stream(args.stream, args.episodes, args.seeds)
    print(f"\nStream {args.stream}: {len(streams)} seeds × "
          f"{len(streams[0])} episodes each")
    print(f"Arms: {args.arms}")
    print(f"Output: {args.out}")

    # ATLAS config (defaults from plan §7.7; chart kind set after E0).
    atlas_cfg = ATLASConfig(
        router="umf",
        tau=0.5,
        q=3,
        hysteresis=0.05,
        lr=5e-4,
        n_probe=20,
        k_max=10,
    )

    raise NotImplementedError(
        "run_e4.py: integrate the jepa-wms CEM planning eval loop.\n"
        "Per-arm per-episode structure:\n"
        "  1. Reset env with episode_spec.seed (via gymnasium env.reset(seed=...))\n"
        "  2. Apply PhysicsRegime(env, episode_spec.regime) — call reset() AFTER applying\n"
        "  3. Run the arm's adaptation loop:\n"
        "     - frozen: plan-only, no adaptation\n"
        "     - adajepa: AdaJEPA(variant='adajepa').reset() per episode\n"
        "     - adajepa_persist: AdaJEPA(variant='persistent'), no reset\n"
        "     - atlas_*: atlas_step() → plan → atlas_refine() (see loop.py)\n"
        "  4. Record per-episode: {arm, seed, segment, regime, success, UMF per chart}\n"
        "  5. Log to atlas_out/e4/episodes.jsonl\n\n"
        "After all episodes: compute T2 and F1 using stats.py and plots.money_plot().\n"
        "Budget calibration: run with --profile --episodes 3 to measure sec/episode."
    )


if __name__ == "__main__":
    main()
