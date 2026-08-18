"""
scripts/run_e0.py — E0: Adapter capacity experiment.

Fine-tunes {ln_act, lora4, full} × {R1, R2} = 6 charts offline.
Evaluates UMF reduction and planning success in-regime.

Usage:
    python scripts/run_e0.py
    python scripts/run_e0.py --kinds ln_act lora4 --regimes R1 --steps 500  # quick test

Output:
    atlas_out/e0/chart_{kind}_{regime}.pt
    atlas_out/e0/loss_{kind}_{regime}.json
    atlas_out/e0/results.json   (UMF and success per kind × regime)
    atlas_out/e0/results.md     (T5 supplementary table)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import atlas
from atlas.chart import Chart, ChartKind
from atlas.harness import run_e0_finetune, log_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="E0: Adapter capacity fine-tune.")
    parser.add_argument("--kinds", nargs="+", default=["ln_act", "lora4", "full"],
                        choices=["ln_act", "lora4", "full"])
    parser.add_argument("--regimes", nargs="+", default=["R1", "R2"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e0")
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

    print(f"\nE0: {len(args.kinds)} kinds × {len(args.regimes)} regimes "
          f"= {len(args.kinds) * len(args.regimes)} fine-tunes")
    print(f"Output: {args.out}")

    results: dict = {}
    for regime in args.regimes:
        results[regime] = {}
        print(f"\n── Regime {regime} ─────────────────────────────────────────────")

        # Load regime trajectories from the dataset.
        # This stub requires jepa-wms dataset loading to be integrated.
        print(f"  Loading trajectories for regime {regime}...")
        raise NotImplementedError(
            f"run_e0.py: integrate jepa-wms dataset loading for regime {regime}. "
            "Load the pusht_noise dataset from DATA_DIR, apply the PhysicsRegime wrapper "
            "for regime-specific rollouts, encode observations with model.encoder, "
            "and pass the list of {encoder_output, actions} dicts to run_e0_finetune(). "
            "See implementation plan §7.1 and harness.run_e0_finetune()."
        )

    # The result table (T5) would be printed and saved here after integration.


if __name__ == "__main__":
    main()
