"""
scripts/run_e2.py — E2: Appearance vs dynamics (2×2 routing accuracy).

Cells:
  A: same appearance, same dynamics (R0 vs R0)
  B: same appearance, different dynamics (R0 vs R1) ← decisive
  C: different appearance, same dynamics (R0 vs R0+colour) ← must not over-expand
  D: different appearance, different dynamics (R0 vs R1+colour) ← realistic

Usage:
    python scripts/run_e2.py --cells A B C D --routers umf sdyn --episodes 40 --seeds 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import atlas


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: Appearance vs dynamics 2×2.")
    parser.add_argument("--cells", nargs="+", default=["A", "B", "C", "D"],
                        choices=["A", "B", "C", "D"])
    parser.add_argument("--routers", nargs="+", default=["umf", "sdyn"],
                        choices=["umf", "sdyn"])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e2")
    args = parser.parse_args()

    print(f"E2: cells={args.cells} routers={args.routers}")

    # Cell definitions from e2.yaml:
    CELL_CONFIGS = {
        "A": dict(regime_a="R0", regime_b="R0", corruption_a="none", corruption_b="none"),
        "B": dict(regime_a="R0", regime_b="R1", corruption_a="none", corruption_b="none"),
        "C": dict(regime_a="R0", regime_b="R0", corruption_a="none", corruption_b="colour_change"),
        "D": dict(regime_a="R0", regime_b="R1", corruption_a="none", corruption_b="colour_change"),
    }

    raise NotImplementedError(
        "run_e2.py: integrate the jepa-wms planning eval loop with the regime/corruption wrappers.\n"
        "For each cell:\n"
        "  1. Build env_a and env_b using atlas.regimes.build_env()\n"
        "  2. Route chart per episode using the specified router\n"
        "  3. Record routing accuracy (did the correct chart get selected?)\n"
        "  4. For Cell C: assert charts_committed == 0 (no over-expansion)\n"
        "  5. Save results to atlas_out/e2/F2a.pdf via atlas.plots.two_panel()"
    )


if __name__ == "__main__":
    main()
