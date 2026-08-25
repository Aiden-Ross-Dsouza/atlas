"""
scripts/make_figures.py — Regenerate all paper figures from logs.

Usage:
    python scripts/make_figures.py --all
    python scripts/make_figures.py --fig F1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import atlas
from atlas.plots import money_plot, two_panel, umf_traces, crosspolicy, umf_vs_sr


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Log file not found: {path}\n"
            f"Run the corresponding experiment script first."
        )
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def make_f1(e4_log: Path, out_dir: Path, summary_path: Path | None = None) -> None:
    """F1 — Money plot: rolling success vs episode for all 7 arms.

    Segment boundaries/regime labels are read from the run's e4_summary.json
    (episodes_per_segment, segment_regimes) rather than hardcoded 20-episode/
    6-segment/R0-R1 defaults (E3_E4_IMPLEMENTATION_PLAN.md §7b) — falls back
    to those defaults only if no summary is found, so a 10-episode R0/R2 run
    (or any other --episodes/--segment-regimes combination) still plots
    correctly.
    """
    episodes = load_jsonl(e4_log)
    arms = sorted(set(ep["arm"] for ep in episodes))

    arm_outcomes: dict[str, np.ndarray] = {}
    for arm in arms:
        arm_eps = [ep for ep in episodes if ep["arm"] == arm]
        arm_eps.sort(key=lambda e: e.get("global_episode_idx", 0))
        arm_outcomes[arm] = np.array([ep["success"] for ep in arm_eps], dtype=float)

    if summary_path is None:
        summary_path = e4_log.parent / "e4_summary.json"
    eps_per_seg = 20
    regime_a, regime_b = "R0", "R1"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        eps_per_seg = summary.get("episodes_per_segment", eps_per_seg)
        seg_regimes = summary.get("segment_regimes")
        if seg_regimes and len(seg_regimes) == 2:
            regime_a, regime_b = seg_regimes
    else:
        # Fall back to inferring episodes-per-segment from the data itself
        # (segment_idx==0's episode count), rather than a hardcoded 20.
        seg0_eps = [ep for ep in episodes if ep.get("arm") == arms[0] and ep.get("segment_idx") == 0]
        if seg0_eps:
            eps_per_seg = len(seg0_eps)

    boundaries = [i * eps_per_seg for i in range(6)]
    regime_labels = [regime_a, regime_b, regime_a, regime_b, regime_a, regime_b]

    # Extract commit/reject episode indices from atlas arm.
    atlas_eps = [ep for ep in episodes if ep["arm"] == "atlas"]
    commits = [ep.get("global_episode_idx", 0) for ep in atlas_eps
               if ep.get("probe_outcome") == "committed"]
    probes_rejected = [ep.get("global_episode_idx", 0) for ep in atlas_eps
                       if ep.get("probe_outcome", "").startswith("rejected")]

    out_dir.mkdir(parents=True, exist_ok=True)
    money_plot(arm_outcomes, boundaries, regime_labels, commits, probes_rejected,
               out_dir / "F1_money_plot.pdf")
    print(f"F1 saved: {out_dir / 'F1_money_plot.pdf'}")


def make_f2(e2_log: Path, e4_log: Path, out_dir: Path) -> None:
    """F2 — Two-panel: routing accuracy (2×2) + library size vs episode."""
    # F2a: routing accuracy from E2.
    e2_eps = load_jsonl(e2_log)
    cells = ["A", "B", "C", "D"]
    routers = sorted(set(ep["router"] for ep in e2_eps))
    routing_accuracy: dict[str, dict[str, float]] = {r: {} for r in routers}
    for router in routers:
        for cell in cells:
            cell_eps = [ep for ep in e2_eps if ep["router"] == router and ep["cell"] == cell]
            if cell_eps:
                routing_accuracy[router][cell] = float(np.mean(
                    [ep.get("routing_correct", 0) for ep in cell_eps]
                ))

    # F2b: library size from E4.
    e4_eps = load_jsonl(e4_log)
    expansion_arms = ["atlas_fixed", "atlas_detect", "atlas"]
    lib_sizes: dict[str, np.ndarray] = {}
    for arm in expansion_arms:
        arm_eps = [ep for ep in e4_eps if ep["arm"] == arm]
        arm_eps.sort(key=lambda e: e.get("global_episode_idx", 0))
        if arm_eps:
            lib_sizes[arm] = np.array([ep.get("library_size", 1) for ep in arm_eps])

    out_dir.mkdir(parents=True, exist_ok=True)
    two_panel(routing_accuracy, lib_sizes, true_regime_count=2, out_path=out_dir / "F2.pdf")
    print(f"F2 saved: {out_dir / 'F2.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate paper figures from logs.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fig", choices=["F1", "F2"])
    args = parser.parse_args()

    if not args.all and args.fig is None:
        parser.print_help()
        return

    out_dir = atlas.OUT_DIR / "figures"

    if args.all or args.fig == "F1":
        make_f1(atlas.OUT_DIR / "e4" / "episodes.jsonl", out_dir)

    if args.all or args.fig == "F2":
        make_f2(
            atlas.OUT_DIR / "e2" / "episodes.jsonl",
            atlas.OUT_DIR / "e4" / "episodes.jsonl",
            out_dir,
        )


if __name__ == "__main__":
    main()
