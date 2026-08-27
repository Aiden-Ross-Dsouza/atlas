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


def make_s1(e4_log: Path, out_dir: Path) -> None:
    """S1 — UMF traces across the stream, per chart, with the selected chart
    shaded. Needs per-episode per-chart UMF, logged by E4's ATLAS arm as
    `umf_all` (list) or `umf_trace`. FIX_SPEC.md A12: raise clearly if the
    required field is absent rather than emitting an empty figure."""
    episodes = load_jsonl(e4_log)
    atlas_eps = sorted((e for e in episodes if e.get("arm") == "atlas"),
                       key=lambda e: e.get("global_episode_idx", 0))
    series = None
    for key in ("umf_all", "umf_per_chart", "umf_trace"):
        if atlas_eps and key in atlas_eps[0] and atlas_eps[0][key] is not None:
            series = key
            break
    if series is None:
        raise ValueError(
            f"S1 needs per-chart UMF per episode (field 'umf_all'/'umf_per_chart'/'umf_trace') "
            f"in {e4_log}; none of the ATLAS-arm records carry it. Re-run E4 with per-chart UMF "
            f"logging enabled, or drop S1 from the figure set."
        )
    n_charts = max(len(e[series]) for e in atlas_eps if e.get(series))
    umf_per_chart = {
        f"c{c}": np.array([(e.get(series) or [np.nan] * n_charts)[c]
                           if c < len(e.get(series) or []) else np.nan
                           for e in atlas_eps], dtype=float)
        for c in range(n_charts)
    }
    selected = [str(e.get("selected_chart", e.get("current_idx", ""))) for e in atlas_eps]
    out_dir.mkdir(parents=True, exist_ok=True)
    umf_traces(umf_per_chart, selected, out_dir / "S1_umf_traces.pdf")
    print(f"S1 saved: {out_dir / 'S1_umf_traces.pdf'}")


def make_s2(matrix_json: Path, out_dir: Path) -> None:
    """S2 — cross-policy UMF heatmap. Needs a [K,K] UMF matrix, produced by
    atlas.harness.build_cross_policy_matrix (E5). Raise clearly if absent."""
    if not matrix_json.exists():
        raise FileNotFoundError(
            f"S2 needs a cross-policy UMF matrix at {matrix_json} "
            f"(atlas.harness.build_cross_policy_matrix / E5 output). Run E5 first, "
            f"or drop S2 from the figure set."
        )
    d = json.loads(matrix_json.read_text())
    M = np.asarray(d["matrix"] if "matrix" in d else d["M"], dtype=float)
    labels = d.get("chart_labels") or [f"c{i}" for i in range(M.shape[0])]
    out_dir.mkdir(parents=True, exist_ok=True)
    crosspolicy(M, labels, out_dir / "S2_crosspolicy.pdf")
    print(f"S2 saved: {out_dir / 'S2_crosspolicy.pdf'}")


def make_s3(planning_jsonl: Path, out_dir: Path) -> None:
    """S3 — UMF vs success-rate scatter with Kendall tau. Needs episode-level
    (umf_mean, success) pairs from a planning run. Raise clearly if absent."""
    if not planning_jsonl.exists():
        raise FileNotFoundError(
            f"S3 needs an episode planning log with per-episode 'umf_mean' + 'success' "
            f"at {planning_jsonl}. Point it at e.g. atlas_out/e0_planning_n100/ln_act_R2.jsonl."
        )
    rows = load_jsonl(planning_jsonl)
    pairs = [(r["umf_mean"], int(r["success"])) for r in rows if r.get("umf_mean") is not None]
    if not pairs:
        raise ValueError(f"S3: no rows in {planning_jsonl} carry a non-null 'umf_mean'.")
    umf_vals = np.array([p[0] for p in pairs], dtype=float)
    sr_vals = np.array([p[1] for p in pairs], dtype=float)
    out_dir.mkdir(parents=True, exist_ok=True)
    umf_vs_sr(umf_vals, sr_vals, out_dir / "S3_umf_vs_sr.pdf")
    print(f"S3 saved: {out_dir / 'S3_umf_vs_sr.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate paper figures from logs.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fig", choices=["F1", "F2", "S1", "S2", "S3"])
    parser.add_argument("--s2-matrix", type=Path,
                        default=atlas.OUT_DIR / "e5" / "cross_policy_matrix.json")
    parser.add_argument("--s3-log", type=Path,
                        default=atlas.OUT_DIR / "e0_planning_n100" / "ln_act_R2.jsonl")
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

    # S1-S3: an explicit --fig raises when its data is missing (no silent
    # no-op, FIX_SPEC.md A12); --all reports the failure and carries on so a
    # missing supplementary figure does not block F1/F2.
    for name, fn in (("S1", lambda: make_s1(atlas.OUT_DIR / "e4" / "episodes.jsonl", out_dir)),
                     ("S2", lambda: make_s2(args.s2_matrix, out_dir)),
                     ("S3", lambda: make_s3(args.s3_log, out_dir))):
        if args.fig == name:
            fn()
        elif args.all:
            try:
                fn()
            except (FileNotFoundError, ValueError) as e:
                print(f"{name} skipped ({type(e).__name__}): {e}")


if __name__ == "__main__":
    main()
