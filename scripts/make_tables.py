"""
scripts/make_tables.py — Regenerate all paper tables from episode logs.

Usage:
    python scripts/make_tables.py --all
    python scripts/make_tables.py --table T1
    python scripts/make_tables.py --table T2

Tables:
  T1  E1 routing:   Router × {SR, routing acc., oracle gap, normalised recovery}
  T2  E4 ladder:    Arm × {SR, SR first visit, SR final revisit, Δ [CI], McNemar p, charts committed, probes rejected}
  T5  E0 capacity:  Adapter × {Params, KB, ΔUMF, Success, % of full}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import atlas
from atlas.stats import normalised_recovery, paired_bootstrap, mcnemar_paired, success_rate_ci


def load_episodes(log_path: Path) -> list[dict]:
    """Load all episode records from a JSONL log file."""
    if not log_path.exists():
        raise FileNotFoundError(
            f"Episode log not found: {log_path}\n"
            f"Run the corresponding experiment script first."
        )
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


def make_t1(e1_log: Path, out_dir: Path) -> None:
    """T1: Routing comparison table."""
    episodes = load_episodes(e1_log)
    # Group by router.
    by_router: dict[str, list[dict]] = {}
    for ep in episodes:
        by_router.setdefault(ep["router"], []).append(ep)

    if "random" not in by_router or "oracle_id" not in by_router:
        raise ValueError(
            "T1 requires 'random' and 'oracle_id' router results in the episode log. "
            "Ensure run_e1.py was run with all 5 routers."
        )

    sr_random = np.mean([ep["success"] for ep in by_router["random"]])
    sr_oracle = np.mean([ep["success"] for ep in by_router["oracle_id"]])

    rows = []
    for router, eps in by_router.items():
        outcomes = np.array([ep["success"] for ep in eps])
        sr, (sr_lo, sr_hi) = success_rate_ci(outcomes)
        routing_acc = np.mean([ep.get("routing_correct", float("nan")) for ep in eps])
        oracle_gap = sr_oracle - sr
        nr = normalised_recovery(sr, sr_oracle, sr_random)
        rows.append({
            "Router": router,
            "SR": f"{sr:.3f} [{sr_lo:.3f}, {sr_hi:.3f}]",
            "Routing acc.": f"{routing_acc:.3f}",
            "Oracle gap": f"{oracle_gap:.3f}",
            "Norm. recovery": f"{nr:.3f}" if nr is not None else "—",
        })

    _print_markdown_table(rows, out_dir / "T1.md", "T1 — E1 Routing Comparison")


def make_t2(e4_log: Path, out_dir: Path) -> None:
    """T2: Ablation ladder and recall table."""
    episodes = load_episodes(e4_log)
    by_arm: dict[str, list[dict]] = {}
    for ep in episodes:
        by_arm.setdefault(ep["arm"], []).append(ep)

    rows = []
    for arm, eps in by_arm.items():
        outcomes = np.array([ep["success"] for ep in eps])
        sr, _ = success_rate_ci(outcomes)

        # First visit A = segment 0 (regime R0), final revisit A = segment 4.
        first_a = [ep["success"] for ep in eps if ep.get("segment_idx") == 0]
        final_a = [ep["success"] for ep in eps if ep.get("segment_idx") == 4]

        sr_first = float(np.mean(first_a)) if first_a else float("nan")
        sr_final = float(np.mean(final_a)) if final_a else float("nan")

        charts_committed = sum(ep.get("charts_committed_cumulative", 0) for ep in eps[-1:])
        probes_rejected = sum(ep.get("probes_rejected_cumulative", 0) for ep in eps[-1:])

        rows.append({
            "Arm": arm,
            "SR overall": f"{sr:.3f}",
            "SR first visit A": f"{sr_first:.3f}",
            "SR final revisit A": f"{sr_final:.3f}",
            "Charts committed": str(charts_committed),
            "Probes rejected": str(probes_rejected),
        })

    _print_markdown_table(rows, out_dir / "T2.md", "T2 — E4 Ablation Ladder + Recall")


def _print_markdown_table(rows: list[dict], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"{title}: no data.")
        return
    headers = list(rows[0].keys())
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    text = "\n".join(lines)
    print(text)
    out_path.write_text(text)
    print(f"\nSaved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate paper tables from logs.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--table", choices=["T1", "T2", "T5"])
    args = parser.parse_args()

    if not args.all and args.table is None:
        parser.print_help()
        return

    if args.all or args.table == "T1":
        make_t1(atlas.OUT_DIR / "e1" / "episodes.jsonl", atlas.OUT_DIR / "e1")
    if args.all or args.table == "T2":
        make_t2(atlas.OUT_DIR / "e4" / "episodes.jsonl", atlas.OUT_DIR / "e4")


if __name__ == "__main__":
    main()
