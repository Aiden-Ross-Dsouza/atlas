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


def make_t2(e4_log: Path, out_dir: Path, baseline_arm: str = "frozen") -> None:
    """T2: Ablation ladder and recall table.

    Pairs on (seed_run, global_episode_idx) -- sorted exactly as
    run_e1.py::compute_t1 does (E3_E4_IMPLEMENTATION_PLAN.md §7a), so paired
    bootstrap / McNemar never compare misaligned episodes across arms. Never
    an unpaired test (CLAUDE.md §5).

    "First visit A" = segment_idx == 0 (the stream's opening R0 segment);
    "final revisit A" = segment_idx == 4 (S2's last R0 segment, index 4 of
    0..5 -- see atlas.streams.stream_s2).

    charts_committed / probes_rejected are read as max(...) over an arm's
    records (its cumulative counters only grow), NOT eps[-1:] -- the old
    eps[-1:] read is fragile if an arm's records interleave with other arms'
    in the JSONL (e.g. a resumed/merged multi-container Modal run), where the
    last record on disk for that arm is not necessarily the temporally last
    episode.

    Knock-away (block_pos_diff > init_block_pos_diff) is E0's pre-registered
    mechanism metric for damping (E0_RECOVERY_PLAN.md §0.5) -- reported here
    per arm as count and mean damage, derived from existing fields (no extra
    logging needed).
    """
    episodes = load_episodes(e4_log)
    by_arm: dict[str, list[dict]] = {}
    for ep in episodes:
        by_arm.setdefault(ep["arm"], []).append(ep)

    def _pair_key(ep: dict) -> tuple:
        return (ep.get("seed_run", 0), ep.get("global_episode_idx", 0))

    def outcomes_for(arm: str) -> np.ndarray:
        rows = sorted(by_arm[arm], key=_pair_key)
        return np.array([1.0 if ep["success"] else 0.0 for ep in rows])

    baseline_outcomes = outcomes_for(baseline_arm) if baseline_arm in by_arm else None

    rows = []
    for arm, eps in by_arm.items():
        outcomes = outcomes_for(arm)
        sr, (sr_lo, sr_hi) = success_rate_ci(outcomes)

        first_a = [ep["success"] for ep in eps if ep.get("segment_idx") == 0]
        final_a = [ep["success"] for ep in eps if ep.get("segment_idx") == 4]
        sr_first = float(np.mean(first_a)) if first_a else float("nan")
        sr_final = float(np.mean(final_a)) if final_a else float("nan")

        charts_committed = max((ep.get("charts_committed_cumulative", 0) for ep in eps), default=0)
        probes_rejected = max((ep.get("probes_rejected_cumulative", 0) for ep in eps), default=0)

        knock_away = [ep["block_pos_diff"] - ep["init_block_pos_diff"] for ep in eps
                      if ep.get("block_pos_diff") is not None and ep.get("init_block_pos_diff") is not None
                      and ep["block_pos_diff"] > ep["init_block_pos_diff"]]
        n_knock_away = len(knock_away)
        mean_damage = float(np.mean(knock_away)) if knock_away else float("nan")

        if baseline_outcomes is not None and arm != baseline_arm and len(outcomes) == len(baseline_outcomes):
            delta_mean, (delta_lo, delta_hi) = paired_bootstrap(outcomes, baseline_outcomes)
            delta_str = f"{delta_mean:.3f} [{delta_lo:.3f}, {delta_hi:.3f}]"
            try:
                p = mcnemar_paired(outcomes.astype(bool), baseline_outcomes.astype(bool))
                mcnemar_str = f"{p:.4f}"
            except ImportError:
                mcnemar_str = "N/A (statsmodels missing)"
        else:
            delta_str = "—" if arm == baseline_arm else "N/A (unpaired lengths)"
            mcnemar_str = "—" if arm == baseline_arm else "N/A"

        rows.append({
            "Arm": arm,
            "SR overall [CI]": f"{sr:.3f} [{sr_lo:.3f}, {sr_hi:.3f}]",
            "SR first visit A": f"{sr_first:.3f}",
            "SR final revisit A": f"{sr_final:.3f}",
            f"paired Δ vs {baseline_arm} [CI]": delta_str,
            "McNemar p": mcnemar_str,
            "Charts committed": str(charts_committed),
            "Probes rejected": str(probes_rejected),
            "Knock-aways": str(n_knock_away),
            "Mean damage": f"{mean_damage:.2f}" if mean_damage == mean_damage else "—",
        })

    _print_markdown_table(rows, out_dir / "T2.md", "T2 — E4 Ablation Ladder + Recall")


def make_t5(e0_dir: Path, out_dir: Path) -> None:
    """T5 — E0 adapter capacity: Adapter x {Params, KB, Eval UMF, dUMF, Success,
    % of full}.

    Reads {e0_dir}/results.json (regime -> kind -> metrics). dUMF is
    (frozen c0 UMF - chart UMF) when a c0 / baseline row is present in the same
    regime; Success is read from {e0_dir}/{kind}_{regime}_summary.json or
    {e0_dir}/planning_{kind}_{regime}.json if present. Columns whose inputs are
    absent render as "-" rather than silently vanishing. Raises if results.json
    itself is missing (FIX_SPEC.md A12: no silent no-op).
    """
    results_path = e0_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"T5 needs E0 capacity results: {results_path} not found. "
            f"Run scripts/run_e0.py (or point --e0-dir at a directory that has results.json)."
        )
    results = json.loads(results_path.read_text())

    def _success(kind: str, regime: str):
        for cand in (e0_dir / f"{kind}_{regime}_summary.json",
                     e0_dir / f"planning_{kind}_{regime}.json",
                     e0_dir / f"{kind}_{regime}.json"):
            if cand.exists():
                d = json.loads(cand.read_text())
                for key in ("success_rate", "sr", "planning_success"):
                    if key in d:
                        return float(d[key])
        return None

    rows = []
    for regime, kinds in results.items():
        baseline_umf = None
        for bkey in ("c0", "baseline", "frozen", "identity"):
            if bkey in kinds and kinds[bkey].get("eval_umf") is not None:
                baseline_umf = float(kinds[bkey]["eval_umf"])
        full_gain = None
        if "full" in kinds and baseline_umf is not None and kinds["full"].get("eval_umf") is not None:
            full_gain = baseline_umf - float(kinds["full"]["eval_umf"])
        for kind, m in kinds.items():
            params = m.get("params_trainable", m.get("params"))
            umf = m.get("eval_umf")
            d_umf = (baseline_umf - float(umf)) if (baseline_umf is not None and umf is not None) else None
            pct_full = (100.0 * d_umf / full_gain) if (d_umf is not None and full_gain not in (None, 0)) else None
            sr = _success(kind, regime)
            rows.append({
                "Regime": regime,
                "Adapter": kind,
                "Params": f"{params:,}" if isinstance(params, int) else "—",
                "KB": f"{params * 4 / 1024:.1f}" if isinstance(params, int) else "—",
                "Eval UMF": f"{umf:.4f}" if umf is not None else "—",
                "ΔUMF": f"{d_umf:+.4f}" if d_umf is not None else "—",
                "Success": f"{sr:.3f}" if sr is not None else "—",
                "% of full": f"{pct_full:.0f}%" if pct_full is not None else "—",
            })

    _print_markdown_table(rows, out_dir / "T5.md", "T5 — E0 Adapter Capacity")


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
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))
    out_path.write_text(text, encoding="utf-8")
    print(f"\nSaved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate paper tables from logs.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--table", choices=["T1", "T2", "T5"])
    parser.add_argument("--e0-dir", type=Path, default=atlas.OUT_DIR / "e0",
                        help="Directory with E0 results.json for T5.")
    args = parser.parse_args()

    if not args.all and args.table is None:
        parser.print_help()
        return

    if args.all or args.table == "T1":
        make_t1(atlas.OUT_DIR / "e1" / "episodes.jsonl", atlas.OUT_DIR / "e1")
    if args.all or args.table == "T2":
        make_t2(atlas.OUT_DIR / "e4" / "episodes.jsonl", atlas.OUT_DIR / "e4")
    if args.all or args.table == "T5":
        make_t5(args.e0_dir, args.e0_dir)


if __name__ == "__main__":
    main()
