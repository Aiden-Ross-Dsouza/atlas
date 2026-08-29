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
from datetime import datetime
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

    def keyed_outcomes_for(arm: str) -> dict[tuple, float]:
        return {_pair_key(ep): (1.0 if ep["success"] else 0.0) for ep in by_arm[arm]}

    baseline_outcomes = outcomes_for(baseline_arm) if baseline_arm in by_arm else None
    baseline_keyed = keyed_outcomes_for(baseline_arm) if baseline_arm in by_arm else None

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

        # FIX_SPEC.md B7: previously paired by EQUAL LENGTH
        # (`len(outcomes) == len(baseline_outcomes)`), not equal KEY SET.
        # With resume (a partial episodes.jsonl restarted and continued),
        # two arms can reach the same COUNT of completed episodes while
        # covering DIFFERENT (seed_run, global_episode_idx) keys -- the
        # sorted-by-key arrays would then silently pair episode i of one
        # arm against a DIFFERENT episode i of the other (same position,
        # different underlying episode), corrupting paired_bootstrap/
        # mcnemar_paired's pairing assumption without any error. Intersect
        # the actual key sets and pair ONLY on the intersection.
        if baseline_keyed is not None and arm != baseline_arm:
            arm_keyed = keyed_outcomes_for(arm)
            common_keys = sorted(set(arm_keyed) & set(baseline_keyed))
            n_arm, n_base = len(arm_keyed), len(baseline_keyed)
            if common_keys:
                paired_arm = np.array([arm_keyed[k] for k in common_keys])
                paired_base = np.array([baseline_keyed[k] for k in common_keys])
                assert len(paired_arm) == len(paired_base) == len(common_keys), (
                    "B7: intersected pairing arrays desynced -- must never happen."
                )
                delta_mean, (delta_lo, delta_hi) = paired_bootstrap(paired_arm, paired_base)
                delta_str = f"{delta_mean:.3f} [{delta_lo:.3f}, {delta_hi:.3f}]"
                if len(common_keys) < max(n_arm, n_base):
                    delta_str += f" (n={len(common_keys)}/{max(n_arm, n_base)} paired)"
                try:
                    p = mcnemar_paired(paired_arm.astype(bool), paired_base.astype(bool))
                    mcnemar_str = f"{p:.4f}"
                except ImportError:
                    mcnemar_str = "N/A (statsmodels missing)"
            else:
                delta_str = "N/A (zero overlapping (seed_run, global_episode_idx) keys)"
                mcnemar_str = "N/A"
        else:
            delta_str = "—" if arm == baseline_arm else "N/A (no baseline data)"
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


# Kinds that are the frozen-model reference row, not an adapter under test.
_T5_BASELINE_KEYS = ("c0", "baseline", "frozen", "identity")
# Smallest real adapter (ln_act) is ~10.7k trainable params. Anything with a
# positive trainable-param count below this is not a real chart — it is the
# pre-A11 / pre-rollout-fix `results.json` (e.g. atlas_out/e0/, which records
# `params: 26/12/69`). Rendering that as a table is the L2 bug this guard closes.
_T5_MIN_PLAUSIBLE_PARAMS = 1000


def _t5_resolve_baseline_umf(regime: str, kinds: dict, results: dict, e0_dir: Path) -> float | None:
    """Frozen-model reference UMF for a regime, tried in priority order:
    regime-level scalar key -> a per-kind baseline row -> {e0_dir}/frozen_baseline.json.
    E0 / E0' (run_e0.py) writes `results[regime]` as {ln_act, lora4, full} with no
    baseline entry, so without this the dUMF / "% of full" columns are dead."""
    for k in ("baseline_umf", "frozen_umf", "c0_umf"):
        if isinstance(kinds.get(k), (int, float)):
            return float(kinds[k])
    for bkey in _T5_BASELINE_KEYS:
        row = kinds.get(bkey)
        if isinstance(row, dict) and row.get("eval_umf") is not None:
            return float(row["eval_umf"])
    fb = e0_dir / "frozen_baseline.json"
    if fb.exists():
        d = json.loads(fb.read_text())
        for key in (regime, f"{regime}_umf", "umf", "eval_umf"):
            v = d.get(key) if isinstance(d, dict) else None
            if isinstance(v, (int, float)):
                return float(v)
    return None


def make_t5(e0_dir: Path, out_dir: Path, strict: bool = True) -> None:
    """T5 — E0 adapter capacity: Adapter x {Params, KB, Eval UMF, dUMF, Success,
    % of full}.

    Reads {e0_dir}/results.json (regime -> kind -> metrics). Params is
    `params_trainable` (falls back to `params`). dUMF is (frozen-model UMF -
    chart UMF); the frozen UMF is resolved by _t5_resolve_baseline_umf (regime
    scalar, a baseline kind row, or {e0_dir}/frozen_baseline.json). Success is
    read from {e0_dir}/{kind}_{regime}_summary.json etc. if present. Columns
    whose inputs are absent render as "-" rather than silently vanishing.

    Raises (FIX_SPEC.md A12: no silent no-op / no silent garbage):
      - FileNotFoundError if results.json is missing;
      - ValueError if results.json is pre-A11 / superseded data (trainable param
        counts below a real adapter's size — the atlas_out/e0/ `params: 26` case).
    With strict=False the ValueError is downgraded to a warning + skip, so
    `--all` against a stale default dir does not abort the other tables.
    """
    results_path = e0_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"T5 needs E0 capacity results: {results_path} not found. "
            f"Run scripts/run_e0.py (or point --e0-dir at a directory that has results.json)."
        )
    results = json.loads(results_path.read_text())

    # Guard: refuse to render a table from superseded / pre-A11 results.json.
    suspect = [
        (regime, kind, m.get("params_trainable", m.get("params")))
        for regime, kinds in results.items()
        for kind, m in kinds.items()
        if isinstance(m, dict) and kind not in _T5_BASELINE_KEYS
        and isinstance(m.get("params_trainable", m.get("params")), int)
        and 0 < m.get("params_trainable", m.get("params")) < _T5_MIN_PLAUSIBLE_PARAMS
    ]
    if suspect:
        msg = (
            f"T5: {results_path} looks like superseded / pre-A11 data — trainable "
            f"param counts below a real adapter ({_T5_MIN_PLAUSIBLE_PARAMS}): "
            f"{suspect[:3]}{'...' if len(suspect) > 3 else ''}. "
            f"Point --e0-dir at the current E0' results directory."
        )
        if strict:
            raise ValueError(msg)
        print(f"[skip] {msg}")
        return

    mtime = datetime.fromtimestamp(results_path.stat().st_mtime)
    provenance = f"_Source: `{results_path}` (mtime {mtime:%Y-%m-%d %H:%M})_"

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
        baseline_umf = _t5_resolve_baseline_umf(regime, kinds, results, e0_dir)
        full_gain = None
        full_row = kinds.get("full")
        if isinstance(full_row, dict) and baseline_umf is not None and full_row.get("eval_umf") is not None:
            full_gain = baseline_umf - float(full_row["eval_umf"])
        for kind, m in kinds.items():
            if kind in _T5_BASELINE_KEYS or not isinstance(m, dict):
                continue  # frozen reference row (or a regime-level scalar) — not an adapter
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

    _print_markdown_table(rows, out_dir / "T5.md", "T5 — E0 Adapter Capacity", note=provenance)


def _print_markdown_table(rows: list[dict], out_path: Path, title: str, note: str | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"{title}: no data.")
        return
    headers = list(rows[0].keys())
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
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
        # `--all` must not abort on a stale default --e0-dir; an explicit
        # `--table T5` raises so the caller sees the bad input.
        make_t5(args.e0_dir, args.e0_dir, strict=(args.table == "T5"))


if __name__ == "__main__":
    main()
