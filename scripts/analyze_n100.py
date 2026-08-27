"""
scripts/analyze_n100.py -- OPUS_REMAINING_TASKS.md Section A, zero-GPU
analysis on planning/cost-ranking logs already on disk. No Modal, no model
calls: pure pandas/scipy/statsmodels post-processing of existing JSONL/JSON.

Items covered (see OPUS_REMAINING_TASKS.md for the exact wording each
answers):
  A.1  Knock-aways + mean progress at N=100, both arms.
  A.2  SR by init-displacement bucket at N=100.
  A.3  Partial/stratified Kendall tau(UMF, success), controlling for
       init_block_pos_diff (and total_contacts).
  A.7  The "bridge" experiment: does cost-ranking's seed construction match
       the N=100 planning run's episode construction for seeds/episodes 0-9,
       and if so, do cost-ranking-inverted seeds correspond to failed
       planning episodes?
  A.8  Verify the training-size sweep's pairing (init_block_pos_diff exact
       match across e0_planning_n100 / e0_planning_sweep_60 / _sweep_100).

A.4/A.5/A.6 (degeneracy + regret/CI on cost-ranking data) live in
diagnose_cem_costs.py's own output once its --capture-iteration and
per-candidate persistence changes land (needs a re-run, not pure
post-processing of what's on disk today) -- see analyze_cost_ranking.py.

Usage:
    python scripts/analyze_n100.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import atlas


def load_episodes(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def knock_away_progress(df: pd.DataFrame) -> dict:
    """Knock-away definition matches make_tables.py::make_t2 exactly
    (block_pos_diff > init_block_pos_diff). Progress = init - final (positive
    = moved closer to goal, matching E0_RESULTS.md's five-arm N=20 table's
    'Mean progress' column sign)."""
    knock_away = df["block_pos_diff"] > df["init_block_pos_diff"]
    progress = df["init_block_pos_diff"] - df["block_pos_diff"]
    return {
        "n": len(df),
        "n_knock_away": int(knock_away.sum()),
        "frac_knock_away": float(knock_away.mean()),
        "mean_progress_px": float(progress.mean()),
    }


def sr_by_bucket(df: pd.DataFrame, edges=(0, 80, 120, 300)) -> pd.DataFrame:
    labels = [f"{edges[i]}-{edges[i+1]}px" for i in range(len(edges) - 1)]
    df = df.copy()
    df["bucket"] = pd.cut(df["init_block_pos_diff"], bins=edges, labels=labels, include_lowest=True)
    out = df.groupby("bucket", observed=True).agg(
        n=("success", "size"), sr=("success", "mean")
    ).reset_index()
    return out


def partial_kendall(df: pd.DataFrame, x_col: str = "umf_mean", y_col: str = "success",
                     controls: tuple[str, ...] = ("init_block_pos_diff", "total_contacts")) -> dict:
    """Residualize x and y on `controls` via OLS, then Kendall tau on the
    residuals -- a standard semipartial-correlation approach. y is binary
    (success); an OLS-on-binary residual (linear probability model) is a
    coarse but standard way to get a rank-comparable residual without
    assuming a link function. Also reports a stratified (tercile-binned)
    tau as a model-free cross-check."""
    import statsmodels.api as sm

    sub = df.dropna(subset=[x_col, y_col, *controls]).copy()
    X = sm.add_constant(sub[list(controls)].astype(float))

    x_resid = sm.OLS(sub[x_col].astype(float), X).fit().resid
    y_resid = sm.OLS(sub[y_col].astype(float), X).fit().resid
    partial_tau, partial_p = kendalltau(x_resid, y_resid)

    # Stratified: tercile bins of the primary control (init_block_pos_diff).
    sub["stratum"] = pd.qcut(sub[controls[0]], q=3, duplicates="drop")
    strat_rows = []
    for s, g in sub.groupby("stratum", observed=True):
        if len(g) < 5:
            continue
        tau, p = kendalltau(g[x_col], g[y_col])
        strat_rows.append({"stratum": str(s), "n": len(g), "tau": float(tau), "p": float(p)})

    unconditional_tau, unconditional_p = kendalltau(sub[x_col], sub[y_col])

    return {
        "n": len(sub),
        "unconditional_tau": float(unconditional_tau), "unconditional_p": float(unconditional_p),
        "partial_tau": float(partial_tau), "partial_p": float(partial_p),
        "stratified": strat_rows,
    }


def catastrophic_episodes(df: pd.DataFrame, k: int = 5) -> list[dict]:
    """Episodes with the largest |umf_mean| under failure or the largest
    umf/success mismatch -- eyeball candidates for 'is one bad episode
    carrying the correlation'."""
    sub = df.dropna(subset=["umf_mean"]).copy()
    sub["mismatch_score"] = sub["umf_mean"] * (1 - sub["success"].astype(int) * 2)
    top = sub.nlargest(k, "mismatch_score")[
        ["episode", "success", "umf_mean", "init_block_pos_diff", "block_pos_diff"]
    ]
    return top.to_dict("records")


def bridge_check(cost_ranking_json: Path, planning_jsonl: Path) -> dict:
    """A.7: compare init_block_pos_diff for cost-ranking's seeds 0-9 against
    planning episodes 0-9 (seed == episode index throughout run_e0_planning.py).
    If they match, cross-reference cost-ranking-inverted seeds (rho < 0)
    against planning failures."""
    cr = json.loads(cost_ranking_json.read_text())
    plan_df = load_episodes(planning_jsonl)
    plan_by_ep = plan_df.set_index("episode")

    rows = []
    for seed_entry in cr["per_seed"]:
        seed = seed_entry["seed"]
        init_state = np.array(seed_entry["init_state"])
        goal_state = np.array(seed_entry["goal_state"])
        cr_block_pos_diff = float(np.linalg.norm(goal_state[2:4] - init_state[2:4]))
        if seed not in plan_by_ep.index:
            continue
        plan_row = plan_by_ep.loc[seed]
        plan_block_pos_diff = float(plan_row["init_block_pos_diff"])
        match = abs(cr_block_pos_diff - plan_block_pos_diff) < 1e-3
        # rho for the (only) chart kind present besides baseline, if any.
        kinds = list(seed_entry["results"].keys())
        chart_kind = next((k for k in kinds if k != "baseline"), kinds[0])
        rho = seed_entry["results"][chart_kind]["spearman_rho"]
        rows.append({
            "seed": seed, "match": match,
            "cr_init_block_pos_diff": cr_block_pos_diff,
            "plan_init_block_pos_diff": plan_block_pos_diff,
            "cost_ranking_rho": rho,
            "planning_success": bool(plan_row["success"]),
        })
    return {"n_compared": len(rows), "n_matched": sum(r["match"] for r in rows), "rows": rows}


def verify_pairing(paths: dict[str, Path], key: str = "init_block_pos_diff") -> dict:
    """A.8: exact pairing check (same standard as the N=100 power confirmation
    section already used: max mismatch to 6 decimal places) across whichever
    of the given files share episode indices."""
    dfs = {name: load_episodes(p).set_index("episode")[key] for name, p in paths.items() if p.exists()}
    names = list(dfs.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = dfs[names[i]], dfs[names[j]]
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            diff = (a.loc[common] - b.loc[common]).abs()
            results[f"{names[i]} vs {names[j]}"] = {
                "n_common": len(common), "max_abs_diff_px": float(diff.max()),
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Section-A zero-GPU analysis on N=100 logs.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--n100-dir", type=Path, default=atlas.OUT_DIR / "e0_planning_n100")
    parser.add_argument("--sweep60-dir", type=Path, default=atlas.OUT_DIR / "e0_planning_sweep_60")
    parser.add_argument("--sweep100-dir", type=Path, default=atlas.OUT_DIR / "e0_planning_sweep_100")
    parser.add_argument("--cost-ranking-json", type=Path,
                         default=atlas.OUT_DIR / "cost_ranking_R2" /
                         "cost_ranking_R2_seeds0-1-2-3-4-5-6-7-8-9.json")
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "analysis_n100.json")
    args = parser.parse_args()

    baseline_path = args.n100_dir / "baseline_R2.jsonl"
    chart_path = args.n100_dir / "ln_act_R2.jsonl"
    baseline_df = load_episodes(baseline_path)
    chart_df = load_episodes(chart_path)

    report: dict = {}

    # A.1
    report["A1_knock_away_progress"] = {
        "baseline": knock_away_progress(baseline_df),
        "ln_act": knock_away_progress(chart_df),
    }

    # A.2
    report["A2_sr_by_bucket"] = {
        "baseline": sr_by_bucket(baseline_df).to_dict("records"),
        "ln_act": sr_by_bucket(chart_df).to_dict("records"),
    }

    # A.3
    report["A3_partial_stratified_kendall"] = {
        "baseline": partial_kendall(baseline_df),
        "ln_act": partial_kendall(chart_df),
    }
    report["A3_catastrophic_episodes"] = {
        "baseline": catastrophic_episodes(baseline_df),
        "ln_act": catastrophic_episodes(chart_df),
    }

    # A.7
    if args.cost_ranking_json.exists():
        report["A7_bridge"] = bridge_check(args.cost_ranking_json, chart_path)
    else:
        report["A7_bridge"] = {"skipped": f"{args.cost_ranking_json} not found"}

    # A.8
    report["A8_pairing_verification"] = verify_pairing({
        "n100_baseline": baseline_path,
        "n100_ln_act": chart_path,
        "sweep60_ln_act": args.sweep60_dir / "ln_act_R2.jsonl",
        "sweep100_ln_act": args.sweep100_dir / "ln_act_R2.jsonl",
    })

    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
