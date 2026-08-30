"""
scripts/c2_planning_screen.py — P0G_FIX_PLAN §4.5 C-2: the catastrophe screen.

Compares a chart arm's planning episodes against an already-run paired baseline
arm (same seeds, same config), reports the discordant-pair split (not just two
rates), and evaluates the two pre-registered RED FLAGS. Pure analysis — reads
two JSONLs, uses atlas/stats.py unmodified, no GPU / no Modal.

    python scripts/c2_planning_screen.py \\
        --chart-jsonl  phase0_v3/c2_p0g_R2/ln_act_R2.jsonl \\
        --baseline-jsonl phase0_v3/p0c/p0c_it10_baseline_R2.jsonl \\
        --out phase0_v3/c2_p0g_R2/c2_screen_summary.json

**The power statement travels with the result, always** (P0G_FIX_PLAN §4.5,
§6-5): n=20 paired detects roughly a 25–30 pp effect. Given N1's well-powered
null (44/100 vs 43/100), the expected real effect here is ≈ 0. **A clean C-2 is
"not catastrophic", NEVER "works".** Do not let a null here be written up as
support — that is the exact CLAUDE.md §1.8 violation the whole audit exists to
prevent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from atlas.stats import mcnemar_paired, paired_bootstrap  # noqa: E402 -- unmodified


def _load(path: Path) -> dict[int, dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    by_ep = {}
    for r in rows:
        if r["episode"] in by_ep:
            raise ValueError(f"{path}: duplicate episode {r['episode']}")
        by_ep[r["episode"]] = r
    return by_ep


def main() -> None:
    p = argparse.ArgumentParser(description="C-2 catastrophe screen.")
    p.add_argument("--chart-jsonl", type=Path, required=True)
    p.add_argument("--baseline-jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--catastrophe-sr-threshold", type=int, default=5,
                   help="RED FLAG 1: chart successes <= this (of the paired n) fires it. "
                        "Default 5 -- '>=25pp below the 10/20 baseline', the lora4xR1 "
                        "signature (4/10 vs 8/10).")
    args = p.parse_args()

    chart = _load(args.chart_jsonl)
    base = _load(args.baseline_jsonl)

    common = sorted(set(chart) & set(base))
    if not common:
        raise ValueError("No overlapping episode indices between the two arms.")
    if set(chart) != set(base):
        print(f"  [WARN] episode sets differ -- chart {sorted(set(chart)-set(base))}, "
              f"baseline {sorted(set(base)-set(chart))}; pairing on the {len(common)} common only.",
              flush=True)

    c_succ = np.array([bool(chart[e]["success"]) for e in common])
    b_succ = np.array([bool(base[e]["success"]) for e in common])
    n = len(common)

    c_sr, b_sr = int(c_succ.sum()), int(b_succ.sum())
    delta, (lo, hi) = paired_bootstrap(c_succ.astype(float), b_succ.astype(float))
    try:
        pval = mcnemar_paired(c_succ, b_succ)
    except ImportError:
        pval = None

    # Discordant-pair split (the thing the plan says to report, not just two rates)
    n_10 = int((c_succ & ~b_succ).sum())   # chart succeeds, baseline fails
    n_01 = int((~c_succ & b_succ).sum())   # baseline succeeds, chart fails
    n_11 = int((c_succ & b_succ).sum())
    n_00 = int((~c_succ & ~b_succ).sum())

    # RED FLAG 2: the mechanism statistic. R2's failure mode is OVERSHOOT, so a
    # chart that lowers UMF but leaves overshoot untouched has (per the project's
    # own theory) learned nothing that matters. "knock-away" = final block
    # further from goal than it started.
    def knockaways_and_finaldist(arm: dict) -> tuple[int, float]:
        ka = sum(1 for e in common
                 if arm[e]["block_pos_diff"] > arm[e]["init_block_pos_diff"])
        mean_final = float(np.mean([arm[e]["block_pos_diff"] for e in common]))
        return ka, mean_final

    c_ka, c_final = knockaways_and_finaldist(chart)
    b_ka, b_final = knockaways_and_finaldist(base)

    rf1 = c_sr <= args.catastrophe_sr_threshold
    # "do NOT decrease" (plan's literal wording) -> >=. Fires when the chart
    # fails to measurably reduce overshoot -- i.e. lowered UMF but left R2's
    # actual failure mode untouched. (A self-test against identical inputs
    # trivially fires this: the "chart" IS the baseline, so nothing decreased.
    # That's correct, not a bug -- RF2 is a "did it demonstrably help" gate.)
    rf2 = (c_ka >= b_ka) and (c_final >= b_final)

    summary = {
        "n_paired": n, "episodes": common,
        "chart_sr": f"{c_sr}/{n}", "baseline_sr": f"{b_sr}/{n}",
        "delta_sr": delta, "delta_sr_ci95": [lo, hi], "mcnemar_p": pval,
        "discordant_split": {
            "chart_succeeds_baseline_fails (n_10)": n_10,
            "baseline_succeeds_chart_fails (n_01)": n_01,
            "both_succeed (n_11)": n_11, "both_fail (n_00)": n_00,
        },
        "mechanism_overshoot": {
            "chart_knockaways": c_ka, "baseline_knockaways": b_ka,
            "chart_mean_final_block_dist_px": c_final,
            "baseline_mean_final_block_dist_px": b_final,
        },
        "RED_FLAG_1_catastrophic_sr": bool(rf1),
        "RED_FLAG_2_mechanism_unmoved": bool(rf2),
        "verdict": ("BLOCK -- red flag fired" if (rf1 or rf2)
                    else "not catastrophic (NOT 'works' -- see power statement)"),
        "power_statement": (
            f"n={n} paired detects ~25-30 pp. Given N1's well-powered null "
            f"(44/100 vs 43/100) the expected effect is ~0. This is a CATASTROPHE "
            f"SCREEN, not an efficacy test. A null means 'not broken', never 'works'."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== C-2 catastrophe screen ({n} paired episodes) ===")
    print(f"  chart SR      : {c_sr}/{n}")
    print(f"  baseline SR   : {b_sr}/{n}")
    print(f"  delta SR      : {delta:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  McNemar p={pval}")
    print(f"  discordant    : chart-only {n_10}, baseline-only {n_01}  (both {n_11}, neither {n_00})")
    print(f"  overshoot     : knock-aways chart {c_ka} vs baseline {b_ka}; "
          f"mean final dist chart {c_final:.1f}px vs baseline {b_final:.1f}px")
    print(f"  RED FLAG 1 (SR <= {args.catastrophe_sr_threshold}): {'FIRED' if rf1 else 'clear'}")
    print(f"  RED FLAG 2 (overshoot unmoved) : {'FIRED' if rf2 else 'clear'}")
    print(f"  VERDICT       : {summary['verdict']}")
    print(f"\n  {summary['power_statement']}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
