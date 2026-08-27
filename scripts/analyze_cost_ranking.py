"""
scripts/analyze_cost_ranking.py -- OPUS_REMAINING_TASKS.md A.4/A.5/A.6:
degeneracy + regret/top-tail metrics on diagnose_cem_costs.py output that
now persists raw per-candidate data (contacts, true_dist, costs).

A.4: contact fraction, min/median/max true_dist, rho restricted to the
     contact-making subset -- already computed per-seed by
     diagnose_cem_costs.py itself (see its "contact_fraction" etc. fields);
     this script just aggregates them across seeds for reporting.
A.5: regret = true_dist of the cost-argmin candidate minus the batch min
     true_dist; mean true_dist of the cost-ranked top-10 vs. the batch mean.
     Needs the raw per-candidate arrays this script reads directly.
A.6: per-seed mean rho as a proper CI -- diagnose_cem_costs.py already
     computes this (ci95_of_mean_seed_rho); reprinted here for convenience.

Usage:
    python scripts/analyze_cost_ranking.py --regime R0 --json atlas_out/cost_ranking_R0/cost_ranking_R0_seeds0-1-2-3-4-5-6-7-8-9.json
    python scripts/analyze_cost_ranking.py --regime R2 --json atlas_out/cost_ranking_R2_v2/cost_ranking_R2_seeds0-1-2-3-4-5-6-7-8-9.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(path: Path) -> dict:
    d = json.loads(path.read_text())
    kinds = d["kinds"]
    report: dict = {"regime": d["regime"], "kinds": kinds, "per_kind": {}}

    for kind in kinds:
        contact_fracs, regrets, top10_means, batch_means = [], [], [], []
        for seed_entry in d["per_seed"]:
            r = seed_entry["results"][kind]
            costs = np.array(r["costs"])
            true_dist = np.array(r["true_dist"])
            contact_fracs.append(r["contact_fraction"])

            argmin_idx = int(costs.argmin())
            regret = true_dist[argmin_idx] - true_dist.min()
            regrets.append(regret)

            top10_idx = np.argsort(costs)[:10]
            top10_means.append(true_dist[top10_idx].mean())
            batch_means.append(true_dist.mean())

        report["per_kind"][kind] = {
            "mean_contact_fraction": float(np.mean(contact_fracs)),
            "min_contact_fraction": float(np.min(contact_fracs)),
            "max_contact_fraction": float(np.max(contact_fracs)),
            "mean_regret_px": float(np.mean(regrets)),
            "median_regret_px": float(np.median(regrets)),
            "regret_per_seed": [float(r) for r in regrets],
            "mean_top10_true_dist_px": float(np.mean(top10_means)),
            "mean_batch_true_dist_px": float(np.mean(batch_means)),
            "top10_vs_batch_gap_px": float(np.mean(batch_means) - np.mean(top10_means)),
            "ci95_of_mean_seed_rho": d["pooled"][kind]["ci95_of_mean_seed_rho"],
            "mean_of_per_seed_rhos": d["pooled"][kind]["mean_of_per_seed_rhos"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
