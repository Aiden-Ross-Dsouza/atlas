"""
FABLE5 Day 1.1 — settle-check analysis (read-only). Regenerates the corrected
distance comparison in research_audit/FABLE5_DAY1_RESULTS.md from the raw JSONLs.

The pass-through settled-SR is 0/20 in every arm (a floor at the 20 px threshold).
The informative comparison is the *distance* the block ends at:
  - pass-through success  -> settled_block_pos_diff  (position after 15 hold steps)
  - otherwise             -> block_pos_diff          (end-of-episode position)
plus the neither-succeeded subset, where BOTH arms use end-of-episode distance and
there is no drift-substitution confound.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

P0 = Path(__file__).resolve().parent.parent / "phase0_v3"

ARMS = {
    "nas2": ("c2_settle_baseline_nas2/baseline_R2.jsonl", "c2_settle_ln_act_nas2/ln_act_R2.jsonl"),
    "nas6": ("c2_settle_baseline_nas6/baseline_R2.jsonl", "c2_settle_ln_act_nas6/ln_act_R2.jsonl"),
}


def load(rel):
    return {json.loads(l)["episode"]: json.loads(l) for l in (P0 / rel).read_text().splitlines() if l.strip()}


def end_distance(r):
    return r["settled_block_pos_diff"] if r.get("passthrough_success") else r["block_pos_diff"]


def main():
    out = {}
    for tag, (bf, cf) in ARMS.items():
        b, c = load(bf), load(cf)
        eps = sorted(set(b) & set(c))
        mism = sum(abs(b[e]["init_block_pos_diff"] - c[e]["init_block_pos_diff"]) > 1e-6 for e in eps)
        bd = np.array([end_distance(b[e]) for e in eps])
        cd = np.array([end_distance(c[e]) for e in eps])
        w = stats.wilcoxon(cd, bd)
        ns = [e for e in eps if not b[e].get("passthrough_success") and not c[e].get("passthrough_success")]
        bns = np.array([b[e]["block_pos_diff"] for e in ns])
        cns = np.array([c[e]["block_pos_diff"] for e in ns])
        wns = stats.wilcoxon(cns, bns)
        out[tag] = {
            "n": len(eps), "init_pairing_mismatches": mism,
            "passthrough_sr": {"chart": f"{sum(c[e]['success'] for e in eps)}/{len(eps)}",
                                "frozen": f"{sum(b[e]['success'] for e in eps)}/{len(eps)}"},
            "settled_sr": {"chart": f"{sum(c[e].get('settled_success', False) for e in eps)}/{len(eps)}",
                            "frozen": f"{sum(b[e].get('settled_success', False) for e in eps)}/{len(eps)}"},
            "end_distance_all": {
                "chart_mean": round(float(cd.mean()), 1), "frozen_mean": round(float(bd.mean()), 1),
                "paired_delta": round(float((cd - bd).mean()), 1),
                "chart_better": f"{int((cd < bd).sum())}/{len(eps)}",
                "wilcoxon_p": round(float(w.pvalue), 4),
                "bands": {f"<{t}px": {"chart": int((cd < t).sum()), "frozen": int((bd < t).sum())}
                          for t in (20, 30, 40, 60, 80)},
            },
            "end_distance_neither_succeeded": {
                "n": len(ns),
                "chart_mean": round(float(cns.mean()), 1) if ns else None,
                "frozen_mean": round(float(bns.mean()), 1) if ns else None,
                "paired_delta": round(float((cns - bns).mean()), 1) if ns else None,
                "chart_better": f"{int((cns < bns).sum())}/{len(ns)}" if ns else None,
                "wilcoxon_p": round(float(wns.pvalue), 4) if ns else None,
            },
        }
    (P0 / "c2_settle_distance_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
