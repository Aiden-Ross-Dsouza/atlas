"""
FABLE5 Day 1.1-R — settle-check RE-RUN analysis (read-only).

The re-run applies the hold-position tail to EVERY episode (`--settle-steps 40`) and
records `settled_trace` (block_success at steps 1/5/15/30/40). So every episode now
carries one clean `settled_block_pos_diff` — no post-drift-vs-end-of-episode mixing.

Reports, per cadence:
  - pass-through SR vs settled SR (settled SR is a floor; kept only for the record)
  - paired Δ of settled block-distance (Wilcoxon + paired bootstrap CI), all episodes
    AND the neither-succeeded subset (identical treatment, no confound)
  - the drift curve from settled_trace (median block-distance at each checkpoint, per arm)
Plus:
  - R0 control: settled SR should stay ~= pass-through SR
  - falsification: settle2 nas6 baseline `settled_trace` step-15 value must reproduce the
    archived settle1 `c2_settle_baseline_nas6` `settled_block_pos_diff` for each success
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

P0 = Path(__file__).resolve().parent.parent / "phase0_v3"

R2 = {
    "nas2": ("c2_settle2_baseline_nas2/baseline_R2.jsonl", "c2_settle2_ln_act_nas2/ln_act_R2.jsonl"),
    "nas6": ("c2_settle2_baseline_nas6/baseline_R2.jsonl", "c2_settle2_ln_act_nas6/ln_act_R2.jsonl"),
}
R0_CONTROL = "c2_settle2_R0_baseline_nas6/baseline_R0.jsonl"


def load(rel):
    p = P0 / rel
    if not p.exists():
        return None
    return {json.loads(l)["episode"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}


def boot_ci(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def trace_at(r, step):
    for t in r.get("settled_trace", []):
        if t["step"] == step:
            return t["block_pos_diff"]
    return None


def main():
    out = {}
    for tag, (bf, cf) in R2.items():
        b, c = load(bf), load(cf)
        if b is None or c is None:
            out[tag] = "PENDING"
            continue
        eps = sorted(set(b) & set(c))
        mism = sum(abs(b[e]["init_block_pos_diff"] - c[e]["init_block_pos_diff"]) > 1e-6 for e in eps)
        bd = np.array([b[e]["settled_block_pos_diff"] for e in eps])
        cd = np.array([c[e]["settled_block_pos_diff"] for e in eps])
        d = cd - bd
        w = stats.wilcoxon(cd, bd)
        lo, hi = boot_ci(d)
        ns = [e for e in eps if not b[e]["passthrough_success"] and not c[e]["passthrough_success"]]
        bns = np.array([b[e]["settled_block_pos_diff"] for e in ns])
        cns = np.array([c[e]["settled_block_pos_diff"] for e in ns])
        wns = stats.wilcoxon(cns, bns) if len(ns) > 1 else None
        drift = {}
        for step in (1, 5, 15, 30, 40):
            fb = [trace_at(b[e], step) for e in eps]
            cc = [trace_at(c[e], step) for e in eps]
            if all(v is not None for v in fb + cc):
                drift[step] = {"frozen_median": round(float(np.median(fb)), 1),
                               "chart_median": round(float(np.median(cc)), 1)}
        out[tag] = {
            "n": len(eps), "init_pairing_mismatches": mism,
            "passthrough_sr": {"chart": int(sum(c[e]["success"] for e in eps)),
                                "frozen": int(sum(b[e]["success"] for e in eps))},
            "settled_sr": {"chart": int(sum(c[e]["settled_success"] for e in eps)),
                            "frozen": int(sum(b[e]["settled_success"] for e in eps))},
            "settled_dist_all": {
                "chart_mean": round(float(cd.mean()), 1), "frozen_mean": round(float(bd.mean()), 1),
                "paired_delta": round(float(d.mean()), 1), "delta_ci95": [round(lo, 1), round(hi, 1)],
                "chart_better": f"{int((d < 0).sum())}/{len(eps)}", "wilcoxon_p": round(float(w.pvalue), 4),
                "bands": {f"<{t}px": {"chart": int((cd < t).sum()), "frozen": int((bd < t).sum())}
                          for t in (20, 30, 40, 60, 80)},
            },
            "settled_dist_neither_succeeded": {
                "n": len(ns),
                "chart_mean": round(float(cns.mean()), 1) if ns else None,
                "frozen_mean": round(float(bns.mean()), 1) if ns else None,
                "paired_delta": round(float((cns - bns).mean()), 1) if ns else None,
                "chart_better": f"{int((cns < bns).sum())}/{len(ns)}" if ns else None,
                "wilcoxon_p": round(float(wns.pvalue), 4) if wns else None,
            },
            "drift_curve_median_px": drift,
        }

    # R0 control
    r0 = load(R0_CONTROL)
    if r0 is not None:
        eps = sorted(r0)
        out["R0_control_baseline_nas6"] = {
            "n": len(eps),
            "passthrough_sr": int(sum(r0[e]["success"] for e in eps)),
            "settled_sr": int(sum(r0[e]["settled_success"] for e in eps)),
            "note": "settled SR should be ~= pass-through SR if real successes survive the hold",
            "per_success": [{"ep": e, "pt_dist": round(r0[e]["block_pos_diff"], 1),
                             "settled_dist": round(r0[e]["settled_block_pos_diff"], 1),
                             "settled_ok": r0[e]["settled_success"]}
                            for e in eps if r0[e]["success"]],
        }
    else:
        out["R0_control_baseline_nas6"] = "PENDING"

    # falsification: settle2 nas6 baseline trace@15 vs archived settle1 settled_block_pos_diff
    new = load("c2_settle2_baseline_nas6/baseline_R2.jsonl")
    old = load("c2_settle_baseline_nas6/baseline_R2.jsonl")
    if new and old:
        checks = []
        for e in sorted(set(new) & set(old)):
            if old[e]["success"] and "settled_block_pos_diff" in old[e]:
                t15 = trace_at(new[e], 15)
                checks.append({"ep": e, "old_settled15": round(old[e]["settled_block_pos_diff"], 3),
                               "new_trace15": round(t15, 3) if t15 is not None else None,
                               "match": t15 is not None and abs(t15 - old[e]["settled_block_pos_diff"]) < 0.5})
        out["falsification_nas6_trace15_vs_archived"] = {
            "n": len(checks), "all_match": all(c["match"] for c in checks), "detail": checks}

    (P0 / "c2_settle2_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
