"""FINAL_FIVE_DAY_PLAN Day 1 — free local analyses 1.G.1 and 1.G.5.

Read-only. Recomputes from raw per-episode JSONL under phase0_v3/. No GPU, no
production code path. Writes phase0_v3/day1_free_analyses.json.

1.G.1  settle-length sensitivity: settled SR + settled block-distance at hold
       in {1,5,15,30,40} raw steps, per arm, from settled_trace.
1.G.5  termination timing: for pass-through successes, the raw step the criterion
       fires at and the unused budget (of 30), per arm.
"""
import json
import statistics as st
from pathlib import Path

P0 = Path("phase0_v3")

ARMS = {
    "R2_baseline_nas2": "c2_settle2_baseline_nas2/baseline_R2.jsonl",
    "R2_ln_act_nas2":   "c2_settle2_ln_act_nas2/ln_act_R2.jsonl",
    "R2_baseline_nas6": "c2_settle2_baseline_nas6/baseline_R2.jsonl",
    "R2_ln_act_nas6":   "c2_settle2_ln_act_nas6/ln_act_R2.jsonl",
    "R0_baseline_nas2": "c2_settle2_R0_baseline_nas2/baseline_R0.jsonl",
    "R0_baseline_nas6": "c2_settle2_R0_baseline_nas6/baseline_R0.jsonl",
    "dmp01_baseline_nas2": "c2_settle2_dmp01_baseline_nas2/baseline_R2.jsonl",
    "dmp01_baseline_nas6": "c2_settle2_dmp01_baseline_nas6/baseline_R2.jsonl",
}

HOLDS = [1, 5, 15, 30, 40]


def load(rel):
    p = P0 / rel
    if not p.exists():
        return None
    return [json.loads(l) for l in p.open()]


def trace_at(row, hold):
    """block_pos_diff and success at the settle checkpoint nearest <= hold.
    settled_trace checkpoints are 1/5/15/30/40/45/N intersect [1,N]."""
    tr = row.get("settled_trace") or []
    pick = None
    for cp in tr:
        if cp["step"] <= hold:
            pick = cp
    if pick is None:
        return None, None
    # success = pos < 20px AND angle < 20deg (0.349 rad); trace 'success' already encodes it
    return pick["block_pos_diff"], pick["success"]


out = {"1.G.1_settle_length_sensitivity": {}, "1.G.5_termination_timing": {}}

for name, rel in ARMS.items():
    rows = load(rel)
    if rows is None:
        out["1.G.1_settle_length_sensitivity"][name] = "MISSING"
        continue
    n = len(rows)

    # ---- 1.G.1 ----
    g1 = {"n": n, "pass_through_SR": sum(r["success"] for r in rows) / n}
    for h in HOLDS:
        dists, succ = [], 0
        miss = 0
        for r in rows:
            d, s = trace_at(r, h)
            if d is None:
                miss += 1
                continue
            dists.append(d)
            succ += bool(s)
        g1[f"hold_{h}"] = {
            "settled_SR": succ / n,
            "settled_dist_mean": round(st.mean(dists), 2) if dists else None,
            "settled_dist_median": round(st.median(dists), 2) if dists else None,
            "n_with_trace": len(dists),
        }
    out["1.G.1_settle_length_sensitivity"][name] = g1

    # ---- 1.G.5 ----
    fire_steps = [r["success_at_step"] for r in rows
                  if r.get("passthrough_success") and r.get("success_at_step") is not None]
    g5 = {
        "n_passthrough_success": len(fire_steps),
        "fire_step_mean": round(st.mean(fire_steps), 2) if fire_steps else None,
        "fire_step_median": round(st.median(fire_steps), 1) if fire_steps else None,
        "budget_unused_mean": round(30 - st.mean(fire_steps), 2) if fire_steps else None,
        "fire_steps": sorted(fire_steps),
    }
    out["1.G.5_termination_timing"][name] = g5

(P0 / "day1_free_analyses.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
