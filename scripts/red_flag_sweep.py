"""
scripts/red_flag_sweep.py — FIX_SPEC.md D1: the "red-flag sweep" flagged in
RESULTS_AUDIT.md/NEXT_ACTIONS.md as never done as a DEDICATED pass (only
covered opportunistically while investigating specific claims). Runs three
checks over every JSONL under atlas_out/:

  1. Exact-zero variance across seeds/episodes for numeric fields that
     SHOULD vary run-to-run (e.g. init_block_pos_diff, umf values, costs) --
     a strong signature of the modal_e4.py-style "relabelled copies of one
     run" bug (SUBMISSION_PLAN.md A-ix), or of a seed never actually being
     threaded through.
  2. Per-arm/per-router/per-regime episode counts vs. what the directory
     name or a companion summary.json claims (paired-episode-count spec
     check) -- best-effort: flags any group whose n differs from the modal
     count for that file, and separately reports raw counts for manual
     comparison against each experiment's registered N.
  3. NaN/None sweep across every numeric field in every record, beyond the
     two already known (E0's sr_by_bucket overflow, A8; run_e0's early
     UMF=None chunks) -- reports the field name, count, and one example
     record's identifying keys (seed/episode/arm) per file.

This is a READ-ONLY diagnostic. It does not modify or judge any number; it
enumerates candidates for a human to inspect. Additions only per FIX_SPEC's
rules -- does not touch atlas/score.py, atlas/stats.py, or
scripts/run_e0_planning.py's planning loop.

Usage:
    python scripts/red_flag_sweep.py [--out research_audit/RED_FLAG_SWEEP.md]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import atlas


def _is_numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def check_nan_sweep(path: Path, records: list[dict]) -> list[str]:
    """Check 3: NaN/null sweep across every numeric-typed field."""
    findings = []
    field_bad_count: dict[str, int] = defaultdict(int)
    field_example: dict[str, dict] = {}
    for r in records:
        for k, v in r.items():
            is_bad = False
            if v is None:
                is_bad = True
            elif isinstance(v, float) and math.isnan(v):
                is_bad = True
            if is_bad:
                field_bad_count[k] += 1
                if k not in field_example:
                    ident = {kk: r.get(kk) for kk in
                              ("seed", "seed_run", "episode", "episode_idx",
                               "global_episode_idx", "arm", "router", "regime", "kind")
                              if kk in r}
                    field_example[k] = ident
    for k, count in field_bad_count.items():
        findings.append(
            f"  - field `{k}`: {count}/{len(records)} records are None/NaN "
            f"(example record: {field_example[k]})"
        )
    return findings


def check_zero_variance(path: Path, records: list[dict]) -> list[str]:
    """Check 1: exact-zero variance across records for fields that plausibly
    vary by seed/episode (init_*, seed-dependent metrics). Grouped by
    (arm/router/kind) if present, else pooled."""
    findings = []
    if len(records) < 3:
        return findings  # too few records for a variance signature to mean anything

    # Candidate fields: numeric, present on most records, name suggests
    # per-episode variation (init_*, umf*, cost*, dist*, seed*).
    VARY_HINTS = ("init_", "umf", "cost", "dist", "seed", "score", "loss")
    field_values: dict[str, list] = defaultdict(list)
    for r in records:
        for k, v in r.items():
            if _is_numeric(v) and any(h in k.lower() for h in VARY_HINTS):
                field_values[k].append(v)

    for k, vals in field_values.items():
        if len(vals) < 3:
            continue
        if len(set(vals)) == 1:
            findings.append(
                f"  - field `{k}`: EXACT-ZERO variance across {len(vals)} "
                f"records, all = {vals[0]!r} -- suspicious for a field whose "
                "name suggests per-episode/per-seed variation."
            )
    return findings


def check_episode_counts(path: Path, records: list[dict]) -> list[str]:
    """Check 2: per-arm/per-router/per-regime episode counts, reported raw
    for manual comparison against each experiment's registered N (this
    script does not know each experiment's spec N a priori)."""
    findings = []
    group_keys = [k for k in ("arm", "router", "kind", "regime", "seed_run")
                  if any(k in r for r in records[:5])]
    if not group_keys:
        findings.append(f"  - {len(records)} total records, no arm/router/kind/"
                         "regime/seed_run field to group by.")
        return findings
    counts: dict[tuple, int] = defaultdict(int)
    for r in records:
        key = tuple(r.get(k, "?") for k in group_keys)
        counts[key] += 1
    counts_str = ", ".join(f"{dict(zip(group_keys, k))}: n={v}"
                            for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])))
    findings.append(f"  - grouped by {group_keys}: {counts_str}")
    # Flag if group sizes are unequal (a paired design should give equal n
    # per arm/router unless episodes were deliberately dropped/resumed).
    ns = list(counts.values())
    if len(set(ns)) > 1:
        findings.append(
            f"  - UNEQUAL group sizes ({sorted(set(ns))}) -- if this file is "
            "meant to be a paired design (same seeds across arms/routers), "
            "this is a red flag for dropped/duplicated/resumed episodes."
        )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 red-flag sweep over atlas_out/*.jsonl")
    parser.add_argument("--out", type=Path,
                         default=atlas.ATLAS_HOME / "research_audit" / "RED_FLAG_SWEEP.md")
    parser.add_argument("--atlas-out", type=Path, default=atlas.OUT_DIR)
    args = parser.parse_args()

    jsonl_files = sorted(args.atlas_out.rglob("*.jsonl"))
    lines = [
        "# ATLAS — Red-flag sweep (FIX_SPEC.md D1)",
        "",
        f"Generated by `scripts/red_flag_sweep.py`. Scanned {len(jsonl_files)} "
        f"JSONL files under `{args.atlas_out}`. Read-only diagnostic -- flags "
        "candidates for human review, does not judge or fix anything.",
        "",
    ]

    any_findings = False
    for path in jsonl_files:
        rel = path.relative_to(args.atlas_out)
        records = _load_jsonl(path)
        if not records:
            lines.append(f"## `{rel}` -- EMPTY or unparseable, 0 records")
            lines.append("")
            any_findings = True
            continue

        nan_findings = check_nan_sweep(path, records)
        var_findings = check_zero_variance(path, records)
        count_findings = check_episode_counts(path, records)

        section = [f"## `{rel}` ({len(records)} records)", ""]
        if nan_findings:
            section.append("**NaN/null sweep:**")
            section.extend(nan_findings)
            any_findings = True
        if var_findings:
            section.append("**Exact-zero variance:**")
            section.extend(var_findings)
            any_findings = True
        section.append("**Episode counts:**")
        section.extend(count_findings)
        section.append("")
        lines.extend(section)

    if not any_findings:
        lines.insert(3, "**No NaN/zero-variance red flags found.** "
                         "(Episode-count groupings are reported for every "
                         "file below regardless, for manual spec comparison.)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Scanned {len(jsonl_files)} files. Report written to {args.out}")
    print(f"any_findings={any_findings}")


if __name__ == "__main__":
    main()
