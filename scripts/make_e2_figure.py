"""
scripts/make_e2_figure.py — F2a and the separability panel, regenerated from E2 logs.

Left panel is plan 7.3's F2a as specified: routing accuracy by cell and router.
Right panel replaces plots.two_panel's F2b (library growth), which belongs to E4
and renders as an empty axis for E2 — with the mechanism E2 actually established:
routing accuracy is governed by whether the chart's UMF advantage clears the
pre-registered hysteresis margin (m=0.05, CLAUDE.md 1.7).

Reads only the committed JSONL/summary artifacts, so the figure regenerates from
logs alone (CLAUDE.md 6, "release artifacts, not just code").

Usage:
    python scripts/make_e2_figure.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import atlas
from atlas.plots import COLUMN_WIDTH_IN, PALETTE

CELLS = ["A", "B", "C", "D"]
HYSTERESIS = 0.05  # CLAUDE.md 1.7, fixed


def load(run_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.load(open(run_dir / "e2_summary.json"))
    records = [json.loads(l) for l in open(run_dir / "e2_episodes.jsonl") if l.strip()]
    return summary, records


def shifted_condition_stats(records: list[dict], cell: str) -> tuple[float, float]:
    """(accuracy, relative UMF gap) for the SHIFTED condition of `cell`.

    Cell-level accuracy pools condition A (always R0) with condition B, which
    dilutes the only comparison that discriminates. The gap is
    (umf_c0 - umf_chart)/umf_c0: positive means the chart predicts better, and it
    is what the router's hysteresis margin is actually compared against.
    """
    sel = [r for r in records
           if r["cell"] == cell and r["condition"] == "B"
           and r.get("router") == "umf" and r["hit"] is not None]
    acc = sum(r["hit"] for r in sel) / len(sel)
    scored = [r["scores"] for r in sel if r.get("scores") and None not in r["scores"][:2]]
    c0 = st.mean(x[0] for x in scored)
    chart = st.mean(x[1] for x in scored)
    return acc, (c0 - chart) / c0


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate E2's F2a from logs.")
    ap.add_argument("--primary", type=Path, default=atlas.OUT_DIR / "e2_R2",
                    help="Run whose cells drive F2a (default: the R2 run).")
    ap.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e2_R2" / "F2a.pdf")
    args = ap.parse_args()

    summary, records = load(args.primary)
    acc = summary["routing_accuracy"]
    routers = [r for r in ("umf", "sdyn") if r in acc]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN * 2, 2.4))

    # ── F2a: routing accuracy by cell and router (plan 7.3) ───────────────────
    x = np.arange(len(CELLS))
    width = 0.36
    for i, router in enumerate(routers):
        vals = [acc[router].get(c, float("nan")) for c in CELLS]
        ax_a.bar(x + i * width, vals, width * 0.92, label=router,
                 color=PALETTE[router], alpha=0.9)
    # Two charts in the library, so an uninformative router sits at 0.5.
    ax_a.axhline(0.5, color="#666666", linestyle=":", linewidth=0.8, zorder=0)
    ax_a.annotate("chance", xy=(len(CELLS) - 0.35, 0.52), fontsize=6, color="#666666")
    ax_a.set_xticks(x + width / 2)
    ax_a.set_xticklabels([f"{c}\n{lbl}" for c, lbl in
                          zip(CELLS, ["same/same", "same/DYN", "APP/same", "APP/DYN"])])
    ax_a.set_ylim(0, 1.05)
    ax_a.set_ylabel("Routing accuracy")
    ax_a.set_title("F2a — appearance vs dynamics")
    ax_a.legend(frameon=False, fontsize=6)

    # ── Right: separability vs accuracy, against the hysteresis margin ────────
    runs = [("$c_{ln\\_act}$ R1", atlas.OUT_DIR / "e2_R1", "B"),
            ("$c_{lora4}$ R1", atlas.OUT_DIR / "e2_R1_lora4", "B"),
            ("$c_{ln\\_act}$ R2", atlas.OUT_DIR / "e2_R2", "B")]
    pts = []
    for label, d, cell in runs:
        if not (d / "e2_episodes.jsonl").exists():
            continue
        _, recs = load(d)
        a, gap = shifted_condition_stats(recs, cell)
        pts.append((label, gap, a))

    ax_b.axvline(HYSTERESIS, color="#666666", linestyle="--", linewidth=0.9, zorder=0)
    ax_b.annotate("hysteresis\nmargin m=0.05", xy=(HYSTERESIS * 1.15, 0.42),
                  fontsize=6, color="#666666")
    ax_b.axhline(0.5, color="#666666", linestyle=":", linewidth=0.8, zorder=0)
    for label, gap, a in pts:
        ax_b.scatter([gap], [a], s=42, color=PALETTE["umf"], zorder=3,
                     edgecolor="white", linewidth=0.8)
        ax_b.annotate(label, xy=(gap, a), xytext=(4, -9), textcoords="offset points",
                      fontsize=6)
    ax_b.set_xlim(0, max(0.17, max((g for _, g, _ in pts), default=0.16) * 1.15))
    ax_b.set_ylim(0.35, 1.0)
    ax_b.set_xlabel("Chart's relative UMF advantage under the shift")
    ax_b.set_ylabel("Routing accuracy (shifted condition)")
    ax_b.set_title("Separability sets routing accuracy")

    for ax in (ax_a, ax_b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out} and {args.out.with_suffix('.png')}")
    for label, gap, a in pts:
        print(f"  {label:<18} gap={gap:+.3f}  accuracy={a:.3f}")


if __name__ == "__main__":
    main()
