"""
plots.py — Figure generation: F1 (money plot), F2 (routing + library size), supplementary.

All figures:
  - Matplotlib only, colour-blind-safe palette
  - Saved as .pdf at 300 dpi
  - Fonts ≥ 8 pt at column width (3.5 in)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed; works on headless GPU servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Colour-blind-safe palette (Wong 2011) ─────────────────────────────────────
PALETTE = {
    "frozen":         "#000000",
    "adajepa":        "#E69F00",
    "persist":        "#56B4E9",
    "atlas_fixed":    "#009E73",
    "atlas_detect":   "#F0E442",
    "atlas":          "#0072B2",
    "oracle":         "#CC79A7",
    "random":         "#D55E00",
    "umf":            "#0072B2",
    "e1":             "#E69F00",
    "sdyn":           "#009E73",
}

COLUMN_WIDTH_IN = 3.5
FONT_SIZE_PT = 8

plt.rcParams.update({
    "font.size": FONT_SIZE_PT,
    "axes.titlesize": FONT_SIZE_PT,
    "axes.labelsize": FONT_SIZE_PT,
    "xtick.labelsize": FONT_SIZE_PT - 1,
    "ytick.labelsize": FONT_SIZE_PT - 1,
    "legend.fontsize": FONT_SIZE_PT - 1,
    "figure.dpi": 300,
    "pdf.fonttype": 42,   # TrueType in PDF
})


# ── F1 — Money plot ────────────────────────────────────────────────────────────

def money_plot(
    arm_episodes: dict[str, np.ndarray],   # arm_name → [N_episodes] binary success
    segment_boundaries: list[int],         # episode indices where regime changes
    regime_labels: list[str],              # label per segment
    commits: list[int],                    # episode indices where a chart was committed
    probes_rejected: list[int],            # episode indices where probe was rejected
    out_path: Path,
    window: int = 5,
) -> None:
    """
    F1: Rolling-mean success vs episode for all arms.
    Regime boundaries are dashed vertical lines.
    ★ marks chart commits, ○ marks rejected probes.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN * 2, 2.2))

    for arm_name, outcomes in arm_episodes.items():
        color = PALETTE.get(arm_name, "#888888")
        rolling = _rolling_mean(outcomes, window)
        ax.plot(rolling, label=arm_name, color=color, linewidth=1.0)

    for boundary in segment_boundaries[1:]:
        ax.axvline(x=boundary, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)

    # Annotate regime labels between boundaries.
    boundaries_with_end = segment_boundaries + [len(next(iter(arm_episodes.values())))]
    for i, label in enumerate(regime_labels):
        mid = (boundaries_with_end[i] + boundaries_with_end[i + 1]) / 2
        ax.text(mid, 0.02, label, ha="center", va="bottom",
                fontsize=FONT_SIZE_PT - 2, color="gray", transform=ax.get_xaxis_transform())

    # Commits (★) and rejected probes (○).
    for ep in commits:
        ax.annotate("★", xy=(ep, 0.95), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=7, color=PALETTE["atlas"])
    for ep in probes_rejected:
        ax.annotate("○", xy=(ep, 0.90), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=7, color=PALETTE["atlas_detect"])

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Success (rolling mean, w={window})")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", ncol=2, framealpha=0.5)
    ax.set_title("F1 — Continual stream S2")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ── F2 — Two-panel ────────────────────────────────────────────────────────────

def two_panel(
    routing_accuracy: dict[str, dict[str, float]],  # router → {cell: accuracy}
    library_size_per_arm: dict[str, np.ndarray],    # arm → [N_episodes] library size
    true_regime_count: int,
    out_path: Path,
) -> None:
    """
    F2a: 2×2 routing accuracy grouped bars (UMF vs S-dyn).
    F2b: Library size vs episode for arms 4/5/6.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN * 2, 2.0))

    # ── F2a ───────────────────────────────────────────────────────────────────
    cells = ["A", "B", "C", "D"]
    routers = list(routing_accuracy.keys())
    x = np.arange(len(cells))
    width = 0.35
    for idx, router in enumerate(routers):
        vals = [routing_accuracy[router].get(c, float("nan")) for c in cells]
        ax_a.bar(x + idx * width, vals, width, label=router,
                 color=PALETTE.get(router, "#888888"), alpha=0.85)
    ax_a.set_xticks(x + width / 2)
    ax_a.set_xticklabels(cells)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_ylabel("Routing accuracy")
    ax_a.set_title("F2a — Appearance vs dynamics")
    ax_a.legend()

    # ── F2b ───────────────────────────────────────────────────────────────────
    for arm_name, sizes in library_size_per_arm.items():
        color = PALETTE.get(arm_name, "#888888")
        ax_b.plot(sizes, label=arm_name, color=color, linewidth=1.0)
    ax_b.axhline(y=true_regime_count, color="black", linestyle=":", linewidth=0.8,
                 label=f"True regimes ({true_regime_count})")
    ax_b.set_xlabel("Episode")
    ax_b.set_ylabel("Library size")
    ax_b.set_title("F2b — Library growth")
    ax_b.legend()

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ── Supplementary plots ───────────────────────────────────────────────────────

def umf_traces(
    umf_per_chart: dict[str, np.ndarray],  # chart_id → [N_episodes] UMF
    selected_chart_per_episode: list[str],
    out_path: Path,
) -> None:
    """S1 — UMF traces across the stream; selected chart shaded."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN * 2, 2.0))
    for chart_id, vals in umf_per_chart.items():
        ax.plot(vals, label=chart_id, linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("UMF")
    ax.set_title("S1 — UMF traces")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def crosspolicy(
    M: np.ndarray,     # [K, K] UMF matrix
    chart_labels: list[str],
    out_path: Path,
) -> None:
    """S2 — Cross-policy UMF heatmap (column-normalised)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    K = M.shape[0]
    col_min = np.nanmin(M, axis=0, keepdims=True)
    col_max = np.nanmax(M, axis=0, keepdims=True)
    M_norm = (M - col_min) / (col_max - col_min + 1e-8)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN))
    im = ax.imshow(M_norm, cmap="Blues_r", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(chart_labels, rotation=45, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(chart_labels)
    ax.set_xlabel("Plans from chart j")
    ax.set_ylabel("Evaluated by chart i")
    ax.set_title("S2 — Cross-policy (col-norm)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def umf_vs_sr(
    umf_vals: np.ndarray,
    sr_vals: np.ndarray,
    out_path: Path,
) -> None:
    """S3 — UMF vs success rate scatter with Kendall τ."""
    from scipy.stats import kendalltau
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tau, p = kendalltau(umf_vals, sr_vals)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN))
    ax.scatter(umf_vals, sr_vals, s=10, alpha=0.6, color=PALETTE["umf"])
    ax.set_xlabel("UMF")
    ax.set_ylabel("Success rate")
    ax.set_title(f"S3 — UMF vs SR  (Kendall τ = {tau:.2f}, p = {p:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ── Utility ───────────────────────────────────────────────────────────────────

def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean with full padding at the start."""
    result = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = x[start:i + 1].mean()
    return result
