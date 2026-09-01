"""Figure 1 — the damping dose-response ladder. From phase0_v3/day1_ladder_analysis.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.loads(Path("phase0_v3/day1_ladder_analysis.json").read_text())["1C_dose_ladder"]
damps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for nas, style in [(6, "-o"), (2, "--s")]:
    pt = [d[f"damp{x}_nas{nas}"]["pass_through_SR"] for x in damps]
    st = [d[f"damp{x}_nas{nas}"]["settled_SR"] for x in damps]
    co = [d[f"damp{x}_nas{nas}"]["coast_median_px"] for x in damps]
    ax[0].plot(damps, pt, style, color="tab:blue", label=f"pass-through, nas={nas}")
    ax[0].plot(damps, st, style, color="tab:red", label=f"settled, nas={nas}")
    ax[1].plot(damps, co, style, color="tab:green", label=f"nas={nas}")

ax[0].set_xlabel("R2 damping"); ax[0].set_ylabel("success rate (n=20)")
ax[0].set_title("H1: pass-through SR plateaus, settled SR → 0"); ax[0].set_ylim(-0.03, 1.03)
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].set_xlabel("R2 damping"); ax[1].set_ylabel("median coast after agent stops (px)")
ax[1].set_title("H6: residual momentum, a dose-response"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"phase0_v3/day1_fig_ladder.{ext}", dpi=130)
print("wrote phase0_v3/day1_fig_ladder.{png,pdf}")
