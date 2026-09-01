"""Fig 4 — the CEM-iteration ladder (paper §4 lead). From day2_controller_family.json.

Frozen c0 only, nas=2, R2 damping 0.5, same 20 paired tasks. One knob (CEM iterations).
Pass-through SR is flat; settled block-distance degrades 4.4x monotonically.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = json.loads(Path("phase0_v3/day2_controller_family.json").read_text())["PRIMARY_iteration_ladder"]
its = [1, 3, 10, 30]
x = list(range(4))
passsr = [L[str(i)]["pass_through_SR"] for i in its]
settled = [L[str(i)]["settled_dist_mean"] for i in its]
toward = [int(L[str(i)]["moved_toward_goal"].split("/")[0]) for i in its]
contacts = [L[str(i)]["contacts_mean"] for i in its]
iv = L["it1_vs_it30_paired"]

fig, ax1 = plt.subplots(figsize=(7.2, 4.9))
l1, = ax1.plot(x, passsr, "-o", color="tab:blue", lw=2.4, ms=9,
               label="pass-through SR  (deployed metric)  — flat")
ax1.set_ylabel("pass-through SR", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(0, 1.0)
ax1.axhline(sum(passsr) / 4, color="tab:blue", ls=":", alpha=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{i}\n({t}/20 toward goal,\n{c:.1f} contacts/ep)"
                     for i, t, c in zip(its, toward, contacts)], fontsize=8)
ax1.set_xlabel("CEM iterations  (planner optimisation budget)")

ax2 = ax1.twinx()
l2, = ax2.plot(x, settled, "-s", color="tab:red", lw=2.4, ms=9,
               label="mean settled block-distance (px)  — 4.4× worse, monotone")
ax2.set_ylabel("settled block-distance (px)  —  lower = better", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(0, 175)
# label each point; the last one is flipped to the left so it is not clipped by the
# right spine (matplotlib does not clip-detect annotations in offset-point space)
for i, (xi, s) in enumerate(zip(x, settled)):
    last = i == len(x) - 1
    ax2.annotate(f"{s:.0f}", (xi, s), color="tab:red", fontsize=9,
                 xytext=(-24 if last else 7, -4), textcoords="offset points")

ax1.legend(handles=[l1, l2], loc="center left", fontsize=8.5, framealpha=0.95)
ax1.set_title("§4 lead — optimising the planner harder monotonically destroys control;\n"
              "the deployed metric registers nothing\n"
              f"frozen c₀, nas=2, same 20 paired tasks ·  it1 vs it30: settled Δ{iv['settled_delta_mean']:.0f} px, "
              f"{iv['it1_closer']}, p={iv['wilcoxon_p_one_sided']:.0e} one-sided", fontsize=9.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"phase0_v3/day2_fig_iteration_ladder.{ext}", dpi=130)
print("wrote phase0_v3/day2_fig_iteration_ladder.{png,pdf}")
