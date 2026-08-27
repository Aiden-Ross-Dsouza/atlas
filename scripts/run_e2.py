"""
scripts/run_e2.py — E2: Appearance vs dynamics (2x2 routing accuracy).

Cells (ATLAS_implementation_plan_v2.md 6.3):
  A control:            same appearance, same dynamics     (R0 vs R0)
  B decisive:           same appearance, differ dynamics   (R0 vs R1)
  C over-expansion:     differ appearance, same dynamics   (R0 vs R0+colour)
  D realistic:          differ appearance, differ dynamics (R0 vs R1+colour)

The two numbers that carry the experiment (plan 7.3) are ROUTING ACCURACY in
Cell B and CHARTS COMMITTED == 0 in Cell C. Neither depends on a chart being
good at planning — they measure the selector (C1) and the expansion verifier
(C2). That is why E2 stays informative after E0's capacity matrix failed
(E0_RESULTS.md), and it is the pivot plan 7.2 pre-registered for this exact
situation.

DELIBERATE DEVIATION from plan 7.3 — record it with any result: episodes here
are COLLECTED TRAJECTORIES, not CEM-planned episodes. Routing accuracy is a
property of UMF scoring on an observed chunk; the planner does not enter the
metric, and running CEM would multiply cost ~100x without changing anything the
metric reads. This is what makes E2 affordable with E0's spend already booked.

One routing decision per episode (each trajectory is scored as a single chunk),
so `--episodes 40 --seeds 3` gives 120 decisions per cell per condition.

Usage:
    python scripts/run_e2.py --cells A B C D --routers umf sdyn --episodes 40 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

import atlas
from atlas.chart import Chart
from atlas.expand import Expander, ExpansionConfig
from atlas.library import Library
from atlas.router import route
from atlas.score import compute_motion_gate

sys.path.insert(0, str(Path(__file__).parent))
from run_e0 import load_regime_trajectories  # noqa: E402

# Condition A is always the frozen base case (R0, uncorrupted), so its correct
# chart is always c0 (index 0). Only condition B varies per cell.
CELL_CONFIGS = {
    # regime_b=None -> filled from --dynamics-regime; corruption_b=None ->
    # filled from --corruption. Cells A/C are the same-dynamics cells, so their
    # regime_b stays pinned to R0 whatever --dynamics-regime says.
    "A": dict(regime_b="R0", corruption_b="none", correct_b=0),
    "B": dict(regime_b=None, corruption_b="none", correct_b=1),
    "C": dict(regime_b="R0", corruption_b=None, correct_b=0),
    "D": dict(regime_b=None, corruption_b=None, correct_b=1),
}


def build_library(charts_dir: Path, predictor, kind: str, regime: str) -> Library:
    """Library = {c0, chart_R1}. Index 0 is the identity chart (frozen
    predictor); index 1 is E0's dynamics chart — the one a correct router must
    select when, and only when, dynamics actually shifted."""
    library = Library(Chart(predictor, kind=kind), max_size=10)
    chart_path = charts_dir / f"chart_{kind}_{regime}.pt"
    if not chart_path.exists():
        raise FileNotFoundError(
            f"{chart_path} not found — E2 needs E0's dynamics chart for {regime}. "
            f"Point --charts-dir at the directory holding chart_{kind}_{regime}.pt."
        )
    library.add(Chart.load(str(chart_path), predictor))
    return library


def build_confusion_library(charts_dir: Path, predictor, kind: str) -> Library:
    """Library = {c0, chart_R1, chart_R2} for the 3-chart confusion-matrix
    diagnostic (E2_RESULTS.md Limitations #4: '2-entry library is the minimum
    that makes routing meaningful' -- this is the 3-entry follow-up, chance
    accuracy 1/3 instead of 1/2). Index i's regime is REGIME_ORDER[i]."""
    library = Library(Chart(predictor, kind=kind), max_size=10)
    for regime in REGIME_ORDER[1:]:  # R0 is index 0 = c0, already added above
        chart_path = charts_dir / f"chart_{kind}_{regime}.pt"
        if not chart_path.exists():
            raise FileNotFoundError(
                f"{chart_path} not found — --confusion-matrix needs a chart_{kind}_{{regime}}.pt "
                f"for every regime in {REGIME_ORDER[1:]}. Point --charts-dir at a directory "
                f"holding both."
            )
        library.add(Chart.load(str(chart_path), predictor))
    return library


REGIME_ORDER = ["R0", "R1", "R2"]  # library index i <-> REGIME_ORDER[i]'s chart


def _best_umf(info: dict) -> float | None:
    scores = info.get("scores") or []
    finite = [s for s in scores if s is not None]
    return min(finite) if finite else None


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: Appearance vs dynamics 2x2.")
    parser.add_argument("--cells", nargs="+", default=["A", "B", "C", "D"],
                        choices=["A", "B", "C", "D"])
    parser.add_argument("--routers", nargs="+", default=["umf", "sdyn"],
                        choices=["umf", "sdyn", "e1", "random"])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--traj-len", type=int, default=50,
                        help="Raw env steps per collected trajectory. 50 matches E0's "
                             "--eval-traj-len: long enough for UMF's displacement "
                             "denominator to be well-behaved (code-review.md Bug #6d).")
    parser.add_argument("--kind", default="ln_act",
                        help="Adapter kind for c0 and the loaded dynamics chart. ln_act is "
                             "E0's only kind with a positive (if small) effect; lora4 and "
                             "full regressed (E0_RESULTS.md P4).")
    parser.add_argument("--chart-regime", default="R1",
                        help="Which E0 chart fills library index 1. Must match the regime "
                             "used as regime_b in cells B/D.")
    parser.add_argument("--corruption", default="colour_change",
                        choices=["colour_change", "dark", "blur", "salt_pepper"],
                        help="Appearance shift for cells C/D. Plan 6.3 names colour, which is "
                             "the default -- but MEASURED on this env it changes only ~5.6%% of "
                             "pixels (Push-T renders are ~97%% white, mean 248, and an HSV hue "
                             "rotation is a no-op on desaturated pixels), so Cell C risks "
                             "passing VACUOUSLY: no appearance shift to over-expand on. 'dark' "
                             "changes 100%% of pixels and is the conservative choice -- it makes "
                             "Cell C harder to pass, not easier. The startup check below reports "
                             "the actual magnitude either way.")
    parser.add_argument("--dynamics-regime", default="R1",
                        help="Regime for the 'dynamics differ' cells (B/D). Plan 6.3 names R1; "
                             "R2 (damping 0.5) is the larger shift (46.7%% vs 66.7%% frozen "
                             "baseline) and is the useful comparison when R1's between-chart UMF "
                             "gap lands under the hysteresis margin. Must match --chart-regime.")
    parser.add_argument("--probe-q", type=int, default=3,
                        help="DIAGNOSTIC OVERRIDE of CLAUDE.md 1.7's fixed q=3. Leave at 3 for "
                             "any reported result. Lower it ONLY to make the Cell C "
                             "over-expansion test non-vacuous: measured UMF exceeds tau=0.5 in "
                             "0-11.5%% of chunks, so three CONSECUTIVE strikes essentially never "
                             "accumulate and the Expander never arms in any cell -- 'committed 0' "
                             "then says nothing about over-expansion. Any deviation is recorded "
                             "in the summary and printed loudly.")
    parser.add_argument("--probe-tau", type=float, default=0.5,
                        help="DIAGNOSTIC OVERRIDE of CLAUDE.md 1.7's fixed tau=0.5. Same rules as "
                             "--probe-q: never change it for a reported number.")
    parser.add_argument("--corruption-severity", type=float, default=0.5)
    parser.add_argument("--charts-dir", type=Path, default=atlas.OUT_DIR / "e0")
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e2")
    parser.add_argument("--confusion-matrix", action="store_true",
                        help="Run the 3-chart {c0, chart_R1, chart_R2} confusion-matrix "
                             "diagnostic instead of the 2x2 cells (E2_RESULTS.md Limitations "
                             "#4: a 2-entry library's chance accuracy is 0.5; this is 1/3). "
                             "Needs chart_{kind}_R1.pt AND chart_{kind}_R2.pt in --charts-dir. "
                             "Ignores --cells/--corruption/--dynamics-regime/--chart-regime.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"E2: cells={args.cells} routers={args.routers} "
          f"episodes={args.episodes} seeds={args.seeds}", flush=True)

    _atlas_home = os.environ.get("ATLAS_HOME", str(atlas.ATLAS_HOME))
    hub_path = str(Path(_atlas_home) / "hub" / "hub" / "facebookresearch_jepa-wms_main")
    model, prep = torch.hub.load(hub_path, "dino_wm_pusht", source="local",
                                 force_reload=False, trust_repo=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = model.to(device)
    wm = wrapper.model if hasattr(wrapper, "model") else wrapper
    for p in wm.encoder.parameters():
        p.requires_grad_(False)
    for m in wm.predictor.modules():
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = True
    torch.set_float32_matmul_precision("high")

    if args.confusion_matrix:
        gate_trajs = load_regime_trajectories(wrapper, prep, "R0", num_trajs=8,
                                              traj_len=args.traj_len, device=device,
                                              seed_offset=90_000, source="dataset")
        gate_displacements = torch.tensor([
            (t["encoder_output"][-1] - t["encoder_output"][0]).norm(p="fro").item()
            for t in gate_trajs
        ])
        motion_gate = compute_motion_gate(gate_displacements)
        print(f"motion_gate (10th pct of R0 train displacement) = {motion_gate:.4f}", flush=True)
        run_confusion_matrix(args, wrapper, prep, device, motion_gate)
        return

    if args.chart_regime != args.dynamics_regime:
        print(f"  [WARN] --chart-regime={args.chart_regime} but "
              f"--dynamics-regime={args.dynamics_regime}: cells B/D would shift into a regime "
              f"the library has no chart for, making 'correct' unreachable by construction.",
              flush=True)
    if (args.probe_q, args.probe_tau) != (3, 0.5):
        print(f"  [DIAGNOSTIC RUN] q={args.probe_q} tau={args.probe_tau} deviate from "
              f"CLAUDE.md 1.7's fixed (3, 0.5). Expansion numbers from this run are a "
              f"diagnostic, NOT a reportable result.", flush=True)
    library = build_library(args.charts_dir, wm.predictor, args.kind, args.chart_regime)
    print(f"Library: {len(library)} charts "
          f"(0=c0 identity, 1=chart_{args.kind}_{args.chart_regime})", flush=True)

    # The same min-motion gate every other experiment uses (CLAUDE.md 1.7):
    # sub-threshold chunks return None and must yield no routing decision, no
    # strike and no probe. Derived once from R0 so a corrupted cell cannot move
    # its own gate.
    gate_trajs = load_regime_trajectories(wrapper, prep, "R0", num_trajs=8,
                                          traj_len=args.traj_len, device=device,
                                          seed_offset=90_000, source="dataset")
    # compute_motion_gate takes a 1-D tensor of per-chunk Frobenius
    # displacements, not the trajectories themselves -- same construction
    # run_e0.py and run_e1.py use, so all three experiments gate identically.
    gate_displacements = torch.tensor([
        (t["encoder_output"][-1] - t["encoder_output"][0]).norm(p="fro").item()
        for t in gate_trajs
    ])
    motion_gate = compute_motion_gate(gate_displacements)

    # Measure the appearance shift instead of assuming it. A corruption that
    # barely moves pixels makes Cell C's "committed == 0" pass for the wrong
    # reason -- the vacuous-verification failure mode G3b's docstring warns
    # about. Recorded in the summary so any result carries its own evidence.
    from atlas.regimes import _corrupt
    import numpy as _np
    _ref = _np.asarray(gate_trajs[0]["_raw_frame"]) if "_raw_frame" in gate_trajs[0] else None
    corruption_check = None
    if _ref is None:
        from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv as _PE
        _e = _PE(render_size=224, with_velocity=True); _e.seed(0)
        _o, _ = _e.reset()
        _ref = _np.asarray(_o["visual"])
    _c = _corrupt(_ref, args.corruption, args.corruption_severity,
                  _np.random.default_rng(0))
    _d = _np.abs(_ref.astype(int) - _c.astype(int))
    corruption_check = {"kind": args.corruption, "severity": args.corruption_severity,
                        "mean_abs_pixel_diff": float(_d.mean()),
                        "frac_pixels_changed": float((_d.sum(-1) > 0).mean())}
    print(f"corruption {args.corruption}@{args.corruption_severity}: "
          f"{100 * corruption_check['frac_pixels_changed']:.1f}% of pixels changed, "
          f"mean |diff| = {corruption_check['mean_abs_pixel_diff']:.2f}", flush=True)
    if corruption_check["frac_pixels_changed"] < 0.20:
        print("  [WARN] appearance shift is small -- Cell C's committed==0 may pass "
              "vacuously. Consider --corruption dark.", flush=True)
    print(f"motion_gate (10th pct of R0 train displacement) = {motion_gate:.4f}", flush=True)

    records: list[dict] = []
    t_start = time.time()

    for cell in args.cells:
        cfg = dict(CELL_CONFIGS[cell])
        if cfg["corruption_b"] is None:
            cfg["corruption_b"] = args.corruption
        if cfg["regime_b"] is None:
            cfg["regime_b"] = args.dynamics_regime
        for cond in ("A", "B"):
            regime = "R0" if cond == "A" else cfg["regime_b"]
            corruption = "none" if cond == "A" else cfg["corruption_b"]
            correct_idx = 0 if cond == "A" else cfg["correct_b"]

            for seed in range(args.seeds):
                # Separates every (cell, cond, seed) draw so no two conditions
                # silently share trajectories, while staying deterministic.
                offset = (200_000 + 10_000 * seed + 1_000 * ord(cell)
                          + (0 if cond == "A" else 500))
                trajs = load_regime_trajectories(
                    wrapper, prep, regime, num_trajs=args.episodes,
                    traj_len=args.traj_len, device=device, seed_offset=offset,
                    source="dataset", corruption=corruption,
                    corruption_severity=args.corruption_severity)

                # One Expander per (cell, cond, seed): strikes accumulate ACROSS
                # episodes within a condition, which is what C2's
                # q-strikes-then-probe rule assumes. Reset between conditions so
                # cell C cannot inherit strikes earned under cell B. Probes run
                # against a CLONED library so a commit cannot leak into the
                # routing library the accuracy metric reads.
                expander = Expander(ExpansionConfig(kind=args.kind, q=args.probe_q,
                                                    tau=args.probe_tau))
                probe_library = Library(library.c0.clone(), max_size=10)
                probe_library.add(library[1].clone())
                commits = 0

                # Sequential hysteresis (E2_RESULTS.md Limitations #2): carry each
                # router's own previously-selected chart forward as the next
                # decision's current_idx, instead of hardcoding 0 every episode.
                # current_idx=0 always meant hysteresis always favoured c0,
                # inflating R0-condition accuracy and deflating shifted-condition
                # accuracy. Reset per (cell, cond, seed) -- a fresh condition is a
                # fresh deployment starting from the identity chart, same as before.
                current_idx_by_router = {router: 0 for router in args.routers}

                for ep, traj in enumerate(tqdm(trajs, desc=f"route_{cell}{cond}_s{seed}",
                                               unit="ep", leave=False)):
                    enc = traj["encoder_output"]
                    acts = traj["actions"]
                    proprio_ctxt = traj["proprio"][0:1].unsqueeze(0)  # [1,1,P_tok,D_p]

                    umf_info = None
                    for router in args.routers:
                        idx, info = route(router, library, wrapper, enc, acts,
                                          current_idx=current_idx_by_router[router],
                                          motion_gate=motion_gate,
                                          proprio_ctxt=proprio_ctxt)
                        current_idx_by_router[router] = idx
                        if router == "umf":
                            umf_info = info
                        gated = bool(info.get("gated", False))
                        records.append({
                            "cell": cell, "condition": cond, "seed": seed, "episode": ep,
                            "regime": regime, "corruption": corruption,
                            "router": router, "selected": idx, "correct": correct_idx,
                            "hit": (None if gated else int(idx == correct_idx)),
                            "gated": gated,
                            "scores": info.get("scores"),
                        })

                    # Expansion is router-independent — it reads the library's
                    # best UMF — so it is recorded once per episode, not once per
                    # router. Reuse the umf router's scores when it already ran.
                    if umf_info is None:
                        _, umf_info = route("umf", library, wrapper, enc, acts,
                                            current_idx=0, motion_gate=motion_gate,
                                            proprio_ctxt=proprio_ctxt)
                    expander.record(_best_umf(umf_info), enc, acts, proprio_ctxt=proprio_ctxt)
                    outcome = "not_ready"
                    if ep + 1 < len(trajs):
                        nxt = trajs[ep + 1]
                        outcome = expander.maybe_expand(
                            probe_library, wrapper, nxt["encoder_output"], nxt["actions"],
                            motion_gate,
                            next_proprio_ctxt=nxt["proprio"][0:1].unsqueeze(0))
                        # ProbeOutcome is a STRING literal ("committed" /
                        # "rejected_score" / "rejected_full" / "not_ready"), not
                        # an object with a .committed attribute.
                        if outcome == "committed":
                            commits += 1
                    # FIX_SPEC.md A3: per-chunk expander state, so N9's commit
                    # count and the K=3 hysteresis binding rate are recomputable
                    # from raw records alone (closes PAPER_FACT_CHECK D4).
                    records.append({
                        "cell": cell, "condition": cond, "seed": seed, "episode": ep,
                        "regime": regime, "corruption": corruption,
                        "router": None, "record_type": "expansion",
                        "strikes": expander._strikes,
                        "probe_outcome": outcome,
                        "relative_gap": (umf_info or {}).get("relative_gap"),
                        "hysteresis_held": bool((umf_info or {}).get("hysteresis_held", False)),
                        "committed": outcome == "committed",
                        "incumbent_debug": dict(getattr(expander, "_last_probe_debug", {})),
                        "library_size": len(probe_library),
                    })

                # Cross-check against the Expander's own counter: if these
                # ever disagree the loop above missed a commit, and Cell C would
                # pass vacuously.
                stats = expander.stats()
                assert stats["charts_committed"] == commits, (
                    f"commit count mismatch: loop={commits} expander={stats}")
                records.append({"cell": cell, "condition": cond, "seed": seed,
                                "episode": None, "router": None,
                                "charts_committed": commits,
                                "expander_stats": stats,
                                "library_size": len(probe_library)})
                print(f"  {cell}/{cond} seed={seed}: {args.episodes} eps, "
                      f"commits={commits}, lib={len(probe_library)}", flush=True)

    jsonl_path = args.out / "e2_episodes.jsonl"
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    routing_accuracy: dict[str, dict[str, float]] = {}
    for router in args.routers:
        routing_accuracy[router] = {}
        for cell in args.cells:
            hits = [r["hit"] for r in records
                    if r.get("router") == router and r["cell"] == cell
                    and r["hit"] is not None]
            routing_accuracy[router][cell] = (sum(hits) / len(hits)) if hits else float("nan")

    commits_by_cell = {
        cell: sum(r.get("charts_committed", 0) for r in records
                  if r["cell"] == cell and r.get("episode") is None)
        for cell in args.cells
    }

    n_decisions = sum(1 for r in records if r.get("router"))
    summary = {
        "routing_accuracy": routing_accuracy,
        "charts_committed": commits_by_cell,
        "gated_fraction": sum(1 for r in records if r.get("gated")) / max(1, n_decisions),
        "config": {"episodes": args.episodes, "seeds": args.seeds, "kind": args.kind,
                   "chart_regime": args.chart_regime, "traj_len": args.traj_len,
                   "dynamics_regime": args.dynamics_regime,
                   "probe_q": args.probe_q, "probe_tau": args.probe_tau,
                   "probe_params_are_preregistered": (args.probe_q, args.probe_tau) == (3, 0.5),
                   "corruption_severity": args.corruption_severity,
                   "corruption_check": corruption_check,
                   "motion_gate": motion_gate,
                   "planner": "none — collected trajectories, see module docstring"},
        "wall_time": time.time() - t_start,
    }
    with open(args.out / "e2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n-- E2 routing accuracy --")
    print("router     " + "  ".join(f"{c:>7}" for c in args.cells))
    for router in args.routers:
        print(f"{router:<10} " + "  ".join(f"{routing_accuracy[router][c]:7.3f}"
                                           for c in args.cells))
    print(f"\ncharts committed: {commits_by_cell}")
    if "C" in commits_by_cell:
        verdict = "PASS" if commits_by_cell["C"] == 0 else "FAIL"
        print(f"Cell C over-expansion test: {verdict} "
              f"(committed {commits_by_cell['C']}, required 0)")

    try:
        from atlas.plots import two_panel
        two_panel(routing_accuracy, {}, true_regime_count=2,
                  out_path=args.out / "F2a.pdf")
        print(f"Figure: {args.out / 'F2a.pdf'}")
    except Exception as e:
        print(f"[warn] F2a not written: {e}")

    print(f"Results: {jsonl_path}")


def run_confusion_matrix(args, wrapper, prep, device, motion_gate: float | None) -> None:
    """3-chart {c0, chart_R1, chart_R2} routing accuracy, chance=1/3 instead of
    the 2x2 cells' 2-entry-library 0.5 (E2_RESULTS.md Limitations #4: 'nothing
    here speaks to selection among 3+ charts'). No appearance corruption --
    pure dynamics discrimination across R0/R1/R2, with the same sequential
    hysteresis fix as the 2x2 path (current_idx carried forward per router,
    reset at the start of each (regime, seed) sequence)."""
    wm = wrapper.model if hasattr(wrapper, "model") else wrapper
    library = build_confusion_library(args.charts_dir, wm.predictor, args.kind)
    print(f"Confusion-matrix library: {len(library)} charts "
          f"(0=c0, 1=chart_{args.kind}_R1, 2=chart_{args.kind}_R2), chance=1/3", flush=True)

    records: list[dict] = []
    t_start = time.time()
    for true_idx, regime in enumerate(REGIME_ORDER):
        for seed in range(args.seeds):
            # Distinct offset block from the 2x2 path's own seeding scheme
            # (200_000+) so a shared --out dir never draws identical episodes.
            offset = 300_000 + 10_000 * seed + 1_000 * true_idx
            trajs = load_regime_trajectories(
                wrapper, prep, regime, num_trajs=args.episodes,
                traj_len=args.traj_len, device=device, seed_offset=offset,
                source="dataset")

            current_idx_by_router = {router: 0 for router in args.routers}
            for ep, traj in enumerate(tqdm(trajs, desc=f"confusion_{regime}_s{seed}",
                                           unit="ep", leave=False)):
                enc = traj["encoder_output"]
                acts = traj["actions"]
                proprio_ctxt = traj["proprio"][0:1].unsqueeze(0)
                for router in args.routers:
                    idx, info = route(router, library, wrapper, enc, acts,
                                      current_idx=current_idx_by_router[router],
                                      motion_gate=motion_gate, proprio_ctxt=proprio_ctxt)
                    current_idx_by_router[router] = idx
                    gated = bool(info.get("gated", False))
                    records.append({
                        "regime": regime, "true_idx": true_idx, "seed": seed, "episode": ep,
                        "router": router, "selected": idx,
                        "hit": (None if gated else int(idx == true_idx)),
                        "gated": gated, "scores": info.get("scores"),
                    })

    jsonl_path = args.out / "e2_confusion_episodes.jsonl"
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    confusion: dict[str, list[list[int]]] = {}
    accuracy: dict[str, float] = {}
    for router in args.routers:
        mat = [[0, 0, 0] for _ in REGIME_ORDER]  # rows=true regime, cols=selected chart idx
        n_ungated = n_correct = 0
        for r in records:
            if r["router"] != router or r["hit"] is None:
                continue
            mat[r["true_idx"]][r["selected"]] += 1
            n_ungated += 1
            n_correct += r["hit"]
        confusion[router] = mat
        accuracy[router] = (n_correct / n_ungated) if n_ungated else float("nan")

    summary = {
        "confusion_matrix": confusion,
        "accuracy": accuracy, "chance_accuracy": 1 / 3,
        "config": {"episodes": args.episodes, "seeds": args.seeds, "kind": args.kind,
                   "traj_len": args.traj_len, "motion_gate": motion_gate,
                   "regime_order": REGIME_ORDER,
                   "planner": "none -- collected trajectories, see module docstring"},
        "wall_time": time.time() - t_start,
    }
    with open(args.out / "e2_confusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n-- E2 3-chart confusion matrix (rows=true regime, cols=selected) --")
    for router in args.routers:
        print(f"\nrouter={router}  accuracy={accuracy[router]:.3f}  (chance=0.333)")
        print("            " + "  ".join(f"sel={r}" for r in REGIME_ORDER))
        for i, row in enumerate(confusion[router]):
            print(f"true={REGIME_ORDER[i]:<4}  " + "  ".join(f"{v:5d}" for v in row))
    print(f"\nResults: {jsonl_path}")


if __name__ == "__main__":
    main()
