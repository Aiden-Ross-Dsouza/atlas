"""Analysis for the three C-2 mechanism experiments (C2_FAILURE_DIAGNOSIS.md Part 5).

Read-only. Recomputes everything from raw JSONL/JSON; imports `mcnemar_paired` and
`paired_bootstrap` from atlas.stats unmodified (CLAUDE.md 1.2).

  --mode dose   : CEM-iteration dose-response, chart vs frozen c0, paired per episode.
  --mode alpha  : proprio-term ablation (alpha=0 vs alpha=0.1), paired.
  --mode c1last : C-1 at the converged CEM iteration vs the existing iteration-0 file.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.stats import mcnemar_paired, paired_bootstrap  # noqa: E402


def load_jsonl(p: Path) -> dict[int, dict]:
    return {json.loads(l)["episode"]: json.loads(l) for l in p.open() if l.strip()}


def arm_stats(rows: dict[int, dict]) -> dict:
    eps = sorted(rows)
    umf = [rows[e]["umf_mean"] for e in eps if rows[e].get("umf_mean") is not None]
    ka = sum(1 for e in eps if rows[e]["block_pos_diff"] > rows[e]["init_block_pos_diff"])
    static = sum(1 for e in eps
                 if abs(rows[e]["block_pos_diff"] - rows[e]["init_block_pos_diff"]) < 0.5)
    return {
        "n": len(eps),
        "sr": sum(1 for e in eps if rows[e]["success"] is True),
        "contacts": st.mean(rows[e]["total_contacts"] for e in eps),
        "zero_contact": sum(1 for e in eps if rows[e]["total_contacts"] == 0),
        "block_static_lt0p5px": static,
        "knockaways": ka,
        "final_dist": st.mean(rows[e]["block_pos_diff"] for e in eps),
        "umf": st.mean(umf) if umf else None,
        "n_umf": len(umf),
    }


def paired_block(chart: dict[int, dict], base: dict[int, dict], label: str) -> dict:
    """Every comparison is paired on episode index; pairing is ASSERTED, not assumed."""
    eps = sorted(set(chart) & set(base))
    for e in eps:
        assert abs(chart[e]["init_block_pos_diff"] - base[e]["init_block_pos_diff"]) < 1e-9, e
        assert abs(chart[e]["init_agent_block_dist"] - base[e]["init_agent_block_dist"]) < 1e-9, e
    c = [bool(chart[e]["success"]) for e in eps]
    b = [bool(base[e]["success"]) for e in eps]
    n10 = sum(1 for x, y in zip(c, b) if x and not y)
    n01 = sum(1 for x, y in zip(c, b) if y and not x)
    mc = mcnemar_paired(c, b)
    bs = paired_bootstrap(np.array(c, dtype=float), np.array(b, dtype=float), n=10000, seed=0)
    cs, bs_ = arm_stats({e: chart[e] for e in eps}), arm_stats({e: base[e] for e in eps})
    out = {
        "label": label, "n_paired": len(eps),
        "chart_sr": f"{cs['sr']}/{len(eps)}", "baseline_sr": f"{bs_['sr']}/{len(eps)}",
        "delta_sr": (cs["sr"] - bs_["sr"]) / len(eps),
        "mcnemar_p": float(mc),
        "paired_bootstrap": bs,
        "discordant": {"chart_only": n10, "baseline_only": n01},
        "chart": cs, "baseline": bs_,
        "init_states_identical": True,
    }
    return out


def fmt_arm(a: dict) -> str:
    u = f"{a['umf']:.3f}(n={a['n_umf']})" if a["umf"] is not None else "n/a"
    return (f"SR {a['sr']}/{a['n']} | contacts {a['contacts']:.2f} | zero-contact "
            f"{a['zero_contact']} | static<0.5px {a['block_static_lt0p5px']} | "
            f"knockaway {a['knockaways']} | final {a['final_dist']:.1f}px | UMF {u}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dose", "alpha", "c1last"])
    ap.add_argument("--root", type=Path, default=Path("phase0_v3"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.mode in ("dose", "alpha"):
        if args.mode == "dose":
            cells = [
                (1, args.root / "c2_dose_it1_ln_act/ln_act_R2.jsonl",
                    args.root / "c2_dose_it1_baseline/baseline_R2.jsonl"),
                (3, args.root / "c2_dose_it3_ln_act/ln_act_R2.jsonl",
                    args.root / "c2_dose_it3_baseline/baseline_R2.jsonl"),
                (10, args.root / "c2_p0g_R2/ln_act_R2.jsonl",
                    args.root / "p0c/p0c_it10_baseline_R2.jsonl"),
            ]
            key = "cem_iterations"
        else:
            cells = [
                (0.0, args.root / "c2_alpha0_ln_act/ln_act_R2.jsonl",
                    args.root / "c2_alpha0_baseline/baseline_R2.jsonl"),
                (0.1, args.root / "c2_p0g_R2/ln_act_R2.jsonl",
                    args.root / "p0c/p0c_it10_baseline_R2.jsonl"),
            ]
            key = "objective_alpha"

        results = []
        for v, cp, bp in cells:
            if not cp.exists() or not bp.exists():
                print(f"[MISSING] {key}={v}: {cp if not cp.exists() else bp}")
                continue
            r = paired_block(load_jsonl(cp), load_jsonl(bp), f"{key}={v}")
            r[key] = v
            results.append(r)
            print(f"\n=== {key} = {v} ===")
            print(f"  chart    : {fmt_arm(r['chart'])}")
            print(f"  frozen c0: {fmt_arm(r['baseline'])}")
            print(f"  delta SR {r['delta_sr']:+.3f} | McNemar {r['mcnemar_p']} | "
                  f"discordant chart-only {r['discordant']['chart_only']} / "
                  f"baseline-only {r['discordant']['baseline_only']}")
            print(f"  paired bootstrap: {r['paired_bootstrap']}")

        if results:
            print(f"\n--- DOSE TABLE ({key}) ---")
            print(f"{key:>18} | chart SR | frozen SR |  dSR   | chart contacts | frozen contacts")
            for r in results:
                print(f"{r[key]:>18} | {r['chart_sr']:>8} | {r['baseline_sr']:>9} | "
                      f"{r['delta_sr']:+.3f} | {r['chart']['contacts']:>14.2f} | "
                      f"{r['baseline']['contacts']:>15.2f}")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
            print(f"\nwrote {args.out}")
        return

    # ---- c1last -------------------------------------------------------------
    it0 = json.load(open(args.root / "cost_ranking_p0g_R2" /
                          "cost_ranking_R2_seeds0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19.json"))
    lastdir = args.root / "cost_ranking_p0g_R2_iterlast"
    cands = sorted(lastdir.glob("cost_ranking_R2_seeds*.json"))
    if not cands:
        print(f"[MISSING] no file in {lastdir}")
        return
    itL = json.load(open(cands[0]))
    print(f"iteration-0 file: iterations={it0['iterations']} capture={it0['capture_iteration']}")
    print(f"converged  file: iterations={itL['iterations']} capture={itL['capture_iteration']}")

    def per_seed(d):
        rows = {}
        for s in d["per_seed"]:
            r = s["results"]
            e = {}
            for k in ("baseline", "ln_act"):
                c = np.array(r[k]["costs"]); td = np.array(r[k]["true_dist"])
                ct = np.array(r[k]["contacts"])
                idx = np.argsort(c)[:10]
                e[k] = {"rho": r[k]["spearman_rho"],
                        "best": r[k]["best_by_cost_true_dist"],
                        "elite10_td": float(td[idx].mean()),
                        "elite10_contact": float((ct[idx] > 0).mean()),
                        "pop_mean_td": float(td.mean()),
                        "pop_contact_frac": float((ct > 0).mean())}
            rows[s["seed"]] = e
        return rows

    A, B = per_seed(it0), per_seed(itL)
    seeds = sorted(set(A) & set(B))
    print(f"\nseeds compared: {len(seeds)}")
    hdr = ("seed |        ITERATION 0 (C-1 as run)        |     CONVERGED (final CEM iter)")
    print(hdr)
    print("     | base_e10  chart_e10   base_bst chart_bst | base_e10  chart_e10   base_bst chart_bst")
    for s in seeds:
        a, b = A[s], B[s]
        print(f"{s:4d} | {a['baseline']['elite10_td']:8.1f} {a['ln_act']['elite10_td']:10.1f} "
              f"{a['baseline']['best']:10.1f} {a['ln_act']['best']:9.1f} | "
              f"{b['baseline']['elite10_td']:8.1f} {b['ln_act']['elite10_td']:10.1f} "
              f"{b['baseline']['best']:10.1f} {b['ln_act']['best']:9.1f}")

    print("\n--- SUMMARY (mean over seeds) ---")
    print(f"{'':30} {'iteration 0':>14} {'converged':>14}")
    for k in ("baseline", "ln_act"):
        for f, name in (("elite10_td", "elite-10 true dist px"),
                        ("best", "best-by-cost true dist px"),
                        ("elite10_contact", "elite-10 contact frac"),
                        ("pop_contact_frac", "population contact frac"),
                        ("rho", "per-seed rho")):
            print(f"{k:>9} {name:<27} {st.mean(A[s][k][f] for s in seeds):>14.3f} "
                  f"{st.mean(B[s][k][f] for s in seeds):>14.3f}")
    print("\n--- THE DECISIVE CONTRAST: chart minus frozen, same CEM stage ---")
    from scipy.stats import wilcoxon
    # pop_mean_td is the headline at convergence: CEM executes mean(elites) of the
    # FINAL population, so that population's true quality is what the planner commits to.
    for stage, D in (("iteration 0", A), ("converged", B)):
        for f, name in (("pop_mean_td", "population mean true dist"),
                        ("elite10_td", "elite-10 true dist"),
                        ("best", "best-by-cost true dist")):
            d = [D[s]["ln_act"][f] - D[s]["baseline"][f] for s in seeds]
            print(f"{stage:>12}: d({name:<26}) = {st.mean(d):+8.2f} px "
                  f"(chart better in {sum(1 for x in d if x < 0):2d}/{len(d)}, "
                  f"wilcoxon p={wilcoxon(d).pvalue:.4f})")
        print()
    print("\nNEGATIVE = chart better. The hypothesis under test (C2_FAILURE_DIAGNOSIS 3.1) "
          "predicts NEGATIVE at iteration 0 and POSITIVE (or null) at convergence.")


if __name__ == "__main__":
    main()
