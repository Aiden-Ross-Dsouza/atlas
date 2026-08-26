"""
scripts/diagnose_umf_locality.py — is UMF's spatial averaging why it fails to
rank charts by planning competence?

UMF averages squared prediction error over the WHOLE latent field (16x16 DINOv2
tokens x 384 dims). In a Push-T frame the block and agent occupy a small
fraction of the image, so most tokens encode static arena background. A chart
can lower global UMF by predicting background better while getting WORSE on the
few tokens that carry block pose -- the only ones CEM's cost depends on.

E0_RESULTS.md's P4 matrix is the motivating evidence: global eval UMF separated
`full` (0.728, catastrophic) from the rest, but could NOT rank `ln_act` above
`lora4` (~0.336 vs ~0.329, essentially tied) even though they differ by 10pp of
planning success in the OPPOSITE direction.

This script recomputes UMF restricted to the top-k most-MOVING tokens, where
"moving" is measured from the data itself (per-token displacement ||z_T - z_0||)
rather than from a pixel-coordinate mapping -- tokens tracking the block and
agent move; background tokens do not. No planner, no env stepping: it re-scores
saved encodings, so it costs minutes.

READ BEFORE USING THE OUTPUT: this is a POST-HOC diagnostic, run after seeing
P4's results. It explains the pre-registered result; it does not replace it, and
score.py's umf() is deliberately left untouched. Reporting a localized variant
as though it had been the metric all along would be exactly the goalpost-move
CLAUDE.md 1.8 forbids.

Usage:
    python scripts/diagnose_umf_locality.py --regime R2 --topk 16 32 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

import atlas
from atlas.chart import Chart
from atlas.score import _make_z_ctxt, _open_loop_rollout

sys.path.insert(0, str(Path(__file__).parent))
from run_e0 import load_regime_trajectories  # noqa: E402

# The five trained charts, as backed up locally. "baseline" is the frozen
# predictor with no chart applied — the reference every column is read against.
DEFAULT_CHARTS = {
    "ln_act/dataset": "e0_v3_dataset/chart_ln_act_R2.pt",
    "ln_act/hybrid": "e0_v3_hybrid/chart_ln_act_R2.pt",
    "lora4/dataset": "e0_v4_lora4/chart_lora4_R2.pt",
    "full/dataset": "e0_v4_full/chart_full_R2.pt",
    "ln_act/closed_loop": "e0_v5_closed_loop/chart_ln_act_R2.pt",
}


def umf_masked(z_true: torch.Tensor, z_pred: torch.Tensor,
               mask: torch.Tensor | None) -> float:
    """UMF over a token subset.

    Same ratio score.umf() computes — sum of squared prediction error over sum
    of squared observed displacement from z_0 — but restricted to `mask`
    tokens. mask=None reproduces the global metric, which is the control: if
    this does not match score.umf()'s own number the comparison is meaningless,
    so the caller checks it.
    """
    z0 = z_true[0]                     # [N, D]
    tgt = z_true[1:]                   # [T, N, D]
    if mask is not None:
        tgt = tgt[:, mask, :]
        z_pred = z_pred[:, mask, :]
        z0 = z0[mask, :]
    err = (z_pred - tgt).pow(2).sum()
    disp = (tgt - z0.unsqueeze(0)).pow(2).sum()
    if disp.item() == 0.0:
        return float("nan")
    return float((err / disp).item())


def token_motion(z_true: torch.Tensor) -> torch.Tensor:
    """Per-token displacement ||z_T - z_0||_2 over the chunk, shape [N].

    This is the data-driven stand-in for "which tokens are task-relevant":
    tokens covering the block and the agent move over a push; arena background
    does not. Avoids hard-coding an env-coords -> patch-grid mapping, which
    would have to track render size, grid size and patch stride to stay correct.
    """
    return (z_true[-1] - z_true[0]).norm(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="UMF spatial-locality diagnostic.")
    parser.add_argument("--regime", default="R2")
    parser.add_argument("--regime-config", default=None,
                        help="JSON, e.g. '{\"damping\": 0.5}'. Defaults to "
                             "regimes.REGIME_CONFIGS' calibrated value.")
    parser.add_argument("--trajs", type=int, default=8,
                        help="Eval trajectories. 8 matches E0's --num-val-trajs.")
    parser.add_argument("--traj-len", type=int, default=50,
                        help="Matches E0's --eval-traj-len.")
    parser.add_argument("--topk", type=int, nargs="+", default=[16, 32, 64],
                        help="Token-subset sizes to score (of 256 total).")
    parser.add_argument("--data-source", default="dataset",
                        help="Must match how the charts' eval UMF was computed, or the "
                             "global column will not reproduce E0's numbers.")
    parser.add_argument("--charts-root", type=Path, default=atlas.OUT_DIR)
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "umf_locality.json")
    args = parser.parse_args()

    if args.regime_config:
        from atlas.regimes import set_regime_config
        set_regime_config(args.regime, json.loads(args.regime_config))

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

    from atlas.regimes import REGIME_CONFIGS
    print(f"regime {args.regime} = {REGIME_CONFIGS.get(args.regime)}", flush=True)

    # seed_offset=10_000 is E0's own EVAL offset — the held-out split these
    # charts' reported eval_umf was measured on, so the global column is
    # comparable to results.json rather than to a fresh unrelated draw.
    trajs = load_regime_trajectories(wrapper, prep, args.regime, num_trajs=args.trajs,
                                     traj_len=args.traj_len, device=device,
                                     seed_offset=10_000, source=args.data_source)
    print(f"{len(trajs)} eval trajectories", flush=True)

    pristine = {k: v.detach().clone() for k, v in wm.predictor.state_dict().items()}

    arms: dict[str, str | None] = {"baseline (frozen)": None}
    for name, rel in DEFAULT_CHARTS.items():
        path = args.charts_root / rel
        if path.exists():
            arms[name] = str(path)
        else:
            print(f"  [skip] {name}: {path} not found", flush=True)

    results: dict[str, dict] = {}
    for arm, chart_path in arms.items():
        chart = None
        if chart_path is not None:
            try:
                chart = Chart.load(chart_path, wm.predictor)
            except Exception as e:
                # A corrupt/incomplete .pt (e.g. a truncated volume download)
                # should cost one arm, not the whole run.
                print(f"  [skip] {arm}: cannot load ({type(e).__name__}: "
                      f"{str(e)[:80]})", flush=True)
                continue
            chart.apply_(wm.predictor)

        per_traj: dict[str, list[float]] = {"global": []}
        for k in args.topk:
            per_traj[f"top{k}"] = []

        for traj in trajs:
            z_true = traj["encoder_output"]
            proprio_ctxt = traj["proprio"][0:1].unsqueeze(0)
            with torch.no_grad():
                z_ctxt = _make_z_ctxt(wrapper, z_true[0:1], proprio_ctxt)
                z_pred = _open_loop_rollout(wrapper, z_ctxt, traj["actions"])
            per_traj["global"].append(umf_masked(z_true, z_pred, None))
            motion = token_motion(z_true)
            for k in args.topk:
                idx = torch.topk(motion, min(k, motion.numel())).indices
                per_traj[f"top{k}"].append(umf_masked(z_true, z_pred, idx))

        if chart is not None:
            # chart.restore_() is NOT "put the pretrained weights back": for
            # non-LoRA kinds it is literally self.apply_(), so it re-applies THIS
            # chart (chart.py:107-127). The documented way back to pretrained is
            # c0.restore_(). For lora4 it does do real work — removing the
            # parametrization — so call it first, then hard-reset values from the
            # pristine snapshot, which is correct for every kind.
            chart.restore_(wm.predictor)
            wm.predictor.load_state_dict(pristine)
            after = wm.predictor.state_dict()
            for key, v0 in pristine.items():
                assert torch.equal(v0, after[key]), (
                    f"predictor not pristine in {key!r} after arm {arm} — "
                    "later arms would be contaminated")

        results[arm] = {k: (sum(v) / len(v)) for k, v in per_traj.items()}
        row = "  ".join(f"{k}={results[arm][k]:.4f}" for k in results[arm])
        print(f"{arm:<22} {row}", flush=True)

    cols = ["global"] + [f"top{k}" for k in args.topk]
    print("\n" + "=" * 72)
    print(f"{'arm':<22}" + "".join(f"{c:>12}" for c in cols))
    for arm, r in results.items():
        print(f"{arm:<22}" + "".join(f"{r[c]:>12.4f}" for c in cols))

    # The question the whole script exists to answer: does any localized variant
    # rank ln_act/dataset BELOW lora4/dataset, matching their planning success
    # (50% vs 40%), where the global metric ranks them the wrong way round?
    a, b = "ln_act/dataset", "lora4/dataset"
    if a in results and b in results:
        print(f"\nOrdering check ({a} should score LOWER — it planned better):")
        for c in cols:
            ok = "correct" if results[a][c] < results[b][c] else "INVERTED"
            print(f"  {c:<8} {a}={results[a][c]:.4f}  {b}={results[b][c]:.4f}  -> {ok}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"regime": args.regime,
                   "regime_config": REGIME_CONFIGS.get(args.regime),
                   "data_source": args.data_source, "trajs": args.trajs,
                   "traj_len": args.traj_len, "topk": args.topk,
                   "results": results}, f, indent=2)
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
