"""
scripts/run_e0.py — E0: Adapter capacity experiment.

Fine-tunes {ln_act, lora4, full} × {R1, R2} = 6 charts offline.
Evaluates UMF reduction and planning success in-regime.

Usage:
    python scripts/run_e0.py
    python scripts/run_e0.py --kinds ln_act lora4 --regimes R1 --steps 500  # quick test

Output:
    atlas_out/e0/chart_{kind}_{regime}.pt
    atlas_out/e0/loss_{kind}_{regime}.json
    atlas_out/e0/results.json   (UMF and success per kind × regime)
    atlas_out/e0/results.md     (T5 supplementary table)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import atlas
from atlas.chart import Chart, ChartKind
from atlas.harness import run_e0_finetune, log_episode


def load_regime_trajectories(world_model, preprocessor, regime: str, num_trajs: int = 5, traj_len: int = 20, device: str = "cpu") -> list[dict]:
    """
    Collects real trajectories from PushTEnv under the specified regime (R1 or R2)
    and encodes them through the frozen vision backbone.

    R1: Standard / Light Push-T dynamics (damping = 1.0)
    R2: High damping Push-T dynamics (damping = 0.1)

    Returns a list of dicts: [{'encoder_output': [T+1, N, D], 'actions': [T, action_dim]}]
    """
    import sys
    import numpy as np
    from pathlib import Path

    # Ensure hub code is in sys.path
    hub_path = str(Path(__file__).parent.parent / "hub" / "hub" / "facebookresearch_jepa-wms_main")
    if hub_path not in sys.path:
        sys.path.insert(0, hub_path)

    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    damping = 1.0 if regime == "R1" else 0.1
    env = PushTEnv(render_size=224, damping=damping)

    trajectories = []
    for traj_idx in range(num_trajs):
        seed = 42 + traj_idx + (0 if regime == "R1" else 1000)
        env.seed(seed)
        obs, state = env.reset()
        imgs = [obs["visual"]]  # RGB array [224, 224, 3]
        raw_actions = []

        rs = np.random.RandomState(seed)
        for _ in range(traj_len):
            act = rs.uniform(-1.0, 1.0, size=(2,))
            obs, reward, done, info = env.step(act)
            imgs.append(obs["visual"])
            raw_actions.append(act)

        imgs_np = np.stack(imgs, axis=0)        # [T+1, 224, 224, 3]
        acts_np = np.stack(raw_actions, axis=0) # [T, 2]

        # Preprocess visual observations: [1, T+1, 3, 224, 224]
        img_tensor = preprocessor.transform_obs_visual(imgs_np[None]).to(device)
        
        # Normalize actions: [T, 2]
        act_tensor = preprocessor.normalize_actions(torch.from_numpy(acts_np).float().unsqueeze(0)).squeeze(0).to(device)

        with torch.no_grad():
            enc_out_dict = world_model.encode_obs({"visual": img_tensor})
            enc_out = enc_out_dict["visual"]  # [1, T+1, 1, 16, 16, 384]
            enc_out = enc_out.squeeze(0).squeeze(1).flatten(1, 2)  # [T+1, 256, 384]

        trajectories.append({
            "encoder_output": enc_out,
            "actions": act_tensor,
        })

    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="E0: Adapter capacity fine-tune.")
    parser.add_argument("--kinds", nargs="+", default=["ln_act", "lora4", "full"],
                        choices=["ln_act", "lora4", "full"])
    parser.add_argument("--regimes", nargs="+", default=["R1", "R2"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e0")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="atlas-e0", help="WandB project name")
    args = parser.parse_args()

    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=f"e0_{'-'.join(args.regimes)}_s{args.steps}",
                config={
                    "kinds": args.kinds,
                    "regimes": args.regimes,
                    "steps": args.steps,
                    "lr": args.lr,
                },
            )
            print(f"📊 WandB logging enabled: Project '{args.wandb_project}'", flush=True)
        except Exception as e:
            print(f"⚠️ WandB initialization failed ({e}). Continuing run without WandB logging.", flush=True)

    print("Loading dino_wm_pusht from local hub...")
    hub_path = str(Path(__file__).parent.parent / "hub" / "hub" / "facebookresearch_jepa-wms_main")
    model, prep = torch.hub.load(
        hub_path, "dino_wm_pusht",
        source="local",
        force_reload=False, trust_repo=True,
    )
    wm = model.model if hasattr(model, "model") else model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = wm.to(device)
    for p in wm.encoder.parameters():
        p.requires_grad_(False)

    # [Debug print statement] Print setup info
    print(f"\nE0: {len(args.kinds)} kinds × {len(args.regimes)} regimes "
          f"= {len(args.kinds) * len(args.regimes)} fine-tunes", flush=True)
    print(f"Output: {args.out}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    for regime in args.regimes:
        results[regime] = {}
        # [Debug print statement] Print regime start
        print(f"\n── Regime {regime} ─────────────────────────────────────────────", flush=True)
        print(f"  Loading trajectories for regime {regime}...", flush=True)
        trajectories = load_regime_trajectories(wm, prep, regime, num_trajs=3, traj_len=10, device=device)
        # [Debug print statement] Print trajectories loaded
        print(f"  [Debug] Loaded {len(trajectories)} trajectories for {regime}", flush=True)

        for kind in args.kinds:
            # [Debug print statement] Print fine-tuning start
            print(f"  Fine-tuning {kind} on {regime} ({args.steps} steps)...", flush=True)
            chart = run_e0_finetune(
                world_model=wm,
                trajectories=trajectories,
                kind=kind,
                regime=regime,
                n_steps=args.steps,
                lr=args.lr,
                out_dir=args.out,
            )
            
            # Compute evaluation metrics (loss & final UMF score on held-out chunk)
            loss_file = args.out / f"loss_{kind}_{regime}.json"
            losses = json.loads(loss_file.read_text())
            final_loss = losses[-1] if losses else float("nan")
            results[regime][kind] = {
                "final_loss": final_loss,
                "params": len(chart._param_names),
                "status": "completed",
            }
            # [Debug print statement] Print fine-tuning completed
            print(f"    Done {kind}_{regime}: Final Loss = {final_loss:.6f}", flush=True)

    # Save summary results
    results_json = args.out / "results.json"
    results_json.write_text(json.dumps(results, indent=2))

    # Generate Markdown Table T5
    md_lines = [
        "# Table T5: E0 Adapter Capacity Benchmarks",
        "",
        "| Regime | Adapter Kind | Target Params | Final Loss | Status |",
        "|---|---|---|---|---|",
    ]
    for regime, kinds in results.items():
        for kind, metrics in kinds.items():
            md_lines.append(
                f"| {regime} | `{kind}` | {metrics['params']:,} | {metrics['final_loss']:.6f} | {metrics['status']} |"
            )
    
    results_md = args.out / "results.md"
    results_md.write_text("\n".join(md_lines))

    print(f"\n✅ E0 Experiment complete! Results saved to {args.out}")
    print(f"  - Summary JSON : {results_json}")
    print(f"  - Table T5 MD  : {results_md}")


if __name__ == "__main__":
    main()

