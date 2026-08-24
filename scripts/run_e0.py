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
import copy
import json
import time
from pathlib import Path

import torch
import atlas
from atlas.chart import Chart, ChartKind
from atlas.harness import run_e0_finetune, log_episode
from atlas.score import compute_motion_gate


def load_regime_trajectories(world_model, preprocessor, regime: str, num_trajs: int = 5, traj_len: int = 50, device: str = "cpu", max_tries: int = 8, seed_offset: int = 0, frameskip: int = 5) -> list[dict]:
    """
    Collects real trajectories from PushTEnv under the specified regime and
    encodes them through the frozen vision backbone.

    Physics regime is applied via atlas.regimes.PhysicsRegime — R0=default,
    R1=high friction, R2=high restitution. R1/R2 are NOT mass/damping-based:
    see ATLAS_implementation_plan_v2.md §6.1a and REGIME_DESIGN_REVIEW.md —
    Push-T's kinematic pusher makes mass-based shifts physically impossible to
    detect at any scale, so R1/R2 were re-targeted onto shape.friction and
    shape.elasticity, which are not subject to the same cancellation.

    Actions are NOT sampled IID per step. A single random target near the
    block is chosen once per trajectory, and the agent aims at it every step
    (small Gaussian noise added for diversity) — pure per-step IID random
    actions (the previous scheme) were empirically found to produce
    agent-block contact in only ~13-17% of rollouts, since a memoryless random
    walk rarely displaces net position over a short horizon. This mirrors how
    every public Push-T lineage (IBC, Diffusion Policy, DINO-WM) generates
    data — none uses fresh per-step random actions either; see
    ACTION_SAMPLING_REVIEW.md. Each trajectory is retried with a new seed (up
    to max_tries) if it produces zero total contact, as a hard guarantee on
    top of the aimed sampling (which only raises the expected contact rate,
    it doesn't guarantee it for every seed/spawn configuration).

    Frame/action chunking (E0_IMPLEMENTATION_PLAN.md T3): raw env steps are
    still what gets simulated (frameskip has no effect on physics), but the
    RETURNED actions/frames are chunked/subsampled to the MODEL time base --
    `_open_loop_rollout` unrolls one model step per `frameskip` raw steps, so
    training/eval data must match that or every step after the first runs on
    a stale, wrong-time-base target (see E0_DIAGNOSIS_AND_PLAN.md's root-cause
    writeup). `world_model.encode()` (not the old
    preprocessor.transform_obs_visual + encode_obs path) is used so real
    proprio is captured and normalized/embedded the same way the planner does
    it -- this checkpoint's predictor concatenates proprio into the token
    channel width (VideoWM.concat_obs_act(), dim=3) and errors without it, so
    it is not an optional enrichment here.

    Args:
        world_model: EncPredWM instance (the object torch.hub.load returns —
                     NOT .model).

    Returns a list of dicts:
        {'encoder_output': [T_model+1, N, D],
         'actions':        [T_model, 10]  (model-chunk, normalized),
         'proprio':        [T_model+1, P_tok, D_p]  (encoded, full sequence),
         'seed':           int}
    """
    import sys
    import numpy as np
    from pathlib import Path

    # Ensure hub code is in sys.path
    hub_path = str(Path(__file__).parent.parent / "hub" / "hub" / "facebookresearch_jepa-wms_main")
    if hub_path not in sys.path:
        sys.path.insert(0, hub_path)

    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
    from atlas.regimes import PhysicsRegime

    if traj_len % frameskip != 0:
        raise ValueError(
            f"traj_len={traj_len} must be divisible by frameskip={frameskip} so "
            f"frame subsampling and action chunking land exactly on model steps."
        )

    # Disjoint seed ranges per regime (unchanged spirit from the original code,
    # extended to cover R0 too since PhysicsRegime now supports all three).
    # seed_offset additionally separates independent calls for the same regime
    # (e.g. train vs. eval trajectories) so they never draw the same seeds --
    # without it, two calls both starting traj_idx at 0 would silently overlap.
    seed_base = {"R0": 2000, "R1": 0, "R2": 1000}.get(regime, 0) + seed_offset

    trajectories = []
    for traj_idx in range(num_trajs):
        for attempt in range(max_tries):
            seed = seed_base + traj_idx * max_tries + attempt
            # with_velocity=True: matches scripts/run_e1.py and the shipped
            # eval YAML (env.with_velocity: true) -- the checkpoint's
            # preprocessor.proprio_std is sized for the 4-dim (x,y,vx,vy)
            # proprio this produces, not the 2-dim default. Confirmed
            # empirically (RuntimeError on proprio_std shape mismatch without
            # it) -- see E0_IMPLEMENTATION_PLAN.md T3.
            env = PhysicsRegime(PushTEnv(render_size=224, with_velocity=True), regime)
            env.seed(seed)
            obs, state = env.reset()
            imgs = [obs["visual"]]  # RGB array [224, 224, 3]
            proprios = [obs["proprio"]]
            raw_actions = []
            total_contacts = 0

            rs = np.random.RandomState(seed)
            block_xy = state[2:4]
            # One persistent random target near the block, sampled once per
            # trajectory — NOT resampled every step, which is what keeps the
            # walk "aimed" instead of cancelling out like the old scheme.
            target = block_xy + rs.uniform(-40.0, 40.0, size=(2,))
            agent_xy = obs["proprio"][:2]

            for _ in range(traj_len):
                # Undo the env's own act*action_scale (+= agent.position when
                # relative=True, the default) to get a raw action that aims
                # the agent at `target` this step.
                direction = (target - agent_xy) / env.action_scale
                noise = rs.normal(0.0, 0.15, size=(2,))
                # ACTION_GAIN: without this, "aim hard at the target every
                # step" produces raw actions with std ~[0.46, 0.43] (often
                # saturating at the +-1 clip bound while the agent is still
                # far away) -- but the checkpoint's own preprocessor.action_std
                # is ~[0.20, 0.20] (real demonstration data takes gentler,
                # smaller per-step actions), so un-scaled actions land ~2.2x
                # outside the distribution the model was calibrated on, which
                # after normalize_actions stretches to std ~2.1-2.3 and values
                # up to +-5 std. Confirmed via direct audit (code-review.md
                # Bug #6d). ACTION_GAIN=0.25 was tuned empirically (clipping
                # is nonlinear, so std doesn't scale linearly with gain) to
                # bring raw action std to ~[0.215, 0.204], matching the
                # checkpoint's ~[0.202, 0.200] almost exactly. This alone
                # drops single-attempt contact rate to ~43%, but the
                # rejection-sampling retry above (max_tries=8) still reaches
                # ~99% expected overall success (1-(1-0.43)^8).
                ACTION_GAIN = 0.25
                act = np.clip((direction + noise) * ACTION_GAIN, -1.0, 1.0)
                obs, reward, done, info = env.step(act)
                agent_xy = obs["proprio"][:2]
                total_contacts += info["n_contacts"]
                imgs.append(obs["visual"])
                proprios.append(obs["proprio"])
                raw_actions.append(act)

            if total_contacts > 0 or attempt == max_tries - 1:
                break  # accept: real contact happened, or retries exhausted

        imgs_np = np.stack(imgs, axis=0)            # [T_raw+1, 224, 224, 3]
        proprios_np = np.stack(proprios, axis=0)    # [T_raw+1, proprio_dim]
        acts_np = np.stack(raw_actions, axis=0)     # [T_raw, 2]

        # Subsample frames to the model time base: keep every `frameskip`-th
        # raw frame (plus frame 0) -> T_raw/frameskip + 1 frames, matching the
        # chunked actions below one-to-one.
        keep_idx = list(range(0, traj_len + 1, frameskip))
        imgs_sub = imgs_np[keep_idx]                # [T_model+1, 224, 224, 3]
        proprios_sub = proprios_np[keep_idx]        # [T_model+1, proprio_dim]

        # Encode visual+proprio TOGETHER via the wrapper's own encode() -- the
        # same obs-dict layout GC_Agent.act() feeds it (raw uint8 visual, raw
        # proprio; encode() does its own /255 + preprocessor.transform +
        # normalize_proprios).
        visual_t = torch.from_numpy(imgs_sub.copy()).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
        proprio_t = torch.from_numpy(proprios_sub.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            enc = world_model.encode({"visual": visual_t, "proprio": proprio_t})
            enc_out = enc["visual"].squeeze(0).squeeze(1).flatten(1, 2)  # [T_model+1, 256, 384]
            proprio_enc = enc["proprio"].squeeze(0)                     # [T_model+1, P_tok, D_p]

        # Chunk raw actions [T_raw, 2] -> model actions [T_raw/frameskip, 10].
        act_tensor_raw = torch.from_numpy(acts_np).float()
        act_norm = preprocessor.normalize_actions(act_tensor_raw.unsqueeze(0)).squeeze(0)  # [T_raw, 2]
        act_model = act_norm.reshape(traj_len // frameskip, frameskip * 2).to(device)      # [T_model, 10]

        trajectories.append({
            "encoder_output": enc_out,
            "actions": act_model,
            "proprio": proprio_enc,
            "seed": seed,  # the ACCEPTED seed (whichever attempt broke the retry loop) --
                            # written to e0_seed_manifest.json so E1/downstream experiments
                            # can be audited for zero overlap with E0's train/eval seeds.
        })

    return trajectories


def evaluate_e0_chart(world_model, chart: Chart, val_trajectories: list[dict],
                       motion_gate: float | None = None) -> tuple[float, float]:
    """
    Evaluates a fine-tuned chart on held-out validation trajectories.

    Args:
        world_model: EncPredWM wrapper (the object torch.hub.load returns —
                     NOT .model). _open_loop_rollout/umf need the wrapper for
                     the canonical unroll(); chart apply/restore and
                     cfgs_loss-based loss reach the inner VideoWM via
                     world_model.model.
        motion_gate: Informative-chunk threshold (gate G6) — see
                     atlas.score.compute_motion_gate. None = skip gate.

    Returns (avg_eval_loss, avg_eval_umf).
    """
    import numpy as np
    from atlas.harness import compute_trajectory_loss
    from atlas.score import _open_loop_rollout, _make_z_ctxt, umf

    losses = []
    umf_scores = []

    with torch.no_grad():
        for traj in val_trajectories:
            enc_out = traj["encoder_output"]
            actions = traj["actions"]
            z_vis = enc_out[0]
            proprio_ctxt = traj["proprio"][0:1].unsqueeze(0)  # [1, 1, P_tok, D_p]
            z_ctxt = _make_z_ctxt(world_model, z_vis, proprio_ctxt)

            # Apply chart specifically for the open-loop rollout, then restore.
            # apply_() is INSIDE the try too (not just the rollout) -- if
            # apply_() itself raises partway through (e.g. a lora4 chart with
            # a corrupted _param_names list), restore_() must still run or the
            # predictor is left with a permanent, never-cleaned-up partial
            # parametrization that corrupts every subsequent chart/fine-tune
            # sharing this predictor object. See code-review.md Bug #6f.
            try:
                chart.apply_(world_model.model.predictor)
                z_preds = _open_loop_rollout(world_model, z_ctxt, actions)
            finally:
                chart.restore_(world_model.model.predictor)

            loss = compute_trajectory_loss(world_model.model, z_preds, enc_out[1:])
            losses.append(loss.item())

            # umf internally handles applying and restoring the chart,
            # expecting the predictor to start in baseline state.
            score = umf(chart, world_model, enc_out, actions, motion_gate=motion_gate,
                        proprio_ctxt=proprio_ctxt)
            if score is not None:
                umf_scores.append(score)

    avg_eval_loss = float(np.mean(losses)) if losses else float("nan")
    avg_eval_umf = float(np.mean(umf_scores)) if umf_scores else float("nan")
    return avg_eval_loss, avg_eval_umf


def main() -> None:
    parser = argparse.ArgumentParser(description="E0: Adapter capacity fine-tune.")
    parser.add_argument("--kinds", nargs="+", default=["ln_act", "lora4", "full"],
                        choices=["ln_act", "lora4", "full"])
    parser.add_argument("--regimes", nargs="+", default=["R1", "R2"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--num-train-trajs", type=int, default=3,
                         help="Number of TRAINING trajectories. run_e0_finetune() loops over "
                              "every trajectory on every step, so compute scales linearly "
                              "with this too.")
    parser.add_argument("--train-traj-len", type=int, default=10,
                         help="Steps per TRAINING trajectory. run_e0_finetune() backprops "
                              "through the full open-loop unroll, so GPU memory scales "
                              "~linearly with this (measured ~0.27GB/step for kind=full on "
                              "this predictor) -- kept short to fit a 6GB GPU. See "
                              "code-review.md Bug #6e.")
    parser.add_argument("--eval-traj-len", type=int, default=50,
                         help="Steps per EVAL (held-out) trajectory. Runs under torch.no_grad "
                              "so length is cheap here -- needs to be long because 10 was too "
                              "short for properly-calibrated (gentle) actions to accumulate "
                              "enough displacement for UMF to be well-behaved. See "
                              "code-review.md Bug #6d. DINO-WM's own training trajectories "
                              "are ~100 steps.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e0")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="atlas-e0", help="WandB project name")
    args = parser.parse_args()

    wandb_group = f"e0_experiment_{int(time.time())}"

    print("Loading dino_wm_pusht from local hub...")
    # Raw ATLAS_HOME env var, not Path(__file__)-relative: on Modal the code
    # (/src) and the volume-mounted hub cache (/atlas_root/hub) live at
    # different paths -- see run_e0_planning.py's HUB_PATH for the same fix.
    import os
    _atlas_home = os.environ.get("ATLAS_HOME", str(atlas.ATLAS_HOME))
    hub_path = str(Path(_atlas_home) / "hub" / "hub" / "facebookresearch_jepa-wms_main")
    model, prep = torch.hub.load(
        hub_path, "dino_wm_pusht",
        source="local",
        force_reload=False, trust_repo=True,
    )
    # `model` is the EncPredWM wrapper (owns ctxt_window/proprio_mode and the
    # canonical unroll() this checkpoint was validated with -- see
    # E0_IMPLEMENTATION_PLAN.md T1/T2). `wm` is the inner VideoWM, needed for
    # predictor state-dict ops, Chart construction, and encoder freezing --
    # wrapper.to(device) moves it too, since it's a registered submodule.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = model.to(device)
    wm = wrapper.model if hasattr(wrapper, "model") else wrapper
    for p in wm.encoder.parameters():
        p.requires_grad_(False)

    # Pristine snapshot of the predictor, taken once before any fine-tuning.
    # run_e0_finetune() mutates wm.predictor in-place for ln_act/full charts and
    # never restores it (Chart.restore_ just re-applies the already-updated chart
    # params, not the original checkpoint weights) — without reloading this
    # snapshot before every fine-tune, later charts would train on top of earlier
    # charts' weights instead of the pretrained baseline. See code-review.md #1.
    pristine_predictor_state = copy.deepcopy(wm.predictor.state_dict())

    # [Debug print statement] Print setup info
    print(f"\nE0: {len(args.kinds)} kinds × {len(args.regimes)} regimes "
          f"= {len(args.kinds) * len(args.regimes)} fine-tunes", flush=True)
    print(f"Output: {args.out}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    seed_manifest: dict = {}  # regime -> {"train": [...], "eval": [...]} accepted seeds --
                              # written to e0_seed_manifest.json so downstream experiments
                              # (E1) can be audited for zero seed overlap with E0.

    for regime in args.regimes:
        results[regime] = {}
        # [Debug print statement] Print regime start
        print(f"\n── Regime {regime} ─────────────────────────────────────────────", flush=True)
        print(f"  Loading trajectories for regime {regime}...", flush=True)
        # Separate lengths: training backprops through the full unroll (memory
        # scales with length, kept short); eval runs under no_grad (cheap, kept
        # long since UMF needs real accumulated displacement). See --train-traj-len
        # / --eval-traj-len help text and code-review.md Bug #6d/#6e.
        train_trajectories = load_regime_trajectories(
            wrapper, prep, regime, num_trajs=args.num_train_trajs, traj_len=args.train_traj_len,
            device=device, seed_offset=0)
        val_trajectories = load_regime_trajectories(
            wrapper, prep, regime, num_trajs=2, traj_len=args.eval_traj_len, device=device,
            seed_offset=10_000)
        seed_manifest[regime] = {
            "train": [t["seed"] for t in train_trajectories],
            "eval": [t["seed"] for t in val_trajectories],
        }
        # [Debug print statement] Print trajectories loaded
        print(f"  [Debug] Loaded {len(train_trajectories)} train & {len(val_trajectories)} eval trajectories for {regime}", flush=True)

        # Gate G6: informative-chunk threshold, computed once from this
        # regime's own training displacements (10th percentile) -- wired into
        # every umf() call below instead of the motion_gate=None the eval
        # path has always used (E0_IMPLEMENTATION_PLAN.md T5).
        train_displacements = torch.tensor([
            (t["encoder_output"][-1] - t["encoder_output"][0]).norm(p="fro").item()
            for t in train_trajectories
        ])
        motion_gate = compute_motion_gate(train_displacements)
        print(f"  [Debug] motion_gate (10th pct of train displacement) = {motion_gate:.4f}", flush=True)

        for kind in args.kinds:
            # Reset the predictor to its pristine pretrained state before every
            # kind's fine-tune/eval. Without this, later charts train on top of
            # earlier charts' weights instead of the checkpoint baseline — see
            # code-review.md #1.
            wm.predictor.load_state_dict(pristine_predictor_state)

            loss_file = args.out / f"loss_{kind}_{regime}.json"
            chart_file = args.out / f"chart_{kind}_{regime}.pt"

            if loss_file.exists() and chart_file.exists():
                # [Resume support] Skip fine-tuning if results already exist on disk / volume
                print(f"  ⏩ [Resume] {kind}_{regime} already completed. Loading cached result...", flush=True)
                losses = json.loads(loss_file.read_text())
                final_loss = losses[-1] if losses else float("nan")
                try:
                    chart = Chart.load(chart_file, wm.predictor)
                    n_params = chart.n_params()  # real parameter count, not a tensor count (T12 #9)
                    eval_loss, eval_umf = evaluate_e0_chart(wrapper, chart, val_trajectories, motion_gate)
                except Exception as e:
                    # Print the real error instead of silently returning NaN --
                    # a swallowed exception here previously masked a real bug
                    # (Chart.load()'s corrupted _param_names for lora4, see
                    # code-review.md Bug #6f) that also left the shared
                    # predictor in a corrupted state for every subsequent
                    # kind/regime, only surfacing as a confusing crash much
                    # later. Reset the predictor defensively so a failure here
                    # can't cascade the same way even if a new bug appears.
                    print(f"    ⚠️ [Resume] Failed to re-evaluate cached {kind}_{regime}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    wm.predictor.load_state_dict(pristine_predictor_state)
                    n_params = 0
                    eval_loss, eval_umf = float("nan"), float("nan")
                results[regime][kind] = {
                    "train_loss": final_loss,
                    "eval_loss": eval_loss,
                    "eval_umf": eval_umf,
                    "params": n_params,
                    "status": "completed (cached)",
                }
                print(f"    Done {kind}_{regime} (Cached): Train Loss = {final_loss:.6f} | Eval Loss = {eval_loss:.6f} | Eval UMF = {eval_umf:.4f}", flush=True)
                continue

            # Initialize separate WandB run for this fine-tuning session under the shared group
            if args.wandb:
                try:
                    import wandb
                    wandb.init(
                        project=args.wandb_project,
                        group=wandb_group,
                        name=f"{kind}_{regime}",
                        reinit=True,
                        config={
                            "kind": kind,
                            "regime": regime,
                            "steps": args.steps,
                            "lr": args.lr,
                        },
                    )
                    wandb.define_metric("step")
                    wandb.define_metric("loss", step_metric="step")
                    wandb.define_metric(f"loss_{kind}_{regime}", step_metric="step")
                except Exception as e:
                    print(f"⚠️ WandB init failed for {kind}_{regime}: {e}", flush=True)

            # [Debug print statement] Print fine-tuning start
            print(f"  Fine-tuning {kind} on {regime} ({args.steps} steps)...", flush=True)
            chart = run_e0_finetune(
                world_model=wrapper,
                trajectories=train_trajectories,
                kind=kind,
                regime=regime,
                n_steps=args.steps,
                lr=args.lr,
                out_dir=args.out,
            )

            # Held-out Evaluation on Validation Trajectories
            eval_loss, eval_umf = evaluate_e0_chart(wrapper, chart, val_trajectories, motion_gate)

            if args.wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"eval_loss": eval_loss, "eval_umf": eval_umf})
                        wandb.finish()
                except Exception:
                    pass
            
            # Compute evaluation metrics
            losses = json.loads(loss_file.read_text())
            final_loss = losses[-1] if losses else float("nan")
            results[regime][kind] = {
                "train_loss": final_loss,
                "eval_loss": eval_loss,
                "eval_umf": eval_umf,
                "params": chart.n_params(),  # real parameter count, not a tensor count (T12 #9)
                "status": "completed",
            }
            # [Debug print statement] Print fine-tuning & eval completed
            print(f"    Done {kind}_{regime}: Train Loss = {final_loss:.6f} | Eval Loss = {eval_loss:.6f} | Eval UMF = {eval_umf:.4f}", flush=True)

    # Save summary results
    results_json = args.out / "results.json"
    results_json.write_text(json.dumps(results, indent=2))

    # Seed manifest: audit trail proving E1 (or any downstream experiment) uses
    # seeds disjoint from every trajectory that went into fine-tuning/evaluating
    # these charts. E1's own seeds (atlas.streams.paired_seed, SHA256-derived)
    # live in a completely different space from these small deterministic
    # integers, but this manifest makes that an auditable fact, not a claim.
    seed_manifest_json = args.out / "e0_seed_manifest.json"
    seed_manifest_json.write_text(json.dumps(seed_manifest, indent=2))
    print(f"  - Seed manifest : {seed_manifest_json}")

    # Generate Markdown Table T5
    md_lines = [
        "# Table T5: E0 Adapter Capacity Benchmarks",
        "",
        "| Regime | Adapter Kind | Target Params | Train Loss | Eval Loss | Eval UMF | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for regime, kinds in results.items():
        for kind, metrics in kinds.items():
            md_lines.append(
                f"| {regime} | `{kind}` | {metrics['params']:,} | {metrics['train_loss']:.6f} | {metrics['eval_loss']:.6f} | {metrics['eval_umf']:.4f} | {metrics['status']} |"
            )
    
    results_md = args.out / "results.md"
    results_md.write_text("\n".join(md_lines))

    # Generate local loss curves plot from saved JSON files
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4), dpi=300)
        colors = {
            "ln_act": "#1f77b4",
            "lora4": "#ff7f0e",
            "full": "#2ca02c",
        }
        linestyles = {
            "R1": "-",
            "R2": "--",
        }

        for regime in args.regimes:
            for kind in args.kinds:
                loss_file = args.out / f"loss_{kind}_{regime}.json"
                if loss_file.exists():
                    losses = json.loads(loss_file.read_text())
                    steps = list(range(1, len(losses) + 1))
                    plt.plot(
                        steps,
                        losses,
                        label=f"{kind} ({regime})",
                        color=colors.get(kind, None),
                        linestyle=linestyles.get(regime, "-"),
                        alpha=0.85,
                        linewidth=1.5,
                    )

        plt.xlabel("Gradient Step (1..2000)")
        plt.ylabel("Prediction Loss")
        plt.title("E0 Adapter Capacity: Fine-Tuning Loss Curves")
        plt.legend(loc="upper right", frameon=True)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()

        plot_png = args.out / "e0_loss_curves.png"
        plot_pdf = args.out / "e0_loss_curves.pdf"
        plt.savefig(plot_png)
        plt.savefig(plot_pdf)
        plt.close()
        print(f"  - Loss Plot PNG: {plot_png}")
        print(f"  - Loss Plot PDF: {plot_pdf}")
    except Exception as e:
        print(f"⚠️ Loss plot generation skipped: {e}")

    print(f"\n✅ E0 Experiment complete! Results saved to {args.out}")
    print(f"  - Summary JSON : {results_json}")
    print(f"  - Table T5 MD  : {results_md}")


if __name__ == "__main__":
    main()

