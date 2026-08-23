"""
harness.py — Offline evaluation harness for E0, E1, E5.

E0 (adapter capacity):
  - Fine-tune chart kinds offline on regime trajectories.
  - Evaluate UMF and planning success in-regime.

E1 (fitness routing):
  - Charts fixed; 2 warmup replans under c₀, then route and plan.
  - Same seeds across all routers (G5 guaranteed by streams.paired_seed).

E5 (cross-policy):
  - M[i,j] = chart i's UMF on chunks from chart j's plans.

All harness functions write per-episode JSONL to the output directory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from atlas.chart import Chart, ChartKind
from atlas.library import Library
from atlas.score import umf as compute_umf
from atlas.router import route, RouterKind
from atlas import LOGS_DIR


def compute_trajectory_loss(world_model, z_preds: torch.Tensor, z_targets: torch.Tensor) -> torch.Tensor:
    """Computes loss on latent predictions following VideoWM.compute_loss configuration."""
    import torch.nn.functional as F
    if getattr(world_model, "cfgs_loss", None):
        cfgs = world_model.cfgs_loss
        cos_w = cfgs.get("cos_loss_weight", 0.0)
        l1_w = cfgs.get("l1_loss_weight", 0.0)
        l2_w = cfgs.get("l2_loss_weight", 1.0)
        smooth_w = cfgs.get("smooth_l1_loss_weight", 0.0)
        
        loss = torch.tensor(0.0, device=z_preds.device)
        if l2_w > 0:
            loss = loss + l2_w * (z_preds - z_targets).pow(2).mean(dim=-1).mean()
        if l1_w > 0:
            loss = loss + l1_w * (z_preds - z_targets).abs().mean(dim=-1).mean()
        if cos_w > 0:
            cos_sim = F.cosine_similarity(z_preds, z_targets, dim=-1)
            loss = loss + cos_w * (-cos_sim).mean()
        if smooth_w > 0:
            loss = loss + smooth_w * F.smooth_l1_loss(z_preds, z_targets, reduction="mean")
        return loss
    else:
        return (z_preds - z_targets).pow(2).mean(dim=-1).mean()


# ── E0 — Adapter capacity ─────────────────────────────────────────────────────

def run_e0_finetune(
    world_model,
    trajectories: list[dict],   # list of {encoder_output, actions} dicts
    kind: ChartKind,
    regime: str,
    n_steps: int,
    lr: float,
    out_dir: Path,
) -> Chart:
    """
    Offline fine-tune a chart of the given kind on provided trajectories.

    Args:
        world_model:  VideoWM instance (EncPredWM.model from torch.hub).
        trajectories: List of trajectory dicts with pre-encoded data.
        kind:         Chart kind to fine-tune.
        regime:       Regime name (for logging).
        n_steps:      Number of gradient steps.
        lr:           Learning rate.
        out_dir:      Directory to save the resulting chart and loss log.

    Returns:
        The fine-tuned Chart.
    """
    import torch.optim as optim
    from tqdm import tqdm
    from atlas.score import _open_loop_rollout

    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = world_model.predictor
    chart = Chart(predictor, kind)
    chart.apply_(predictor)
    
    # Freeze non-chart parameters and enable gradients ONLY on chart parameters
    for n, p in predictor.named_parameters():
        if kind == "lora4" and ("lora_A" in n or "lora_B" in n):
            p.requires_grad_(True)
        elif kind != "lora4" and n in chart._param_names:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    params = [p for n, p in predictor.named_parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    loss_log: list[float] = []
    pbar = tqdm(range(n_steps), desc=f"{kind}_{regime}", unit="step")
    for step in pbar:
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=next(predictor.parameters()).device)
        for traj in trajectories:
            enc_out: torch.Tensor = traj["encoder_output"]  # [T+1, N, D]
            actions: torch.Tensor = traj["actions"]         # [T, action_dim]
            z_vis = enc_out[0]
            
            # Predict visual latents open-loop
            z_preds = _open_loop_rollout(world_model, z_vis, actions) # [T, N, D]
            loss = compute_trajectory_loss(world_model, z_preds, enc_out[1:])
            total_loss = total_loss + loss
            
        (total_loss / len(trajectories)).backward()
        optimizer.step()
        avg_loss = total_loss.item() / len(trajectories)
        loss_log.append(avg_loss)
        pbar.set_postfix(loss=f"{avg_loss:.6f}")

        # [WandB Logging] Log step and loss to active WandB run if initialized
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "step": step + 1,
                    "loss": avg_loss,
                    f"loss_{kind}_{regime}": avg_loss,
                    "kind": kind,
                    "regime": regime,
                })
        except ImportError:
            pass

        # [Debug print statement] Print progress every 100 steps and on step 1.
        # pbar.write (not print) so it doesn't corrupt the tqdm bar's line.
        if (step + 1) == 1 or (step + 1) % 100 == 0 or (step + 1) == n_steps:
            pbar.write(f"    [Debug] [{kind}_{regime}] Step {step+1}/{n_steps} - Loss: {avg_loss:.6f}")

    chart.update_from_predictor_(predictor)
    chart.restore_(predictor)

    chart_path = out_dir / f"chart_{kind}_{regime}.pt"
    chart.save(chart_path)
    (out_dir / f"loss_{kind}_{regime}.json").write_text(json.dumps(loss_log))
    return chart


# ── E1 — Fitness routing ──────────────────────────────────────────────────────

def _make_obs_td(visual_hw3_uint8, proprio_vec, device: str):
    """Build a batchless-per-field TensorDict; GC_Agent.set_goal()/.act() add
    their own leading batch dim internally (see scripts/run_e0_planning.py)."""
    import numpy as np
    from tensordict import TensorDict

    visual = torch.from_numpy(visual_hw3_uint8.copy()).permute(2, 0, 1).float().unsqueeze(0)
    proprio = torch.from_numpy(np.asarray(proprio_vec, dtype=np.float32)).unsqueeze(0)
    return TensorDict({"visual": visual, "proprio": proprio}, batch_size=[]).to(device)


def _prepare_env(base_env, regime, seed: int, state) -> tuple[dict, dict]:
    """
    Reset to a controlled state with physics reapplied.

    Uses regime.reset() (PhysicsRegime.reset(), which calls _apply_physics()
    automatically) rather than PushTWrapper's reset()/step() — PushTWrapper
    discards obs["visual"] (it expects a separate PixelWrapper to re-render),
    which E1 needs for encoding. reset_to_state is set on base_env directly
    (not on the regime wrapper) since gym.Wrapper does not proxy attribute
    writes to the wrapped env.
    """
    base_env.seed(seed)
    base_env.reset_to_state = state
    return regime.reset()


def run_e1_episode(
    library: Library,
    agent,                      # GC_Agent, pre-configured with E1's CEM hyperparameters
    world_model,                # VideoWM: has .predictor, .encode_obs, .encode_act, .forward_pred
    base_env,                   # raw PushTEnv
    regime,                     # PhysicsRegime(base_env, ...) — physics fixed for the whole episode
    goal_utils,                 # PushTWrapper(base_env) — only .sample_random_init_goal_states/.eval_state used
    router: RouterKind,
    episode_seed: int,
    n_warmup_replans: int,
    n_replans_target: int,
    frameskip: int,
    num_act_stepped: int,       # model-chunk units -- matches agent's CEMPlanner.horizon units
    motion_gate: float | None,
    hysteresis: float,
    out_dir: Path,
    episode_id: str,
    *,
    regime_label: int | None = None,
    label_to_chart: dict[int, int] | None = None,
) -> dict[str, Any]:
    """
    Run one E1 episode with a specified router.

    Prequential order per replan (never reorder — see loop.py's module docstring
    for the same invariant in the full ATLAS loop): SCORE the chunk executed by
    the PREVIOUS replan -> SELECT a chart -> EXECUTE this replan with it. The
    first `n_warmup_replans` replans skip scoring/selection entirely and stay on
    library[0] (c0) — this is deliberate: E1 tests whether the router can find
    the right chart using only c0's warmup data, not whether c0 itself is a good
    fit. Charts are frozen throughout (no atlas_refine call) — E1's charts come
    fixed from E0 ("Charts from E0, fixed" per the proposal); this also means
    E1 never touches atlas.loop.atlas_step()/atlas_refine() at all.

    Regime is FIXED for the whole episode (set via `regime` before this is
    called) — E1 does not switch regimes mid-episode; that's E3/E4's stream.
    """
    device = agent.device
    predictor = world_model.predictor

    init_state, goal_state = goal_utils.sample_random_init_goal_states(episode_seed)

    goal_obs, _ = _prepare_env(base_env, regime, episode_seed, goal_state)
    agent.set_goal(_make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))

    obs, _ = _prepare_env(base_env, regime, episode_seed, init_state)

    current_idx = 0  # c0, always the starting chart
    elapsed = 0
    success = False
    selected_trace: list[int] = []
    umf_trace: list[list[float | None]] = []
    raw_steps_per_replan: list[int] = []
    prev_chunk: tuple[torch.Tensor, torch.Tensor] | None = None  # (encoder_output, actions)

    for replan_idx in range(n_replans_target):
        if replan_idx >= n_warmup_replans and prev_chunk is not None:
            enc_out, acts = prev_chunk
            current_idx, route_info = route(
                kind=router, library=library, world_model=world_model,
                encoder_output=enc_out, actions=acts, current_idx=current_idx,
                motion_gate=motion_gate, hysteresis=hysteresis,
                regime_label=regime_label, label_to_chart=label_to_chart,
            )
            umf_trace.append(route_info["scores"])
        else:
            umf_trace.append([None] * len(library))  # warmup: no scoring, stays on c0
        selected_trace.append(current_idx)

        chart = library[current_idx]
        chart.apply_(predictor)
        try:
            obs_td = _make_obs_td(obs["visual"], obs["proprio"], device)
            # steps_left must be in MODEL-chunk units -- CEMPlanner.plan() does
            # plan_length = min(self.horizon, steps_left), and self.horizon is
            # unambiguously model-chunk units. Multiplying by frameskip here
            # (an earlier version of this code did) mixes units with horizon --
            # fixed to match num_act_stepped's units instead.
            steps_left_model = (n_replans_target - replan_idx) * num_act_stepped
            action = agent.act(obs_td, steps_left=max(steps_left_model, 1))
        finally:
            chart.restore_(predictor)  # chart only needs to be applied during planning

        from einops import rearrange
        raw_actions = rearrange(action.cpu(), "t (f d) -> (t f) d", d=2)
        raw_actions = agent.preprocessor.denormalize_actions(raw_actions).numpy()

        imgs = [obs["visual"]]
        step_actions = []
        for a in raw_actions:
            obs, reward, done, info = base_env.step(a)
            imgs.append(obs["visual"])
            step_actions.append(a)
            elapsed += 1
            if goal_utils.eval_state(goal_state, info["state"])["success"]:
                success = True
                break
        raw_steps_per_replan.append(len(step_actions))

        # Encode the just-executed chunk so the NEXT replan can score against it.
        import numpy as np
        imgs_np = np.stack(imgs, axis=0)[None]                      # [1, T+1, H, W, 3]
        acts_np = np.stack(step_actions, axis=0)                    # [T, 2]
        img_tensor = agent.preprocessor.transform_obs_visual(imgs_np).to(device)
        act_tensor = agent.preprocessor.normalize_actions(
            torch.from_numpy(acts_np).float().unsqueeze(0)
        ).squeeze(0).to(device)
        with torch.no_grad():
            enc_out_dict = world_model.encode_obs({"visual": img_tensor})
            enc_out = enc_out_dict["visual"].squeeze(0).squeeze(1).flatten(1, 2)  # [T+1, N, D]
        prev_chunk = (enc_out, act_tensor)

        if success:
            break

    record = {
        "episode_id": episode_id,
        "router": router,
        "seed": episode_seed,
        "success": success,
        "elapsed_raw_steps": elapsed,
        "n_replans": len(selected_trace),
        "raw_steps_per_replan": raw_steps_per_replan,
        "selected_trace": selected_trace,
        "umf_trace": umf_trace,
        "regime_label": regime_label,
    }
    log_episode(out_dir, record)
    return record


# ── E5 — Cross-policy matrix ──────────────────────────────────────────────────

def build_cross_policy_matrix(
    library: Library,
    world_model,
    chunks_per_chart: list[list[dict]],   # chunks_per_chart[j] = chunks from chart j's plans
    motion_gate: float | None,
) -> torch.Tensor:
    """
    Compute M[i,j] = chart i's mean UMF on chunks generated by chart j's plans.

    Args:
        library:          Chart library with K charts.
        world_model:      VideoWM instance (EncPredWM.model from torch.hub).
        chunks_per_chart: List of K lists; chunks_per_chart[j] are the encoded
                          chunks collected while chart j was active.
        motion_gate:      Informative-chunk gate.

    Returns:
        [K, K] matrix of mean UMF values (NaN where all chunks are gated).
    """
    K = len(library)
    if len(chunks_per_chart) != K:
        raise ValueError(
            f"chunks_per_chart has {len(chunks_per_chart)} entries but library has {K} charts."
        )
    M = torch.full((K, K), float("nan"))
    for i, chart_i in enumerate(library):
        for j, chunk_list in enumerate(chunks_per_chart):
            scores = []
            for chunk in chunk_list:
                s = compute_umf(chart_i, world_model,
                                chunk["encoder_output"], chunk["actions"], motion_gate)
                if s is not None:
                    scores.append(s)
            if scores:
                M[i, j] = sum(scores) / len(scores)
    return M


# ── JSONL logging ─────────────────────────────────────────────────────────────

def log_episode(out_dir: Path, record: dict) -> None:
    """Append one episode record to a JSONL log file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "episodes.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
