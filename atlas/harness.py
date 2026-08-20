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
    for step in range(n_steps):
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

        # [WandB Logging] Log to active WandB run if initialized
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    f"e0/{regime}/{kind}/loss": avg_loss,
                    "e0/step": step + 1,
                    "kind": kind,
                    "regime": regime,
                })
        except ImportError:
            pass

        # [Debug print statement] Print progress every 100 steps and on step 1
        if (step + 1) == 1 or (step + 1) % 100 == 0 or (step + 1) == n_steps:
            print(f"    [Debug] [{kind}_{regime}] Step {step+1}/{n_steps} - Loss: {avg_loss:.6f}", flush=True)

    chart.update_from_predictor_(predictor)
    chart.restore_(predictor)

    chart_path = out_dir / f"chart_{kind}_{regime}.pt"
    chart.save(chart_path)
    (out_dir / f"loss_{kind}_{regime}.json").write_text(json.dumps(loss_log))
    return chart


# ── E1 — Fitness routing ──────────────────────────────────────────────────────

def run_e1_episode(
    library: Library,
    predictor,
    encoder,
    env,
    router: RouterKind,
    episode_seed: int,
    n_warmup_replans: int,
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

    Warmup: 2 replans under c₀. Then route using *router* for the rest.
    Returns per-episode metrics dict (also written to JSONL).

    This function is a HARNESS STUB — the actual planning loop depends on
    the jepa-wms CEM planner API. The caller must integrate this with the
    upstream planning eval code from jepa-wms. See scripts/run_e1.py for
    the integration point.
    """
    raise NotImplementedError(
        "run_e1_episode() is an integration stub. "
        "Wire this into the jepa-wms planning eval loop in scripts/run_e1.py. "
        "See the implementation plan §7.2 for the required episode structure."
    )


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
