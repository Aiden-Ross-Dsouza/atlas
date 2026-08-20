"""
score.py — UMF (Unexplained Motion Fraction) and informative-chunk gating.

C3: UMF(c; Q) = Σ_k ‖ẑ^c_k − z_k‖² / Σ_k ‖z_k − z₀‖²

  Numerator:   open-loop latent prediction error under chart c.
  Denominator: total latent motion in chunk Q relative to z₀ (first frame).
  Result:      0 = perfect predictor; ≈1 = no better than predicting stasis.

Prequential invariant (enforced here, not just documented):
  score() is a PURE FORWARD PASS with no_grad. Refinement happens AFTER scoring.

Informative-chunk gate:
  A chunk is informative iff its observed displacement exceeds a threshold
  computed from the training set (10th percentile of training displacement).
  Non-informative chunks return None; callers must not count them as strikes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atlas.chart import Chart


@torch.no_grad()
def umf(
    chart: "Chart",
    world_model,
    encoder_output: torch.Tensor,   # [T+1, N_patches, D] — already encoded
    actions: torch.Tensor,          # [T, wm.action_dim] — model actions (raw_dim * frameskip)
    motion_gate: float | None = None,
) -> float | None:
    """
    Compute UMF for *chart* on chunk Q = (encoder_output, actions).

    Args:
        chart:           The chart to evaluate.
        world_model:     VideoWM instance (EncPredWM.model from torch.hub). The chart
                         parameters are swapped into world_model.predictor (ViTPredictor).
        encoder_output:  Pre-computed visual latent states z_0 … z_T, shape [T+1, N, D].
        actions:         Executed raw actions a_0 … a_{T-1}, shape [T, action_dim].
        motion_gate:     If not None, return None when observed displacement
                         ||z_T - z_0||_F <= motion_gate (uninformative chunk).

    Returns:
        UMF value in [0, ∞), or None if chunk is uninformative.

    Raises:
        ValueError: if tensor shapes are inconsistent.
    """
    if encoder_output.ndim != 3:
        raise ValueError(
            f"encoder_output must be [T+1, N_patches, D], got {encoder_output.shape}"
        )
    T = actions.shape[0]
    if encoder_output.shape[0] != T + 1:
        raise ValueError(
            f"encoder_output has {encoder_output.shape[0]} frames but {T} actions imply {T+1} frames"
        )

    z = encoder_output  # [T+1, N, D]
    z0 = z[0]          # [N, D] — first frame = null predictor output

    # ── Informative-chunk gate ────────────────────────────────────────────────
    observed_displacement = (z[-1] - z0).norm(p="fro").item()
    if motion_gate is not None and observed_displacement <= motion_gate:
        return None  # uninformative chunk — caller must not count as a strike

    # ── Open-loop unroll under chart ──────────────────────────────────────────
    # chart.apply_/restore_ swap params on ViTPredictor (world_model.predictor).
    predictor = world_model.predictor
    chart.apply_(predictor)
    try:
        z_hat = _open_loop_rollout(world_model, z0, actions)   # [T, N, D]
    finally:
        chart.restore_(predictor)

    # ── UMF numerator: Σ_k ||ẑ_k − z_k||² ───────────────────────────────────
    z_targets = z[1:]  # [T, N, D]
    numerator   = (z_hat - z_targets).pow(2).sum().item()

    # ── UMF denominator: Σ_k ||z_k − z_0||² ─────────────────────────────────
    displacement = (z_targets - z0.unsqueeze(0)).pow(2).sum().item()

    if displacement == 0.0:
        # Denominator is exactly 0 → chunk is static; return None (gate should
        # have caught this, but be defensive).
        return None

    return numerator / displacement


def _open_loop_rollout(world_model, z_vis: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Roll out the predictor open-loop from z_vis for len(actions) steps.

    This uses VideoWM.forward_pred() — the correct entry point for the dino_wm
    architecture — rather than calling ViTPredictor.forward() directly.

    ViTPredictor expects a single pre-concatenated [vis ‖ proprio ‖ action] tensor
    assembled by VideoWM.concat_obs_act() after encode_act(). Calling the bare
    predictor with raw (z, a) arguments is not a valid call signature.

    Args:
        world_model: VideoWM instance (accessible as EncPredWM.model from torch.hub).
        z_vis:       Initial visual latent [N_patches, D] (one frame, no batch).
        actions:     Raw action sequence [T, action_dim].

    Returns:
        Predicted visual latent states ẑ_1 … ẑ_T, shape [T, N_patches, D].
    """
    from einops import rearrange

    T = actions.shape[0]
    grid = world_model.grid_size      # e.g. 16 for ViT-S/14 @ 224
    D = z_vis.shape[-1]               # 384

    # Reshape z_vis [N, D] → [B=1, tau=1, V=1, H, W, D]
    z_cur = z_vis.reshape(1, 1, 1, grid, grid, D)

    # Encode all raw actions at once → [B=1, T, action_tokens, D] or [B=1, T, A]
    a_seq = actions.unsqueeze(0)                          # [1, T, action_dim]
    act_feats_all = world_model.encode_act(a_seq)         # [1, T, ...]

    z_preds = []
    act_buf = None
    proprio_buf = None
    if getattr(world_model, "proprio_encoder", None) is not None:
        prop_dim = world_model.proprio_encoder.embed_dim
        # Create a dummy proprio feature of shape [1, 1, grid * grid, prop_dim]
        prop_feat = torch.zeros(1, 1, grid * grid, prop_dim, device=z_vis.device)
        proprio_buf = prop_feat

    for t in range(T):
        new_act = act_feats_all[:, t : t + 1]            # [1, 1, ...]
        act_buf = new_act if act_buf is None else torch.cat([act_buf, new_act], dim=1)
        
        # Also accumulate dummy proprio if needed
        curr_prop = None
        if proprio_buf is not None:
            curr_prop = proprio_buf.expand(1, act_buf.shape[1], -1, -1)

        pred_video_features, _, _ = world_model.forward_pred(
            z_cur[:, -1:],            # [1, 1, V, H, W, D] — one frame context
            act_buf[:, -1:],          # [1, 1, ...] — one action step
            curr_prop[:, -1:] if curr_prop is not None else None,
        )
        # pred_video_features: [1, 1, V, H, W, D]  (T_pred=1)
        next_z = pred_video_features[:, -1:]             # [1, 1, V, H, W, D]
        z_preds.append(next_z.reshape(grid * grid, D))   # [N, D]
        z_cur = next_z

    return torch.stack(z_preds, dim=0)                   # [T, N, D]


def compute_motion_gate(training_displacements: torch.Tensor, percentile: float = 10.0) -> float:
    """
    Compute the informative-chunk gate threshold from training data.

    Args:
        training_displacements: 1-D tensor of per-chunk Frobenius displacements
                                collected from the training dataset.
        percentile:             Gate chunks below this percentile (default 10th).

    Returns:
        Scalar threshold; chunks with displacement ≤ this value return UMF=None.
    """
    if training_displacements.numel() == 0:
        raise ValueError("training_displacements is empty; cannot compute motion gate.")
    return float(torch.quantile(training_displacements.float(), percentile / 100.0).item())
