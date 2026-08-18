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
    predictor,
    encoder_output: torch.Tensor,   # [T+1, N_patches, D] — already encoded
    actions: torch.Tensor,          # [T, action_dim]
    motion_gate: float | None = None,
) -> float | None:
    """
    Compute UMF for *chart* on chunk Q = (encoder_output, actions).

    Args:
        chart:           The chart to evaluate.
        predictor:       The frozen JEPA predictor (modified in-place by chart.apply_).
        encoder_output:  Pre-computed latent states z_0 … z_T, shape [T+1, N, D].
        actions:         Executed actions a_0 … a_{T-1}, shape [T, action_dim].
        motion_gate:     If not None, return None when observed displacement
                         ‖z_T − z_0‖_F  ≤  motion_gate  (uninformative chunk).

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
    chart.apply_(predictor)
    try:
        z_hat = _open_loop_rollout(predictor, z0, actions)   # [T, N, D]
    finally:
        chart.restore_(predictor)

    # ── UMF numerator: Σ_k ‖ẑ_k − z_k‖² ─────────────────────────────────────
    z_targets = z[1:]  # [T, N, D]
    numerator   = (z_hat - z_targets).pow(2).sum().item()

    # ── UMF denominator: Σ_k ‖z_k − z_0‖² ───────────────────────────────────
    displacement = (z_targets - z0.unsqueeze(0)).pow(2).sum().item()

    if displacement == 0.0:
        # Denominator is exactly 0 → chunk is static; return None (gate should
        # have caught this, but be defensive).
        return None

    return numerator / displacement


@torch.no_grad()
def _open_loop_rollout(predictor, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Roll out the predictor open-loop from z0 for len(actions) steps.

    Args:
        predictor: the (temporarily chart-swapped) JEPA predictor.
        z0:        initial latent state [N_patches, D].
        actions:   action sequence [T, action_dim].

    Returns:
        Predicted latent states ẑ_1 … ẑ_T, shape [T, N, D].
    """
    T = actions.shape[0]
    z_preds = []
    z_cur = z0.unsqueeze(0)  # [1, N, D] — add batch dim
    for t in range(T):
        a_t = actions[t].unsqueeze(0)  # [1, action_dim]
        z_next = predictor(z_cur, a_t)
        z_preds.append(z_next.squeeze(0))
        z_cur = z_next
    return torch.stack(z_preds, dim=0)  # [T, N, D]


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
