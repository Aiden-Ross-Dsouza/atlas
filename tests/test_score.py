"""
tests/test_score.py — Unit tests for atlas.score (UMF computation).
"""

import torch
import pytest
from unittest.mock import MagicMock

from atlas.score import umf, compute_motion_gate


def make_dummy_predictor(output_shape):
    """A predictor that returns zeros of the specified shape."""
    predictor = MagicMock()
    predictor.named_parameters.return_value = []
    predictor.return_value = torch.zeros(*output_shape).unsqueeze(0)
    predictor.side_effect = lambda z, a: torch.zeros_like(z)
    return predictor


def make_dummy_chart():
    chart = MagicMock()
    chart.apply_ = MagicMock()
    chart.restore_ = MagicMock()
    chart._param_names = []
    return chart


def test_umf_static_chunk_returns_none():
    """Static chunk (no motion) → denominator 0 → UMF returns None."""
    N, D, T = 10, 16, 3
    z0 = torch.randn(N, D)
    encoder_output = z0.unsqueeze(0).expand(T + 1, -1, -1).clone()
    actions = torch.zeros(T, 2)

    chart = make_dummy_chart()
    predictor = MagicMock()
    predictor.side_effect = lambda z, a: z  # identity predictor

    result = umf(chart, predictor, encoder_output, actions, motion_gate=None)
    assert result is None, f"Expected None for static chunk, got {result}"


def test_umf_gated_chunk_returns_none():
    """Chunk below motion_gate → returns None."""
    N, D, T = 10, 16, 3
    z0 = torch.randn(N, D)
    z_end = z0 + 0.001  # tiny displacement
    encoder_output = torch.stack([z0] + [z0] * (T - 1) + [z_end])
    actions = torch.zeros(T, 2)

    chart = make_dummy_chart()
    predictor = MagicMock()
    predictor.side_effect = lambda z, a: z

    # Gate is larger than the displacement → should be gated.
    result = umf(chart, predictor, encoder_output, actions, motion_gate=1.0)
    assert result is None


def test_umf_shape_validation():
    """Wrong encoder_output shape raises ValueError."""
    chart = make_dummy_chart()
    predictor = MagicMock()
    with pytest.raises(ValueError, match="encoder_output must be"):
        umf(chart, predictor, torch.randn(5, 16), torch.zeros(3, 2))


def test_umf_frame_count_mismatch():
    """T+1 frame count mismatch raises ValueError."""
    chart = make_dummy_chart()
    predictor = MagicMock()
    with pytest.raises(ValueError, match="frames"):
        umf(chart, predictor, torch.randn(5, 10, 16), torch.zeros(3, 2))


def test_compute_motion_gate_empty():
    """Empty tensor raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        compute_motion_gate(torch.tensor([]))


def test_compute_motion_gate_percentile():
    """10th percentile of [0..9] is 0.9."""
    displacements = torch.arange(10, dtype=torch.float32)
    gate = compute_motion_gate(displacements, percentile=10.0)
    assert abs(gate - 0.9) < 0.1  # approximate due to linear interpolation
