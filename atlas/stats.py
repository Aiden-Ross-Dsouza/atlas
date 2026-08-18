"""
stats.py — Statistical utilities.

Functions exactly as specified in implementation plan §8:
  normalised_recovery   — with min_spread guard
  paired_bootstrap      — paired per-episode binary outcomes
  mcnemar_paired        — exact McNemar test (statsmodels)
"""

from __future__ import annotations

import numpy as np


def normalised_recovery(
    sr_fit: float,
    sr_oracle: float,
    sr_random: float,
    min_spread: float = 0.10,
) -> float | None:
    """
    Normalised recovery of *sr_fit* relative to oracle and random.

    Args:
        sr_fit:      Success rate of the method being evaluated.
        sr_oracle:   Success rate of oracle routing.
        sr_random:   Success rate of random routing.
        min_spread:  Minimum oracle–random spread to report a value.
                     Returns None if spread < min_spread (denominator too small).

    Returns:
        Normalised recovery in [0, 1] (may exceed 1 if sr_fit > sr_oracle),
        or None if the denominator is too small to be meaningful.
    """
    spread = sr_oracle - sr_random
    if spread < min_spread:
        return None
    return (sr_fit - sr_random) / spread


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n: int = 10_000,
    seed: int = 0,
    ci: float = 95.0,
) -> tuple[float, tuple[float, float]]:
    """
    Paired bootstrap confidence interval for the mean difference a − b.

    Args:
        a:    Binary outcomes for arm A, shape [N_episodes].
        b:    Binary outcomes for arm B, in the SAME episode order as *a*.
        n:    Number of bootstrap resamples.
        seed: RNG seed for reproducibility.
        ci:   Confidence level (default 95 → [2.5, 97.5] percentiles).

    Returns:
        (mean_diff, (lower_ci, upper_ci))

    Raises:
        ValueError: if a and b have different lengths.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape; got {a.shape} vs {b.shape}")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    bootstrap_means = d[idx].mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo, hi = float(np.percentile(bootstrap_means, alpha)), float(np.percentile(bootstrap_means, 100.0 - alpha))
    return float(d.mean()), (lo, hi)


def mcnemar_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Exact McNemar test for paired binary outcomes.

    Args:
        a: Binary outcomes for arm A, shape [N_episodes].
        b: Binary outcomes for arm B, in the SAME episode order.

    Returns:
        Two-sided p-value.

    Raises:
        ImportError: if statsmodels is not installed.
        ValueError:  if a and b have different lengths or are not binary.
    """
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for McNemar test. "
            "Install it with: uv pip install statsmodels"
        ) from e

    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape; got {a.shape} vs {b.shape}")

    table = [
        [(( a) & ( b)).sum(), (( a) & (~b)).sum()],
        [((~a) & ( b)).sum(), ((~a) & (~b)).sum()],
    ]
    result = mcnemar(table, exact=True)
    return float(result.pvalue)


def success_rate_ci(
    outcomes: np.ndarray,
    ci: float = 95.0,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """
    Bootstrap confidence interval for a success rate.

    Args:
        outcomes:    Binary success/failure array, shape [N].
        ci:          Confidence level.
        n_bootstrap: Number of resamples.
        seed:        RNG seed.

    Returns:
        (mean, (lower, upper))
    """
    outcomes = np.asarray(outcomes, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(outcomes), (n_bootstrap, len(outcomes)))
    bootstrap_means = outcomes[idx].mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(bootstrap_means, alpha))
    hi = float(np.percentile(bootstrap_means, 100.0 - alpha))
    return float(outcomes.mean()), (lo, hi)
