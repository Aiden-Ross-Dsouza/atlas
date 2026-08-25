"""
streams.py — Deployment stream drivers and paired seeding.

S2 stream: A, B, A, B, A, B  (regimes configurable, 20 episodes per segment,
3 seeds by default)

Paired seeding: episode i of segment s uses seed hash(s, i) for EVERY arm,
guaranteeing identical start states and goals across arms (G5 gate).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Generator, Literal

StreamName = Literal["s2"]

# S2: alternating regime_A / regime_B, 6 segments. Kept as the DEFAULT label
# sequence for back-compat / documentation; stream_s2()'s actual regime
# sequence is built from its `regimes` argument (see below), not this
# constant directly.
S2_REGIMES = ["R0", "R1", "R0", "R1", "R0", "R1"]


@dataclass(frozen=True)
class EpisodeSpec:
    """Specification for a single episode in the stream."""
    segment_idx: int
    episode_idx: int
    regime: str
    seed: int
    # segment_idx * episodes_per_segment + episode_idx -- a single monotone
    # index across the whole stream. make_f1/make_f2 sort on this so plots
    # render in stream order regardless of how records land in the JSONL.
    global_episode_idx: int


def paired_seed(segment_idx: int, episode_idx: int, arm: str = "") -> int:
    """
    Deterministic seed for (segment, episode).
    The arm is intentionally NOT included so all arms share the same seed.

    Args:
        segment_idx:  0-indexed segment number.
        episode_idx:  0-indexed episode number within the segment.
        arm:          (unused, kept for API clarity that arm is excluded)
    """
    key = f"seg={segment_idx}:ep={episode_idx}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big")


def stream_s2(
    episodes_per_segment: int = 20,
    seeds: int = 3,
    stream_seed_offset: int = 0,
    regimes: tuple[str, str] = ("R0", "R2"),
) -> Generator[list[EpisodeSpec], None, None]:
    """S2 = A,B,A,B,A,B over (regime_A, regime_B).

    Plan §6.1/§7.4 says R0/R1. Default changed to R0/R2 (damping 0.5): E0's
    calibration (E0_RECOVERY_PLAN §0.5) measured R2 at 40pp degradation vs R1's
    20pp, and every trained/evaluated chart in this project targets R2.
    Overridable via --segment-regimes; the resolved pair is recorded in the
    run's summary JSON. Documented deviation, in the style of plan §7.0a.

    Yields one list per seed, each containing all episodes in order.
    Each call with a different stream_seed_offset gives a different seed set.

    Args:
        episodes_per_segment: Episodes per A or B segment.
        seeds:                Number of independent seeds to generate.
        stream_seed_offset:   Integer offset added to seed computations to
                              produce independent replications.
        regimes:              (regime_A, regime_B) — the alternating pair.

    Yields:
        List[EpisodeSpec] — one full stream (all segments) per seed.
    """
    seq = [regimes[i % 2] for i in range(6)]
    for seed_run in range(seeds):
        episodes: list[EpisodeSpec] = []
        for seg_idx, regime in enumerate(seq):
            for ep_idx in range(episodes_per_segment):
                seed = paired_seed(seg_idx + stream_seed_offset * 1000,
                                   ep_idx + seed_run * 10_000)
                episodes.append(EpisodeSpec(
                    segment_idx=seg_idx,
                    episode_idx=ep_idx,
                    regime=regime,
                    seed=seed,
                    global_episode_idx=seg_idx * episodes_per_segment + ep_idx,
                ))
        yield episodes


def get_stream(
    name: StreamName,
    episodes_per_segment: int = 20,
    seeds: int = 3,
    regimes: tuple[str, str] = ("R0", "R2"),
) -> list[list[EpisodeSpec]]:
    """
    Return all seeds' episode lists for the named stream.

    Args:
        name:                 'S2'
        episodes_per_segment: Episodes per segment.
        seeds:                Number of independent seeds.
        regimes:              (regime_A, regime_B) — the alternating pair
                              (S2 only).

    Returns:
        List of length *seeds*, each a list of EpisodeSpec.
    """
    if name == "s2":
        return list(stream_s2(episodes_per_segment, seeds, regimes=regimes))
    raise ValueError(f"Unknown stream: {name!r}. Supported: ['s2']")
