"""
tests/test_streams.py — Unit tests for atlas.streams (paired seeding).
"""

from atlas.streams import paired_seed, get_stream, S2_REGIMES


def test_paired_seed_arm_independent():
    """G5: seed must not depend on the arm argument."""
    for seg in range(6):
        for ep in range(5):
            assert paired_seed(seg, ep, arm="atlas") == paired_seed(seg, ep, arm="frozen")
            assert paired_seed(seg, ep, arm="adajepa") == paired_seed(seg, ep, arm="oracle_id")


def test_paired_seed_deterministic():
    """Same args always produce the same seed."""
    assert paired_seed(0, 0) == paired_seed(0, 0)
    assert paired_seed(3, 17) == paired_seed(3, 17)


def test_paired_seed_unique():
    """Different (seg, ep) pairs produce different seeds (collision-free for S2 scale)."""
    seeds = set()
    for seg in range(6):
        for ep in range(20):
            s = paired_seed(seg, ep)
            assert s not in seeds, f"Collision at seg={seg}, ep={ep}"
            seeds.add(s)


def test_s2_structure():
    """S2 has 6 segments of alternating R0/R1."""
    assert S2_REGIMES == ["R0", "R1", "R0", "R1", "R0", "R1"]


def test_get_stream_s2():
    streams = get_stream("s2", episodes_per_segment=5, seeds=2)
    assert len(streams) == 2
    for stream in streams:
        assert len(stream) == 6 * 5
        for ep in stream:
            assert ep.regime in ("R0", "R1")


def test_get_stream_unknown():
    import pytest
    with pytest.raises(ValueError, match="Unknown stream"):
        get_stream("s99")
