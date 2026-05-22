"""Tests de DifficultyBucketTracker."""
from __future__ import annotations

from mw_ia.config import TrainingConfig
from mw_ia.training.metrics import DifficultyBucketTracker


def _tracker() -> DifficultyBucketTracker:
    return DifficultyBucketTracker(TrainingConfig())


def test_bucket_routing_low():
    t = _tracker()
    t.record_episode(success=True, reward=1.0, length=10, difficulty=0.15)
    wr = t.winrate_per_bucket()
    assert wr[0] == 1.0
    for i in range(1, 5):
        assert wr[i] is None


def test_bucket_routing_high():
    t = _tracker()
    t.record_episode(success=True, reward=1.0, length=10, difficulty=0.85)
    wr = t.winrate_per_bucket()
    assert wr[4] == 1.0
    for i in range(4):
        assert wr[i] is None


def test_bucket_routing_max_difficulty_inclusive():
    """difficulty=1.0 doit aller dans le bucket 4, pas 5 (out of bounds)."""
    t = _tracker()
    t.record_episode(success=True, reward=1.0, length=10, difficulty=1.0)
    wr = t.winrate_per_bucket()
    assert wr[4] == 1.0


def test_bucket_empty_returns_none():
    t = _tracker()
    wr = t.winrate_per_bucket()
    assert wr == [None, None, None, None, None]


def test_bucket_winrate_per_bucket_returns_5_values():
    t = _tracker()
    for d, s in [(0.1, True), (0.3, False), (0.5, True), (0.7, True), (0.9, False)]:
        t.record_episode(success=s, reward=1.0 if s else 0.0, length=10, difficulty=d)
    wr = t.winrate_per_bucket()
    assert len(wr) == 5
    assert all(w is not None for w in wr)
