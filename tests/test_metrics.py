"""Tests MetricsTracker (winrate + niveau)."""
from __future__ import annotations

import pytest

from mw_ia.config import TrainingConfig
from mw_ia.training.metrics import Level, MetricsTracker


def test_winrate_rolling_window_default_10() -> None:
    cfg = TrainingConfig(winrate_window=10)
    m = MetricsTracker(cfg)
    for i in range(10):
        m.record_episode(reward=1.0, length=5, success=(i % 2 == 0))
    assert m.winrate() == pytest.approx(0.5)


def test_winrate_window_caps() -> None:
    cfg = TrainingConfig(winrate_window=3)
    m = MetricsTracker(cfg)
    m.record_episode(0.0, 1, success=False)
    m.record_episode(0.0, 1, success=False)
    m.record_episode(0.0, 1, success=True)
    m.record_episode(0.0, 1, success=True)
    m.record_episode(0.0, 1, success=True)
    assert m.winrate() == pytest.approx(1.0)


def test_level_thresholds() -> None:
    cfg = TrainingConfig(level_thresholds=(0.3, 0.6, 0.85), winrate_window=10)
    m = MetricsTracker(cfg)
    assert m.level(winrate=0.10) is Level.BEGINNER
    assert m.level(winrate=0.40) is Level.INTERMEDIATE
    assert m.level(winrate=0.70) is Level.ADVANCED
    assert m.level(winrate=0.90) is Level.EXPERT


def test_total_episodes_and_best() -> None:
    cfg = TrainingConfig()
    m = MetricsTracker(cfg)
    m.record_episode(0.5, 10, success=False)
    m.record_episode(0.9, 12, success=True)
    m.record_episode(0.7, 8, success=True)
    assert m.total_episodes == 3
    assert m.best_reward == pytest.approx(0.9)


def test_running_loss_updates() -> None:
    cfg = TrainingConfig()
    m = MetricsTracker(cfg)
    m.record_loss(0.5)
    m.record_loss(0.3)
    assert m.last_loss == pytest.approx(0.3)
