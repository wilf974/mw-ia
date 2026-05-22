"""Tests de AdaptiveDifficultyScheduler."""
from __future__ import annotations

import pytest

from mw_ia.config import SchedulerConfig
from mw_ia.training.scheduler import AdaptiveDifficultyScheduler


def test_scheduler_starts_at_initial():
    s = AdaptiveDifficultyScheduler(SchedulerConfig(initial_difficulty=0.2))
    assert s.current == 0.2


def test_scheduler_winrate_above_up_threshold_increases():
    s = AdaptiveDifficultyScheduler(
        SchedulerConfig(initial_difficulty=0.5, up_threshold=0.80, step=0.05)
    )
    new = s.update(winrate=0.9)
    assert new == pytest.approx(0.55)
    assert s.current == pytest.approx(0.55)


def test_scheduler_winrate_below_down_threshold_decreases():
    s = AdaptiveDifficultyScheduler(
        SchedulerConfig(initial_difficulty=0.5, down_threshold=0.30, step=0.05)
    )
    new = s.update(winrate=0.2)
    assert new == pytest.approx(0.45)


def test_scheduler_winrate_in_neutral_zone_unchanged():
    s = AdaptiveDifficultyScheduler(
        SchedulerConfig(initial_difficulty=0.5,
                        up_threshold=0.80, down_threshold=0.30, step=0.05)
    )
    new = s.update(winrate=0.5)
    assert new == 0.5
    assert s.current == 0.5


def test_scheduler_clamps_to_max():
    s = AdaptiveDifficultyScheduler(
        SchedulerConfig(initial_difficulty=0.95, up_threshold=0.80,
                        max_difficulty=1.0, step=0.10)
    )
    new = s.update(winrate=0.9)
    assert new == 1.0  # 0.95 + 0.10 = 1.05 → clamp à 1.0


def test_scheduler_clamps_to_min():
    s = AdaptiveDifficultyScheduler(
        SchedulerConfig(initial_difficulty=0.05, down_threshold=0.30,
                        min_difficulty=0.0, step=0.10)
    )
    new = s.update(winrate=0.1)
    assert new == 0.0  # 0.05 - 0.10 = -0.05 → clamp à 0.0


def test_scheduler_stable_at_neutral_winrate():
    s = AdaptiveDifficultyScheduler(SchedulerConfig(initial_difficulty=0.5))
    for _ in range(100):
        s.update(winrate=0.5)
    assert s.current == 0.5
