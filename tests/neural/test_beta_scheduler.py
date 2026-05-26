"""Tests V2-B0 BetaScheduler — annealing linéaire IS exponent."""
from __future__ import annotations

import pytest

from mw_ia.neural.prioritized_sequence_buffer import BetaScheduler


def test_beta_at_zero_returns_beta_start() -> None:
    scheduler = BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=5000)
    assert scheduler.beta(0) == pytest.approx(0.4)


def test_beta_at_total_returns_beta_end() -> None:
    scheduler = BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=5000)
    assert scheduler.beta(5000) == pytest.approx(1.0)


def test_beta_overshoot_clamped_to_beta_end() -> None:
    """Episode > total → clamp à beta_end (cas buffer plein mais épisodes continuent)."""
    scheduler = BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=5000)
    assert scheduler.beta(10_000) == pytest.approx(1.0)
    assert scheduler.beta(50_000) == pytest.approx(1.0)


def test_beta_midpoint_linear() -> None:
    """beta(total / 2) == (beta_start + beta_end) / 2 (linéaire)."""
    scheduler = BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=5000)
    assert scheduler.beta(2500) == pytest.approx(0.7)


def test_validation_beta_start_out_of_range() -> None:
    with pytest.raises(ValueError, match="beta_start"):
        BetaScheduler(beta_start=-0.1, beta_end=1.0, total_episodes=5000)
    with pytest.raises(ValueError, match="beta_start"):
        BetaScheduler(beta_start=1.5, beta_end=1.0, total_episodes=5000)


def test_validation_beta_end_out_of_range() -> None:
    with pytest.raises(ValueError, match="beta_end"):
        BetaScheduler(beta_start=0.4, beta_end=-0.1, total_episodes=5000)
    with pytest.raises(ValueError, match="beta_end"):
        BetaScheduler(beta_start=0.4, beta_end=1.5, total_episodes=5000)


def test_validation_total_episodes_positive() -> None:
    with pytest.raises(ValueError, match="total_episodes"):
        BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=0)
    with pytest.raises(ValueError, match="total_episodes"):
        BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=-1)
