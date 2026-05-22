"""Tests V2-V de BestCheckpointTracker (sauvegarde au pic eval_winrate)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from mw_ia.agents.conv_dqn import ConvDQNAgent
from mw_ia.config import ConvDQNConfig
from mw_ia.training.checkpoint_tracker import BestCheckpointTracker


def _build_agent() -> ConvDQNAgent:
    cfg = ConvDQNConfig(min_replay_to_learn=4, batch_size=2, train_every=1)
    return ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )


def test_init_no_best() -> None:
    """best_winrate = -inf, best_episode = None à l'init."""
    tracker = BestCheckpointTracker(path=None)
    assert math.isinf(tracker.best_winrate) and tracker.best_winrate < 0
    assert tracker.best_episode is None
    assert tracker.best_difficulty is None


def test_first_update_always_saves(tmp_path: Path) -> None:
    """Premier eval triggers save (improvement vs -inf)."""
    path = tmp_path / "best.pt"
    tracker = BestCheckpointTracker(path=path)
    agent = _build_agent()
    metrics = {"winrate": 0.0, "difficulty": 0.05}
    improved = tracker.update(metrics, agent, episode=100)
    assert improved is True
    assert path.exists()
    assert tracker.best_winrate == 0.0
    assert tracker.best_episode == 100
    assert tracker.best_difficulty == 0.05


def test_lower_winrate_does_not_save(tmp_path: Path) -> None:
    """eval_winrate < best → skip, fichier inchangé."""
    path = tmp_path / "best.pt"
    tracker = BestCheckpointTracker(path=path)
    agent = _build_agent()
    tracker.update({"winrate": 0.5, "difficulty": 0.10}, agent, episode=100)
    mtime_before = path.stat().st_mtime_ns
    improved = tracker.update({"winrate": 0.3, "difficulty": 0.15}, agent, episode=200)
    assert improved is False
    assert tracker.best_winrate == 0.5
    assert tracker.best_episode == 100
    assert path.stat().st_mtime_ns == mtime_before


def test_higher_winrate_saves_and_updates_best(tmp_path: Path) -> None:
    """eval_winrate > best → save + update state."""
    path = tmp_path / "best.pt"
    tracker = BestCheckpointTracker(path=path)
    agent = _build_agent()
    tracker.update({"winrate": 0.5, "difficulty": 0.10}, agent, episode=100)
    improved = tracker.update({"winrate": 0.7, "difficulty": 0.20}, agent, episode=200)
    assert improved is True
    assert tracker.best_winrate == 0.7
    assert tracker.best_episode == 200
    assert tracker.best_difficulty == 0.20


def test_path_none_no_save_attempted() -> None:
    """path=None → tracking en mémoire, pas d'IO."""
    tracker = BestCheckpointTracker(path=None)
    agent = _build_agent()
    improved = tracker.update({"winrate": 0.5, "difficulty": 0.10}, agent, episode=100)
    assert improved is True
    assert tracker.best_winrate == 0.5
    assert tracker.best_episode == 100


def test_equal_winrate_does_not_save(tmp_path: Path) -> None:
    """Idempotence : eval_winrate == best → skip."""
    path = tmp_path / "best.pt"
    tracker = BestCheckpointTracker(path=path)
    agent = _build_agent()
    tracker.update({"winrate": 0.5, "difficulty": 0.10}, agent, episode=100)
    mtime_before = path.stat().st_mtime_ns
    improved = tracker.update({"winrate": 0.5, "difficulty": 0.15}, agent, episode=200)
    assert improved is False
    assert tracker.best_episode == 100
    assert path.stat().st_mtime_ns == mtime_before
