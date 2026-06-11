"""Tests de DRQNConfig."""
from __future__ import annotations

import pytest

from mw_ia.config import DRQNConfig


def test_drqn_config_defaults_valid():
    cfg = DRQNConfig()
    assert cfg.fc_hidden == 256
    assert cfg.lstm_hidden == 128
    assert cfg.sequence_length == 32
    assert cfg.replay_capacity == 5000
    assert cfg.min_episodes_to_learn == 100
    assert cfg.train_steps_per_episode == 4
    assert cfg.epsilon_decay_steps == 200_000


def test_drqn_config_is_frozen():
    cfg = DRQNConfig()
    with pytest.raises(Exception):
        cfg.sequence_length = 64  # type: ignore[misc]


def test_drqn_config_sequence_length_too_large_raises():
    with pytest.raises(ValueError, match="sequence_length"):
        DRQNConfig(sequence_length=300)  # > max_steps_per_episode=200


def test_drqn_config_sequence_length_zero_raises():
    with pytest.raises(ValueError, match="sequence_length"):
        DRQNConfig(sequence_length=0)


def test_drqn_config_replay_capacity_zero_raises():
    with pytest.raises(ValueError, match="replay_capacity"):
        DRQNConfig(replay_capacity=0)


def test_drqn_config_epsilon_inverted_raises():
    with pytest.raises(ValueError, match="epsilon"):
        DRQNConfig(epsilon_start=0.1, epsilon_end=0.9)


def test_drqn_polyak_tau_default_and_validation() -> None:
    """V2-U : default polyak_tau=0.0, validation dans [0, 1]."""
    cfg = DRQNConfig()
    assert cfg.polyak_tau == 0.0
    cfg2 = DRQNConfig(polyak_tau=0.005)
    assert cfg2.polyak_tau == 0.005
    with pytest.raises(ValueError):
        DRQNConfig(polyak_tau=-0.001)
    with pytest.raises(ValueError):
        DRQNConfig(polyak_tau=1.001)
