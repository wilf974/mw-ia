"""Tests de SequenceReplayBuffer."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.neural.sequence_buffer import SequenceReplayBuffer


def _trajectory(length: int, obs_dim: int = 4) -> list[tuple]:
    """Crée une trajectoire factice de longueur `length`."""
    return [
        (
            np.zeros(obs_dim, dtype=np.float32),    # state
            i % 4,                                   # action
            float(i),                                # reward
            np.zeros(obs_dim, dtype=np.float32),    # next_state
            i == length - 1,                         # done sur dernier step
        )
        for i in range(length)
    ]


def test_buffer_empty_at_init():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    assert len(buf) == 0


def test_push_trajectory_valid():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    buf.push_trajectory(_trajectory(length=18))
    assert len(buf) == 1


def test_push_trajectory_empty_raises():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    with pytest.raises(ValueError, match="longueur"):
        buf.push_trajectory([])


def test_push_trajectory_too_long_raises():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    with pytest.raises(ValueError, match="longueur"):
        buf.push_trajectory(_trajectory(length=201))


def test_push_capacity_circular():
    """Buffer circulaire : push capacity+5 trajectoires, len reste à capacity."""
    buf = SequenceReplayBuffer(capacity=5, obs_dim=4, max_steps=200, seed=0)
    for _ in range(10):
        buf.push_trajectory(_trajectory(length=10))
    assert len(buf) == 5
