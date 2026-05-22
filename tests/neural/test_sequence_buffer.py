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


def test_sample_before_min_raises():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    with pytest.raises(ValueError, match="buffer trop petit"):
        buf.sample(batch_size=4, seq_len=32)


def test_sample_returns_correct_shapes():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    for _ in range(5):
        buf.push_trajectory(_trajectory(length=50))
    batch = buf.sample(batch_size=4, seq_len=32)
    assert batch.states.shape == (32, 4, 4)
    assert batch.actions.shape == (32, 4)
    assert batch.rewards.shape == (32, 4)
    assert batch.next_states.shape == (32, 4, 4)
    assert batch.dones.shape == (32, 4)
    assert batch.mask.shape == (32, 4)


def test_sample_padding_short_trajectory():
    """Trajectoire de 18 steps + seq_len=32 → mask = [1]*18 + [0]*14 dans la fenêtre."""
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=42)
    # Pousser une seule trajectoire courte
    buf.push_trajectory(_trajectory(length=18))
    # Sampler avec batch=1, seq_len=32. Comme la traj a 18 steps, l'offset valide est 0,
    # et les 14 steps suivants sont padding.
    batch = buf.sample(batch_size=1, seq_len=32)
    # mask[0:18, 0] doit valoir 1.0, mask[18:32, 0] doit valoir 0.0
    assert batch.mask[:18, 0].sum() == 18.0
    assert batch.mask[18:, 0].sum() == 0.0


def test_sample_random_offset_within_long_trajectory():
    """Trajectoire longue (50 steps) + seq_len=10 → mask doit être tout à 1 (pas de padding)."""
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    buf.push_trajectory(_trajectory(length=50))
    batch = buf.sample(batch_size=1, seq_len=10)
    assert batch.mask.sum() == 10.0   # tous les steps sont des vrais steps
