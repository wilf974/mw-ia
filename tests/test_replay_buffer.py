"""Tests ReplayBuffer."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.neural.replay_buffer import ReplayBuffer


def test_push_and_len() -> None:
    rb = ReplayBuffer(capacity=10, obs_dim=4, seed=0)
    for _ in range(5):
        rb.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
    assert len(rb) == 5


def test_capacity_is_circular() -> None:
    rb = ReplayBuffer(capacity=3, obs_dim=2, seed=0)
    for i in range(5):
        rb.push(np.array([i, i], dtype=np.float32), i % 2, float(i),
                np.zeros(2), False)
    assert len(rb) == 3
    rewards_in_buffer = set(rb._rewards[:len(rb)].tolist())
    assert 2.0 in rewards_in_buffer
    assert 4.0 in rewards_in_buffer


def test_sample_shapes() -> None:
    rb = ReplayBuffer(capacity=100, obs_dim=4, seed=0)
    for _ in range(50):
        rb.push(np.random.randn(4).astype(np.float32), 1, 0.5,
                np.random.randn(4).astype(np.float32), False)
    batch = rb.sample(16)
    assert batch.states.shape == (16, 4)
    assert batch.actions.shape == (16,)
    assert batch.rewards.shape == (16,)
    assert batch.next_states.shape == (16, 4)
    assert batch.dones.shape == (16,)


def test_sample_requires_enough() -> None:
    rb = ReplayBuffer(capacity=10, obs_dim=4, seed=0)
    rb.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
    with pytest.raises(ValueError):
        rb.sample(8)
