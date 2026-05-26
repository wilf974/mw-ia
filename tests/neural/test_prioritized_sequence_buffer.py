"""Tests V2-B0 PrioritizedSequenceReplayBuffer (sum tree + IS correction)."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.neural.prioritized_sequence_buffer import (
    PrioritizedBatchSeq,
    PrioritizedSequenceReplayBuffer,
)


def _make_trajectory(length: int, obs_dim: int, seed: int = 0) -> list[tuple]:
    """Génère une trajectoire synthétique pour les tests."""
    rng = np.random.default_rng(seed)
    traj = []
    for t in range(length):
        s = rng.normal(size=(obs_dim,)).astype(np.float32)
        a = int(rng.integers(0, 4))
        r = float(rng.uniform(-1, 1))
        sp = rng.normal(size=(obs_dim,)).astype(np.float32)
        d = (t == length - 1)
        traj.append((s, a, r, sp, d))
    return traj


def test_init_validates_capacity_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        PrioritizedSequenceReplayBuffer(capacity=0, obs_dim=10)


def test_init_validates_alpha_in_range() -> None:
    with pytest.raises(ValueError, match="alpha"):
        PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=10, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=10, alpha=1.5)


def test_init_validates_epsilon_positive() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=10, epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=10, epsilon=-1e-6)


def test_push_assigns_max_priority_to_new_trajectory() -> None:
    """Nouvelle trajectoire reçoit la priorité max courante (greedy init)."""
    buf = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=4, seed=0)
    traj = _make_trajectory(5, obs_dim=4)
    buf.push_trajectory(traj)
    assert len(buf) == 1
    # _max_priority initial = 1.0 → première trajectoire a priorité 1.0
    assert buf._sum_tree.total() == pytest.approx(1.0)


def test_sample_smaller_than_buffer_raises() -> None:
    buf = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=4, seed=0)
    buf.push_trajectory(_make_trajectory(5, obs_dim=4))
    with pytest.raises(ValueError, match="buffer"):
        buf.sample(batch_size=10, seq_len=5, beta=0.5)


def test_sample_seq_len_out_of_range_raises() -> None:
    buf = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=4, max_steps=10, seed=0)
    for _ in range(5):
        buf.push_trajectory(_make_trajectory(5, obs_dim=4))
    with pytest.raises(ValueError, match="seq_len"):
        buf.sample(batch_size=4, seq_len=0, beta=0.5)
    with pytest.raises(ValueError, match="seq_len"):
        buf.sample(batch_size=4, seq_len=11, beta=0.5)


def test_sample_returns_prioritized_batch_seq() -> None:
    """sample() retourne PrioritizedBatchSeq avec batch / weights / tree_indices."""
    buf = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=4, max_steps=20, seed=0)
    for i in range(10):
        buf.push_trajectory(_make_trajectory(10, obs_dim=4, seed=i))
    prio_batch = buf.sample(batch_size=4, seq_len=8, beta=0.5)
    assert isinstance(prio_batch, PrioritizedBatchSeq)
    # batch shape (seq, batch, obs_dim) selon BatchSeq V2-Y
    assert prio_batch.batch.states.shape == (8, 4, 4)
    assert prio_batch.weights.shape == (4,)
    assert prio_batch.weights.dtype == np.float32
    assert prio_batch.tree_indices.shape == (4,)


def test_sample_is_weights_normalized_by_max() -> None:
    """IS weights normalisés : max(w) == 1.0 ± epsilon."""
    buf = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=4, max_steps=20, seed=0)
    for i in range(20):
        buf.push_trajectory(_make_trajectory(10, obs_dim=4, seed=i))
    # Donner des priorités différentes
    indices = np.arange(20)
    td_errors = np.linspace(0.1, 5.0, 20).astype(np.float32)
    buf.update_priorities(indices, td_errors)
    prio_batch = buf.sample(batch_size=8, seq_len=8, beta=0.5)
    assert prio_batch.weights.max() == pytest.approx(1.0, abs=1e-6)
    assert (prio_batch.weights > 0).all()


def test_update_priorities_modifies_sampling_distribution() -> None:
    """Après update_priorities avec td_errors différents, sampling devient biaisé."""
    buf = PrioritizedSequenceReplayBuffer(
        capacity=10, obs_dim=4, max_steps=20, alpha=1.0, seed=0,
    )
    for i in range(10):
        buf.push_trajectory(_make_trajectory(10, obs_dim=4, seed=i))
    # Donner priorité 100x plus haute à trajectoire 0
    indices = np.arange(10)
    td_errors = np.ones(10, dtype=np.float32)
    td_errors[0] = 100.0
    buf.update_priorities(indices, td_errors)
    # 1000 samples, trajectoire 0 doit être >>50%
    counts = np.zeros(10, dtype=int)
    for _ in range(1000):
        prio_batch = buf.sample(batch_size=1, seq_len=8, beta=0.0)
        counts[prio_batch.tree_indices[0]] += 1
    assert counts[0] > 500  # majoritairement trajectoire 0


def test_update_priorities_updates_max_priority() -> None:
    """_max_priority croît si nouvelle priorité dépasse l'ancienne."""
    buf = PrioritizedSequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=20, seed=0)
    buf.push_trajectory(_make_trajectory(5, obs_dim=4))
    initial_max = buf._max_priority
    # td_error élevé → priorité élevée
    buf.update_priorities(np.array([0]), np.array([10.0], dtype=np.float32))
    assert buf._max_priority > initial_max


def test_first_trajectory_is_sampleable() -> None:
    """Greedy init : première trajectoire (priority=_max_priority=1.0) est sampleable."""
    buf = PrioritizedSequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=20, seed=0)
    for _ in range(4):
        buf.push_trajectory(_make_trajectory(5, obs_dim=4))
    # Aucun update_priorities encore → toutes priorités == 1.0
    prio_batch = buf.sample(batch_size=4, seq_len=5, beta=0.5)
    assert prio_batch.batch.states.shape == (5, 4, 4)


def test_reproducibility_with_seed() -> None:
    """Même seed → même séquence de samples."""
    obs_dim = 4
    buf1 = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=obs_dim, max_steps=20, seed=42)
    buf2 = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=obs_dim, max_steps=20, seed=42)
    for i in range(10):
        traj = _make_trajectory(10, obs_dim=obs_dim, seed=i)
        buf1.push_trajectory(traj)
        buf2.push_trajectory(traj)
    pb1 = buf1.sample(batch_size=4, seq_len=8, beta=0.5)
    pb2 = buf2.sample(batch_size=4, seq_len=8, beta=0.5)
    assert np.array_equal(pb1.tree_indices, pb2.tree_indices)


def test_capacity_5000_v2zy_default() -> None:
    """capacity=5000 (V2-ZY 10×10 default) fonctionne sans crash."""
    buf = PrioritizedSequenceReplayBuffer(
        capacity=5000, obs_dim=300, max_steps=200, seed=0,
    )
    for i in range(200):
        buf.push_trajectory(_make_trajectory(30, obs_dim=300, seed=i))
    prio_batch = buf.sample(batch_size=128, seq_len=32, beta=0.5)
    assert prio_batch.batch.states.shape == (32, 128, 300)
    assert prio_batch.weights.shape == (128,)
