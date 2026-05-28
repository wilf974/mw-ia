"""Tests V2-B1a SnapshotTrajectoryStore.

Invariant architectural central :
    Une fois capturee, une trajectoire snapshot N'EST JAMAIS modifiee,
    re-evaluee, re-encodee, ou re-rolled-out. Test #11 (test_immutability_after_capture)
    est le test qui formalise cet invariant.
"""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.neural.sequence_buffer import BatchSeq, SequenceReplayBuffer
from mw_ia.training.snapshot_store import SnapshotTrajectoryStore


def _make_buffer_with_trajectories(
    capacity: int = 100, obs_dim: int = 4, max_steps: int = 10, seed: int = 0,
) -> SequenceReplayBuffer:
    """Helper : buffer rempli avec mix success / fail trajectoires."""
    buf = SequenceReplayBuffer(capacity=capacity, obs_dim=obs_dim, max_steps=max_steps, seed=seed)
    rng = np.random.default_rng(seed)
    for i in range(20):
        traj_len = int(rng.integers(3, max_steps))
        # Even i = success (terminated last step + positive reward)
        # Odd i = fail (truncated, negative reward)
        is_success = (i % 2 == 0)
        traj = []
        for t in range(traj_len):
            s = rng.normal(size=(obs_dim,)).astype(np.float32)
            a = int(rng.integers(0, 4))
            if t == traj_len - 1 and is_success:
                r = 1.0  # goal reward
                d = True
            elif t == traj_len - 1 and not is_success:
                r = -1.0  # obstacle penalty
                d = True
            else:
                r = -0.01  # step penalty
                d = False
            sp = rng.normal(size=(obs_dim,)).astype(np.float32)
            traj.append((s, a, r, sp, d))
        buf.push_trajectory(traj)
    return buf


def test_init_validates_args() -> None:
    """obs_dim, max_steps, n_windows, snapshot_size doivent être > 0."""
    with pytest.raises(ValueError, match="obs_dim"):
        SnapshotTrajectoryStore(obs_dim=0)
    with pytest.raises(ValueError, match="max_steps"):
        SnapshotTrajectoryStore(obs_dim=4, max_steps=0)
    with pytest.raises(ValueError, match="n_windows"):
        SnapshotTrajectoryStore(obs_dim=4, n_windows=0)
    with pytest.raises(ValueError, match="snapshot_size"):
        SnapshotTrajectoryStore(obs_dim=4, snapshot_size=0)


def test_empty_store_has_zero_length() -> None:
    """Store fraîchement initialisé contient 0 trajectoires."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, n_windows=3, snapshot_size=50, seed=0)
    assert len(store) == 0
    assert store.n_captures == 0


def test_capture_from_empty_buffer_returns_zero() -> None:
    """Buffer source vide → capture retourne 0, store reste vide."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=10, seed=0)
    empty_buf = SequenceReplayBuffer(capacity=50, obs_dim=4, max_steps=10, seed=0)
    n = store.capture_from(empty_buf)
    assert n == 0
    assert len(store) == 0


def test_capture_filters_terminated_with_positive_reward() -> None:
    """Filtre succès strict : terminated_last_step AND total_reward > 0."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=20, seed=0)
    buf = _make_buffer_with_trajectories()
    # buf contient 20 trajectoires : 10 success (even), 10 fail (odd)
    n = store.capture_from(buf)
    # Doit capturer seulement les 10 success (filtre rejette les fail)
    assert n == 10
    assert len(store) == 10


def test_capture_takes_recent_first() -> None:
    """Iteration arrière depuis current_idx : trajectoires récentes prioritaires."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=3, seed=0)
    buf = _make_buffer_with_trajectories()
    # buf contient 20 trajectoires (10 success aux index pairs)
    # On capture seulement snapshot_size=3 → doit prendre les 3 success les + récentes
    n = store.capture_from(buf)
    assert n == 3
    # Les success les + récentes sont index 18, 16, 14 (puisque current_idx=20 mod 100 = 20)
    # On ne peut pas inspecter directement les indices mais on peut vérifier
    # que les trajectoires capturées ont total_reward > 0 (sample en mode greedy)
    batch = store.sample(batch_size=3, seq_len=10)
    # Au moins une étape avec done=1.0 dans chaque trajectoire (terminated)
    # et reward > 0 quelque part
    for b in range(3):
        valid_mask = batch.mask[:, b] > 0
        traj_rewards = batch.rewards[:, b][valid_mask]
        assert traj_rewards.sum() > 0.0


def test_sliding_window_evicts_oldest_after_n_captures() -> None:
    """Après n_windows+1 captures, la window 0 (la + ancienne) est écrasée."""
    store = SnapshotTrajectoryStore(
        obs_dim=4, max_steps=10, n_windows=2, snapshot_size=5, seed=0,
    )
    buf = _make_buffer_with_trajectories()
    # Capture 1 : remplit window 0
    n1 = store.capture_from(buf)
    assert len(store) == n1
    cap1_count = store.n_captures
    # Capture 2 : remplit window 1
    n2 = store.capture_from(buf)
    assert len(store) == n1 + n2
    cap2_count = store.n_captures
    # Capture 3 : devrait écraser window 0 (oldest)
    n3 = store.capture_from(buf)
    # len(store) doit rester ~ n_windows × snapshot_size, pas grandir au-delà
    assert len(store) <= 2 * 5  # n_windows × snapshot_size
    assert store.n_captures == cap2_count + 1
    assert store.n_captures == 3


def test_n_captures_tracks_total_unbounded() -> None:
    """n_captures continue d'incrémenter même après n_windows captures."""
    store = SnapshotTrajectoryStore(
        obs_dim=4, max_steps=10, n_windows=2, snapshot_size=5, seed=0,
    )
    buf = _make_buffer_with_trajectories()
    for _ in range(5):
        store.capture_from(buf)
    assert store.n_captures == 5  # 5 captures totales, même si seulement 2 windows actives


def test_sample_returns_batchseq_with_correct_shape() -> None:
    """sample() retourne BatchSeq shape (seq, batch, obs_dim)."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=20, seed=0)
    buf = _make_buffer_with_trajectories()
    store.capture_from(buf)
    batch = store.sample(batch_size=4, seq_len=8)
    assert isinstance(batch, BatchSeq)
    assert batch.states.shape == (8, 4, 4)
    assert batch.actions.shape == (8, 4)
    assert batch.mask.shape == (8, 4)


def test_sample_uniform_distribution() -> None:
    """10000 samples : fréquence par slot converge vers 1/total à ±2%."""
    store = SnapshotTrajectoryStore(
        obs_dim=2, max_steps=4, snapshot_size=5, seed=42,
    )
    # Capture manuellement 5 trajectoires distinguables via 1ère valeur de state
    buf = SequenceReplayBuffer(capacity=20, obs_dim=2, max_steps=4, seed=0)
    for marker in range(5):
        traj = [
            (np.array([marker, 0], dtype=np.float32), 0, 0.01, np.array([marker, 1], dtype=np.float32), False),
            (np.array([marker, 1], dtype=np.float32), 0, 0.01, np.array([marker, 2], dtype=np.float32), False),
            (np.array([marker, 2], dtype=np.float32), 0, 1.0, np.array([marker, 3], dtype=np.float32), True),
        ]
        buf.push_trajectory(traj)
    store.capture_from(buf)
    assert len(store) == 5

    # 10000 samples de batch_size=1, comptage par marker (1ère value de state)
    counts = np.zeros(5, dtype=int)
    n_samples = 10_000
    for _ in range(n_samples):
        batch = store.sample(batch_size=1, seq_len=3)
        marker = int(batch.states[0, 0, 0])
        if 0 <= marker < 5:
            counts[marker] += 1
    expected_freq = 1.0 / 5
    empirical_freq = counts / n_samples
    assert np.allclose(empirical_freq, expected_freq, atol=0.02)


def test_sample_raises_if_too_few_trajectories() -> None:
    """sample(batch_size > len(store)) lève ValueError."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=10, seed=0)
    buf = _make_buffer_with_trajectories()
    store.capture_from(buf)
    # Capture peut donner < 10 (filtre success), donc on prend un batch_size > tout possible
    with pytest.raises(ValueError, match="trop petit"):
        store.sample(batch_size=100, seq_len=5, )


def test_immutability_after_capture() -> None:
    """INVARIANT ARCHITECTURAL : modifier source post-capture ne change pas le store.

    Une trajectoire capturée est frozen — copies, pas références.
    """
    store = SnapshotTrajectoryStore(obs_dim=2, max_steps=4, snapshot_size=5, seed=0)
    buf = SequenceReplayBuffer(capacity=10, obs_dim=2, max_steps=4, seed=0)
    # Push 3 success trajectoires
    for marker in range(3):
        traj = [
            (np.array([marker, 0], dtype=np.float32), 0, 0.01, np.array([marker, 1], dtype=np.float32), False),
            (np.array([marker, 1], dtype=np.float32), 0, 1.0, np.array([marker, 2], dtype=np.float32), True),
        ]
        buf.push_trajectory(traj)
    store.capture_from(buf)
    assert len(store) == 3

    # Sample baseline pour comparaison
    rng_state = np.random.default_rng(123).bit_generator.state
    store._rng.bit_generator.state = rng_state
    batch_before = store.sample(batch_size=3, seq_len=2)
    states_before = batch_before.states.copy()

    # MUTATION DESTRUCTRICE du source buffer
    buf._states[:] = 999.0
    buf._actions[:] = 999
    buf._rewards[:] = -999.0
    buf._dones[:] = 999.0

    # Re-sample : le store DOIT retourner les mêmes valeurs qu'avant
    store._rng.bit_generator.state = rng_state
    batch_after = store.sample(batch_size=3, seq_len=2)
    assert np.array_equal(batch_after.states, states_before), (
        "INVARIANT VIOLATION : modifier source buffer a contaminé le snapshot store"
    )


def test_reproducibility_with_seed() -> None:
    """Même seed → même séquence de samples."""
    buf = _make_buffer_with_trajectories(seed=0)
    store1 = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=20, seed=42)
    store2 = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=20, seed=42)
    store1.capture_from(buf)
    store2.capture_from(buf)
    b1 = store1.sample(batch_size=4, seq_len=8)
    b2 = store2.sample(batch_size=4, seq_len=8)
    assert np.array_equal(b1.states, b2.states)
    assert np.array_equal(b1.actions, b2.actions)
