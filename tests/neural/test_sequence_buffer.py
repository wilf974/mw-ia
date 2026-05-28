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


# === V2-B1a : concat_batchseq tests ===
from mw_ia.neural.sequence_buffer import BatchSeq, concat_batchseq


def _make_batchseq(seq_len: int, batch_size: int, obs_dim: int, fill_value: float = 0.0) -> BatchSeq:
    """Helper pour construire BatchSeq synthétique."""
    return BatchSeq(
        states=np.full((seq_len, batch_size, obs_dim), fill_value, dtype=np.float32),
        actions=np.zeros((seq_len, batch_size), dtype=np.int64),
        rewards=np.zeros((seq_len, batch_size), dtype=np.float32),
        next_states=np.full((seq_len, batch_size, obs_dim), fill_value, dtype=np.float32),
        dones=np.zeros((seq_len, batch_size), dtype=np.float32),
        mask=np.ones((seq_len, batch_size), dtype=np.float32),
    )


def test_concat_batchseq_axis_correct() -> None:
    """concat_batchseq concat le long de la dimension batch (axis=1), pas seq (axis=0)."""
    a = _make_batchseq(seq_len=8, batch_size=4, obs_dim=10, fill_value=1.0)
    b = _make_batchseq(seq_len=8, batch_size=3, obs_dim=10, fill_value=2.0)
    c = concat_batchseq(a, b)
    # Shape attendue : (8, 4+3, 10) - concat sur axis=1
    assert c.states.shape == (8, 7, 10)
    assert c.actions.shape == (8, 7)
    assert c.rewards.shape == (8, 7)
    assert c.next_states.shape == (8, 7, 10)
    assert c.dones.shape == (8, 7)
    assert c.mask.shape == (8, 7)


def test_concat_batchseq_preserves_content() -> None:
    """Les valeurs de a et b sont preservees dans l'ordre."""
    a = _make_batchseq(seq_len=4, batch_size=2, obs_dim=5, fill_value=1.0)
    b = _make_batchseq(seq_len=4, batch_size=3, obs_dim=5, fill_value=2.0)
    c = concat_batchseq(a, b)
    # Les 2 premieres colonnes batch viennent de a (fill_value=1.0)
    assert np.all(c.states[:, :2] == 1.0)
    # Les 3 dernieres colonnes batch viennent de b (fill_value=2.0)
    assert np.all(c.states[:, 2:] == 2.0)


def test_concat_batchseq_dtype_preserved() -> None:
    """Dtypes float32 / int64 preserves (pas de cast implicite)."""
    a = _make_batchseq(seq_len=4, batch_size=2, obs_dim=5)
    b = _make_batchseq(seq_len=4, batch_size=2, obs_dim=5)
    c = concat_batchseq(a, b)
    assert c.states.dtype == np.float32
    assert c.actions.dtype == np.int64
    assert c.rewards.dtype == np.float32
    assert c.next_states.dtype == np.float32
    assert c.dones.dtype == np.float32
    assert c.mask.dtype == np.float32
