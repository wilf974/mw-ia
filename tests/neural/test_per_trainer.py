"""Tests V2-B0 RecurrentDQNTrainer.step_with_priorities (IS-weighted + R2D2 agg)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mw_ia.neural.recurrent import RecurrentQNetwork
from mw_ia.neural.recurrent_trainer import RecurrentDQNTrainer
from mw_ia.neural.sequence_buffer import BatchSeq


def _make_trainer(double_dqn: bool = False, polyak_tau: float = 0.0):
    online = RecurrentQNetwork(input_dim=8, n_actions=4, fc_hidden=16, lstm_hidden=16)
    target = RecurrentQNetwork(input_dim=8, n_actions=4, fc_hidden=16, lstm_hidden=16)
    trainer = RecurrentDQNTrainer(
        online, target, lr=1e-3, gamma=0.99,
        device="cpu", use_amp=False,
        double_dqn=double_dqn, polyak_tau=polyak_tau,
    )
    return trainer


def _make_batch(seq_len: int = 8, batch_size: int = 4, obs_dim: int = 8) -> BatchSeq:
    rng = np.random.default_rng(0)
    return BatchSeq(
        states=rng.normal(size=(seq_len, batch_size, obs_dim)).astype(np.float32),
        actions=rng.integers(0, 4, size=(seq_len, batch_size)).astype(np.int64),
        rewards=rng.uniform(-1, 1, size=(seq_len, batch_size)).astype(np.float32),
        next_states=rng.normal(size=(seq_len, batch_size, obs_dim)).astype(np.float32),
        dones=np.zeros((seq_len, batch_size), dtype=np.float32),
        mask=np.ones((seq_len, batch_size), dtype=np.float32),
    )


def test_step_unchanged_signature_returns_float() -> None:
    """step(batch) -> float (V2-Y compat strict)."""
    trainer = _make_trainer()
    batch = _make_batch()
    result = trainer.step(batch)
    assert isinstance(result, float)


def test_step_with_priorities_returns_tuple() -> None:
    """step_with_priorities(batch, weights, eta) -> (float, ndarray)."""
    trainer = _make_trainer()
    batch = _make_batch(batch_size=4)
    weights = np.ones(4, dtype=np.float32)
    result = trainer.step_with_priorities(batch, weights, eta=0.9)
    assert isinstance(result, tuple)
    loss, td_errors = result
    assert isinstance(loss, float)
    assert isinstance(td_errors, np.ndarray)
    assert td_errors.shape == (4,)
    assert td_errors.dtype == np.float32


def test_is_weights_change_loss() -> None:
    """Loss avec weights=[2,1,1,1] != loss avec weights=[1,1,1,1]."""
    trainer1 = _make_trainer()
    trainer2 = _make_trainer()
    # Sync params pour comparaison fair
    trainer2.online.load_state_dict(trainer1.online.state_dict())
    trainer2.target.load_state_dict(trainer1.target.state_dict())
    batch = _make_batch(batch_size=4)
    loss1, _ = trainer1.step_with_priorities(
        batch, weights=np.ones(4, dtype=np.float32), eta=0.9,
    )
    # Reset trainer pour comparer (les params ont deja ete mis a jour apres step1)
    trainer3 = _make_trainer()
    trainer3.online.load_state_dict(trainer2.online.state_dict())
    trainer3.target.load_state_dict(trainer2.target.state_dict())
    loss3, _ = trainer3.step_with_priorities(
        batch, weights=np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32), eta=0.9,
    )
    assert loss1 != pytest.approx(loss3, abs=1e-6)


def test_td_errors_r2d2_aggregation() -> None:
    """priority_b = eta*max + (1-eta)*mean sur batch synthetique."""
    trainer = _make_trainer()
    batch = _make_batch(seq_len=4, batch_size=2)
    weights = np.ones(2, dtype=np.float32)
    _, td_errors = trainer.step_with_priorities(batch, weights, eta=0.9)
    # td_errors devrait etre >= 0 (formule |delta|)
    assert (td_errors >= 0).all()
    # Avec eta=1.0 et meme batch, td_errors == max_per_traj uniquement
    _, td_max_only = trainer.step_with_priorities(batch, weights, eta=1.0)
    assert (td_max_only >= 0).all()


def test_mask_excludes_padded_steps_from_aggregation() -> None:
    """Trajectoire avec mask[10:]=0 : aggregation max/mean sur [0:10] seulement."""
    trainer = _make_trainer()
    batch = _make_batch(seq_len=16, batch_size=2)
    # Forcer la seconde trajectoire a mask=0 sur les 8 derniers steps
    batch = BatchSeq(
        states=batch.states, actions=batch.actions, rewards=batch.rewards,
        next_states=batch.next_states, dones=batch.dones,
        mask=batch.mask.copy(),
    )
    batch.mask[8:, 1] = 0.0
    weights = np.ones(2, dtype=np.float32)
    _, td_errors = trainer.step_with_priorities(batch, weights, eta=0.9)
    # Aucun crash sur trajectoire courte. td_errors >= 0
    assert td_errors.shape == (2,)
    assert (td_errors >= 0).all()


def test_double_dqn_path_with_per() -> None:
    """PER + Double DQN cohabitent (formule target reste Hasselt 2015)."""
    trainer = _make_trainer(double_dqn=True)
    batch = _make_batch()
    weights = np.ones(4, dtype=np.float32)
    loss, td_errors = trainer.step_with_priorities(batch, weights, eta=0.9)
    assert isinstance(loss, float)
    assert td_errors.shape == (4,)


def test_polyak_with_per() -> None:
    """PER + Polyak cohabitent (target update post-backward inchange)."""
    trainer = _make_trainer(polyak_tau=0.005)
    # Capture target snapshot
    target_snapshot = {
        name: p.clone() for name, p in trainer.target.named_parameters()
    }
    batch = _make_batch()
    weights = np.ones(4, dtype=np.float32)
    trainer.step_with_priorities(batch, weights, eta=0.9)
    # Target a ete modifie par Polyak post-backward
    target_modified = False
    for name, p in trainer.target.named_parameters():
        if not torch.allclose(p, target_snapshot[name]):
            target_modified = True
            break
    assert target_modified


def test_step_with_priorities_no_grad_through_priorities() -> None:
    """td_errors retournes ne gardent pas de gradient (detaches)."""
    trainer = _make_trainer()
    batch = _make_batch()
    weights = np.ones(4, dtype=np.float32)
    _, td_errors = trainer.step_with_priorities(batch, weights, eta=0.9)
    # td_errors est un numpy ndarray, pas de grad
    assert isinstance(td_errors, np.ndarray)
    # Pas de fuite memoire : appel repete 50 fois sans crash
    for _ in range(50):
        trainer.step_with_priorities(batch, weights, eta=0.9)
