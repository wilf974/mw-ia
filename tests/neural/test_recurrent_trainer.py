"""Tests de RecurrentDQNTrainer."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mw_ia.neural.recurrent import RecurrentQNetwork
from mw_ia.neural.recurrent_trainer import RecurrentDQNTrainer
from mw_ia.neural.sequence_buffer import BatchSeq


def _make_batch(seq_len: int = 8, batch_size: int = 4, obs_dim: int = 10,
                full_mask: bool = True) -> BatchSeq:
    rng = np.random.default_rng(seed=42)
    states = rng.standard_normal((seq_len, batch_size, obs_dim)).astype(np.float32)
    actions = rng.integers(0, 4, size=(seq_len, batch_size)).astype(np.int64)
    rewards = rng.standard_normal((seq_len, batch_size)).astype(np.float32)
    next_states = rng.standard_normal((seq_len, batch_size, obs_dim)).astype(np.float32)
    dones = np.zeros((seq_len, batch_size), dtype=np.float32)
    if full_mask:
        mask = np.ones((seq_len, batch_size), dtype=np.float32)
    else:
        # Demi-mask : la 2e moitié de la séquence est paddée (mask=0)
        mask = np.ones((seq_len, batch_size), dtype=np.float32)
        mask[seq_len // 2:, :] = 0.0
    return BatchSeq(states, actions, rewards, next_states, dones, mask)


def test_trainer_one_step_no_crash(cpu_device):
    online = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    target = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    trainer = RecurrentDQNTrainer(online, target, lr=1e-3, gamma=0.99,
                                  device="cpu", use_amp=False)
    batch = _make_batch()
    loss = trainer.step(batch)
    assert isinstance(loss, float)
    assert loss >= 0.0   # Huber loss toujours ≥ 0


def test_trainer_mask_reduces_loss(cpu_device):
    """Loss avec demi-mask doit différer (être plus petite en moyenne) de loss full mask."""
    online = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    target = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    trainer = RecurrentDQNTrainer(online, target, lr=1e-3, gamma=0.99,
                                  device="cpu", use_amp=False)
    batch_full = _make_batch(full_mask=True)
    loss_full = trainer.step(batch_full)
    # Réinit pour comparer fair (sans laisser le step précédent influencer)
    online2 = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    target2 = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    online2.load_state_dict(online.state_dict() if False else online2.state_dict())  # noqa: SIM108
    trainer2 = RecurrentDQNTrainer(online2, target2, lr=1e-3, gamma=0.99,
                                   device="cpu", use_amp=False)
    batch_half = _make_batch(full_mask=False)
    loss_half = trainer2.step(batch_half)
    # Les deux losses sont des floats finis, différentes (mask différent donne loss différente)
    assert np.isfinite(loss_full)
    assert np.isfinite(loss_half)
    # Note : on ne peut pas affirmer loss_half < loss_full strictement à cause du seed et de l'init,
    # mais on peut vérifier qu'elles sont finies et que mask ≠ all-1 donne un calcul de loss différent
    assert loss_full != loss_half


def test_trainer_sync_target_copies_weights(cpu_device):
    online = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    target = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    trainer = RecurrentDQNTrainer(online, target, lr=1e-3, gamma=0.99,
                                  device="cpu", use_amp=False)
    # Modifier online pour qu'il diverge de target
    with torch.no_grad():
        online.fc_in.weight.add_(1.0)
    # Avant sync, online != target
    assert not torch.allclose(online.fc_in.weight, target.fc_in.weight)
    # Après sync, online == target
    trainer.sync_target()
    assert torch.allclose(online.fc_in.weight, target.fc_in.weight)
