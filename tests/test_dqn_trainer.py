"""Tests du DQNTrainer (loss + step + AMP)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mw_ia.neural.network import QNetwork
from mw_ia.neural.replay_buffer import ReplayBuffer
from mw_ia.neural.trainer import DQNTrainer


def _fill_buffer(rb: ReplayBuffer, n: int = 64) -> None:
    for _ in range(n):
        rb.push(
            np.random.randn(4).astype(np.float32),
            int(np.random.randint(0, 2)),
            float(np.random.randn()),
            np.random.randn(4).astype(np.float32),
            False,
        )


def test_train_step_returns_loss_scalar() -> None:
    online = QNetwork(4, 2, hidden_layers=(32,))
    target = QNetwork(4, 2, hidden_layers=(32,))
    trainer = DQNTrainer(online, target, lr=1e-3, gamma=0.99, device="cpu", use_amp=False)
    rb = ReplayBuffer(capacity=128, obs_dim=4, seed=0)
    _fill_buffer(rb)
    loss = trainer.step(rb.sample(32))
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_target_sync_copies_weights() -> None:
    online = QNetwork(4, 2, hidden_layers=(8,))
    target = QNetwork(4, 2, hidden_layers=(8,))
    trainer = DQNTrainer(online, target, lr=1e-3, gamma=0.99, device="cpu", use_amp=False)
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    trainer.sync_target()
    for po, pt in zip(online.parameters(), target.parameters()):
        assert torch.equal(po, pt)


def test_loss_decreases_on_repeated_steps() -> None:
    online = QNetwork(4, 2, hidden_layers=(32,))
    target = QNetwork(4, 2, hidden_layers=(32,))
    trainer = DQNTrainer(online, target, lr=1e-2, gamma=0.99, device="cpu", use_amp=False)
    rb = ReplayBuffer(capacity=128, obs_dim=4, seed=0)
    _fill_buffer(rb, n=64)
    batch = rb.sample(32)
    first = trainer.step(batch)
    for _ in range(50):
        trainer.step(batch)
    last = trainer.step(batch)
    assert last < first, f"la loss doit decroitre : first={first:.4f} last={last:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA indisponible")
def test_amp_step_on_cuda() -> None:
    online = QNetwork(4, 2, hidden_layers=(32,)).to("cuda")
    target = QNetwork(4, 2, hidden_layers=(32,)).to("cuda")
    trainer = DQNTrainer(online, target, lr=1e-3, gamma=0.99, device="cuda", use_amp=True)
    rb = ReplayBuffer(capacity=128, obs_dim=4, seed=0)
    _fill_buffer(rb)
    loss = trainer.step(rb.sample(32))
    assert loss >= 0.0
