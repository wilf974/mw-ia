"""Tests de RNDModule (V2-C0 RND) — proprietes centrales."""
from __future__ import annotations

import numpy as np
import torch

from mw_ia.neural.rnd import RNDModule


def _obs(seed: int) -> np.ndarray:
    return np.random.RandomState(seed).rand(3, 4, 4).astype(np.float32)


def test_predictor_error_decreases_on_repeated_obs():
    # Propriete RND centrale : le predictor apprend l'obs -> erreur baisse.
    mod = RNDModule(in_channels=3, rows=4, cols=4, warmup_steps=0, seed=0, device="cpu")
    x = _obs(0)
    losses = [mod.update(x) for _ in range(50)]
    assert losses[-1] < losses[0]


def test_bonus_non_negative():
    mod = RNDModule(in_channels=3, rows=4, cols=4, warmup_steps=0, seed=0, device="cpu")
    assert mod.compute_bonus(_obs(1)) >= 0.0


def test_warmup_returns_zero():
    mod = RNDModule(in_channels=3, rows=4, cols=4, warmup_steps=5, seed=0, device="cpu")
    x = _obs(2)
    for _ in range(5):
        assert mod.compute_bonus(x) == 0.0  # steps 1..5 <= warmup
    assert mod.compute_bonus(x) >= 0.0      # step 6 > warmup


def test_bonus_clipped():
    mod = RNDModule(
        in_channels=3, rows=4, cols=4, warmup_steps=0, clip=0.001, seed=0, device="cpu"
    )
    assert mod.compute_bonus(_obs(3)) <= 0.001


def test_target_is_frozen():
    mod = RNDModule(in_channels=3, rows=4, cols=4, warmup_steps=0, seed=0, device="cpu")
    before = [p.detach().clone() for p in mod.target.parameters()]
    x = _obs(4)
    for _ in range(10):
        mod.update(x)
    for b, p in zip(before, mod.target.parameters()):
        assert torch.equal(b, p)


def test_predictor_is_trained():
    mod = RNDModule(in_channels=3, rows=4, cols=4, warmup_steps=0, seed=0, device="cpu")
    before = [p.detach().clone() for p in mod.predictor.parameters()]
    x = _obs(5)
    for _ in range(10):
        mod.update(x)
    changed = any(
        not torch.equal(b, p) for b, p in zip(before, mod.predictor.parameters())
    )
    assert changed
