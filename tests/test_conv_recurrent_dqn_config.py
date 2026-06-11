"""Tests V2-ZY de ConvRecurrentDQNConfig (combo CNN + LSTM + Double DQN + V2-V)."""
from __future__ import annotations

import pytest

from mw_ia.config import ConvRecurrentDQNConfig
from mw_ia.guardrails import VariantSpec, verify_formal


def test_defaults() -> None:
    """V2-ZY defaults : combo Conv + LSTM + Double DQN + V2-V activé."""
    cfg = ConvRecurrentDQNConfig()
    assert cfg.conv_channels == (32, 64)
    assert cfg.kernel_size == 3
    assert cfg.padding == 1
    assert cfg.lstm_hidden == 128
    assert cfg.sequence_length == 32
    assert cfg.double_dqn is True
    assert cfg.eval_enabled is True
    assert cfg.eval_every_episodes == 100
    assert cfg.eval_seeds == tuple(range(10_000, 10_010))
    assert cfg.eval_target_difficulty == 0.30
    assert cfg.best_checkpoint_path is None


def test_validation_conv_channels_positive() -> None:
    """conv_channels doivent être > 0 et non-vide."""
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(conv_channels=())
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(conv_channels=(0, 64))
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(conv_channels=(-1, 64))


def test_validation_lstm_hidden_positive() -> None:
    """lstm_hidden doit être > 0."""
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(lstm_hidden=0)
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(lstm_hidden=-1)


def test_validation_eval_target_difficulty_bounds() -> None:
    """eval_target_difficulty dans [0, 1]."""
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(eval_target_difficulty=-0.1)
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(eval_target_difficulty=1.1)
    cfg = ConvRecurrentDQNConfig(eval_target_difficulty=0.50)
    assert cfg.eval_target_difficulty == 0.50


def test_v2zy_polyak_tau_default_and_validation() -> None:
    """V2-U : default polyak_tau=0.0, validation dans [0, 1]."""
    cfg = ConvRecurrentDQNConfig()
    assert cfg.polyak_tau == 0.0
    cfg2 = ConvRecurrentDQNConfig(polyak_tau=0.005)
    assert cfg2.polyak_tau == 0.005
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(polyak_tau=-0.001)
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(polyak_tau=1.001)


def test_aether_compat() -> None:
    """VariantSpec dérivé du ConvRecurrentDQNConfig passe les invariants Aether I1-I8."""
    cfg = ConvRecurrentDQNConfig()
    spec = VariantSpec(
        gamma=cfg.gamma, lr=cfg.lr,
        epsilon_start=cfg.epsilon_start, epsilon_end=cfg.epsilon_end,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
        batch_size=cfg.batch_size,
        replay_capacity=cfg.replay_capacity,
        target_sync_steps=cfg.target_sync_steps,
    )
    report = verify_formal(spec)
    assert report.passed, f"violations: {report.violations}"
