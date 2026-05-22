"""Tests V2-Z de ConvDQNConfig (frozen dataclass + validation)."""
from __future__ import annotations

import pytest

from mw_ia.config import ConvDQNConfig
from mw_ia.guardrails import VariantSpec, verify_formal


def test_defaults() -> None:
    """Defaults V2-Z : (32, 64) conv, kernel 3, pad 1, fc 256."""
    cfg = ConvDQNConfig()
    assert cfg.conv_channels == (32, 64)
    assert cfg.kernel_size == 3
    assert cfg.padding == 1
    assert cfg.fc_hidden == 256
    assert cfg.gamma == 0.99
    assert cfg.batch_size == 128


def test_post_init_validation_positive() -> None:
    """Channels/kernel/fc_hidden doivent être > 0."""
    with pytest.raises(ValueError):
        ConvDQNConfig(conv_channels=(0, 64))
    with pytest.raises(ValueError):
        ConvDQNConfig(kernel_size=0)
    with pytest.raises(ValueError):
        ConvDQNConfig(fc_hidden=-1)
    with pytest.raises(ValueError):
        ConvDQNConfig(padding=-1)


def test_conv_channels_arbitrary_length() -> None:
    """(16,) ou (32, 64, 128) acceptés."""
    cfg1 = ConvDQNConfig(conv_channels=(16,))
    assert cfg1.conv_channels == (16,)
    cfg3 = ConvDQNConfig(conv_channels=(32, 64, 128))
    assert cfg3.conv_channels == (32, 64, 128)
    # Tuple vide rejeté
    with pytest.raises(ValueError):
        ConvDQNConfig(conv_channels=())


def test_aether_compat() -> None:
    """VariantSpec dérivé du ConvDQNConfig passe les invariants Aether I1-I8."""
    cfg = ConvDQNConfig()
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


def test_double_dqn_default_true() -> None:
    """V2-W : default double_dqn=True (V2-W est l'amélioration recommandée)."""
    cfg = ConvDQNConfig()
    assert cfg.double_dqn is True


def test_double_dqn_can_be_disabled() -> None:
    """double_dqn=False pour reproduire V2-Z baseline."""
    cfg = ConvDQNConfig(double_dqn=False)
    assert cfg.double_dqn is False


def test_eval_defaults() -> None:
    """V2-V defaults : eval activé, 100 ép, seeds 10000-10009, max 200 steps."""
    cfg = ConvDQNConfig()
    assert cfg.eval_enabled is True
    assert cfg.eval_every_episodes == 100
    assert cfg.eval_seeds == tuple(range(10_000, 10_010))
    assert cfg.eval_max_steps == 200
    assert cfg.best_checkpoint_path is None


def test_eval_can_be_disabled() -> None:
    """eval_enabled=False pour reproduire baseline pre-V2-V."""
    cfg = ConvDQNConfig(eval_enabled=False)
    assert cfg.eval_enabled is False


def test_eval_validation() -> None:
    """eval_every_episodes ≤ 0 et eval_seeds vide rejetés."""
    with pytest.raises(ValueError):
        ConvDQNConfig(eval_every_episodes=0)
    with pytest.raises(ValueError):
        ConvDQNConfig(eval_every_episodes=-1)
    with pytest.raises(ValueError):
        ConvDQNConfig(eval_seeds=())
    with pytest.raises(ValueError):
        ConvDQNConfig(eval_max_steps=0)
