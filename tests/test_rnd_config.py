"""Tests des flags V2-C0 RND sur ConvRecurrentDQNConfig."""
from __future__ import annotations

import pytest

from mw_ia.config import ConvRecurrentDQNConfig


def test_defaults_are_no_op():
    cfg = ConvRecurrentDQNConfig()
    assert cfg.rnd_enabled is False
    assert cfg.rnd_beta == 0.5
    assert cfg.rnd_lr == 1e-4
    assert cfg.rnd_embed_dim == 128
    assert cfg.rnd_clip == 5.0
    assert cfg.rnd_warmup_steps == 1000
    assert cfg.rnd_ratio_warn == 10.0


def test_enabled_accepted():
    cfg = ConvRecurrentDQNConfig(rnd_enabled=True, rnd_beta=0.2)
    assert cfg.rnd_enabled is True
    assert cfg.rnd_beta == 0.2


def test_negative_beta_rejected():
    with pytest.raises(ValueError, match="rnd_beta"):
        ConvRecurrentDQNConfig(rnd_beta=-0.1)


def test_non_positive_clip_rejected():
    with pytest.raises(ValueError, match="rnd_clip"):
        ConvRecurrentDQNConfig(rnd_clip=0.0)


def test_negative_warmup_rejected():
    with pytest.raises(ValueError, match="rnd_warmup_steps"):
        ConvRecurrentDQNConfig(rnd_warmup_steps=-1)


def test_non_positive_embed_rejected():
    with pytest.raises(ValueError, match="rnd_embed_dim"):
        ConvRecurrentDQNConfig(rnd_embed_dim=0)
