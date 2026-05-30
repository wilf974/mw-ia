"""Tests des flags V2-BX sur ConvRecurrentDQNConfig."""
from __future__ import annotations

import pytest

from mw_ia.config import ConvRecurrentDQNConfig


def test_defaults_are_no_op():
    cfg = ConvRecurrentDQNConfig()
    assert cfg.bx_repr_oracle == "none"
    assert cfg.bx_novelty_beta == 0.0


def test_valid_oracle_modes_accepted():
    for mode in ("none", "scalar", "field"):
        cfg = ConvRecurrentDQNConfig(bx_repr_oracle=mode)
        assert cfg.bx_repr_oracle == mode


def test_invalid_oracle_mode_rejected():
    with pytest.raises(ValueError, match="bx_repr_oracle"):
        ConvRecurrentDQNConfig(bx_repr_oracle="bogus")


def test_negative_novelty_beta_rejected():
    with pytest.raises(ValueError, match="bx_novelty_beta"):
        ConvRecurrentDQNConfig(bx_novelty_beta=-0.1)


def test_positive_novelty_beta_accepted():
    cfg = ConvRecurrentDQNConfig(bx_novelty_beta=0.1)
    assert cfg.bx_novelty_beta == 0.1
