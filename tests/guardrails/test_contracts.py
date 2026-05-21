"""Tests des dataclasses contracts du module guardrails."""
from __future__ import annotations

import pytest

from mw_ia.guardrails.contracts import Severity, Violation, VariantSpec


def test_severity_has_hard_and_soft():
    assert Severity.HARD.value == "hard"
    assert Severity.SOFT.value == "soft"


def test_violation_basic_fields():
    v = Violation(invariant_id="I1", message="gamma=1.5 hors (0,1)", severity=Severity.HARD)
    assert v.invariant_id == "I1"
    assert v.message == "gamma=1.5 hors (0,1)"
    assert v.severity == Severity.HARD
    assert v.counter_example is None


def test_violation_is_frozen():
    v = Violation(invariant_id="I1", message="x", severity=Severity.HARD)
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        v.message = "y"  # type: ignore[misc]


def test_violation_with_counter_example():
    v = Violation(
        invariant_id="I2",
        message="contraction violée",
        severity=Severity.HARD,
        counter_example={"Q_diff": 1.0, "TQ_diff": 1.5, "gamma": 0.9},
    )
    assert v.counter_example["TQ_diff"] == 1.5


def _valid_spec_kwargs() -> dict:
    return dict(
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50_000,
        batch_size=128,
        replay_capacity=100_000,
        target_sync_steps=1_000,
    )


def test_variant_spec_valid_construction():
    spec = VariantSpec(**_valid_spec_kwargs())
    assert spec.gamma == 0.99
    assert spec.reward_min is None
    assert spec.reward_max is None


def test_variant_spec_with_reward_bounds():
    spec = VariantSpec(**_valid_spec_kwargs(), reward_min=-1.0, reward_max=1.0)
    assert spec.reward_min == -1.0
    assert spec.reward_max == 1.0


def test_variant_spec_zero_replay_capacity_raises():
    kw = _valid_spec_kwargs() | {"replay_capacity": 0}
    with pytest.raises(ValueError, match="replay_capacity"):
        VariantSpec(**kw)


def test_variant_spec_negative_lr_raises():
    kw = _valid_spec_kwargs() | {"lr": -1e-3}
    with pytest.raises(ValueError, match="lr"):
        VariantSpec(**kw)


def test_variant_spec_epsilon_out_of_unit_raises():
    kw = _valid_spec_kwargs() | {"epsilon_start": 1.5}
    with pytest.raises(ValueError, match="epsilon"):
        VariantSpec(**kw)


def test_variant_spec_is_frozen():
    spec = VariantSpec(**_valid_spec_kwargs())
    with pytest.raises(Exception):
        spec.gamma = 0.5  # type: ignore[misc]
