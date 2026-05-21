"""Tests unitaires des 8 invariants du catalogue v1."""
from __future__ import annotations

import pytest

from mw_ia.guardrails.contracts import Severity, VariantSpec
from mw_ia.guardrails.registry import _REGISTRY
# Import du module pour exécuter ses @invariant et peupler le registry :
import mw_ia.guardrails.invariants  # noqa: F401


def _spec(**overrides):
    base = dict(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )
    base.update(overrides)
    return VariantSpec(**base)


# --- I1 -----------------------------------------------------------------------
def test_i1_pass_with_gamma_in_open_unit():
    inv = _REGISTRY["I1"]
    assert inv.check(_spec(gamma=0.99)) is None


def test_i1_violates_when_gamma_one():
    inv = _REGISTRY["I1"]
    v = inv.check(_spec(gamma=1.0))
    assert v is not None
    assert v.invariant_id == "I1"
    assert v.severity == Severity.HARD


# --- I2 -----------------------------------------------------------------------
def test_i2_pass_with_gamma_099():
    inv = _REGISTRY["I2"]
    assert inv.check(_spec(gamma=0.99)) is None


def test_i2_violates_when_gamma_too_close_to_one():
    """Si γ=1, le test runtime échantillonné DOIT détecter une non-contraction.
    Note : I1 attrape γ=1 avant I2 en pratique. Ce test isole I2 sur γ=1.0
    qui par construction est non-contractant (peut tester égalité, pas <)."""
    inv = _REGISTRY["I2"]
    v = inv.check(_spec(gamma=1.0))
    assert v is not None
    assert v.invariant_id == "I2"
    assert v.severity == Severity.HARD
    assert v.counter_example is not None


# --- I3 -----------------------------------------------------------------------
def test_i3_huber_nonneg_passes():
    inv = _REGISTRY["I3"]
    assert inv.check(_spec()) is None


# --- I4 -----------------------------------------------------------------------
def test_i4_winrate_bounds_passes():
    inv = _REGISTRY["I4"]
    assert inv.check(_spec()) is None


# --- I5 -----------------------------------------------------------------------
def test_i5_pass_with_decreasing_schedule():
    inv = _REGISTRY["I5"]
    assert inv.check(_spec(epsilon_start=1.0, epsilon_end=0.05)) is None


def test_i5_violates_when_end_greater_than_start():
    inv = _REGISTRY["I5"]
    v = inv.check(_spec(epsilon_start=0.05, epsilon_end=1.0))
    assert v is not None
    assert v.invariant_id == "I5"
