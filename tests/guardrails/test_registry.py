"""Tests du registry et du décorateur @invariant."""
from __future__ import annotations

import pytest

from mw_ia.guardrails.contracts import Severity, VariantSpec, Violation
from mw_ia.guardrails.registry import Invariant, _REGISTRY, invariant, applicable_invariants


@pytest.fixture(autouse=True)
def _clear_registry():
    """Isole chaque test du registry global."""
    backup = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(backup)


def test_decorator_registers_invariant():
    @invariant("ITEST", applies_to=["gamma"])
    def check_test(spec: VariantSpec):
        return None

    assert "ITEST" in _REGISTRY
    inv = _REGISTRY["ITEST"]
    assert isinstance(inv, Invariant)
    assert inv.id == "ITEST"
    assert inv.applies_to == ("gamma",)


def test_decorator_check_callable():
    @invariant("IX", applies_to=[])
    def check_x(spec: VariantSpec):
        return Violation("IX", "always", Severity.HARD)

    inv = _REGISTRY["IX"]
    spec = VariantSpec(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )
    result = inv.check(spec)
    assert result is not None
    assert result.invariant_id == "IX"


def _spec(**overrides):
    base = dict(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )
    base.update(overrides)
    return VariantSpec(**base)


def test_applicable_all_when_fields_present():
    @invariant("IA", applies_to=["gamma"])
    def a(spec): return None

    @invariant("IB", applies_to=["reward_min", "reward_max"])
    def b(spec): return None

    spec = _spec(reward_min=-1.0, reward_max=1.0)
    ids = [inv.id for inv in applicable_invariants(spec)]
    assert set(ids) == {"IA", "IB"}


def test_applicable_filters_when_optional_field_none():
    @invariant("IA", applies_to=["gamma"])
    def a(spec): return None

    @invariant("IB", applies_to=["reward_min", "reward_max"])
    def b(spec): return None

    spec = _spec()  # reward_min/max = None
    ids = [inv.id for inv in applicable_invariants(spec)]
    assert ids == ["IA"]


def test_applicable_order_is_stable():
    @invariant("I3", applies_to=[])
    def c(spec): return None

    @invariant("I1", applies_to=[])
    def a(spec): return None

    @invariant("I2", applies_to=[])
    def b(spec): return None

    spec = _spec()
    ids = [inv.id for inv in applicable_invariants(spec)]
    assert ids == ["I3", "I1", "I2"]  # ordre d'insertion (dict ordered Py3.7+)
