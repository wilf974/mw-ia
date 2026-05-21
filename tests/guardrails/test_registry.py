"""Tests du registry et du décorateur @invariant."""
from __future__ import annotations

import pytest

from mw_ia.guardrails.contracts import Severity, VariantSpec, Violation
from mw_ia.guardrails.registry import Invariant, _REGISTRY, invariant


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
