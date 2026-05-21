"""Tests des dataclasses contracts du module guardrails."""
from __future__ import annotations

import pytest

from mw_ia.guardrails.contracts import Severity, Violation


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
