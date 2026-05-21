"""Tests de InvariantViolationError."""
from __future__ import annotations

import pytest

from mw_ia.guardrails.contracts import Severity, VariantSpec, VerdictReport, Violation
from mw_ia.guardrails.exceptions import InvariantViolationError


def _spec():
    return VariantSpec(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )


def test_error_exposes_report():
    v = Violation("I1", "gamma hors (0,1)", Severity.HARD)
    report = VerdictReport(passed=False, violations=(v,), spec=_spec(), duration_ms=1.0)
    err = InvariantViolationError(report)
    assert err.report is report
    assert "I1" in str(err)


def test_error_is_raisable():
    v = Violation("I1", "x", Severity.HARD)
    report = VerdictReport(passed=False, violations=(v,), spec=_spec(), duration_ms=1.0)
    with pytest.raises(InvariantViolationError) as exc_info:
        raise InvariantViolationError(report)
    assert exc_info.value.report.violations[0].invariant_id == "I1"
