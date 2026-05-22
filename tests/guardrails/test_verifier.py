"""Tests de verify_formal."""
from __future__ import annotations

import pytest

from mw_ia.guardrails.contracts import VariantSpec
import mw_ia.guardrails.invariants  # noqa: F401  (peuple le registry)
from mw_ia.guardrails.verifier import verify_formal


def _spec(**overrides):
    base = dict(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )
    base.update(overrides)
    return VariantSpec(**base)


def test_verify_formal_passes_on_valid_spec():
    report = verify_formal(_spec())
    assert report.passed is True
    assert report.violations == ()
    assert report.duration_ms > 0


def test_verify_formal_collects_multiple_violations_no_shortcircuit():
    # gamma=1.0 viole I1 ET I2. Pas de court-circuit attendu.
    report = verify_formal(_spec(gamma=1.0))
    assert report.passed is False
    ids = {v.invariant_id for v in report.violations}
    assert "I1" in ids
    assert "I2" in ids


def test_verify_formal_is_deterministic():
    r1 = verify_formal(_spec(gamma=1.0))
    r2 = verify_formal(_spec(gamma=1.0))
    assert {v.invariant_id for v in r1.violations} == {v.invariant_id for v in r2.violations}
