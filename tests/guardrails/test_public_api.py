"""Smoke test de l'API publique du package guardrails."""
from __future__ import annotations


def test_public_api_imports():
    from mw_ia.guardrails import (
        InvariantViolationError,
        Severity,
        VariantSpec,
        VerdictReport,
        Violation,
        verify_formal,
        verify_or_raise,
    )
    # Smoke : tous importables, on construit un spec valide
    spec = VariantSpec(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )
    report = verify_formal(spec)
    assert report.passed is True
    assert isinstance(report, VerdictReport)
