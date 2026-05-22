"""Module guardrails MW_IA V2-A : invariants formels Aether + runtime Python.

API publique consommée par le futur sous-projet E (auto-modification).

Usage minimal :
    from mw_ia.guardrails import VariantSpec, verify_formal
    spec = VariantSpec(gamma=0.99, lr=1e-3, ...)
    report = verify_formal(spec)
    if report.passed:
        ...
"""
from __future__ import annotations

# Import des invariants pour peupler le registry au moment de l'import du package.
from mw_ia.guardrails import invariants as _invariants  # noqa: F401
from mw_ia.guardrails.contracts import Severity, VariantSpec, VerdictReport, Violation
from mw_ia.guardrails.exceptions import InvariantViolationError
from mw_ia.guardrails.verifier import verify_formal, verify_or_raise

__all__ = [
    "Severity",
    "VariantSpec",
    "VerdictReport",
    "Violation",
    "InvariantViolationError",
    "verify_formal",
    "verify_or_raise",
]
