"""Catalogue v1 des 8 invariants RL — implémentations runtime Python.

Chaque @invariant doit avoir son équivalent Aether dans aether/invariants/.
Cohérence vérifiée par tests/guardrails/test_aether_python_sync.py.
"""
from __future__ import annotations

from typing import Optional

from mw_ia.guardrails.contracts import Severity, VariantSpec, Violation
from mw_ia.guardrails.registry import invariant


@invariant("I1", applies_to=["gamma"])
def gamma_in_open_unit(spec: VariantSpec) -> Optional[Violation]:
    """γ doit être dans l'intervalle ouvert (0, 1) pour garantir la contraction."""
    if not (0.0 < spec.gamma < 1.0):
        return Violation(
            invariant_id="I1",
            message=f"gamma={spec.gamma} hors (0,1)",
            severity=Severity.HARD,
        )
    return None
