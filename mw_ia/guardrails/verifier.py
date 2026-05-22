"""API publique : verify_formal et verify_or_raise."""
from __future__ import annotations

import time

from mw_ia.guardrails.contracts import VariantSpec, VerdictReport
from mw_ia.guardrails.exceptions import InvariantViolationError
from mw_ia.guardrails.registry import applicable_invariants


def verify_formal(spec: VariantSpec) -> VerdictReport:
    """Vérifie spec contre tous les invariants applicables.

    Ne lève jamais d'exception. Collecte TOUTES les violations (pas de
    court-circuit) pour permettre à E de réparer en parallèle plusieurs
    paramètres.

    Stateless, déterministe (les invariants stochastiques utilisent un
    RNG seedé).
    """
    t0 = time.perf_counter()
    violations = []
    for inv in applicable_invariants(spec):
        v = inv.check(spec)
        if v is not None:
            violations.append(v)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    return VerdictReport(
        passed=(len(violations) == 0),
        violations=tuple(violations),
        spec=spec,
        duration_ms=duration_ms,
    )


def verify_or_raise(spec: VariantSpec) -> VerdictReport:
    """Wrapper de verify_formal : lève InvariantViolationError si non passé.

    Réservé aux contextes "tout ou rien" (CI, pre-commit). Pour
    l'inspection/réparation côté E, utiliser verify_formal directement.
    """
    report = verify_formal(spec)
    if not report.passed:
        raise InvariantViolationError(report)
    return report
