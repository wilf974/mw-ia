"""Exceptions du module guardrails."""
from __future__ import annotations

from mw_ia.guardrails.contracts import VerdictReport


class InvariantViolationError(Exception):
    """Levée par verify_or_raise quand au moins un invariant viole."""

    def __init__(self, report: VerdictReport) -> None:
        self.report = report
        ids = ", ".join(v.invariant_id for v in report.violations)
        super().__init__(
            f"{len(report.violations)} invariant(s) violé(s) : [{ids}]"
        )
