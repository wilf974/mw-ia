"""Tests des dataclasses contracts du module guardrails."""
from __future__ import annotations

from mw_ia.guardrails.contracts import Severity


def test_severity_has_hard_and_soft():
    assert Severity.HARD.value == "hard"
    assert Severity.SOFT.value == "soft"
