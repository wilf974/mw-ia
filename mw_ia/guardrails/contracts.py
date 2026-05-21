"""Dataclasses du contrat public guardrails."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Severity(Enum):
    """Gravité d'une violation d'invariant."""

    HARD = "hard"   # mathématiquement faux — bloque le déploiement
    SOFT = "soft"   # atypique mais valide — signale sans bloquer (réservé v2)


@dataclass(frozen=True)
class Violation:
    """Violation d'un invariant pour un VariantSpec donné."""

    invariant_id: str
    message: str
    severity: Severity
    counter_example: Optional[dict] = None
