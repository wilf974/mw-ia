"""Dataclasses du contrat public guardrails."""
from __future__ import annotations

from enum import Enum


class Severity(Enum):
    """Gravité d'une violation d'invariant."""

    HARD = "hard"   # mathématiquement faux — bloque le déploiement
    SOFT = "soft"   # atypique mais valide — signale sans bloquer (réservé v2)
