"""Interface commune à tous les agents (tabulaires + neural)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Agent(ABC):
    """Contrat minimal d'un agent RL."""

    @abstractmethod
    def act(self, state: Any, *, greedy: bool = False) -> int:
        """Sélectionne une action depuis un état."""

    @abstractmethod
    def learn(self, transition: Any) -> dict[str, float]:
        """Met à jour la politique à partir d'une transition.

        Retourne un dict de métriques (ex: {"loss": 0.12}). Peut être vide.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persiste l'état de l'agent sur disque."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Recharge l'état depuis disque."""
