"""BestCheckpointTracker — sauvegarde automatique du meilleur modèle observé (V2-V).

Maintient best_eval_winrate en mémoire et sauvegarde le modèle au pic.
Découple "savoir quand sauver" de "où sauver" — path=None pour tracking
en mémoire pur (utile pour tests / dry-run).

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-training-protocol-stabilization-design.md §2
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol


class _SaveableAgent(Protocol):
    """Contrat minimal qu'un agent doit respecter pour être sauvegardé."""

    def save(self, path: str | Path) -> None: ...


class BestCheckpointTracker:
    """Sauvegarde le modèle au pic d'eval_winrate observé.

    Idempotent : un eval avec winrate égal ou inférieur au meilleur observé
    ne déclenche aucune sauvegarde.
    """

    def __init__(self, path: str | Path | None) -> None:
        self._path: Path | None = Path(path) if path is not None else None
        self._best_winrate: float = -math.inf
        self._best_episode: int | None = None
        self._best_difficulty: float | None = None

    @property
    def best_winrate(self) -> float:
        return self._best_winrate

    @property
    def best_episode(self) -> int | None:
        return self._best_episode

    @property
    def best_difficulty(self) -> float | None:
        return self._best_difficulty

    @property
    def path(self) -> Path | None:
        return self._path

    def update(
        self, eval_metrics: dict[str, float], agent: _SaveableAgent, episode: int,
    ) -> bool:
        """Update best si nouveau pic. Retourne True si save déclenché.

        Args:
            eval_metrics: dict avec keys 'winrate' et 'difficulty'.
            agent: objet exposant save(path).
            episode: épisode courant (pour traçabilité du pic).

        Returns:
            True si nouveau best sauvegardé, False sinon (idempotent sur égalité).
        """
        wr = float(eval_metrics["winrate"])
        if wr > self._best_winrate:
            self._best_winrate = wr
            self._best_episode = int(episode)
            self._best_difficulty = float(eval_metrics["difficulty"])
            if self._path is not None:
                agent.save(self._path)
            return True
        return False
