"""Prioritized Experience Replay au niveau trajectoire pour V2-ZY+Polyak.

Voir spec : docs/superpowers/specs/2026-05-26-mw-ia-per-trajectory-design.md

Contient :
- BetaScheduler : annealing linéaire β_start → β_end pour IS correction
- PrioritizedSequenceReplayBuffer + PrioritizedBatchSeq : ajouté Phase 4
"""
from __future__ import annotations


class BetaScheduler:
    """Annealing linéaire β_start → β_end sur total_episodes.

    β contrôle l'intensité de la correction Importance Sampling. β=0 = aucune
    correction, β=1 = correction complète. Standard Schaul 2015 : β annealé
    progressivement pour stabiliser l'apprentissage initial puis converger
    vers une estimation non-biaisée.
    """

    def __init__(self, beta_start: float, beta_end: float, total_episodes: int) -> None:
        if not (0.0 <= beta_start <= 1.0):
            raise ValueError(
                f"beta_start doit etre dans [0, 1], recu {beta_start}"
            )
        if not (0.0 <= beta_end <= 1.0):
            raise ValueError(
                f"beta_end doit etre dans [0, 1], recu {beta_end}"
            )
        if total_episodes <= 0:
            raise ValueError(
                f"total_episodes doit etre > 0, recu {total_episodes}"
            )
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_episodes = total_episodes

    def beta(self, episode: int) -> float:
        """Retourne beta a l'episode donne. Clamp aux extremites."""
        if episode <= 0:
            return self.beta_start
        if episode >= self.total_episodes:
            return self.beta_end
        progress = episode / self.total_episodes
        return self.beta_start + (self.beta_end - self.beta_start) * progress
