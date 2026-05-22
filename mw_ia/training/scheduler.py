"""Scheduler adaptatif de difficulté piloté par winrate."""
from __future__ import annotations

import numpy as np

from mw_ia.config import SchedulerConfig


class AdaptiveDifficultyScheduler:
    """Monte/descend la difficulté selon le winrate observé.

    Règle :
        winrate >= up_threshold   → current += step (cap max_difficulty)
        winrate <= down_threshold → current -= step (floor min_difficulty)
        sinon                     → inchangé

    L'instance est mutable (self.current change), mais la config est frozen.
    """

    def __init__(self, cfg: SchedulerConfig) -> None:
        self.cfg = cfg
        self.current: float = cfg.initial_difficulty

    def update(self, *, winrate: float) -> float:
        """Met à jour current selon winrate et retourne la nouvelle valeur."""
        if winrate >= self.cfg.up_threshold:
            self.current = float(np.clip(
                self.current + self.cfg.step, self.cfg.min_difficulty, self.cfg.max_difficulty
            ))
        elif winrate <= self.cfg.down_threshold:
            self.current = float(np.clip(
                self.current - self.cfg.step, self.cfg.min_difficulty, self.cfg.max_difficulty
            ))
        return self.current
