"""Suivi de métriques pour la GUI (winrate glissant + niveau IA)."""
from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Deque

from mw_ia.config import TrainingConfig


class Level(Enum):
    BEGINNER = "Débutant"
    INTERMEDIATE = "Intermédiaire"
    ADVANCED = "Avancé"
    EXPERT = "Expert"


class MetricsTracker:
    """Conserve une fenêtre glissante des succès + meilleurs scores."""

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self._success_window: Deque[bool] = deque(maxlen=cfg.winrate_window)
        self._reward_history: list[float] = []
        self._length_history: list[int] = []
        self.best_reward: float = float("-inf")
        self.last_loss: float | None = None
        self.last_epsilon: float | None = None

    @property
    def total_episodes(self) -> int:
        return len(self._reward_history)

    def record_episode(self, reward: float, length: int, *, success: bool) -> None:
        self._success_window.append(success)
        self._reward_history.append(reward)
        self._length_history.append(length)
        if reward > self.best_reward:
            self.best_reward = reward

    def record_loss(self, loss: float) -> None:
        self.last_loss = loss

    def record_epsilon(self, eps: float) -> None:
        self.last_epsilon = eps

    def winrate(self) -> float:
        if not self._success_window:
            return 0.0
        return sum(self._success_window) / len(self._success_window)

    def level(self, *, winrate: float | None = None) -> Level:
        wr = self.winrate() if winrate is None else winrate
        low, mid, hi = self.cfg.level_thresholds
        if wr < low:
            return Level.BEGINNER
        if wr < mid:
            return Level.INTERMEDIATE
        if wr < hi:
            return Level.ADVANCED
        return Level.EXPERT
