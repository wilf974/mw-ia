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
        self._loss_history: list[float] = []
        self.best_reward: float = float("-inf")
        self.last_loss: float | None = None
        self.last_epsilon: float | None = None

    @property
    def total_episodes(self) -> int:
        return len(self._reward_history)

    @property
    def episode_rewards(self) -> list[float]:
        return self._reward_history

    @property
    def losses(self) -> list[float]:
        return self._loss_history

    def record_episode(self, reward: float, length: int, *, success: bool) -> None:
        self._success_window.append(success)
        self._reward_history.append(reward)
        self._length_history.append(length)
        if reward > self.best_reward:
            self.best_reward = reward

    def record_loss(self, loss: float) -> None:
        self.last_loss = loss
        self._loss_history.append(loss)

    def record_epsilon(self, eps: float) -> None:
        self.last_epsilon = eps

    def winrate(self) -> float:
        if not self._success_window:
            return 0.0
        return sum(self._success_window) / len(self._success_window)

    def has_data(self) -> bool:
        """True si au moins un épisode a été enregistré."""
        return bool(self._success_window)

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


class DifficultyBucketTracker:
    """5 MetricsTracker, un par bucket de difficulté [0,0.2), [0.2,0.4), ...

    Routing : bucket = min(4, int(difficulty * 5)). Le min(4, ...) gère
    difficulty=1.0 qui donnerait sinon bucket=5 (out of bounds).
    """

    N_BUCKETS = 5

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self._trackers: list[MetricsTracker] = [
            MetricsTracker(cfg) for _ in range(self.N_BUCKETS)
        ]

    def record_episode(
        self, *, success: bool, reward: float, length: int, difficulty: float
    ) -> None:
        bucket = min(self.N_BUCKETS - 1, int(difficulty * self.N_BUCKETS))
        self._trackers[bucket].record_episode(reward, length, success=success)

    def winrate_per_bucket(self) -> list[float | None]:
        return [t.winrate() if t.has_data() else None for t in self._trackers]
