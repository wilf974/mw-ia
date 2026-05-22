"""Hyperparamètres centralisés MW_IA — voir design doc §5."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class GridWorldConfig:
    """Paramètres de l'environnement GridWorld."""

    rows: int = 10
    cols: int = 10
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (9, 9)
    obstacles: tuple[tuple[int, int], ...] = (
        (2, 2), (2, 3), (2, 4),
        (5, 5), (5, 6),
        (7, 1), (7, 2),
    )
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    obstacle_penalty: float = -1.0
    max_steps: int = 200


@dataclass(frozen=True)
class QLearningConfig:
    """Q-Learning tabulaire."""

    alpha: float = 0.1          # learning rate
    gamma: float = 0.99         # discount
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500
    episodes: int = 1000


@dataclass(frozen=True)
class DQNConfig:
    """Deep Q-Network."""

    hidden_layers: tuple[int, ...] = (128, 128)
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    replay_capacity: int = 100_000
    min_replay_to_learn: int = 1_000
    target_sync_steps: int = 1_000
    train_every: int = 4
    use_amp: bool = True
    episodes: int = 500
    max_steps_per_episode: int = 200


@dataclass(frozen=True)
class TrainingConfig:
    """Réglages communs (logging, niveau IA, métriques)."""

    winrate_window: int = 100
    level_thresholds: tuple[float, float, float] = (0.30, 0.60, 0.85)
    log_every_episodes: int = 10
    seed: int = 42


@dataclass(frozen=True)
class Config:
    """Bundle global."""

    gridworld: GridWorldConfig = field(default_factory=GridWorldConfig)
    qlearning: QLearningConfig = field(default_factory=QLearningConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


DEFAULT = Config()


@dataclass(frozen=True)
class ProceduralEnvConfig:
    """Configuration de l'environnement procédural V2-X."""

    mode: Literal["obstacles", "maze"]
    max_rows: int = 10
    max_cols: int = 10
    min_density: float = 0.0            # mode obstacles uniquement (démarrage curriculum sans obstacles)
    max_density: float = 0.50
    min_size: int = 4                   # mode maze uniquement
    max_size: int = 20
    max_attempts_bfs: int = 100         # mode obstacles : tentatives avant RuntimeError

    def __post_init__(self) -> None:
        if self.mode not in ("obstacles", "maze"):
            raise ValueError(f"mode doit être 'obstacles' ou 'maze', reçu {self.mode!r}")
        if not (0.0 <= self.min_density <= self.max_density <= 1.0):
            raise ValueError(
                f"densités invalides : min_density={self.min_density}, "
                f"max_density={self.max_density}"
            )
        if not (2 <= self.min_size < self.max_size):
            raise ValueError(
                f"tailles invalides : min_size={self.min_size}, max_size={self.max_size}"
            )
        if self.max_attempts_bfs <= 0:
            raise ValueError(f"max_attempts_bfs doit être > 0, reçu {self.max_attempts_bfs}")


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration du scheduler adaptatif de difficulté."""

    initial_difficulty: float = 0.0
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    up_threshold: float = 0.80
    down_threshold: float = 0.30
    step: float = 0.05
    update_interval: int = 50           # épisodes

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_difficulty <= self.max_difficulty <= 1.0):
            raise ValueError(
                f"difficultés invalides : min={self.min_difficulty}, max={self.max_difficulty}"
            )
        if not (self.min_difficulty <= self.initial_difficulty <= self.max_difficulty):
            raise ValueError(
                f"initial_difficulty={self.initial_difficulty} hors "
                f"[{self.min_difficulty}, {self.max_difficulty}]"
            )
        if self.up_threshold <= self.down_threshold:
            raise ValueError(
                f"up_threshold ({self.up_threshold}) doit être > "
                f"down_threshold ({self.down_threshold})"
            )
        if not (0.0 < self.step <= 1.0):
            raise ValueError(f"step doit être ∈ (0,1], reçu {self.step}")
        if self.update_interval <= 0:
            raise ValueError(f"update_interval doit être > 0, reçu {self.update_interval}")
