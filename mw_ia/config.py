"""Hyperparamètres centralisés MW_IA — voir design doc §5."""
from __future__ import annotations

from dataclasses import dataclass, field


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
