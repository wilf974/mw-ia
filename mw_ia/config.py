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
    update_interval: int = 200          # épisodes (default V2-X consolidé 2026-05-22 : 50 trop agressif sur procedural, l'agent décroche entre paliers 0.05 → 0.10)

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


@dataclass(frozen=True)
class DRQNConfig:
    """Deep Recurrent Q-Network (LSTM). Successeur V2-Y de DQNConfig.

    ATTENTION : replay_capacity compte des TRAJECTOIRES (pas transitions
    comme DQNConfig.replay_capacity).
    """

    # Réseau
    fc_hidden: int = 256                # couche FC avant LSTM
    lstm_hidden: int = 128              # taille du hidden state LSTM

    # Sequence
    sequence_length: int = 32

    # Replay (NOMBRE DE TRAJECTOIRES, pas transitions)
    replay_capacity: int = 5_000
    min_episodes_to_learn: int = 100
    train_steps_per_episode: int = 4

    # Optimisation (identique V1)
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000  # default V2-X gagnant
    target_sync_steps: int = 1_000
    use_amp: bool = True

    # Training
    episodes: int = 5_000
    max_steps_per_episode: int = 200

    def __post_init__(self) -> None:
        if not (1 <= self.sequence_length <= self.max_steps_per_episode):
            raise ValueError(
                f"sequence_length {self.sequence_length} hors [1, {self.max_steps_per_episode}]"
            )
        if self.replay_capacity <= 0:
            raise ValueError(f"replay_capacity doit être > 0, reçu {self.replay_capacity}")
        if self.min_episodes_to_learn <= 0:
            raise ValueError(
                f"min_episodes_to_learn doit être > 0, reçu {self.min_episodes_to_learn}"
            )
        if self.train_steps_per_episode <= 0:
            raise ValueError(
                f"train_steps_per_episode doit être > 0, reçu {self.train_steps_per_episode}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size doit être > 0, reçu {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr doit être > 0, reçu {self.lr}")
        if not (0.0 < self.gamma < 1.0):
            raise ValueError(f"gamma doit être ∈ (0,1), reçu {self.gamma}")
        if not (0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0):
            raise ValueError(
                f"epsilon invalide : start={self.epsilon_start}, end={self.epsilon_end}"
            )
        if self.epsilon_decay_steps <= 0:
            raise ValueError(
                f"epsilon_decay_steps doit être > 0, reçu {self.epsilon_decay_steps}"
            )
        if self.target_sync_steps <= 0:
            raise ValueError(
                f"target_sync_steps doit être > 0, reçu {self.target_sync_steps}"
            )
        if self.fc_hidden <= 0 or self.lstm_hidden <= 0:
            raise ValueError(
                f"fc_hidden et lstm_hidden doivent être > 0, reçu fc={self.fc_hidden}, lstm={self.lstm_hidden}"
            )
        if self.episodes <= 0:
            raise ValueError(f"episodes doit être > 0, reçu {self.episodes}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(
                f"max_steps_per_episode doit être > 0, reçu {self.max_steps_per_episode}"
            )
