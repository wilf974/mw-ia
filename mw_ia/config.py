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
    max_steps: int = 200                # truncation per épisode du GridWorld interne
                                        # (normalisation d'horizon quand la grille scale :
                                        #  10x10 → 200, 15x15 → 400, 20x20 → 600-800)

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
        if self.max_steps <= 0:
            raise ValueError(f"max_steps doit être > 0, reçu {self.max_steps}")


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration du scheduler adaptatif de difficulté."""

    initial_difficulty: float = 0.0
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    up_threshold: float = 0.80
    down_threshold: float = 0.30
    step: float = 0.05                  # default V2-X consolidé 2026-05-22 : 0.025 testé empiriquement régresse (oscille 3 paliers), 0.05 reste le meilleur compromis stabilité sur DQN feedforward
    update_interval: int = 200          # épisodes (default V2-X consolidé 2026-05-22 : 50 trop agressif sur procedural, 200 = agent consolide vraiment chaque palier)

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
    polyak_tau: float = 0.0   # V2-U : 0.0 = hard sync, >0 = soft Polyak (Lillicrap 2015).

    # V2-B0 : Prioritized Experience Replay trajectory-level (Schaul 2015 + R2D2)
    per_enabled: bool = False
    per_alpha: float = 0.6              # priority exponent (Schaul 2015)
    per_beta_start: float = 0.4         # IS exponent initial
    per_beta_end: float = 1.0           # IS exponent final
    per_eta: float = 0.9                # R2D2 aggregation : eta*max + (1-eta)*mean
    per_epsilon: float = 1e-6           # small constant garantit priority > 0

    # V2-B1a : Policy Snapshot Rehearsal (sliding window N captures x snapshot_size traj)
    b1a_enabled: bool = False
    b1a_snapshot_size: int = 50      # nombre de trajectoires capturees par best
    b1a_n_windows: int = 3           # sliding window FIFO
    b1a_mix_ratio: float = 0.2       # fraction du batch venant du snapshot

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
        if not (0.0 <= self.polyak_tau <= 1.0):
            raise ValueError(
                f"polyak_tau doit être ∈ [0, 1], reçu {self.polyak_tau}"
            )
        if not (0.0 <= self.per_alpha <= 1.0):
            raise ValueError(f"per_alpha doit etre dans [0, 1], recu {self.per_alpha}")
        if not (0.0 <= self.per_beta_start <= 1.0):
            raise ValueError(
                f"per_beta_start doit etre dans [0, 1], recu {self.per_beta_start}"
            )
        if not (0.0 <= self.per_beta_end <= 1.0):
            raise ValueError(
                f"per_beta_end doit etre dans [0, 1], recu {self.per_beta_end}"
            )
        if not (0.0 <= self.per_eta <= 1.0):
            raise ValueError(f"per_eta doit etre dans [0, 1], recu {self.per_eta}")
        if self.per_epsilon <= 0.0:
            raise ValueError(f"per_epsilon doit etre > 0, recu {self.per_epsilon}")
        if self.b1a_snapshot_size <= 0:
            raise ValueError(f"b1a_snapshot_size doit etre > 0, recu {self.b1a_snapshot_size}")
        if self.b1a_n_windows <= 0:
            raise ValueError(f"b1a_n_windows doit etre > 0, recu {self.b1a_n_windows}")
        if not (0.0 < self.b1a_mix_ratio < 1.0):
            raise ValueError(
                f"b1a_mix_ratio doit etre dans ]0, 1[, recu {self.b1a_mix_ratio}"
            )


@dataclass(frozen=True)
class ConvDQNConfig:
    """Convolutional Deep Q-Network (V2-Z).

    Architecture : (Conv2d → ReLU)* → Flatten → Linear → ReLU → Linear.
    Input attendu : tensor (B, 3, max_rows, max_cols) via
    encode_procedural_observation_2d.

    Champs dupliqués depuis DQNConfig (pas d'héritage) pour rester frozen
    et explicit, cohérent V2-X ProceduralEnvConfig et V2-Y DRQNConfig.
    """

    # Conv-spécifique
    conv_channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    padding: int = 1
    fc_hidden: int = 256

    # Champs partagés avec DQNConfig (duplication assumée)
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000   # Default V2-X gagnant (vs V1 50000)
    replay_capacity: int = 100_000
    min_replay_to_learn: int = 1_000
    target_sync_steps: int = 1_000
    train_every: int = 4
    use_amp: bool = True
    double_dqn: bool = True   # V2-W : Hasselt 2015. False = V2-Z baseline DQN classique.
    polyak_tau: float = 0.0   # V2-U : 0.0 = hard sync, >0 = soft Polyak (Lillicrap 2015).
    # V2-V : Training Protocol Stabilization (eval périodique + best-checkpoint)
    eval_enabled: bool = True
    eval_every_episodes: int = 100
    eval_seeds: tuple[int, ...] = tuple(range(10_000, 10_010))
    eval_max_steps: int = 200
    # V2-V fix : eval à diff FIXE (pas scheduler.current). Sans ça, le best
    # capture l'agent trivial à diff=0.00 (winrate 100% sur mazes vides) et
    # n'est jamais battu par l'agent compétent à diff supérieure.
    eval_target_difficulty: float = 0.30
    best_checkpoint_path: str | None = None
    episodes: int = 5_000
    max_steps_per_episode: int = 200

    def __post_init__(self) -> None:
        if len(self.conv_channels) == 0:
            raise ValueError("conv_channels ne peut pas être vide")
        if any(c <= 0 for c in self.conv_channels):
            raise ValueError(
                f"conv_channels doivent être > 0, reçu {self.conv_channels}"
            )
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size doit être > 0, reçu {self.kernel_size}")
        if self.padding < 0:
            raise ValueError(f"padding doit être >= 0, reçu {self.padding}")
        if self.fc_hidden <= 0:
            raise ValueError(f"fc_hidden doit être > 0, reçu {self.fc_hidden}")
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
        if self.replay_capacity <= 0:
            raise ValueError(f"replay_capacity doit être > 0, reçu {self.replay_capacity}")
        if self.min_replay_to_learn <= 0:
            raise ValueError(
                f"min_replay_to_learn doit être > 0, reçu {self.min_replay_to_learn}"
            )
        if self.target_sync_steps <= 0:
            raise ValueError(
                f"target_sync_steps doit être > 0, reçu {self.target_sync_steps}"
            )
        if self.train_every <= 0:
            raise ValueError(f"train_every doit être > 0, reçu {self.train_every}")
        if self.episodes <= 0:
            raise ValueError(f"episodes doit être > 0, reçu {self.episodes}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(
                f"max_steps_per_episode doit être > 0, reçu {self.max_steps_per_episode}"
            )
        if self.eval_every_episodes <= 0:
            raise ValueError(
                f"eval_every_episodes doit être > 0, reçu {self.eval_every_episodes}"
            )
        if len(self.eval_seeds) == 0:
            raise ValueError("eval_seeds ne peut pas être vide")
        if self.eval_max_steps <= 0:
            raise ValueError(f"eval_max_steps doit être > 0, reçu {self.eval_max_steps}")
        if not (0.0 <= self.eval_target_difficulty <= 1.0):
            raise ValueError(
                f"eval_target_difficulty doit être ∈ [0,1], reçu {self.eval_target_difficulty}"
            )
        if not (0.0 <= self.polyak_tau <= 1.0):
            raise ValueError(
                f"polyak_tau doit être ∈ [0, 1], reçu {self.polyak_tau}"
            )


@dataclass(frozen=True)
class ConvRecurrentDQNConfig:
    """V2-ZY : Conv2D + LSTM + Double DQN combiné, avec V2-V eval activé.

    Combo des 3 leviers livrés V2-Z (perception spatiale), V2-Y (mémoire),
    V2-W (Double DQN). Réseau ConvRecurrentQNetwork, buffer SequenceReplayBuffer
    V2-Y, trainer RecurrentDQNTrainer V2-Y étendu avec flag double_dqn.

    Champs combinés V2-Y DRQNConfig + V2-Z ConvDQNConfig + V2-W double_dqn + V2-V eval_*.
    """

    # Conv-spécifique (V2-Z pattern)
    conv_channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    padding: int = 1

    # LSTM (V2-Y pattern)
    lstm_hidden: int = 128
    sequence_length: int = 32

    # Replay (TRAJECTOIRES, pas transitions — V2-Y pattern)
    replay_capacity: int = 5_000
    min_episodes_to_learn: int = 100
    train_steps_per_episode: int = 4

    # Optimisation
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    target_sync_steps: int = 1_000
    use_amp: bool = True

    # V2-W : Double DQN activé par défaut V2-ZY (combo des 3 leviers)
    double_dqn: bool = True
    polyak_tau: float = 0.0   # V2-U : 0.0 = hard sync, >0 = soft Polyak.

    # V2-B0 : Prioritized Experience Replay trajectory-level (Schaul 2015 + R2D2)
    per_enabled: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_eta: float = 0.9
    per_epsilon: float = 1e-6

    # V2-B1a : Policy Snapshot Rehearsal (sliding window N captures x snapshot_size traj)
    b1a_enabled: bool = False
    b1a_snapshot_size: int = 50
    b1a_n_windows: int = 3
    b1a_mix_ratio: float = 0.2

    # V2-V : Training Protocol Stabilization (activé par défaut V2-ZY)
    eval_enabled: bool = True
    eval_every_episodes: int = 100
    eval_seeds: tuple[int, ...] = tuple(range(10_000, 10_010))
    eval_max_steps: int = 200
    eval_target_difficulty: float = 0.30
    best_checkpoint_path: str | None = None

    # Training
    episodes: int = 5_000
    max_steps_per_episode: int = 200

    def __post_init__(self) -> None:
        if len(self.conv_channels) == 0:
            raise ValueError("conv_channels ne peut pas être vide")
        if any(c <= 0 for c in self.conv_channels):
            raise ValueError(
                f"conv_channels doivent être > 0, reçu {self.conv_channels}"
            )
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size doit être > 0, reçu {self.kernel_size}")
        if self.padding < 0:
            raise ValueError(f"padding doit être >= 0, reçu {self.padding}")
        if self.lstm_hidden <= 0:
            raise ValueError(f"lstm_hidden doit être > 0, reçu {self.lstm_hidden}")
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
        if self.eval_every_episodes <= 0:
            raise ValueError(
                f"eval_every_episodes doit être > 0, reçu {self.eval_every_episodes}"
            )
        if len(self.eval_seeds) == 0:
            raise ValueError("eval_seeds ne peut pas être vide")
        if self.eval_max_steps <= 0:
            raise ValueError(f"eval_max_steps doit être > 0, reçu {self.eval_max_steps}")
        if not (0.0 <= self.eval_target_difficulty <= 1.0):
            raise ValueError(
                f"eval_target_difficulty doit être ∈ [0,1], reçu {self.eval_target_difficulty}"
            )
        if self.episodes <= 0:
            raise ValueError(f"episodes doit être > 0, reçu {self.episodes}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(
                f"max_steps_per_episode doit être > 0, reçu {self.max_steps_per_episode}"
            )
        if not (0.0 <= self.polyak_tau <= 1.0):
            raise ValueError(
                f"polyak_tau doit être ∈ [0, 1], reçu {self.polyak_tau}"
            )
        if not (0.0 <= self.per_alpha <= 1.0):
            raise ValueError(f"per_alpha doit etre dans [0, 1], recu {self.per_alpha}")
        if not (0.0 <= self.per_beta_start <= 1.0):
            raise ValueError(
                f"per_beta_start doit etre dans [0, 1], recu {self.per_beta_start}"
            )
        if not (0.0 <= self.per_beta_end <= 1.0):
            raise ValueError(
                f"per_beta_end doit etre dans [0, 1], recu {self.per_beta_end}"
            )
        if not (0.0 <= self.per_eta <= 1.0):
            raise ValueError(f"per_eta doit etre dans [0, 1], recu {self.per_eta}")
        if self.per_epsilon <= 0.0:
            raise ValueError(f"per_epsilon doit etre > 0, recu {self.per_epsilon}")
        if self.b1a_snapshot_size <= 0:
            raise ValueError(f"b1a_snapshot_size doit etre > 0, recu {self.b1a_snapshot_size}")
        if self.b1a_n_windows <= 0:
            raise ValueError(f"b1a_n_windows doit etre > 0, recu {self.b1a_n_windows}")
        if not (0.0 < self.b1a_mix_ratio < 1.0):
            raise ValueError(
                f"b1a_mix_ratio doit etre dans ]0, 1[, recu {self.b1a_mix_ratio}"
            )
