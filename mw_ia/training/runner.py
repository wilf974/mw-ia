"""Boucles d'entraînement (tabulaire + DQN) callback-friendly.

Aucune dépendance Qt ici : la GUI fournira un wrapper QThread qui
branche `RunnerCallbacks` sur ses `pyqtSignal`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mw_ia.agents.conv_dqn import ConvDQNAgent
from mw_ia.agents.dqn import DQNAgent
from mw_ia.agents.q_learning import QLearningAgent, Transition
from mw_ia.agents.recurrent_dqn import RecurrentDQNAgent
from mw_ia.config import ConvDQNConfig, DQNConfig, DRQNConfig, ProceduralEnvConfig, QLearningConfig, SchedulerConfig, TrainingConfig
from mw_ia.envs.gridworld import GridWorld
from mw_ia.envs.procedural_env import ProceduralGridWorld, encode_procedural_observation, encode_procedural_observation_2d
from mw_ia.training.checkpoint_tracker import BestCheckpointTracker
from mw_ia.training.evaluator import PeriodicEvaluator
from mw_ia.training.metrics import DifficultyBucketTracker, MetricsTracker
from mw_ia.training.scheduler import AdaptiveDifficultyScheduler


StepCb = Callable[..., None]
EpisodeCb = Callable[..., None]
LossCb = Callable[[int, float], None]
EpsilonCb = Callable[[int, float], None]
LogCb = Callable[[str, str], None]


@dataclass
class RunnerCallbacks:
    on_step: StepCb | None = None
    on_episode: EpisodeCb | None = None
    on_loss: LossCb | None = None
    on_epsilon: EpsilonCb | None = None
    on_log: LogCb | None = None
    on_maze_changed: Callable[..., None] | None = None
    on_difficulty_updated: Callable[..., None] | None = None
    on_eval: Callable[..., None] | None = None

    def fire_step(self, **kw: object) -> None:
        if self.on_step:
            self.on_step(**kw)

    def fire_episode(self, **kw: object) -> None:
        if self.on_episode:
            self.on_episode(**kw)

    def fire_loss(self, step: int, loss: float) -> None:
        if self.on_loss:
            self.on_loss(step, loss)

    def fire_epsilon(self, step: int, eps: float) -> None:
        if self.on_epsilon:
            self.on_epsilon(step, eps)

    def fire_log(self, level: str, msg: str) -> None:
        if self.on_log:
            self.on_log(level, msg)

    def fire_maze_changed(self, **kw: object) -> None:
        if self.on_maze_changed:
            self.on_maze_changed(**kw)

    def fire_difficulty_updated(self, **kw: object) -> None:
        if self.on_difficulty_updated:
            self.on_difficulty_updated(**kw)

    def fire_evaluation(self, **kw: object) -> None:
        if self.on_eval:
            self.on_eval(**kw)


class _BaseRunner:
    """Logique commune : pause / stop."""

    def __init__(self, train_cfg: TrainingConfig, callbacks: RunnerCallbacks) -> None:
        self.train_cfg = train_cfg
        self.callbacks = callbacks
        self.metrics = MetricsTracker(train_cfg)
        self._stop = False
        self._paused = False

    def request_stop(self) -> None:
        self._stop = True

    def request_pause(self, paused: bool) -> None:
        self._paused = paused

    def is_running(self) -> bool:
        return not self._stop


class TabularRunner(_BaseRunner):
    """Boucle Q-Learning tabulaire."""

    def __init__(
        self,
        env: GridWorld,
        qcfg: QLearningConfig,
        train_cfg: TrainingConfig,
        callbacks: RunnerCallbacks,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__(train_cfg, callbacks)
        self.env = env
        self.qcfg = qcfg
        self.agent = QLearningAgent(env, qcfg, seed=seed)

    def run(self) -> None:
        self.callbacks.fire_log("info", "Tabular Q-Learning : démarrage")
        for ep in range(self.qcfg.episodes):
            if self._stop:
                self.callbacks.fire_log("warning", "Arrêt demandé")
                return
            state, _ = self.env.reset()
            self.agent.start_episode(ep)
            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated):
                if self._stop:
                    return
                while self._paused and not self._stop:
                    pass
                a = self.agent.act(state)
                s2, r, terminated, truncated, _ = self.env.step(a)
                self.agent.learn(Transition(state, a, r, s2, terminated, truncated))
                self.callbacks.fire_step(state=state, action=a, reward=r, next_state=s2)
                state = s2
                ep_reward += r
                ep_len += 1
            self.metrics.record_episode(ep_reward, ep_len, success=terminated)
            self.metrics.record_epsilon(self.agent.epsilon)
            self.callbacks.fire_epsilon(ep, self.agent.epsilon)
            self.callbacks.fire_episode(ep=ep, reward=ep_reward, length=ep_len, success=terminated)
            if ep % self.train_cfg.log_every_episodes == 0:
                wr = self.metrics.winrate()
                self.callbacks.fire_log(
                    "info",
                    f"ep {ep:>4}  R={ep_reward:+.2f}  L={ep_len:>3}  "
                    f"eps={self.agent.epsilon:.3f}  winrate={wr:.2%}",
                )


class DQNRunner(_BaseRunner):
    """Boucle DQN avec replay + target sync."""

    def __init__(
        self,
        env: GridWorld,
        dcfg: DQNConfig,
        train_cfg: TrainingConfig,
        callbacks: RunnerCallbacks,
        *,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        super().__init__(train_cfg, callbacks)
        self.env = env
        self.dcfg = dcfg
        self.agent = DQNAgent(env, dcfg, device=device, seed=seed)

    def _state_vec(self, s: tuple[int, int]) -> np.ndarray:
        v = np.zeros(self.env.n_states, dtype=np.float32)
        v[self.env.state_to_index(s)] = 1.0
        return v

    def run(self) -> None:
        self.callbacks.fire_log("info", f"DQN sur device={self.agent.device} démarrage")
        for ep in range(self.dcfg.episodes):
            if self._stop:
                return
            state, _ = self.env.reset()
            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated) and ep_len < self.dcfg.max_steps_per_episode:
                if self._stop:
                    return
                while self._paused and not self._stop:
                    pass
                sv = self._state_vec(state)
                a = self.agent.act(sv)
                s2, r, terminated, truncated, _ = self.env.step(a)
                m = self.agent.observe(sv, a, r, self._state_vec(s2),
                                       terminated or truncated)
                if "loss" in m:
                    self.metrics.record_loss(m["loss"])
                    self.callbacks.fire_loss(self.agent.global_step, m["loss"])
                self.callbacks.fire_epsilon(self.agent.global_step, m["epsilon"])
                self.metrics.record_epsilon(m["epsilon"])
                self.callbacks.fire_step(state=state, action=a, reward=r, next_state=s2)
                state = s2
                ep_reward += r
                ep_len += 1
            self.metrics.record_episode(ep_reward, ep_len, success=terminated)
            self.callbacks.fire_episode(ep=ep, reward=ep_reward, length=ep_len, success=terminated)
            if ep % self.train_cfg.log_every_episodes == 0:
                self.callbacks.fire_log(
                    "info",
                    f"ep {ep:>4}  R={ep_reward:+.2f}  L={ep_len:>3}  "
                    f"eps={self.agent.epsilon:.3f}  winrate={self.metrics.winrate():.2%}",
                )


class ProceduralDQNRunner(_BaseRunner):
    """Boucle DQN sur environnement procédural avec curriculum adaptatif.

    Différences avec DQNRunner V1 :
    - env régénère le maze à chaque reset()
    - observation = position_one_hot + grid_flatten (dim 2*max_rows*max_cols)
    - scheduler adapte la difficulté toutes les update_interval épisodes
    - bucket_tracker route les épisodes par bucket de difficulté
    - callbacks GUI étendus : on_maze_changed, on_difficulty_updated

    NOTE : le replay buffer mélange volontairement transitions inter-épisodes.
    C'est précisément le but du curriculum : apprendre une politique générale,
    pas une politique map-spécifique.
    """

    def __init__(
        self,
        *,
        env: ProceduralGridWorld,
        proc_cfg: ProceduralEnvConfig,
        dqn_cfg: DQNConfig,
        sched_cfg: SchedulerConfig,
        train_cfg: TrainingConfig,
        callbacks: RunnerCallbacks,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        super().__init__(train_cfg, callbacks)
        self.env = env
        self.proc_cfg = proc_cfg
        self.dqn_cfg = dqn_cfg
        self.sched_cfg = sched_cfg
        self.scheduler = AdaptiveDifficultyScheduler(sched_cfg)
        self.bucket_tracker = DifficultyBucketTracker(train_cfg)
        # Observation dim = 2 * max_rows * max_cols (one-hot pos + grid flatten)
        obs_dim = 2 * proc_cfg.max_rows * proc_cfg.max_cols
        # DQNAgent attend un env avec n_states/n_actions ; on adapte
        # en injectant directement obs_dim via un objet léger.
        self.agent = self._make_agent(obs_dim=obs_dim, device=device, seed=seed)

    def _make_agent(self, *, obs_dim: int, device: str, seed: int) -> "DQNAgent":
        """Construit un DQNAgent qui consomme des observations de dim obs_dim.

        Hack léger : DQNAgent V1 dérive obs_dim de env.n_states. On bypass en
        wrappant l'env procedural dans un objet exposant n_states=obs_dim et
        n_actions=4.
        """
        class _ObsDimEnv:
            n_states = obs_dim
            n_actions = 4

        return DQNAgent(_ObsDimEnv(), self.dqn_cfg, device=device, seed=seed)

    def run(self) -> None:
        obs_dim = 2 * self.proc_cfg.max_rows * self.proc_cfg.max_cols
        self.callbacks.fire_log(
            "info",
            f"Procedural DQN ({self.proc_cfg.mode}) sur {self.agent.device} démarrage"
        )
        self.callbacks.fire_log(
            "info",
            f"Config: hidden={self.dqn_cfg.hidden_layers} "
            f"epsilon_decay_steps={self.dqn_cfg.epsilon_decay_steps} "
            f"min_density={self.proc_cfg.min_density} "
            f"max_density={self.proc_cfg.max_density} "
            f"obs_dim={obs_dim} seed={self.train_cfg.seed}"
        )
        for ep in range(self.dqn_cfg.episodes):
            if self._stop:
                return

            self.env.set_difficulty(self.scheduler.current)
            state, info = self.env.reset(seed=ep)
            maze = info["maze"]
            difficulty = info["difficulty"]
            self.callbacks.fire_maze_changed(maze=maze, episode_id=ep, difficulty=difficulty)

            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated) and ep_len < self.dqn_cfg.max_steps_per_episode:
                if self._stop:
                    return
                while self._paused and not self._stop:
                    pass
                obs = encode_procedural_observation(
                    state=state, grid=maze,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                a = self.agent.act(obs)
                s2, r, terminated, truncated, _ = self.env.step(a)
                next_obs = encode_procedural_observation(
                    state=s2, grid=maze,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                m = self.agent.observe(obs, a, r, next_obs, terminated or truncated)
                if "loss" in m:
                    self.metrics.record_loss(m["loss"])
                    self.callbacks.fire_loss(self.agent.global_step, m["loss"])
                self.callbacks.fire_epsilon(self.agent.global_step, m["epsilon"])
                self.metrics.record_epsilon(m["epsilon"])
                self.callbacks.fire_step(state=state, action=a, reward=r, next_state=s2)
                state = s2
                ep_reward += r
                ep_len += 1

            self.metrics.record_episode(ep_reward, ep_len, success=terminated)
            self.bucket_tracker.record_episode(
                success=terminated, reward=ep_reward, length=ep_len, difficulty=difficulty,
            )
            self.callbacks.fire_episode(ep=ep, reward=ep_reward, length=ep_len, success=terminated)

            if (ep + 1) % self.sched_cfg.update_interval == 0:
                new_diff = self.scheduler.update(winrate=self.metrics.winrate())
                self.callbacks.fire_difficulty_updated(difficulty=new_diff, episode_id=ep)

            if ep % self.train_cfg.log_every_episodes == 0:
                self.callbacks.fire_log(
                    "info",
                    f"ep {ep:>4}  R={ep_reward:+.2f}  L={ep_len:>3}  "
                    f"eps={self.agent.epsilon:.3f}  winrate={self.metrics.winrate():.2%}  "
                    f"diff={self.scheduler.current:.2f}"
                )


class RecurrentProceduralDQNRunner(_BaseRunner):
    """Boucle DRQN sur environnement procédural avec curriculum adaptatif.

    Différences avec ProceduralDQNRunner V2-X :
    - Agent récurrent (LSTM) au lieu de DQN feedforward
    - Hidden state runtime reset à chaque épisode (agent.reset_hidden())
    - Train step à la FIN de l'épisode (agent.end_episode()), pas à chaque step
    - Observation et callbacks GUI identiques V2-X
    """

    def __init__(
        self,
        *,
        env: ProceduralGridWorld,
        proc_cfg: ProceduralEnvConfig,
        drqn_cfg: DRQNConfig,
        sched_cfg: SchedulerConfig,
        train_cfg: TrainingConfig,
        callbacks: RunnerCallbacks,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        super().__init__(train_cfg, callbacks)
        self.env = env
        self.proc_cfg = proc_cfg
        self.drqn_cfg = drqn_cfg
        self.sched_cfg = sched_cfg
        self.scheduler = AdaptiveDifficultyScheduler(sched_cfg)
        self.bucket_tracker = DifficultyBucketTracker(train_cfg)
        obs_dim = 2 * proc_cfg.max_rows * proc_cfg.max_cols
        self.agent = RecurrentDQNAgent(
            obs_dim=obs_dim, n_actions=4, cfg=drqn_cfg, device=device, seed=seed,
        )

    def run(self) -> None:
        obs_dim = 2 * self.proc_cfg.max_rows * self.proc_cfg.max_cols
        self.callbacks.fire_log(
            "info",
            f"Recurrent DQN ({self.proc_cfg.mode}) sur {self.agent.device} démarrage"
        )
        self.callbacks.fire_log(
            "info",
            f"Config: fc_hidden={self.drqn_cfg.fc_hidden} "
            f"lstm_hidden={self.drqn_cfg.lstm_hidden} "
            f"sequence_length={self.drqn_cfg.sequence_length} "
            f"epsilon_decay_steps={self.drqn_cfg.epsilon_decay_steps} "
            f"min_density={self.proc_cfg.min_density} "
            f"max_density={self.proc_cfg.max_density} "
            f"obs_dim={obs_dim} seed={self.train_cfg.seed}"
        )
        for ep in range(self.drqn_cfg.episodes):
            if self._stop:
                return

            self.env.set_difficulty(self.scheduler.current)
            state, info = self.env.reset(seed=ep)
            maze = info["maze"]
            difficulty = info["difficulty"]
            self.callbacks.fire_maze_changed(maze=maze, episode_id=ep, difficulty=difficulty)

            self.agent.reset_hidden()
            self.agent.begin_episode()

            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated) and ep_len < self.drqn_cfg.max_steps_per_episode:
                if self._stop:
                    return
                while self._paused and not self._stop:
                    pass
                obs = encode_procedural_observation(
                    state=state, grid=maze,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                a = self.agent.act(obs)
                s2, r, terminated, truncated, _ = self.env.step(a)
                next_obs = encode_procedural_observation(
                    state=s2, grid=maze,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                self.agent.observe(obs, a, r, next_obs, terminated or truncated)
                self.callbacks.fire_step(state=state, action=a, reward=r, next_state=s2)
                state = s2
                ep_reward += r
                ep_len += 1

            m = self.agent.end_episode()
            if "loss" in m:
                self.metrics.record_loss(m["loss"])
                self.callbacks.fire_loss(self.agent.global_step, m["loss"])
            self.callbacks.fire_epsilon(self.agent.global_step, m["epsilon"])
            self.metrics.record_epsilon(m["epsilon"])

            self.metrics.record_episode(ep_reward, ep_len, success=terminated)
            self.bucket_tracker.record_episode(
                success=terminated, reward=ep_reward, length=ep_len, difficulty=difficulty,
            )
            self.callbacks.fire_episode(ep=ep, reward=ep_reward, length=ep_len, success=terminated)

            if (ep + 1) % self.sched_cfg.update_interval == 0:
                new_diff = self.scheduler.update(winrate=self.metrics.winrate())
                self.callbacks.fire_difficulty_updated(difficulty=new_diff, episode_id=ep)

            if ep % self.train_cfg.log_every_episodes == 0:
                self.callbacks.fire_log(
                    "info",
                    f"ep {ep:>4}  R={ep_reward:+.2f}  L={ep_len:>3}  "
                    f"eps={self.agent.epsilon:.3f}  winrate={self.metrics.winrate():.2%}  "
                    f"diff={self.scheduler.current:.2f}"
                )


class ConvProceduralDQNRunner(_BaseRunner):
    """Boucle DQN procedural avec perception spatiale (V2-Z).

    Différences avec ProceduralDQNRunner V2-X :
    - Agent ConvDQNAgent (Conv2d) au lieu de DQNAgent (MLP).
    - Observation encode_procedural_observation_2d shape (3, R, C).
    - Callbacks GUI identiques V2-X (signaux maze_changed + difficulty_updated).

    Scheduler defaults V2-X (`update_interval=200`, `step=0.05`) — cohérent
    DQN feedforward, distinct du runner V2-Y LSTM.
    """

    def __init__(
        self,
        *,
        env: ProceduralGridWorld,
        proc_cfg: ProceduralEnvConfig,
        dqn_cfg: ConvDQNConfig,
        sched_cfg: SchedulerConfig,
        train_cfg: TrainingConfig,
        callbacks: RunnerCallbacks,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        super().__init__(train_cfg, callbacks)
        self.env = env
        self.proc_cfg = proc_cfg
        self.dqn_cfg = dqn_cfg
        self.sched_cfg = sched_cfg
        self.scheduler = AdaptiveDifficultyScheduler(sched_cfg)
        self.bucket_tracker = DifficultyBucketTracker(train_cfg)
        self.agent = ConvDQNAgent(
            in_channels=3, rows=proc_cfg.max_rows, cols=proc_cfg.max_cols,
            n_actions=4, cfg=dqn_cfg, device=device, seed=seed,
        )

        # V2-V : periodic assessment + best-checkpoint
        if dqn_cfg.eval_enabled:
            # Env d'assessment séparé avec MÊME proc_cfg que training mais générateur fresh
            eval_gen = type(env.generator).__new__(type(env.generator))
            eval_gen.__dict__.update(env.generator.__dict__)
            eval_env = ProceduralGridWorld(cfg=proc_cfg, generator=eval_gen)
            self.evaluator: PeriodicEvaluator | None = PeriodicEvaluator(
                eval_env=eval_env,
                eval_seeds=dqn_cfg.eval_seeds,
                max_steps=dqn_cfg.eval_max_steps,
                observation_encoder=encode_procedural_observation_2d,
                proc_cfg=proc_cfg,
            )
            self.best_tracker: BestCheckpointTracker | None = BestCheckpointTracker(
                path=dqn_cfg.best_checkpoint_path,
            )
        else:
            self.evaluator = None
            self.best_tracker = None

    def run(self) -> None:
        self.callbacks.fire_log(
            "info",
            f"Procedural Conv-DQN ({self.proc_cfg.mode}) sur {self.agent.device} démarrage"
        )
        self.callbacks.fire_log(
            "info",
            f"Config: conv_channels={self.dqn_cfg.conv_channels} "
            f"fc_hidden={self.dqn_cfg.fc_hidden} "
            f"epsilon_decay_steps={self.dqn_cfg.epsilon_decay_steps} "
            f"min_density={self.proc_cfg.min_density} "
            f"max_density={self.proc_cfg.max_density} "
            f"obs_shape=(3, {self.proc_cfg.max_rows}, {self.proc_cfg.max_cols}) "
            f"seed={self.train_cfg.seed}"
        )
        for ep in range(self.dqn_cfg.episodes):
            if self._stop:
                return

            self.env.set_difficulty(self.scheduler.current)
            state, info = self.env.reset(seed=ep)
            maze = info["maze"]
            difficulty = info["difficulty"]
            goal = self.env.inner.cfg.goal
            self.callbacks.fire_maze_changed(maze=maze, episode_id=ep, difficulty=difficulty)

            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated) and ep_len < self.dqn_cfg.max_steps_per_episode:
                if self._stop:
                    return
                while self._paused and not self._stop:
                    pass
                obs = encode_procedural_observation_2d(
                    state=state, grid=maze, goal=goal,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                a = self.agent.act(obs)
                s2, r, terminated, truncated, _ = self.env.step(a)
                next_obs = encode_procedural_observation_2d(
                    state=s2, grid=maze, goal=goal,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                m = self.agent.observe(obs, a, r, next_obs, terminated or truncated)
                if "loss" in m:
                    self.metrics.record_loss(m["loss"])
                    self.callbacks.fire_loss(self.agent.global_step, m["loss"])
                self.callbacks.fire_epsilon(self.agent.global_step, m["epsilon"])
                self.metrics.record_epsilon(m["epsilon"])
                self.callbacks.fire_step(state=state, action=a, reward=r, next_state=s2)
                state = s2
                ep_reward += r
                ep_len += 1

            self.metrics.record_episode(ep_reward, ep_len, success=terminated)
            self.bucket_tracker.record_episode(
                success=terminated, reward=ep_reward, length=ep_len, difficulty=difficulty,
            )
            self.callbacks.fire_episode(ep=ep, reward=ep_reward, length=ep_len, success=terminated)

            if (ep + 1) % self.sched_cfg.update_interval == 0:
                new_diff = self.scheduler.update(winrate=self.metrics.winrate())
                self.callbacks.fire_difficulty_updated(difficulty=new_diff, episode_id=ep)

            if ep % self.train_cfg.log_every_episodes == 0:
                self.callbacks.fire_log(
                    "info",
                    f"ep {ep:>4}  R={ep_reward:+.2f}  L={ep_len:>3}  "
                    f"eps={self.agent.epsilon:.3f}  winrate={self.metrics.winrate():.2%}  "
                    f"diff={self.scheduler.current:.2f}"
                )

            # V2-V : assessment périodique + best-checkpoint
            if (
                self.evaluator is not None
                and (ep + 1) % self.dqn_cfg.eval_every_episodes == 0
            ):
                assessment_metrics = self.evaluator.evaluate(self.agent, self.scheduler.current)
                improved = self.best_tracker.update(assessment_metrics, self.agent, episode=ep)
                self.callbacks.fire_evaluation(
                    ep=ep,
                    eval_winrate=assessment_metrics["winrate"],
                    eval_diff=assessment_metrics["difficulty"],
                    best_winrate=self.best_tracker.best_winrate,
                    best_episode=self.best_tracker.best_episode,
                    improved=improved,
                )
                self.callbacks.fire_log(
                    "info",
                    f"assessment ep {ep:>4} : winrate={assessment_metrics['winrate']:.2%} "
                    f"@ diff={assessment_metrics['difficulty']:.2f}  "
                    f"best={self.best_tracker.best_winrate:.2%} "
                    f"@ ep {self.best_tracker.best_episode}"
                    + ("  NEW BEST" if improved else "")
                )
