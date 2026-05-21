"""Boucles d'entraînement (tabulaire + DQN) callback-friendly.

Aucune dépendance Qt ici : la GUI fournira un wrapper QThread qui
branche `RunnerCallbacks` sur ses `pyqtSignal`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mw_ia.agents.dqn import DQNAgent
from mw_ia.agents.q_learning import QLearningAgent, Transition
from mw_ia.config import DQNConfig, QLearningConfig, TrainingConfig
from mw_ia.envs.gridworld import GridWorld
from mw_ia.training.metrics import MetricsTracker


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
