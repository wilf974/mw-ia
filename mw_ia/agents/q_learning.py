"""Q-Learning tabulaire avec ε-greedy decay linéaire."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mw_ia.agents.base import Agent
from mw_ia.config import QLearningConfig
from mw_ia.envs.gridworld import GridWorld


@dataclass(frozen=True)
class Transition:
    state: tuple[int, int]
    action: int
    reward: float
    next_state: tuple[int, int]
    terminated: bool
    truncated: bool


class QLearningAgent(Agent):
    """Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]."""

    def __init__(self, env: GridWorld, cfg: QLearningConfig, *, seed: int = 0) -> None:
        self.env = env
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self.Q: np.ndarray = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
        self._epsilon: float = cfg.epsilon_start
        self._episode: int = 0

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def start_episode(self, episode: int) -> None:
        self._episode = episode
        if self.cfg.epsilon_decay_episodes <= 0:
            self._epsilon = self.cfg.epsilon_end
        else:
            frac = min(1.0, episode / self.cfg.epsilon_decay_episodes)
            self._epsilon = (
                self.cfg.epsilon_start
                + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)
            )

    def act(self, state: tuple[int, int], *, greedy: bool = False) -> int:
        if (not greedy) and self._rng.random() < self._epsilon:
            return int(self._rng.integers(0, self.env.n_actions))
        idx = self.env.state_to_index(state)
        row = self.Q[idx]
        max_v = row.max()
        best = np.flatnonzero(row == max_v)
        return int(self._rng.choice(best))

    def learn(self, transition: Transition) -> dict[str, float]:
        s = self.env.state_to_index(transition.state)
        s2 = self.env.state_to_index(transition.next_state)
        target = transition.reward
        if not transition.terminated:
            target += self.cfg.gamma * self.Q[s2].max()
        td_error = target - self.Q[s, transition.action]
        self.Q[s, transition.action] += self.cfg.alpha * td_error
        return {"td_error": float(td_error)}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p.with_suffix(".npz"), Q=self.Q, epsilon=np.array([self._epsilon]))

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path).with_suffix(".npz"))
        self.Q = data["Q"]
        self._epsilon = float(data["epsilon"][0])
