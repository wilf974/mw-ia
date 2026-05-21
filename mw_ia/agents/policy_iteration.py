"""Policy Iteration — évaluation puis amélioration."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mw_ia.agents.base import Agent
from mw_ia.envs.gridworld import GridWorld


class PolicyIteration(Agent):
    """Alterne évaluation de politique et amélioration greedy."""

    def __init__(self, env: GridWorld, gamma: float = 0.99, theta: float = 1e-6) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V: np.ndarray = np.zeros(env.n_states, dtype=np.float64)
        self.policy: np.ndarray = np.random.randint(
            0, env.n_actions, size=env.n_states,
        ).astype(np.int64)

    def _simulate(self, state: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool]:
        saved_state, saved_step = self.env._state, self.env._step_count
        self.env._state = state
        self.env._step_count = 0
        s2, r, terminated, _truncated, _info = self.env.step(action)
        self.env._state, self.env._step_count = saved_state, saved_step
        return s2, r, terminated

    def _evaluate(self, max_sweeps: int = 1000) -> None:
        for _ in range(max_sweeps):
            delta = 0.0
            for idx in range(self.env.n_states):
                state = self.env.index_to_state(idx)
                if state == self.env.cfg.goal:
                    self.V[idx] = 0.0
                    continue
                a = int(self.policy[idx])
                s2, r, terminated = self._simulate(state, a)
                v2 = 0.0 if terminated else self.V[self.env.state_to_index(s2)]
                new_v = r + self.gamma * v2
                delta = max(delta, abs(new_v - self.V[idx]))
                self.V[idx] = new_v
            if delta < self.theta:
                return

    def _improve(self) -> bool:
        """Renvoie True si la politique est stable (inchangée)."""
        stable = True
        for idx in range(self.env.n_states):
            state = self.env.index_to_state(idx)
            if state == self.env.cfg.goal:
                continue
            old = int(self.policy[idx])
            best = -np.inf
            best_a = old
            for a in range(self.env.n_actions):
                s2, r, terminated = self._simulate(state, a)
                v2 = 0.0 if terminated else self.V[self.env.state_to_index(s2)]
                q = r + self.gamma * v2
                if q > best:
                    best = q
                    best_a = a
            self.policy[idx] = best_a
            if best_a != old:
                stable = False
        return stable

    def solve(self, max_iters: int = 50) -> int:
        for it in range(max_iters):
            self._evaluate()
            if self._improve():
                return it + 1
        return max_iters

    def act(self, state: tuple[int, int], *, greedy: bool = True) -> int:
        return int(self.policy[self.env.state_to_index(state)])

    def learn(self, transition: object) -> dict[str, float]:
        return {}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p.with_suffix(".npz"), V=self.V, policy=self.policy,
                 gamma=np.array([self.gamma]))

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path).with_suffix(".npz"))
        self.V = data["V"]
        self.policy = data["policy"]
        self.gamma = float(data["gamma"][0])
