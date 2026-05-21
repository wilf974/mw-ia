"""Value Iteration — Bellman exact sur MDP déterministe connu."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mw_ia.agents.base import Agent
from mw_ia.envs.gridworld import GridWorld


class ValueIteration(Agent):
    """Résout exactement le MDP par itération de Bellman.

    V*(s) = max_a [R(s,a,s') + γ V*(s')]   (transitions déterministes)
    """

    def __init__(self, env: GridWorld, gamma: float = 0.99, theta: float = 1e-6) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V: np.ndarray = np.zeros(env.n_states, dtype=np.float64)
        self.policy: np.ndarray = np.zeros(env.n_states, dtype=np.int64)

    def _simulate(self, state: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool]:
        """Simulation déterministe sans modifier l'env réel."""
        saved_state, saved_step = self.env._state, self.env._step_count
        self.env._state = state
        self.env._step_count = 0
        s2, r, terminated, _truncated, _info = self.env.step(action)
        self.env._state, self.env._step_count = saved_state, saved_step
        return s2, r, terminated

    def solve(self, max_iters: int = 1000) -> int:
        """Boucle d'itération de Bellman, renvoie le nombre d'itérations."""
        for it in range(max_iters):
            delta = 0.0
            new_V = self.V.copy()
            for idx in range(self.env.n_states):
                state = self.env.index_to_state(idx)
                if state == self.env.cfg.goal:
                    new_V[idx] = 0.0
                    continue
                best = -np.inf
                best_a = 0
                for a in range(self.env.n_actions):
                    s2, r, terminated = self._simulate(state, a)
                    v2 = 0.0 if terminated else self.V[self.env.state_to_index(s2)]
                    q = r + self.gamma * v2
                    if q > best:
                        best = q
                        best_a = a
                new_V[idx] = best
                self.policy[idx] = best_a
                delta = max(delta, abs(new_V[idx] - self.V[idx]))
            self.V = new_V
            if delta < self.theta:
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
