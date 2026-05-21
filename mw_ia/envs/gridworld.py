"""GridWorld 2D — environnement RL pédagogique.

API inspirée de Gymnasium : reset() / step(action) renvoient un tuple normalisé.
État = (row, col). Actions = haut/bas/gauche/droite.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

from mw_ia.config import GridWorldConfig


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


class GridWorld:
    """GridWorld déterministe avec obstacles et reward shaping."""

    def __init__(self, cfg: GridWorldConfig | None = None) -> None:
        self.cfg = cfg or GridWorldConfig()
        self._obstacles: frozenset[tuple[int, int]] = frozenset(self.cfg.obstacles)
        assert self.cfg.start not in self._obstacles, "start ne peut pas être un obstacle"
        assert self.cfg.goal not in self._obstacles, "goal ne peut pas être un obstacle"
        self._state: tuple[int, int] = self.cfg.start
        self._step_count: int = 0

    @property
    def n_states(self) -> int:
        return self.cfg.rows * self.cfg.cols

    @property
    def n_actions(self) -> int:
        return len(Action)

    @property
    def state(self) -> tuple[int, int]:
        return self._state

    def state_to_index(self, s: tuple[int, int]) -> int:
        r, c = s
        return r * self.cfg.cols + c

    def index_to_state(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.cfg.cols)

    def reset(self, *, seed: int | None = None) -> tuple[tuple[int, int], dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self._state = self.cfg.start
        self._step_count = 0
        return self._state, {"step": 0}

    def step(
        self, action: Action | int
    ) -> tuple[tuple[int, int], float, bool, bool, dict[str, Any]]:
        action = Action(int(action))
        dr, dc = _DELTAS[action]
        r, c = self._state
        nr, nc = r + dr, c + dc

        reward = self.cfg.step_penalty
        moved_into_obstacle = False

        if not (0 <= nr < self.cfg.rows and 0 <= nc < self.cfg.cols):
            nr, nc = r, c
        elif (nr, nc) in self._obstacles:
            reward += self.cfg.obstacle_penalty
            moved_into_obstacle = True
            nr, nc = r, c

        self._state = (nr, nc)
        self._step_count += 1

        terminated = self._state == self.cfg.goal
        if terminated:
            reward += self.cfg.goal_reward

        truncated = (not terminated) and self._step_count >= self.cfg.max_steps

        info: dict[str, Any] = {
            "step": self._step_count,
            "obstacle_hit": moved_into_obstacle,
        }
        return self._state, reward, terminated, truncated, info

    def render_ascii(self) -> str:
        """Rendu texte pour debug headless."""
        lines: list[str] = []
        for r in range(self.cfg.rows):
            row_chars: list[str] = []
            for c in range(self.cfg.cols):
                if (r, c) == self._state:
                    row_chars.append("A")
                elif (r, c) == self.cfg.goal:
                    row_chars.append("G")
                elif (r, c) in self._obstacles:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)
