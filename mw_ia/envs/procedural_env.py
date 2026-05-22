"""Wrapper procédural sur GridWorld V1 qui régénère le maze à chaque reset."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from mw_ia.config import GridWorldConfig, ProceduralEnvConfig
from mw_ia.envs.gridworld import Action, GridWorld


class MazeGenerator(Protocol):
    """Interface commune des générateurs de mazes."""

    def generate(self, *, seed: int, difficulty: float) -> np.ndarray:
        ...


class ProceduralGridWorld:
    """Régénère un nouveau maze à chaque reset() via un MazeGenerator.

    Délègue step() à un GridWorld V1 interne reconstruit à chaque reset.
    """

    def __init__(
        self,
        *,
        cfg: ProceduralEnvConfig,
        generator: MazeGenerator,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
    ) -> None:
        self.cfg = cfg
        self.generator = generator
        self.start = start
        # Goal par défaut : coin opposé. Si maze de taille < max_*, sera ajusté
        # à goal réel au reset() (mais GridWorld interne aura ses propres coords).
        self.goal = goal if goal is not None else (cfg.max_rows - 1, cfg.max_cols - 1)
        self._difficulty: float = 0.0
        self._inner: GridWorld | None = None

    def set_difficulty(self, difficulty: float) -> None:
        self._difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def reset(self, *, seed: int) -> tuple[tuple[int, int], dict[str, Any]]:
        maze = self.generator.generate(seed=seed, difficulty=self._difficulty)
        rows, cols = maze.shape
        goal = (rows - 1, cols - 1)
        obstacles = tuple(
            (int(r), int(c))
            for r, c in zip(*np.where(maze))
        )
        gw_cfg = GridWorldConfig(
            rows=rows, cols=cols,
            start=self.start, goal=goal,
            obstacles=obstacles,
        )
        self._inner = GridWorld(gw_cfg)
        state, _ = self._inner.reset()
        info = {
            "maze": maze,
            "difficulty": self._difficulty,
            "episode_id": seed,
            "step": 0,
        }
        return state, info

    def step(
        self, action: Action | int
    ) -> tuple[tuple[int, int], float, bool, bool, dict[str, Any]]:
        assert self._inner is not None, "reset() doit être appelé avant step()"
        return self._inner.step(action)

    @property
    def inner(self) -> GridWorld:
        assert self._inner is not None, "reset() doit être appelé avant inner"
        return self._inner
