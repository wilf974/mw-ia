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


def encode_procedural_observation(
    *, state: tuple[int, int], grid: np.ndarray, max_rows: int, max_cols: int
) -> np.ndarray:
    """Encode l'observation procédural pour QNetwork.

    Format : concat(position_one_hot, grid_flatten) → np.float32 de dim 2*max_rows*max_cols.

    Pour les mazes plus petits que max_rows × max_cols (mode maze parfait avec
    difficulté variable), la grille est placée top-left dans une zone paddée
    de zéros (= cellules libres). Conséquence : l'agent voit des bordures
    artificielles qu'il apprend à ignorer.

    Args:
        state: position (row, col) de l'agent.
        grid: maze actuel (rows ≤ max_rows, cols ≤ max_cols), True = obstacle.
        max_rows: nombre de rangées max (dim du QNetwork).
        max_cols: nombre de colonnes max.

    Returns:
        np.ndarray[float32] de shape (2 * max_rows * max_cols,).
    """
    rows, cols = grid.shape
    assert rows <= max_rows and cols <= max_cols, (
        f"grid {grid.shape} > max ({max_rows}, {max_cols})"
    )

    n_cells = max_rows * max_cols
    obs = np.zeros(2 * n_cells, dtype=np.float32)

    # Position one-hot dans la grille max_rows × max_cols
    r, c = state
    obs[r * max_cols + c] = 1.0

    # Grid paddé top-left dans la deuxième moitié
    padded = np.zeros((max_rows, max_cols), dtype=np.float32)
    padded[:rows, :cols] = grid.astype(np.float32)
    obs[n_cells:] = padded.flatten()

    return obs
