"""Wrapper procédural sur GridWorld V1 qui régénère le maze à chaque reset."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from mw_ia.config import GridWorldConfig, ProceduralEnvConfig
from mw_ia.envs.gridworld import Action, GridWorld
from mw_ia.envs.maze_generators import bfs_distance_field


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
            max_steps=self.cfg.max_steps,
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


def encode_procedural_observation_2d(
    *,
    state: tuple[int, int],
    grid: np.ndarray,
    goal: tuple[int, int],
    max_rows: int,
    max_cols: int,
    oracle_mode: str = "none",
) -> np.ndarray:
    """Encode l'observation procédural pour ConvQNetwork (V2-Z).

    Format : tensor 3D shape (3 ou 4, max_rows, max_cols) float32 :
    - canal 0 : position agent one-hot (un seul 1 en (row, col))
    - canal 1 : obstacles (grid.astype(float32))
    - canal 2 : goal one-hot (un seul 1 en (goal_r, goal_c))
    - canal 3 (optionnel) : distance BFS normalisée ou scalar si oracle_mode != "none"

    Pour les mazes plus petits que max_rows × max_cols, la grille est placée
    top-left, les cellules hors maze restent à zéro sur les canaux 0-2 (cellules
    libres, pas d'obstacle, pas de goal). L'agent CNN voit des bordures
    artificielles qu'il apprend à ignorer.

    Args:
        state: position (row, col) de l'agent.
        grid: maze actuel (rows ≤ max_rows, cols ≤ max_cols), True = obstacle.
        goal: position (goal_r, goal_c) du goal dans max_rows × max_cols.
        max_rows: nombre de rangées max (dim du ConvQNetwork).
        max_cols: nombre de colonnes max.
        oracle_mode: "none" (défaut, 3 canaux), "scalar" (4ᵉ canal = distance agent→goal),
                     ou "field" (4ᵉ canal = champ de distance BFS normalisé).

    Returns:
        np.ndarray[float32] de shape (3 ou 4, max_rows, max_cols).
        Shape est (3, ...) si oracle_mode="none", (4, ...) sinon.
    """
    rows, cols = grid.shape
    assert rows <= max_rows and cols <= max_cols, (
        f"grid {grid.shape} > max ({max_rows}, {max_cols})"
    )
    assert 0 <= state[0] < rows and 0 <= state[1] < cols, (
        f"state {state} hors grid {grid.shape}"
    )
    assert 0 <= goal[0] < max_rows and 0 <= goal[1] < max_cols, (
        f"goal {goal} hors max ({max_rows}, {max_cols})"
    )

    obs = np.zeros((3, max_rows, max_cols), dtype=np.float32)
    obs[0, state[0], state[1]] = 1.0
    obs[1, :rows, :cols] = grid.astype(np.float32)
    obs[2, goal[0], goal[1]] = 1.0

    if oracle_mode == "none":
        return obs
    if oracle_mode not in ("scalar", "field"):
        raise ValueError(
            f"oracle_mode doit etre none|scalar|field, recu {oracle_mode}"
        )

    dist_norm = float(max_rows * max_cols)
    dist = bfs_distance_field(grid, goal=goal)  # (rows, cols), inf hors-atteignable
    # Normalisation + sentinelle 1.0 pour obstacle / non-atteignable.
    norm_field = np.where(np.isfinite(dist), dist / dist_norm, 1.0)
    norm_field = np.clip(norm_field, 0.0, 1.0).astype(np.float32)

    oracle_chan = np.ones((max_rows, max_cols), dtype=np.float32)  # padding = 1.0
    if oracle_mode == "field":
        oracle_chan[:rows, :cols] = norm_field
    else:  # scalar : plan uniforme = distance de l'agent au goal
        oracle_chan[:] = norm_field[state[0], state[1]]

    return np.concatenate([obs, oracle_chan[np.newaxis, :, :]], axis=0)
