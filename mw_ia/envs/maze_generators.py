"""Générateurs de mazes procéduraux + helper de solvabilité BFS.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-procedural-env-design.md §2.1
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def maze_bfs_check(
    grid: np.ndarray, *, start: tuple[int, int], goal: tuple[int, int]
) -> bool:
    """Retourne True ssi un chemin existe entre start et goal en évitant les obstacles.

    Args:
        grid: tableau (rows, cols) de booléens. True = obstacle.
        start: (row, col) de la position de départ.
        goal: (row, col) de la position d'arrivée.

    Returns:
        True si un chemin 4-connexe existe ; False sinon.

    Raises:
        AssertionError: si start ou goal est hors grille ou sur un obstacle.
    """
    rows, cols = grid.shape
    sr, sc = start
    gr, gc = goal
    assert 0 <= sr < rows and 0 <= sc < cols, f"start {start} hors grille {grid.shape}"
    assert 0 <= gr < rows and 0 <= gc < cols, f"goal {goal} hors grille {grid.shape}"
    assert not grid[sr, sc], f"start {start} sur obstacle"
    assert not grid[gr, gc], f"goal {goal} sur obstacle"

    if start == goal:
        return True

    visited = np.zeros_like(grid, dtype=bool)
    visited[sr, sc] = True
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if visited[nr, nc] or grid[nr, nc]:
                continue
            if (nr, nc) == goal:
                return True
            visited[nr, nc] = True
            queue.append((nr, nc))
    return False


@dataclass(frozen=True)
class RandomObstaclesGenerator:
    """Generator par placement aléatoire d'obstacles avec BFS-check.

    La difficulté ∈ [0,1] interpole linéairement entre min_density et max_density.

    Pattern de génération (contrainte non-négociable, cf. spec) :
        1. tire density*rows*cols obstacles aléatoires
        2. exclut start et goal du tirage
        3. maze_bfs_check → True : retourner, sinon retry
        4. après max_attempts tentatives : RuntimeError
    """

    rows: int
    cols: int
    start: tuple[int, int]
    goal: tuple[int, int]
    min_density: float = 0.10
    max_density: float = 0.50
    max_attempts: int = 100

    def __post_init__(self) -> None:
        assert self.rows > 0 and self.cols > 0
        assert 0 <= self.start[0] < self.rows and 0 <= self.start[1] < self.cols
        assert 0 <= self.goal[0] < self.rows and 0 <= self.goal[1] < self.cols
        if not (0.0 <= self.min_density <= self.max_density <= 1.0):
            raise ValueError(
                f"densités invalides : min={self.min_density}, max={self.max_density}"
            )
        if self.max_attempts <= 0:
            raise ValueError(f"max_attempts doit être > 0, reçu {self.max_attempts}")

    def generate(self, *, seed: int, difficulty: float) -> np.ndarray:
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        density = self.min_density + (self.max_density - self.min_density) * difficulty
        n_cells = self.rows * self.cols
        n_obstacles = int(round(density * n_cells))

        rng = np.random.default_rng(seed=seed)
        sr, sc = self.start
        gr, gc = self.goal
        # Cellules candidates : toutes sauf start et goal
        all_indices = np.arange(n_cells)
        forbidden = {sr * self.cols + sc, gr * self.cols + gc}
        candidates = np.array([i for i in all_indices if i not in forbidden])

        for attempt in range(self.max_attempts):
            picks = rng.choice(candidates, size=min(n_obstacles, len(candidates)), replace=False)
            grid = np.zeros((self.rows, self.cols), dtype=bool)
            grid.flat[picks] = True
            if maze_bfs_check(grid, start=self.start, goal=self.goal):
                return grid

        raise RuntimeError(
            f"density={density:.2f} unreachable after {self.max_attempts} attempts "
            f"(rows={self.rows}, cols={self.cols}, start={self.start}, goal={self.goal})"
        )
