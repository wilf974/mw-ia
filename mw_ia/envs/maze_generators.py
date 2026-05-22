"""Générateurs de mazes procéduraux + helper de solvabilité BFS.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-procedural-env-design.md §2.1
"""
from __future__ import annotations

from collections import deque

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
