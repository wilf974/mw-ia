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
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(
                f"rows et cols doivent être > 0, reçu rows={self.rows}, cols={self.cols}"
            )
        if not (0 <= self.start[0] < self.rows and 0 <= self.start[1] < self.cols):
            raise ValueError(f"start {self.start} hors grille {self.rows}x{self.cols}")
        if not (0 <= self.goal[0] < self.rows and 0 <= self.goal[1] < self.cols):
            raise ValueError(f"goal {self.goal} hors grille {self.rows}x{self.cols}")
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


@dataclass(frozen=True)
class PerfectMazeGenerator:
    """Generator de maze quasi-parfait via DFS recursive backtracker.

    Un vrai maze parfait a un et un seul chemin entre deux cellules. Le DFS
    classique creuse uniquement les cellules aux coordonnées paires ; pour
    des tailles paires (4, 6, ..., 20), goal=(size-1, size-1) est en position
    impaire/impaire et n'est pas dans la sous-grille DFS. On force son
    accessibilité en ouvrant explicitement les murs intermédiaires vers le
    voisin pair-pair le plus proche.

    Conséquence : pour size paire, goal est accessible par 1 ou 2 chemins
    additionnels sur ses dernières cellules (quasi-parfait localement). La
    solvabilité reste garantie par construction (pas besoin de BFS-check).

    La difficulté ∈ [0,1] interpole la TAILLE entre min_size et max_size.
    """

    min_size: int = 4
    max_size: int = 20

    def __post_init__(self) -> None:
        if self.min_size < 2:
            raise ValueError(f"min_size doit être >= 2, reçu {self.min_size}")
        if self.min_size >= self.max_size:
            raise ValueError(
                f"min_size ({self.min_size}) doit être < max_size ({self.max_size})"
            )

    def generate(self, *, seed: int, difficulty: float) -> np.ndarray:
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        size = self.min_size + int(round((self.max_size - self.min_size) * difficulty))

        # Initialement, toutes les cellules sont des obstacles. Le DFS creuse
        # des couloirs en marquant False (cellule libre).
        grid = np.ones((size, size), dtype=bool)
        rng = np.random.default_rng(seed=seed)

        # Carve depuis (0,0). Pour garantir start (0,0) et goal (size-1, size-1)
        # libres, le DFS classique sur maze sur grid creuse en partant de start.
        # On ouvre start et goal explicitement à la fin pour cohérence.
        stack: list[tuple[int, int]] = [(0, 0)]
        grid[0, 0] = False
        while stack:
            r, c = stack[-1]
            # Voisins non visités à distance 2 (cellule + mur)
            neighbors: list[tuple[int, int, int, int]] = []
            for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and grid[nr, nc]:
                    # mur entre (r,c) et (nr,nc) : (r+dr/2, c+dc/2)
                    wr, wc = r + dr // 2, c + dc // 2
                    neighbors.append((nr, nc, wr, wc))
            if not neighbors:
                stack.pop()
                continue
            pick = neighbors[rng.integers(0, len(neighbors))]
            nr, nc, wr, wc = pick
            grid[wr, wc] = False  # casser le mur
            grid[nr, nc] = False  # ouvrir la cellule
            stack.append((nr, nc))

        # Garantir goal libre ET accessible.
        # Le DFS creuse uniquement les cellules aux positions paires (coordonnées divisibles
        # par 2). Pour size paire, goal = (size-1, size-1) est impair/impair et reste fermé
        # après le DFS. On ouvre goal et on casse un mur vers son voisin pair-pair le plus
        # proche : (size-2, size-2) → (size-1, size-2) → (size-1, size-1)
        # ou (size-2, size-2) → (size-2, size-1) → (size-1, size-1).
        gr, gc = size - 1, size - 1
        grid[gr, gc] = False  # ouvrir goal
        # Connecter goal à (gr-1, gc) si (gr-1, gc) est libre, sinon à (gr, gc-1)
        # puis via le carrefour commun (gr-1, gc-1) → (gr-1, gc) ou (gr, gc-1)
        # Stratégie simple : ouvrir le mur intermédiaire vers la cellule paire-paire voisine
        cell_r = (gr // 2) * 2  # cellule logique en ligne : gr ou gr-1 selon parité
        cell_c = (gc // 2) * 2  # cellule logique en colonne
        # Ouvrir le corridor de (cell_r, cell_c) à (gr, gc) si pas déjà accessible
        if grid[cell_r, gc]:   # mur horizontal sur la même colonne que goal
            grid[cell_r, gc] = False
        if grid[gr, cell_c]:   # mur vertical sur la même ligne que goal
            grid[gr, cell_c] = False
        return grid
