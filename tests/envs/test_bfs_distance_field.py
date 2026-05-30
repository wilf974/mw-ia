"""Tests du champ de distance BFS au goal (V2-BX Sonde C)."""
from __future__ import annotations

import numpy as np

from mw_ia.envs.maze_generators import bfs_distance_field


def test_goal_cell_is_zero():
    grid = np.zeros((3, 3), dtype=bool)
    dist = bfs_distance_field(grid, goal=(2, 2))
    assert dist[2, 2] == 0.0


def test_open_grid_manhattan_distances():
    # Grille vide 3x3, goal en (0,0) -> distance = Manhattan (pas d'obstacle).
    grid = np.zeros((3, 3), dtype=bool)
    dist = bfs_distance_field(grid, goal=(0, 0))
    assert dist[0, 1] == 1.0
    assert dist[1, 0] == 1.0
    assert dist[1, 1] == 2.0
    assert dist[2, 2] == 4.0


def test_obstacle_cells_are_inf():
    grid = np.zeros((3, 3), dtype=bool)
    grid[1, 1] = True  # obstacle au centre
    dist = bfs_distance_field(grid, goal=(0, 0))
    assert np.isinf(dist[1, 1])


def test_wall_detour_increases_distance():
    # Mur vertical en colonne 1 sur lignes 0 et 1 force un detour par le bas.
    grid = np.zeros((3, 3), dtype=bool)
    grid[0, 1] = True
    grid[1, 1] = True
    dist = bfs_distance_field(grid, goal=(0, 0))
    # (0,2) atteignable seulement via (2,1) : 0,0->1,0->2,0->2,1->2,2->1,2->0,2 = 6
    assert dist[0, 2] == 6.0


def test_unreachable_region_is_inf():
    # Cellule (0,2) emmuree : obstacles en (0,1) et (1,2).
    grid = np.zeros((3, 3), dtype=bool)
    grid[0, 1] = True
    grid[1, 2] = True
    dist = bfs_distance_field(grid, goal=(0, 0))
    assert np.isinf(dist[0, 2])


def test_returns_float_array_of_grid_shape():
    grid = np.zeros((4, 5), dtype=bool)
    dist = bfs_distance_field(grid, goal=(3, 4))
    assert dist.shape == (4, 5)
    assert dist.dtype == np.float64
