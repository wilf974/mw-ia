"""Tests de maze_generators."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.envs.maze_generators import maze_bfs_check


def test_bfs_trivial_empty_grid():
    grid = np.zeros((5, 5), dtype=bool)
    assert maze_bfs_check(grid, start=(0, 0), goal=(4, 4)) is True


def test_bfs_with_obstacles_solvable():
    grid = np.zeros((5, 5), dtype=bool)
    grid[1, 1] = True  # obstacle isolé
    grid[3, 3] = True
    assert maze_bfs_check(grid, start=(0, 0), goal=(4, 4)) is True


def test_bfs_blocked_full_wall():
    grid = np.zeros((5, 5), dtype=bool)
    grid[2, :] = True  # mur horizontal complet à la rangée 2
    assert maze_bfs_check(grid, start=(0, 0), goal=(4, 4)) is False


def test_bfs_start_equals_goal():
    grid = np.zeros((5, 5), dtype=bool)
    assert maze_bfs_check(grid, start=(2, 2), goal=(2, 2)) is True


def test_bfs_start_on_obstacle_raises():
    grid = np.zeros((5, 5), dtype=bool)
    grid[0, 0] = True
    with pytest.raises(AssertionError):
        maze_bfs_check(grid, start=(0, 0), goal=(4, 4))


def test_bfs_goal_on_obstacle_raises():
    grid = np.zeros((5, 5), dtype=bool)
    grid[4, 4] = True
    with pytest.raises(AssertionError):
        maze_bfs_check(grid, start=(0, 0), goal=(4, 4))


def test_bfs_start_out_of_grid_raises():
    grid = np.zeros((5, 5), dtype=bool)
    with pytest.raises(AssertionError):
        maze_bfs_check(grid, start=(-1, 0), goal=(4, 4))
