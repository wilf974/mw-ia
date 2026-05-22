"""Tests de maze_generators."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mw_ia.envs.maze_generators import (
    PerfectMazeGenerator,
    RandomObstaclesGenerator,
    maze_bfs_check,
)


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


def _gen() -> RandomObstaclesGenerator:
    # max_density=0.40 : à density=0.40 sur 10x10, ~10.8% de mazes solvables par tirage
    # → avec max_attempts=100, probabilité de succès ≈ 99.9% (fiable pour les tests).
    # max_density=0.50 produirait ~0.6% de succès → RuntimeError fréquent sur seed fixe.
    return RandomObstaclesGenerator(
        rows=10, cols=10, start=(0, 0), goal=(9, 9),
        min_density=0.10, max_density=0.40, max_attempts=100,
    )


def test_random_obstacles_density_zero():
    gen = _gen()
    grid = gen.generate(seed=42, difficulty=0.0)
    # density interpolated to min_density (par défaut 0.0 → 0 obstacle, fixture _gen aussi à 0.10)
    # On vérifie qu'on n'est pas au-delà de la borne attendue.
    n_obstacles = int(grid.sum())
    assert 0 <= n_obstacles <= 15


def test_random_obstacles_density_max():
    gen = _gen()
    grid = gen.generate(seed=42, difficulty=1.0)
    # difficulty=1.0 → max_density=0.40 → ~40 obstacles
    n_obstacles = int(grid.sum())
    assert 30 <= n_obstacles <= 42


def test_random_obstacles_seed_deterministic():
    gen = _gen()
    g1 = gen.generate(seed=42, difficulty=0.5)
    g2 = gen.generate(seed=42, difficulty=0.5)
    assert np.array_equal(g1, g2)


def test_random_obstacles_start_goal_never_obstacle():
    gen = _gen()
    for seed in range(50):
        grid = gen.generate(seed=seed, difficulty=0.5)
        assert grid[0, 0] == False, f"seed={seed}: start sur obstacle"
        assert grid[9, 9] == False, f"seed={seed}: goal sur obstacle"


def test_random_obstacles_always_solvable():
    gen = _gen()
    for seed in range(50):
        grid = gen.generate(seed=seed, difficulty=0.5)
        assert maze_bfs_check(grid, start=(0, 0), goal=(9, 9)), \
            f"seed={seed}: maze non solvable"


def test_random_obstacles_pathological_density_raises():
    # max_density=0.95 → quasi-aucun chemin possible avec start=(0,0), goal=(9,9)
    gen = RandomObstaclesGenerator(
        rows=10, cols=10, start=(0, 0), goal=(9, 9),
        min_density=0.95, max_density=0.95, max_attempts=10,
    )
    with pytest.raises(RuntimeError, match="unreachable after"):
        gen.generate(seed=42, difficulty=0.5)


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=50, deadline=None)
def test_random_obstacles_property_solvability(seed: int):
    """Property : tout maze généré est solvable."""
    gen = _gen()
    grid = gen.generate(seed=seed, difficulty=0.5)
    assert maze_bfs_check(grid, start=(0, 0), goal=(9, 9))


def test_perfect_maze_size_4_solvable():
    gen = PerfectMazeGenerator(min_size=4, max_size=20)
    grid = gen.generate(seed=42, difficulty=0.0)
    rows, cols = grid.shape
    assert rows == cols == 4
    assert maze_bfs_check(grid, start=(0, 0), goal=(rows - 1, cols - 1))


def test_perfect_maze_size_20_solvable():
    gen = PerfectMazeGenerator(min_size=4, max_size=20)
    grid = gen.generate(seed=42, difficulty=1.0)
    rows, cols = grid.shape
    assert rows == cols == 20
    assert maze_bfs_check(grid, start=(0, 0), goal=(rows - 1, cols - 1))


def test_perfect_maze_seed_deterministic():
    gen = PerfectMazeGenerator(min_size=4, max_size=20)
    g1 = gen.generate(seed=42, difficulty=0.5)
    g2 = gen.generate(seed=42, difficulty=0.5)
    assert np.array_equal(g1, g2)


def test_perfect_maze_invalid_size_raises():
    with pytest.raises(ValueError):
        PerfectMazeGenerator(min_size=20, max_size=4)  # min > max
    with pytest.raises(ValueError):
        PerfectMazeGenerator(min_size=1, max_size=10)  # min < 2


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=30, deadline=None)
def test_perfect_maze_property_solvability(seed: int):
    """Property : tout maze parfait est solvable par construction."""
    gen = PerfectMazeGenerator(min_size=4, max_size=20)
    grid = gen.generate(seed=seed, difficulty=0.5)
    rows, cols = grid.shape
    assert maze_bfs_check(grid, start=(0, 0), goal=(rows - 1, cols - 1))
