"""Tests de ProceduralGridWorld."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.config import ProceduralEnvConfig
from mw_ia.envs.gridworld import Action
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld, encode_procedural_observation


def _env() -> ProceduralGridWorld:
    cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                              min_density=0.10, max_density=0.30)
    gen = RandomObstaclesGenerator(rows=10, cols=10, start=(0, 0), goal=(9, 9),
                                   min_density=cfg.min_density, max_density=cfg.max_density)
    return ProceduralGridWorld(cfg=cfg, generator=gen)


def test_reset_returns_state_and_info_with_maze():
    env = _env()
    env.set_difficulty(0.5)
    state, info = env.reset(seed=42)
    assert state == (0, 0)
    assert "maze" in info
    assert info["maze"].shape == (10, 10)
    assert info["difficulty"] == 0.5
    assert info["episode_id"] == 42


def test_reset_with_different_seeds_gives_different_mazes():
    env = _env()
    env.set_difficulty(0.5)
    _, info1 = env.reset(seed=1)
    _, info2 = env.reset(seed=2)
    assert not np.array_equal(info1["maze"], info2["maze"])


def test_reset_with_same_seed_is_deterministic():
    env = _env()
    env.set_difficulty(0.5)
    _, info1 = env.reset(seed=42)
    _, info2 = env.reset(seed=42)
    assert np.array_equal(info1["maze"], info2["maze"])


def test_step_delegates_to_v1_gridworld():
    """Vérifie que step() respecte les rewards V1."""
    env = _env()
    env.set_difficulty(0.0)  # densité min, moins de chance d'obstacle hit
    env.reset(seed=42)
    _, reward, terminated, truncated, info = env.step(Action.RIGHT)
    # Au moins le step_penalty doit être appliqué
    assert reward <= 0.0 or terminated  # négatif sauf si on tombe sur goal directement
    assert "step" in info


def test_changing_difficulty_changes_maze():
    env = _env()
    env.set_difficulty(0.0)
    _, info_easy = env.reset(seed=42)
    env.set_difficulty(1.0)
    _, info_hard = env.reset(seed=42)
    assert not np.array_equal(info_easy["maze"], info_hard["maze"]) \
        or info_easy["maze"].sum() != info_hard["maze"].sum()


def test_encode_dim_matches_2x_max_size():
    """Observation = position_one_hot (R*C dim) + grid_flatten (R*C dim) = 2*R*C."""
    grid = np.zeros((10, 10), dtype=bool)
    obs = encode_procedural_observation(state=(0, 0), grid=grid, max_rows=10, max_cols=10)
    assert obs.shape == (200,)  # 2 * 10 * 10
    assert obs.dtype == np.float32


def test_encode_position_one_hot():
    grid = np.zeros((10, 10), dtype=bool)
    obs = encode_procedural_observation(state=(0, 0), grid=grid, max_rows=10, max_cols=10)
    # Première moitié = position one-hot. State (0,0) = index 0.
    assert obs[0] == 1.0
    assert obs[1:100].sum() == 0.0


def test_encode_grid_in_second_half():
    grid = np.zeros((10, 10), dtype=bool)
    grid[2, 2] = True  # obstacle à (2,2) = index 22
    obs = encode_procedural_observation(state=(0, 0), grid=grid, max_rows=10, max_cols=10)
    # Deuxième moitié contient le grid. obstacle à index 22 dans la deuxième moitié.
    assert obs[100 + 22] == 1.0


def test_encode_padding_when_grid_smaller_than_max():
    """Maze 4x4 dans grille max 10x10 → padding top-left avec zéros."""
    grid = np.ones((4, 4), dtype=bool)
    grid[0, 0] = False  # start libre
    grid[3, 3] = False  # goal libre
    obs = encode_procedural_observation(state=(0, 0), grid=grid, max_rows=10, max_cols=10)
    assert obs.shape == (200,)
    # Le grid 4x4 est placé top-left dans une 10x10. Les obstacles sont aux
    # positions (r,c) avec r,c ∈ [0,3]. La cellule (3,3) (goal) doit être 0,
    # et la cellule (4,4) (hors maze original) doit être 0 aussi.
    assert obs[100 + 33] == 0.0  # goal libre
    assert obs[100 + 44] == 0.0  # padding


def test_encode_grid_too_large_raises():
    grid = np.zeros((11, 11), dtype=bool)
    with pytest.raises(AssertionError):
        encode_procedural_observation(state=(0, 0), grid=grid, max_rows=10, max_cols=10)
