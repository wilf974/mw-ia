"""Tests V2-Z de encode_procedural_observation_2d (3 canaux pour CNN)."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.envs.procedural_env import encode_procedural_observation_2d


def test_encode_shape_default():
    """Encoding standard 10x10 → shape (3, 10, 10) float32."""
    grid = np.zeros((10, 10), dtype=bool)
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=grid, goal=(9, 9), max_rows=10, max_cols=10,
    )
    assert obs.shape == (3, 10, 10)
    assert obs.dtype == np.float32


def test_encode_agent_channel():
    """Canal 0 : un seul 1 en (row, col), zéros partout ailleurs."""
    grid = np.zeros((10, 10), dtype=bool)
    obs = encode_procedural_observation_2d(
        state=(3, 5), grid=grid, goal=(9, 9), max_rows=10, max_cols=10,
    )
    assert obs[0, 3, 5] == 1.0
    assert obs[0].sum() == 1.0


def test_encode_obstacles_channel():
    """Canal 1 : matches grid.astype(float32)."""
    grid = np.zeros((10, 10), dtype=bool)
    grid[2, 2] = True
    grid[5, 7] = True
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=grid, goal=(9, 9), max_rows=10, max_cols=10,
    )
    assert obs[1, 2, 2] == 1.0
    assert obs[1, 5, 7] == 1.0
    assert obs[1].sum() == 2.0


def test_encode_goal_channel():
    """Canal 2 : un seul 1 en (goal_r, goal_c)."""
    grid = np.zeros((10, 10), dtype=bool)
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=grid, goal=(7, 4), max_rows=10, max_cols=10,
    )
    assert obs[2, 7, 4] == 1.0
    assert obs[2].sum() == 1.0


def test_encode_padding_smaller_maze():
    """Maze 6x6 dans max 10x10 → padding zéros top-left sur les 3 canaux."""
    grid = np.ones((6, 6), dtype=bool)
    grid[0, 0] = False  # start libre
    grid[5, 5] = False  # goal libre
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=grid, goal=(5, 5), max_rows=10, max_cols=10,
    )
    assert obs.shape == (3, 10, 10)
    # Zone hors maze : padding zéros sur tous les canaux
    assert obs[1, 6:, :].sum() == 0.0  # pas d'obstacles dans le padding
    assert obs[1, :, 6:].sum() == 0.0
    assert obs[2, 6:, :].sum() == 0.0  # pas de goal dans le padding
    # Goal correctement placé
    assert obs[2, 5, 5] == 1.0


def test_encode_asserts_invalid_inputs():
    """state hors grille, grid > max, goal hors max → AssertionError."""
    grid = np.zeros((10, 10), dtype=bool)
    # grid trop grand
    too_big = np.zeros((11, 11), dtype=bool)
    with pytest.raises(AssertionError):
        encode_procedural_observation_2d(
            state=(0, 0), grid=too_big, goal=(9, 9), max_rows=10, max_cols=10,
        )
    # goal hors max
    with pytest.raises(AssertionError):
        encode_procedural_observation_2d(
            state=(0, 0), grid=grid, goal=(10, 5), max_rows=10, max_cols=10,
        )
    # state hors grille réelle
    with pytest.raises(AssertionError):
        encode_procedural_observation_2d(
            state=(10, 0), grid=grid, goal=(9, 9), max_rows=10, max_cols=10,
        )
