"""Tests de l'encodeur d'observation 4-canaux oracle (V2-BX Sonde C)."""
from __future__ import annotations

import numpy as np

from mw_ia.envs.procedural_env import encode_procedural_observation_2d


def _empty_grid(n: int = 4) -> np.ndarray:
    return np.zeros((n, n), dtype=bool)


def test_none_mode_is_three_channels():
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=_empty_grid(), goal=(3, 3),
        max_rows=4, max_cols=4, oracle_mode="none",
    )
    assert obs.shape == (3, 4, 4)


def test_scalar_mode_adds_uniform_fourth_channel():
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=_empty_grid(), goal=(3, 3),
        max_rows=4, max_cols=4, oracle_mode="scalar",
    )
    assert obs.shape == (4, 4, 4)
    chan = obs[3]
    # Plan uniforme : toutes les cellules egales.
    assert np.allclose(chan, chan.flat[0])
    # Valeur = BFS(agent->goal)/ (rows*cols) = 6 / 16 sur grille vide 4x4.
    assert np.isclose(chan.flat[0], 6.0 / 16.0)


def test_field_mode_per_cell_distances():
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=_empty_grid(), goal=(0, 0),
        max_rows=4, max_cols=4, oracle_mode="field",
    )
    assert obs.shape == (4, 4, 4)
    chan = obs[3]
    # goal en (0,0) -> distance 0 ; (3,3) -> 6/16.
    assert np.isclose(chan[0, 0], 0.0)
    assert np.isclose(chan[3, 3], 6.0 / 16.0)


def test_field_mode_obstacle_sentinel_is_one():
    grid = _empty_grid()
    grid[1, 1] = True
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=grid, goal=(3, 3),
        max_rows=4, max_cols=4, oracle_mode="field",
    )
    # Cellule-obstacle = sentinelle 1.0 (plus loin que tout).
    assert np.isclose(obs[3, 1, 1], 1.0)


def test_first_three_channels_unchanged_by_oracle():
    base = encode_procedural_observation_2d(
        state=(1, 2), grid=_empty_grid(), goal=(3, 3),
        max_rows=4, max_cols=4, oracle_mode="none",
    )
    withc = encode_procedural_observation_2d(
        state=(1, 2), grid=_empty_grid(), goal=(3, 3),
        max_rows=4, max_cols=4, oracle_mode="scalar",
    )
    assert np.array_equal(base, withc[:3])


def test_invalid_oracle_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="oracle_mode"):
        encode_procedural_observation_2d(
            state=(0, 0), grid=_empty_grid(), goal=(3, 3),
            max_rows=4, max_cols=4, oracle_mode="bogus",
        )
