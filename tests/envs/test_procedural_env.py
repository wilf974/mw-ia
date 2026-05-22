"""Tests de ProceduralGridWorld."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.config import ProceduralEnvConfig
from mw_ia.envs.gridworld import Action
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld


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
