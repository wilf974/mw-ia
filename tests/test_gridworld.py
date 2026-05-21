"""Tests GridWorld."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.config import GridWorldConfig
from mw_ia.envs.gridworld import Action, GridWorld


@pytest.fixture
def env() -> GridWorld:
    cfg = GridWorldConfig(
        rows=5, cols=5, start=(0, 0), goal=(4, 4),
        obstacles=((2, 2),),
        step_penalty=-0.01, goal_reward=1.0, obstacle_penalty=-1.0,
        max_steps=50,
    )
    return GridWorld(cfg)


def test_reset_returns_start_state(env: GridWorld) -> None:
    state, info = env.reset()
    assert state == (0, 0)
    assert info["step"] == 0


def test_action_enum_size() -> None:
    assert len(list(Action)) == 4


def test_step_moves_agent_right(env: GridWorld) -> None:
    env.reset()
    state, reward, terminated, truncated, info = env.step(Action.RIGHT)
    assert state == (0, 1)
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False


def test_step_blocked_by_wall(env: GridWorld) -> None:
    env.reset()
    state, _, _, _, _ = env.step(Action.UP)
    assert state == (0, 0), "ne doit pas sortir de la grille en haut"


def test_step_blocked_by_obstacle(env: GridWorld) -> None:
    env.reset()
    env.step(Action.DOWN)        # (1,0)
    env.step(Action.DOWN)        # (2,0)
    state2, reward2, _, _, _ = env.step(Action.RIGHT)
    state3, reward3, _, _, _ = env.step(Action.RIGHT)
    assert state3 != (2, 2), "ne doit jamais entrer dans une case obstacle"


def test_step_reaches_goal_and_terminates(env: GridWorld) -> None:
    env.reset()
    for a in [Action.DOWN] * 4 + [Action.RIGHT] * 4:
        state, reward, terminated, _, _ = env.step(a)
    assert state == (4, 4)
    assert terminated is True
    assert reward == pytest.approx(1.0 - 0.01)


def test_step_truncates_at_max_steps(env: GridWorld) -> None:
    env.reset()
    last = None
    for _ in range(60):
        last = env.step(Action.LEFT)   # reste sur (0,0)
        if last[3]:
            break
    assert last is not None
    _, _, _, truncated, info = last
    assert truncated is True
    assert info["step"] == env.cfg.max_steps


def test_observation_space_indexable(env: GridWorld) -> None:
    assert env.n_states == 25
    assert env.n_actions == 4
    assert env.state_to_index((1, 2)) == 1 * 5 + 2


def test_render_ascii_shape(env: GridWorld) -> None:
    env.reset()
    rendered = env.render_ascii()
    assert isinstance(rendered, str)
    assert "A" in rendered
    assert "G" in rendered
    assert "#" in rendered
