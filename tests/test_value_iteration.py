"""Tests Value Iteration."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.agents.value_iteration import ValueIteration
from mw_ia.config import GridWorldConfig
from mw_ia.envs.gridworld import GridWorld


@pytest.fixture
def env() -> GridWorld:
    return GridWorld(GridWorldConfig(
        rows=4, cols=4, start=(0, 0), goal=(3, 3), obstacles=(),
        step_penalty=-0.01, goal_reward=1.0, obstacle_penalty=-1.0, max_steps=30,
    ))


def test_value_iteration_converges(env: GridWorld) -> None:
    vi = ValueIteration(env, gamma=0.99, theta=1e-6)
    iters = vi.solve(max_iters=500)
    assert iters < 500
    V = vi.V
    assert V[env.state_to_index((3, 3))] == pytest.approx(0.0, abs=1e-6)
    assert V[env.state_to_index((0, 0))] > 0.5


def test_policy_reaches_goal(env: GridWorld) -> None:
    vi = ValueIteration(env, gamma=0.99)
    vi.solve(max_iters=500)
    state, _ = env.reset()
    for _ in range(env.cfg.max_steps):
        action = vi.act(state)
        state, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    assert state == (3, 3), "la politique optimale doit atteindre le goal"
