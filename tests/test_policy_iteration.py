"""Tests Policy Iteration."""
from __future__ import annotations

import pytest

from mw_ia.agents.policy_iteration import PolicyIteration
from mw_ia.config import GridWorldConfig
from mw_ia.envs.gridworld import GridWorld


@pytest.fixture
def env() -> GridWorld:
    return GridWorld(GridWorldConfig(
        rows=4, cols=4, start=(0, 0), goal=(3, 3), obstacles=(),
        step_penalty=-0.01, goal_reward=1.0, obstacle_penalty=-1.0, max_steps=30,
    ))


def test_policy_iteration_converges(env: GridWorld) -> None:
    pi = PolicyIteration(env, gamma=0.99, theta=1e-6)
    iters = pi.solve(max_iters=50)
    assert iters < 50


def test_policy_iteration_reaches_goal(env: GridWorld) -> None:
    pi = PolicyIteration(env, gamma=0.99)
    pi.solve(max_iters=50)
    state, _ = env.reset()
    for _ in range(env.cfg.max_steps):
        a = pi.act(state)
        state, _, terminated, _, _ = env.step(a)
        if terminated:
            break
    assert state == (3, 3)
