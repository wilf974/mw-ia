"""Tests Q-Learning tabulaire."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.agents.q_learning import QLearningAgent, Transition
from mw_ia.config import GridWorldConfig, QLearningConfig
from mw_ia.envs.gridworld import GridWorld


@pytest.fixture
def env() -> GridWorld:
    return GridWorld(GridWorldConfig(
        rows=5, cols=5, start=(0, 0), goal=(4, 4), obstacles=(),
        step_penalty=-0.01, goal_reward=1.0, obstacle_penalty=-1.0, max_steps=100,
    ))


def test_qtable_initial_zero(env: GridWorld) -> None:
    cfg = QLearningConfig(alpha=0.1, gamma=0.99, epsilon_start=0.0, epsilon_end=0.0,
                          epsilon_decay_episodes=1, episodes=1)
    agent = QLearningAgent(env, cfg, seed=0)
    assert agent.Q.shape == (env.n_states, env.n_actions)
    assert np.allclose(agent.Q, 0.0)


def test_act_greedy_when_epsilon_zero(env: GridWorld) -> None:
    cfg = QLearningConfig(alpha=0.1, gamma=0.99, epsilon_start=0.0, epsilon_end=0.0,
                          epsilon_decay_episodes=1, episodes=1)
    agent = QLearningAgent(env, cfg, seed=0)
    agent.Q[env.state_to_index((0, 0)), 1] = 5.0
    assert agent.act((0, 0), greedy=True) == 1
    assert agent.act((0, 0)) == 1


def test_learn_updates_q_value(env: GridWorld) -> None:
    cfg = QLearningConfig(alpha=0.5, gamma=0.9, epsilon_start=0.0, epsilon_end=0.0,
                          epsilon_decay_episodes=1, episodes=1)
    agent = QLearningAgent(env, cfg, seed=0)
    t = Transition(state=(0, 0), action=3, reward=-0.01, next_state=(0, 1),
                   terminated=False, truncated=False)
    metrics = agent.learn(t)
    expected = 0.5 * (-0.01 + 0.9 * 0.0 - 0.0)
    assert agent.Q[env.state_to_index((0, 0)), 3] == pytest.approx(expected)
    assert "td_error" in metrics


def test_convergence_on_small_gridworld(env: GridWorld) -> None:
    cfg = QLearningConfig(alpha=0.5, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                          epsilon_decay_episodes=500, episodes=1000)
    agent = QLearningAgent(env, cfg, seed=42)
    last_100: list[bool] = []
    for ep in range(cfg.episodes):
        state, _ = env.reset()
        terminated = truncated = False
        agent.start_episode(ep)
        while not (terminated or truncated):
            a = agent.act(state)
            s2, r, terminated, truncated, _ = env.step(a)
            agent.learn(Transition(state, a, r, s2, terminated, truncated))
            state = s2
        last_100.append(terminated)
        if len(last_100) > 100:
            last_100.pop(0)
    winrate = sum(last_100) / len(last_100)
    assert winrate > 0.9, f"winrate final attendu >90 %, vu {winrate:.2f}"


def test_save_load_roundtrip(env: GridWorld, tmp_path) -> None:
    cfg = QLearningConfig()
    agent = QLearningAgent(env, cfg, seed=0)
    agent.Q[3, 2] = 1.234
    path = tmp_path / "qtable"
    agent.save(path)
    agent2 = QLearningAgent(env, cfg, seed=1)
    agent2.load(path)
    assert agent2.Q[3, 2] == pytest.approx(1.234)
