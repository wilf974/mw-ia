"""Smoke test DQN (chaîne complète sans crash + epsilon decay + sync target)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mw_ia.agents.dqn import DQNAgent
from mw_ia.config import DQNConfig, GridWorldConfig
from mw_ia.envs.gridworld import GridWorld


@pytest.fixture
def env() -> GridWorld:
    return GridWorld(GridWorldConfig(rows=5, cols=5, start=(0, 0), goal=(4, 4),
                                     obstacles=(), max_steps=50))


def _state_vec(env: GridWorld, s: tuple[int, int]) -> np.ndarray:
    v = np.zeros(env.n_states, dtype=np.float32)
    v[env.state_to_index(s)] = 1.0
    return v


def test_dqn_smoke_100_steps(env: GridWorld) -> None:
    cfg = DQNConfig(hidden_layers=(64,), batch_size=16, replay_capacity=500,
                    min_replay_to_learn=20, target_sync_steps=50,
                    epsilon_decay_steps=200, use_amp=False)
    agent = DQNAgent(env, cfg, device="cpu", seed=0)
    state, _ = env.reset()
    for _ in range(100):
        a = agent.act(_state_vec(env, state))
        s2, r, terminated, truncated, _ = env.step(a)
        agent.observe(_state_vec(env, state), a, r, _state_vec(env, s2),
                      terminated or truncated)
        state = s2
        if terminated or truncated:
            state, _ = env.reset()
    assert agent.epsilon < cfg.epsilon_start


def test_dqn_target_sync_triggers(env: GridWorld) -> None:
    cfg = DQNConfig(hidden_layers=(8,), batch_size=8, replay_capacity=100,
                    min_replay_to_learn=8, target_sync_steps=10,
                    epsilon_decay_steps=50, use_amp=False, train_every=1)
    agent = DQNAgent(env, cfg, device="cpu", seed=0)
    state, _ = env.reset()
    for _ in range(50):
        a = agent.act(_state_vec(env, state))
        s2, r, term, trunc, _ = env.step(a)
        agent.observe(_state_vec(env, state), a, r, _state_vec(env, s2), term or trunc)
        state = s2
        if term or trunc:
            state, _ = env.reset()
    assert agent.global_step >= 50
    assert agent.target_syncs > 0


def test_dqn_save_load_roundtrip(env: GridWorld, tmp_path) -> None:
    cfg = DQNConfig(hidden_layers=(16,), batch_size=4, replay_capacity=50,
                    min_replay_to_learn=4, use_amp=False)
    agent = DQNAgent(env, cfg, device="cpu", seed=0)
    state, _ = env.reset()
    for _ in range(20):
        a = agent.act(_state_vec(env, state))
        s2, r, term, trunc, _ = env.step(a)
        agent.observe(_state_vec(env, state), a, r, _state_vec(env, s2), term or trunc)
        state = s2
        if term or trunc:
            state, _ = env.reset()
    path = tmp_path / "dqn.pt"
    agent.save(path)
    agent2 = DQNAgent(env, cfg, device="cpu", seed=1)
    agent2.load(path)
    for p1, p2 in zip(agent.online.parameters(), agent2.online.parameters()):
        assert torch.equal(p1.cpu(), p2.cpu())
