"""Tests de RecurrentDQNAgent."""
from __future__ import annotations

import numpy as np
import pytest
pytest.importorskip("torch", reason="suite complete : requiert torch")

import torch

from mw_ia.agents.recurrent_dqn import RecurrentDQNAgent
from mw_ia.config import DRQNConfig


def _agent(seed: int = 0) -> RecurrentDQNAgent:
    cfg = DRQNConfig(episodes=20, replay_capacity=20, min_episodes_to_learn=5,
                     batch_size=4, sequence_length=8, use_amp=False,
                     epsilon_decay_steps=1000, target_sync_steps=100,
                     max_steps_per_episode=50)
    return RecurrentDQNAgent(obs_dim=10, n_actions=4, cfg=cfg, device="cpu", seed=seed)


def test_recurrent_agent_init():
    agent = _agent()
    assert agent.global_step == 0
    assert agent._hidden_state is None
    assert agent._episode_trajectory == []


def test_act_returns_valid_action():
    agent = _agent()
    obs = np.zeros(10, dtype=np.float32)
    action = agent.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < 4


def test_act_greedy_deterministic_with_reset_hidden():
    """Greedy + même obs + hidden reset entre 2 calls → même action."""
    agent = _agent(seed=42)
    obs = np.random.default_rng(0).standard_normal(10).astype(np.float32)
    agent.reset_hidden()
    a1 = agent.act(obs, greedy=True)
    agent.reset_hidden()
    a2 = agent.act(obs, greedy=True)
    assert a1 == a2


def test_reset_hidden_clears_state():
    agent = _agent()
    obs = np.zeros(10, dtype=np.float32)
    agent.act(obs)
    # Après act, le hidden state est non-None
    assert agent._hidden_state is not None
    agent.reset_hidden()
    assert agent._hidden_state is None


def test_begin_episode_clears_trajectory():
    agent = _agent()
    agent._episode_trajectory = [("fake", 0, 0.0, "fake", False)]
    agent.begin_episode()
    assert agent._episode_trajectory == []


def test_observe_appends_to_trajectory():
    agent = _agent()
    agent.begin_episode()
    obs = np.zeros(10, dtype=np.float32)
    next_obs = np.ones(10, dtype=np.float32)
    metrics = agent.observe(obs, action=1, reward=0.5, next_state=next_obs, done=False)
    assert len(agent._episode_trajectory) == 1
    assert agent.global_step == 1
    assert "epsilon" in metrics


def test_end_episode_pushes_trajectory_to_buffer():
    agent = _agent()
    agent.begin_episode()
    obs = np.zeros(10, dtype=np.float32)
    for _ in range(8):
        agent.observe(obs, action=0, reward=0.0, next_state=obs, done=False)
    agent.observe(obs, action=0, reward=1.0, next_state=obs, done=True)
    metrics = agent.end_episode()
    assert len(agent.buffer) == 1
    # min_episodes_to_learn=5 mais on n'a que 1 épisode → pas de train step
    assert "loss" not in metrics


def test_end_episode_triggers_train_after_min_episodes():
    """Pousser min_episodes_to_learn=5 épisodes → end_episode() doit déclencher train_step."""
    agent = _agent()
    obs = np.zeros(10, dtype=np.float32)
    for ep in range(6):
        agent.begin_episode()
        for _ in range(8):
            agent.observe(obs, action=0, reward=0.0, next_state=obs, done=False)
        agent.observe(obs, action=0, reward=1.0, next_state=obs, done=True)
        metrics = agent.end_episode()
    # Au 6e épisode (>= min_episodes_to_learn=5), train step déclenché
    assert "loss" in metrics
    assert agent.last_loss is not None


def test_v2y_polyak_tau_skips_hard_sync() -> None:
    """V2-U : V2-Y agent avec polyak_tau > 0 skip hard sync périodique."""
    cfg = DRQNConfig(
        polyak_tau=0.005,
        target_sync_steps=2,
        min_episodes_to_learn=10_000,  # disable train_step
        use_amp=False,
        sequence_length=4,
        max_steps_per_episode=8,
    )
    agent = RecurrentDQNAgent(
        obs_dim=200, n_actions=4, cfg=cfg, device="cpu", seed=0,
    )
    obs = np.zeros(200, dtype=np.float32)
    # Simuler 5 épisodes courts
    for _ in range(5):
        agent.reset_hidden()
        agent.begin_episode()
        for _ in range(4):
            agent.observe(obs, action=0, reward=0.0, next_state=obs, done=False)
        agent.end_episode()
    # target_syncs jamais incrémenté
    assert agent.target_syncs == 0
