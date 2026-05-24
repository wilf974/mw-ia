"""Tests V2-ZY de ConvRecurrentDQNAgent (combo CNN + LSTM + Double DQN)."""
from __future__ import annotations

import numpy as np
import torch

from mw_ia.agents.conv_recurrent_dqn import ConvRecurrentDQNAgent
from mw_ia.config import ConvRecurrentDQNConfig
from mw_ia.guardrails import VariantSpec, verify_formal


def _make_agent(cfg: ConvRecurrentDQNConfig | None = None, seed: int = 0) -> ConvRecurrentDQNAgent:
    cfg = cfg or ConvRecurrentDQNConfig(
        min_episodes_to_learn=2, batch_size=2, train_steps_per_episode=1,
        sequence_length=4, max_steps_per_episode=8,
        replay_capacity=10, use_amp=False,
    )
    return ConvRecurrentDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=seed,
    )


def _obs() -> np.ndarray:
    return np.zeros((3, 10, 10), dtype=np.float32)


def test_init() -> None:
    """Online + target nets, sequence buffer empty, global_step=0, hidden None."""
    agent = _make_agent()
    assert agent.global_step == 0
    assert len(agent.buffer) == 0
    assert agent._hidden_state is None
    for p_o, p_t in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.allclose(p_o, p_t)


def test_reset_hidden() -> None:
    """reset_hidden() remet _hidden_state à None."""
    agent = _make_agent()
    agent.act(_obs())
    assert agent._hidden_state is not None
    agent.reset_hidden()
    assert agent._hidden_state is None


def test_begin_episode_resets_hidden_and_starts_trajectory() -> None:
    """begin_episode() reset hidden + vide la trajectoire en cours."""
    agent = _make_agent()
    agent.act(_obs())
    agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    assert len(agent._episode_trajectory) == 1
    agent.begin_episode()
    assert agent._hidden_state is None
    assert len(agent._episode_trajectory) == 0


def test_act_maintains_hidden_state_even_in_exploration() -> None:
    """eps=1.0 → action random MAIS hidden state updated par forward LSTM."""
    cfg = ConvRecurrentDQNConfig(epsilon_start=1.0, epsilon_end=1.0)
    agent = ConvRecurrentDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=42,
    )
    assert agent._hidden_state is None
    agent.act(_obs())
    assert agent._hidden_state is not None


def test_act_greedy_deterministic() -> None:
    """Eps=0 + même hidden + même obs → mêmes actions."""
    cfg = ConvRecurrentDQNConfig(epsilon_start=0.0, epsilon_end=0.0)
    agent = ConvRecurrentDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=42,
    )
    obs = _obs()
    agent.reset_hidden()
    a1 = agent.act(obs)
    agent.reset_hidden()
    a2 = agent.act(obs)
    assert a1 == a2


def test_end_episode_trains_when_buffer_full() -> None:
    """end_episode() train_steps quand buffer >= max(min_episodes_to_learn, batch_size)."""
    cfg = ConvRecurrentDQNConfig(
        min_episodes_to_learn=2, batch_size=2, train_steps_per_episode=1,
        sequence_length=4, max_steps_per_episode=8,
        replay_capacity=10, use_amp=False,
    )
    agent = ConvRecurrentDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    agent.begin_episode()
    for _ in range(4):
        agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    m1 = agent.end_episode()
    assert "loss" not in m1
    agent.begin_episode()
    for _ in range(4):
        agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    m2 = agent.end_episode()
    assert "loss" in m2
    assert np.isfinite(m2["loss"])


def test_aether_smoke() -> None:
    """Smoke E2E : VariantSpec dérivé d'un agent V2-ZY passe Aether I1-I8."""
    cfg = ConvRecurrentDQNConfig()
    spec = VariantSpec(
        gamma=cfg.gamma, lr=cfg.lr,
        epsilon_start=cfg.epsilon_start, epsilon_end=cfg.epsilon_end,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
        batch_size=cfg.batch_size,
        replay_capacity=cfg.replay_capacity,
        target_sync_steps=cfg.target_sync_steps,
    )
    assert verify_formal(spec).passed


def test_v2zy_polyak_tau_skips_hard_sync() -> None:
    """V2-U : V2-ZY agent avec polyak_tau > 0 skip hard sync périodique."""
    cfg = ConvRecurrentDQNConfig(
        polyak_tau=0.005,
        target_sync_steps=2,
        min_episodes_to_learn=10_000,
        use_amp=False,
        sequence_length=4,
        max_steps_per_episode=8,
    )
    agent = ConvRecurrentDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    obs = np.zeros((3, 10, 10), dtype=np.float32)
    for _ in range(5):
        agent.reset_hidden()
        agent.begin_episode()
        for _ in range(4):
            agent.observe(obs, action=0, reward=0.0, next_state=obs, done=False)
        agent.end_episode()
    assert agent.target_syncs == 0
