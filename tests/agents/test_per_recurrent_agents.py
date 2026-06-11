"""Tests V2-B0 integration PER dans RecurrentDQNAgent (V2-Y) et
ConvRecurrentDQNAgent (V2-ZY). Parametrized sur les 2 agents."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mw_ia.agents.recurrent_dqn import RecurrentDQNAgent
from mw_ia.agents.conv_recurrent_dqn import ConvRecurrentDQNAgent
from mw_ia.config import DRQNConfig, ConvRecurrentDQNConfig
from mw_ia.neural.prioritized_sequence_buffer import (
    BetaScheduler,
    PrioritizedSequenceReplayBuffer,
)
from mw_ia.neural.sequence_buffer import SequenceReplayBuffer


def _build_drqn(per_enabled: bool, **kwargs) -> RecurrentDQNAgent:
    cfg = DRQNConfig(
        per_enabled=per_enabled,
        replay_capacity=50,
        min_episodes_to_learn=5,
        batch_size=4,
        sequence_length=4,
        max_steps_per_episode=10,
        episodes=100,
        use_amp=False,
        **kwargs,
    )
    return RecurrentDQNAgent(obs_dim=8, n_actions=4, cfg=cfg, device="cpu", seed=0)


def _build_conv_recurrent(per_enabled: bool, **kwargs) -> ConvRecurrentDQNAgent:
    cfg = ConvRecurrentDQNConfig(
        per_enabled=per_enabled,
        replay_capacity=50,
        min_episodes_to_learn=5,
        batch_size=4,
        sequence_length=4,
        max_steps_per_episode=10,
        episodes=100,
        use_amp=False,
        eval_enabled=False,
        **kwargs,
    )
    return ConvRecurrentDQNAgent(
        in_channels=3, rows=4, cols=4, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )


AGENT_BUILDERS = [
    pytest.param(_build_drqn, "drqn", id="v2y_drqn"),
    pytest.param(_build_conv_recurrent, "conv_recurrent", id="v2zy_conv_recurrent"),
]


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_per_disabled_instantiates_sequence_buffer(builder, agent_kind) -> None:
    """per_enabled=False -> SequenceReplayBuffer V2-Y baseline strict."""
    agent = builder(per_enabled=False)
    assert isinstance(agent.buffer, SequenceReplayBuffer)
    assert not isinstance(agent.buffer, PrioritizedSequenceReplayBuffer)
    assert agent._beta_scheduler is None


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_per_enabled_instantiates_prioritized_buffer(builder, agent_kind) -> None:
    """per_enabled=True -> PrioritizedSequenceReplayBuffer + BetaScheduler."""
    agent = builder(per_enabled=True)
    assert isinstance(agent.buffer, PrioritizedSequenceReplayBuffer)
    assert isinstance(agent._beta_scheduler, BetaScheduler)


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_episode_count_increments_independently_of_buffer_len(builder, agent_kind) -> None:
    """_episode_count croit meme quand le buffer plafonne a capacity."""
    agent = builder(per_enabled=True)
    # Pousser 100 trajectoires (capacity=50)
    for ep in range(100):
        if agent_kind == "drqn":
            traj = [(np.zeros(8, dtype=np.float32), 0, 0.1, np.zeros(8, dtype=np.float32), False)
                    for _ in range(5)]
        else:
            obs = np.zeros((3, 4, 4), dtype=np.float32)
            traj = [(obs.flatten(), 0, 0.1, obs.flatten(), False) for _ in range(5)]
        agent._episode_trajectory = traj
        agent.end_episode()
    assert len(agent.buffer) == 50  # capacity
    assert agent._episode_count == 100  # croit independamment


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_beta_anneals_with_episode_count(builder, agent_kind) -> None:
    """beta(0) == beta_start, beta(episodes) == beta_end."""
    agent = builder(per_enabled=True)
    assert agent._beta_scheduler.beta(0) == pytest.approx(0.4)
    assert agent._beta_scheduler.beta(100) == pytest.approx(1.0)


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_end_episode_per_path_emits_metrics(builder, agent_kind) -> None:
    """end_episode avec PER actif retourne 'per_beta' dans metrics quand train se declenche."""
    agent = builder(per_enabled=True)
    # Pousser assez de trajectoires pour passer le seuil
    for _ in range(10):
        if agent_kind == "drqn":
            traj = [(np.zeros(8, dtype=np.float32), 0, 0.1, np.zeros(8, dtype=np.float32), False)
                    for _ in range(5)]
        else:
            obs = np.zeros((3, 4, 4), dtype=np.float32)
            traj = [(obs.flatten(), 0, 0.1, obs.flatten(), False) for _ in range(5)]
        agent._episode_trajectory = traj
        metrics = agent.end_episode()
    # Au 10e episode, le buffer a >= min_episodes_to_learn (5), train se declenche
    assert "per_beta" in metrics
    assert "loss" in metrics


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_end_episode_per_disabled_skips_per_beta(builder, agent_kind) -> None:
    """end_episode avec PER desactive n'emet PAS 'per_beta' dans metrics."""
    agent = builder(per_enabled=False)
    for _ in range(10):
        if agent_kind == "drqn":
            traj = [(np.zeros(8, dtype=np.float32), 0, 0.1, np.zeros(8, dtype=np.float32), False)
                    for _ in range(5)]
        else:
            obs = np.zeros((3, 4, 4), dtype=np.float32)
            traj = [(obs.flatten(), 0, 0.1, obs.flatten(), False) for _ in range(5)]
        agent._episode_trajectory = traj
        metrics = agent.end_episode()
    assert "per_beta" not in metrics
    assert "loss" in metrics


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_polyak_and_per_cohabit(builder, agent_kind) -> None:
    """Polyak + PER : target update a chaque train_step, sans crash."""
    agent = builder(per_enabled=True, polyak_tau=0.005)
    target_snapshot = {
        name: p.clone() for name, p in agent.target.named_parameters()
    }
    for _ in range(10):
        if agent_kind == "drqn":
            traj = [(np.zeros(8, dtype=np.float32), 0, 0.1, np.zeros(8, dtype=np.float32), False)
                    for _ in range(5)]
        else:
            obs = np.zeros((3, 4, 4), dtype=np.float32)
            traj = [(obs.flatten(), 0, 0.1, obs.flatten(), False) for _ in range(5)]
        agent._episode_trajectory = traj
        agent.end_episode()
    # Target a ete modifie par Polyak
    target_modified = False
    for name, p in agent.target.named_parameters():
        if not torch.allclose(p, target_snapshot[name]):
            target_modified = True
            break
    assert target_modified


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_save_load_unchanged_with_per(builder, agent_kind, tmp_path) -> None:
    """save/load fonctionnent avec PER actif (priorite PER non sauvegardee, intentionnel)."""
    agent = builder(per_enabled=True)
    for _ in range(6):
        if agent_kind == "drqn":
            traj = [(np.zeros(8, dtype=np.float32), 0, 0.1, np.zeros(8, dtype=np.float32), False)
                    for _ in range(5)]
        else:
            obs = np.zeros((3, 4, 4), dtype=np.float32)
            traj = [(obs.flatten(), 0, 0.1, obs.flatten(), False) for _ in range(5)]
        agent._episode_trajectory = traj
        agent.end_episode()
    ckpt_path = tmp_path / "agent.pt"
    agent.save(str(ckpt_path))
    assert ckpt_path.exists()
    # Reload
    agent2 = builder(per_enabled=True)
    agent2.load(str(ckpt_path))
    # Online params identiques
    for name, p1 in agent.online.named_parameters():
        p2 = dict(agent2.online.named_parameters())[name]
        assert torch.allclose(p1, p2)
