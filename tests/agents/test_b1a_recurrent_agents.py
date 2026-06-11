"""Tests V2-B1a integration agents V2-Y RecurrentDQNAgent et V2-ZY ConvRecurrentDQNAgent.

Parametrized sur les 2 agents.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
pytest.importorskip("torch", reason="suite complete : requiert torch")

import torch

from mw_ia.agents.recurrent_dqn import RecurrentDQNAgent
from mw_ia.agents.conv_recurrent_dqn import ConvRecurrentDQNAgent
from mw_ia.config import DRQNConfig, ConvRecurrentDQNConfig
from mw_ia.training.snapshot_store import SnapshotTrajectoryStore


def _build_drqn(b1a_enabled: bool, per_enabled: bool = False, **kwargs) -> RecurrentDQNAgent:
    cfg = DRQNConfig(
        b1a_enabled=b1a_enabled,
        per_enabled=per_enabled,
        replay_capacity=50,
        min_episodes_to_learn=5,
        batch_size=10,
        sequence_length=4,
        max_steps_per_episode=8,
        episodes=100,
        use_amp=False,
        b1a_snapshot_size=10,
        b1a_n_windows=2,
        b1a_mix_ratio=0.2,
        **kwargs,
    )
    return RecurrentDQNAgent(obs_dim=4, n_actions=4, cfg=cfg, device="cpu", seed=0)


def _build_conv_recurrent(b1a_enabled: bool, per_enabled: bool = False, **kwargs) -> ConvRecurrentDQNAgent:
    cfg = ConvRecurrentDQNConfig(
        b1a_enabled=b1a_enabled,
        per_enabled=per_enabled,
        replay_capacity=50,
        min_episodes_to_learn=5,
        batch_size=10,
        sequence_length=4,
        max_steps_per_episode=8,
        episodes=100,
        use_amp=False,
        eval_enabled=False,
        b1a_snapshot_size=10,
        b1a_n_windows=2,
        b1a_mix_ratio=0.2,
        **kwargs,
    )
    return ConvRecurrentDQNAgent(
        in_channels=2, rows=2, cols=2, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )


AGENT_BUILDERS = [
    pytest.param(_build_drqn, "drqn", 4, id="v2y_drqn"),
    pytest.param(_build_conv_recurrent, "conv_recurrent", 8, id="v2zy_conv_recurrent"),
]


def _make_success_trajectory(agent_kind: str, obs_dim: int, length: int = 5):
    """Trajectoire successful synthetique (terminated AND total_reward > 0)."""
    traj = []
    for t in range(length):
        if agent_kind == "drqn":
            s = np.zeros(obs_dim, dtype=np.float32)
            sp = np.zeros(obs_dim, dtype=np.float32)
        else:
            obs = np.zeros((2, 2, 2), dtype=np.float32)
            s = obs.flatten()
            sp = obs.flatten()
        r = 1.0 if t == length - 1 else -0.01
        d = (t == length - 1)
        traj.append((s, 0, r, sp, d))
    return traj


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_disabled_no_snapshot_store(builder, agent_kind, obs_dim) -> None:
    """b1a_enabled=False -> snapshot_store is None, on_new_best() retourne 0."""
    agent = builder(b1a_enabled=False)
    assert agent.snapshot_store is None
    assert agent.on_new_best() == 0


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_enabled_instantiates_snapshot_store(builder, agent_kind, obs_dim) -> None:
    """b1a_enabled=True -> SnapshotTrajectoryStore instancie, len == 0 init."""
    agent = builder(b1a_enabled=True)
    assert isinstance(agent.snapshot_store, SnapshotTrajectoryStore)
    assert len(agent.snapshot_store) == 0


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_on_new_best_triggers_capture(builder, agent_kind, obs_dim) -> None:
    """on_new_best() apres push de trajectoires successful -> len > 0, n_captures == 1."""
    agent = builder(b1a_enabled=True)
    # Push 10 success trajectoires dans le main buffer
    for _ in range(10):
        traj = _make_success_trajectory(agent_kind, obs_dim)
        agent._episode_trajectory = traj
        agent.end_episode()
    n_captured = agent.on_new_best()
    assert n_captured > 0
    assert agent.snapshot_store.n_captures == 1
    assert len(agent.snapshot_store) == n_captured


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_threshold_strict_pure_main_below_threshold(builder, agent_kind, obs_dim) -> None:
    """Snapshot avec moins que snapshot_B traj -> batch tire purement main (no mix).

    Strategy : on capture peu de trajectoires (< snapshot_B), puis on appelle
    _sample_training_batch et on verifie que batch_size == main_B (pas mix).
    """
    agent = builder(b1a_enabled=True)
    # snapshot_B = int(B * mix_ratio) = int(10 * 0.2) = 2
    # Pour rester sous threshold, on doit avoir < 2 traj dans le snapshot
    # Push 1 success seulement
    traj = _make_success_trajectory(agent_kind, obs_dim)
    agent._episode_trajectory = traj
    agent.end_episode()
    agent.on_new_best()  # capture 1 traj
    assert len(agent.snapshot_store) == 1  # < snapshot_B=2

    # Maintenant remplit le main buffer avec assez de trajectoires
    for _ in range(20):
        traj = _make_success_trajectory(agent_kind, obs_dim)
        agent._episode_trajectory = traj
        agent.end_episode()

    # Call _sample_training_batch : doit returner batch shape (seq, B, ...) car b1a inactif
    tb = agent._sample_training_batch()
    # B=10, snapshot_B=2, len(snapshot)=1 < 2 -> b1a_active=False -> main_B=B=10
    assert tb.batch.states.shape[1] == agent.cfg.batch_size


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_active_mixes_batch_shape(builder, agent_kind, obs_dim) -> None:
    """Snapshot rempli -> batch shape (seq, B, ...) avec B*0.2 derniers du snapshot."""
    agent = builder(b1a_enabled=True)
    # Push assez de success traj pour avoir snapshot_size=10 dans le store
    for _ in range(15):
        traj = _make_success_trajectory(agent_kind, obs_dim)
        agent._episode_trajectory = traj
        agent.end_episode()
    agent.on_new_best()
    assert len(agent.snapshot_store) >= 2  # snapshot_B = int(10*0.2) = 2

    # Sample : doit mixer 8 main + 2 snapshot = 10 total
    tb = agent._sample_training_batch()
    assert tb.batch.states.shape[1] == agent.cfg.batch_size  # B=10
    # Pas de verification fine sur quelles colonnes viennent de quoi
    # (cf test suivant pour weights)


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_per_on_weights_concat_correct(builder, agent_kind, obs_dim) -> None:
    """PER + B1a -> weights shape (B,), weights[main_B:] == 1.0 strict."""
    agent = builder(b1a_enabled=True, per_enabled=True)
    # Push assez de traj
    for _ in range(15):
        traj = _make_success_trajectory(agent_kind, obs_dim)
        agent._episode_trajectory = traj
        agent.end_episode()
    agent.on_new_best()
    # Update some priorities pour rendre PER actif (sinon all uniform = bias)
    for _ in range(3):
        agent._episode_trajectory = _make_success_trajectory(agent_kind, obs_dim)
        agent.end_episode()

    tb = agent._sample_training_batch()
    # main_B = 8, snapshot_B = 2
    assert tb.weights is not None
    assert tb.weights.shape == (agent.cfg.batch_size,)
    # Snapshot portion (indices [main_B:]) doit avoir weights == 1.0
    main_B = agent.cfg.batch_size - int(agent.cfg.batch_size * agent.cfg.b1a_mix_ratio)
    snapshot_weights = tb.weights[main_B:]
    assert np.allclose(snapshot_weights, 1.0)


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_per_on_priorities_updated_main_only(builder, agent_kind, obs_dim) -> None:
    """update_priorities appele uniquement sur td_errors[:main_B] (pas le batch entier)."""
    agent = builder(b1a_enabled=True, per_enabled=True)
    for _ in range(15):
        traj = _make_success_trajectory(agent_kind, obs_dim)
        agent._episode_trajectory = traj
        agent.end_episode()
    agent.on_new_best()
    assert len(agent.snapshot_store) >= 2

    # Mock buffer.update_priorities pour intercepter les appels
    original_update_priorities = agent.buffer.update_priorities
    captured_calls = []

    def mock_update(indices, td_errors):
        captured_calls.append((indices.copy(), td_errors.copy()))
        return original_update_priorities(indices, td_errors)

    agent.buffer.update_priorities = mock_update

    # Trigger train step
    traj = _make_success_trajectory(agent_kind, obs_dim)
    agent._episode_trajectory = traj
    agent.end_episode()

    # Verifier que update_priorities a ete appele avec td_errors[:main_B]
    assert len(captured_calls) > 0
    last_call_indices, last_call_td = captured_calls[-1]
    main_B = agent.cfg.batch_size - int(agent.cfg.batch_size * agent.cfg.b1a_mix_ratio)
    assert last_call_indices.shape[0] == main_B
    assert last_call_td.shape[0] == main_B
