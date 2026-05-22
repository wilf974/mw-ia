"""Tests V2-Z de ConvDQNAgent (DQN à perception spatiale)."""
from __future__ import annotations

import numpy as np
import torch

from mw_ia.agents.conv_dqn import ConvDQNAgent
from mw_ia.config import ConvDQNConfig
from mw_ia.guardrails import VariantSpec, verify_formal


def _make_agent(cfg: ConvDQNConfig | None = None, seed: int = 0) -> ConvDQNAgent:
    cfg = cfg or ConvDQNConfig(min_replay_to_learn=4, batch_size=2, train_every=1)
    return ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=seed,
    )


def _obs() -> np.ndarray:
    return np.zeros((3, 10, 10), dtype=np.float32)


def test_init() -> None:
    """Online + target nets, buffer vide, optimizer Adam, global_step=0."""
    agent = _make_agent()
    assert agent.global_step == 0
    assert len(agent.buffer) == 0
    assert agent.epsilon == agent.cfg.epsilon_start
    # online et target ont les mêmes poids dès l'init (sync_target dans Trainer.__init__)
    for p_o, p_t in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.allclose(p_o, p_t)


def test_act_random_when_eps_high() -> None:
    """Eps=1.0 → action ∈ {0,1,2,3}, distribution non dégénérée."""
    cfg = ConvDQNConfig(epsilon_start=1.0, epsilon_end=1.0, min_replay_to_learn=10_000)
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=42,
    )
    actions = {agent.act(_obs()) for _ in range(50)}
    assert actions.issubset({0, 1, 2, 3})
    assert len(actions) > 1  # pas dégénéré sur 50 tirages avec seed=42


def test_act_greedy_when_eps_zero() -> None:
    """Eps=0 → action = argmax Q-values (déterministe pour même obs)."""
    cfg = ConvDQNConfig(epsilon_start=0.0, epsilon_end=0.0, min_replay_to_learn=10_000)
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=42,
    )
    obs = _obs()
    a1 = agent.act(obs)
    a2 = agent.act(obs)
    assert a1 == a2  # déterministe


def test_observe_pushes_buffer() -> None:
    """1 observe → buffer.size = 1, global_step = 1."""
    agent = _make_agent()
    agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    assert len(agent.buffer) == 1
    assert agent.global_step == 1


def test_target_sync() -> None:
    """Après target_sync_steps updates, target_qnet == online_qnet."""
    cfg = ConvDQNConfig(
        target_sync_steps=3, min_replay_to_learn=10_000, train_every=1_000_000,
    )
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    # Modifier manuellement online pour qu'il diffère de target
    with torch.no_grad():
        for p in agent.online.parameters():
            p.add_(1.0)
    # Vérifier qu'ils diffèrent
    diff_before = sum(
        (po - pt).abs().sum().item()
        for po, pt in zip(agent.online.parameters(), agent.target.parameters())
    )
    assert diff_before > 0.0
    # 3 observes → sync target au step 3
    for _ in range(3):
        agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    diff_after = sum(
        (po - pt).abs().sum().item()
        for po, pt in zip(agent.online.parameters(), agent.target.parameters())
    )
    assert diff_after == 0.0


def test_train_trigger_min_replay() -> None:
    """train_step ne fire pas avant len(buffer) >= max(min_replay, batch_size)."""
    cfg = ConvDQNConfig(min_replay_to_learn=5, batch_size=8, train_every=1)
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    # 7 observes : buffer.size = 7 < max(5, 8) = 8 → pas de train
    for _ in range(7):
        m = agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
        assert "loss" not in m
    # 1 observe de plus : buffer.size = 8 >= 8 → train_step fire
    m = agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    assert "loss" in m
    assert np.isfinite(m["loss"])


def test_aether_smoke() -> None:
    """Smoke E2E : VariantSpec dérivé d'un agent V2-Z passe Aether I1-I8."""
    cfg = ConvDQNConfig()
    spec = VariantSpec(
        gamma=cfg.gamma, lr=cfg.lr,
        epsilon_start=cfg.epsilon_start, epsilon_end=cfg.epsilon_end,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
        batch_size=cfg.batch_size,
        replay_capacity=cfg.replay_capacity,
        target_sync_steps=cfg.target_sync_steps,
    )
    assert verify_formal(spec).passed


def test_double_dqn_branch_differs_from_standard() -> None:
    """V2-W : avec online ≠ target, les formules DQN et Double DQN divergent.

    - DQN classique :  q_next = max_a Q_target(s', a)
    - Double DQN :     q_next = Q_target(s', argmax_a Q_online(s', a))

    Si argmax_online ≠ argmax_target, les deux formules donnent des q_next
    différents. Pour rendre ça déterministe, on désynchronise volontairement
    online/target avant comparaison.
    """
    from mw_ia.neural.conv_network import ConvQNetwork

    online = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    )
    target = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    )
    # Sync initial pour partir de poids identiques
    target.load_state_dict(online.state_dict())
    # Désynchroniser online en ajoutant un offset
    with torch.no_grad():
        for p in online.parameters():
            p.add_(0.5)

    torch.manual_seed(42)
    next_states = torch.randn(4, 3, 10, 10)

    with torch.no_grad():
        # Formule DQN classique (V2-Z baseline)
        q_next_dqn = target(next_states).max(dim=1).values
        # Formule Double DQN (V2-W)
        next_actions = online(next_states).argmax(dim=1)
        q_next_double = target(next_states).gather(1, next_actions.view(-1, 1)).squeeze(1)

    # Avec online ≠ target, les 2 formules DOIVENT diverger sur au moins une transition
    assert not torch.allclose(q_next_dqn, q_next_double), (
        "Double DQN doit différer de DQN classique quand online ≠ target"
    )
