"""Tests V2-U de polyak_update sur _ConvDQNTrainer (formule pure)."""
from __future__ import annotations

import torch

from mw_ia.agents.conv_dqn import ConvDQNAgent
from mw_ia.config import ConvDQNConfig


def _build_agent_pair() -> tuple[ConvDQNAgent, dict[str, torch.Tensor]]:
    """Construit un agent et capture une snapshot des params target initiaux."""
    cfg = ConvDQNConfig(min_replay_to_learn=10_000, use_amp=False)
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    # Désynchroniser online pour avoir online != target
    with torch.no_grad():
        for p in agent.online.parameters():
            p.add_(0.5)
    target_snapshot = {
        name: p.clone()
        for name, p in agent.target.named_parameters()
    }
    return agent, target_snapshot


def test_polyak_tau_zero_is_noop() -> None:
    """polyak_update(0.0) ne modifie pas target."""
    agent, snapshot = _build_agent_pair()
    agent.trainer.polyak_update(0.0)
    for name, p in agent.target.named_parameters():
        assert torch.allclose(p, snapshot[name]), f"{name} modified by tau=0.0"


def test_polyak_tau_one_copies_online_to_target() -> None:
    """polyak_update(1.0) rend target identique à online."""
    agent, _ = _build_agent_pair()
    agent.trainer.polyak_update(1.0)
    for p_target, p_online in zip(
        agent.target.parameters(), agent.online.parameters()
    ):
        assert torch.allclose(p_target, p_online)


def test_polyak_intermediate_tau() -> None:
    """polyak_update(0.5) produit target = 0.5 × old_target + 0.5 × online."""
    agent, snapshot = _build_agent_pair()
    # Capture online snapshot before update (in case Polyak modifies online by mistake)
    online_snapshot = {
        name: p.clone() for name, p in agent.online.named_parameters()
    }
    agent.trainer.polyak_update(0.5)
    for name, p_target_new in agent.target.named_parameters():
        expected = 0.5 * snapshot[name] + 0.5 * online_snapshot[name]
        assert torch.allclose(p_target_new, expected, atol=1e-6), (
            f"{name} mismatch with formula 0.5 × old_target + 0.5 × online"
        )


def test_polyak_idempotent_when_online_equals_target() -> None:
    """Si online == target, polyak_update(τ) ne change rien quel que soit τ."""
    cfg = ConvDQNConfig(min_replay_to_learn=10_000, use_amp=False)
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    # online == target dès l'init (sync_target dans Trainer.__init__)
    snapshot = {
        name: p.clone() for name, p in agent.target.named_parameters()
    }
    for tau in [0.1, 0.5, 0.9, 1.0]:
        agent.trainer.polyak_update(tau)
        for name, p in agent.target.named_parameters():
            assert torch.allclose(p, snapshot[name]), (
                f"tau={tau}: {name} modifié alors que online == target"
            )


def test_polyak_does_not_modify_online() -> None:
    """polyak_update ne touche qu'aux paramètres de target, online inchangé."""
    agent, _ = _build_agent_pair()
    online_snapshot = {
        name: p.clone() for name, p in agent.online.named_parameters()
    }
    agent.trainer.polyak_update(0.5)
    for name, p in agent.online.named_parameters():
        assert torch.allclose(p, online_snapshot[name]), (
            f"{name} (online) modifié par polyak_update"
        )
