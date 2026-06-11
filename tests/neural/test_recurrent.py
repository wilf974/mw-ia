"""Tests de RecurrentQNetwork."""
from __future__ import annotations

import pytest
pytest.importorskip("torch", reason="suite complete : requiert torch")

import torch

from mw_ia.neural.recurrent import RecurrentQNetwork


def test_recurrent_qnetwork_instantiation(cpu_device):
    net = RecurrentQNetwork(input_dim=200, n_actions=4, fc_hidden=256, lstm_hidden=128)
    net.to(cpu_device)
    assert isinstance(net, torch.nn.Module)


def test_recurrent_qnetwork_forward_single_step(cpu_device):
    net = RecurrentQNetwork(input_dim=200, n_actions=4, fc_hidden=256, lstm_hidden=128).to(cpu_device)
    obs = torch.zeros((1, 1, 200), device=cpu_device)   # (seq=1, batch=1, input_dim)
    q, hidden = net(obs, None)
    assert q.shape == (1, 1, 4)
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2
    h, c = hidden
    assert h.shape == (1, 1, 128)   # (num_layers=1, batch, lstm_hidden)
    assert c.shape == (1, 1, 128)


def test_recurrent_qnetwork_forward_with_none_hidden(cpu_device):
    """hidden=None doit auto-init zéros et retourner hidden non-None."""
    net = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    obs = torch.randn((1, 1, 10), device=cpu_device)
    q, hidden = net(obs, None)
    assert q.shape == (1, 1, 4)
    assert hidden is not None


def test_recurrent_qnetwork_forward_sequence_batch(cpu_device):
    """Forward sur séquence (seq=32, batch=4) → Q (32, 4, n_actions)."""
    net = RecurrentQNetwork(input_dim=200, n_actions=4, fc_hidden=256, lstm_hidden=128).to(cpu_device)
    obs = torch.zeros((32, 4, 200), device=cpu_device)
    q, hidden = net(obs, None)
    assert q.shape == (32, 4, 4)
    h, c = hidden
    assert h.shape == (1, 4, 128)
    assert c.shape == (1, 4, 128)


def test_recurrent_qnetwork_hidden_state_changes_output(cpu_device):
    """Sanity LSTM : passer un hidden non-zéro change la sortie vs hidden zéro."""
    net = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    obs = torch.randn((1, 1, 10), device=cpu_device)
    torch.manual_seed(0)
    q_zero, _ = net(obs, None)
    nonzero_h = torch.randn((1, 1, 8), device=cpu_device)
    nonzero_c = torch.randn((1, 1, 8), device=cpu_device)
    q_nonzero, _ = net(obs, (nonzero_h, nonzero_c))
    assert not torch.allclose(q_zero, q_nonzero)


def test_recurrent_qnetwork_determinism_same_inputs(cpu_device):
    """Même obs + même hidden + même seed → même Q (déterminisme)."""
    torch.manual_seed(42)
    net = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    obs = torch.randn((1, 1, 10), device=cpu_device)
    h = torch.zeros((1, 1, 8), device=cpu_device)
    c = torch.zeros((1, 1, 8), device=cpu_device)
    q1, _ = net(obs, (h.clone(), c.clone()))
    q2, _ = net(obs, (h.clone(), c.clone()))
    assert torch.allclose(q1, q2)


def test_recurrent_qnetwork_backward_pass_propagates_gradient(cpu_device):
    """Backward sur loss simple → gradients non-None et non-zéro sur tous les params."""
    net = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    obs = torch.randn((4, 2, 10), device=cpu_device, requires_grad=False)
    q, _ = net(obs, None)
    loss = q.pow(2).mean()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"param {name} sans gradient"
        assert not torch.allclose(p.grad, torch.zeros_like(p.grad)), \
            f"param {name} gradient nul (devrait être non-zéro pour obs non-trivial)"
