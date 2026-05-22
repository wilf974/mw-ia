"""Tests de RecurrentQNetwork."""
from __future__ import annotations

import pytest
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
