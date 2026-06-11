"""Tests V2-ZY de ConvRecurrentQNetwork (Conv + LSTM + FC)."""
from __future__ import annotations

import torch

from mw_ia.neural.conv_recurrent import ConvRecurrentQNetwork


def test_forward_single_step_with_hidden(cpu_device: torch.device) -> None:
    """Input (seq=1, batch=1, in_channels * rows * cols) + hidden None → q (1, 1, 4) + new hidden."""
    net = ConvRecurrentQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, lstm_hidden=128,
    ).to(cpu_device)
    x = torch.zeros(1, 1, 3 * 10 * 10, device=cpu_device)
    q, hidden = net(x, None)
    assert q.shape == (1, 1, 4)
    assert isinstance(hidden, tuple) and len(hidden) == 2
    h, c = hidden
    assert h.shape == (1, 1, 128)
    assert c.shape == (1, 1, 128)


def test_forward_batch_sequence(cpu_device: torch.device) -> None:
    """Input (seq=32, batch=8, 300) → q (32, 8, 4)."""
    net = ConvRecurrentQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, lstm_hidden=128,
    ).to(cpu_device)
    x = torch.randn(32, 8, 3 * 10 * 10, device=cpu_device)
    q, hidden = net(x, None)
    assert q.shape == (32, 8, 4)
    assert hidden[0].shape == (1, 8, 128)
    assert hidden[1].shape == (1, 8, 128)


def test_hidden_state_propagation(cpu_device: torch.device) -> None:
    """Forward avec hidden propagé != forward avec hidden None."""
    net = ConvRecurrentQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, lstm_hidden=128,
    ).to(cpu_device)
    torch.manual_seed(42)
    x1 = torch.randn(1, 1, 300, device=cpu_device)
    x2 = torch.randn(1, 1, 300, device=cpu_device)
    _, h1 = net(x1, None)
    q_with_hidden, _ = net(x2, h1)
    q_no_hidden, _ = net(x2, None)
    assert not torch.allclose(q_with_hidden, q_no_hidden)


def test_gradient_flow(cpu_device: torch.device) -> None:
    """loss.backward() produit des grads non-nuls sur Conv1, Conv2, LSTM, FC."""
    net = ConvRecurrentQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, lstm_hidden=128,
    ).to(cpu_device)
    x = torch.randn(8, 4, 300, device=cpu_device, requires_grad=False)
    q, _ = net(x, None)
    loss = q.sum()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} grad is None"
        assert p.grad.abs().sum().item() > 0.0, f"{name} grad sum is 0"


def test_params_count(cpu_device: torch.device) -> None:
    """Total params ≈ 3.3M (±5%) pour défauts 3x10x10 + LSTM 128."""
    net = ConvRecurrentQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, lstm_hidden=128,
    ).to(cpu_device)
    total = sum(p.numel() for p in net.parameters())
    expected = 896 + 18_496 + 3_343_488 + 516
    assert abs(total - expected) <= expected * 0.05
