"""Tests V2-Z de ConvQNetwork (Conv2d + FC pour DQN spatial)."""
from __future__ import annotations

import torch

from mw_ia.neural.conv_network import ConvQNetwork


def test_forward_single_sample(cpu_device: torch.device) -> None:
    """Input (1, 3, 10, 10) → output (1, 4)."""
    net = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    x = torch.zeros(1, 3, 10, 10, device=cpu_device)
    y = net(x)
    assert y.shape == (1, 4)


def test_forward_batch(cpu_device: torch.device) -> None:
    """Input (32, 3, 10, 10) → output (32, 4)."""
    net = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    x = torch.randn(32, 3, 10, 10, device=cpu_device)
    y = net(x)
    assert y.shape == (32, 4)


def test_params_count(cpu_device: torch.device) -> None:
    """Total params ≈ 1.66M (tolerance ±5%) pour défauts (3, 10, 10)."""
    net = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    total = sum(p.numel() for p in net.parameters())
    # Conv1=3*32*9+32=896, Conv2=32*64*9+64=18496, FC1=6400*256+256=1638656, FC2=256*4+4=1028
    expected = 896 + 18_496 + 1_638_656 + 1_028
    assert abs(total - expected) <= expected * 0.05


def test_gradient_flow(cpu_device: torch.device) -> None:
    """loss.backward() produit des grads non-nulls sur toutes les couches."""
    net = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    x = torch.randn(8, 3, 10, 10, device=cpu_device, requires_grad=False)
    y = net(x)
    loss = y.sum()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name}: pas de gradient"
        assert p.grad.abs().sum().item() > 0.0, f"{name}: gradient nul"


def test_state_dict_compat(cpu_device: torch.device) -> None:
    """target.load_state_dict(online.state_dict()) round-trip exact."""
    online = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    target = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    target.load_state_dict(online.state_dict())
    x = torch.randn(4, 3, 10, 10, device=cpu_device)
    assert torch.allclose(online(x), target(x))
