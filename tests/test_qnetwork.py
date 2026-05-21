"""Tests QNetwork."""
from __future__ import annotations

import pytest
import torch

from mw_ia.neural.network import QNetwork


def test_forward_shape() -> None:
    net = QNetwork(input_dim=4, output_dim=4, hidden_layers=(64, 64))
    x = torch.randn(8, 4)
    out = net(x)
    assert out.shape == (8, 4)


def test_configurable_hidden_layers() -> None:
    net = QNetwork(input_dim=2, output_dim=3, hidden_layers=(32, 16, 8))
    x = torch.randn(2, 2)
    out = net(x)
    assert out.shape == (2, 3)
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 4  # 3 hidden + 1 output


def test_parameters_require_grad() -> None:
    net = QNetwork(input_dim=4, output_dim=4)
    assert any(p.requires_grad for p in net.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA indisponible")
def test_runs_on_cuda() -> None:
    net = QNetwork(input_dim=4, output_dim=4).to("cuda")
    x = torch.randn(2, 4, device="cuda")
    out = net(x)
    assert out.device.type == "cuda"
