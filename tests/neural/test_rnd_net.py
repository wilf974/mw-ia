"""Tests du reseau _RNDNet (V2-C0 RND)."""
from __future__ import annotations

import torch

from mw_ia.neural.rnd import _RNDNet


def test_output_shape():
    net = _RNDNet(in_channels=3, rows=4, cols=4, embed_dim=8)
    x = torch.zeros(2, 3, 4, 4)
    out = net(x)
    assert out.shape == (2, 8)


def test_two_nets_differ_on_random_init():
    torch.manual_seed(0)
    a = _RNDNet(in_channels=3, rows=4, cols=4, embed_dim=8)
    b = _RNDNet(in_channels=3, rows=4, cols=4, embed_dim=8)
    x = torch.ones(1, 3, 4, 4)
    assert not torch.allclose(a(x), b(x))
