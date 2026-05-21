"""Reseau Q (FC + ReLU) - prepare pour evolution Dueling."""
from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """MLP simple : input -> [hidden + ReLU]* -> output (Q par action)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
