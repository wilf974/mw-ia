"""ConvQNetwork (Conv2d) pour V2-Z DQN à perception spatiale.

Architecture (defaults pour input 3 × 10 × 10) :
    Conv(3→32, k=3, pad=1) → ReLU
    Conv(32→64, k=3, pad=1) → ReLU
    Flatten
    Linear(64*R*C → 256) → ReLU
    Linear(256 → n_actions)

Pas de pooling pour préserver l'info spatiale sur grilles 10×10.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-cnn-perception-design.md §2
"""
from __future__ import annotations

import torch
from torch import nn


class ConvQNetwork(nn.Module):
    """Conv2d → FC pour Q-values d'un GridWorld procedural."""

    def __init__(
        self,
        *,
        in_channels: int,
        rows: int,
        cols: int,
        n_actions: int,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        padding: int = 1,
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()
        conv_layers: list[nn.Module] = []
        prev = in_channels
        for ch in conv_channels:
            conv_layers.append(nn.Conv2d(prev, ch, kernel_size=kernel_size, padding=padding))
            conv_layers.append(nn.ReLU(inplace=True))
            prev = ch
        self.conv = nn.Sequential(*conv_layers)
        flat_dim = prev * rows * cols
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, R, C) → (B, n_actions)."""
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.fc(h)
