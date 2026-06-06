"""RND (Random Network Distillation, Burda et al. 2018) — exploration intrinseque.

V2-C0 single-stream : le bonus = erreur de prediction d'un predictor entraine
a imiter une target figee a init aleatoire. Etats nouveaux -> grosse erreur ->
gros bonus. Le bonus normalise est ajoute au reward extrinseque dans le runner.

Voir spec : docs/superpowers/specs/2026-06-06-mw-ia-v2c-rnd-design.md
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mw_ia.neural.running_mean_std import RunningMeanStd


class _RNDNet(nn.Module):
    """Petit CNN (C,R,Cc) -> embedding. Reutilise pour target (figee) et predictor."""

    def __init__(self, *, in_channels: int, rows: int, cols: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * rows * cols, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
