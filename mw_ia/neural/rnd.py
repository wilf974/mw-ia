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


class RNDModule:
    """RND single-stream : bonus de nouveaute appris, normalise et clippe.

    compute_bonus(obs) : met a jour le normaliseur d'obs, calcule l'erreur de
    prediction, la normalise par sa running-std, clippe, retourne 0 pendant le
    warmup. update(obs) : un pas de gradient sur le predictor.

    L'agent DQN ne connait pas RND : le runner ajoute beta*compute_bonus au reward.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        rows: int,
        cols: int,
        embed_dim: int = 128,
        lr: float = 1e-4,
        clip: float = 5.0,
        warmup_steps: int = 1000,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        self.device = torch.device(device)
        # target construite en premier, predictor ensuite -> inits aleatoires distinctes
        self.target = _RNDNet(
            in_channels=in_channels, rows=rows, cols=cols, embed_dim=embed_dim
        ).to(self.device)
        self.predictor = _RNDNet(
            in_channels=in_channels, rows=rows, cols=cols, embed_dim=embed_dim
        ).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.obs_rms = RunningMeanStd(shape=(in_channels, rows, cols))
        self.bonus_rms = RunningMeanStd(shape=())
        self.clip = clip
        self.warmup_steps = warmup_steps
        self._step = 0

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        o = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(o, -5.0, 5.0).astype(np.float32)

    def compute_bonus(self, obs: np.ndarray) -> float:
        self.obs_rms.update(obs[None])
        self._step += 1
        o = self._normalize_obs(obs)
        t = torch.from_numpy(o[None]).to(self.device)
        with torch.no_grad():
            err = float(((self.predictor(t) - self.target(t)) ** 2).mean().item())
        self.bonus_rms.update(np.array([err], dtype=np.float64))
        if self._step <= self.warmup_steps:
            return 0.0
        bonus = err / (float(self.bonus_rms.std) + 1e-8)
        return float(np.clip(bonus, 0.0, self.clip))

    def update(self, obs: np.ndarray) -> float:
        o = self._normalize_obs(obs)
        t = torch.from_numpy(o[None]).to(self.device)
        with torch.no_grad():
            tgt = self.target(t)
        pred = self.predictor(t)
        loss = ((pred - tgt) ** 2).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
