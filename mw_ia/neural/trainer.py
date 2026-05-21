"""Boucle d'optimisation DQN : Huber Loss + Adam + AMP optionnel."""
from __future__ import annotations

import torch
from torch import nn

from mw_ia.neural.network import QNetwork
from mw_ia.neural.replay_buffer import Batch


class DQNTrainer:
    """Encapsule online net + target net + optimizer + AMP."""

    def __init__(
        self,
        online: QNetwork,
        target: QNetwork,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cuda",
        use_amp: bool = True,
    ) -> None:
        self.online = online
        self.target = target
        self.gamma = gamma
        self.device = torch.device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self._scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.sync_target()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def step(self, batch: Batch) -> float:
        states = torch.from_numpy(batch.states).to(self.device, non_blocking=True)
        actions = torch.from_numpy(batch.actions).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(batch.rewards).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(batch.next_states).to(self.device, non_blocking=True)
        dones = torch.from_numpy(batch.dones).to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
            q_pred = self.online(states).gather(1, actions.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                q_next = self.target(next_states).max(dim=1).values
                target_q = rewards + self.gamma * q_next * (1.0 - dones)
            loss = self.loss_fn(q_pred, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
            self.optimizer.step()

        return float(loss.detach().item())
