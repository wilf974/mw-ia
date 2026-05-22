"""RecurrentDQNTrainer — boucle d'optimisation BPTT pour DRQN.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md §2.1

DRQN simple (Hausknecht & Stone 2015) : hidden state zero-init au début de
chaque séquence de training (pas de burn-in en V2-Y MVP).
"""
from __future__ import annotations

import torch
from torch import nn

from mw_ia.neural.recurrent import RecurrentQNetwork
from mw_ia.neural.sequence_buffer import BatchSeq


class RecurrentDQNTrainer:
    """Encapsule online net + target net + optimizer + AMP, BPTT 32 steps avec mask."""

    def __init__(
        self,
        online: RecurrentQNetwork,
        target: RecurrentQNetwork,
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
        # SmoothL1Loss avec reduction='none' pour pouvoir appliquer le mask manuellement
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self._scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.sync_target()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def step(self, batch: BatchSeq) -> float:
        states = torch.from_numpy(batch.states).to(self.device, non_blocking=True)
        actions = torch.from_numpy(batch.actions).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(batch.rewards).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(batch.next_states).to(self.device, non_blocking=True)
        dones = torch.from_numpy(batch.dones).to(self.device, non_blocking=True)
        mask = torch.from_numpy(batch.mask).to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
            # Forward online sur la séquence complète. Hidden=None → zéro-init.
            q_pred_all, _ = self.online(states, None)
            # q_pred_all shape : (seq, batch, n_actions)
            # Gather sur l'action prise
            q_pred = q_pred_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)
            # q_pred shape : (seq, batch)

            with torch.no_grad():
                q_next_all, _ = self.target(next_states, None)
                q_next = q_next_all.max(dim=-1).values
                target_q = rewards + self.gamma * q_next * (1.0 - dones)

            # Huber loss element-wise, puis mask, puis moyenne sur vrais steps
            elem_loss = self.loss_fn(q_pred, target_q)
            masked_loss = elem_loss * mask
            n_valid = mask.sum().clamp(min=1.0)
            loss = masked_loss.sum() / n_valid

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
