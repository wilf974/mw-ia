"""RecurrentDQNTrainer — boucle d'optimisation BPTT pour DRQN.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md §2.1

DRQN simple (Hausknecht & Stone 2015) : hidden state zero-init au début de
chaque séquence de training (pas de burn-in en V2-Y MVP).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from mw_ia.neural.sequence_buffer import BatchSeq

if TYPE_CHECKING:
    from mw_ia.neural.recurrent import RecurrentQNetwork


class RecurrentDQNTrainer:
    """Encapsule online net + target net + optimizer + AMP, BPTT 32 steps avec mask."""

    def __init__(
        self,
        online: nn.Module,  # RecurrentQNetwork ou ConvRecurrentQNetwork
        target: nn.Module,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cuda",
        use_amp: bool = True,
        double_dqn: bool = False,
        polyak_tau: float = 0.0,
    ) -> None:
        self.online = online
        self.target = target
        self.gamma = gamma
        self.double_dqn = double_dqn
        self.polyak_tau = polyak_tau
        self.device = torch.device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        # SmoothL1Loss avec reduction='none' pour pouvoir appliquer le mask manuellement
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self._scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.sync_target()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def polyak_update(self, tau: float) -> None:
        """Soft update target ← τ × online + (1−τ) × target, in-place.

        Voir spec V2-U : docs/superpowers/specs/2026-05-24-mw-ia-polyak-soft-target-design.md
        """
        with torch.no_grad():
            for p_target, p_online in zip(
                self.target.parameters(), self.online.parameters()
            ):
                p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)

    def step(self, batch: BatchSeq) -> float:
        """V2-Y baseline : sans IS, retourne loss seul (signature stricte)."""
        loss, _ = self._step_impl(batch, weights=None, eta=0.0)
        return loss

    def step_with_priorities(
        self,
        batch: BatchSeq,
        weights: np.ndarray,
        eta: float = 0.9,
    ) -> tuple[float, np.ndarray]:
        """V2-B0 : sample IS-weighted + retourne (loss, td_errors agréges R2D2).

        Voir spec V2-B0 : docs/superpowers/specs/2026-05-26-mw-ia-per-trajectory.md
        """
        loss, td_errors = self._step_impl(batch, weights=weights, eta=eta)
        assert td_errors is not None
        return loss, td_errors

    def _step_impl(
        self,
        batch: BatchSeq,
        weights: np.ndarray | None = None,
        eta: float = 0.9,
    ) -> tuple[float, np.ndarray | None]:
        """Pipeline unifié BPTT.

        weights=None → V2-Y baseline strict (loss masquée uniforme, td_errors=None).
        weights fourni → IS-weighted loss + retourne td_errors agrégés par trajectoire
        selon la formule R2D2 : priority_b = eta × max + (1 − eta) × mean.
        """
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
                if self.double_dqn:
                    # V2-W branche appliquée au BPTT recurrent :
                    # online sélectionne, target évalue (Hasselt 2015)
                    q_online_next_all, _ = self.online(next_states, None)
                    next_actions = q_online_next_all.argmax(dim=-1)
                    q_target_all, _ = self.target(next_states, None)
                    q_next = q_target_all.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
                else:
                    # V2-Y baseline DQN classique
                    q_next_all, _ = self.target(next_states, None)
                    q_next = q_next_all.max(dim=-1).values
                target_q = rewards + self.gamma * q_next * (1.0 - dones)

            # Huber loss element-wise, puis mask + (optionnel) IS weights
            elem_loss = self.loss_fn(q_pred, target_q)
            if weights is None:
                masked_loss = elem_loss * mask
            else:
                w = torch.from_numpy(weights).to(
                    self.device, non_blocking=True
                ).unsqueeze(0)  # (1, batch) broadcast vers (seq, batch)
                masked_loss = elem_loss * mask * w
            n_valid = mask.sum().clamp(min=1.0)
            loss = masked_loss.sum() / n_valid

        # Backward + grad clip + optimizer step (V2-Y inchangé)
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

        # V2-U : soft Polyak update à chaque train_step si tau > 0
        if self.polyak_tau > 0.0:
            self.polyak_update(self.polyak_tau)

        loss_value = float(loss.detach().item())

        # V2-B0 : aggregation R2D2 si PER
        if weights is None:
            return loss_value, None

        with torch.no_grad():
            td_step = (target_q - q_pred).detach().abs()        # (seq, batch)
            masked_td = td_step * mask                          # (seq, batch)
            max_per_traj = masked_td.max(dim=0).values          # (batch,)
            sum_per_traj = masked_td.sum(dim=0)                 # (batch,)
            length_per_traj = mask.sum(dim=0).clamp(min=1.0)    # (batch,)
            mean_per_traj = sum_per_traj / length_per_traj
            priorities = eta * max_per_traj + (1.0 - eta) * mean_per_traj

        return loss_value, priorities.cpu().numpy().astype(np.float32)
