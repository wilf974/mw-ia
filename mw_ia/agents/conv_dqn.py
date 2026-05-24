"""ConvDQNAgent — DQN feedforward à perception spatiale (V2-Z).

Diffère de DQNAgent V1 par :
- Réseau ConvQNetwork (Conv2d) au lieu de QNetwork (MLP)
- Observations d'entrée shape (3, R, C) au lieu de (R*C,)
- Réutilise ReplayBuffer V1 inchangé via flatten/reshape autour de push/sample.
  Le buffer stocke des np.ndarray 1D de dim `in_channels * rows * cols` ;
  le train_step reshape en (B, C, R, C) avant le forward.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-cnn-perception-design.md §3
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from mw_ia.config import ConvDQNConfig
from mw_ia.neural.conv_network import ConvQNetwork
from mw_ia.neural.replay_buffer import Batch, ReplayBuffer


class _ConvDQNTrainer:
    """Trainer Huber + Adam + AMP + grad clip pour ConvQNetwork.

    Variante du V1 DQNTrainer qui reshape les obs flat en (B, C, R, C) avant
    le forward. Pattern AMP/grad-clip strictement identique pour rester
    cohérent avec V1/V2-X.
    """

    def __init__(
        self,
        online: ConvQNetwork,
        target: ConvQNetwork,
        *,
        in_channels: int,
        rows: int,
        cols: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cuda",
        use_amp: bool = True,
        double_dqn: bool = True,
        polyak_tau: float = 0.0,
    ) -> None:
        self.online = online
        self.target = target
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.device = torch.device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.double_dqn = double_dqn
        self.polyak_tau = polyak_tau
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
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

    def step(self, batch: Batch) -> float:
        B = batch.states.shape[0]
        shape = (B, self.in_channels, self.rows, self.cols)
        states = (
            torch.from_numpy(batch.states)
            .to(self.device, non_blocking=True)
            .view(*shape)
        )
        next_states = (
            torch.from_numpy(batch.next_states)
            .to(self.device, non_blocking=True)
            .view(*shape)
        )
        actions = torch.from_numpy(batch.actions).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(batch.rewards).to(self.device, non_blocking=True)
        dones = torch.from_numpy(batch.dones).to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
            q_pred = self.online(states).gather(1, actions.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                if self.double_dqn:
                    # V2-W : online sélectionne, target évalue (Hasselt 2015)
                    next_actions = self.online(next_states).argmax(dim=1)
                    q_next = self.target(next_states).gather(
                        1, next_actions.view(-1, 1)
                    ).squeeze(1)
                else:
                    # V2-Z baseline : target sélectionne ET évalue (DQN classique)
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
        # V2-U : soft Polyak update à chaque train_step si tau > 0
        if self.polyak_tau > 0.0:
            self.polyak_update(self.polyak_tau)
        return float(loss.detach().item())


class ConvDQNAgent:
    """DQN à perception spatiale (Conv2d). Contrat compatible avec runner."""

    def __init__(
        self,
        *,
        in_channels: int,
        rows: int,
        cols: int,
        n_actions: int,
        cfg: ConvDQNConfig,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        wants_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if wants_cuda else "cpu")
        self.online = ConvQNetwork(
            in_channels=in_channels, rows=rows, cols=cols, n_actions=n_actions,
            conv_channels=cfg.conv_channels, kernel_size=cfg.kernel_size,
            padding=cfg.padding, fc_hidden=cfg.fc_hidden,
        ).to(self.device)
        self.target = ConvQNetwork(
            in_channels=in_channels, rows=rows, cols=cols, n_actions=n_actions,
            conv_channels=cfg.conv_channels, kernel_size=cfg.kernel_size,
            padding=cfg.padding, fc_hidden=cfg.fc_hidden,
        ).to(self.device)
        self.trainer = _ConvDQNTrainer(
            self.online, self.target,
            in_channels=in_channels, rows=rows, cols=cols,
            lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
            double_dqn=cfg.double_dqn,
            polyak_tau=cfg.polyak_tau,
        )
        obs_dim = in_channels * rows * cols
        self.buffer = ReplayBuffer(cfg.replay_capacity, obs_dim, seed=seed)
        self.global_step: int = 0
        self.target_syncs: int = 0
        self.last_loss: float | None = None

    @property
    def epsilon(self) -> float:
        if self.cfg.epsilon_decay_steps <= 0:
            return self.cfg.epsilon_end
        frac = min(1.0, self.global_step / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, state: np.ndarray, *, greedy: bool = False) -> int:
        assert state.shape == (self.in_channels, self.rows, self.cols), (
            f"state {state.shape} != expected ({self.in_channels}, {self.rows}, {self.cols})"
        )
        if (not greedy) and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.online(x)
            return int(q.argmax(dim=1).item())

    def observe(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, done: bool,
    ) -> dict[str, float]:
        assert state.shape == (self.in_channels, self.rows, self.cols)
        assert next_state.shape == (self.in_channels, self.rows, self.cols)
        self.buffer.push(state.flatten(), action, reward, next_state.flatten(), done)
        self.global_step += 1
        metrics: dict[str, float] = {"epsilon": self.epsilon}
        train_threshold = max(self.cfg.min_replay_to_learn, self.cfg.batch_size)
        if (
            len(self.buffer) >= train_threshold
            and self.global_step % self.cfg.train_every == 0
        ):
            batch = self.buffer.sample(self.cfg.batch_size)
            self.last_loss = self.trainer.step(batch)
            metrics["loss"] = self.last_loss
        if self.global_step % self.cfg.target_sync_steps == 0:
            self.trainer.sync_target()
            self.target_syncs += 1
        return metrics

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "global_step": self.global_step,
                "cfg": self.cfg.__dict__,
            },
            p,
        )

    def load(self, path: str | Path) -> None:
        data = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data["target"])
        self.global_step = int(data.get("global_step", 0))
