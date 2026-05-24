"""ConvRecurrentDQNAgent — V2-ZY combo CNN + LSTM + Double DQN.

Hidden state runtime maintenu entre act() consécutifs (pattern V2-Y).
Forward LSTM appliqué AUSSI en eps-greedy random.

Observations 3D `(in_channels, rows, cols)` flatten 1D pour storage dans
SequenceReplayBuffer V2-Y. Network reshape interne 1D → 3D pour Conv block.

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-cnn-lstm-double-dqn-design.md §2
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mw_ia.agents.base import Agent
from mw_ia.config import ConvRecurrentDQNConfig
from mw_ia.neural.conv_recurrent import ConvRecurrentQNetwork
from mw_ia.neural.recurrent_trainer import RecurrentDQNTrainer
from mw_ia.neural.sequence_buffer import SequenceReplayBuffer


class ConvRecurrentDQNAgent(Agent):
    """Agent V2-ZY combinant perception spatiale (Conv) + mémoire (LSTM) + Double DQN."""

    def __init__(
        self,
        *,
        in_channels: int,
        rows: int,
        cols: int,
        n_actions: int,
        cfg: ConvRecurrentDQNConfig,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        self.n_actions = n_actions
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        wants_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if wants_cuda else "cpu")
        self.online = ConvRecurrentQNetwork(
            in_channels=in_channels, rows=rows, cols=cols, n_actions=n_actions,
            conv_channels=cfg.conv_channels, kernel_size=cfg.kernel_size,
            padding=cfg.padding, lstm_hidden=cfg.lstm_hidden,
        ).to(self.device)
        self.target = ConvRecurrentQNetwork(
            in_channels=in_channels, rows=rows, cols=cols, n_actions=n_actions,
            conv_channels=cfg.conv_channels, kernel_size=cfg.kernel_size,
            padding=cfg.padding, lstm_hidden=cfg.lstm_hidden,
        ).to(self.device)
        self.trainer = RecurrentDQNTrainer(
            self.online, self.target,
            lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
            double_dqn=cfg.double_dqn,
            polyak_tau=cfg.polyak_tau,
        )
        obs_dim_flat = in_channels * rows * cols
        self.buffer = SequenceReplayBuffer(
            cfg.replay_capacity, obs_dim_flat, cfg.max_steps_per_episode, seed=seed,
        )
        self.global_step: int = 0
        self.target_syncs: int = 0
        self.last_loss: float | None = None
        self._hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self._episode_trajectory: list[tuple] = []

    @property
    def epsilon(self) -> float:
        if self.cfg.epsilon_decay_steps <= 0:
            return self.cfg.epsilon_end
        frac = min(1.0, self.global_step / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def reset_hidden(self) -> None:
        """Reset le hidden state LSTM."""
        self._hidden_state = None

    def begin_episode(self) -> None:
        """Reset hidden + vide la trajectoire courante. Compat V2-V duck-typing."""
        self._hidden_state = None
        self._episode_trajectory = []

    def act(self, state: np.ndarray, *, greedy: bool = False) -> int:
        """Forward LSTM toujours appliqué (maintient hidden state runtime)."""
        assert state.shape == (self.in_channels, self.rows, self.cols), (
            f"state {state.shape} != ({self.in_channels}, {self.rows}, {self.cols})"
        )
        with torch.no_grad():
            x = torch.from_numpy(state.flatten()).float().to(self.device)
            x = x.unsqueeze(0).unsqueeze(0)
            q, new_hidden = self.online(x, self._hidden_state)
            self._hidden_state = new_hidden
        if (not greedy) and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(q.argmax(dim=-1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        """Accumule la transition. Train step PAS déclenché ici (cf. end_episode())."""
        assert state.shape == (self.in_channels, self.rows, self.cols)
        assert next_state.shape == (self.in_channels, self.rows, self.cols)
        self._episode_trajectory.append(
            (state.flatten(), action, reward, next_state.flatten(), done)
        )
        self.global_step += 1
        return {"epsilon": self.epsilon}

    def end_episode(self) -> dict[str, float]:
        """Push trajectoire dans buffer + train_steps_per_episode batches BPTT."""
        if self._episode_trajectory:
            self.buffer.push_trajectory(self._episode_trajectory)
        metrics: dict[str, float] = {"epsilon": self.epsilon}
        train_threshold = max(self.cfg.min_episodes_to_learn, self.cfg.batch_size)
        if len(self.buffer) >= train_threshold:
            losses: list[float] = []
            for _ in range(self.cfg.train_steps_per_episode):
                batch = self.buffer.sample(
                    batch_size=self.cfg.batch_size, seq_len=self.cfg.sequence_length,
                )
                losses.append(self.trainer.step(batch))
            if losses:
                self.last_loss = sum(losses) / len(losses)
                metrics["loss"] = self.last_loss
        # V2-U : skip hard sync périodique si Polyak activé.
        if self.cfg.polyak_tau == 0.0:
            if self.global_step // self.cfg.target_sync_steps > self.target_syncs:
                self.trainer.sync_target()
                self.target_syncs += 1
        return metrics

    def learn(self, transition: Any) -> dict[str, float]:
        raise NotImplementedError("Utiliser observe() + end_episode() pour DRQN")

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
