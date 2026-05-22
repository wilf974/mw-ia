"""RecurrentDQNAgent — DRQN avec LSTM, hidden state runtime maintenu par épisode.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md §2.1
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mw_ia.agents.base import Agent
from mw_ia.config import DRQNConfig
from mw_ia.neural.recurrent import RecurrentQNetwork
from mw_ia.neural.recurrent_trainer import RecurrentDQNTrainer
from mw_ia.neural.sequence_buffer import SequenceReplayBuffer


class RecurrentDQNAgent(Agent):
    """DRQN avec LSTM. Hidden state runtime maintenu entre act() consécutifs.

    Différences clés avec DQNAgent V1 :
    - Forward 1 timestep dans act() avec hidden state runtime
    - reset_hidden() / begin_episode() appelés par le runner à chaque épisode
    - observe() accumule transitions dans une trajectoire courante (PAS de train step)
    - end_episode() push la trajectoire dans le buffer + déclenche les train steps
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: DRQNConfig,
        *,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        wants_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if wants_cuda else "cpu")
        self.online = RecurrentQNetwork(
            obs_dim, n_actions, cfg.fc_hidden, cfg.lstm_hidden
        ).to(self.device)
        self.target = RecurrentQNetwork(
            obs_dim, n_actions, cfg.fc_hidden, cfg.lstm_hidden
        ).to(self.device)
        self.trainer = RecurrentDQNTrainer(
            self.online, self.target, lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
        )
        self.buffer = SequenceReplayBuffer(
            cfg.replay_capacity, obs_dim, cfg.max_steps_per_episode, seed=seed,
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
        """Reset le hidden state runtime. Appelé par le runner au début de chaque épisode."""
        self._hidden_state = None

    def begin_episode(self) -> None:
        """Vide la trajectoire de l'épisode courant. Appelé par le runner après reset_hidden()."""
        self._episode_trajectory = []

    def act(self, state: np.ndarray, *, greedy: bool = False) -> int:
        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
            # x shape : (seq=1, batch=1, obs_dim)
            # Forward TOUJOURS pour maintenir le hidden state runtime continu
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
        """Accumule la transition dans la trajectoire courante.

        Train step PAS déclenché ici (cf. end_episode()).
        """
        self._episode_trajectory.append((state, action, reward, next_state, done))
        self.global_step += 1
        return {"epsilon": self.epsilon}

    def end_episode(self) -> dict[str, float]:
        """Push la trajectoire dans le buffer + train_steps_per_episode batches.

        Doit être appelé par le runner après la boucle step de l'épisode.
        """
        if self._episode_trajectory:
            self.buffer.push_trajectory(self._episode_trajectory)
        metrics: dict[str, float] = {"epsilon": self.epsilon}
        if len(self.buffer) >= self.cfg.min_episodes_to_learn:
            losses: list[float] = []
            for _ in range(self.cfg.train_steps_per_episode):
                batch = self.buffer.sample(
                    batch_size=self.cfg.batch_size, seq_len=self.cfg.sequence_length,
                )
                losses.append(self.trainer.step(batch))
            if losses:
                self.last_loss = sum(losses) / len(losses)
                metrics["loss"] = self.last_loss
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
