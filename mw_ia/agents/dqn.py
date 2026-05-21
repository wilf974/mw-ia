"""Agent DQN — orchestre QNetwork, ReplayBuffer, Trainer, ε-greedy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mw_ia.agents.base import Agent
from mw_ia.config import DQNConfig
from mw_ia.envs.gridworld import GridWorld
from mw_ia.neural.network import QNetwork
from mw_ia.neural.replay_buffer import ReplayBuffer
from mw_ia.neural.trainer import DQNTrainer


class DQNAgent(Agent):
    """Vue état = one-hot indexant la cellule. Sortie = Q(s, a)."""

    def __init__(
        self,
        env: GridWorld,
        cfg: DQNConfig,
        *,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        wants_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if wants_cuda else "cpu")
        obs_dim = env.n_states
        self.online = QNetwork(obs_dim, env.n_actions, cfg.hidden_layers).to(self.device)
        self.target = QNetwork(obs_dim, env.n_actions, cfg.hidden_layers).to(self.device)
        self.trainer = DQNTrainer(
            self.online, self.target, lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
        )
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
        if (not greedy) and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.env.n_actions))
        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.online(x)
            return int(q.argmax(dim=1).item())

    def observe(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, done: bool,
    ) -> dict[str, float]:
        self.buffer.push(state, action, reward, next_state, done)
        self.global_step += 1
        metrics: dict[str, float] = {"epsilon": self.epsilon}
        if (
            len(self.buffer) >= self.cfg.min_replay_to_learn
            and self.global_step % self.cfg.train_every == 0
        ):
            batch = self.buffer.sample(self.cfg.batch_size)
            self.last_loss = self.trainer.step(batch)
            metrics["loss"] = self.last_loss
        if self.global_step % self.cfg.target_sync_steps == 0:
            self.trainer.sync_target()
            self.target_syncs += 1
        return metrics

    def learn(self, transition: object) -> dict[str, float]:
        raise NotImplementedError("Utiliser observe() pour DQN")

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
