"""RecurrentDQNAgent — DRQN avec LSTM, hidden state runtime maintenu par épisode.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md §2.1
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mw_ia.agents.base import Agent
from mw_ia.config import DRQNConfig
from mw_ia.neural.prioritized_sequence_buffer import (
    BetaScheduler,
    PrioritizedSequenceReplayBuffer,
)
from mw_ia.neural.recurrent import RecurrentQNetwork
from mw_ia.neural.recurrent_trainer import RecurrentDQNTrainer
from mw_ia.neural.sequence_buffer import BatchSeq, SequenceReplayBuffer, concat_batchseq
from mw_ia.training.snapshot_store import SnapshotTrajectoryStore


@dataclass
class _TrainingBatch:
    """Conteneur batch + weights + tree_indices pour _sample_training_batch."""
    batch: BatchSeq
    weights: "np.ndarray | None"
    tree_indices: "np.ndarray | None"


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
            polyak_tau=cfg.polyak_tau,
        )
        if cfg.per_enabled:
            self.buffer: SequenceReplayBuffer | PrioritizedSequenceReplayBuffer = (
                PrioritizedSequenceReplayBuffer(
                    cfg.replay_capacity, obs_dim, cfg.max_steps_per_episode,
                    alpha=cfg.per_alpha, epsilon=cfg.per_epsilon, seed=seed,
                )
            )
            self._beta_scheduler: BetaScheduler | None = BetaScheduler(
                cfg.per_beta_start, cfg.per_beta_end, cfg.episodes,
            )
        else:
            self.buffer = SequenceReplayBuffer(
                cfg.replay_capacity, obs_dim, cfg.max_steps_per_episode, seed=seed,
            )
            self._beta_scheduler = None
        if cfg.b1a_enabled:
            self.snapshot_store: SnapshotTrajectoryStore | None = SnapshotTrajectoryStore(
                obs_dim=obs_dim,
                max_steps=cfg.max_steps_per_episode,
                n_windows=cfg.b1a_n_windows,
                snapshot_size=cfg.b1a_snapshot_size,
                seed=seed,
            )
        else:
            self.snapshot_store = None
        self._episode_count: int = 0
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

    def on_new_best(self) -> int:
        """Hook appele par le runner quand BestCheckpointTracker detecte un nouveau peak.

        Si B1a active : capture jusqu'a snapshot_size trajectoires successful recentes
        depuis self.buffer dans self.snapshot_store (sliding window N).
        Si B1a desactive : no-op.

        Returns: nombre de trajectoires effectivement capturees.
        """
        if not self.cfg.b1a_enabled:
            return 0
        assert self.snapshot_store is not None
        return self.snapshot_store.capture_from(self.buffer)

    def _sample_training_batch(self) -> _TrainingBatch:
        """Build batch combine main + snapshot avec mix 80/20 si B1a actif.

        Gere les 4 combinaisons PER x B1a. Retourne (batch, weights or None,
        tree_indices or None for update_priorities main portion).
        """
        B = self.cfg.batch_size
        L = self.cfg.sequence_length

        snapshot_B = int(B * self.cfg.b1a_mix_ratio) if self.cfg.b1a_enabled else 0
        b1a_active = (
            self.cfg.b1a_enabled
            and snapshot_B > 0
            and self.snapshot_store is not None
            and len(self.snapshot_store) >= snapshot_B
        )
        main_B = B - snapshot_B if b1a_active else B

        # --- Sample main portion ---
        if self.cfg.per_enabled:
            assert self._beta_scheduler is not None
            beta = self._beta_scheduler.beta(self._episode_count)
            prio = self.buffer.sample(main_B, L, beta=beta)  # type: ignore[call-arg]
            main_batch = prio.batch
            main_weights = prio.weights
            tree_indices = prio.tree_indices
        else:
            main_batch = self.buffer.sample(main_B, L)
            main_weights = None
            tree_indices = None

        if not b1a_active:
            return _TrainingBatch(batch=main_batch, weights=main_weights, tree_indices=tree_indices)

        # --- Sample snapshot portion ---
        assert self.snapshot_store is not None
        snapshot_batch = self.snapshot_store.sample(snapshot_B, L)
        combined_batch = concat_batchseq(main_batch, snapshot_batch)

        if self.cfg.per_enabled:
            snapshot_weights = np.ones(snapshot_B, dtype=np.float32)
            combined_weights = np.concatenate([main_weights, snapshot_weights])
            return _TrainingBatch(
                batch=combined_batch, weights=combined_weights, tree_indices=tree_indices,
            )
        else:
            return _TrainingBatch(batch=combined_batch, weights=None, tree_indices=None)

    def end_episode(self) -> dict[str, float]:
        """Push la trajectoire dans le buffer + train_steps_per_episode batches.

        Doit être appelé par le runner après la boucle step de l'épisode.
        V2-B0 : branche PER (sample IS-weighted + update_priorities) si
        cfg.per_enabled, sinon V2-Y baseline strict.
        V2-B1a : mixe snapshot store (frozen success trajectories) dans batch
        si cfg.b1a_enabled ET snapshot rempli (>= snapshot_B).
        """
        if self._episode_trajectory:
            self.buffer.push_trajectory(self._episode_trajectory)
        self._episode_count += 1
        metrics: dict[str, float] = {"epsilon": self.epsilon}
        if len(self.buffer) >= max(self.cfg.min_episodes_to_learn, self.cfg.batch_size):
            losses: list[float] = []
            for _ in range(self.cfg.train_steps_per_episode):
                tb = self._sample_training_batch()
                if tb.weights is not None:
                    loss, td_errors = self.trainer.step_with_priorities(
                        tb.batch, tb.weights, eta=self.cfg.per_eta,
                    )
                    if tb.tree_indices is not None:
                        main_B = len(tb.tree_indices)
                        self.buffer.update_priorities(  # type: ignore[union-attr]
                            tb.tree_indices, td_errors[:main_B],
                        )
                    losses.append(loss)
                else:
                    loss = self.trainer.step(tb.batch)
                    losses.append(loss)
            if losses:
                self.last_loss = sum(losses) / len(losses)
                metrics["loss"] = self.last_loss
            if self.cfg.per_enabled and self._beta_scheduler is not None:
                metrics["per_beta"] = self._beta_scheduler.beta(self._episode_count)
        # V2-U : skip hard sync périodique si Polyak activé (le trainer.step()
        # appelle déjà polyak_update à chaque train_step).
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
