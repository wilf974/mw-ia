"""Prioritized Experience Replay au niveau trajectoire pour V2-ZY+Polyak.

Voir spec : docs/superpowers/specs/2026-05-26-mw-ia-per-trajectory-design.md

Contient :
- BetaScheduler : annealing lineaire beta_start -> beta_end pour IS correction
- PrioritizedSequenceReplayBuffer + PrioritizedBatchSeq : sum tree + IS weights
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mw_ia.neural.sequence_buffer import BatchSeq
from mw_ia.neural.sum_tree import SumTree


class BetaScheduler:
    """Annealing linéaire β_start → β_end sur total_episodes.

    β contrôle l'intensité de la correction Importance Sampling. β=0 = aucune
    correction, β=1 = correction complète. Standard Schaul 2015 : β annealé
    progressivement pour stabiliser l'apprentissage initial puis converger
    vers une estimation non-biaisée.
    """

    def __init__(self, beta_start: float, beta_end: float, total_episodes: int) -> None:
        if not (0.0 <= beta_start <= 1.0):
            raise ValueError(
                f"beta_start doit etre dans [0, 1], recu {beta_start}"
            )
        if not (0.0 <= beta_end <= 1.0):
            raise ValueError(
                f"beta_end doit etre dans [0, 1], recu {beta_end}"
            )
        if total_episodes <= 0:
            raise ValueError(
                f"total_episodes doit etre > 0, recu {total_episodes}"
            )
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_episodes = total_episodes

    def beta(self, episode: int) -> float:
        """Retourne beta a l'episode donne. Clamp aux extremites."""
        if episode <= 0:
            return self.beta_start
        if episode >= self.total_episodes:
            return self.beta_end
        progress = episode / self.total_episodes
        return self.beta_start + (self.beta_end - self.beta_start) * progress


@dataclass
class PrioritizedBatchSeq:
    """Batch enrichi PER : BatchSeq V2-Y + IS weights + tree indices."""

    batch: BatchSeq              # BatchSeq standard V2-Y (states/.../mask)
    weights: np.ndarray          # (B,) float32 — IS weights normalises par max(w)
    tree_indices: np.ndarray     # (B,) int64 — leaf indices pour update_priorities


class PrioritizedSequenceReplayBuffer:
    """Buffer circulaire de trajectoires avec PER (Schaul 2015) trajectory-level.

    Capacity = nombre de trajectoires (coherent SequenceReplayBuffer V2-Y).
    Priorite par trajectoire stockee comme `(|td_error| + epsilon)^alpha`.
    Sampling stratifie sum tree + IS weights normalises par max(w).
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        max_steps: int = 200,
        *,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity doit etre > 0, recu {capacity}")
        if obs_dim <= 0:
            raise ValueError(f"obs_dim doit etre > 0, recu {obs_dim}")
        if max_steps <= 0:
            raise ValueError(f"max_steps doit etre > 0, recu {max_steps}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha doit etre dans [0, 1], recu {alpha}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon doit etre > 0, recu {epsilon}")

        self.capacity = capacity
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)

        # Storage identique a SequenceReplayBuffer V2-Y
        self._states = np.zeros((capacity, max_steps, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, max_steps), dtype=np.int64)
        self._rewards = np.zeros((capacity, max_steps), dtype=np.float32)
        self._next_states = np.zeros((capacity, max_steps, obs_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, max_steps), dtype=np.float32)
        self._lengths = np.zeros(capacity, dtype=np.int64)
        self._idx = 0
        self._size = 0

        # PER : sum tree + max priority tracker
        self._sum_tree = SumTree(capacity)
        self._max_priority = 1.0  # Greedy init pour nouvelles trajectoires

    def __len__(self) -> int:
        return self._size

    def push_trajectory(self, trajectory: list[tuple]) -> None:
        """Stocke trajectoire (pattern V2-Y) + assigne priorite = max courante."""
        n = len(trajectory)
        if not (1 <= n <= self.max_steps):
            raise ValueError(
                f"longueur trajectoire {n} hors [1, {self.max_steps}]"
            )
        i = self._idx
        # Reset slot
        self._states[i, :, :] = 0.0
        self._actions[i, :] = 0
        self._rewards[i, :] = 0.0
        self._next_states[i, :, :] = 0.0
        self._dones[i, :] = 0.0
        for t, (s, a, r, sp, d) in enumerate(trajectory):
            self._states[i, t] = s
            self._actions[i, t] = a
            self._rewards[i, t] = r
            self._next_states[i, t] = sp
            self._dones[i, t] = 1.0 if d else 0.0
        self._lengths[i] = n
        # Greedy init : nouvelle trajectoire recoit la priorite max courante
        self._sum_tree.update(i, self._max_priority)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int, beta: float) -> PrioritizedBatchSeq:
        """Sampling stratifie sum tree + extraction fenetre + IS weights."""
        if self._size < batch_size:
            raise ValueError(
                f"buffer trop petit ({self._size}) pour batch={batch_size}"
            )
        if seq_len <= 0 or seq_len > self.max_steps:
            raise ValueError(
                f"seq_len {seq_len} hors ]0, {self.max_steps}]"
            )

        total = self._sum_tree.total()
        # Sampling stratifie : segments egaux sur [0, total]
        segment = total / batch_size
        traj_idxs = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        for b in range(batch_size):
            low = segment * b
            high = segment * (b + 1)
            value = self._rng.uniform(low, high)
            leaf_idx, prio = self._sum_tree.find(value)
            traj_idxs[b] = leaf_idx
            priorities[b] = prio

        # Construction BatchSeq (offset aleatoire + padding + mask, pattern V2-Y)
        states = np.zeros((seq_len, batch_size, self.obs_dim), dtype=np.float32)
        actions = np.zeros((seq_len, batch_size), dtype=np.int64)
        rewards = np.zeros((seq_len, batch_size), dtype=np.float32)
        next_states = np.zeros((seq_len, batch_size, self.obs_dim), dtype=np.float32)
        dones = np.zeros((seq_len, batch_size), dtype=np.float32)
        mask = np.zeros((seq_len, batch_size), dtype=np.float32)

        for b, traj_i in enumerate(traj_idxs):
            length = int(self._lengths[traj_i])
            max_offset = max(0, length - seq_len)
            offset = int(self._rng.integers(0, max_offset + 1))
            real_len = min(seq_len, length - offset)
            states[:real_len, b] = self._states[traj_i, offset:offset + real_len]
            actions[:real_len, b] = self._actions[traj_i, offset:offset + real_len]
            rewards[:real_len, b] = self._rewards[traj_i, offset:offset + real_len]
            next_states[:real_len, b] = self._next_states[traj_i, offset:offset + real_len]
            dones[:real_len, b] = self._dones[traj_i, offset:offset + real_len]
            mask[:real_len, b] = 1.0

        batch_seq = BatchSeq(
            states=states, actions=actions, rewards=rewards,
            next_states=next_states, dones=dones, mask=mask,
        )

        # IS weights : w_i = (1/N * 1/P_i)^beta, normalises par max(w)
        # P_i = priority_i / total
        # Pour eviter overflow numerique si priority tres petite, clamp epsilon-level
        probs = np.maximum(priorities / max(total, 1e-12), 1e-12)
        weights = (1.0 / (self._size * probs)) ** beta
        weights = weights / weights.max()
        weights = weights.astype(np.float32)

        return PrioritizedBatchSeq(
            batch=batch_seq,
            weights=weights,
            tree_indices=traj_idxs,
        )

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """Met a jour les priorites : new_priority = (|td_error| + epsilon)^alpha."""
        if tree_indices.shape != td_errors.shape:
            raise ValueError(
                f"tree_indices {tree_indices.shape} != td_errors {td_errors.shape}"
            )
        # Formule Schaul : (|td| + eps)^alpha
        new_priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for leaf_idx, prio in zip(tree_indices, new_priorities):
            self._sum_tree.update(int(leaf_idx), float(prio))
            if prio > self._max_priority:
                self._max_priority = float(prio)
