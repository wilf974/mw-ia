"""SequenceReplayBuffer — buffer circulaire de trajectoires complètes pour DRQN.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md §2.1

ATTENTION : capacity = nombre de TRAJECTOIRES (pas transitions, contrairement
au ReplayBuffer V1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BatchSeq:
    """Batch d'entraînement DRQN. Convention batch_first=False (seq en premier)."""

    states: np.ndarray       # (seq, batch, obs_dim) float32
    actions: np.ndarray      # (seq, batch) int64
    rewards: np.ndarray      # (seq, batch) float32
    next_states: np.ndarray  # (seq, batch, obs_dim) float32
    dones: np.ndarray        # (seq, batch) float32 (0/1)
    mask: np.ndarray         # (seq, batch) float32 — 1 pour vrais steps, 0 pour padding


class SequenceReplayBuffer:
    """Buffer circulaire de trajectoires complètes.

    Capacity = nombre de trajectoires. Chaque trajectoire = liste de
    (state, action, reward, next_state, done), longueur ∈ [1, max_steps].

    sample(batch_size, seq_len) : tire B trajectoires aléatoires avec remise,
    pour chacune tire un offset aléatoire et extrait une fenêtre seq_len.
    Padding zéros + mask=0 si trajectoire plus courte que seq_len.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        max_steps: int = 200,
        *,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity doit être > 0, reçu {capacity}")
        if obs_dim <= 0:
            raise ValueError(f"obs_dim doit être > 0, reçu {obs_dim}")
        if max_steps <= 0:
            raise ValueError(f"max_steps doit être > 0, reçu {max_steps}")
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        # Pré-allocation : (capacity, max_steps, ...) — réutilisé en circulaire.
        self._states = np.zeros((capacity, max_steps, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, max_steps), dtype=np.int64)
        self._rewards = np.zeros((capacity, max_steps), dtype=np.float32)
        self._next_states = np.zeros((capacity, max_steps, obs_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, max_steps), dtype=np.float32)
        self._lengths = np.zeros(capacity, dtype=np.int64)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push_trajectory(self, trajectory: list[tuple]) -> None:
        """Ajoute une trajectoire de longueur ∈ [1, max_steps]."""
        n = len(trajectory)
        if not (1 <= n <= self.max_steps):
            raise ValueError(
                f"longueur trajectoire {n} hors [1, {self.max_steps}]"
            )
        i = self._idx
        # Reset les anciennes valeurs au-delà de la nouvelle longueur (sécurité)
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
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
