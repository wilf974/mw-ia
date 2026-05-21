"""Replay buffer circulaire a echantillonnage uniforme."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    states: np.ndarray       # (B, obs_dim) float32
    actions: np.ndarray      # (B,) int64
    rewards: np.ndarray      # (B,) float32
    next_states: np.ndarray  # (B, obs_dim) float32
    dones: np.ndarray        # (B,) float32 (0/1)


class ReplayBuffer:
    """Buffer circulaire numpy - prepare pour futurs Prioritized ER."""

    def __init__(self, capacity: int, obs_dim: int, *, seed: int = 0) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._rng = np.random.default_rng(seed)
        self._states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self._idx
        self._states[i] = state
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._dones[i] = 1.0 if done else 0.0
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if self._size < batch_size:
            raise ValueError(
                f"buffer trop petit ({self._size}) pour batch={batch_size}"
            )
        idxs = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            states=self._states[idxs],
            actions=self._actions[idxs],
            rewards=self._rewards[idxs],
            next_states=self._next_states[idxs],
            dones=self._dones[idxs],
        )
