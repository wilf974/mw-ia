"""SnapshotTrajectoryStore -- sliding window FIFO de captures de trajectoires.

Voir spec V2-B1a : docs/superpowers/specs/2026-05-29-mw-ia-policy-snapshot-rehearsal-design.md

Invariant architectural central :
    Une fois capturee, une trajectoire snapshot N'EST JAMAIS modifiee,
    re-evaluee, re-encodee, ou re-rolled-out. Elle reste un temoin frozen
    de la politique au pic.

Storage : pre-alloue (n_windows * snapshot_size, max_steps, ...) arrays.
Filtre succes : terminated_last_step AND sum(rewards) > 0.
Sample : uniforme parmi les slots valides (pas de PER interne).
"""
from __future__ import annotations

from typing import Union

import numpy as np

from mw_ia.neural.sequence_buffer import BatchSeq, SequenceReplayBuffer


class SnapshotTrajectoryStore:
    """Stock frozen de trajectoires snapshot, sliding window FIFO."""

    def __init__(
        self,
        obs_dim: int,
        max_steps: int = 200,
        *,
        n_windows: int = 3,
        snapshot_size: int = 50,
        seed: int = 0,
    ) -> None:
        if obs_dim <= 0:
            raise ValueError(f"obs_dim doit etre > 0, recu {obs_dim}")
        if max_steps <= 0:
            raise ValueError(f"max_steps doit etre > 0, recu {max_steps}")
        if n_windows <= 0:
            raise ValueError(f"n_windows doit etre > 0, recu {n_windows}")
        if snapshot_size <= 0:
            raise ValueError(f"snapshot_size doit etre > 0, recu {snapshot_size}")

        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.n_windows = n_windows
        self.snapshot_size = snapshot_size
        self._rng = np.random.default_rng(seed)

        total_slots = n_windows * snapshot_size
        # Storage layout : slots [w*snapshot_size : (w+1)*snapshot_size] = window w
        self._states = np.zeros((total_slots, max_steps, obs_dim), dtype=np.float32)
        self._actions = np.zeros((total_slots, max_steps), dtype=np.int64)
        self._rewards = np.zeros((total_slots, max_steps), dtype=np.float32)
        self._next_states = np.zeros((total_slots, max_steps, obs_dim), dtype=np.float32)
        self._dones = np.zeros((total_slots, max_steps), dtype=np.float32)
        self._lengths = np.zeros(total_slots, dtype=np.int64)

        # Tracking
        self._window_sizes = np.zeros(n_windows, dtype=np.int64)  # combien remplis par window
        self._oldest_window_idx = 0  # window a ecraser au prochain capture si plein
        self._n_captures = 0

    def __len__(self) -> int:
        return int(self._window_sizes.sum())

    @property
    def n_captures(self) -> int:
        return self._n_captures

    @staticmethod
    def _is_successful(
        source: SequenceReplayBuffer, slot: int,
    ) -> bool:
        """Filtre succes : terminated_last_step AND total_reward > 0."""
        length = int(source._lengths[slot])
        if length == 0:
            return False
        terminated_at_end = source._dones[slot, length - 1] == 1.0
        total_reward = float(np.sum(source._rewards[slot, :length]))
        return terminated_at_end and total_reward > 0.0

    def capture_from(
        self,
        source_buffer: Union[SequenceReplayBuffer, "PrioritizedSequenceReplayBuffer"],
    ) -> int:
        """Extrait jusqu'a snapshot_size trajectoires successful recentes.

        Iterate source buffer en arriere depuis current_idx. Filtre success.
        Storage : remplit la prochaine window, ou ecrase oldest si sliding window plein.
        Copies tous les arrays (immutabilite garantie).

        Returns: nombre de trajectoires effectivement capturees.
        """
        # Determine la window cible
        if self._n_captures < self.n_windows:
            target_window = self._n_captures
        else:
            target_window = self._oldest_window_idx
            self._oldest_window_idx = (self._oldest_window_idx + 1) % self.n_windows

        # Reset le window slot (ecrasage propre)
        window_start = target_window * self.snapshot_size
        window_end = window_start + self.snapshot_size
        self._states[window_start:window_end] = 0.0
        self._actions[window_start:window_end] = 0
        self._rewards[window_start:window_end] = 0.0
        self._next_states[window_start:window_end] = 0.0
        self._dones[window_start:window_end] = 0.0
        self._lengths[window_start:window_end] = 0

        # Iterate source buffer en arriere depuis current_idx
        n_captured = 0
        src_size = source_buffer._size
        src_capacity = source_buffer.capacity
        if src_size == 0:
            self._window_sizes[target_window] = 0
            self._n_captures += 1
            return 0
        # current_idx pointe sur la prochaine ecriture, donc most recent = (current_idx - 1) mod capacity
        start_idx = (source_buffer._idx - 1) % src_capacity

        for offset in range(src_size):
            if n_captured >= self.snapshot_size:
                break
            slot_idx = (start_idx - offset) % src_capacity
            if not self._is_successful(source_buffer, slot_idx):
                continue
            # Copy (immutabilite : pas de reference)
            dest_slot = window_start + n_captured
            length = int(source_buffer._lengths[slot_idx])
            self._states[dest_slot, :length] = source_buffer._states[slot_idx, :length]
            self._actions[dest_slot, :length] = source_buffer._actions[slot_idx, :length]
            self._rewards[dest_slot, :length] = source_buffer._rewards[slot_idx, :length]
            self._next_states[dest_slot, :length] = source_buffer._next_states[slot_idx, :length]
            self._dones[dest_slot, :length] = source_buffer._dones[slot_idx, :length]
            self._lengths[dest_slot] = length
            n_captured += 1

        self._window_sizes[target_window] = n_captured
        self._n_captures += 1
        return n_captured

    def sample(self, batch_size: int, seq_len: int) -> BatchSeq:
        """Sample uniforme parmi les slots valides. Pattern V2-Y padding + mask."""
        total = int(self._window_sizes.sum())
        if total < batch_size:
            raise ValueError(
                f"snapshot store trop petit ({total}) pour batch={batch_size}"
            )
        if seq_len <= 0 or seq_len > self.max_steps:
            raise ValueError(
                f"seq_len {seq_len} hors ]0, {self.max_steps}]"
            )

        # Enumerate valid slots
        valid_slots = self._enumerate_valid_slots()
        chosen = self._rng.choice(valid_slots, size=batch_size, replace=False)

        # Build BatchSeq pattern V2-Y SequenceReplayBuffer.sample()
        states = np.zeros((seq_len, batch_size, self.obs_dim), dtype=np.float32)
        actions = np.zeros((seq_len, batch_size), dtype=np.int64)
        rewards = np.zeros((seq_len, batch_size), dtype=np.float32)
        next_states = np.zeros((seq_len, batch_size, self.obs_dim), dtype=np.float32)
        dones = np.zeros((seq_len, batch_size), dtype=np.float32)
        mask = np.zeros((seq_len, batch_size), dtype=np.float32)

        for b, slot in enumerate(chosen):
            length = int(self._lengths[slot])
            max_offset = max(0, length - seq_len)
            offset = int(self._rng.integers(0, max_offset + 1))
            real_len = min(seq_len, length - offset)
            states[:real_len, b] = self._states[slot, offset:offset + real_len]
            actions[:real_len, b] = self._actions[slot, offset:offset + real_len]
            rewards[:real_len, b] = self._rewards[slot, offset:offset + real_len]
            next_states[:real_len, b] = self._next_states[slot, offset:offset + real_len]
            dones[:real_len, b] = self._dones[slot, offset:offset + real_len]
            mask[:real_len, b] = 1.0

        return BatchSeq(
            states=states, actions=actions, rewards=rewards,
            next_states=next_states, dones=dones, mask=mask,
        )

    def _enumerate_valid_slots(self) -> np.ndarray:
        """Liste tous les slot indices remplis (windows actives)."""
        slots = []
        for w in range(self.n_windows):
            n = int(self._window_sizes[w])
            start = w * self.snapshot_size
            for k in range(n):
                slots.append(start + k)
        return np.array(slots, dtype=np.int64)
