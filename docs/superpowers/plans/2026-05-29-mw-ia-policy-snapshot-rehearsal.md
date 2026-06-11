# MW_IA V2-B1a Policy Snapshot Rehearsal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter Policy Snapshot Rehearsal (B1a) opt-in via `b1a_enabled: bool = False` sur les agents récurrents V2-Y et V2-ZY. Capture 50 trajectoires successful récentes du buffer au moment du best detected par V2-V (sliding window N=3), inject 20% dans chaque batch d'entraînement (mix avec main buffer). Hypothèse : préserver les trajectoires near-frontier suffit à éviter le forgetting et améliorer V2-ZY+Polyak en 15×15.

**Architecture:** Nouveau `SnapshotTrajectoryStore` (`mw_ia/training/snapshot_store.py`) — sliding window FIFO de N captures × snapshot_size trajectoires, immutable after capture, sample uniforme, filtre succès strict (`terminated AND total_reward > 0`). Helper `concat_batchseq` dans `sequence_buffer.py`. Hook `agent.on_new_best()` appelé par runner après `best_tracker.update()` retourne True. Sample-time mix 80/20 dans `end_episode()` géré par helper `_sample_training_batch()` (4 combinaisons PER × B1a). Default `b1a_enabled=False` strict → V2-Y/V2-ZY/V2-W/V2-U/V2-B0 baselines reproductibles strict.

**Tech Stack:** Python 3.13, PyTorch 2.11+cu128, numpy. Réutilise infrastructure V2-V (BestCheckpointTracker, PeriodicEvaluator) + V2-B0 (PrioritizedSequenceReplayBuffer).

**Spec source:** `docs/superpowers/specs/2026-05-29-mw-ia-policy-snapshot-rehearsal-design.md` (commit `f3b44a9`)

**État initial:** Branche `main`, 10 tags posés (jusqu'à `v0.2.0-b0`). **323 tests pytest verts**. Dernier commit avant V2-B1a : `f3b44a9` (spec V2-B1a).

---

## Phase 1 — Scaffold

### Task 1 : Créer les 2 fichiers de tests vides

**Files:**
- Create: `tests/training/test_snapshot_store.py` (vide)
- Create: `tests/agents/test_b1a_recurrent_agents.py` (vide)

- [ ] **Step 1 — Verify initial state**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `323 passed`.

- [ ] **Step 2 — Create 2 empty test files**

Créer 2 fichiers de 0 byte aux chemins indiqués.

- [ ] **Step 3 — Verify pytest still passes**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `323 passed`.

- [ ] **Step 4 — Commit**

```bash
git add tests/training/test_snapshot_store.py tests/agents/test_b1a_recurrent_agents.py
git commit -m "chore(v2-b1a): scaffold 2 test files for policy snapshot rehearsal"
```

---

## Phase 2 — `concat_batchseq` helper

### Task 2 : `concat_batchseq` dans `sequence_buffer.py` + 3 tests TDD

**Files:**
- Modify: `mw_ia/neural/sequence_buffer.py` (ajouter `concat_batchseq` après le dataclass `BatchSeq`)
- Test: `tests/neural/test_sequence_buffer.py` (extension)

- [ ] **Step 1 — Write the 3 failing tests**

Ajouter à la fin de `tests/neural/test_sequence_buffer.py` (créer le fichier s'il n'existe pas, sinon ajouter à la suite des tests existants) :

```python
# === V2-B1a : concat_batchseq tests ===
import numpy as np
import pytest

from mw_ia.neural.sequence_buffer import BatchSeq, concat_batchseq


def _make_batchseq(seq_len: int, batch_size: int, obs_dim: int, fill_value: float = 0.0) -> BatchSeq:
    """Helper pour construire BatchSeq synthétique."""
    return BatchSeq(
        states=np.full((seq_len, batch_size, obs_dim), fill_value, dtype=np.float32),
        actions=np.zeros((seq_len, batch_size), dtype=np.int64),
        rewards=np.zeros((seq_len, batch_size), dtype=np.float32),
        next_states=np.full((seq_len, batch_size, obs_dim), fill_value, dtype=np.float32),
        dones=np.zeros((seq_len, batch_size), dtype=np.float32),
        mask=np.ones((seq_len, batch_size), dtype=np.float32),
    )


def test_concat_batchseq_axis_correct() -> None:
    """concat_batchseq concat le long de la dimension batch (axis=1), pas seq (axis=0)."""
    a = _make_batchseq(seq_len=8, batch_size=4, obs_dim=10, fill_value=1.0)
    b = _make_batchseq(seq_len=8, batch_size=3, obs_dim=10, fill_value=2.0)
    c = concat_batchseq(a, b)
    # Shape attendue : (8, 4+3, 10) — concat sur axis=1
    assert c.states.shape == (8, 7, 10)
    assert c.actions.shape == (8, 7)
    assert c.rewards.shape == (8, 7)
    assert c.next_states.shape == (8, 7, 10)
    assert c.dones.shape == (8, 7)
    assert c.mask.shape == (8, 7)


def test_concat_batchseq_preserves_content() -> None:
    """Les valeurs de a et b sont préservées dans l'ordre."""
    a = _make_batchseq(seq_len=4, batch_size=2, obs_dim=5, fill_value=1.0)
    b = _make_batchseq(seq_len=4, batch_size=3, obs_dim=5, fill_value=2.0)
    c = concat_batchseq(a, b)
    # Les 2 premières colonnes batch viennent de a (fill_value=1.0)
    assert np.all(c.states[:, :2] == 1.0)
    # Les 3 dernières colonnes batch viennent de b (fill_value=2.0)
    assert np.all(c.states[:, 2:] == 2.0)


def test_concat_batchseq_dtype_preserved() -> None:
    """Dtypes float32 / int64 préservés (pas de cast implicite)."""
    a = _make_batchseq(seq_len=4, batch_size=2, obs_dim=5)
    b = _make_batchseq(seq_len=4, batch_size=2, obs_dim=5)
    c = concat_batchseq(a, b)
    assert c.states.dtype == np.float32
    assert c.actions.dtype == np.int64
    assert c.rewards.dtype == np.float32
    assert c.next_states.dtype == np.float32
    assert c.dones.dtype == np.float32
    assert c.mask.dtype == np.float32
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_sequence_buffer.py -v 2>&1 | tail -15
```

Attendu : 3 fails (`ImportError: cannot import name 'concat_batchseq'`).

- [ ] **Step 3 — Implement concat_batchseq**

Modifier `mw_ia/neural/sequence_buffer.py`. Ajouter après le dataclass `BatchSeq` (avant ou après la classe `SequenceReplayBuffer`, peu importe) :

```python
def concat_batchseq(a: BatchSeq, b: BatchSeq) -> BatchSeq:
    """Concat 2 BatchSeq le long de la dimension batch (axis=1).

    Préconditions : seq_len identique, obs_dim identique. Aucun cast de dtype.
    Utilisé par V2-B1a pour mixer main buffer + snapshot_store dans batch.
    """
    return BatchSeq(
        states=np.concatenate([a.states, b.states], axis=1),
        actions=np.concatenate([a.actions, b.actions], axis=1),
        rewards=np.concatenate([a.rewards, b.rewards], axis=1),
        next_states=np.concatenate([a.next_states, b.next_states], axis=1),
        dones=np.concatenate([a.dones, b.dones], axis=1),
        mask=np.concatenate([a.mask, b.mask], axis=1),
    )
```

Vérifier que `import numpy as np` est déjà en haut du fichier (sinon ajouter).

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_sequence_buffer.py -v 2>&1 | tail -15
```

Attendu : `3 passed` (ou plus si des tests V2-Y existaient déjà dans le fichier).

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `326 passed` (323 + 3).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/neural/sequence_buffer.py tests/neural/test_sequence_buffer.py
git commit -m "feat(v2-b1a): add concat_batchseq helper for V2-B1a sample-time mix (3 tests)"
```

---

## Phase 3 — `SnapshotTrajectoryStore`

### Task 3 : `SnapshotTrajectoryStore` + 12 tests TDD (incl. invariant immutability)

**Files:**
- Create: `mw_ia/training/snapshot_store.py`
- Test: `tests/training/test_snapshot_store.py`

- [ ] **Step 1 — Write the 12 failing tests**

Contenu de `tests/training/test_snapshot_store.py` :

```python
"""Tests V2-B1a SnapshotTrajectoryStore.

Invariant architectural central :
    Une fois capturée, une trajectoire snapshot N'EST JAMAIS modifiée,
    re-evaluee, re-encodee, ou re-rolled-out. Test #11 (test_immutability_after_capture)
    est le test qui formalise cet invariant.
"""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.neural.sequence_buffer import BatchSeq, SequenceReplayBuffer
from mw_ia.training.snapshot_store import SnapshotTrajectoryStore


def _make_buffer_with_trajectories(
    capacity: int = 100, obs_dim: int = 4, max_steps: int = 10, seed: int = 0,
) -> SequenceReplayBuffer:
    """Helper : buffer rempli avec mix success / fail trajectoires."""
    buf = SequenceReplayBuffer(capacity=capacity, obs_dim=obs_dim, max_steps=max_steps, seed=seed)
    rng = np.random.default_rng(seed)
    for i in range(20):
        traj_len = int(rng.integers(3, max_steps))
        # Even i = success (terminated last step + positive reward)
        # Odd i = fail (truncated, negative reward)
        is_success = (i % 2 == 0)
        traj = []
        for t in range(traj_len):
            s = rng.normal(size=(obs_dim,)).astype(np.float32)
            a = int(rng.integers(0, 4))
            if t == traj_len - 1 and is_success:
                r = 1.0  # goal reward
                d = True
            elif t == traj_len - 1 and not is_success:
                r = -1.0  # obstacle penalty
                d = True
            else:
                r = -0.01  # step penalty
                d = False
            sp = rng.normal(size=(obs_dim,)).astype(np.float32)
            traj.append((s, a, r, sp, d))
        buf.push_trajectory(traj)
    return buf


def test_init_validates_args() -> None:
    """obs_dim, max_steps, n_windows, snapshot_size doivent être > 0."""
    with pytest.raises(ValueError, match="obs_dim"):
        SnapshotTrajectoryStore(obs_dim=0)
    with pytest.raises(ValueError, match="max_steps"):
        SnapshotTrajectoryStore(obs_dim=4, max_steps=0)
    with pytest.raises(ValueError, match="n_windows"):
        SnapshotTrajectoryStore(obs_dim=4, n_windows=0)
    with pytest.raises(ValueError, match="snapshot_size"):
        SnapshotTrajectoryStore(obs_dim=4, snapshot_size=0)


def test_empty_store_has_zero_length() -> None:
    """Store fraîchement initialisé contient 0 trajectoires."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, n_windows=3, snapshot_size=50, seed=0)
    assert len(store) == 0
    assert store.n_captures == 0


def test_capture_from_empty_buffer_returns_zero() -> None:
    """Buffer source vide → capture retourne 0, store reste vide."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=10, seed=0)
    empty_buf = SequenceReplayBuffer(capacity=50, obs_dim=4, max_steps=10, seed=0)
    n = store.capture_from(empty_buf)
    assert n == 0
    assert len(store) == 0


def test_capture_filters_terminated_with_positive_reward() -> None:
    """Filtre succès strict : terminated_last_step AND total_reward > 0."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=20, seed=0)
    buf = _make_buffer_with_trajectories()
    # buf contient 20 trajectoires : 10 success (even), 10 fail (odd)
    n = store.capture_from(buf)
    # Doit capturer seulement les 10 success (filtre rejette les fail)
    assert n == 10
    assert len(store) == 10


def test_capture_takes_recent_first() -> None:
    """Iteration arrière depuis current_idx : trajectoires récentes prioritaires."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=3, seed=0)
    buf = _make_buffer_with_trajectories()
    # buf contient 20 trajectoires (10 success aux index pairs)
    # On capture seulement snapshot_size=3 → doit prendre les 3 success les + récentes
    n = store.capture_from(buf)
    assert n == 3
    # Les success les + récentes sont index 18, 16, 14 (puisque current_idx=20 mod 100 = 20)
    # On ne peut pas inspecter directement les indices mais on peut vérifier
    # que les trajectoires capturées ont total_reward > 0 (sample en mode greedy)
    batch = store.sample(batch_size=3, seq_len=10)
    # Au moins une étape avec done=1.0 dans chaque trajectoire (terminated)
    # et reward > 0 quelque part
    for b in range(3):
        valid_mask = batch.mask[:, b] > 0
        traj_rewards = batch.rewards[:, b][valid_mask]
        assert traj_rewards.sum() > 0.0


def test_sliding_window_evicts_oldest_after_n_captures() -> None:
    """Après n_windows+1 captures, la window 0 (la + ancienne) est écrasée."""
    store = SnapshotTrajectoryStore(
        obs_dim=4, max_steps=10, n_windows=2, snapshot_size=5, seed=0,
    )
    buf = _make_buffer_with_trajectories()
    # Capture 1 : remplit window 0
    n1 = store.capture_from(buf)
    assert len(store) == n1
    cap1_count = store.n_captures
    # Capture 2 : remplit window 1
    n2 = store.capture_from(buf)
    assert len(store) == n1 + n2
    cap2_count = store.n_captures
    # Capture 3 : devrait écraser window 0 (oldest)
    n3 = store.capture_from(buf)
    # len(store) doit rester ~ n_windows × snapshot_size, pas grandir au-delà
    assert len(store) <= 2 * 5  # n_windows × snapshot_size
    assert store.n_captures == cap2_count + 1
    assert store.n_captures == 3


def test_n_captures_tracks_total_unbounded() -> None:
    """n_captures continue d'incrémenter même après n_windows captures."""
    store = SnapshotTrajectoryStore(
        obs_dim=4, max_steps=10, n_windows=2, snapshot_size=5, seed=0,
    )
    buf = _make_buffer_with_trajectories()
    for _ in range(5):
        store.capture_from(buf)
    assert store.n_captures == 5  # 5 captures totales, même si seulement 2 windows actives


def test_sample_returns_batchseq_with_correct_shape() -> None:
    """sample() retourne BatchSeq shape (seq, batch, obs_dim)."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=20, seed=0)
    buf = _make_buffer_with_trajectories()
    store.capture_from(buf)
    batch = store.sample(batch_size=4, seq_len=8)
    assert isinstance(batch, BatchSeq)
    assert batch.states.shape == (8, 4, 4)
    assert batch.actions.shape == (8, 4)
    assert batch.mask.shape == (8, 4)


def test_sample_uniform_distribution() -> None:
    """10000 samples : fréquence par slot converge vers 1/total à ±2%."""
    store = SnapshotTrajectoryStore(
        obs_dim=2, max_steps=4, snapshot_size=5, seed=42,
    )
    # Capture manuellement 5 trajectoires distinguables via 1ère valeur de state
    buf = SequenceReplayBuffer(capacity=20, obs_dim=2, max_steps=4, seed=0)
    for marker in range(5):
        traj = [
            (np.array([marker, 0], dtype=np.float32), 0, 0.01, np.array([marker, 1], dtype=np.float32), False),
            (np.array([marker, 1], dtype=np.float32), 0, 0.01, np.array([marker, 2], dtype=np.float32), False),
            (np.array([marker, 2], dtype=np.float32), 0, 1.0, np.array([marker, 3], dtype=np.float32), True),
        ]
        buf.push_trajectory(traj)
    store.capture_from(buf)
    assert len(store) == 5

    # 10000 samples de batch_size=1, comptage par marker (1ère value de state)
    counts = np.zeros(5, dtype=int)
    n_samples = 10_000
    for _ in range(n_samples):
        batch = store.sample(batch_size=1, seq_len=3)
        marker = int(batch.states[0, 0, 0])
        if 0 <= marker < 5:
            counts[marker] += 1
    expected_freq = 1.0 / 5
    empirical_freq = counts / n_samples
    assert np.allclose(empirical_freq, expected_freq, atol=0.02)


def test_sample_raises_if_too_few_trajectories() -> None:
    """sample(batch_size > len(store)) lève ValueError."""
    store = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=10, seed=0)
    buf = _make_buffer_with_trajectories()
    store.capture_from(buf)
    # Capture peut donner < 10 (filtre success), donc on prend un batch_size > tout possible
    with pytest.raises(ValueError, match="trop petit"):
        store.sample(batch_size=100, seq_len=5, )


def test_immutability_after_capture() -> None:
    """INVARIANT ARCHITECTURAL : modifier source post-capture ne change pas le store.

    Une trajectoire capturée est frozen — copies, pas références.
    """
    store = SnapshotTrajectoryStore(obs_dim=2, max_steps=4, snapshot_size=5, seed=0)
    buf = SequenceReplayBuffer(capacity=10, obs_dim=2, max_steps=4, seed=0)
    # Push 3 success trajectoires
    for marker in range(3):
        traj = [
            (np.array([marker, 0], dtype=np.float32), 0, 0.01, np.array([marker, 1], dtype=np.float32), False),
            (np.array([marker, 1], dtype=np.float32), 0, 1.0, np.array([marker, 2], dtype=np.float32), True),
        ]
        buf.push_trajectory(traj)
    store.capture_from(buf)
    assert len(store) == 3

    # Sample baseline pour comparaison
    rng_state = np.random.default_rng(123).bit_generator.state
    store._rng.bit_generator.state = rng_state
    batch_before = store.sample(batch_size=3, seq_len=2)
    states_before = batch_before.states.copy()

    # MUTATION DESTRUCTRICE du source buffer
    buf._states[:] = 999.0
    buf._actions[:] = 999
    buf._rewards[:] = -999.0
    buf._dones[:] = 999.0

    # Re-sample : le store DOIT retourner les mêmes valeurs qu'avant
    store._rng.bit_generator.state = rng_state
    batch_after = store.sample(batch_size=3, seq_len=2)
    assert np.array_equal(batch_after.states, states_before), (
        "INVARIANT VIOLATION : modifier source buffer a contaminé le snapshot store"
    )


def test_reproducibility_with_seed() -> None:
    """Même seed → même séquence de samples."""
    buf = _make_buffer_with_trajectories(seed=0)
    store1 = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=20, seed=42)
    store2 = SnapshotTrajectoryStore(obs_dim=4, max_steps=10, snapshot_size=20, seed=42)
    store1.capture_from(buf)
    store2.capture_from(buf)
    b1 = store1.sample(batch_size=4, seq_len=8)
    b2 = store2.sample(batch_size=4, seq_len=8)
    assert np.array_equal(b1.states, b2.states)
    assert np.array_equal(b1.actions, b2.actions)
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/training/test_snapshot_store.py -v 2>&1 | tail -20
```

Attendu : 12 fails (`ModuleNotFoundError: No module named 'mw_ia.training.snapshot_store'`).

- [ ] **Step 3 — Implement SnapshotTrajectoryStore**

Créer `mw_ia/training/snapshot_store.py` :

```python
"""SnapshotTrajectoryStore — sliding window FIFO de captures de trajectoires.

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
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/training/test_snapshot_store.py -v 2>&1 | tail -20
```

Attendu : `12 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `338 passed` (326 + 12).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/training/snapshot_store.py tests/training/test_snapshot_store.py
git commit -m "feat(v2-b1a): add SnapshotTrajectoryStore with immutable sliding window (12 tests)"
```

---

## Phase 4 — Extension Config (DRQNConfig + ConvRecurrentDQNConfig)

### Task 4 : Ajouter 4 champs B1a × 2 dataclasses + 4 tests validation

**Files:**
- Modify: `mw_ia/config.py` (DRQNConfig + ConvRecurrentDQNConfig)
- Test: `tests/neural/test_prioritized_sequence_buffer.py` (extension fin de fichier)

- [ ] **Step 1 — Add 4 fields + validation to DRQNConfig**

Dans `mw_ia/config.py`, localiser la classe `DRQNConfig`. Ajouter après les champs `per_*` (V2-B0) :

```python
    # V2-B1a : Policy Snapshot Rehearsal (sliding window N captures × snapshot_size traj)
    b1a_enabled: bool = False
    b1a_snapshot_size: int = 50      # nombre de trajectoires capturees par best
    b1a_n_windows: int = 3           # sliding window FIFO
    b1a_mix_ratio: float = 0.2       # fraction du batch venant du snapshot
```

Dans `DRQNConfig.__post_init__`, ajouter à la fin (après les validations `per_*`) :

```python
        if self.b1a_snapshot_size <= 0:
            raise ValueError(f"b1a_snapshot_size doit etre > 0, recu {self.b1a_snapshot_size}")
        if self.b1a_n_windows <= 0:
            raise ValueError(f"b1a_n_windows doit etre > 0, recu {self.b1a_n_windows}")
        if not (0.0 < self.b1a_mix_ratio < 1.0):
            raise ValueError(
                f"b1a_mix_ratio doit etre dans ]0, 1[, recu {self.b1a_mix_ratio}"
            )
```

- [ ] **Step 2 — Add same 4 fields + validation to ConvRecurrentDQNConfig**

Dans `mw_ia/config.py`, localiser la classe `ConvRecurrentDQNConfig`. Ajouter après les champs `per_*` :

```python
    # V2-B1a : Policy Snapshot Rehearsal (sliding window N captures × snapshot_size traj)
    b1a_enabled: bool = False
    b1a_snapshot_size: int = 50
    b1a_n_windows: int = 3
    b1a_mix_ratio: float = 0.2
```

Dans `ConvRecurrentDQNConfig.__post_init__`, à la fin :

```python
        if self.b1a_snapshot_size <= 0:
            raise ValueError(f"b1a_snapshot_size doit etre > 0, recu {self.b1a_snapshot_size}")
        if self.b1a_n_windows <= 0:
            raise ValueError(f"b1a_n_windows doit etre > 0, recu {self.b1a_n_windows}")
        if not (0.0 < self.b1a_mix_ratio < 1.0):
            raise ValueError(
                f"b1a_mix_ratio doit etre dans ]0, 1[, recu {self.b1a_mix_ratio}"
            )
```

- [ ] **Step 3 — Run full suite, verify no regression (338 still passes)**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `338 passed` (defaults `b1a_enabled=False` ne change rien).

- [ ] **Step 4 — Add 4 validation tests**

Ajouter à la fin de `tests/neural/test_prioritized_sequence_buffer.py` :

```python
# === V2-B1a config validation tests ===
def test_drqn_config_b1a_snapshot_size_zero_raises() -> None:
    with pytest.raises(ValueError, match="b1a_snapshot_size"):
        DRQNConfig(b1a_snapshot_size=0)
    with pytest.raises(ValueError, match="b1a_snapshot_size"):
        DRQNConfig(b1a_snapshot_size=-1)


def test_drqn_config_b1a_n_windows_zero_raises() -> None:
    with pytest.raises(ValueError, match="b1a_n_windows"):
        DRQNConfig(b1a_n_windows=0)
    with pytest.raises(ValueError, match="b1a_n_windows"):
        DRQNConfig(b1a_n_windows=-2)


def test_drqn_config_b1a_mix_ratio_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="b1a_mix_ratio"):
        DRQNConfig(b1a_mix_ratio=0.0)
    with pytest.raises(ValueError, match="b1a_mix_ratio"):
        DRQNConfig(b1a_mix_ratio=1.0)
    with pytest.raises(ValueError, match="b1a_mix_ratio"):
        DRQNConfig(b1a_mix_ratio=1.5)
    with pytest.raises(ValueError, match="b1a_mix_ratio"):
        DRQNConfig(b1a_mix_ratio=-0.1)


def test_conv_recurrent_config_b1a_validation_parallel() -> None:
    with pytest.raises(ValueError, match="b1a_mix_ratio"):
        ConvRecurrentDQNConfig(b1a_mix_ratio=-0.1)
    with pytest.raises(ValueError, match="b1a_snapshot_size"):
        ConvRecurrentDQNConfig(b1a_snapshot_size=0)
```

- [ ] **Step 5 — Run new tests + full suite**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_prioritized_sequence_buffer.py -v 2>&1 | tail -10
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : 4 nouveaux tests passing dans test_prioritized_sequence_buffer.py, `342 passed` total (338 + 4).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/config.py tests/neural/test_prioritized_sequence_buffer.py
git commit -m "feat(v2-b1a): add 4 B1a fields x 2 configs (DRQN + ConvRecurrent) with validation (4 tests)"
```

---

## Phase 5 — Agents PER+B1a integration

### Task 5 : Branche conditionnelle B1a + sample-time mix (7 parametrized × 2 agents = 14 cases)

**Files:**
- Modify: `mw_ia/agents/recurrent_dqn.py` (constructor + on_new_best + _sample_training_batch + end_episode)
- Modify: `mw_ia/agents/conv_recurrent_dqn.py` (idem, pattern parallèle)
- Test: `tests/agents/test_b1a_recurrent_agents.py`

- [ ] **Step 1 — Write the 7 parametrized failing tests**

Contenu de `tests/agents/test_b1a_recurrent_agents.py` :

```python
"""Tests V2-B1a integration agents V2-Y RecurrentDQNAgent et V2-ZY ConvRecurrentDQNAgent.

Parametrized sur les 2 agents.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mw_ia.agents.recurrent_dqn import RecurrentDQNAgent
from mw_ia.agents.conv_recurrent_dqn import ConvRecurrentDQNAgent
from mw_ia.config import DRQNConfig, ConvRecurrentDQNConfig
from mw_ia.training.snapshot_store import SnapshotTrajectoryStore


def _build_drqn(b1a_enabled: bool, per_enabled: bool = False, **kwargs) -> RecurrentDQNAgent:
    cfg = DRQNConfig(
        b1a_enabled=b1a_enabled,
        per_enabled=per_enabled,
        replay_capacity=50,
        min_episodes_to_learn=5,
        batch_size=10,
        sequence_length=4,
        max_steps_per_episode=8,
        episodes=100,
        use_amp=False,
        b1a_snapshot_size=10,
        b1a_n_windows=2,
        b1a_mix_ratio=0.2,
        **kwargs,
    )
    return RecurrentDQNAgent(obs_dim=4, n_actions=4, cfg=cfg, device="cpu", seed=0)


def _build_conv_recurrent(b1a_enabled: bool, per_enabled: bool = False, **kwargs) -> ConvRecurrentDQNAgent:
    cfg = ConvRecurrentDQNConfig(
        b1a_enabled=b1a_enabled,
        per_enabled=per_enabled,
        replay_capacity=50,
        min_episodes_to_learn=5,
        batch_size=10,
        sequence_length=4,
        max_steps_per_episode=8,
        episodes=100,
        use_amp=False,
        eval_enabled=False,
        b1a_snapshot_size=10,
        b1a_n_windows=2,
        b1a_mix_ratio=0.2,
        **kwargs,
    )
    return ConvRecurrentDQNAgent(
        in_channels=2, rows=2, cols=2, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )


AGENT_BUILDERS = [
    pytest.param(_build_drqn, "drqn", 4, id="v2y_drqn"),
    pytest.param(_build_conv_recurrent, "conv_recurrent", 8, id="v2zy_conv_recurrent"),
]


def _make_success_trajectory(agent_kind: str, obs_dim: int, length: int = 5):
    """Trajectoire successful synthetique (terminated AND total_reward > 0)."""
    traj = []
    for t in range(length):
        if agent_kind == "drqn":
            s = np.zeros(obs_dim, dtype=np.float32)
            sp = np.zeros(obs_dim, dtype=np.float32)
        else:
            obs = np.zeros((2, 2, 2), dtype=np.float32)
            s = obs.flatten()
            sp = obs.flatten()
        r = 1.0 if t == length - 1 else -0.01
        d = (t == length - 1)
        traj.append((s, 0, r, sp, d))
    return traj


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_disabled_no_snapshot_store(builder, agent_kind, obs_dim) -> None:
    """b1a_enabled=False → snapshot_store is None, on_new_best() retourne 0."""
    agent = builder(b1a_enabled=False)
    assert agent.snapshot_store is None
    assert agent.on_new_best() == 0


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_enabled_instantiates_snapshot_store(builder, agent_kind, obs_dim) -> None:
    """b1a_enabled=True → SnapshotTrajectoryStore instancie, len == 0 init."""
    agent = builder(b1a_enabled=True)
    assert isinstance(agent.snapshot_store, SnapshotTrajectoryStore)
    assert len(agent.snapshot_store) == 0


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_on_new_best_triggers_capture(builder, agent_kind, obs_dim) -> None:
    """on_new_best() apres push de trajectoires successful → len > 0, n_captures == 1."""
    agent = builder(b1a_enabled=True)
    # Push 10 success trajectoires dans le main buffer
    for _ in range(10):
        traj = _make_success_trajectory(agent_kind, obs_dim)
        agent._episode_trajectory = traj
        agent.end_episode()
    n_captured = agent.on_new_best()
    assert n_captured > 0
    assert agent.snapshot_store.n_captures == 1
    assert len(agent.snapshot_store) == n_captured


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_threshold_strict_pure_main_below_threshold(builder, agent_kind, obs_dim) -> None:
    """Snapshot avec moins que snapshot_B traj → batch tire purement main (no mix).

    Strategy : on capture peu de trajectoires (< snapshot_B), puis on appelle
    _sample_training_batch et on verifie que batch_size == main_B (pas mix).
    """
    agent = builder(b1a_enabled=True)
    # snapshot_B = int(B * mix_ratio) = int(10 * 0.2) = 2
    # Pour rester sous threshold, on doit avoir < 2 traj dans le snapshot
    # Push 1 success seulement
    traj = _make_success_trajectory(agent_kind, obs_dim)
    agent._episode_trajectory = traj
    agent.end_episode()
    agent.on_new_best()  # capture 1 traj
    assert len(agent.snapshot_store) == 1  # < snapshot_B=2

    # Maintenant remplit le main buffer avec assez de trajectoires
    for _ in range(20):
        traj = _make_success_trajectory(agent_kind, obs_dim)
        agent._episode_trajectory = traj
        agent.end_episode()

    # Call _sample_training_batch : doit returner batch shape (seq, B, ...) car b1a inactif
    tb = agent._sample_training_batch()
    # B=10, snapshot_B=2, len(snapshot)=1 < 2 → b1a_active=False → main_B=B=10
    assert tb.batch.states.shape[1] == agent.cfg.batch_size


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_active_mixes_batch_shape(builder, agent_kind, obs_dim) -> None:
    """Snapshot rempli → batch shape (seq, B, ...) avec B*0.2 derniers du snapshot."""
    agent = builder(b1a_enabled=True)
    # Push assez de success traj pour avoir snapshot_size=10 dans le store
    for _ in range(15):
        traj = _make_success_trajectory(agent_kind, obs_dim)
        agent._episode_trajectory = traj
        agent.end_episode()
    agent.on_new_best()
    assert len(agent.snapshot_store) >= 2  # snapshot_B = int(10*0.2) = 2

    # Sample : doit mixer 8 main + 2 snapshot = 10 total
    tb = agent._sample_training_batch()
    assert tb.batch.states.shape[1] == agent.cfg.batch_size  # B=10
    # Pas de verification fine sur quelles colonnes viennent de quoi
    # (cf test suivant pour weights)


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_per_on_weights_concat_correct(builder, agent_kind, obs_dim) -> None:
    """PER + B1a → weights shape (B,), weights[main_B:] == 1.0 strict."""
    agent = builder(b1a_enabled=True, per_enabled=True)
    # Push assez de traj
    for _ in range(15):
        traj = _make_success_trajectory(agent_kind, obs_dim)
        agent._episode_trajectory = traj
        agent.end_episode()
    agent.on_new_best()
    # Update some priorities pour rendre PER actif (sinon all uniform = bias)
    for _ in range(3):
        agent._episode_trajectory = _make_success_trajectory(agent_kind, obs_dim)
        agent.end_episode()

    tb = agent._sample_training_batch()
    # main_B = 8, snapshot_B = 2
    assert tb.weights is not None
    assert tb.weights.shape == (agent.cfg.batch_size,)
    # Snapshot portion (indices [main_B:]) doit avoir weights == 1.0
    main_B = agent.cfg.batch_size - int(agent.cfg.batch_size * agent.cfg.b1a_mix_ratio)
    snapshot_weights = tb.weights[main_B:]
    assert np.allclose(snapshot_weights, 1.0)


@pytest.mark.parametrize("builder,agent_kind,obs_dim", AGENT_BUILDERS)
def test_b1a_per_on_priorities_updated_main_only(builder, agent_kind, obs_dim) -> None:
    """update_priorities appele uniquement sur td_errors[:main_B] (pas le batch entier)."""
    agent = builder(b1a_enabled=True, per_enabled=True)
    for _ in range(15):
        traj = _make_success_trajectory(agent_kind, obs_dim)
        agent._episode_trajectory = traj
        agent.end_episode()
    agent.on_new_best()
    assert len(agent.snapshot_store) >= 2

    # Mock buffer.update_priorities pour intercepter les appels
    original_update_priorities = agent.buffer.update_priorities
    captured_calls = []

    def mock_update(indices, td_errors):
        captured_calls.append((indices.copy(), td_errors.copy()))
        return original_update_priorities(indices, td_errors)

    agent.buffer.update_priorities = mock_update

    # Trigger train step
    traj = _make_success_trajectory(agent_kind, obs_dim)
    agent._episode_trajectory = traj
    agent.end_episode()

    # Verifier que update_priorities a ete appele avec td_errors[:main_B]
    assert len(captured_calls) > 0
    last_call_indices, last_call_td = captured_calls[-1]
    main_B = agent.cfg.batch_size - int(agent.cfg.batch_size * agent.cfg.b1a_mix_ratio)
    assert last_call_indices.shape[0] == main_B
    assert last_call_td.shape[0] == main_B
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_b1a_recurrent_agents.py -v 2>&1 | tail -20
```

Attendu : 14 fails (8 attribute errors `snapshot_store`, 6 method errors `_sample_training_batch`/`on_new_best`).

- [ ] **Step 3 — Modify `RecurrentDQNAgent`**

Dans `mw_ia/agents/recurrent_dqn.py`, ajouter en haut :

```python
from dataclasses import dataclass

from mw_ia.neural.sequence_buffer import concat_batchseq
from mw_ia.training.snapshot_store import SnapshotTrajectoryStore
```

Ajouter un dataclass privé au module (top-level, hors classe) :

```python
@dataclass
class _TrainingBatch:
    """Conteneur batch + weights + tree_indices pour _sample_training_batch."""
    batch: "BatchSeq"
    weights: "np.ndarray | None"
    tree_indices: "np.ndarray | None"
```

Dans `RecurrentDQNAgent.__init__`, ajouter APRÈS l'instantiation du buffer (V2-B0 conditional branch existant), avant la ligne `self.global_step: int = 0` :

```python
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
```

Ajouter la méthode `on_new_best()` dans `RecurrentDQNAgent` (n'importe où, je suggère après `begin_episode()`) :

```python
    def on_new_best(self) -> int:
        """Hook appele par le runner quand BestCheckpointTracker detecte un nouveau peak.

        Si B1a active : capture jusqu'a snapshot_size trajectoires successful recentes
        depuis self.buffer dans self.snapshot_store (sliding window N=3).
        Si B1a desactive : no-op.

        Returns: nombre de trajectoires effectivement capturees.
        """
        if not self.cfg.b1a_enabled:
            return 0
        return self.snapshot_store.capture_from(self.buffer)
```

Ajouter la méthode `_sample_training_batch()` (privée, helper pour `end_episode`) :

```python
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
            prio = self.buffer.sample(main_B, L, beta=beta)
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
```

Remplacer la boucle train dans `end_episode()` (la version V2-B0 actuelle) par :

```python
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
                        self.buffer.update_priorities(tb.tree_indices, td_errors[:main_B])
                    losses.append(loss)
                    if self.cfg.per_enabled:
                        # B1a actif ou pas, on emet per_beta (signal d'observabilite)
                        metrics["per_beta"] = self._beta_scheduler.beta(self._episode_count)
                else:
                    loss = self.trainer.step(tb.batch)
                    losses.append(loss)
            if losses:
                self.last_loss = sum(losses) / len(losses)
                metrics["loss"] = self.last_loss
```

**IMPORTANT** : il faut conserver le comportement V2-B0 strict (`per_beta` dans metrics quand PER actif). La nouvelle structure doit faire ça correctement. Si tu trouves que le code original V2-B0 émettait `per_beta` ailleurs, garde cette logique cohérente.

- [ ] **Step 4 — Modify `ConvRecurrentDQNAgent` (pattern identique)**

Dans `mw_ia/agents/conv_recurrent_dqn.py`, ajouter en haut :

```python
from dataclasses import dataclass

from mw_ia.neural.sequence_buffer import concat_batchseq
from mw_ia.training.snapshot_store import SnapshotTrajectoryStore
```

Ajouter le dataclass `_TrainingBatch` au top-level du module (identique à V2-Y, si déjà importable d'ailleurs on peut share, mais ici duplicate accepté pour clarity).

Dans `ConvRecurrentDQNAgent.__init__`, ajouter APRÈS le bloc V2-B0 conditional buffer, avant `self.global_step` :

```python
        if cfg.b1a_enabled:
            self.snapshot_store: SnapshotTrajectoryStore | None = SnapshotTrajectoryStore(
                obs_dim=obs_dim_flat,
                max_steps=cfg.max_steps_per_episode,
                n_windows=cfg.b1a_n_windows,
                snapshot_size=cfg.b1a_snapshot_size,
                seed=seed,
            )
        else:
            self.snapshot_store = None
```

Ajouter `on_new_best()` et `_sample_training_batch()` méthodes (copies exactes du code V2-Y ci-dessus) sur `ConvRecurrentDQNAgent`.

Remplacer la boucle train dans `end_episode()` par la même structure que V2-Y.

- [ ] **Step 5 — Run new tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_b1a_recurrent_agents.py -v 2>&1 | tail -20
```

Attendu : `14 passed`.

- [ ] **Step 6 — Run full suite (V2-B0 et V2-Y/V2-ZY préservés)**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `356 passed` (342 + 14). Tous les tests V2-B0 (16 cases) et V2-Y/V2-ZY existants doivent rester verts grâce à `b1a_enabled=False` default.

- [ ] **Step 7 — Commit**

```bash
git add mw_ia/agents/recurrent_dqn.py mw_ia/agents/conv_recurrent_dqn.py tests/agents/test_b1a_recurrent_agents.py
git commit -m "feat(v2-b1a): integrate B1a (snapshot + sample mix) in V2-Y + V2-ZY agents (14 parametrized tests)"
```

---

## Phase 6 — Runner hook B1a

### Task 6 : Ajouter `agent.on_new_best()` call dans `ConvRecurrentProceduralDQNRunner`

**Files:**
- Modify: `mw_ia/training/runner.py` (classe `ConvRecurrentProceduralDQNRunner.run()`)

- [ ] **Step 1 — Inspect existing runner code**

```bash
grep -n "best_tracker.update" mw_ia/training/runner.py
```

Localiser l'appel `improved = self.best_tracker.update(...)` dans `ConvRecurrentProceduralDQNRunner.run()`. Il devrait être autour de la ligne 734 (peut varier).

- [ ] **Step 2 — Add B1a hook after best_tracker.update**

Trouver la section dans `ConvRecurrentProceduralDQNRunner.run()` qui ressemble à :

```python
                eval_metrics = self.evaluator.evaluate(
                    self.agent, self.dqn_cfg.eval_target_difficulty,
                )
                improved = self.best_tracker.update(eval_metrics, self.agent, episode=ep)
                self.callbacks.fire_evaluation(...)
```

Ajouter ENTRE `improved = ...` ET `self.callbacks.fire_evaluation(...)` :

```python
                if improved:
                    n_captured = self.agent.on_new_best()
                    if n_captured > 0:
                        self.callbacks.on_log(
                            "info",
                            f"B1a snapshot capture : {n_captured} trajectoires "
                            f"(capture #{self.agent.snapshot_store.n_captures})",
                        )
```

- [ ] **Step 3 — Verify full pytest still passes**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `356 passed` (pas de regression).

- [ ] **Step 4 — Sanity smoke local (B1a + V2-V eval)**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 30 --mode obstacles --device cpu \
    --eval-every-episodes 5 --best-checkpoint-path checkpoints/sanity_b1a_runner.pt 2>&1 | tail -20
```

Attendu : run complet, no crash. Le `b1a_enabled` par défaut est False donc on ne devrait PAS voir de log "B1a snapshot capture". C'est attendu — on teste juste la baseline runner ne casse pas.

- [ ] **Step 5 — Sanity smoke local AVEC --b1a (flag pas encore exposé en CLI mais on peut le tester via Python directe ou en passant la config)**

Pour tester le hook B1a effectivement, il faut utiliser le flag `--b1a` qui sera ajouté en Task 7. Ce smoke complet attendra donc Task 7. On valide juste le pytest pour Task 6.

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/training/runner.py
git commit -m "feat(v2-b1a): add agent.on_new_best() hook in ConvRecurrentProceduralDQNRunner"
```

---

## Phase 7 — CLI flags V2-Y + V2-ZY scripts

### Task 7 : Ajouter 4 flags B1a aux 2 scripts CLI

**Files:**
- Modify: `scripts/train_drqn_procedural.py`
- Modify: `scripts/train_cnn_lstm_dqn_procedural.py`

- [ ] **Step 1 — Add 4 B1a flags to `train_drqn_procedural.py`**

Dans `scripts/train_drqn_procedural.py`, localiser le bloc `parser.add_argument` PER (ajouté V2-B0). Ajouter les flags B1a APRÈS les flags `--per-*` et `--max-attempts-bfs` :

```python
    parser.add_argument(
        "--b1a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="V2-B1a : Policy Snapshot Rehearsal — capture des trajectoires successful "
             "depuis le buffer au moment du best eval (V2-V), inject 20%% dans chaque batch. "
             "Default False = pas de rehearsal (V2-U / V2-B0 baseline).",
    )
    parser.add_argument("--b1a-snapshot-size", type=int, default=50,
                        help="V2-B1a : nb de trajectoires capturees par best (default 50).")
    parser.add_argument("--b1a-n-windows", type=int, default=3,
                        help="V2-B1a : sliding window des N derniers bests (default 3).")
    parser.add_argument("--b1a-mix-ratio", type=float, default=0.2,
                        help="V2-B1a : fraction du batch venant du snapshot (default 0.2 = 20%%).")
```

Localiser la construction `dqn_cfg = DRQNConfig(...)`. Ajouter les 4 champs B1a :

```python
    dqn_cfg = DRQNConfig(
        # ... champs existants V2-Y/B0 ...
        b1a_enabled=args.b1a,
        b1a_snapshot_size=args.b1a_snapshot_size,
        b1a_n_windows=args.b1a_n_windows,
        b1a_mix_ratio=args.b1a_mix_ratio,
    )
```

- [ ] **Step 2 — Add same 4 B1a flags to `train_cnn_lstm_dqn_procedural.py`**

Dans `scripts/train_cnn_lstm_dqn_procedural.py`, ajouter les mêmes 4 flags après les flags `--per-*` et `--max-attempts-bfs`.

Localiser la construction `dqn_cfg = ConvRecurrentDQNConfig(...)`. Ajouter les 4 champs B1a (idem V2-Y).

- [ ] **Step 3 — Sanity smoke V2-Y CLI avec --b1a**

```bash
source .venv/Scripts/activate && python scripts/train_drqn_procedural.py --episodes 30 --mode obstacles --device cpu --b1a 2>&1 | tail -10
```

Attendu : exécution complète, no crash. Logs `B1a snapshot capture` peuvent apparaître si V2-Y a aussi un hook runner — sinon non (V2-Y runner asymmetry documentée dans spec).

- [ ] **Step 4 — Sanity smoke V2-ZY CLI avec --b1a**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_lstm_dqn_procedural.py --episodes 30 --mode obstacles --device cpu --b1a --best-checkpoint-path checkpoints/sanity_b1a.pt 2>&1 | tail -15
```

Attendu : run complet, fichier checkpoint créé, logs `B1a snapshot capture : N trajectoires (capture #1)` ou similaire devraient apparaître au moins une fois (premier best detected déclenche capture).

- [ ] **Step 5 — Run full pytest**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `356 passed`.

- [ ] **Step 6 — Commit**

```bash
git add scripts/train_drqn_procedural.py scripts/train_cnn_lstm_dqn_procedural.py
git commit -m "feat(v2-b1a-cli): expose --b1a + 3 hyperparams on V2-Y + V2-ZY scripts"
```

---

## Phase 8 — CI smoke

### Task 8 : Ajouter 2 smoke jobs CI (B1a seul + B1a + PER cohabit)

**Files:**
- Modify: `.github/workflows/aether_verify.yml`

- [ ] **Step 1 — Add 2 new smoke jobs**

Dans `.github/workflows/aether_verify.yml`, ajouter après les smoke V2-B0 (la dernière étape `Smoke test V2-B0 PER + Polyak cohabit (sanity check)`) :

```yaml
      - name: Smoke test V2-B1a (snapshot rehearsal seul) on V2-ZY
        run: |
          mkdir -p checkpoints
          python scripts/train_cnn_lstm_dqn_procedural.py \
            --episodes 30 --mode obstacles --device cpu \
            --b1a --eval-every-episodes 5 \
            --best-checkpoint-path checkpoints/ci_v2b1a_best.pt
          test -f checkpoints/ci_v2b1a_best.pt
      - name: Smoke test V2-B1a + PER cohabit (4e bras factoriel)
        run: |
          mkdir -p checkpoints
          python scripts/train_cnn_lstm_dqn_procedural.py \
            --episodes 30 --mode obstacles --device cpu \
            --b1a --per --polyak-tau 0.005 --eval-every-episodes 5 \
            --best-checkpoint-path checkpoints/ci_v2b1a_per_best.pt
          test -f checkpoints/ci_v2b1a_per_best.pt
```

**Note `--episodes 30`** (vs `--episodes 10` pour V2-B0) : il faut au moins 2 evaluations (à `eval_every_episodes=5`) pour qu'un best soit captured et exercer le pipeline complet de capture B1a.

- [ ] **Step 2 — Smoke local equivalent**

```bash
source .venv/Scripts/activate
mkdir -p checkpoints
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 30 --mode obstacles --device cpu --b1a --eval-every-episodes 5 --best-checkpoint-path checkpoints/ci_v2b1a_best.pt
test -f checkpoints/ci_v2b1a_best.pt && echo "V2-B1a smoke OK"
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 30 --mode obstacles --device cpu --b1a --per --polyak-tau 0.005 --eval-every-episodes 5 --best-checkpoint-path checkpoints/ci_v2b1a_per_best.pt
test -f checkpoints/ci_v2b1a_per_best.pt && echo "V2-B1a + PER cohabit smoke OK"
```

Attendu : 2 messages "OK".

- [ ] **Step 3 — Commit**

```bash
git add .github/workflows/aether_verify.yml
git commit -m "ci(v2-b1a): add 2 smoke jobs (B1a seul + B1a+PER cohabit) on V2-ZY"
```

---

## Phase 9 — Sanity verification finale

### Task 9 : Verify all 356 tests + Aether + baselines preservées

**Files:** aucun, vérifications uniquement.

- [ ] **Step 1 — Full pytest run**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `356 passed` (323 baseline V2-B0 + 33 V2-B1a tests).

- [ ] **Step 2 — Test count précis**

```bash
source .venv/Scripts/activate && pytest --co 2>&1 | tail -1
```

Attendu : `356 tests collected`.

- [ ] **Step 3 — Aether harness check**

```bash
bash aether/verify_all.sh 2>&1 | tail -5
```

Attendu : 8 OK (inchangé).

- [ ] **Step 4 — V2-U baseline reproducible sans --b1a**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_lstm_dqn_procedural.py --episodes 20 --mode obstacles --device cpu --seed 0 --polyak-tau 0.005 2>&1 | tail -5
```

Attendu : exécution complète, comportement V2-U identique (pas de log B1a). `b1a_enabled=False` default préserve la baseline strict.

- [ ] **Step 5 — V2-B0 PER baseline reproducible sans --b1a**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_lstm_dqn_procedural.py --episodes 20 --mode obstacles --device cpu --seed 0 --polyak-tau 0.005 --per 2>&1 | tail -5
```

Attendu : exécution complète, comportement V2-B0 identique (logs `per_beta` mais pas de log B1a).

- [ ] **Step 6 — No commit (vérification pure)**

---

## Phase 10 — Documentation README + CLAUDE.md (livraison code)

### Task 10 : Section V2-B1a dans README + CLAUDE.md (sans bench results)

**Files:**
- Modify: `README.md` (nouvelle section V2-B1a après V2-B0)
- Modify: `CLAUDE.md` (nouvelle section V2-B1a phases + composants + décisions + pièges, sans bench results)

- [ ] **Step 1 — Add V2-B1a section to README.md**

Ajouter dans `README.md` après la section V2-B0 (avant `## Roadmap (V2+)`) :

```markdown
## V2-B1a — Policy Snapshot Rehearsal (code livré, bench pending)

**Tests** : 356 verts (323 baseline V2-B0 + 33 V2-B1a). **Tag** : `v0.2.0-b1a` prévu après bench n=5.

Sous-projet B phase 1a : capture des trajectoires successful récentes du buffer au moment du best eval V2-V, mix 20% dans chaque batch d'entraînement. Variante "frozen rehearsal" minimaliste du sous-projet B.

### Hypothèse

Préserver les trajectoires near-frontier (frozen, sliding window des 3 derniers bests × 50 trajectoires) suffit-il à éviter le forgetting et améliorer V2-ZY+Polyak en 15×15 ?

### Usage CLI (opt-in via `--b1a`)

```bash
# V2-B1a seul (Bras 3 du bench factoriel 2x2)
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda \
    --seed {N} --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --b1a --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b1a_15x15_seed{N}.pt

# V2-B1a + PER (Bras 4, test interaction)
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda \
    --seed {N} --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --b1a --per --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b1a_per_15x15_seed{N}.pt
```

### Hyperparams V2-B1a (défauts)

| Flag | Default | Source |
|---|---|---|
| `--b1a-snapshot-size` | 50 | Spec V2-B1a brainstorm |
| `--b1a-n-windows` | 3 | Sliding window FIFO |
| `--b1a-mix-ratio` | 0.2 | 20% snapshot / 80% main |

### Architecture (résumé)

- `SnapshotTrajectoryStore` (`mw_ia/training/snapshot_store.py`) : sliding window FIFO, **immutable after capture**
- Filtre succès strict : `terminated AND total_reward > 0`
- Sample uniforme (pas de PER interne sur le snapshot)
- Hook `agent.on_new_best()` appelé par runner quand `BestCheckpointTracker.update()` retourne True
- Sample-time mix dans `end_episode()` : `_sample_training_batch()` gère les 4 combinaisons PER × B1a

### Critère succès (bench factoriel 2×2 pré-enregistré)

**Phase 1 — Bras 3 B1a seul vs baseline V2-U** : mean ≥ 55 %, std ≤ 20 pp, 0/5 collapse, diff_max ≥ 0.30.

**Phase 2 — Bras 4 B1a+PER vs V2-B0+PER seul** : mean > 56 % (recovery ≥ 10 pp). Bonus : mean ≥ 64 % (full recovery), mean > 64 % AND > Bras 3 (positive interaction → phase-dépendance confirmée).
```

- [ ] **Step 2 — Add V2-B1a section to CLAUDE.md**

Dans `CLAUDE.md`, ajouter une section V2-B1a après la section V2-B0 (qui doit déjà exister depuis V2-B0). La section doit inclure :

1. **"V2-B1a — état final des phases (livraison 2026-05-29)"** : table 13 phases × Statut × Tests × Commits (Phases 1-10 complétées au moment de cette livraison, Phases 11-13 pending GPU bench)

2. **"Composants V2-B1a livrés"** : table Composant × Fichier × Rôle :
   - `SnapshotTrajectoryStore` (`mw_ia/training/snapshot_store.py`)
   - `concat_batchseq` (`mw_ia/neural/sequence_buffer.py`)
   - 4 champs config × 2 dataclasses
   - `agent.on_new_best()` + `agent._sample_training_batch()` × 2 agents
   - Hook runner B1a
   - 4 flags CLI × 2 scripts
   - 2 smoke CI

3. **"Décisions techniques V2-B1a"** : invariant immutability central, filtre succès strict, threshold "snapshot ready", IS weights snapshot = 1.0, sliding window FIFO N=3, `per_enabled × b1a_enabled` orthogonaux

4. **"V2-B1a — pièges connus"** : 12 pièges spec section 10.1 (synthétisés)

5. **Mise à jour des compteurs** : `pytest -q` attendu **356 passed** (vs 323 baseline V2-B0)

6. **Mise à jour de la liste "modules livrés à ne pas toucher"** : ajouter `mw_ia/training/snapshot_store.py`, `mw_ia/neural/sequence_buffer.py` (extension `concat_batchseq`)

- [ ] **Step 3 — Verify pytest still passes**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `356 passed`.

- [ ] **Step 4 — Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs(v2-b1a): add V2-B1a section to README + CLAUDE.md (code livraison)"
```

---

## Phase 11 — Bench n=5 same-seed 15×15 Bras 3 (B1a seul)

### Task 11 : Exécuter et documenter bench V2-ZY+Polyak+B1a 15×15

**Files:**
- Modify: `CLAUDE.md` (section bench V2-B1a Bras 3)
- Output: `checkpoints/v2b1a_15x15_seed{0..4}.pt` (5 fichiers) + `logs/v2b1a_15x15_seed{0..4}.log`

- [ ] **Step 1 — Run 5 same-seed benches 15×15 (GPU)**

```bash
source .venv/Scripts/activate
mkdir -p checkpoints logs
for seed in 0 1 2 3 4; do
  python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 5000 --mode obstacles --device cuda --seed $seed \
    --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --b1a --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b1a_15x15_seed${seed}.pt \
    2>&1 | tee logs/v2b1a_15x15_seed${seed}.log
done
```

Compute estimé : 5 runs × ~1h RTX 3060 = ~5h GPU total.

- [ ] **Step 2 — Extraire les métriques par seed**

Pour chaque seed, extraire de `logs/v2b1a_15x15_seed{N}.log` :
- Final winrate + diff
- Best @ diff=0.30 winrate + episode capté
- Per-bucket winrate
- Nombre de captures B1a (compter logs `B1a snapshot capture`)

- [ ] **Step 3 — Calculer agrégats n=5**

Calculer :
- Mean best winrate @ diff=0.30
- Std (n−1)
- Min, Max
- Late-stage collapse incidence
- Diff_max training mean

- [ ] **Step 4 — Vérifier critères acceptance Phase 1**

Comparer aux critères pré-enregistrés (spec section 9.3 Phase 1) :
- Mean ≥ **55 %** (vs baseline V2-U 64 %, tolérance −9 pp)
- Min ≥ **30 %** (pas en dessous de V2-B0+PER worst)
- Late-stage collapse = **0/5**
- Diff_max training mean ≥ **0.30**

**Si critères atteints** → continuer Phase 12 (Bras 4). **Si critères ratés** → STOP plan, B1a alone failed. Tag avec finding négatif.

- [ ] **Step 5 — Documenter résultats Bras 3 dans CLAUDE.md**

Ajouter section "V2-B1a — bench n=5 same-seed 15×15 (Bras 3, B1a seul, 2026-05-29)" avec :
- Table résultats par seed
- Table agrégats n=5
- Comparaison vs baseline V2-U 15×15
- Verdict Phase 1 (PASSED / FAILED + détail des critères)

- [ ] **Step 6 — Commit**

```bash
git add CLAUDE.md
git commit -m "docs(v2-b1a): bench V2-ZY+Polyak+B1a 15x15 n=5 — [verdict: PASSED/FAILED summary]"
```

(Adapter le commit message au verdict réel.)

---

## Phase 12 — Bench n=5 same-seed 15×15 Bras 4 (B1a + PER)

### Task 12 : Exécuter et documenter bench V2-ZY+Polyak+B1a+PER 15×15 (test interaction)

**Files:**
- Modify: `CLAUDE.md` (section bench V2-B1a Bras 4 + analyse marginale 2×2)
- Output: `checkpoints/v2b1a_per_15x15_seed{0..4}.pt` + `logs/v2b1a_per_15x15_seed{0..4}.log`

- [ ] **Step 1 — Run 5 same-seed benches 15×15 (GPU)**

```bash
source .venv/Scripts/activate
mkdir -p checkpoints logs
for seed in 0 1 2 3 4; do
  python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 5000 --mode obstacles --device cuda --seed $seed \
    --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --b1a --per --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b1a_per_15x15_seed${seed}.pt \
    2>&1 | tee logs/v2b1a_per_15x15_seed${seed}.log
done
```

Compute estimé : 5 runs × ~1h RTX 3060 = ~5h GPU total.

- [ ] **Step 2 — Extraire les métriques par seed**

Pour chaque seed, extraire (idem Phase 11 step 2).

- [ ] **Step 3 — Calculer agrégats n=5**

Identique Phase 11 step 3.

- [ ] **Step 4 — Vérifier critères acceptance Phase 2**

Comparer aux critères pré-enregistrés (spec section 9.3 Phase 2) :
- Mean > **56 %** (recovery ≥ 10 pp vs V2-B0+PER 46 %)
- Bonus : Mean ≥ 64 % (full recovery)
- Bonus fort : Mean > 64 % AND > Bras 3 mean (positive interaction)

- [ ] **Step 5 — Calculer analyse marginale 2×2 factorielle**

Données :
- Bras 1 (baseline V2-U) : mean 64 % (déjà documenté CLAUDE.md V2-U)
- Bras 2 (V2-B0+PER) : mean 46 % (déjà documenté CLAUDE.md V2-B0)
- Bras 3 (V2-B1a) : mean = ? (Phase 11)
- Bras 4 (V2-B1a+PER) : mean = ? (Phase 12)

Calculer :
```text
PER main effect    = (Bras 2 + Bras 4) / 2 − (Bras 1 + Bras 3) / 2
B1a main effect    = (Bras 3 + Bras 4) / 2 − (Bras 1 + Bras 2) / 2
Interaction        = (Bras 4 + Bras 1) − (Bras 2 + Bras 3)
```

Si `|Interaction| > 10 pp` : interaction significative.

- [ ] **Step 6 — Documenter résultats Bras 4 + analyse marginale dans CLAUDE.md**

Ajouter section "V2-B1a — bench n=5 same-seed 15×15 (Bras 4, B1a + PER interaction, 2026-05-29)" avec :
- Table résultats par seed
- Table agrégats n=5
- Comparaison vs V2-B0+PER seul

Puis section "V2-B1a — Finding scientifique consolidé (factoriel 2×2)" avec :
- Table 4 bras
- Calculs marginal effects + interaction
- Verdict selon matrice de décision (spec section 9.5)
- Story scientifique 3-5 lignes

- [ ] **Step 7 — Commit**

```bash
git add CLAUDE.md
git commit -m "docs(v2-b1a): bench V2-ZY+Polyak+B1a+PER 15x15 n=5 + factorial 2x2 analysis — [verdict summary]"
```

---

## Phase 13 — Tag v0.2.0-b1a + clôture

### Task 13 : Tag livraison V2-B1a

**Files:** aucun (opération git pure).

- [ ] **Step 1 — Verify DoD complète**

Checklist finale (spec section 11) :
- [x] 33 nouveaux tests pytest verts (snapshot_store 12 + concat_batchseq 3 + config 4 + agents B1a parametrized 14)
- [x] `pytest -q` total : 356 passed
- [x] `bash aether/verify_all.sh` → 8 OK
- [x] V2-U et V2-B0 baselines reproductibles sans `--b1a`
- [x] CI : 2 nouveaux smoke jobs passent
- [x] Bench Phase 1 (Bras 3) — 15×15 n=5 documenté
- [x] Bench Phase 2 (Bras 4) — 15×15 n=5 + factoriel 2×2 documenté
- [x] Section V2-B1a complète dans CLAUDE.md
- [x] Section V2-B1a dans README
- [x] Spec dans `docs/superpowers/specs/`
- [x] Plan dans `docs/superpowers/plans/`
- [x] Cartographie bottlenecks RL mise à jour dans `~/.claude/projects/.../memory/projet_mw_ia_phase_dependence_finding.md` (row B1a)

- [ ] **Step 2 — Verify git tree clean**

```bash
git status --short
```

Attendu : nothing to commit (working tree clean).

- [ ] **Step 3 — Update memory file with B1a row**

Éditer `C:\Users\Wilfred\.claude\projects\C--Users-Wilfred-Documents-IA-Inst-MW-IA\memory\projet_mw_ia_phase_dependence_finding.md` : ajouter une ligne dans la "Cartographie des bottlenecks RL" :

```markdown
| V2-B1a | Forgetting / drift politique | Frozen snapshot rehearsal sliding window | Régimes en croissance (15×15) | [verdict bench post-bench] |
```

- [ ] **Step 4 — Commit memory update**

```bash
git status  # Verifier qu'il y a un changement dans memory (techniquement, .claude/ est gitignore donc on ne commit pas ça)
```

Note : le fichier mémoire est dans `~/.claude/`, gitignored. Pas de commit nécessaire — la mise à jour est locale persistante.

- [ ] **Step 5 — Tag**

```bash
git tag v0.2.0-b1a -m "$(cat <<'EOF'
V2-B1a — Policy Snapshot Rehearsal pour V2-ZY+Polyak : code livre + bench n=5 factoriel 2x2

Sous-projet B phase 1a (rehearsal frozen, variante MVP de B1). Implementation
canonique : SnapshotTrajectoryStore avec sliding window FIFO N=3 captures x 50
trajectoires, filtre succes strict (terminated AND total_reward > 0), sample
uniforme, IMMUTABLE after capture.

CODE LIVRAISON : 13 commits, 33 nouveaux tests (356 total vs 323 baseline V2-B0).
Backwards compat strict via b1a_enabled=False default.

BENCH 2x2 factoriel same-seed (PER x B1a) sur 15x15 :

Bras 1 (V2-U baseline, deja documente)         : mean 64 %
Bras 2 (V2-B0 PER seul, deja documente)        : mean 46 %
Bras 3 (V2-B1a seul, NOUVEAU)                  : mean [X] %
Bras 4 (V2-B1a + PER, NOUVEAU)                 : mean [X] %

Analyse marginale :
  PER main effect : [X] pp
  B1a main effect : [X] pp
  Interaction     : [X] pp (signification : [synergie / antagonisme / nulle])

VERDICT : [resume verdict selon matrice de decision spec section 9.5]

Voir CLAUDE.md sections "V2-B1a — bench n=5 same-seed 15x15 (Bras 3)",
"V2-B1a — bench n=5 same-seed 15x15 (Bras 4)" et
"V2-B1a — Finding scientifique consolide (factoriel 2x2)".
EOF
)"
```

(Remplacer `[X]` et `[verdict]` par les vraies données du bench.)

- [ ] **Step 6 — Verify tag**

```bash
git tag --list | grep b1a
git show v0.2.0-b1a --no-patch | head -30
```

Attendu : `v0.2.0-b1a` listé, message complet affiché.

- [ ] **Step 7 — No commit (tag uniquement)**

---

## Self-review checklist

- [x] **Spec coverage** : chaque section de la spec mappée à une task
  - Spec §1.2 Invariant architectural → Task 3 (test_immutability_after_capture)
  - Spec §4 Architecture → Tasks 2-7
  - Spec §5.1 SnapshotTrajectoryStore → Task 3
  - Spec §5.2 concat_batchseq → Task 2
  - Spec §5.3-5.4 Agent constructor + on_new_best → Task 5
  - Spec §5.5 Runner hook → Task 6
  - Spec §5.6-5.7 _sample_training_batch + train loop → Task 5
  - Spec §6 Config & CLI → Task 4 + Task 7
  - Spec §7 Tests → tests intégrés dans chaque task
  - Spec §8 CI smoke → Task 8
  - Spec §9 Bench protocol → Tasks 11-12
  - Spec §11 DoD → Task 13 verification
- [x] **Placeholder scan** : aucun TBD/TODO. Les `[X] %` et `[verdict]` dans Task 13 step 5 sont des marqueurs intentionnels à remplir au moment du tag avec les vraies données.
- [x] **Type consistency** : `SnapshotTrajectoryStore.capture_from()` et `agent.on_new_best()` signatures cohérentes. `_sample_training_batch()` retourne `_TrainingBatch` dataclass cohérent entre Task 5 et la boucle train. `concat_batchseq(a, b) -> BatchSeq` cohérent Task 2 et utilisation Task 5.

---

## Phase totale récapitulée

| Phase | Tâche | Tests cumul | Commits |
|---|---|---|---|
| 1 — Scaffold | 1 | 323 | 1 |
| 2 — concat_batchseq | 1 | 326 (+3) | 1 |
| 3 — SnapshotTrajectoryStore | 1 | 338 (+12) | 1 |
| 4 — Config extension | 1 | 342 (+4) | 1 |
| 5 — Agents PER+B1a integration | 1 | 356 (+14) | 1 |
| 6 — Runner hook | 1 | 356 | 1 |
| 7 — CLI flags | 1 | 356 | 1 |
| 8 — CI smoke | 1 | 356 | 1 |
| 9 — Sanity verification | 1 | 356 | 0 |
| 10 — Doc README + CLAUDE.md | 1 | 356 | 1 |
| 11 — Bench Bras 3 (B1a seul) | 1 | 356 | 1 |
| 12 — Bench Bras 4 (B1a+PER) + factoriel 2×2 | 1 | 356 | 1 |
| 13 — Tag v0.2.0-b1a | 1 | 356 | 0 (tag uniquement) |

**Total** : 13 tâches, 33 nouveaux tests (356 final), 11 commits + 1 tag.

**Compute total bench** : ~10 h GPU RTX 3060 (5h Bras 3 + 5h Bras 4).
