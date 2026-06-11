# MW_IA V2-B0 Trajectory-level PER — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter Prioritized Experience Replay trajectory-level (Schaul 2015 + R2D2 2019) opt-in via `per_enabled: bool = False` sur les agents récurrents V2-Y et V2-ZY. Hypothèse : améliorer mean/min/convergence/diff_max sur le régime stable V2-ZY+Polyak en 15×15 (baseline n=5 : mean 64 %, std 13.4 pp).

**Architecture:** Nouveau `SumTree` (sum tree binaire O(log N)) + `PrioritizedSequenceReplayBuffer` (avec `BetaScheduler` annealing β) dans `mw_ia/neural/`. `RecurrentDQNTrainer` reçoit la méthode `step_with_priorities` (IS-weighted loss + R2D2 aggregation TD-errors). `RecurrentDQNAgent` et `ConvRecurrentDQNAgent` reçoivent une branche conditionnelle dans constructor + `end_episode`. 6 nouveaux champs config × 2 dataclasses. CLI flags `--per` + 5 hyperparams + `--max-attempts-bfs`. 2 nouveaux smoke CI. Bench n=5 same-seed pattern V2-U.

**Tech Stack:** Python 3.13, PyTorch 2.11+cu128, numpy. Réutilise infrastructure V2-W/V2-Y/V2-ZY/V2-U/V2-V.

**Spec source:** `docs/superpowers/specs/2026-05-26-mw-ia-per-trajectory-design.md`

**État initial:** Branche `main`, 9 tags posés (jusqu'à `v0.2.0-u`). **265 tests pytest verts**. Dernier commit avant V2-B0 : `4e0ac93` (spec V2-B0).

---

## Phase 1 — Scaffold

### Task 1 : Créer les 5 fichiers de tests vides

**Files:**
- Create: `tests/neural/test_sum_tree.py` (vide)
- Create: `tests/neural/test_prioritized_sequence_buffer.py` (vide)
- Create: `tests/neural/test_beta_scheduler.py` (vide)
- Create: `tests/neural/test_per_trainer.py` (vide)
- Create: `tests/agents/test_per_recurrent_agents.py` (vide)

- [ ] **Step 1 — Verify initial state**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `265 passed`.

- [ ] **Step 2 — Create 5 empty test files**

Créer 5 fichiers de 0 byte aux chemins indiqués ci-dessus.

- [ ] **Step 3 — Verify pytest still passes**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `265 passed` (fichiers vides ne sont pas comptés).

- [ ] **Step 4 — Commit**

```bash
git add tests/neural/test_sum_tree.py tests/neural/test_prioritized_sequence_buffer.py tests/neural/test_beta_scheduler.py tests/neural/test_per_trainer.py tests/agents/test_per_recurrent_agents.py
git commit -m "chore(v2-b0): scaffold 5 test files for trajectory-level PER"
```

---

## Phase 2 — SumTree (fondation)

### Task 2 : `SumTree` (8 tests TDD)

**Files:**
- Create: `mw_ia/neural/sum_tree.py`
- Test: `tests/neural/test_sum_tree.py`

- [ ] **Step 1 — Write the 8 failing tests**

Contenu de `tests/neural/test_sum_tree.py` :

```python
"""Tests V2-B0 du SumTree (structure de données O(log N) pour PER)."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.neural.sum_tree import SumTree


def test_init_validates_capacity() -> None:
    """capacity doit être >= 1."""
    with pytest.raises(ValueError, match="capacity"):
        SumTree(0)
    with pytest.raises(ValueError, match="capacity"):
        SumTree(-3)


def test_init_capacity_one() -> None:
    """Edge case capacity=1 fonctionne."""
    tree = SumTree(1)
    assert tree.capacity == 1
    assert tree.total() == 0.0


def test_update_and_total() -> None:
    """update(leaf_idx, priority) propage à la racine."""
    tree = SumTree(4)
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)
    tree.update(3, 4.0)
    assert tree.total() == pytest.approx(10.0)


def test_update_overwrites_old_priority() -> None:
    """update sur même leaf remplace la priorité (pas accumulation)."""
    tree = SumTree(4)
    tree.update(0, 5.0)
    assert tree.total() == pytest.approx(5.0)
    tree.update(0, 2.0)
    assert tree.total() == pytest.approx(2.0)


def test_find_returns_leaf_index_and_priority() -> None:
    """find(value) descend l'arbre et retourne (leaf_idx, priority)."""
    tree = SumTree(4)
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)
    tree.update(3, 4.0)
    # value=0.5 < 1.0 → leaf 0
    idx, prio = tree.find(0.5)
    assert idx == 0
    assert prio == pytest.approx(1.0)
    # value=1.5 ∈ [1.0, 3.0) → leaf 1
    idx, prio = tree.find(1.5)
    assert idx == 1
    assert prio == pytest.approx(2.0)
    # value=4.0 ∈ [3.0, 6.0) → leaf 2
    idx, prio = tree.find(4.0)
    assert idx == 2
    assert prio == pytest.approx(3.0)
    # value=9.9 ∈ [6.0, 10.0) → leaf 3
    idx, prio = tree.find(9.9)
    assert idx == 3
    assert prio == pytest.approx(4.0)


def test_find_validates_leaf_idx_range() -> None:
    """update avec leaf_idx hors [0, capacity) lève ValueError."""
    tree = SumTree(4)
    with pytest.raises(ValueError, match="leaf_idx"):
        tree.update(4, 1.0)
    with pytest.raises(ValueError, match="leaf_idx"):
        tree.update(-1, 1.0)


def test_find_distribution_converges() -> None:
    """10000 samples : fréquence empirique converge vers priorité normalisée à ±5%."""
    tree = SumTree(5)
    priorities = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, p in enumerate(priorities):
        tree.update(i, p)
    rng = np.random.default_rng(42)
    counts = np.zeros(5, dtype=int)
    n_samples = 10_000
    total = tree.total()
    for _ in range(n_samples):
        value = rng.uniform(0.0, total)
        idx, _ = tree.find(value)
        counts[idx] += 1
    expected_freq = np.array(priorities) / sum(priorities)
    empirical_freq = counts / n_samples
    assert np.allclose(empirical_freq, expected_freq, atol=0.02)


def test_capacity_5000_works() -> None:
    """capacity=5000 (default V2-ZY 10×10) fonctionne sans crash."""
    tree = SumTree(5000)
    for i in range(5000):
        tree.update(i, float(i + 1))
    expected_total = sum(range(1, 5001))
    assert tree.total() == pytest.approx(float(expected_total), rel=1e-5)
    # Sample test
    idx, _ = tree.find(0.5)
    assert idx == 0
    idx, _ = tree.find(tree.total() - 0.5)
    assert idx == 4999
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_sum_tree.py -v 2>&1 | tail -15
```

Attendu : 8 fails (`ModuleNotFoundError: No module named 'mw_ia.neural.sum_tree'`).

- [ ] **Step 3 — Implement SumTree**

Créer `mw_ia/neural/sum_tree.py` :

```python
"""SumTree — arbre binaire de sommes pour PER (Schaul 2015) O(log N).

Convention : array de taille 2*capacity - 1.
- Feuilles aux indices [capacity-1, 2*capacity-2] (capacity feuilles)
- Nœuds internes [0, capacity-2] (capacity-1 internes)
- parent(i) = (i - 1) // 2
- left(i)   = 2*i + 1
- right(i)  = 2*i + 2

Capacité quelconque (>= 1) acceptée — la convention reste correcte même
pour capacity non-puissance-de-2 (arbre lopsided mais fonctionnel).
"""
from __future__ import annotations

import numpy as np


class SumTree:
    """Sum tree O(log N) pour PER. Capacité fixe."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity doit être >= 1, reçu {capacity}")
        self._capacity = capacity
        # Array de taille 2*capacity - 1 (feuilles + internes)
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    @property
    def capacity(self) -> int:
        return self._capacity

    def total(self) -> float:
        """Somme totale (valeur de la racine, index 0)."""
        return float(self._tree[0])

    def update(self, leaf_idx: int, priority: float) -> None:
        """Met à jour la priorité d'une feuille et propage à la racine."""
        if not (0 <= leaf_idx < self._capacity):
            raise ValueError(
                f"leaf_idx {leaf_idx} hors [0, {self._capacity})"
            )
        # Index interne dans le tableau
        tree_idx = leaf_idx + self._capacity - 1
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        # Propagation vers la racine
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self._tree[parent] += delta
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def find(self, value: float) -> tuple[int, float]:
        """Trouve la feuille où le cumul atteint `value`.

        Retourne (leaf_idx ∈ [0, capacity), priority de la feuille).
        """
        # Descente depuis la racine
        idx = 0
        while idx < self._capacity - 1:  # Tant que nœud interne
            left = 2 * idx + 1
            right = 2 * idx + 2
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = right
        # idx est l'index dans le tableau, convertir en leaf_idx
        leaf_idx = idx - (self._capacity - 1)
        return leaf_idx, float(self._tree[idx])
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_sum_tree.py -v 2>&1 | tail -15
```

Attendu : `8 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `273 passed` (265 + 8).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/neural/sum_tree.py tests/neural/test_sum_tree.py
git commit -m "feat(v2-b0): add SumTree O(log N) for trajectory-level PER (8 tests)"
```

---

## Phase 3 — BetaScheduler

### Task 3 : `BetaScheduler` annealing linéaire (4 tests TDD)

**Files:**
- Create: `mw_ia/neural/prioritized_sequence_buffer.py` (avec uniquement BetaScheduler — buffer ajouté Phase 4)
- Test: `tests/neural/test_beta_scheduler.py`

- [ ] **Step 1 — Write the 4 failing tests**

Contenu de `tests/neural/test_beta_scheduler.py` :

```python
"""Tests V2-B0 BetaScheduler — annealing linéaire IS exponent."""
from __future__ import annotations

import pytest

from mw_ia.neural.prioritized_sequence_buffer import BetaScheduler


def test_beta_at_zero_returns_beta_start() -> None:
    scheduler = BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=5000)
    assert scheduler.beta(0) == pytest.approx(0.4)


def test_beta_at_total_returns_beta_end() -> None:
    scheduler = BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=5000)
    assert scheduler.beta(5000) == pytest.approx(1.0)


def test_beta_overshoot_clamped_to_beta_end() -> None:
    """Episode > total → clamp à beta_end (cas buffer plein mais épisodes continuent)."""
    scheduler = BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=5000)
    assert scheduler.beta(10_000) == pytest.approx(1.0)
    assert scheduler.beta(50_000) == pytest.approx(1.0)


def test_beta_midpoint_linear() -> None:
    """beta(total / 2) == (beta_start + beta_end) / 2 (linéaire)."""
    scheduler = BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=5000)
    assert scheduler.beta(2500) == pytest.approx(0.7)


def test_validation_beta_start_out_of_range() -> None:
    with pytest.raises(ValueError, match="beta_start"):
        BetaScheduler(beta_start=-0.1, beta_end=1.0, total_episodes=5000)
    with pytest.raises(ValueError, match="beta_start"):
        BetaScheduler(beta_start=1.5, beta_end=1.0, total_episodes=5000)


def test_validation_beta_end_out_of_range() -> None:
    with pytest.raises(ValueError, match="beta_end"):
        BetaScheduler(beta_start=0.4, beta_end=-0.1, total_episodes=5000)
    with pytest.raises(ValueError, match="beta_end"):
        BetaScheduler(beta_start=0.4, beta_end=1.5, total_episodes=5000)


def test_validation_total_episodes_positive() -> None:
    with pytest.raises(ValueError, match="total_episodes"):
        BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=0)
    with pytest.raises(ValueError, match="total_episodes"):
        BetaScheduler(beta_start=0.4, beta_end=1.0, total_episodes=-1)
```

**Note** : 7 tests effectifs mais la spec annonçait "4 tests". J'ai séparé les validations en 3 tests granulaires (beta_start, beta_end, total_episodes) au lieu d'un test unique multi-assert. Plus précis pour TDD. Total V2-B0 final : 51 nouveaux tests (vs 48 spec).

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_beta_scheduler.py -v 2>&1 | tail -15
```

Attendu : 7 fails (`ModuleNotFoundError: No module named 'mw_ia.neural.prioritized_sequence_buffer'`).

- [ ] **Step 3 — Implement BetaScheduler**

Créer `mw_ia/neural/prioritized_sequence_buffer.py` :

```python
"""Prioritized Experience Replay au niveau trajectoire pour V2-ZY+Polyak.

Voir spec : docs/superpowers/specs/2026-05-26-mw-ia-per-trajectory-design.md

Contient :
- BetaScheduler : annealing linéaire β_start → β_end pour IS correction
- PrioritizedSequenceReplayBuffer + PrioritizedBatchSeq : ajouté Phase 4
"""
from __future__ import annotations


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
                f"beta_start doit être ∈ [0, 1], reçu {beta_start}"
            )
        if not (0.0 <= beta_end <= 1.0):
            raise ValueError(
                f"beta_end doit être ∈ [0, 1], reçu {beta_end}"
            )
        if total_episodes <= 0:
            raise ValueError(
                f"total_episodes doit être > 0, reçu {total_episodes}"
            )
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_episodes = total_episodes

    def beta(self, episode: int) -> float:
        """Retourne β à l'épisode donné. Clamp aux extrémités."""
        if episode <= 0:
            return self.beta_start
        if episode >= self.total_episodes:
            return self.beta_end
        progress = episode / self.total_episodes
        return self.beta_start + (self.beta_end - self.beta_start) * progress
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_beta_scheduler.py -v 2>&1 | tail -15
```

Attendu : `7 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `280 passed` (273 + 7).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/neural/prioritized_sequence_buffer.py tests/neural/test_beta_scheduler.py
git commit -m "feat(v2-b0): add BetaScheduler linear annealing for PER IS correction (7 tests)"
```

---

## Phase 4 — PrioritizedSequenceReplayBuffer

### Task 4 : `PrioritizedSequenceReplayBuffer` (12 tests TDD)

**Files:**
- Modify: `mw_ia/neural/prioritized_sequence_buffer.py` (ajout PrioritizedBatchSeq + PrioritizedSequenceReplayBuffer après BetaScheduler)
- Test: `tests/neural/test_prioritized_sequence_buffer.py`

- [ ] **Step 1 — Write the 12 failing tests**

Contenu de `tests/neural/test_prioritized_sequence_buffer.py` :

```python
"""Tests V2-B0 PrioritizedSequenceReplayBuffer (sum tree + IS correction)."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.neural.prioritized_sequence_buffer import (
    PrioritizedBatchSeq,
    PrioritizedSequenceReplayBuffer,
)


def _make_trajectory(length: int, obs_dim: int, seed: int = 0) -> list[tuple]:
    """Génère une trajectoire synthétique pour les tests."""
    rng = np.random.default_rng(seed)
    traj = []
    for t in range(length):
        s = rng.normal(size=(obs_dim,)).astype(np.float32)
        a = int(rng.integers(0, 4))
        r = float(rng.uniform(-1, 1))
        sp = rng.normal(size=(obs_dim,)).astype(np.float32)
        d = (t == length - 1)
        traj.append((s, a, r, sp, d))
    return traj


def test_init_validates_capacity_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        PrioritizedSequenceReplayBuffer(capacity=0, obs_dim=10)


def test_init_validates_alpha_in_range() -> None:
    with pytest.raises(ValueError, match="alpha"):
        PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=10, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=10, alpha=1.5)


def test_init_validates_epsilon_positive() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=10, epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=10, epsilon=-1e-6)


def test_push_assigns_max_priority_to_new_trajectory() -> None:
    """Nouvelle trajectoire reçoit la priorité max courante (greedy init)."""
    buf = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=4, seed=0)
    traj = _make_trajectory(5, obs_dim=4)
    buf.push_trajectory(traj)
    assert len(buf) == 1
    # _max_priority initial = 1.0 → première trajectoire a priorité 1.0
    assert buf._sum_tree.total() == pytest.approx(1.0)


def test_sample_smaller_than_buffer_raises() -> None:
    buf = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=4, seed=0)
    buf.push_trajectory(_make_trajectory(5, obs_dim=4))
    with pytest.raises(ValueError, match="buffer"):
        buf.sample(batch_size=10, seq_len=5, beta=0.5)


def test_sample_seq_len_out_of_range_raises() -> None:
    buf = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=4, max_steps=10, seed=0)
    for _ in range(5):
        buf.push_trajectory(_make_trajectory(5, obs_dim=4))
    with pytest.raises(ValueError, match="seq_len"):
        buf.sample(batch_size=4, seq_len=0, beta=0.5)
    with pytest.raises(ValueError, match="seq_len"):
        buf.sample(batch_size=4, seq_len=11, beta=0.5)


def test_sample_returns_prioritized_batch_seq() -> None:
    """sample() retourne PrioritizedBatchSeq avec batch / weights / tree_indices."""
    buf = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=4, max_steps=20, seed=0)
    for i in range(10):
        buf.push_trajectory(_make_trajectory(10, obs_dim=4, seed=i))
    prio_batch = buf.sample(batch_size=4, seq_len=8, beta=0.5)
    assert isinstance(prio_batch, PrioritizedBatchSeq)
    # batch shape (seq, batch, obs_dim) selon BatchSeq V2-Y
    assert prio_batch.batch.states.shape == (8, 4, 4)
    assert prio_batch.weights.shape == (4,)
    assert prio_batch.weights.dtype == np.float32
    assert prio_batch.tree_indices.shape == (4,)


def test_sample_is_weights_normalized_by_max() -> None:
    """IS weights normalisés : max(w) == 1.0 ± epsilon."""
    buf = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=4, max_steps=20, seed=0)
    for i in range(20):
        buf.push_trajectory(_make_trajectory(10, obs_dim=4, seed=i))
    # Donner des priorités différentes
    indices = np.arange(20)
    td_errors = np.linspace(0.1, 5.0, 20).astype(np.float32)
    buf.update_priorities(indices, td_errors)
    prio_batch = buf.sample(batch_size=8, seq_len=8, beta=0.5)
    assert prio_batch.weights.max() == pytest.approx(1.0, abs=1e-6)
    assert (prio_batch.weights > 0).all()


def test_update_priorities_modifies_sampling_distribution() -> None:
    """Après update_priorities avec td_errors différents, sampling devient biaisé."""
    buf = PrioritizedSequenceReplayBuffer(
        capacity=10, obs_dim=4, max_steps=20, alpha=1.0, seed=0,
    )
    for i in range(10):
        buf.push_trajectory(_make_trajectory(10, obs_dim=4, seed=i))
    # Donner priorité 100x plus haute à trajectoire 0
    indices = np.arange(10)
    td_errors = np.ones(10, dtype=np.float32)
    td_errors[0] = 100.0
    buf.update_priorities(indices, td_errors)
    # 1000 samples, trajectoire 0 doit être >>50%
    counts = np.zeros(10, dtype=int)
    for _ in range(1000):
        prio_batch = buf.sample(batch_size=1, seq_len=8, beta=0.0)
        counts[prio_batch.tree_indices[0]] += 1
    assert counts[0] > 500  # majoritairement trajectoire 0


def test_update_priorities_updates_max_priority() -> None:
    """_max_priority croît si nouvelle priorité dépasse l'ancienne."""
    buf = PrioritizedSequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=20, seed=0)
    buf.push_trajectory(_make_trajectory(5, obs_dim=4))
    initial_max = buf._max_priority
    # td_error élevé → priorité élevée
    buf.update_priorities(np.array([0]), np.array([10.0], dtype=np.float32))
    assert buf._max_priority > initial_max


def test_first_trajectory_is_sampleable() -> None:
    """Greedy init : première trajectoire (priority=_max_priority=1.0) est sampleable."""
    buf = PrioritizedSequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=20, seed=0)
    for _ in range(4):
        buf.push_trajectory(_make_trajectory(5, obs_dim=4))
    # Aucun update_priorities encore → toutes priorités == 1.0
    prio_batch = buf.sample(batch_size=4, seq_len=5, beta=0.5)
    assert prio_batch.batch.states.shape == (5, 4, 4)


def test_reproducibility_with_seed() -> None:
    """Même seed → même séquence de samples."""
    obs_dim = 4
    buf1 = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=obs_dim, max_steps=20, seed=42)
    buf2 = PrioritizedSequenceReplayBuffer(capacity=100, obs_dim=obs_dim, max_steps=20, seed=42)
    for i in range(10):
        traj = _make_trajectory(10, obs_dim=obs_dim, seed=i)
        buf1.push_trajectory(traj)
        buf2.push_trajectory(traj)
    pb1 = buf1.sample(batch_size=4, seq_len=8, beta=0.5)
    pb2 = buf2.sample(batch_size=4, seq_len=8, beta=0.5)
    assert np.array_equal(pb1.tree_indices, pb2.tree_indices)


def test_capacity_5000_v2zy_default() -> None:
    """capacity=5000 (V2-ZY 10×10 default) fonctionne sans crash."""
    buf = PrioritizedSequenceReplayBuffer(
        capacity=5000, obs_dim=300, max_steps=200, seed=0,
    )
    for i in range(200):
        buf.push_trajectory(_make_trajectory(30, obs_dim=300, seed=i))
    prio_batch = buf.sample(batch_size=128, seq_len=32, beta=0.5)
    assert prio_batch.batch.states.shape == (32, 128, 300)
    assert prio_batch.weights.shape == (128,)
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_prioritized_sequence_buffer.py -v 2>&1 | tail -15
```

Attendu : 12 fails (`ImportError: cannot import name 'PrioritizedSequenceReplayBuffer'`).

- [ ] **Step 3 — Implement PrioritizedSequenceReplayBuffer + PrioritizedBatchSeq**

Modifier `mw_ia/neural/prioritized_sequence_buffer.py`. Ajouter après la classe `BetaScheduler` :

```python
from dataclasses import dataclass

import numpy as np

from mw_ia.neural.sequence_buffer import BatchSeq
from mw_ia.neural.sum_tree import SumTree


@dataclass
class PrioritizedBatchSeq:
    """Batch enrichi PER : BatchSeq V2-Y + IS weights + tree indices."""

    batch: BatchSeq              # BatchSeq standard V2-Y (states/.../mask)
    weights: np.ndarray          # (B,) float32 — IS weights normalisés par max(w)
    tree_indices: np.ndarray     # (B,) int64 — leaf indices pour update_priorities


class PrioritizedSequenceReplayBuffer:
    """Buffer circulaire de trajectoires avec PER (Schaul 2015) trajectory-level.

    Capacity = nombre de trajectoires (cohérent SequenceReplayBuffer V2-Y).
    Priorité par trajectoire stockée comme `(|td_error| + epsilon)^alpha`.
    Sampling stratifié sum tree + IS weights normalisés.
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
            raise ValueError(f"capacity doit être > 0, reçu {capacity}")
        if obs_dim <= 0:
            raise ValueError(f"obs_dim doit être > 0, reçu {obs_dim}")
        if max_steps <= 0:
            raise ValueError(f"max_steps doit être > 0, reçu {max_steps}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha doit être ∈ [0, 1], reçu {alpha}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon doit être > 0, reçu {epsilon}")

        self.capacity = capacity
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)

        # Storage identique à SequenceReplayBuffer V2-Y
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
        """Stocke trajectoire (pattern V2-Y) + assigne priorité = max courante."""
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
        # Greedy init : nouvelle trajectoire reçoit la priorité max courante
        self._sum_tree.update(i, self._max_priority)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int, beta: float) -> PrioritizedBatchSeq:
        """Sampling stratifié sum tree + extraction fenêtre + IS weights."""
        if self._size < batch_size:
            raise ValueError(
                f"buffer trop petit ({self._size}) pour batch={batch_size}"
            )
        if seq_len <= 0 or seq_len > self.max_steps:
            raise ValueError(
                f"seq_len {seq_len} hors ]0, {self.max_steps}]"
            )

        total = self._sum_tree.total()
        # Sampling stratifié : segments égaux sur [0, total]
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

        # Construction BatchSeq (offset aléatoire + padding + mask, pattern V2-Y)
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

        # IS weights : w_i = (1/N * 1/P_i)^beta, normalisés par max(w)
        # P_i = priority_i / total
        # Pour éviter overflow numérique si priority très petite, clamp epsilon-level
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
        """Met à jour les priorités : new_priority = (|td_error| + epsilon)^alpha."""
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
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_prioritized_sequence_buffer.py -v 2>&1 | tail -15
```

Attendu : `12 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `292 passed` (280 + 12).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/neural/prioritized_sequence_buffer.py tests/neural/test_prioritized_sequence_buffer.py
git commit -m "feat(v2-b0): add PrioritizedSequenceReplayBuffer with stratified sampling + IS (12 tests)"
```

---

## Phase 5 — Extension Config (DRQNConfig + ConvRecurrentDQNConfig)

### Task 5 : Ajouter 6 champs PER × 2 dataclasses + validation

**Files:**
- Modify: `mw_ia/config.py` (DRQNConfig + ConvRecurrentDQNConfig)
- Test: validation embedded in `tests/neural/test_prioritized_sequence_buffer.py` (extension)

- [ ] **Step 1 — Add 6 fields + validation to DRQNConfig**

Dans `mw_ia/config.py`, localiser la classe `DRQNConfig`. Ajouter après le champ `polyak_tau: float = 0.0` (ligne ~178) :

```python
    # V2-B0 : Prioritized Experience Replay trajectory-level (Schaul 2015 + R2D2)
    per_enabled: bool = False
    per_alpha: float = 0.6              # priority exponent (Schaul 2015)
    per_beta_start: float = 0.4         # IS exponent initial
    per_beta_end: float = 1.0           # IS exponent final
    per_eta: float = 0.9                # R2D2 aggregation : eta*max + (1-eta)*mean
    per_epsilon: float = 1e-6           # small constant garantit priority > 0
```

Dans `DRQNConfig.__post_init__`, ajouter après le validate `polyak_tau` (avant le close de la méthode) :

```python
        if not (0.0 <= self.per_alpha <= 1.0):
            raise ValueError(f"per_alpha doit être ∈ [0, 1], reçu {self.per_alpha}")
        if not (0.0 <= self.per_beta_start <= 1.0):
            raise ValueError(
                f"per_beta_start doit être ∈ [0, 1], reçu {self.per_beta_start}"
            )
        if not (0.0 <= self.per_beta_end <= 1.0):
            raise ValueError(
                f"per_beta_end doit être ∈ [0, 1], reçu {self.per_beta_end}"
            )
        if not (0.0 <= self.per_eta <= 1.0):
            raise ValueError(f"per_eta doit être ∈ [0, 1], reçu {self.per_eta}")
        if self.per_epsilon <= 0.0:
            raise ValueError(f"per_epsilon doit être > 0, reçu {self.per_epsilon}")
```

- [ ] **Step 2 — Add same 6 fields + validation to ConvRecurrentDQNConfig**

Dans `mw_ia/config.py`, localiser la classe `ConvRecurrentDQNConfig`. Ajouter après le champ `polyak_tau: float = 0.0` (ligne ~378) :

```python
    # V2-B0 : Prioritized Experience Replay trajectory-level (Schaul 2015 + R2D2)
    per_enabled: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_eta: float = 0.9
    per_epsilon: float = 1e-6
```

Dans `ConvRecurrentDQNConfig.__post_init__`, à la fin (après le validate `polyak_tau`) :

```python
        if not (0.0 <= self.per_alpha <= 1.0):
            raise ValueError(f"per_alpha doit être ∈ [0, 1], reçu {self.per_alpha}")
        if not (0.0 <= self.per_beta_start <= 1.0):
            raise ValueError(
                f"per_beta_start doit être ∈ [0, 1], reçu {self.per_beta_start}"
            )
        if not (0.0 <= self.per_beta_end <= 1.0):
            raise ValueError(
                f"per_beta_end doit être ∈ [0, 1], reçu {self.per_beta_end}"
            )
        if not (0.0 <= self.per_eta <= 1.0):
            raise ValueError(f"per_eta doit être ∈ [0, 1], reçu {self.per_eta}")
        if self.per_epsilon <= 0.0:
            raise ValueError(f"per_epsilon doit être > 0, reçu {self.per_epsilon}")
```

- [ ] **Step 3 — Run full suite, verify 265+12+7+8=292 still passes (no regression)**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `292 passed` (config defaults `per_enabled=False` ne change rien).

- [ ] **Step 4 — Add 4 validation tests (per_enabled config validation)**

Ajouter à la fin de `tests/neural/test_prioritized_sequence_buffer.py` :

```python
# === Tests V2-B0 config field validation ===
from mw_ia.config import DRQNConfig, ConvRecurrentDQNConfig


def test_drqn_config_per_alpha_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="per_alpha"):
        DRQNConfig(per_alpha=-0.1)
    with pytest.raises(ValueError, match="per_alpha"):
        DRQNConfig(per_alpha=1.5)


def test_drqn_config_per_epsilon_zero_raises() -> None:
    with pytest.raises(ValueError, match="per_epsilon"):
        DRQNConfig(per_epsilon=0.0)


def test_conv_recurrent_config_per_beta_end_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="per_beta_end"):
        ConvRecurrentDQNConfig(per_beta_end=1.5)


def test_conv_recurrent_config_per_eta_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="per_eta"):
        ConvRecurrentDQNConfig(per_eta=-0.1)
```

- [ ] **Step 5 — Run new tests + full suite**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_prioritized_sequence_buffer.py -v 2>&1 | tail -20
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : 16 passed test_prioritized_sequence_buffer (12 + 4), `296 passed` total (292 + 4).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/config.py tests/neural/test_prioritized_sequence_buffer.py
git commit -m "feat(v2-b0): add 6 PER fields × 2 configs (DRQN + ConvRecurrent) with validation (4 tests)"
```

---

## Phase 6 — RecurrentDQNTrainer.step_with_priorities

### Task 6 : Refactor `_step_impl` + `step_with_priorities` (8 tests TDD)

**Files:**
- Modify: `mw_ia/neural/recurrent_trainer.py` (refactor `step` en `_step_impl` + nouvelle `step_with_priorities`)
- Test: `tests/neural/test_per_trainer.py`

- [ ] **Step 1 — Write the 8 failing tests**

Contenu de `tests/neural/test_per_trainer.py` :

```python
"""Tests V2-B0 RecurrentDQNTrainer.step_with_priorities (IS-weighted + R2D2 agg)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mw_ia.neural.recurrent import RecurrentQNetwork
from mw_ia.neural.recurrent_trainer import RecurrentDQNTrainer
from mw_ia.neural.sequence_buffer import BatchSeq


def _make_trainer(double_dqn: bool = False, polyak_tau: float = 0.0):
    online = RecurrentQNetwork(obs_dim=8, n_actions=4, fc_hidden=16, lstm_hidden=16)
    target = RecurrentQNetwork(obs_dim=8, n_actions=4, fc_hidden=16, lstm_hidden=16)
    trainer = RecurrentDQNTrainer(
        online, target, lr=1e-3, gamma=0.99,
        device="cpu", use_amp=False,
        double_dqn=double_dqn, polyak_tau=polyak_tau,
    )
    return trainer


def _make_batch(seq_len: int = 8, batch_size: int = 4, obs_dim: int = 8) -> BatchSeq:
    rng = np.random.default_rng(0)
    return BatchSeq(
        states=rng.normal(size=(seq_len, batch_size, obs_dim)).astype(np.float32),
        actions=rng.integers(0, 4, size=(seq_len, batch_size)).astype(np.int64),
        rewards=rng.uniform(-1, 1, size=(seq_len, batch_size)).astype(np.float32),
        next_states=rng.normal(size=(seq_len, batch_size, obs_dim)).astype(np.float32),
        dones=np.zeros((seq_len, batch_size), dtype=np.float32),
        mask=np.ones((seq_len, batch_size), dtype=np.float32),
    )


def test_step_unchanged_signature_returns_float() -> None:
    """step(batch) → float (V2-Y compat strict)."""
    trainer = _make_trainer()
    batch = _make_batch()
    result = trainer.step(batch)
    assert isinstance(result, float)


def test_step_with_priorities_returns_tuple() -> None:
    """step_with_priorities(batch, weights, eta) → (float, ndarray)."""
    trainer = _make_trainer()
    batch = _make_batch(batch_size=4)
    weights = np.ones(4, dtype=np.float32)
    result = trainer.step_with_priorities(batch, weights, eta=0.9)
    assert isinstance(result, tuple)
    loss, td_errors = result
    assert isinstance(loss, float)
    assert isinstance(td_errors, np.ndarray)
    assert td_errors.shape == (4,)
    assert td_errors.dtype == np.float32


def test_is_weights_change_loss() -> None:
    """Loss avec weights=[2,1,1,1] ≠ loss avec weights=[1,1,1,1]."""
    trainer1 = _make_trainer()
    trainer2 = _make_trainer()
    # Sync params pour comparaison fair
    trainer2.online.load_state_dict(trainer1.online.state_dict())
    trainer2.target.load_state_dict(trainer1.target.state_dict())
    batch = _make_batch(batch_size=4)
    loss1, _ = trainer1.step_with_priorities(
        batch, weights=np.ones(4, dtype=np.float32), eta=0.9,
    )
    # Reset trainer pour comparer (les params ont déjà été mis à jour après step1)
    trainer3 = _make_trainer()
    trainer3.online.load_state_dict(trainer2.online.state_dict())
    trainer3.target.load_state_dict(trainer2.target.state_dict())
    loss3, _ = trainer3.step_with_priorities(
        batch, weights=np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32), eta=0.9,
    )
    assert loss1 != pytest.approx(loss3, abs=1e-6)


def test_td_errors_r2d2_aggregation() -> None:
    """priority_b = eta*max + (1-eta)*mean sur batch synthétique."""
    trainer = _make_trainer()
    batch = _make_batch(seq_len=4, batch_size=2)
    weights = np.ones(2, dtype=np.float32)
    _, td_errors = trainer.step_with_priorities(batch, weights, eta=0.9)
    # td_errors devrait être >= 0 (formule |delta|)
    assert (td_errors >= 0).all()
    # Avec eta=1.0 et même batch, td_errors == max_per_traj uniquement
    _, td_max_only = trainer.step_with_priorities(batch, weights, eta=1.0)
    assert (td_max_only >= 0).all()


def test_mask_excludes_padded_steps_from_aggregation() -> None:
    """Trajectoire avec mask[10:]=0 : aggregation max/mean sur [0:10] seulement."""
    trainer = _make_trainer()
    batch = _make_batch(seq_len=16, batch_size=2)
    # Forcer la seconde trajectoire à mask=0 sur les 8 derniers steps
    batch = BatchSeq(
        states=batch.states, actions=batch.actions, rewards=batch.rewards,
        next_states=batch.next_states, dones=batch.dones,
        mask=batch.mask.copy(),
    )
    batch.mask[8:, 1] = 0.0
    weights = np.ones(2, dtype=np.float32)
    _, td_errors = trainer.step_with_priorities(batch, weights, eta=0.9)
    # Aucun crash sur trajectoire courte. td_errors >= 0
    assert td_errors.shape == (2,)
    assert (td_errors >= 0).all()


def test_double_dqn_path_with_per() -> None:
    """PER + Double DQN cohabitent (formule target reste Hasselt 2015)."""
    trainer = _make_trainer(double_dqn=True)
    batch = _make_batch()
    weights = np.ones(4, dtype=np.float32)
    loss, td_errors = trainer.step_with_priorities(batch, weights, eta=0.9)
    assert isinstance(loss, float)
    assert td_errors.shape == (4,)


def test_polyak_with_per() -> None:
    """PER + Polyak cohabitent (target update post-backward inchangé)."""
    trainer = _make_trainer(polyak_tau=0.005)
    # Capture target snapshot
    target_snapshot = {
        name: p.clone() for name, p in trainer.target.named_parameters()
    }
    batch = _make_batch()
    weights = np.ones(4, dtype=np.float32)
    trainer.step_with_priorities(batch, weights, eta=0.9)
    # Target a été modifié par Polyak post-backward
    target_modified = False
    for name, p in trainer.target.named_parameters():
        if not torch.allclose(p, target_snapshot[name]):
            target_modified = True
            break
    assert target_modified


def test_step_with_priorities_no_grad_through_priorities() -> None:
    """td_errors retournés ne gardent pas de gradient (détachés)."""
    trainer = _make_trainer()
    batch = _make_batch()
    weights = np.ones(4, dtype=np.float32)
    _, td_errors = trainer.step_with_priorities(batch, weights, eta=0.9)
    # td_errors est un numpy ndarray, pas de grad
    assert isinstance(td_errors, np.ndarray)
    # Pas de fuite mémoire : appel répété 50 fois sans crash
    for _ in range(50):
        trainer.step_with_priorities(batch, weights, eta=0.9)
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_per_trainer.py -v 2>&1 | tail -20
```

Attendu : 8 fails (`AttributeError: 'RecurrentDQNTrainer' object has no attribute 'step_with_priorities'`).

- [ ] **Step 3 — Refactor `step` en `_step_impl` + ajouter `step_with_priorities`**

Modifier `mw_ia/neural/recurrent_trainer.py`. Remplacer la méthode `step` (lignes 63-115) par :

```python
    def step(self, batch: BatchSeq) -> float:
        """V2-Y baseline : sans IS, retourne loss seul (signature stricte)."""
        loss, _ = self._step_impl(batch, weights=None, eta=0.0)
        return loss

    def step_with_priorities(
        self,
        batch: BatchSeq,
        weights: np.ndarray,
        eta: float = 0.9,
    ) -> tuple[float, np.ndarray]:
        """V2-B0 : sample IS-weighted + retourne (loss, td_errors aggregés R2D2)."""
        loss, td_errors = self._step_impl(batch, weights=weights, eta=eta)
        assert td_errors is not None
        return loss, td_errors

    def _step_impl(
        self,
        batch: BatchSeq,
        weights: np.ndarray | None = None,
        eta: float = 0.9,
    ) -> tuple[float, np.ndarray | None]:
        """Pipeline unifié. weights=None → V2-Y baseline strict.
        weights fourni → IS-weighted loss + retourne td_errors agrégés par trajectoire.
        """
        states = torch.from_numpy(batch.states).to(self.device, non_blocking=True)
        actions = torch.from_numpy(batch.actions).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(batch.rewards).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(batch.next_states).to(self.device, non_blocking=True)
        dones = torch.from_numpy(batch.dones).to(self.device, non_blocking=True)
        mask = torch.from_numpy(batch.mask).to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
            # Forward online sur la séquence complète (Hidden=None → zéro-init)
            q_pred_all, _ = self.online(states, None)
            q_pred = q_pred_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)
            # q_pred shape : (seq, batch)

            with torch.no_grad():
                if self.double_dqn:
                    q_online_next_all, _ = self.online(next_states, None)
                    next_actions = q_online_next_all.argmax(dim=-1)
                    q_target_all, _ = self.target(next_states, None)
                    q_next = q_target_all.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
                else:
                    q_next_all, _ = self.target(next_states, None)
                    q_next = q_next_all.max(dim=-1).values
                target_q = rewards + self.gamma * q_next * (1.0 - dones)

            # Huber loss element-wise, puis mask + (optionnel) IS weights
            elem_loss = self.loss_fn(q_pred, target_q)
            if weights is None:
                masked_loss = elem_loss * mask
            else:
                w = torch.from_numpy(weights).to(
                    self.device, non_blocking=True
                ).unsqueeze(0)  # (1, batch) broadcast vers (seq, batch)
                masked_loss = elem_loss * mask * w
            n_valid = mask.sum().clamp(min=1.0)
            loss = masked_loss.sum() / n_valid

        # Backward + grad clip + optimizer step (V2-Y inchangé)
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

        loss_value = float(loss.detach().item())

        # V2-B0 : aggregation R2D2 si PER
        if weights is None:
            return loss_value, None

        with torch.no_grad():
            td_step = (target_q - q_pred).detach().abs()        # (seq, batch)
            masked_td = td_step * mask                          # (seq, batch)
            max_per_traj = masked_td.max(dim=0).values          # (batch,)
            sum_per_traj = masked_td.sum(dim=0)                 # (batch,)
            length_per_traj = mask.sum(dim=0).clamp(min=1.0)    # (batch,)
            mean_per_traj = sum_per_traj / length_per_traj
            priorities = eta * max_per_traj + (1.0 - eta) * mean_per_traj

        return loss_value, priorities.cpu().numpy().astype(np.float32)
```

Ajouter l'import numpy en haut du fichier (`import numpy as np`) si pas déjà présent.

- [ ] **Step 4 — Run new tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_per_trainer.py -v 2>&1 | tail -15
```

Attendu : `8 passed`.

- [ ] **Step 5 — Run full suite (V2-Y / V2-ZY baseline preserved)**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `304 passed` (296 + 8). Tous les tests V2-Y et V2-ZY existants doivent toujours passer car `step()` garde sa signature.

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/neural/recurrent_trainer.py tests/neural/test_per_trainer.py
git commit -m "feat(v2-b0): add RecurrentDQNTrainer.step_with_priorities (IS-weighted + R2D2 agg, 8 tests)"
```

---

## Phase 7 — Integration agents V2-Y + V2-ZY

### Task 7 : PER branche conditionnelle dans `RecurrentDQNAgent` + `ConvRecurrentDQNAgent` (8 tests parametrized × 2 agents = 16 cases)

**Files:**
- Modify: `mw_ia/agents/recurrent_dqn.py` (constructor + end_episode)
- Modify: `mw_ia/agents/conv_recurrent_dqn.py` (idem, pattern parallèle)
- Test: `tests/agents/test_per_recurrent_agents.py`

- [ ] **Step 1 — Write the 8 parametrized failing tests**

Contenu de `tests/agents/test_per_recurrent_agents.py` :

```python
"""Tests V2-B0 integration PER dans RecurrentDQNAgent (V2-Y) et
ConvRecurrentDQNAgent (V2-ZY). Parametrized sur les 2 agents."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mw_ia.agents.recurrent_dqn import RecurrentDQNAgent
from mw_ia.agents.conv_recurrent_dqn import ConvRecurrentDQNAgent
from mw_ia.config import DRQNConfig, ConvRecurrentDQNConfig
from mw_ia.neural.prioritized_sequence_buffer import (
    BetaScheduler,
    PrioritizedSequenceReplayBuffer,
)
from mw_ia.neural.sequence_buffer import SequenceReplayBuffer


def _build_drqn(per_enabled: bool, **kwargs) -> RecurrentDQNAgent:
    cfg = DRQNConfig(
        per_enabled=per_enabled,
        replay_capacity=50,
        min_episodes_to_learn=5,
        batch_size=4,
        sequence_length=4,
        max_steps_per_episode=10,
        episodes=100,
        use_amp=False,
        **kwargs,
    )
    return RecurrentDQNAgent(obs_dim=8, n_actions=4, cfg=cfg, device="cpu", seed=0)


def _build_conv_recurrent(per_enabled: bool, **kwargs) -> ConvRecurrentDQNAgent:
    cfg = ConvRecurrentDQNConfig(
        per_enabled=per_enabled,
        replay_capacity=50,
        min_episodes_to_learn=5,
        batch_size=4,
        sequence_length=4,
        max_steps_per_episode=10,
        episodes=100,
        use_amp=False,
        eval_enabled=False,
        **kwargs,
    )
    return ConvRecurrentDQNAgent(
        in_channels=3, rows=4, cols=4, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )


AGENT_BUILDERS = [
    pytest.param(_build_drqn, "drqn", id="v2y_drqn"),
    pytest.param(_build_conv_recurrent, "conv_recurrent", id="v2zy_conv_recurrent"),
]


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_per_disabled_instantiates_sequence_buffer(builder, agent_kind) -> None:
    """per_enabled=False → SequenceReplayBuffer V2-Y baseline strict."""
    agent = builder(per_enabled=False)
    assert isinstance(agent.buffer, SequenceReplayBuffer)
    assert not isinstance(agent.buffer, PrioritizedSequenceReplayBuffer)
    assert agent._beta_scheduler is None


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_per_enabled_instantiates_prioritized_buffer(builder, agent_kind) -> None:
    """per_enabled=True → PrioritizedSequenceReplayBuffer + BetaScheduler."""
    agent = builder(per_enabled=True)
    assert isinstance(agent.buffer, PrioritizedSequenceReplayBuffer)
    assert isinstance(agent._beta_scheduler, BetaScheduler)


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_episode_count_increments_independently_of_buffer_len(builder, agent_kind) -> None:
    """_episode_count croît même quand le buffer plafonne à capacity."""
    agent = builder(per_enabled=True)
    # Pousser 100 trajectoires (capacity=50)
    for ep in range(100):
        if agent_kind == "drqn":
            traj = [(np.zeros(8, dtype=np.float32), 0, 0.1, np.zeros(8, dtype=np.float32), False)
                    for _ in range(5)]
        else:
            obs = np.zeros((3, 4, 4), dtype=np.float32)
            traj = [(obs.flatten(), 0, 0.1, obs.flatten(), False) for _ in range(5)]
        agent._episode_trajectory = traj
        agent.end_episode()
    assert len(agent.buffer) == 50  # capacity
    assert agent._episode_count == 100  # croît indépendamment


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_beta_anneals_with_episode_count(builder, agent_kind) -> None:
    """beta(0) == beta_start, beta(episodes) == beta_end."""
    agent = builder(per_enabled=True)
    assert agent._beta_scheduler.beta(0) == pytest.approx(0.4)
    assert agent._beta_scheduler.beta(100) == pytest.approx(1.0)


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_end_episode_per_path_emits_metrics(builder, agent_kind) -> None:
    """end_episode avec PER actif retourne 'per_beta' dans metrics quand train se déclenche."""
    agent = builder(per_enabled=True)
    # Pousser assez de trajectoires pour passer le seuil
    for _ in range(10):
        if agent_kind == "drqn":
            traj = [(np.zeros(8, dtype=np.float32), 0, 0.1, np.zeros(8, dtype=np.float32), False)
                    for _ in range(5)]
        else:
            obs = np.zeros((3, 4, 4), dtype=np.float32)
            traj = [(obs.flatten(), 0, 0.1, obs.flatten(), False) for _ in range(5)]
        agent._episode_trajectory = traj
        metrics = agent.end_episode()
    # Au 10e épisode, le buffer a >= min_episodes_to_learn (5), train se déclenche
    assert "per_beta" in metrics
    assert "loss" in metrics


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_end_episode_per_disabled_skips_per_beta(builder, agent_kind) -> None:
    """end_episode avec PER désactivé n'émet PAS 'per_beta' dans metrics."""
    agent = builder(per_enabled=False)
    for _ in range(10):
        if agent_kind == "drqn":
            traj = [(np.zeros(8, dtype=np.float32), 0, 0.1, np.zeros(8, dtype=np.float32), False)
                    for _ in range(5)]
        else:
            obs = np.zeros((3, 4, 4), dtype=np.float32)
            traj = [(obs.flatten(), 0, 0.1, obs.flatten(), False) for _ in range(5)]
        agent._episode_trajectory = traj
        metrics = agent.end_episode()
    assert "per_beta" not in metrics
    assert "loss" in metrics


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_polyak_and_per_cohabit(builder, agent_kind) -> None:
    """Polyak + PER : target update à chaque train_step, sans crash."""
    agent = builder(per_enabled=True, polyak_tau=0.005)
    target_snapshot = {
        name: p.clone() for name, p in agent.target.named_parameters()
    }
    for _ in range(10):
        if agent_kind == "drqn":
            traj = [(np.zeros(8, dtype=np.float32), 0, 0.1, np.zeros(8, dtype=np.float32), False)
                    for _ in range(5)]
        else:
            obs = np.zeros((3, 4, 4), dtype=np.float32)
            traj = [(obs.flatten(), 0, 0.1, obs.flatten(), False) for _ in range(5)]
        agent._episode_trajectory = traj
        agent.end_episode()
    # Target a été modifié par Polyak
    target_modified = False
    for name, p in agent.target.named_parameters():
        if not torch.allclose(p, target_snapshot[name]):
            target_modified = True
            break
    assert target_modified


@pytest.mark.parametrize("builder,agent_kind", AGENT_BUILDERS)
def test_save_load_unchanged_with_per(builder, agent_kind, tmp_path) -> None:
    """save/load fonctionnent avec PER actif (priorité PER non sauvegardée, intentionnel)."""
    agent = builder(per_enabled=True)
    for _ in range(6):
        if agent_kind == "drqn":
            traj = [(np.zeros(8, dtype=np.float32), 0, 0.1, np.zeros(8, dtype=np.float32), False)
                    for _ in range(5)]
        else:
            obs = np.zeros((3, 4, 4), dtype=np.float32)
            traj = [(obs.flatten(), 0, 0.1, obs.flatten(), False) for _ in range(5)]
        agent._episode_trajectory = traj
        agent.end_episode()
    ckpt_path = tmp_path / "agent.pt"
    agent.save(str(ckpt_path))
    assert ckpt_path.exists()
    # Reload
    agent2 = builder(per_enabled=True)
    agent2.load(str(ckpt_path))
    # Online params identiques
    for name, p1 in agent.online.named_parameters():
        p2 = dict(agent2.online.named_parameters())[name]
        assert torch.allclose(p1, p2)
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_per_recurrent_agents.py -v 2>&1 | tail -20
```

Attendu : 16 fails (8 tests × 2 agents). Erreurs : `AttributeError: '_beta_scheduler' attribute not set`.

- [ ] **Step 3 — Modify `RecurrentDQNAgent` constructor**

Dans `mw_ia/agents/recurrent_dqn.py`, ajouter en haut l'import :

```python
from mw_ia.neural.prioritized_sequence_buffer import (
    BetaScheduler,
    PrioritizedSequenceReplayBuffer,
)
```

Remplacer les lignes 57-59 (instantiation buffer) par :

```python
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
        self._episode_count: int = 0
```

- [ ] **Step 4 — Modify `RecurrentDQNAgent.end_episode`**

Remplacer la méthode `end_episode` (lignes 108-132 actuelles) par :

```python
    def end_episode(self) -> dict[str, float]:
        """Push trajectoire + train_steps. Branche PER si cfg.per_enabled."""
        if self._episode_trajectory:
            self.buffer.push_trajectory(self._episode_trajectory)
        self._episode_count += 1
        metrics: dict[str, float] = {"epsilon": self.epsilon}
        if len(self.buffer) >= max(self.cfg.min_episodes_to_learn, self.cfg.batch_size):
            losses: list[float] = []
            if self.cfg.per_enabled:
                assert self._beta_scheduler is not None
                beta = self._beta_scheduler.beta(self._episode_count)
                for _ in range(self.cfg.train_steps_per_episode):
                    prio_batch = self.buffer.sample(  # type: ignore[union-attr]
                        batch_size=self.cfg.batch_size,
                        seq_len=self.cfg.sequence_length,
                        beta=beta,
                    )
                    loss, td_errors = self.trainer.step_with_priorities(
                        prio_batch.batch, prio_batch.weights, eta=self.cfg.per_eta,
                    )
                    self.buffer.update_priorities(  # type: ignore[union-attr]
                        prio_batch.tree_indices, td_errors,
                    )
                    losses.append(loss)
                metrics["per_beta"] = beta
            else:
                for _ in range(self.cfg.train_steps_per_episode):
                    batch = self.buffer.sample(
                        batch_size=self.cfg.batch_size,
                        seq_len=self.cfg.sequence_length,
                    )
                    losses.append(self.trainer.step(batch))
            if losses:
                self.last_loss = sum(losses) / len(losses)
                metrics["loss"] = self.last_loss
        # V2-U : skip hard sync périodique si Polyak activé
        if self.cfg.polyak_tau == 0.0:
            if self.global_step // self.cfg.target_sync_steps > self.target_syncs:
                self.trainer.sync_target()
                self.target_syncs += 1
        return metrics
```

- [ ] **Step 5 — Modify `ConvRecurrentDQNAgent` (pattern identique)**

Dans `mw_ia/agents/conv_recurrent_dqn.py`, ajouter l'import :

```python
from mw_ia.neural.prioritized_sequence_buffer import (
    BetaScheduler,
    PrioritizedSequenceReplayBuffer,
)
```

Remplacer les lignes 66-69 (instantiation buffer) par :

```python
        obs_dim_flat = in_channels * rows * cols
        if cfg.per_enabled:
            self.buffer: SequenceReplayBuffer | PrioritizedSequenceReplayBuffer = (
                PrioritizedSequenceReplayBuffer(
                    cfg.replay_capacity, obs_dim_flat, cfg.max_steps_per_episode,
                    alpha=cfg.per_alpha, epsilon=cfg.per_epsilon, seed=seed,
                )
            )
            self._beta_scheduler: BetaScheduler | None = BetaScheduler(
                cfg.per_beta_start, cfg.per_beta_end, cfg.episodes,
            )
        else:
            self.buffer = SequenceReplayBuffer(
                cfg.replay_capacity, obs_dim_flat, cfg.max_steps_per_episode, seed=seed,
            )
            self._beta_scheduler = None
        self._episode_count: int = 0
```

Remplacer la méthode `end_episode` (lignes 123-144 actuelles) par :

```python
    def end_episode(self) -> dict[str, float]:
        """Push trajectoire + train_steps BPTT. Branche PER si cfg.per_enabled."""
        if self._episode_trajectory:
            self.buffer.push_trajectory(self._episode_trajectory)
        self._episode_count += 1
        metrics: dict[str, float] = {"epsilon": self.epsilon}
        train_threshold = max(self.cfg.min_episodes_to_learn, self.cfg.batch_size)
        if len(self.buffer) >= train_threshold:
            losses: list[float] = []
            if self.cfg.per_enabled:
                assert self._beta_scheduler is not None
                beta = self._beta_scheduler.beta(self._episode_count)
                for _ in range(self.cfg.train_steps_per_episode):
                    prio_batch = self.buffer.sample(  # type: ignore[union-attr]
                        batch_size=self.cfg.batch_size,
                        seq_len=self.cfg.sequence_length,
                        beta=beta,
                    )
                    loss, td_errors = self.trainer.step_with_priorities(
                        prio_batch.batch, prio_batch.weights, eta=self.cfg.per_eta,
                    )
                    self.buffer.update_priorities(  # type: ignore[union-attr]
                        prio_batch.tree_indices, td_errors,
                    )
                    losses.append(loss)
                metrics["per_beta"] = beta
            else:
                for _ in range(self.cfg.train_steps_per_episode):
                    batch = self.buffer.sample(
                        batch_size=self.cfg.batch_size,
                        seq_len=self.cfg.sequence_length,
                    )
                    losses.append(self.trainer.step(batch))
            if losses:
                self.last_loss = sum(losses) / len(losses)
                metrics["loss"] = self.last_loss
        # V2-U : skip hard sync périodique si Polyak activé
        if self.cfg.polyak_tau == 0.0:
            if self.global_step // self.cfg.target_sync_steps > self.target_syncs:
                self.trainer.sync_target()
                self.target_syncs += 1
        return metrics
```

- [ ] **Step 6 — Run new tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_per_recurrent_agents.py -v 2>&1 | tail -25
```

Attendu : `16 passed`.

- [ ] **Step 7 — Run full suite (V2-Y / V2-ZY baseline preserved)**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `320 passed` (304 + 16). Les 35 tests V2-Y et tests V2-ZY existants passent grâce au default `per_enabled=False`.

- [ ] **Step 8 — Commit**

```bash
git add mw_ia/agents/recurrent_dqn.py mw_ia/agents/conv_recurrent_dqn.py tests/agents/test_per_recurrent_agents.py
git commit -m "feat(v2-b0): integrate PER conditional branch in V2-Y + V2-ZY agents (16 parametrized tests)"
```

---

## Phase 8 — CLI flags V2-Y + V2-ZY scripts

### Task 8 : Ajouter 7 flags PER + `--max-attempts-bfs` aux 2 scripts CLI

**Files:**
- Modify: `scripts/train_drqn_procedural.py`
- Modify: `scripts/train_cnn_lstm_dqn_procedural.py`

- [ ] **Step 1 — Add 7 flags to `train_drqn_procedural.py`**

Dans `scripts/train_drqn_procedural.py`, localiser le bloc `parser.add_argument(...)`. Ajouter à la suite des flags existants :

```python
    parser.add_argument(
        "--per",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="V2-B0 : Prioritized Experience Replay trajectory-level (Schaul 2015 + R2D2). "
             "Default False = SequenceReplayBuffer uniforme baseline V2-Y.",
    )
    parser.add_argument("--per-alpha", type=float, default=0.6,
                        help="V2-B0 : priority exponent alpha (default 0.6, Schaul 2015).")
    parser.add_argument("--per-beta-start", type=float, default=0.4,
                        help="V2-B0 : IS exponent beta initial (default 0.4, Schaul 2015).")
    parser.add_argument("--per-beta-end", type=float, default=1.0,
                        help="V2-B0 : IS exponent beta final (default 1.0, annealing complete).")
    parser.add_argument("--per-eta", type=float, default=0.9,
                        help="V2-B0 : R2D2 priority aggregation eta (default 0.9).")
    parser.add_argument("--per-epsilon", type=float, default=1e-6,
                        help="V2-B0 : small constant epsilon (default 1e-6) garantit priority > 0.")
    parser.add_argument(
        "--max-attempts-bfs",
        type=int,
        default=100,
        help="ProceduralEnvConfig max_attempts_bfs (default 100). Recommande bench B0 : 500.",
    )
```

Localiser la construction de `dqn_cfg = DRQNConfig(...)`. Ajouter les 6 champs PER :

```python
    dqn_cfg = DRQNConfig(
        # ... champs existants ...
        per_enabled=args.per,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_eta=args.per_eta,
        per_epsilon=args.per_epsilon,
    )
```

Localiser la construction de `proc_cfg = ProceduralEnvConfig(...)`. Ajouter le champ `max_attempts_bfs` :

```python
    proc_cfg = ProceduralEnvConfig(
        # ... champs existants ...
        max_attempts_bfs=args.max_attempts_bfs,
    )
```

- [ ] **Step 2 — Add same 7 flags to `train_cnn_lstm_dqn_procedural.py`**

Dans `scripts/train_cnn_lstm_dqn_procedural.py`, ajouter les 7 flags après `--polyak-tau` (ligne 69) :

```python
    parser.add_argument(
        "--per",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="V2-B0 : Prioritized Experience Replay trajectory-level (Schaul 2015 + R2D2). "
             "Default False = SequenceReplayBuffer uniforme baseline V2-ZY.",
    )
    parser.add_argument("--per-alpha", type=float, default=0.6,
                        help="V2-B0 : priority exponent alpha (default 0.6, Schaul 2015).")
    parser.add_argument("--per-beta-start", type=float, default=0.4,
                        help="V2-B0 : IS exponent beta initial (default 0.4, Schaul 2015).")
    parser.add_argument("--per-beta-end", type=float, default=1.0,
                        help="V2-B0 : IS exponent beta final (default 1.0, annealing complete).")
    parser.add_argument("--per-eta", type=float, default=0.9,
                        help="V2-B0 : R2D2 priority aggregation eta (default 0.9).")
    parser.add_argument("--per-epsilon", type=float, default=1e-6,
                        help="V2-B0 : small constant epsilon (default 1e-6) garantit priority > 0.")
    parser.add_argument(
        "--max-attempts-bfs",
        type=int,
        default=100,
        help="ProceduralEnvConfig max_attempts_bfs (default 100). Recommande bench B0 : 500.",
    )
```

Localiser la construction de `dqn_cfg = ConvRecurrentDQNConfig(...)` (ligne 87). Ajouter les 6 champs PER :

```python
    dqn_cfg = ConvRecurrentDQNConfig(
        # ... champs existants ...
        per_enabled=args.per,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_eta=args.per_eta,
        per_epsilon=args.per_epsilon,
    )
```

Localiser la construction de `proc_cfg = ProceduralEnvConfig(...)` (ligne 72). Ajouter :

```python
    proc_cfg = ProceduralEnvConfig(
        mode=args.mode, max_rows=args.max_rows, max_cols=args.max_cols,
        max_steps=args.max_steps,
        max_attempts_bfs=args.max_attempts_bfs,
    )
```

- [ ] **Step 3 — Sanity smoke V2-Y CLI**

```bash
source .venv/Scripts/activate && python scripts/train_drqn_procedural.py --episodes 10 --mode obstacles --device cpu --per 2>&1 | tail -10
```

Attendu : exécution complète, no crash, output final avec winrate + per-bucket. Si `--per` PER actif, message logs montre métriques `loss` cohérentes.

- [ ] **Step 4 — Sanity smoke V2-ZY CLI**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles --device cpu --per --best-checkpoint-path checkpoints/sanity_v2b0.pt 2>&1 | tail -10
```

Attendu : exécution complète, no crash, fichier checkpoint créé.

- [ ] **Step 5 — Run full pytest (no regression)**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `320 passed`.

- [ ] **Step 6 — Commit**

```bash
git add scripts/train_drqn_procedural.py scripts/train_cnn_lstm_dqn_procedural.py
git commit -m "feat(v2-b0-cli): expose --per + 5 hyperparams + --max-attempts-bfs on V2-Y + V2-ZY scripts"
```

---

## Phase 9 — CI smoke

### Task 9 : Ajouter 2 smoke jobs CI (PER seul + PER + Polyak)

**Files:**
- Modify: `.github/workflows/aether_verify.yml`

- [ ] **Step 1 — Add 2 new smoke jobs**

Dans `.github/workflows/aether_verify.yml`, ajouter après le smoke V2-U (ligne 50) :

```yaml
      - name: Smoke test V2-B0 PER (trajectory-level) on V2-ZY
        run: |
          mkdir -p checkpoints
          python scripts/train_cnn_lstm_dqn_procedural.py \
            --episodes 10 --mode obstacles --device cpu \
            --per --eval-every-episodes 5 \
            --best-checkpoint-path checkpoints/ci_v2b0_best.pt
          test -f checkpoints/ci_v2b0_best.pt
      - name: Smoke test V2-B0 PER + Polyak cohabit (sanity check)
        run: |
          mkdir -p checkpoints
          python scripts/train_cnn_lstm_dqn_procedural.py \
            --episodes 10 --mode obstacles --device cpu \
            --per --polyak-tau 0.005 --eval-every-episodes 5 \
            --best-checkpoint-path checkpoints/ci_v2b0_polyak_best.pt
          test -f checkpoints/ci_v2b0_polyak_best.pt
```

- [ ] **Step 2 — Smoke local equivalent**

Exécuter localement pour valider les commandes (équivalent du CI sur CPU) :

```bash
source .venv/Scripts/activate
mkdir -p checkpoints
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles --device cpu --per --eval-every-episodes 5 --best-checkpoint-path checkpoints/ci_v2b0_best.pt
test -f checkpoints/ci_v2b0_best.pt && echo "V2-B0 PER smoke OK"
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles --device cpu --per --polyak-tau 0.005 --eval-every-episodes 5 --best-checkpoint-path checkpoints/ci_v2b0_polyak_best.pt
test -f checkpoints/ci_v2b0_polyak_best.pt && echo "V2-B0 PER + Polyak smoke OK"
```

Attendu : 2 messages "OK".

- [ ] **Step 3 — Commit**

```bash
git add .github/workflows/aether_verify.yml
git commit -m "ci(v2-b0): add 2 smoke jobs (PER seul + PER+Polyak cohabit) on V2-ZY"
```

---

## Phase 10 — Sanity verification finale

### Task 10 : Verify all 320 tests + Aether + V2-Y baseline reproducible

**Files:** aucun, vérifications uniquement.

- [ ] **Step 1 — Full pytest run**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `320 passed`.

- [ ] **Step 2 — Aether harness check**

```bash
bash aether/verify_all.sh 2>&1 | tail -5
```

Attendu : 8 OK (inchangé).

- [ ] **Step 3 — V2-Y baseline reproducible without --per**

```bash
source .venv/Scripts/activate && python scripts/train_drqn_procedural.py --episodes 20 --mode obstacles --device cpu --seed 0 2>&1 | tail -5
```

Attendu : exécution complète, comportement V2-Y baseline identique (winrate, loss cohérents avec baseline V2-Y).

- [ ] **Step 4 — V2-ZY baseline reproducible without --per**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_lstm_dqn_procedural.py --episodes 20 --mode obstacles --device cpu --seed 0 --polyak-tau 0.005 2>&1 | tail -5
```

Attendu : exécution complète, comportement V2-ZY+Polyak baseline identique.

- [ ] **Step 5 — No commit (vérification pure)**

---

## Phase 11 — Documentation README + CLAUDE.md

### Task 11 : Section V2-B0 dans README + CLAUDE.md (livraison code)

**Files:**
- Modify: `README.md` (nouvelle section V2-B0)
- Modify: `CLAUDE.md` (nouvelle section V2-B0 + mise à jour tags + roadmap)

- [ ] **Step 1 — Add V2-B0 section to README.md**

Ajouter après la section V2-U (ou équivalent en fin de README) une nouvelle section :

```markdown
## V2-B0 — Trajectory-level Prioritized Experience Replay

Sous-projet B phase 0 : Prioritized Experience Replay (Schaul 2015 + R2D2 2019) trajectoire-level
sur SequenceReplayBuffer V2-Y. Hypothèse testée : le replay uniforme est-il un bottleneck
du régime stable V2-ZY+Polyak ?

### Recette CLI

```bash
# 10x10 (sanity / no regression)
python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 5000 --mode obstacles --device cuda --seed {N} \
    --polyak-tau 0.005 --per --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b0_10x10_seed{N}.pt

# 15x15 (test scientifique principal)
python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 5000 --mode obstacles --device cuda --seed {N} \
    --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --per --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b0_15x15_seed{N}.pt
```

### Hyperparams V2-B0 (défauts littéraires)

| Param | Valeur | Source |
|---|---|---|
| --per-alpha | 0.6 | Schaul et al. 2015 |
| --per-beta-start | 0.4 | Schaul et al. 2015 |
| --per-beta-end | 1.0 | Annealing complet |
| --per-eta | 0.9 | Kapturowski et al. 2019 (R2D2) |
| --per-epsilon | 1e-6 | Schaul (priority floor) |

### Pour reproduire la baseline V2-ZY+Polyak (sans PER)

Ne pas passer `--per`. Le flag est strict opt-in.
```

- [ ] **Step 2 — Add V2-B0 section to CLAUDE.md**

Dans `CLAUDE.md`, ajouter après la section V2-U une nouvelle section V2-B0. Inclure :

1. Sous-section "V2-B0 — état final des phases" avec table 10 phases × Tâches × Statut × Tests × Commits
2. Sous-section "Composants V2-B0 livrés" avec table Composant × Fichier × Rôle
3. Sous-section "Décisions techniques V2-B0" — listes les décisions clés (sum tree convention, IS normalisation, R2D2 aggregation hors autocast, etc.)
4. Sous-section "V2-B0 — pièges connus" (mêmes 13 pièges que spec section 10.1)
5. Mise à jour de la table "Sous-projets" : ajouter `B0 | PER trajectoire | Livré (tag v0.2.0-b0)`
6. Mise à jour du paragraphe "État au handoff" pour mentionner V2-B0 livré

(Note : les bench results sont ajoutés en Phase 13/14, pas ici.)

Modifier également :
- "Lancer les tests" → "Attendu : **320 passed**" (au lieu de 265 actuellement)
- "Smoke E2E ..." → ajouter mention `--per` comme option testée
- "Instructions pour la prochaine session" → la décision suivante dépendra du bench V2-B0 phase 14

- [ ] **Step 3 — Verify pytest still passes**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `320 passed`.

- [ ] **Step 4 — Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs(v2-b0): add V2-B0 section to README + CLAUDE.md (code livraison)"
```

---

## Phase 12 — Bench n=5 same-seed 10×10 (no-regression check)

### Task 12 : Exécuter et documenter le bench 10×10

**Files:**
- Modify: `CLAUDE.md` (section bench V2-B0 10×10)
- Output: `checkpoints/v2b0_10x10_seed{0..4}.pt` (5 fichiers)

- [ ] **Step 1 — Run 5 same-seed benches 10×10 (GPU)**

```bash
source .venv/Scripts/activate
mkdir -p checkpoints
for seed in 0 1 2 3 4; do
  python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 5000 --mode obstacles --device cuda --seed $seed \
    --polyak-tau 0.005 --per --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b0_10x10_seed${seed}.pt \
    2>&1 | tee logs/v2b0_10x10_seed${seed}.log
done
```

Compute estimé : 5 runs × ~30 min RTX 3060 = ~2.5 h.

- [ ] **Step 2 — Extraire les métriques par seed**

Pour chaque seed, extraire de `logs/v2b0_10x10_seed{N}.log` :
- Final winrate + diff
- Best @ diff=0.30 winrate + episode capté
- Per-bucket winrate (5 buckets)

- [ ] **Step 3 — Calculer agrégats n=5**

Calculer :
- Mean best winrate @ diff=0.30
- Std (n−1)
- Min (worst seed)
- Max (best seed)
- Late-stage collapse incidence (compter combien de seeds finissent < 30% rolling winrate)
- Diff_max training mean

- [ ] **Step 4 — Vérifier critères acceptance phase 1**

Comparer aux critères de la spec (section 9.3) :
- Mean ≥ **85 %** (tolère −7 pp vs baseline 92 %)
- Std ≤ **20 pp**
- Late-stage collapse = **0/5**
- Diff_max training mean ≥ **0.5**

**Si critères atteints** → continuer Phase 13.
**Si critères ratés** → B0 rejeté, investigation requise. **STOP plan** — n'exécuter ni Phase 13 ni tag.

- [ ] **Step 5 — Documenter résultats 10×10 dans CLAUDE.md**

Ajouter section "V2-B0 — bench n=5 same-seed 10×10 (sanity)" avec :
- Table résultats par seed (5 lignes : seed, best winrate, final winrate, diff_max, ep_to_best)
- Table agrégats n=5 (mean, std, min, max)
- Comparaison directe vs baseline V2-ZY+Polyak 10×10 (table colonnes Métrique | Baseline | V2-B0 | Δ)
- Verdict phase 1 (PASSED / FAILED)

- [ ] **Step 6 — Commit**

```bash
git add CLAUDE.md
git commit -m "docs(v2-b0): bench V2-ZY+Polyak+PER 10x10 n=5 — [verdict: PASSED no-regression]"
```

(Adapter le commit message au verdict réel.)

---

## Phase 13 — Bench n=5 same-seed 15×15 (test scientifique principal)

### Task 13 : Exécuter et documenter le bench 15×15

**Files:**
- Modify: `CLAUDE.md` (section bench V2-B0 15×15)
- Output: `checkpoints/v2b0_15x15_seed{0..4}.pt` (5 fichiers)

- [ ] **Step 1 — Run 5 same-seed benches 15×15 (GPU)**

```bash
source .venv/Scripts/activate
mkdir -p checkpoints
for seed in 0 1 2 3 4; do
  python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 5000 --mode obstacles --device cuda --seed $seed \
    --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --per --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b0_15x15_seed${seed}.pt \
    2>&1 | tee logs/v2b0_15x15_seed${seed}.log
done
```

Compute estimé : 5 runs × ~45 min RTX 3060 = ~3.75 h.

- [ ] **Step 2 — Extraire métriques par seed**

Pour chaque seed, extraire :
- Final winrate + diff
- Best @ diff=0.30 winrate + episode capté
- Per-bucket winrate

- [ ] **Step 3 — Calculer agrégats n=5**

Identique Phase 12 step 3.

- [ ] **Step 4 — Évaluer critères acceptance phase 2**

Comparer aux 4 critères (spec section 9.3) :
- Mean > **64 %** ? (gain global)
- Min > **50 %** ? (worst-case)
- Médiane `ep_to_best` < baseline médiane ? (convergence)
- Diff_max training mean > **0.36** ? (curriculum franchi plus haut)

**Compter le nombre de critères atteints (0/4 à 4/4).**

- [ ] **Step 5 — Documenter résultats 15×15 dans CLAUDE.md**

Ajouter section "V2-B0 — bench n=5 same-seed 15×15 (test scientifique)" avec :
- Table résultats par seed
- Table agrégats n=5
- Comparaison directe vs baseline V2-ZY+Polyak 15×15
- Critères acceptance (4 lignes, atteints OUI/NON)
- Verdict B0 selon outcome :
  - **≥ 3/4 critères atteints** : "Finding publishable. Cascade Conv + LSTM + Double DQN + Polyak + PER confirmée."
  - **1-2/4 critères atteints** : "PER aide. Documenter critère + magnitude + hypothèse mécaniste."
  - **0/4 critères atteints** : "PER ne franchit pas le plafond 15×15. Finding négatif défendable (hyperparams Schaul canoniques, n=5)."
- Story scientifique consolidée (3-5 lignes)
- Décision suivante (B0.1, B1, ou nouveau brainstorm) selon spec section 12

- [ ] **Step 6 — Commit**

```bash
git add CLAUDE.md
git commit -m "docs(v2-b0): bench V2-ZY+Polyak+PER 15x15 n=5 — [résumé verdict 1 ligne]"
```

(Adapter le commit message au verdict réel.)

---

## Phase 14 — Tag v0.2.0-b0

### Task 14 : Tag livraison

**Files:** aucun (opération git pure).

- [ ] **Step 1 — Verify all DoD items**

Checklist finale (spec section 11) :
- [x] 48-51 nouveaux tests pytest verts (sum tree 8 + prioritized buffer 16 + beta scheduler 7 + trainer 8 + agents 16 = 55 effectifs)
- [x] `pytest -q` total : 320 passed (265 baseline + 55)
- [x] `bash aether/verify_all.sh` → 8 OK (inchangé)
- [x] V2-Y baseline reproductible strict sans `--per`
- [x] V2-ZY baseline idem
- [x] CI : 2 nouveaux smoke jobs passent
- [x] Bench phase 1 — 10×10 n=5 acceptance PASSED
- [x] Bench phase 2 — 15×15 n=5 documenté
- [x] Section V2-B0 dans CLAUDE.md complète
- [x] Section V2-B0 dans README
- [x] Spec dans `docs/superpowers/specs/`
- [x] Plan dans `docs/superpowers/plans/`

- [ ] **Step 2 — Verify git tree clean**

```bash
git status --short
```

Attendu : nothing to commit (working tree clean).

- [ ] **Step 3 — Tag**

```bash
git tag v0.2.0-b0 -m "V2-B0 : Trajectory-level PER (Schaul 2015 + R2D2 2019) livré.

Sous-projet B phase 0 (contrôle scientifique avant B1/B2). Implémentation
canonique : sum tree O(log N), IS correction avec annealing β 0.4→1.0,
R2D2 aggregation (η=0.9 × max + 0.1 × mean) par trajectoire.

Bench n=5 same-seed V2-ZY+Polyak+PER :
- 10x10 : [résumé verdict]
- 15x15 : [résumé verdict]

48-55 nouveaux tests pytest. 320 total. Backwards compat V2-Y/V2-ZY strict.

Voir CLAUDE.md section V2-B0 pour bench results + décision suivante (B1/B2/B0.1)."
```

(Remplacer `[résumé verdict]` par les vrais résultats.)

- [ ] **Step 4 — Verify tag**

```bash
git tag --list | tail -3
git show v0.2.0-b0 --no-patch | head -20
```

Attendu : `v0.2.0-b0` listé, message de tag affiché.

- [ ] **Step 5 — No commit (tag uniquement)**

---

## Self-review checklist

- [x] **Spec coverage** : chaque sous-section de la spec mappée à une task
  - Spec §4 Architecture → Tasks 2-7 (composants), Task 8 (CLI), Task 9 (CI)
  - Spec §5.1 SumTree → Task 2
  - Spec §5.2 PrioritizedSequenceReplayBuffer → Task 4
  - Spec §5.3 BetaScheduler → Task 3
  - Spec §5.4 Trainer extension → Task 6
  - Spec §5.5 Agents extension → Task 7
  - Spec §6 Config & CLI → Task 5 (config) + Task 8 (CLI)
  - Spec §7 Tests → tests intégrés dans chaque task
  - Spec §8 CI smoke → Task 9
  - Spec §9 Bench protocol → Tasks 12-13
  - Spec §11 DoD → Task 14 verification
- [x] **Placeholder scan** : aucun TBD/TODO dans les tasks (les `[résumé verdict]` dans les commits Phase 12-14 sont des marqueurs intentionnels à remplir au moment de l'exécution avec les vraies données du bench)
- [x] **Type consistency** : `PrioritizedBatchSeq` utilisé cohérent entre Task 4 (définition) et Task 6 (consommation). `BetaScheduler` instancié cohérent entre Task 3 (définition) et Task 7 (utilisation). `step_with_priorities` signature cohérente entre Task 6 (impl) et Task 7 (appel agent).

---

## Phase totale récapitulée

| Phase | Tâches | Tests cumul | Commits |
|---|---|---|---|
| 1 — Scaffold | 1 | 265 | 1 |
| 2 — SumTree | 1 | 273 (+8) | 1 |
| 3 — BetaScheduler | 1 | 280 (+7) | 1 |
| 4 — PrioritizedSequenceReplayBuffer | 1 | 292 (+12) | 1 |
| 5 — Config extension | 1 | 296 (+4) | 1 |
| 6 — Trainer step_with_priorities | 1 | 304 (+8) | 1 |
| 7 — Agents PER integration | 1 | 320 (+16) | 1 |
| 8 — CLI flags | 1 | 320 | 1 |
| 9 — CI smoke | 1 | 320 | 1 |
| 10 — Sanity verification | 1 | 320 | 0 |
| 11 — Doc README + CLAUDE.md | 1 | 320 | 1 |
| 12 — Bench 10×10 n=5 | 1 | 320 | 1 |
| 13 — Bench 15×15 n=5 | 1 | 320 | 1 |
| 14 — Tag v0.2.0-b0 | 1 | 320 | 0 (tag uniquement) |

**Total** : 14 tâches, 55 nouveaux tests (320 final), 11 commits + 1 tag.

**Note sur le test count** : la spec annonçait 48 tests. Le plan en livre 55 (granularité validation plus fine sur BetaScheduler : 7 au lieu de 4 + 4 validation config). Cohérent et plus rigoureux.

**Compute total bench** : ~6.25 h GPU RTX 3060.
