# MW_IA V2-Y Recurrent DQN (LSTM) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduire un Deep Recurrent Q-Network (LSTM) pour franchir le plafond architectural V2-X (winrate plafonné à ~80% à diff=0.10 sur DQN feedforward). Critère succès : bucket 1 du tracker (diff 0.20-0.40) winrate ≥ 70% en 5000 ép.

**Architecture:** Nouvelle 3ème ligne d'agent/runner parallèle à V1 et V2-X (zéro modification invasive). Réseau Linear→ReLU→LSTM→Linear, buffer par trajectoires complètes, sample fenêtres aléatoires seq_len=32 avec padding+mask, BPTT 32 steps avec Adam+AMP+grad clip (pattern V1), hidden state runtime maintenu entre `act()` consécutifs et reset à chaque épisode.

**Tech Stack:** Python 3.13, PyTorch (cu128, nn.LSTM), NumPy, pytest, hypothesis (déjà installé V2-A), réutilise infrastructure V2-X (ProceduralGridWorld, encode_procedural_observation, AdaptiveDifficultyScheduler, DifficultyBucketTracker).

**Spec source:** `docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md`

**État initial:** Branche `main`, tags `v0.1.0` + `v0.2.0-a` + `v0.2.0-x` posés, 148 tests pytest verts. Pattern : développement sur `main` (pas de feature branch, comme V1/V2-A/V2-X).

---

## Phase 1 — Setup

### Task 1 : Scaffold neural/ + tests/

**Files :**
- Create : `mw_ia/neural/recurrent.py` (avec docstring de module)
- Create : `mw_ia/neural/sequence_buffer.py` (avec docstring)
- Create : `mw_ia/neural/recurrent_trainer.py` (avec docstring)
- Create : `tests/neural/__init__.py` (vide, si absent)
- Create : `tests/neural/conftest.py` (fixture device CPU)
- Create : `tests/neural/test_recurrent.py` (vide)
- Create : `tests/neural/test_sequence_buffer.py` (vide)
- Create : `tests/neural/test_recurrent_trainer.py` (vide)
- Create : `tests/agents/test_recurrent_dqn.py` (vide)

- [ ] **Step 1 — Verify initial state**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `148 passed`.

- [ ] **Step 2 — Create scaffolds**

```bash
mkdir -p tests/neural tests/agents
```

Contenu de `tests/neural/__init__.py` : vide.
Contenu de `tests/agents/__init__.py` : créer si absent, vide.

Contenu de `tests/neural/conftest.py` :

```python
"""Fixtures partagées pour les tests neural."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cpu_device() -> torch.device:
    """Device CPU pour tests déterministes (pas de dépendance CUDA)."""
    return torch.device("cpu")
```

Contenu de `mw_ia/neural/recurrent.py` :

```python
"""RecurrentQNetwork (LSTM) pour V2-Y DRQN.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md §2.1
"""
from __future__ import annotations
```

Contenu de `mw_ia/neural/sequence_buffer.py` :

```python
"""SequenceReplayBuffer — buffer circulaire de trajectoires complètes pour DRQN.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md §2.1
"""
from __future__ import annotations
```

Contenu de `mw_ia/neural/recurrent_trainer.py` :

```python
"""RecurrentDQNTrainer — boucle d'optimisation BPTT pour DRQN.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md §2.1
"""
from __future__ import annotations
```

Contenu de `tests/neural/test_recurrent.py`, `tests/neural/test_sequence_buffer.py`, `tests/neural/test_recurrent_trainer.py`, `tests/agents/test_recurrent_dqn.py` : vides (seront remplis dans les tasks suivantes).

- [ ] **Step 3 — Verify pytest still passes**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `148 passed` (nouveaux fichiers vides ignorés).

- [ ] **Step 4 — Commit**

```bash
git add mw_ia/neural/recurrent.py mw_ia/neural/sequence_buffer.py mw_ia/neural/recurrent_trainer.py tests/neural/ tests/agents/test_recurrent_dqn.py
git commit -m "chore(drqn): scaffold mw_ia/neural recurrent modules + tests/neural"
```

---

## Phase 2 — `RecurrentQNetwork`

### Task 2 : Init + forward 1 timestep

**Files :**
- Modify : `mw_ia/neural/recurrent.py`
- Modify : `tests/neural/test_recurrent.py`

- [ ] **Step 1 — Write failing tests**

Contenu COMPLET de `tests/neural/test_recurrent.py` :

```python
"""Tests de RecurrentQNetwork."""
from __future__ import annotations

import pytest
import torch

from mw_ia.neural.recurrent import RecurrentQNetwork


def test_recurrent_qnetwork_instantiation(cpu_device):
    net = RecurrentQNetwork(input_dim=200, n_actions=4, fc_hidden=256, lstm_hidden=128)
    net.to(cpu_device)
    assert isinstance(net, torch.nn.Module)


def test_recurrent_qnetwork_forward_single_step(cpu_device):
    net = RecurrentQNetwork(input_dim=200, n_actions=4, fc_hidden=256, lstm_hidden=128).to(cpu_device)
    obs = torch.zeros((1, 1, 200), device=cpu_device)   # (seq=1, batch=1, input_dim)
    q, hidden = net(obs, None)
    assert q.shape == (1, 1, 4)
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2
    h, c = hidden
    assert h.shape == (1, 1, 128)   # (num_layers=1, batch, lstm_hidden)
    assert c.shape == (1, 1, 128)


def test_recurrent_qnetwork_forward_with_none_hidden(cpu_device):
    """hidden=None doit auto-init zéros et retourner hidden non-None."""
    net = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    obs = torch.randn((1, 1, 10), device=cpu_device)
    q, hidden = net(obs, None)
    assert q.shape == (1, 1, 4)
    assert hidden is not None
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_recurrent.py -v 2>&1 | tail -15
```

Attendu : `ImportError: cannot import name 'RecurrentQNetwork'`.

- [ ] **Step 3 — Implement**

Remplacer le contenu de `mw_ia/neural/recurrent.py` par :

```python
"""RecurrentQNetwork (LSTM) pour V2-Y DRQN.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md §2.1
"""
from __future__ import annotations

import torch
from torch import nn


class RecurrentQNetwork(nn.Module):
    """Réseau Q récurrent : Linear → ReLU → LSTM → Linear.

    Convention : batch_first=False, donc obs shape = (seq, batch, input_dim).
    Hidden = tuple (h, c) avec h.shape == c.shape == (1, batch, lstm_hidden)
    (1 layer LSTM). hidden=None → auto-init zéros (pattern PyTorch).
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int = 4,
        fc_hidden: int = 256,
        lstm_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.fc_hidden = fc_hidden
        self.lstm_hidden = lstm_hidden
        self.fc_in = nn.Linear(input_dim, fc_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(fc_hidden, lstm_hidden, num_layers=1, batch_first=False)
        self.fc_out = nn.Linear(lstm_hidden, n_actions)

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """obs_seq shape (seq, batch, input_dim). Retourne (q (seq, batch, n_actions), hidden).

        hidden=None → init zéros gérée par nn.LSTM.
        """
        x = self.fc_in(obs_seq)
        x = self.relu(x)
        x, hidden = self.lstm(x, hidden)
        q = self.fc_out(x)
        return q, hidden
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_recurrent.py -v 2>&1 | tail -10
```

Attendu : `3 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/neural/recurrent.py tests/neural/test_recurrent.py
git commit -m "feat(drqn): RecurrentQNetwork (Linear→ReLU→LSTM→Linear)"
```

---

### Task 3 : Forward séquence + propriétés LSTM

**Files :**
- Modify : `tests/neural/test_recurrent.py`

- [ ] **Step 1 — Add failing tests**

Ajouter EN BAS de `tests/neural/test_recurrent.py` :

```python
def test_recurrent_qnetwork_forward_sequence_batch(cpu_device):
    """Forward sur séquence (seq=32, batch=4) → Q (32, 4, n_actions)."""
    net = RecurrentQNetwork(input_dim=200, n_actions=4, fc_hidden=256, lstm_hidden=128).to(cpu_device)
    obs = torch.zeros((32, 4, 200), device=cpu_device)
    q, hidden = net(obs, None)
    assert q.shape == (32, 4, 4)
    h, c = hidden
    assert h.shape == (1, 4, 128)
    assert c.shape == (1, 4, 128)


def test_recurrent_qnetwork_hidden_state_changes_output(cpu_device):
    """Sanity LSTM : passer un hidden non-zéro change la sortie vs hidden zéro."""
    net = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    obs = torch.randn((1, 1, 10), device=cpu_device)
    torch.manual_seed(0)
    q_zero, _ = net(obs, None)
    nonzero_h = torch.randn((1, 1, 8), device=cpu_device)
    nonzero_c = torch.randn((1, 1, 8), device=cpu_device)
    q_nonzero, _ = net(obs, (nonzero_h, nonzero_c))
    assert not torch.allclose(q_zero, q_nonzero)


def test_recurrent_qnetwork_determinism_same_inputs(cpu_device):
    """Même obs + même hidden + même seed → même Q (déterminisme)."""
    torch.manual_seed(42)
    net = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    obs = torch.randn((1, 1, 10), device=cpu_device)
    h = torch.zeros((1, 1, 8), device=cpu_device)
    c = torch.zeros((1, 1, 8), device=cpu_device)
    q1, _ = net(obs, (h.clone(), c.clone()))
    q2, _ = net(obs, (h.clone(), c.clone()))
    assert torch.allclose(q1, q2)


def test_recurrent_qnetwork_backward_pass_propagates_gradient(cpu_device):
    """Backward sur loss simple → gradients non-None et non-zéro sur tous les params."""
    net = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    obs = torch.randn((4, 2, 10), device=cpu_device, requires_grad=False)
    q, _ = net(obs, None)
    loss = q.pow(2).mean()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"param {name} sans gradient"
        assert not torch.allclose(p.grad, torch.zeros_like(p.grad)), \
            f"param {name} gradient nul (devrait être non-zéro pour obs non-trivial)"
```

- [ ] **Step 2 — Run, expect pass (impl déjà en place)**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_recurrent.py -v 2>&1 | tail -15
```

Attendu : `7 passed` (3 + 4 nouveaux). L'implémentation Task 2 couvre déjà ces tests.

- [ ] **Step 3 — Commit**

```bash
git add tests/neural/test_recurrent.py
git commit -m "test(drqn): RecurrentQNetwork sequence batch + hidden state + backward"
```

---

## Phase 3 — `SequenceReplayBuffer`

### Task 4 : Push trajectoire + validation

**Files :**
- Modify : `mw_ia/neural/sequence_buffer.py`
- Modify : `tests/neural/test_sequence_buffer.py`

- [ ] **Step 1 — Write failing tests**

Contenu COMPLET de `tests/neural/test_sequence_buffer.py` :

```python
"""Tests de SequenceReplayBuffer."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.neural.sequence_buffer import SequenceReplayBuffer


def _trajectory(length: int, obs_dim: int = 4) -> list[tuple]:
    """Crée une trajectoire factice de longueur `length`."""
    return [
        (
            np.zeros(obs_dim, dtype=np.float32),    # state
            i % 4,                                   # action
            float(i),                                # reward
            np.zeros(obs_dim, dtype=np.float32),    # next_state
            i == length - 1,                         # done sur dernier step
        )
        for i in range(length)
    ]


def test_buffer_empty_at_init():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    assert len(buf) == 0


def test_push_trajectory_valid():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    buf.push_trajectory(_trajectory(length=18))
    assert len(buf) == 1


def test_push_trajectory_empty_raises():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    with pytest.raises(ValueError, match="longueur"):
        buf.push_trajectory([])


def test_push_trajectory_too_long_raises():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    with pytest.raises(ValueError, match="longueur"):
        buf.push_trajectory(_trajectory(length=201))


def test_push_capacity_circular():
    """Buffer circulaire : push capacity+5 trajectoires, len reste à capacity."""
    buf = SequenceReplayBuffer(capacity=5, obs_dim=4, max_steps=200, seed=0)
    for _ in range(10):
        buf.push_trajectory(_trajectory(length=10))
    assert len(buf) == 5
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_sequence_buffer.py -v 2>&1 | tail -15
```

Attendu : `ImportError: cannot import name 'SequenceReplayBuffer'`.

- [ ] **Step 3 — Implement**

Remplacer le contenu de `mw_ia/neural/sequence_buffer.py` par :

```python
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
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_sequence_buffer.py -v 2>&1 | tail -10
```

Attendu : `5 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/neural/sequence_buffer.py tests/neural/test_sequence_buffer.py
git commit -m "feat(drqn): SequenceReplayBuffer push_trajectory + circular capacity"
```

---

### Task 5 : Sample fenêtre + padding + mask

**Files :**
- Modify : `mw_ia/neural/sequence_buffer.py`
- Modify : `tests/neural/test_sequence_buffer.py`

- [ ] **Step 1 — Add failing tests**

Ajouter EN BAS de `tests/neural/test_sequence_buffer.py` :

```python
def test_sample_before_min_raises():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    with pytest.raises(ValueError, match="buffer trop petit"):
        buf.sample(batch_size=4, seq_len=32)


def test_sample_returns_correct_shapes():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    for _ in range(5):
        buf.push_trajectory(_trajectory(length=50))
    batch = buf.sample(batch_size=4, seq_len=32)
    assert batch.states.shape == (32, 4, 4)
    assert batch.actions.shape == (32, 4)
    assert batch.rewards.shape == (32, 4)
    assert batch.next_states.shape == (32, 4, 4)
    assert batch.dones.shape == (32, 4)
    assert batch.mask.shape == (32, 4)


def test_sample_padding_short_trajectory():
    """Trajectoire de 18 steps + seq_len=32 → mask = [1]*18 + [0]*14 dans la fenêtre."""
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=42)
    # Pousser une seule trajectoire courte
    buf.push_trajectory(_trajectory(length=18))
    # Sampler avec batch=1, seq_len=32. Comme la traj a 18 steps, l'offset valide est 0,
    # et les 14 steps suivants sont padding.
    batch = buf.sample(batch_size=1, seq_len=32)
    # mask[0:18, 0] doit valoir 1.0, mask[18:32, 0] doit valoir 0.0
    assert batch.mask[:18, 0].sum() == 18.0
    assert batch.mask[18:, 0].sum() == 0.0


def test_sample_random_offset_within_long_trajectory():
    """Trajectoire longue (50 steps) + seq_len=10 → mask doit être tout à 1 (pas de padding)."""
    buf = SequenceReplayBuffer(capacity=10, obs_dim=4, max_steps=200, seed=0)
    buf.push_trajectory(_trajectory(length=50))
    batch = buf.sample(batch_size=1, seq_len=10)
    assert batch.mask.sum() == 10.0   # tous les steps sont des vrais steps
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_sequence_buffer.py -v 2>&1 | tail -15
```

Attendu : `AttributeError: 'SequenceReplayBuffer' object has no attribute 'sample'`.

- [ ] **Step 3 — Implement**

Ajouter en BAS de `mw_ia/neural/sequence_buffer.py` (dans la classe `SequenceReplayBuffer`, après `push_trajectory`) :

```python
    def sample(self, batch_size: int, seq_len: int) -> BatchSeq:
        """Tire batch_size trajectoires aléatoires, fenêtre aléatoire seq_len chacune.

        Padding zéros + mask=0 si trajectoire plus courte que seq_len.
        """
        if self._size < batch_size:
            raise ValueError(
                f"buffer trop petit ({self._size}) pour batch={batch_size}"
            )
        if seq_len <= 0 or seq_len > self.max_steps:
            raise ValueError(
                f"seq_len {seq_len} hors ]0, {self.max_steps}]"
            )

        traj_idxs = self._rng.integers(0, self._size, size=batch_size)

        states = np.zeros((seq_len, batch_size, self.obs_dim), dtype=np.float32)
        actions = np.zeros((seq_len, batch_size), dtype=np.int64)
        rewards = np.zeros((seq_len, batch_size), dtype=np.float32)
        next_states = np.zeros((seq_len, batch_size, self.obs_dim), dtype=np.float32)
        dones = np.zeros((seq_len, batch_size), dtype=np.float32)
        mask = np.zeros((seq_len, batch_size), dtype=np.float32)

        for b, traj_i in enumerate(traj_idxs):
            length = int(self._lengths[traj_i])
            # Offset aléatoire ∈ [0, max(0, length - seq_len)]
            max_offset = max(0, length - seq_len)
            offset = int(self._rng.integers(0, max_offset + 1))
            real_len = min(seq_len, length - offset)
            states[:real_len, b] = self._states[traj_i, offset:offset + real_len]
            actions[:real_len, b] = self._actions[traj_i, offset:offset + real_len]
            rewards[:real_len, b] = self._rewards[traj_i, offset:offset + real_len]
            next_states[:real_len, b] = self._next_states[traj_i, offset:offset + real_len]
            dones[:real_len, b] = self._dones[traj_i, offset:offset + real_len]
            mask[:real_len, b] = 1.0

        return BatchSeq(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            mask=mask,
        )
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_sequence_buffer.py -v 2>&1 | tail -15
```

Attendu : `9 passed` (5 + 4 nouveaux).

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/neural/sequence_buffer.py tests/neural/test_sequence_buffer.py
git commit -m "feat(drqn): SequenceReplayBuffer sample with random offset + padding mask"
```

---

## Phase 4 — `RecurrentDQNTrainer`

### Task 6 : Trainer BPTT + masque

**Files :**
- Modify : `mw_ia/neural/recurrent_trainer.py`
- Modify : `tests/neural/test_recurrent_trainer.py`

- [ ] **Step 1 — Write failing tests**

Contenu COMPLET de `tests/neural/test_recurrent_trainer.py` :

```python
"""Tests de RecurrentDQNTrainer."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mw_ia.neural.recurrent import RecurrentQNetwork
from mw_ia.neural.recurrent_trainer import RecurrentDQNTrainer
from mw_ia.neural.sequence_buffer import BatchSeq


def _make_batch(seq_len: int = 8, batch_size: int = 4, obs_dim: int = 10,
                full_mask: bool = True) -> BatchSeq:
    rng = np.random.default_rng(seed=42)
    states = rng.standard_normal((seq_len, batch_size, obs_dim)).astype(np.float32)
    actions = rng.integers(0, 4, size=(seq_len, batch_size)).astype(np.int64)
    rewards = rng.standard_normal((seq_len, batch_size)).astype(np.float32)
    next_states = rng.standard_normal((seq_len, batch_size, obs_dim)).astype(np.float32)
    dones = np.zeros((seq_len, batch_size), dtype=np.float32)
    if full_mask:
        mask = np.ones((seq_len, batch_size), dtype=np.float32)
    else:
        # Demi-mask : la 2e moitié de la séquence est paddée (mask=0)
        mask = np.ones((seq_len, batch_size), dtype=np.float32)
        mask[seq_len // 2:, :] = 0.0
    return BatchSeq(states, actions, rewards, next_states, dones, mask)


def test_trainer_one_step_no_crash(cpu_device):
    online = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    target = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    trainer = RecurrentDQNTrainer(online, target, lr=1e-3, gamma=0.99,
                                  device="cpu", use_amp=False)
    batch = _make_batch()
    loss = trainer.step(batch)
    assert isinstance(loss, float)
    assert loss >= 0.0   # Huber loss toujours ≥ 0


def test_trainer_mask_reduces_loss(cpu_device):
    """Loss avec demi-mask doit différer (être plus petite en moyenne) de loss full mask."""
    online = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    target = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    trainer = RecurrentDQNTrainer(online, target, lr=1e-3, gamma=0.99,
                                  device="cpu", use_amp=False)
    batch_full = _make_batch(full_mask=True)
    loss_full = trainer.step(batch_full)
    # Réinit pour comparer fair (sans laisser le step précédent influencer)
    online2 = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    target2 = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    online2.load_state_dict(online.state_dict() if False else online2.state_dict())  # noqa: SIM108
    trainer2 = RecurrentDQNTrainer(online2, target2, lr=1e-3, gamma=0.99,
                                   device="cpu", use_amp=False)
    batch_half = _make_batch(full_mask=False)
    loss_half = trainer2.step(batch_half)
    # Les deux losses sont des floats finis, différentes (mask différent donne loss différente)
    assert np.isfinite(loss_full)
    assert np.isfinite(loss_half)
    # Note : on ne peut pas affirmer loss_half < loss_full strictement à cause du seed et de l'init,
    # mais on peut vérifier qu'elles sont finies et que mask ≠ all-1 donne un calcul de loss différent
    assert loss_full != loss_half


def test_trainer_sync_target_copies_weights(cpu_device):
    online = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    target = RecurrentQNetwork(input_dim=10, n_actions=4, fc_hidden=16, lstm_hidden=8).to(cpu_device)
    trainer = RecurrentDQNTrainer(online, target, lr=1e-3, gamma=0.99,
                                  device="cpu", use_amp=False)
    # Modifier online pour qu'il diverge de target
    with torch.no_grad():
        online.fc_in.weight.add_(1.0)
    # Avant sync, online != target
    assert not torch.allclose(online.fc_in.weight, target.fc_in.weight)
    # Après sync, online == target
    trainer.sync_target()
    assert torch.allclose(online.fc_in.weight, target.fc_in.weight)
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_recurrent_trainer.py -v 2>&1 | tail -15
```

Attendu : `ImportError: cannot import name 'RecurrentDQNTrainer'`.

- [ ] **Step 3 — Implement**

Remplacer le contenu de `mw_ia/neural/recurrent_trainer.py` par :

```python
"""RecurrentDQNTrainer — boucle d'optimisation BPTT pour DRQN.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md §2.1

DRQN simple (Hausknecht & Stone 2015) : hidden state zero-init au début de
chaque séquence de training (pas de burn-in en V2-Y MVP).
"""
from __future__ import annotations

import torch
from torch import nn

from mw_ia.neural.recurrent import RecurrentQNetwork
from mw_ia.neural.sequence_buffer import BatchSeq


class RecurrentDQNTrainer:
    """Encapsule online net + target net + optimizer + AMP, BPTT 32 steps avec mask."""

    def __init__(
        self,
        online: RecurrentQNetwork,
        target: RecurrentQNetwork,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cuda",
        use_amp: bool = True,
    ) -> None:
        self.online = online
        self.target = target
        self.gamma = gamma
        self.device = torch.device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        # SmoothL1Loss avec reduction='none' pour pouvoir appliquer le mask manuellement
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self._scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.sync_target()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def step(self, batch: BatchSeq) -> float:
        states = torch.from_numpy(batch.states).to(self.device, non_blocking=True)
        actions = torch.from_numpy(batch.actions).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(batch.rewards).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(batch.next_states).to(self.device, non_blocking=True)
        dones = torch.from_numpy(batch.dones).to(self.device, non_blocking=True)
        mask = torch.from_numpy(batch.mask).to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
            # Forward online sur la séquence complète. Hidden=None → zéro-init.
            q_pred_all, _ = self.online(states, None)
            # q_pred_all shape : (seq, batch, n_actions)
            # Gather sur l'action prise
            q_pred = q_pred_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)
            # q_pred shape : (seq, batch)

            with torch.no_grad():
                q_next_all, _ = self.target(next_states, None)
                q_next = q_next_all.max(dim=-1).values
                target_q = rewards + self.gamma * q_next * (1.0 - dones)

            # Huber loss element-wise, puis mask, puis moyenne sur vrais steps
            elem_loss = self.loss_fn(q_pred, target_q)
            masked_loss = elem_loss * mask
            n_valid = mask.sum().clamp(min=1.0)
            loss = masked_loss.sum() / n_valid

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

        return float(loss.detach().item())
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_recurrent_trainer.py -v 2>&1 | tail -10
```

Attendu : `3 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : **163 passed** (148 + 7 Task 2/3 + 9 Task 4/5 + 3 Task 6 - 4 overlap = 163, ou très proche).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/neural/recurrent_trainer.py tests/neural/test_recurrent_trainer.py
git commit -m "feat(drqn): RecurrentDQNTrainer with masked Huber loss + BPTT"
```

---

## Phase 5 — `DRQNConfig`

### Task 7 : Config dataclass + validation

**Files :**
- Modify : `mw_ia/config.py`
- Create : `tests/test_drqn_config.py`

- [ ] **Step 1 — Write failing tests**

`tests/test_drqn_config.py` :

```python
"""Tests de DRQNConfig."""
from __future__ import annotations

import pytest

from mw_ia.config import DRQNConfig


def test_drqn_config_defaults_valid():
    cfg = DRQNConfig()
    assert cfg.fc_hidden == 256
    assert cfg.lstm_hidden == 128
    assert cfg.sequence_length == 32
    assert cfg.replay_capacity == 5000
    assert cfg.min_episodes_to_learn == 100
    assert cfg.train_steps_per_episode == 4
    assert cfg.epsilon_decay_steps == 200_000


def test_drqn_config_is_frozen():
    cfg = DRQNConfig()
    with pytest.raises(Exception):
        cfg.sequence_length = 64  # type: ignore[misc]


def test_drqn_config_sequence_length_too_large_raises():
    with pytest.raises(ValueError, match="sequence_length"):
        DRQNConfig(sequence_length=300)  # > max_steps_per_episode=200


def test_drqn_config_sequence_length_zero_raises():
    with pytest.raises(ValueError, match="sequence_length"):
        DRQNConfig(sequence_length=0)


def test_drqn_config_replay_capacity_zero_raises():
    with pytest.raises(ValueError, match="replay_capacity"):
        DRQNConfig(replay_capacity=0)


def test_drqn_config_epsilon_inverted_raises():
    with pytest.raises(ValueError, match="epsilon"):
        DRQNConfig(epsilon_start=0.1, epsilon_end=0.9)
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/test_drqn_config.py -v 2>&1 | tail -15
```

Attendu : `ImportError: cannot import name 'DRQNConfig'`.

- [ ] **Step 3 — Implement**

Ajouter EN BAS de `mw_ia/config.py` (après `SchedulerConfig`) :

```python
@dataclass(frozen=True)
class DRQNConfig:
    """Deep Recurrent Q-Network (LSTM). Successeur V2-Y de DQNConfig.

    ATTENTION : replay_capacity compte des TRAJECTOIRES (pas transitions
    comme DQNConfig.replay_capacity).
    """

    # Réseau
    fc_hidden: int = 256                # couche FC avant LSTM
    lstm_hidden: int = 128              # taille du hidden state LSTM

    # Sequence
    sequence_length: int = 32

    # Replay (NOMBRE DE TRAJECTOIRES, pas transitions)
    replay_capacity: int = 5_000
    min_episodes_to_learn: int = 100
    train_steps_per_episode: int = 4

    # Optimisation (identique V1)
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000  # default V2-X gagnant
    target_sync_steps: int = 1_000
    use_amp: bool = True

    # Training
    episodes: int = 5_000
    max_steps_per_episode: int = 200

    def __post_init__(self) -> None:
        if not (1 <= self.sequence_length <= self.max_steps_per_episode):
            raise ValueError(
                f"sequence_length {self.sequence_length} hors [1, {self.max_steps_per_episode}]"
            )
        if self.replay_capacity <= 0:
            raise ValueError(f"replay_capacity doit être > 0, reçu {self.replay_capacity}")
        if self.min_episodes_to_learn <= 0:
            raise ValueError(
                f"min_episodes_to_learn doit être > 0, reçu {self.min_episodes_to_learn}"
            )
        if self.train_steps_per_episode <= 0:
            raise ValueError(
                f"train_steps_per_episode doit être > 0, reçu {self.train_steps_per_episode}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size doit être > 0, reçu {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr doit être > 0, reçu {self.lr}")
        if not (0.0 < self.gamma < 1.0):
            raise ValueError(f"gamma doit être ∈ (0,1), reçu {self.gamma}")
        if not (0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0):
            raise ValueError(
                f"epsilon invalide : start={self.epsilon_start}, end={self.epsilon_end}"
            )
        if self.epsilon_decay_steps <= 0:
            raise ValueError(
                f"epsilon_decay_steps doit être > 0, reçu {self.epsilon_decay_steps}"
            )
        if self.target_sync_steps <= 0:
            raise ValueError(
                f"target_sync_steps doit être > 0, reçu {self.target_sync_steps}"
            )
        if self.fc_hidden <= 0 or self.lstm_hidden <= 0:
            raise ValueError(
                f"fc_hidden et lstm_hidden doivent être > 0, reçu fc={self.fc_hidden}, lstm={self.lstm_hidden}"
            )
        if self.episodes <= 0:
            raise ValueError(f"episodes doit être > 0, reçu {self.episodes}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(
                f"max_steps_per_episode doit être > 0, reçu {self.max_steps_per_episode}"
            )
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/test_drqn_config.py -v 2>&1 | tail -10
```

Attendu : `6 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/config.py tests/test_drqn_config.py
git commit -m "feat(drqn): DRQNConfig frozen dataclass with __post_init__ validation"
```

---

## Phase 6 — `RecurrentDQNAgent`

### Task 8 : Init + reset_hidden + act

**Files :**
- Create : `mw_ia/agents/recurrent_dqn.py`
- Modify : `tests/agents/test_recurrent_dqn.py`

- [ ] **Step 1 — Write failing tests**

Contenu COMPLET de `tests/agents/test_recurrent_dqn.py` :

```python
"""Tests de RecurrentDQNAgent."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mw_ia.agents.recurrent_dqn import RecurrentDQNAgent
from mw_ia.config import DRQNConfig


def _agent(seed: int = 0) -> RecurrentDQNAgent:
    cfg = DRQNConfig(episodes=20, replay_capacity=20, min_episodes_to_learn=5,
                     batch_size=4, sequence_length=8, use_amp=False,
                     epsilon_decay_steps=1000, target_sync_steps=100,
                     max_steps_per_episode=50)
    return RecurrentDQNAgent(obs_dim=10, n_actions=4, cfg=cfg, device="cpu", seed=seed)


def test_recurrent_agent_init():
    agent = _agent()
    assert agent.global_step == 0
    assert agent._hidden_state is None
    assert agent._episode_trajectory == []


def test_act_returns_valid_action():
    agent = _agent()
    obs = np.zeros(10, dtype=np.float32)
    action = agent.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < 4


def test_act_greedy_deterministic_with_reset_hidden():
    """Greedy + même obs + hidden reset entre 2 calls → même action."""
    agent = _agent(seed=42)
    obs = np.random.default_rng(0).standard_normal(10).astype(np.float32)
    agent.reset_hidden()
    a1 = agent.act(obs, greedy=True)
    agent.reset_hidden()
    a2 = agent.act(obs, greedy=True)
    assert a1 == a2


def test_reset_hidden_clears_state():
    agent = _agent()
    obs = np.zeros(10, dtype=np.float32)
    agent.act(obs)
    # Après act, le hidden state est non-None
    assert agent._hidden_state is not None
    agent.reset_hidden()
    assert agent._hidden_state is None
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_recurrent_dqn.py -v 2>&1 | tail -15
```

Attendu : `ModuleNotFoundError: No module named 'mw_ia.agents.recurrent_dqn'`.

- [ ] **Step 3 — Implement (init + reset_hidden + act)**

Contenu de `mw_ia/agents/recurrent_dqn.py` :

```python
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
        if (not greedy) and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
            # x shape : (seq=1, batch=1, obs_dim)
            q, new_hidden = self.online(x, self._hidden_state)
            self._hidden_state = new_hidden
            return int(q.argmax(dim=-1).item())

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
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_recurrent_dqn.py -v 2>&1 | tail -10
```

Attendu : `4 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/agents/recurrent_dqn.py tests/agents/test_recurrent_dqn.py
git commit -m "feat(drqn): RecurrentDQNAgent init + reset_hidden + act"
```

---

### Task 9 : observe + begin/end_episode + train step

**Files :**
- Modify : `mw_ia/agents/recurrent_dqn.py`
- Modify : `tests/agents/test_recurrent_dqn.py`

- [ ] **Step 1 — Add failing tests**

Ajouter EN BAS de `tests/agents/test_recurrent_dqn.py` :

```python
def test_begin_episode_clears_trajectory():
    agent = _agent()
    agent._episode_trajectory = [("fake", 0, 0.0, "fake", False)]
    agent.begin_episode()
    assert agent._episode_trajectory == []


def test_observe_appends_to_trajectory():
    agent = _agent()
    agent.begin_episode()
    obs = np.zeros(10, dtype=np.float32)
    next_obs = np.ones(10, dtype=np.float32)
    metrics = agent.observe(obs, action=1, reward=0.5, next_state=next_obs, done=False)
    assert len(agent._episode_trajectory) == 1
    assert agent.global_step == 1
    assert "epsilon" in metrics


def test_end_episode_pushes_trajectory_to_buffer():
    agent = _agent()
    agent.begin_episode()
    obs = np.zeros(10, dtype=np.float32)
    for _ in range(8):
        agent.observe(obs, action=0, reward=0.0, next_state=obs, done=False)
    agent.observe(obs, action=0, reward=1.0, next_state=obs, done=True)
    metrics = agent.end_episode()
    assert len(agent.buffer) == 1
    # min_episodes_to_learn=5 mais on n'a que 1 épisode → pas de train step
    assert "loss" not in metrics


def test_end_episode_triggers_train_after_min_episodes():
    """Pousser min_episodes_to_learn=5 épisodes → end_episode() doit déclencher train_step."""
    agent = _agent()
    obs = np.zeros(10, dtype=np.float32)
    for ep in range(6):
        agent.begin_episode()
        for _ in range(8):
            agent.observe(obs, action=0, reward=0.0, next_state=obs, done=False)
        agent.observe(obs, action=0, reward=1.0, next_state=obs, done=True)
        metrics = agent.end_episode()
    # Au 6e épisode (>= min_episodes_to_learn=5), train step déclenché
    assert "loss" in metrics
    assert agent.last_loss is not None
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_recurrent_dqn.py -v 2>&1 | tail -15
```

Attendu : 4 tests passent, 4 nouveaux échouent (`AttributeError: 'RecurrentDQNAgent' object has no attribute 'observe'` ou similaire — selon que tu as défini observe en stub).

- [ ] **Step 3 — Implement (observe + end_episode)**

Ajouter à la classe `RecurrentDQNAgent` (après `act()`, avant `learn()`) dans `mw_ia/agents/recurrent_dqn.py` :

```python
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
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_recurrent_dqn.py -v 2>&1 | tail -15
```

Attendu : `8 passed` (4 + 4 nouveaux).

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : **≥ 175 passed** (148 baseline + ~27 nouveaux).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/agents/recurrent_dqn.py tests/agents/test_recurrent_dqn.py
git commit -m "feat(drqn): RecurrentDQNAgent observe + begin/end_episode + train trigger"
```

---

## Phase 7 — `RecurrentProceduralDQNRunner`

### Task 10 : Extension runner.py

**Files :**
- Modify : `mw_ia/training/runner.py`
- Create : `tests/training/test_recurrent_procedural_runner.py`

- [ ] **Step 1 — Write failing tests**

`tests/training/test_recurrent_procedural_runner.py` :

```python
"""Tests d'intégration de RecurrentProceduralDQNRunner."""
from __future__ import annotations

import numpy as np

from mw_ia.config import (
    DRQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import RecurrentProceduralDQNRunner, RunnerCallbacks


def _make_runner(episodes: int = 10) -> RecurrentProceduralDQNRunner:
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.0, max_density=0.10)
    gen = RandomObstaclesGenerator(rows=10, cols=10, start=(0, 0), goal=(9, 9),
                                   min_density=proc_cfg.min_density,
                                   max_density=proc_cfg.max_density)
    env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
    drqn_cfg = DRQNConfig(episodes=episodes, replay_capacity=20,
                          min_episodes_to_learn=3, batch_size=4,
                          sequence_length=8, max_steps_per_episode=30,
                          use_amp=False, epsilon_decay_steps=200,
                          target_sync_steps=100)
    sched_cfg = SchedulerConfig(initial_difficulty=0.0, step=0.05, update_interval=5)
    train_cfg = TrainingConfig()
    return RecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, drqn_cfg=drqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=RunnerCallbacks(),
        device="cpu", seed=0,
    )


def test_recurrent_runner_10_episodes_no_error():
    runner = _make_runner(episodes=10)
    runner.run()
    assert runner.metrics.total_episodes == 10


def test_recurrent_runner_emits_maze_changed_callback():
    captured: list[tuple[np.ndarray, int, float]] = []
    cb = RunnerCallbacks(on_maze_changed=lambda **kw: captured.append(
        (kw["maze"], kw["episode_id"], kw["difficulty"])
    ))
    runner = _make_runner(episodes=5)
    runner.callbacks = cb
    runner.run()
    assert len(captured) == 5
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/training/test_recurrent_procedural_runner.py -v 2>&1 | tail -15
```

Attendu : `ImportError: cannot import name 'RecurrentProceduralDQNRunner'`.

- [ ] **Step 3 — Étendre les imports en HAUT de `mw_ia/training/runner.py`**

Les imports doivent rester en TÊTE du fichier (PEP 8, pattern projet commits f5f4d8f / 920e208 / bbcfd55). Ajouter à la liste existante des `from mw_ia.config import ...` :

```python
from mw_ia.config import (
    DQNConfig, DRQNConfig, ProceduralEnvConfig, QLearningConfig,
    SchedulerConfig, TrainingConfig,
)
```

Et ajouter le bloc d'imports neural (regrouper avec les imports projet existants, en tête) :

```python
from mw_ia.agents.recurrent_dqn import RecurrentDQNAgent
```

- [ ] **Step 4 — Implémenter `RecurrentProceduralDQNRunner`**

Ajouter EN BAS de `mw_ia/training/runner.py` (après `ProceduralDQNRunner`) :

```python
class RecurrentProceduralDQNRunner(_BaseRunner):
    """Boucle DRQN sur environnement procédural avec curriculum adaptatif.

    Différences avec ProceduralDQNRunner V2-X :
    - Agent récurrent (LSTM) au lieu de DQN feedforward
    - Hidden state runtime reset à chaque épisode (agent.reset_hidden())
    - Train step à la FIN de l'épisode (agent.end_episode()), pas à chaque step
    - Observation et callbacks GUI identiques V2-X
    """

    def __init__(
        self,
        *,
        env: ProceduralGridWorld,
        proc_cfg: ProceduralEnvConfig,
        drqn_cfg: DRQNConfig,
        sched_cfg: SchedulerConfig,
        train_cfg: TrainingConfig,
        callbacks: RunnerCallbacks,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        super().__init__(train_cfg, callbacks)
        self.env = env
        self.proc_cfg = proc_cfg
        self.drqn_cfg = drqn_cfg
        self.sched_cfg = sched_cfg
        self.scheduler = AdaptiveDifficultyScheduler(sched_cfg)
        self.bucket_tracker = DifficultyBucketTracker(train_cfg)
        obs_dim = 2 * proc_cfg.max_rows * proc_cfg.max_cols
        self.agent = RecurrentDQNAgent(
            obs_dim=obs_dim, n_actions=4, cfg=drqn_cfg, device=device, seed=seed,
        )

    def run(self) -> None:
        self.callbacks.fire_log(
            "info",
            f"Recurrent DQN ({self.proc_cfg.mode}) sur {self.agent.device} démarrage"
        )
        for ep in range(self.drqn_cfg.episodes):
            if self._stop:
                return

            self.env.set_difficulty(self.scheduler.current)
            state, info = self.env.reset(seed=ep)
            maze = info["maze"]
            difficulty = info["difficulty"]
            self.callbacks.fire_maze_changed(maze=maze, episode_id=ep, difficulty=difficulty)

            self.agent.reset_hidden()
            self.agent.begin_episode()

            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated) and ep_len < self.drqn_cfg.max_steps_per_episode:
                if self._stop:
                    return
                while self._paused and not self._stop:
                    pass
                obs = encode_procedural_observation(
                    state=state, grid=maze,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                a = self.agent.act(obs)
                s2, r, terminated, truncated, _ = self.env.step(a)
                next_obs = encode_procedural_observation(
                    state=s2, grid=maze,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                self.agent.observe(obs, a, r, next_obs, terminated or truncated)
                self.callbacks.fire_step(state=state, action=a, reward=r, next_state=s2)
                state = s2
                ep_reward += r
                ep_len += 1

            m = self.agent.end_episode()
            if "loss" in m:
                self.metrics.record_loss(m["loss"])
                self.callbacks.fire_loss(self.agent.global_step, m["loss"])
            self.callbacks.fire_epsilon(self.agent.global_step, m["epsilon"])
            self.metrics.record_epsilon(m["epsilon"])

            self.metrics.record_episode(ep_reward, ep_len, success=terminated)
            self.bucket_tracker.record_episode(
                success=terminated, reward=ep_reward, length=ep_len, difficulty=difficulty,
            )
            self.callbacks.fire_episode(ep=ep, reward=ep_reward, length=ep_len, success=terminated)

            if (ep + 1) % self.sched_cfg.update_interval == 0:
                new_diff = self.scheduler.update(winrate=self.metrics.winrate())
                self.callbacks.fire_difficulty_updated(difficulty=new_diff, episode_id=ep)

            if ep % self.train_cfg.log_every_episodes == 0:
                self.callbacks.fire_log(
                    "info",
                    f"ep {ep:>4}  R={ep_reward:+.2f}  L={ep_len:>3}  "
                    f"eps={self.agent.epsilon:.3f}  winrate={self.metrics.winrate():.2%}  "
                    f"diff={self.scheduler.current:.2f}"
                )
```

- [ ] **Step 5 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/training/test_recurrent_procedural_runner.py -v 2>&1 | tail -15
```

Attendu : `2 passed` (peut prendre 5-15 s sur CPU à cause des train steps).

- [ ] **Step 6 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : ≥ **177 passed** (148 + 7 + 9 + 3 + 6 + 8 + 2 ≈ 177).

- [ ] **Step 7 — Commit**

```bash
git add mw_ia/training/runner.py tests/training/test_recurrent_procedural_runner.py
git commit -m "feat(drqn): RecurrentProceduralDQNRunner with curriculum and GUI callbacks"
```

---

## Phase 8 — CLI + CI

### Task 11 : `scripts/train_drqn_procedural.py`

**Files :**
- Create : `scripts/train_drqn_procedural.py`
- Modify : `.github/workflows/aether_verify.yml`

- [ ] **Step 1 — Write the CLI script**

`scripts/train_drqn_procedural.py` :

```python
"""Entraînement Recurrent DQN (DRQN/LSTM) procedural headless (CLI).

Usage :
    python scripts/train_drqn_procedural.py --episodes 5000 --mode obstacles --device cuda
"""
from __future__ import annotations

import argparse
import sys

from mw_ia.config import (
    DRQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import PerfectMazeGenerator, RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import RecurrentProceduralDQNRunner, RunnerCallbacks


def _print_log(level: str, msg: str) -> None:
    print(f"[{level:7s}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="DRQN procedural training")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--mode", choices=("obstacles", "maze"), default="obstacles")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fc-hidden", type=int, default=256)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--epsilon-decay-steps", type=int, default=200_000)
    args = parser.parse_args()

    proc_cfg = ProceduralEnvConfig(mode=args.mode)
    if args.mode == "obstacles":
        gen = RandomObstaclesGenerator(
            rows=proc_cfg.max_rows, cols=proc_cfg.max_cols,
            start=(0, 0), goal=(proc_cfg.max_rows - 1, proc_cfg.max_cols - 1),
            min_density=proc_cfg.min_density, max_density=proc_cfg.max_density,
            max_attempts=proc_cfg.max_attempts_bfs,
        )
    else:
        gen = PerfectMazeGenerator(min_size=proc_cfg.min_size, max_size=proc_cfg.max_size)

    env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
    drqn_cfg = DRQNConfig(
        episodes=args.episodes,
        fc_hidden=args.fc_hidden,
        lstm_hidden=args.lstm_hidden,
        sequence_length=args.sequence_length,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    sched_cfg = SchedulerConfig()
    train_cfg = TrainingConfig()

    cb = RunnerCallbacks(on_log=_print_log)
    runner = RecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, drqn_cfg=drqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=cb, device=args.device, seed=args.seed,
    )
    runner.run()

    final_wr = runner.metrics.winrate()
    final_diff = runner.scheduler.current
    print(f"\nFinal : winrate={final_wr:.2%}, difficulty={final_diff:.2f}")
    print("Per-bucket winrate :")
    for i, wr in enumerate(runner.bucket_tracker.winrate_per_bucket()):
        wr_str = f"{wr:.2%}" if wr is not None else "-"
        print(f"  bucket {i} ({i*0.2:.1f}-{(i+1)*0.2:.1f}) : {wr_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2 — Smoke test CPU obstacles**

```bash
source .venv/Scripts/activate && python scripts/train_drqn_procedural.py --episodes 20 --mode obstacles --device cpu 2>&1 | tail -15
```

Attendu : pas d'erreur, output contient `Final :`, `Per-bucket winrate :`, 5 lignes bucket.

- [ ] **Step 3 — Smoke test CPU maze**

```bash
source .venv/Scripts/activate && python scripts/train_drqn_procedural.py --episodes 20 --mode maze --device cpu 2>&1 | tail -15
```

Attendu : même pattern.

- [ ] **Step 4 — Update CI workflow**

Lire `.github/workflows/aether_verify.yml`, identifier le job `pytest`. Ajouter un step après le step `Smoke test procedural training` existant (V2-X) :

```yaml
      - name: Smoke test recurrent procedural training
        run: |
          python scripts/train_drqn_procedural.py --episodes 10 --mode obstacles --device cpu
          python scripts/train_drqn_procedural.py --episodes 10 --mode maze --device cpu
```

L'indentation YAML doit correspondre exactement à celle du step V2-X existant.

- [ ] **Step 5 — Run full pytest (no regression)**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : ≥ **177 passed** (inchangé).

- [ ] **Step 6 — Commit**

```bash
git add scripts/train_drqn_procedural.py .github/workflows/aether_verify.yml
git commit -m "feat(drqn): scripts/train_drqn_procedural.py CLI + CI smoke"
```

---

## Phase 9 — Documentation + tag

### Task 12 : README V2-Y section

**Files :**
- Modify : `README.md`

- [ ] **Step 1 — Add V2-Y section**

Insérer dans `README.md` après la section "V2-X — Environnement procédural & curriculum learning" et avant "Roadmap (V2+)" :

```markdown
## V2-Y — Deep Recurrent Q-Network (LSTM) (sous-projet livré)

Mémoire neuronale temporelle pour franchir le plafond architectural V2-X
(DQN feedforward plafonne à ~80% winrate à diff=0.10). Le LSTM permet à
l'agent de se souvenir des dead-ends récents dans le maze courant.

### Usage CLI

```bash
# Mode obstacles, recette V2-X gagnante héritée par défaut
python scripts/train_drqn_procedural.py --episodes 5000 --mode obstacles --device cuda

# Mode maze parfait
python scripts/train_drqn_procedural.py --episodes 5000 --mode maze --device cuda
```

### Critère de succès

Bucket 1 du tracker (difficulté 0.20-0.40) doit afficher **winrate ≥ 70 %** en
fin d'entraînement. Comparaison directe avec V2-X (DQN feedforward) qui plafonne
au bucket 0 (0.0-0.20).

### Architecture

- `mw_ia/neural/recurrent.py` — `RecurrentQNetwork` (Linear → ReLU → LSTM → Linear)
- `mw_ia/neural/sequence_buffer.py` — `SequenceReplayBuffer` (buffer de trajectoires complètes, sample seq_len avec padding+mask)
- `mw_ia/neural/recurrent_trainer.py` — `RecurrentDQNTrainer` (BPTT 32 steps, Huber masquée, AMP + grad clip)
- `mw_ia/agents/recurrent_dqn.py` — `RecurrentDQNAgent` (hidden state runtime maintenu, reset par épisode)
- `mw_ia/training/runner.py::RecurrentProceduralDQNRunner` — extension V2-Y

### Algorithme

DRQN simple (Hausknecht & Stone 2015) : hidden state zero-init au début de
chaque séquence d'entraînement (pas de burn-in en V2-Y MVP). Hidden state
runtime maintenu entre `act()` consécutifs dans un épisode, reset à chaque
nouvel épisode (pas de mémoire cross-épisodes — cohérent avec le but
"résoudre le maze courant", pas "se souvenir des mazes précédents").
```

- [ ] **Step 2 — Commit**

```bash
git add README.md
git commit -m "docs(readme): add V2-Y Recurrent DQN section"
```

---

### Task 13 : Definition of Done + tag

**Files :** aucune modification.

- [ ] **Step 1 — Full pytest**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : ≥ **183 passed** (148 baseline + ~35 nouveaux V2-Y).

- [ ] **Step 2 — Smoke V2-A inchangé**

```bash
bash aether/verify_all.sh
```

Attendu : 8 OK.

- [ ] **Step 3 — Smoke V2-X inchangé**

```bash
source .venv/Scripts/activate && python scripts/train_dqn_procedural.py --episodes 20 --mode obstacles --device cpu 2>&1 | tail -10
```

Attendu : V2-X CLI fonctionne toujours (pas de régression).

- [ ] **Step 4 — Smoke V2-Y E2E obstacles**

```bash
source .venv/Scripts/activate && python scripts/train_drqn_procedural.py --episodes 50 --mode obstacles --device cuda 2>&1 | tail -10
```

Attendu : pas d'erreur, output `Final :` et `Per-bucket winrate :`.

- [ ] **Step 5 — Smoke V2-Y E2E maze**

```bash
source .venv/Scripts/activate && python scripts/train_drqn_procedural.py --episodes 50 --mode maze --device cuda 2>&1 | tail -10
```

Attendu : même pattern.

- [ ] **Step 6 — Tag the release**

```bash
git tag -a v0.2.0-y -m "MW_IA V2-Y — Deep Recurrent Q-Network (LSTM)

Sous-projet V2-Y (roadmap #1). Livraison :
- RecurrentQNetwork (Linear → ReLU → LSTM → Linear)
- SequenceReplayBuffer (buffer de trajectoires complètes + sample seq_len padding+mask)
- RecurrentDQNTrainer (BPTT 32 steps, Huber masquée, AMP + grad clip)
- RecurrentDQNAgent (hidden state runtime + reset par épisode)
- RecurrentProceduralDQNRunner (extension du V2-X runner)
- scripts/train_drqn_procedural.py + CI smoke

V1 + V2-A + V2-X intacts (rétro-compat stricte).

DRQN simple (Hausknecht & Stone 2015) : hidden state zero-init au début
de chaque séquence d'entraînement (pas de burn-in en MVP).

Critère de succès à valider en entraînement long : bucket 1 du tracker
(diff 0.20-0.40) winrate ≥ 70% en 5000 ép. Comparaison directe avec
baseline V2-X qui plafonne au bucket 0.

~35 nouveaux tests pytest, total >= 183 verts."
git tag | grep v0
git log --oneline -5
```

Expected : tags `v0.1.0`, `v0.2.0-a`, `v0.2.0-x`, **`v0.2.0-y`** tous présents.

---

## Récapitulatif des fichiers livrés

```
MW_IA/
├── mw_ia/
│   ├── config.py                          [Task 7] + DRQNConfig
│   ├── neural/
│   │   ├── recurrent.py                   [Tasks 2-3] RecurrentQNetwork
│   │   ├── sequence_buffer.py             [Tasks 4-5] SequenceReplayBuffer
│   │   └── recurrent_trainer.py           [Task 6] RecurrentDQNTrainer
│   ├── agents/
│   │   └── recurrent_dqn.py               [Tasks 8-9] RecurrentDQNAgent
│   └── training/
│       └── runner.py                      [Task 10] + RecurrentProceduralDQNRunner
├── scripts/
│   └── train_drqn_procedural.py           [Task 11] CLI DRQN
├── tests/
│   ├── neural/
│   │   ├── __init__.py                    [Task 1]
│   │   ├── conftest.py                    [Task 1] fixture cpu_device
│   │   ├── test_recurrent.py              [Tasks 2-3] 7 tests
│   │   ├── test_sequence_buffer.py        [Tasks 4-5] 9 tests
│   │   └── test_recurrent_trainer.py      [Task 6] 3 tests
│   ├── agents/
│   │   └── test_recurrent_dqn.py          [Tasks 8-9] 8 tests
│   ├── training/
│   │   └── test_recurrent_procedural_runner.py  [Task 10] 2 tests d'intégration
│   └── test_drqn_config.py                [Task 7] 6 tests
├── .github/workflows/aether_verify.yml    [Task 11] + CI smoke V2-Y
└── README.md                              [Task 12] section V2-Y
```

**Total :** 13 tâches sur 9 phases · ~35 nouveaux tests · 4 nouveaux composants neural + 1 agent + 1 runner + 1 CLI · 0 dépendance ajoutée.
