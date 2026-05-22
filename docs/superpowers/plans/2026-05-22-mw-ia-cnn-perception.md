# MW_IA V2-Z CNN perception spatiale — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduire un Convolutional DQN feedforward pour franchir le plafond architectural V2-X/V2-Y (winrate plafonné à `diff ≈ 0.05`). Critère succès : match V2-Y @ diff=0.05 (≥ 95% winrate) + scheduler maintient `diff ≥ 0.10` avec winrate bucket ≥ 70%.

**Architecture:** Nouvelle 4ème ligne agent/runner parallèle à V1, V2-X et V2-Y (zéro modification invasive). Encoder 2D `(3, R, C)` (canaux agent + obstacles + goal), réseau `Conv(3→32) → ReLU → Conv(32→64) → ReLU → Flatten → FC(256) → FC(4)` sans pooling sur 10×10 (~1.66M params). Réutilise `ReplayBuffer` V1 via flatten/reshape autour de `push`/`sample`. Defaults scheduler V2-X (`update=200`, `step=0.05`).

**Tech Stack:** Python 3.13, PyTorch (cu128, nn.Conv2d), NumPy, pytest. Réutilise infrastructure V2-X (`ProceduralGridWorld`, `AdaptiveDifficultyScheduler`, `DifficultyBucketTracker`, `MetricsTracker`) + buffer V1 (`ReplayBuffer`, `Batch`) + trainer V1 (`DQNTrainer` pattern AMP/Huber/grad clip).

**Spec source:** `docs/superpowers/specs/2026-05-22-mw-ia-cnn-perception-design.md`

**État initial:** Branche `main`, tags `v0.1.0` + `v0.2.0-a` + `v0.2.0-x` + `v0.2.0-y` posés, 183 tests pytest verts. Pattern : développement sur `main` (pas de feature branch).

---

## Phase 1 — Setup

### Task 1 : Scaffold neural/, agents/, tests/

**Files :**
- Create : `mw_ia/neural/conv_network.py` (docstring seulement)
- Create : `mw_ia/agents/conv_dqn.py` (docstring seulement)
- Create : `tests/neural/test_conv_network.py` (vide)
- Create : `tests/envs/test_procedural_env_2d.py` (vide)
- Create : `tests/agents/test_conv_dqn.py` (vide)
- Create : `tests/training/test_conv_procedural_runner.py` (vide)
- Create : `tests/test_conv_dqn_config.py` (vide)

- [ ] **Step 1 — Verify initial state**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `183 passed`.

- [ ] **Step 2 — Create scaffold files**

Contenu de `mw_ia/neural/conv_network.py` :

```python
"""ConvQNetwork (Conv2d) pour V2-Z DQN à perception spatiale.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-cnn-perception-design.md §2
"""
from __future__ import annotations
```

Contenu de `mw_ia/agents/conv_dqn.py` :

```python
"""ConvDQNAgent — DQN feedforward à perception spatiale (V2-Z).

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-cnn-perception-design.md §2
"""
from __future__ import annotations
```

Les 5 fichiers de tests (`test_conv_network.py`, `test_procedural_env_2d.py`, `test_conv_dqn.py`, `test_conv_procedural_runner.py`, `test_conv_dqn_config.py`) restent vides à ce stade.

- [ ] **Step 3 — Verify pytest still passes**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `183 passed` (les nouveaux fichiers vides ne cassent rien).

- [ ] **Step 4 — Commit scaffold**

```bash
git add mw_ia/neural/conv_network.py mw_ia/agents/conv_dqn.py \
        tests/neural/test_conv_network.py tests/envs/test_procedural_env_2d.py \
        tests/agents/test_conv_dqn.py tests/training/test_conv_procedural_runner.py \
        tests/test_conv_dqn_config.py
git commit -m "chore(v2-z): scaffold conv_network/conv_dqn modules + empty test files"
```

---

## Phase 2 — `encode_procedural_observation_2d`

### Task 2 : Encoder 2D (3 canaux : agent + obstacles + goal)

**Files :**
- Modify : `mw_ia/envs/procedural_env.py` (ajout fonction, V2-X 1D conservé)
- Test : `tests/envs/test_procedural_env_2d.py`

- [ ] **Step 1 — Write the 6 failing tests**

Contenu de `tests/envs/test_procedural_env_2d.py` :

```python
"""Tests V2-Z de encode_procedural_observation_2d (3 canaux pour CNN)."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.envs.procedural_env import encode_procedural_observation_2d


def test_encode_shape_default():
    """Encoding standard 10x10 → shape (3, 10, 10) float32."""
    grid = np.zeros((10, 10), dtype=bool)
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=grid, goal=(9, 9), max_rows=10, max_cols=10,
    )
    assert obs.shape == (3, 10, 10)
    assert obs.dtype == np.float32


def test_encode_agent_channel():
    """Canal 0 : un seul 1 en (row, col), zéros partout ailleurs."""
    grid = np.zeros((10, 10), dtype=bool)
    obs = encode_procedural_observation_2d(
        state=(3, 5), grid=grid, goal=(9, 9), max_rows=10, max_cols=10,
    )
    assert obs[0, 3, 5] == 1.0
    assert obs[0].sum() == 1.0


def test_encode_obstacles_channel():
    """Canal 1 : matches grid.astype(float32)."""
    grid = np.zeros((10, 10), dtype=bool)
    grid[2, 2] = True
    grid[5, 7] = True
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=grid, goal=(9, 9), max_rows=10, max_cols=10,
    )
    assert obs[1, 2, 2] == 1.0
    assert obs[1, 5, 7] == 1.0
    assert obs[1].sum() == 2.0


def test_encode_goal_channel():
    """Canal 2 : un seul 1 en (goal_r, goal_c)."""
    grid = np.zeros((10, 10), dtype=bool)
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=grid, goal=(7, 4), max_rows=10, max_cols=10,
    )
    assert obs[2, 7, 4] == 1.0
    assert obs[2].sum() == 1.0


def test_encode_padding_smaller_maze():
    """Maze 6x6 dans max 10x10 → padding zéros top-left sur les 3 canaux."""
    grid = np.ones((6, 6), dtype=bool)
    grid[0, 0] = False  # start libre
    grid[5, 5] = False  # goal libre
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=grid, goal=(5, 5), max_rows=10, max_cols=10,
    )
    assert obs.shape == (3, 10, 10)
    # Zone hors maze : padding zéros sur tous les canaux
    assert obs[1, 6:, :].sum() == 0.0  # pas d'obstacles dans le padding
    assert obs[1, :, 6:].sum() == 0.0
    assert obs[2, 6:, :].sum() == 0.0  # pas de goal dans le padding
    # Goal correctement placé
    assert obs[2, 5, 5] == 1.0


def test_encode_asserts_invalid_inputs():
    """state hors grille, grid > max, goal hors max → AssertionError."""
    grid = np.zeros((10, 10), dtype=bool)
    # grid trop grand
    too_big = np.zeros((11, 11), dtype=bool)
    with pytest.raises(AssertionError):
        encode_procedural_observation_2d(
            state=(0, 0), grid=too_big, goal=(9, 9), max_rows=10, max_cols=10,
        )
    # goal hors max
    with pytest.raises(AssertionError):
        encode_procedural_observation_2d(
            state=(0, 0), grid=grid, goal=(10, 5), max_rows=10, max_cols=10,
        )
    # state hors grille réelle
    with pytest.raises(AssertionError):
        encode_procedural_observation_2d(
            state=(10, 0), grid=grid, goal=(9, 9), max_rows=10, max_cols=10,
        )
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_procedural_env_2d.py -v 2>&1 | tail -15
```

Attendu : `ImportError` (la fonction `encode_procedural_observation_2d` n'existe pas).

- [ ] **Step 3 — Implement `encode_procedural_observation_2d`**

À ajouter à la fin de `mw_ia/envs/procedural_env.py` (V2-X 1D inchangé) :

```python
def encode_procedural_observation_2d(
    *,
    state: tuple[int, int],
    grid: np.ndarray,
    goal: tuple[int, int],
    max_rows: int,
    max_cols: int,
) -> np.ndarray:
    """Encode l'observation procédural pour ConvQNetwork (V2-Z).

    Format : tensor 3D shape (3, max_rows, max_cols) float32 :
    - canal 0 : position agent one-hot (un seul 1 en (row, col))
    - canal 1 : obstacles (grid.astype(float32))
    - canal 2 : goal one-hot (un seul 1 en (goal_r, goal_c))

    Pour les mazes plus petits que max_rows × max_cols, la grille est placée
    top-left, les cellules hors maze restent à zéro sur les 3 canaux (cellules
    libres, pas d'obstacle, pas de goal). L'agent CNN voit des bordures
    artificielles qu'il apprend à ignorer.

    Args:
        state: position (row, col) de l'agent.
        grid: maze actuel (rows ≤ max_rows, cols ≤ max_cols), True = obstacle.
        goal: position (goal_r, goal_c) du goal dans max_rows × max_cols.
        max_rows: nombre de rangées max (dim du ConvQNetwork).
        max_cols: nombre de colonnes max.

    Returns:
        np.ndarray[float32] de shape (3, max_rows, max_cols).
    """
    rows, cols = grid.shape
    assert rows <= max_rows and cols <= max_cols, (
        f"grid {grid.shape} > max ({max_rows}, {max_cols})"
    )
    assert 0 <= state[0] < rows and 0 <= state[1] < cols, (
        f"state {state} hors grid {grid.shape}"
    )
    assert 0 <= goal[0] < max_rows and 0 <= goal[1] < max_cols, (
        f"goal {goal} hors max ({max_rows}, {max_cols})"
    )

    obs = np.zeros((3, max_rows, max_cols), dtype=np.float32)
    obs[0, state[0], state[1]] = 1.0
    obs[1, :rows, :cols] = grid.astype(np.float32)
    obs[2, goal[0], goal[1]] = 1.0
    return obs
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_procedural_env_2d.py -v 2>&1 | tail -10
```

Attendu : `6 passed`.

- [ ] **Step 5 — Run full test suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `189 passed` (183 + 6).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/envs/procedural_env.py tests/envs/test_procedural_env_2d.py
git commit -m "feat(v2-z): add encode_procedural_observation_2d (3 channels: agent+obstacles+goal)"
```

---

## Phase 3 — `ConvQNetwork`

### Task 3 : Réseau Conv2D + FC

**Files :**
- Modify : `mw_ia/neural/conv_network.py` (impl)
- Test : `tests/neural/test_conv_network.py`

- [ ] **Step 1 — Write the 5 failing tests**

Contenu de `tests/neural/test_conv_network.py` :

```python
"""Tests V2-Z de ConvQNetwork (Conv2d + FC pour DQN spatial)."""
from __future__ import annotations

import torch

from mw_ia.neural.conv_network import ConvQNetwork


def test_forward_single_sample(cpu_device: torch.device) -> None:
    """Input (1, 3, 10, 10) → output (1, 4)."""
    net = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    x = torch.zeros(1, 3, 10, 10, device=cpu_device)
    y = net(x)
    assert y.shape == (1, 4)


def test_forward_batch(cpu_device: torch.device) -> None:
    """Input (32, 3, 10, 10) → output (32, 4)."""
    net = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    x = torch.randn(32, 3, 10, 10, device=cpu_device)
    y = net(x)
    assert y.shape == (32, 4)


def test_params_count(cpu_device: torch.device) -> None:
    """Total params ≈ 1.66M (tolerance ±5%) pour défauts (3, 10, 10)."""
    net = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    total = sum(p.numel() for p in net.parameters())
    # Conv1=3*32*9+32=896, Conv2=32*64*9+64=18496, FC1=6400*256+256=1638656, FC2=256*4+4=1028
    expected = 896 + 18_496 + 1_638_656 + 1_028
    assert abs(total - expected) <= expected * 0.05


def test_gradient_flow(cpu_device: torch.device) -> None:
    """loss.backward() produit des grads non-nulls sur toutes les couches."""
    net = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    x = torch.randn(8, 3, 10, 10, device=cpu_device, requires_grad=False)
    y = net(x)
    loss = y.sum()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name}: pas de gradient"
        assert p.grad.abs().sum().item() > 0.0, f"{name}: gradient nul"


def test_state_dict_compat(cpu_device: torch.device) -> None:
    """target.load_state_dict(online.state_dict()) round-trip exact."""
    online = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    target = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    ).to(cpu_device)
    target.load_state_dict(online.state_dict())
    x = torch.randn(4, 3, 10, 10, device=cpu_device)
    assert torch.allclose(online(x), target(x))
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_conv_network.py -v 2>&1 | tail -10
```

Attendu : `ImportError` (`ConvQNetwork` n'existe pas).

- [ ] **Step 3 — Implement `ConvQNetwork`**

Remplacer le contenu de `mw_ia/neural/conv_network.py` par :

```python
"""ConvQNetwork (Conv2d) pour V2-Z DQN à perception spatiale.

Architecture (defaults pour input 3 × 10 × 10) :
    Conv(3→32, k=3, pad=1) → ReLU
    Conv(32→64, k=3, pad=1) → ReLU
    Flatten
    Linear(64*R*C → 256) → ReLU
    Linear(256 → n_actions)

Pas de pooling pour préserver l'info spatiale sur grilles 10×10.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-cnn-perception-design.md §2
"""
from __future__ import annotations

import torch
from torch import nn


class ConvQNetwork(nn.Module):
    """Conv2d → FC pour Q-values d'un GridWorld procedural."""

    def __init__(
        self,
        *,
        in_channels: int,
        rows: int,
        cols: int,
        n_actions: int,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        padding: int = 1,
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()
        conv_layers: list[nn.Module] = []
        prev = in_channels
        for ch in conv_channels:
            conv_layers.append(nn.Conv2d(prev, ch, kernel_size=kernel_size, padding=padding))
            conv_layers.append(nn.ReLU(inplace=True))
            prev = ch
        self.conv = nn.Sequential(*conv_layers)
        flat_dim = prev * rows * cols
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, R, C) → (B, n_actions)."""
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.fc(h)
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_conv_network.py -v 2>&1 | tail -10
```

Attendu : `5 passed`.

- [ ] **Step 5 — Run full test suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `194 passed` (189 + 5).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/neural/conv_network.py tests/neural/test_conv_network.py
git commit -m "feat(v2-z): add ConvQNetwork (2 conv + 2 FC, ~1.66M params for 3x10x10)"
```

---

## Phase 4 — `ConvDQNConfig`

### Task 4 : Frozen dataclass + validation

**Files :**
- Modify : `mw_ia/config.py` (ajout de `ConvDQNConfig` en fin de fichier)
- Test : `tests/test_conv_dqn_config.py`

- [ ] **Step 1 — Write the 4 failing tests**

Contenu de `tests/test_conv_dqn_config.py` :

```python
"""Tests V2-Z de ConvDQNConfig (frozen dataclass + validation)."""
from __future__ import annotations

import pytest

from mw_ia.config import ConvDQNConfig
from mw_ia.guardrails import VariantSpec, verify_formal


def test_defaults() -> None:
    """Defaults V2-Z : (32, 64) conv, kernel 3, pad 1, fc 256."""
    cfg = ConvDQNConfig()
    assert cfg.conv_channels == (32, 64)
    assert cfg.kernel_size == 3
    assert cfg.padding == 1
    assert cfg.fc_hidden == 256
    assert cfg.gamma == 0.99
    assert cfg.batch_size == 128


def test_post_init_validation_positive() -> None:
    """Channels/kernel/fc_hidden doivent être > 0."""
    with pytest.raises(ValueError):
        ConvDQNConfig(conv_channels=(0, 64))
    with pytest.raises(ValueError):
        ConvDQNConfig(kernel_size=0)
    with pytest.raises(ValueError):
        ConvDQNConfig(fc_hidden=-1)
    with pytest.raises(ValueError):
        ConvDQNConfig(padding=-1)


def test_conv_channels_arbitrary_length() -> None:
    """(16,) ou (32, 64, 128) acceptés."""
    cfg1 = ConvDQNConfig(conv_channels=(16,))
    assert cfg1.conv_channels == (16,)
    cfg3 = ConvDQNConfig(conv_channels=(32, 64, 128))
    assert cfg3.conv_channels == (32, 64, 128)
    # Tuple vide rejeté
    with pytest.raises(ValueError):
        ConvDQNConfig(conv_channels=())


def test_aether_compat() -> None:
    """VariantSpec dérivé du ConvDQNConfig passe les invariants Aether I1-I8."""
    cfg = ConvDQNConfig()
    spec = VariantSpec(
        gamma=cfg.gamma, lr=cfg.lr,
        epsilon_start=cfg.epsilon_start, epsilon_end=cfg.epsilon_end,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
        batch_size=cfg.batch_size,
        replay_capacity=cfg.replay_capacity,
        target_sync_steps=cfg.target_sync_steps,
    )
    report = verify_formal(spec)
    assert report.passed, f"violations: {report.violations}"
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/test_conv_dqn_config.py -v 2>&1 | tail -10
```

Attendu : `ImportError` (`ConvDQNConfig` n'existe pas).

- [ ] **Step 3 — Implement `ConvDQNConfig`**

À ajouter à la fin de `mw_ia/config.py` :

```python
@dataclass(frozen=True)
class ConvDQNConfig:
    """Convolutional Deep Q-Network (V2-Z).

    Architecture : (Conv2d → ReLU)* → Flatten → Linear → ReLU → Linear.
    Input attendu : tensor (B, 3, max_rows, max_cols) via
    encode_procedural_observation_2d.

    Champs dupliqués depuis DQNConfig (pas d'héritage) pour rester frozen
    et explicit, cohérent V2-X ProceduralEnvConfig et V2-Y DRQNConfig.
    """

    # Conv-spécifique
    conv_channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    padding: int = 1
    fc_hidden: int = 256

    # Champs partagés avec DQNConfig (duplication assumée)
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000   # Default V2-X gagnant (vs V1 50000)
    replay_capacity: int = 100_000
    min_replay_to_learn: int = 1_000
    target_sync_steps: int = 1_000
    train_every: int = 4
    use_amp: bool = True
    episodes: int = 5_000
    max_steps_per_episode: int = 200

    def __post_init__(self) -> None:
        if len(self.conv_channels) == 0:
            raise ValueError("conv_channels ne peut pas être vide")
        if any(c <= 0 for c in self.conv_channels):
            raise ValueError(
                f"conv_channels doivent être > 0, reçu {self.conv_channels}"
            )
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size doit être > 0, reçu {self.kernel_size}")
        if self.padding < 0:
            raise ValueError(f"padding doit être >= 0, reçu {self.padding}")
        if self.fc_hidden <= 0:
            raise ValueError(f"fc_hidden doit être > 0, reçu {self.fc_hidden}")
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
        if self.replay_capacity <= 0:
            raise ValueError(f"replay_capacity doit être > 0, reçu {self.replay_capacity}")
        if self.min_replay_to_learn <= 0:
            raise ValueError(
                f"min_replay_to_learn doit être > 0, reçu {self.min_replay_to_learn}"
            )
        if self.target_sync_steps <= 0:
            raise ValueError(
                f"target_sync_steps doit être > 0, reçu {self.target_sync_steps}"
            )
        if self.train_every <= 0:
            raise ValueError(f"train_every doit être > 0, reçu {self.train_every}")
        if self.episodes <= 0:
            raise ValueError(f"episodes doit être > 0, reçu {self.episodes}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(
                f"max_steps_per_episode doit être > 0, reçu {self.max_steps_per_episode}"
            )
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/test_conv_dqn_config.py -v 2>&1 | tail -10
```

Attendu : `4 passed`.

- [ ] **Step 5 — Run full test suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `198 passed` (194 + 4).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/config.py tests/test_conv_dqn_config.py
git commit -m "feat(v2-z): add ConvDQNConfig (frozen dataclass + validation + Aether compat)"
```

---

## Phase 5 — `ConvDQNAgent`

### Task 5 : Agent CNN-DQN (init + act + observe + train)

**Files :**
- Modify : `mw_ia/agents/conv_dqn.py` (impl)
- Test : `tests/agents/test_conv_dqn.py`

- [ ] **Step 1 — Write the 7 failing tests**

Contenu de `tests/agents/test_conv_dqn.py` :

```python
"""Tests V2-Z de ConvDQNAgent (DQN à perception spatiale)."""
from __future__ import annotations

import numpy as np
import torch

from mw_ia.agents.conv_dqn import ConvDQNAgent
from mw_ia.config import ConvDQNConfig
from mw_ia.guardrails import VariantSpec, verify_formal


def _make_agent(cfg: ConvDQNConfig | None = None, seed: int = 0) -> ConvDQNAgent:
    cfg = cfg or ConvDQNConfig(min_replay_to_learn=4, batch_size=2, train_every=1)
    return ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=seed,
    )


def _obs() -> np.ndarray:
    return np.zeros((3, 10, 10), dtype=np.float32)


def test_init() -> None:
    """Online + target nets, buffer vide, optimizer Adam, global_step=0."""
    agent = _make_agent()
    assert agent.global_step == 0
    assert len(agent.buffer) == 0
    assert agent.epsilon == agent.cfg.epsilon_start
    # online et target ont les mêmes poids dès l'init (sync_target dans Trainer.__init__)
    for p_o, p_t in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.allclose(p_o, p_t)


def test_act_random_when_eps_high() -> None:
    """Eps=1.0 → action ∈ {0,1,2,3}, distribution non dégénérée."""
    cfg = ConvDQNConfig(epsilon_start=1.0, epsilon_end=1.0, min_replay_to_learn=10_000)
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=42,
    )
    actions = {agent.act(_obs()) for _ in range(50)}
    assert actions.issubset({0, 1, 2, 3})
    assert len(actions) > 1  # pas dégénéré sur 50 tirages avec seed=42


def test_act_greedy_when_eps_zero() -> None:
    """Eps=0 → action = argmax Q-values (déterministe pour même obs)."""
    cfg = ConvDQNConfig(epsilon_start=0.0, epsilon_end=0.0, min_replay_to_learn=10_000)
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=42,
    )
    obs = _obs()
    a1 = agent.act(obs)
    a2 = agent.act(obs)
    assert a1 == a2  # déterministe


def test_observe_pushes_buffer() -> None:
    """1 observe → buffer.size = 1, global_step = 1."""
    agent = _make_agent()
    agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    assert len(agent.buffer) == 1
    assert agent.global_step == 1


def test_target_sync() -> None:
    """Après target_sync_steps updates, target_qnet == online_qnet."""
    cfg = ConvDQNConfig(
        target_sync_steps=3, min_replay_to_learn=10_000, train_every=1_000_000,
    )
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    # Modifier manuellement online pour qu'il diffère de target
    with torch.no_grad():
        for p in agent.online.parameters():
            p.add_(1.0)
    # Vérifier qu'ils diffèrent
    diff_before = sum(
        (po - pt).abs().sum().item()
        for po, pt in zip(agent.online.parameters(), agent.target.parameters())
    )
    assert diff_before > 0.0
    # 3 observes → sync target au step 3
    for _ in range(3):
        agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    diff_after = sum(
        (po - pt).abs().sum().item()
        for po, pt in zip(agent.online.parameters(), agent.target.parameters())
    )
    assert diff_after == 0.0


def test_train_trigger_min_replay() -> None:
    """train_step ne fire pas avant len(buffer) >= max(min_replay, batch_size)."""
    cfg = ConvDQNConfig(min_replay_to_learn=5, batch_size=8, train_every=1)
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    # 7 observes : buffer.size = 7 < max(5, 8) = 8 → pas de train
    for _ in range(7):
        m = agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
        assert "loss" not in m
    # 1 observe de plus : buffer.size = 8 >= 8 → train_step fire
    m = agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    assert "loss" in m
    assert np.isfinite(m["loss"])


def test_aether_smoke() -> None:
    """Smoke E2E : VariantSpec dérivé d'un agent V2-Z passe Aether I1-I8."""
    cfg = ConvDQNConfig()
    spec = VariantSpec(
        gamma=cfg.gamma, lr=cfg.lr,
        epsilon_start=cfg.epsilon_start, epsilon_end=cfg.epsilon_end,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
        batch_size=cfg.batch_size,
        replay_capacity=cfg.replay_capacity,
        target_sync_steps=cfg.target_sync_steps,
    )
    assert verify_formal(spec).passed
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_conv_dqn.py -v 2>&1 | tail -10
```

Attendu : `ImportError` (`ConvDQNAgent` n'existe pas).

- [ ] **Step 3 — Implement `ConvDQNAgent`**

Remplacer le contenu de `mw_ia/agents/conv_dqn.py` par :

```python
"""ConvDQNAgent — DQN feedforward à perception spatiale (V2-Z).

Diffère de DQNAgent V1 par :
- Réseau ConvQNetwork (Conv2d) au lieu de QNetwork (MLP)
- Observations d'entrée shape (3, R, C) au lieu de (R*C,)
- Réutilise ReplayBuffer V1 inchangé via flatten/reshape autour de push/sample.
  Le buffer stocke des np.ndarray 1D de dim `in_channels * rows * cols` ;
  le train_step reshape en (B, C, R, C) avant le forward.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-cnn-perception-design.md §3
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from mw_ia.config import ConvDQNConfig
from mw_ia.neural.conv_network import ConvQNetwork
from mw_ia.neural.replay_buffer import Batch, ReplayBuffer


class _ConvDQNTrainer:
    """Trainer Huber + Adam + AMP + grad clip pour ConvQNetwork.

    Variante du V1 DQNTrainer qui reshape les obs flat en (B, C, R, C) avant
    le forward. Pattern AMP/grad-clip strictement identique pour rester
    cohérent avec V1/V2-X.
    """

    def __init__(
        self,
        online: ConvQNetwork,
        target: ConvQNetwork,
        *,
        in_channels: int,
        rows: int,
        cols: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cuda",
        use_amp: bool = True,
    ) -> None:
        self.online = online
        self.target = target
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.device = torch.device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self._scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.sync_target()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def step(self, batch: Batch) -> float:
        B = batch.states.shape[0]
        shape = (B, self.in_channels, self.rows, self.cols)
        states = (
            torch.from_numpy(batch.states)
            .to(self.device, non_blocking=True)
            .view(*shape)
        )
        next_states = (
            torch.from_numpy(batch.next_states)
            .to(self.device, non_blocking=True)
            .view(*shape)
        )
        actions = torch.from_numpy(batch.actions).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(batch.rewards).to(self.device, non_blocking=True)
        dones = torch.from_numpy(batch.dones).to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
            q_pred = self.online(states).gather(1, actions.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                q_next = self.target(next_states).max(dim=1).values
                target_q = rewards + self.gamma * q_next * (1.0 - dones)
            loss = self.loss_fn(q_pred, target_q)

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


class ConvDQNAgent:
    """DQN à perception spatiale (Conv2d). Contrat compatible avec runner."""

    def __init__(
        self,
        *,
        in_channels: int,
        rows: int,
        cols: int,
        n_actions: int,
        cfg: ConvDQNConfig,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        wants_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if wants_cuda else "cpu")
        self.online = ConvQNetwork(
            in_channels=in_channels, rows=rows, cols=cols, n_actions=n_actions,
            conv_channels=cfg.conv_channels, kernel_size=cfg.kernel_size,
            padding=cfg.padding, fc_hidden=cfg.fc_hidden,
        ).to(self.device)
        self.target = ConvQNetwork(
            in_channels=in_channels, rows=rows, cols=cols, n_actions=n_actions,
            conv_channels=cfg.conv_channels, kernel_size=cfg.kernel_size,
            padding=cfg.padding, fc_hidden=cfg.fc_hidden,
        ).to(self.device)
        self.trainer = _ConvDQNTrainer(
            self.online, self.target,
            in_channels=in_channels, rows=rows, cols=cols,
            lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
        )
        obs_dim = in_channels * rows * cols
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
        assert state.shape == (self.in_channels, self.rows, self.cols), (
            f"state {state.shape} != expected ({self.in_channels}, {self.rows}, {self.cols})"
        )
        if (not greedy) and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.online(x)
            return int(q.argmax(dim=1).item())

    def observe(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, done: bool,
    ) -> dict[str, float]:
        assert state.shape == (self.in_channels, self.rows, self.cols)
        assert next_state.shape == (self.in_channels, self.rows, self.cols)
        self.buffer.push(state.flatten(), action, reward, next_state.flatten(), done)
        self.global_step += 1
        metrics: dict[str, float] = {"epsilon": self.epsilon}
        train_threshold = max(self.cfg.min_replay_to_learn, self.cfg.batch_size)
        if (
            len(self.buffer) >= train_threshold
            and self.global_step % self.cfg.train_every == 0
        ):
            batch = self.buffer.sample(self.cfg.batch_size)
            self.last_loss = self.trainer.step(batch)
            metrics["loss"] = self.last_loss
        if self.global_step % self.cfg.target_sync_steps == 0:
            self.trainer.sync_target()
            self.target_syncs += 1
        return metrics

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

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_conv_dqn.py -v 2>&1 | tail -15
```

Attendu : `7 passed`.

- [ ] **Step 5 — Run full test suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `205 passed` (198 + 7).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/agents/conv_dqn.py tests/agents/test_conv_dqn.py
git commit -m "feat(v2-z): add ConvDQNAgent (Conv2d agent + internal trainer, reuses V1 buffer)"
```

---

## Phase 6 — `ConvProceduralDQNRunner`

### Task 6 : Runner intégrant scheduler, bucket tracker et CNN

**Files :**
- Modify : `mw_ia/training/runner.py` (ajout classe en fin de fichier)
- Test : `tests/training/test_conv_procedural_runner.py`

- [ ] **Step 1 — Write the 3 failing tests**

Contenu de `tests/training/test_conv_procedural_runner.py` :

```python
"""Tests V2-Z d'intégration de ConvProceduralDQNRunner."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from mw_ia.config import (
    ConvDQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ConvProceduralDQNRunner, RunnerCallbacks


def _build_env() -> ProceduralGridWorld:
    cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                              min_density=0.0, max_density=0.20)
    gen = RandomObstaclesGenerator(
        rows=10, cols=10, start=(0, 0), goal=(9, 9),
        min_density=cfg.min_density, max_density=cfg.max_density,
    )
    return ProceduralGridWorld(cfg=cfg, generator=gen)


def _build_runner(
    *, episodes: int, callbacks: RunnerCallbacks | None = None,
) -> ConvProceduralDQNRunner:
    env = _build_env()
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.0, max_density=0.20)
    dqn_cfg = ConvDQNConfig(
        episodes=episodes, max_steps_per_episode=30,
        batch_size=8, min_replay_to_learn=8, train_every=1,
        epsilon_decay_steps=200, target_sync_steps=50,
        replay_capacity=500, use_amp=False,
    )
    sched_cfg = SchedulerConfig(initial_difficulty=0.0)
    train_cfg = TrainingConfig(log_every_episodes=100)
    return ConvProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=callbacks or RunnerCallbacks(),
        device="cpu", seed=0,
    )


def test_single_episode_runs() -> None:
    """1 épisode sans crash, metrics récoltés."""
    runner = _build_runner(episodes=1)
    runner.run()
    assert len(runner.metrics.episode_rewards) == 1


def test_callbacks_fired() -> None:
    """on_maze_changed, on_episode, on_step appelés au moins une fois."""
    counts: dict[str, int] = {"maze": 0, "ep": 0, "step": 0}

    def on_maze(**kw: Any) -> None:
        counts["maze"] += 1

    def on_ep(**kw: Any) -> None:
        counts["ep"] += 1

    def on_step(**kw: Any) -> None:
        counts["step"] += 1

    cb = RunnerCallbacks(on_maze_changed=on_maze, on_episode=on_ep, on_step=on_step)
    runner = _build_runner(episodes=2, callbacks=cb)
    runner.run()
    assert counts["maze"] >= 2
    assert counts["ep"] == 2
    assert counts["step"] >= 1


def test_smoke_10_episodes_no_nan() -> None:
    """10 épisodes : metrics.losses tous finis, winrate ∈ [0, 1]."""
    runner = _build_runner(episodes=10)
    runner.run()
    losses = runner.metrics.losses
    assert all(math.isfinite(l) for l in losses), f"loss non-finite : {losses}"
    wr = runner.metrics.winrate()
    assert 0.0 <= wr <= 1.0
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/training/test_conv_procedural_runner.py -v 2>&1 | tail -10
```

Attendu : `ImportError` (`ConvProceduralDQNRunner` n'existe pas).

- [ ] **Step 3 — Implement `ConvProceduralDQNRunner`**

À ajouter à la fin de `mw_ia/training/runner.py`. D'abord adapter l'import en haut du fichier — remplacer la ligne :

```python
from mw_ia.config import DQNConfig, DRQNConfig, ProceduralEnvConfig, QLearningConfig, SchedulerConfig, TrainingConfig
```

par :

```python
from mw_ia.config import ConvDQNConfig, DQNConfig, DRQNConfig, ProceduralEnvConfig, QLearningConfig, SchedulerConfig, TrainingConfig
```

Et ajouter en haut, à côté des imports d'agents :

```python
from mw_ia.agents.conv_dqn import ConvDQNAgent
```

Puis à côté de l'import `encode_procedural_observation`, ajouter :

```python
from mw_ia.envs.procedural_env import ProceduralGridWorld, encode_procedural_observation, encode_procedural_observation_2d
```

Puis ajouter en fin de fichier :

```python
class ConvProceduralDQNRunner(_BaseRunner):
    """Boucle DQN procedural avec perception spatiale (V2-Z).

    Différences avec ProceduralDQNRunner V2-X :
    - Agent ConvDQNAgent (Conv2d) au lieu de DQNAgent (MLP).
    - Observation encode_procedural_observation_2d shape (3, R, C).
    - Callbacks GUI identiques V2-X (signaux maze_changed + difficulty_updated).

    Scheduler defaults V2-X (`update_interval=200`, `step=0.05`) — cohérent
    DQN feedforward, distinct du runner V2-Y LSTM.
    """

    def __init__(
        self,
        *,
        env: ProceduralGridWorld,
        proc_cfg: ProceduralEnvConfig,
        dqn_cfg: ConvDQNConfig,
        sched_cfg: SchedulerConfig,
        train_cfg: TrainingConfig,
        callbacks: RunnerCallbacks,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        super().__init__(train_cfg, callbacks)
        self.env = env
        self.proc_cfg = proc_cfg
        self.dqn_cfg = dqn_cfg
        self.sched_cfg = sched_cfg
        self.scheduler = AdaptiveDifficultyScheduler(sched_cfg)
        self.bucket_tracker = DifficultyBucketTracker(train_cfg)
        self.agent = ConvDQNAgent(
            in_channels=3, rows=proc_cfg.max_rows, cols=proc_cfg.max_cols,
            n_actions=4, cfg=dqn_cfg, device=device, seed=seed,
        )

    def run(self) -> None:
        self.callbacks.fire_log(
            "info",
            f"Procedural Conv-DQN ({self.proc_cfg.mode}) sur {self.agent.device} démarrage"
        )
        self.callbacks.fire_log(
            "info",
            f"Config: conv_channels={self.dqn_cfg.conv_channels} "
            f"fc_hidden={self.dqn_cfg.fc_hidden} "
            f"epsilon_decay_steps={self.dqn_cfg.epsilon_decay_steps} "
            f"min_density={self.proc_cfg.min_density} "
            f"max_density={self.proc_cfg.max_density} "
            f"obs_shape=(3, {self.proc_cfg.max_rows}, {self.proc_cfg.max_cols}) "
            f"seed={self.train_cfg.seed}"
        )
        for ep in range(self.dqn_cfg.episodes):
            if self._stop:
                return

            self.env.set_difficulty(self.scheduler.current)
            state, info = self.env.reset(seed=ep)
            maze = info["maze"]
            difficulty = info["difficulty"]
            goal = self.env.inner.cfg.goal
            self.callbacks.fire_maze_changed(maze=maze, episode_id=ep, difficulty=difficulty)

            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated) and ep_len < self.dqn_cfg.max_steps_per_episode:
                if self._stop:
                    return
                while self._paused and not self._stop:
                    pass
                obs = encode_procedural_observation_2d(
                    state=state, grid=maze, goal=goal,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                a = self.agent.act(obs)
                s2, r, terminated, truncated, _ = self.env.step(a)
                next_obs = encode_procedural_observation_2d(
                    state=s2, grid=maze, goal=goal,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                m = self.agent.observe(obs, a, r, next_obs, terminated or truncated)
                if "loss" in m:
                    self.metrics.record_loss(m["loss"])
                    self.callbacks.fire_loss(self.agent.global_step, m["loss"])
                self.callbacks.fire_epsilon(self.agent.global_step, m["epsilon"])
                self.metrics.record_epsilon(m["epsilon"])
                self.callbacks.fire_step(state=state, action=a, reward=r, next_state=s2)
                state = s2
                ep_reward += r
                ep_len += 1

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

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/training/test_conv_procedural_runner.py -v 2>&1 | tail -10
```

Attendu : `3 passed`.

- [ ] **Step 5 — Run full test suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `208 passed` (205 + 3).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/training/runner.py tests/training/test_conv_procedural_runner.py
git commit -m "feat(v2-z): add ConvProceduralDQNRunner (CNN-DQN + scheduler + bucket tracker)"
```

---

## Phase 7 — CLI + GUI + CI smoke

### Task 7 : CLI script `train_cnn_dqn_procedural.py`

**Files :**
- Create : `scripts/train_cnn_dqn_procedural.py`

- [ ] **Step 1 — Write the CLI script**

Contenu de `scripts/train_cnn_dqn_procedural.py` :

```python
"""Entraînement CNN-DQN procedural headless (V2-Z, CLI).

Usage :
    python scripts/train_cnn_dqn_procedural.py --episodes 200 --mode obstacles --device cpu
"""
from __future__ import annotations

import argparse
import sys

from mw_ia.config import (
    ConvDQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import PerfectMazeGenerator, RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ConvProceduralDQNRunner, RunnerCallbacks


def _print_log(level: str, msg: str) -> None:
    print(f"[{level:7s}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="CNN-DQN procedural training (V2-Z)")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--mode", choices=("obstacles", "maze"), default="obstacles")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conv-channels", type=int, nargs="+", default=[32, 64],
                        help="Tailles des couches conv (ex: --conv-channels 16 32)")
    parser.add_argument("--fc-hidden", type=int, default=256,
                        help="Taille de la couche FC après le bloc conv (default 256)")
    parser.add_argument("--epsilon-decay-steps", type=int, default=200_000,
                        help="Steps pour passer ε de start à end (default V2-Z : 200000)")
    parser.add_argument("--target-sync-steps", type=int, default=1_000,
                        help="Périodicité de la sync target ← online (default 1000)")
    parser.add_argument("--scheduler-update-interval", type=int, default=200,
                        help="Périodicité (ép) du scheduler (default V2-X : 200)")
    parser.add_argument("--scheduler-step", type=float, default=0.05,
                        help="Pas de difficulté du scheduler (default V2-X : 0.05)")
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
    dqn_cfg = ConvDQNConfig(
        episodes=args.episodes,
        conv_channels=tuple(args.conv_channels),
        fc_hidden=args.fc_hidden,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_sync_steps=args.target_sync_steps,
    )
    sched_cfg = SchedulerConfig(
        update_interval=args.scheduler_update_interval,
        step=args.scheduler_step,
    )
    train_cfg = TrainingConfig()

    cb = RunnerCallbacks(on_log=_print_log)
    runner = ConvProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
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

- [ ] **Step 2 — Run a CPU smoke test (10 episodes)**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_dqn_procedural.py --episodes 10 --mode obstacles --device cpu
```

Attendu :
- pas de crash
- `Final : winrate=X.XX%, difficulty=0.00` (10 ép trop court pour scheduler update)
- 5 buckets listés

- [ ] **Step 3 — Run full test suite (should still be 208)**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `208 passed`.

- [ ] **Step 4 — Commit**

```bash
git add scripts/train_cnn_dqn_procedural.py
git commit -m "feat(v2-z): add scripts/train_cnn_dqn_procedural.py CLI"
```

---

### Task 8 : Bouton GUI + slot `on_start_procedural_cnn`

**Files :**
- Modify : `mw_ia/gui/widgets/control_panel.py`
- Modify : `mw_ia/gui/app.py`

- [ ] **Step 1 — Extend ControlPanel**

Modifier `mw_ia/gui/widgets/control_panel.py` — 5 edits ciblés (anchored sur strings existantes pour robustesse) :

1. Ajouter le nouveau signal — sous `start_procedural_clicked = pyqtSignal()`, insérer :

```python
    start_procedural_cnn_clicked = pyqtSignal()
```

2. Ajouter le bouton — sous la ligne `self.btn_start_procedural = QPushButton("Démarrer (procedural)")`, insérer :

```python
        self.btn_start_procedural_cnn = QPushButton("Démarrer (procedural CNN)")
```

3. Étendre le tuple du layout — remplacer la ligne `for b in (self.btn_start, self.btn_start_procedural, self.btn_pause, self.btn_reset, self.btn_save, self.btn_load):` par :

```python
        for b in (
            self.btn_start, self.btn_start_procedural, self.btn_start_procedural_cnn,
            self.btn_pause, self.btn_reset, self.btn_save, self.btn_load,
        ):
```

4. Connecter le signal — sous la ligne `self.btn_start_procedural.clicked.connect(self.start_procedural_clicked)`, insérer :

```python
        self.btn_start_procedural_cnn.clicked.connect(self.start_procedural_cnn_clicked)
```

5. Adapter `set_running()` — remplacer le corps entier de la méthode `set_running` par :

```python
    def set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_start_procedural.setEnabled(not running)
        self.btn_start_procedural_cnn.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_reset.setEnabled(not running)
        self.btn_load.setEnabled(not running)
```

- [ ] **Step 2 — Extend MainWindow.on_start_procedural_cnn**

Modifier `mw_ia/gui/app.py` — 4 edits ciblés (anchored sur strings existantes) :

1. Étendre l'import config. Remplacer la ligne `from mw_ia.config import Config, DQNConfig, ProceduralEnvConfig, SchedulerConfig` par :

```python
from mw_ia.config import Config, ConvDQNConfig, DQNConfig, ProceduralEnvConfig, SchedulerConfig
```

2. Étendre l'import runner. Remplacer la ligne `from mw_ia.training.runner import DQNRunner, ProceduralDQNRunner, RunnerCallbacks` par :

```python
from mw_ia.training.runner import ConvProceduralDQNRunner, DQNRunner, ProceduralDQNRunner, RunnerCallbacks
```

3. Brancher le signal dans `MainWindow.__init__`. Sous la ligne `self.controls.start_procedural_clicked.connect(self.on_start_procedural)`, insérer :

```python
        self.controls.start_procedural_cnn_clicked.connect(self.on_start_procedural_cnn)
```

4. Ajouter le slot juste après le slot `on_start_procedural` (avant `on_pause`) :

```python
    @pyqtSlot()
    def on_start_procedural_cnn(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc_cfg = ProceduralEnvConfig(mode="obstacles")
        gen = RandomObstaclesGenerator(
            rows=proc_cfg.max_rows, cols=proc_cfg.max_cols,
            start=(0, 0), goal=(proc_cfg.max_rows - 1, proc_cfg.max_cols - 1),
            min_density=proc_cfg.min_density, max_density=proc_cfg.max_density,
        )
        proc_env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
        # Defaults V2-Z : ConvDQNConfig() défini avec defaults gagnants
        # (epsilon_decay_steps=200000, conv_channels=(32, 64), fc_hidden=256).
        cnn_cfg = ConvDQNConfig(episodes=self.config.dqn.episodes)
        runner = ConvProceduralDQNRunner(
            env=proc_env, proc_cfg=proc_cfg, dqn_cfg=cnn_cfg,
            sched_cfg=SchedulerConfig(), train_cfg=self.config.training,
            callbacks=RunnerCallbacks(), device=device,
            seed=self.config.training.seed,
        )
        self.thread = TrainingThread(runner)
        self.thread.step_signal.connect(self._on_step)
        self.thread.episode_signal.connect(self._on_episode)
        self.thread.loss_signal.connect(self._on_loss)
        self.thread.epsilon_signal.connect(self._on_epsilon)
        self.thread.log_signal.connect(self.log.append)
        self.thread.finished_signal.connect(self._on_finished)
        self.thread.maze_changed_signal.connect(self.gridview.on_maze_changed)
        self.thread.maze_changed_signal.connect(self.difficulty_label.on_maze_changed)
        self.thread.difficulty_signal.connect(self._on_difficulty)
        self.controls.set_running(True)
        self.thread.start()
```

- [ ] **Step 3 — Smoke test GUI manuel**

```bash
source .venv/Scripts/activate && python scripts/launch_gui.py
```

Attendu (manuel) :
- 3 boutons "Démarrer" affichés (V1, procedural, procedural CNN)
- Clic sur "Démarrer (procedural CNN)" → training démarre, courbes se rafraîchissent, label difficulty visible, pas de crash dans la console
- Stop avec "Réinitialiser" ou fermer la fenêtre

- [ ] **Step 4 — Run full test suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `208 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/gui/widgets/control_panel.py mw_ia/gui/app.py
git commit -m "feat(v2-z): add 'Démarrer (procedural CNN)' GUI button + on_start_procedural_cnn slot"
```

---

### Task 9 : CI workflow smoke

**Files :**
- Modify : `.github/workflows/aether_verify.yml`

- [ ] **Step 1 — Add V2-Z smoke step after V2-Y**

Dans `.github/workflows/aether_verify.yml`, ajouter un nouveau step après le step `Smoke test recurrent procedural training` (qui se termine ligne 31, par `--mode maze --device cpu`). Pattern strict identique aux 2 steps existants :

```yaml
      - name: Smoke test conv procedural training
        run: |
          python scripts/train_cnn_dqn_procedural.py --episodes 10 --mode obstacles --device cpu
          python scripts/train_cnn_dqn_procedural.py --episodes 10 --mode maze --device cpu
```

Insérer entre les lignes 31 et 32 (donc juste après le bloc Smoke recurrent et avant le job `aether-files:`).

- [ ] **Step 3 — Vérifier le YAML syntaxiquement**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/aether_verify.yml'))" && echo "YAML OK"
```

Attendu : `YAML OK`.

- [ ] **Step 4 — Run full test suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `208 passed`.

- [ ] **Step 5 — Commit**

```bash
git add .github/workflows/aether_verify.yml
git commit -m "ci(v2-z): add smoke train_cnn_dqn_procedural to aether_verify.yml"
```

---

## Phase 8 — README + DoD + tag

### Task 10 : Section V2-Z dans README.md et CLAUDE.md, smoke DoD, tag

**Files :**
- Modify : `README.md`
- Modify : `CLAUDE.md`

- [ ] **Step 1 — Smoke E2E manuel GPU 50 ép**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_dqn_procedural.py --episodes 50 --mode obstacles --device cuda
```

Attendu :
- Pas de crash
- `Final : winrate=X.XX%, difficulty=0.00` (50 ép trop court pour franchir diff=0.05)
- Loss finite tout du long

- [ ] **Step 2 — Aether re-verify**

```bash
bash aether/verify_all.sh
```

Attendu : `8 OK`.

- [ ] **Step 3 — Add V2-Z section to README.md**

Localiser la section V2-Y dans `README.md` (chercher `V2-Y` ou `v0.2.0-y`). Ajouter en dessous une section parallèle :

```markdown
## V2-Z — CNN perception spatiale (DQN feedforward)

**Tag** : `v0.2.0-z` — **Tests** : 208 verts (183 baseline + 25 V2-Z)

**Motivation** : V2-X (MLP) et V2-Y (LSTM) plafonnent tous deux à `diff ≈ 0.05`.
Le bottleneck est la représentation spatiale 1D — un encoding `concat(position_one_hot, grid_flatten)`
détruit la structure 2D du maze. V2-Z remplace l'encoder par un tensor 3-canaux
(agent + obstacles + goal) et le réseau par un Conv2D.

**Architecture** : `Conv(3→32, k=3, pad=1) → ReLU → Conv(32→64, k=3, pad=1) → ReLU →
Flatten → Linear(64·R·C → 256) → ReLU → Linear(256 → 4)`. Pas de pooling pour préserver
l'info spatiale sur 10×10. ~1.66M params.

**Lancer V2-Z procedural CNN headless** :

\`\`\`bash
source .venv/Scripts/activate && python scripts/train_cnn_dqn_procedural.py \\
    --episodes 5000 --mode obstacles --device cuda
\`\`\`

**Defaults gagnants V2-Z** (recette consolidée à itérer empiriquement) :
- `--conv-channels 32 64` (default)
- `--fc-hidden 256` (default)
- `--epsilon-decay-steps 200000` (default V2-Z, hérité V2-X)
- `--scheduler-update-interval 200` (default V2-X)
- `--scheduler-step 0.05` (default V2-X)

**GUI** : bouton "Démarrer (procedural CNN)" disponible dans `python scripts/launch_gui.py`.
```

- [ ] **Step 4 — Add V2-Z section to CLAUDE.md**

Dans `CLAUDE.md`, mettre à jour le tableau "Sous-projets — décomposition" en ajoutant une ligne Z :

```markdown
| **Z** | CNN perception spatiale (roadmap #2) | ✅ Livré (tag `v0.2.0-z`) |
```

Ajouter une section après "V2-Y — état final des phases" :

```markdown
### V2-Z — état final des phases (livraison 2026-05-22)

> **Note exécutant** : remplacer `2026-05-22` par la date réelle du tag `v0.2.0-z` si différente (format YYYY-MM-DD).

| Phase | Tâches | Statut | Tests | Commits |
|---|---|---|---|---|
| 1 — Setup scaffold | T1 | ✅ | — | 1 |
| 2 — `encode_procedural_observation_2d` | T2 | ✅ | 6 | 1 |
| 3 — `ConvQNetwork` | T3 | ✅ | 5 | 1 |
| 4 — `ConvDQNConfig` + validation + Aether compat | T4 | ✅ | 4 | 1 |
| 5 — `ConvDQNAgent` (+ `_ConvDQNTrainer` interne) | T5 | ✅ | 7 | 1 |
| 6 — `ConvProceduralDQNRunner` | T6 | ✅ | 3 | 1 |
| 7 — CLI + GUI button + CI smoke | T7-T9 | ✅ | — | 3 |
| 8 — README V2-Z + DoD + tag `v0.2.0-z` | T10 | ✅ | — | 1 + tag |

### Composants V2-Z livrés

| Composant | Fichier | Rôle |
|---|---|---|
| `encode_procedural_observation_2d` | `mw_ia/envs/procedural_env.py` | Encoder 3 canaux (agent + obstacles + goal) shape `(3, R, C)` |
| `ConvQNetwork` | `mw_ia/neural/conv_network.py` | Conv(3→32) → Conv(32→64) → FC(256) → FC(4), ~1.66M params pour 10×10 |
| `ConvDQNConfig` | `mw_ia/config.py` | Frozen dataclass + validation + champs DQN dupliqués (pas d'héritage) |
| `ConvDQNAgent` + `_ConvDQNTrainer` | `mw_ia/agents/conv_dqn.py` | DQN à perception spatiale, réutilise `ReplayBuffer` V1 (flatten/reshape autour de push/sample) |
| `ConvProceduralDQNRunner` | `mw_ia/training/runner.py` | Extension parallèle à V2-X/V2-Y, scheduler defaults V2-X |
| CLI | `scripts/train_cnn_dqn_procedural.py` | Flags : `--conv-channels`, `--fc-hidden`, `--epsilon-decay-steps`, `--target-sync-steps`, `--scheduler-update-interval`, `--scheduler-step` |
| GUI button | `mw_ia/gui/widgets/control_panel.py` + `mw_ia/gui/app.py` | "Démarrer (procedural CNN)" + slot `on_start_procedural_cnn` |

### Décisions techniques V2-Z

- **3 canaux** (agent + obstacles + goal) plutôt que 2 : goal explicite → robuste si on varie taille/position du goal plus tard. Standard DeepMind.
- **Pas de pooling** : grille 10×10 trop petite pour downsampling sans perte d'info.
- **`ReplayBuffer` V1 réutilisé via flatten/reshape** : évite la duplication d'un `SpatialReplayBuffer`. Le `_ConvDQNTrainer` interne au module `conv_dqn.py` reshape `(B, 3, R, C)` au moment du train_step.
- **Pas d'héritage de `DQNConfig`** : duplication explicite des champs partagés pour rester frozen et explicit, cohérent V2-X `ProceduralEnvConfig` et V2-Y `DRQNConfig`.
- **Scheduler defaults V2-X** (`update=200`, `step=0.05`) : CNN feedforward, distinct V2-Y LSTM (`update=50`).
- **Asserts ajoutés** : `state` hors grille, `grid` > max_size, `goal` hors max_size → `AssertionError` (cohérent V2-X).

### V2-Z — pièges connus

1. **Padding zéros = bordure artificielle sur 3 canaux** : pour 10×10 fixe sans effet, mais si max_size > taille réelle du maze, le CNN voit une zone "vide" en bas-droite. Mitigation possible : 4ᵉ canal "valid region mask". Pas en MVP.
2. **VRAM si on monte à `max_size=20`** : FC1 = `20*20*64*256 ≈ 6.5M params` (vs 1.64M pour 10×10). OK sur 12 GB mais penser à `AdaptiveAvgPool2d` ou stride=2 si on va plus large.
3. **`conv_channels=(32, 64)` peut être overkill** : 99% des params dans FC1. Tester `--conv-channels 16 32` voire `8 16` post-livraison.
4. **Scheduler `update=200` peut être trop patient pour CNN** : à confirmer empiriquement vs `--scheduler-update-interval 100` (intermédiaire entre V2-X 200 et V2-Y 50).
```

Mettre à jour la section "Instructions pour la prochaine session" pour refléter le nouvel état : V2-Z livré, prochain sous-projet à brainstormer (V2-W Double DQN selon outcome empirique de V2-Z).

- [ ] **Step 5 — Run full test suite + Aether final**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
bash aether/verify_all.sh
```

Attendu : `208 passed` + `8 OK`.

- [ ] **Step 6 — Commit doc**

```bash
git add README.md CLAUDE.md
git commit -m "docs(v2-z): add V2-Z section (CNN perception) to README + CLAUDE.md"
```

- [ ] **Step 7 — Tag**

```bash
git tag v0.2.0-z
git tag --list | tail -5
```

Attendu : `v0.1.0`, `v0.2.0-a`, `v0.2.0-x`, `v0.2.0-y`, `v0.2.0-z`.

- [ ] **Step 8 — DoD final récap**

Imprimer le récap (output texte direct, pas via tool) :

```
=== V2-Z DoD CHECKLIST ===
[ ] pytest -q → 208 passed
[ ] bash aether/verify_all.sh → 8 OK
[ ] smoke train_cnn_dqn_procedural.py --episodes 50 --device cuda OK
[ ] smoke GUI manuel : bouton "Démarrer (procedural CNN)" lance le runner
[ ] V2-Z section dans README.md + CLAUDE.md
[ ] Tag v0.2.0-z posé
[ ] V2-A / V2-X / V2-Y tags intacts (pas de force-push)
==> Phase post-livraison : 2 entraînements GPU 5000 ép pour valider critère succès
    final winrate ≥ 95% @ diff=0.05 + diff ≥ 0.10 stablement atteinte.
```

---

## Récapitulatif

- **10 tasks** réparties sur **8 phases**
- **25 nouveaux tests** (183 → 208)
- **~10 commits** sur `main`
- **Tag livraison** : `v0.2.0-z`
- **DoD bloquante** : pytest 208 + Aether 8 OK + smoke E2E manuel GPU + GUI + tag
- **DoD non-bloquante (objectif scientifique)** : 2 runs 5000 ép GPU reproductibles, match V2-Y @ diff=0.05 et franchir diff=0.10. Si KO → escalade vers V2-W Double DQN.
