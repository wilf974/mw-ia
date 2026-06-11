# MW_IA V2-ZY CNN + LSTM + Double DQN combiné — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combiner les 3 leviers livrés V2-Z (Conv2D perception spatiale), V2-Y (LSTM mémoire temporelle), V2-W (Double DQN stabilité Q-target) en un agent unifié, validé via V2-V eval rigoureux. Hypothèse : combo nécessaire et suffisant pour franchir la cible "4/5 seeds best ≥ 70 % @ diff=0.30 en eval rigoureux".

**Architecture:** Réseau `Conv → Flatten → LSTM → FC` (Hausknecht-style). Réutilise `SequenceReplayBuffer` V2-Y (obs flatten 1D, network reshape interne en 3D pour conv). Réutilise `RecurrentDQNTrainer` V2-Y étendu avec flag `double_dqn` (default False, V2-ZY active True). Nouveau runner `ConvRecurrentProceduralDQNRunner` intègre V2-V eval+best-checkpoint dès l'origine. Extension V2-V `PeriodicEvaluator.evaluate()` duck-types `agent.begin_episode()` pour reset hidden state entre rollouts eval.

**Tech Stack:** Python 3.13, PyTorch (cu128, nn.Conv2d + nn.LSTM), NumPy, pytest. Réutilise infrastructure V2-Y/V2-Z/V2-W/V2-V intégralement.

**Spec source:** `docs/superpowers/specs/2026-05-23-mw-ia-cnn-lstm-double-dqn-design.md`

**État initial:** Branche `main`, tags `v0.1.0` + `v0.2.0-a` + `v0.2.0-x` + `v0.2.0-y` + `v0.2.0-z` + `v0.2.0-w` + `v0.2.0-v` posés. **231 tests pytest verts**. Dernier commit avant V2-ZY : `7e3324c` (spec V2-ZY).

---

## Phase 1 — Setup scaffold

### Task 1 : Créer les fichiers code + tests vides

**Files :**
- Create : `mw_ia/neural/conv_recurrent.py` (docstring seulement)
- Create : `mw_ia/agents/conv_recurrent_dqn.py` (docstring seulement)
- Create : `tests/neural/test_conv_recurrent.py` (vide)
- Create : `tests/agents/test_conv_recurrent_dqn.py` (vide)
- Create : `tests/training/test_conv_recurrent_procedural_runner.py` (vide)
- Create : `tests/test_conv_recurrent_dqn_config.py` (vide)

- [ ] **Step 1 — Verify initial state**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `231 passed`.

- [ ] **Step 2 — Create scaffold files**

Contenu de `mw_ia/neural/conv_recurrent.py` :

```python
"""ConvRecurrentQNetwork — Conv2D + LSTM + FC pour V2-ZY (combo CNN + LSTM + Double DQN).

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-cnn-lstm-double-dqn-design.md §2
"""
from __future__ import annotations
```

Contenu de `mw_ia/agents/conv_recurrent_dqn.py` :

```python
"""ConvRecurrentDQNAgent — agent V2-ZY combinant perception spatiale (Conv) + mémoire (LSTM) + Double DQN.

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-cnn-lstm-double-dqn-design.md §2
"""
from __future__ import annotations
```

Les 4 fichiers de tests restent vides à ce stade.

- [ ] **Step 3 — Verify pytest still passes**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `231 passed`.

- [ ] **Step 4 — Commit scaffold**

```bash
git add mw_ia/neural/conv_recurrent.py mw_ia/agents/conv_recurrent_dqn.py \
        tests/neural/test_conv_recurrent.py tests/agents/test_conv_recurrent_dqn.py \
        tests/training/test_conv_recurrent_procedural_runner.py \
        tests/test_conv_recurrent_dqn_config.py
git commit -m "chore(v2-zy): scaffold conv_recurrent + conv_recurrent_dqn modules + empty test files"
```

---

## Phase 2 — `ConvRecurrentQNetwork`

### Task 2 : Réseau Conv2D + LSTM + FC

**Files :**
- Modify : `mw_ia/neural/conv_recurrent.py` (impl)
- Test : `tests/neural/test_conv_recurrent.py`

- [ ] **Step 1 — Write the 5 failing tests**

Contenu de `tests/neural/test_conv_recurrent.py` :

```python
"""Tests V2-ZY de ConvRecurrentQNetwork (Conv + LSTM + FC)."""
from __future__ import annotations

import torch

from mw_ia.neural.conv_recurrent import ConvRecurrentQNetwork


def test_forward_single_step_with_hidden(cpu_device: torch.device) -> None:
    """Input (seq=1, batch=1, in_channels * rows * cols) + hidden None → q (1, 1, 4) + new hidden."""
    net = ConvRecurrentQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, lstm_hidden=128,
    ).to(cpu_device)
    x = torch.zeros(1, 1, 3 * 10 * 10, device=cpu_device)
    q, hidden = net(x, None)
    assert q.shape == (1, 1, 4)
    assert isinstance(hidden, tuple) and len(hidden) == 2
    h, c = hidden
    assert h.shape == (1, 1, 128)
    assert c.shape == (1, 1, 128)


def test_forward_batch_sequence(cpu_device: torch.device) -> None:
    """Input (seq=32, batch=8, 300) → q (32, 8, 4)."""
    net = ConvRecurrentQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, lstm_hidden=128,
    ).to(cpu_device)
    x = torch.randn(32, 8, 3 * 10 * 10, device=cpu_device)
    q, hidden = net(x, None)
    assert q.shape == (32, 8, 4)
    assert hidden[0].shape == (1, 8, 128)
    assert hidden[1].shape == (1, 8, 128)


def test_hidden_state_propagation(cpu_device: torch.device) -> None:
    """Forward avec hidden propagé != forward avec hidden None."""
    net = ConvRecurrentQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, lstm_hidden=128,
    ).to(cpu_device)
    torch.manual_seed(42)
    x1 = torch.randn(1, 1, 300, device=cpu_device)
    x2 = torch.randn(1, 1, 300, device=cpu_device)
    _, h1 = net(x1, None)
    q_with_hidden, _ = net(x2, h1)
    q_no_hidden, _ = net(x2, None)
    assert not torch.allclose(q_with_hidden, q_no_hidden)


def test_gradient_flow(cpu_device: torch.device) -> None:
    """loss.backward() produit des grads non-nuls sur Conv1, Conv2, LSTM, FC."""
    net = ConvRecurrentQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, lstm_hidden=128,
    ).to(cpu_device)
    x = torch.randn(8, 4, 300, device=cpu_device, requires_grad=False)
    q, _ = net(x, None)
    loss = q.sum()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} grad is None"
        assert p.grad.abs().sum().item() > 0.0, f"{name} grad sum is 0"


def test_params_count(cpu_device: torch.device) -> None:
    """Total params ≈ 3.3M (±5%) pour défauts 3x10x10 + LSTM 128."""
    net = ConvRecurrentQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, lstm_hidden=128,
    ).to(cpu_device)
    total = sum(p.numel() for p in net.parameters())
    expected = 896 + 18_496 + 3_343_488 + 516
    assert abs(total - expected) <= expected * 0.05
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_conv_recurrent.py -v 2>&1 | tail -15
```

Attendu : `ImportError` (`ConvRecurrentQNetwork` n'existe pas).

- [ ] **Step 3 — Implement `ConvRecurrentQNetwork`**

Remplacer le contenu de `mw_ia/neural/conv_recurrent.py` par :

```python
"""ConvRecurrentQNetwork — Conv2D + LSTM + FC pour V2-ZY.

Architecture (pattern Hausknecht DRQN appliqué à CNN) :
    Input  : (seq, batch, in_channels * rows * cols)   ← obs flat from buffer
    Reshape: (seq*batch, in_channels, rows, cols)
    Conv2d block → ReLU
    Flatten → (seq*batch, conv_out_channels * rows * cols)
    Reshape: (seq, batch, conv_features)
    LSTM(batch_first=False)
    FC(lstm_hidden → n_actions)
    Output : (seq, batch, n_actions)

Convention V2-Y : batch_first=False.

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-cnn-lstm-double-dqn-design.md §2
"""
from __future__ import annotations

import torch
from torch import nn


class ConvRecurrentQNetwork(nn.Module):
    """Conv block + LSTM + FC pour DRQN spatial.

    Accepte des observations 1D-flat depuis le buffer (compat
    SequenceReplayBuffer V2-Y) et reshape internement en 3D pour le Conv block.
    """

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
        lstm_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        self.n_actions = n_actions
        self.lstm_hidden = lstm_hidden

        conv_layers: list[nn.Module] = []
        prev = in_channels
        for ch in conv_channels:
            conv_layers.append(nn.Conv2d(prev, ch, kernel_size=kernel_size, padding=padding))
            conv_layers.append(nn.ReLU(inplace=True))
            prev = ch
        self.conv = nn.Sequential(*conv_layers)
        self._conv_out_features = prev * rows * cols

        self.lstm = nn.LSTM(
            input_size=self._conv_out_features,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=False,
        )

        self.fc_out = nn.Linear(lstm_hidden, n_actions)

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """obs_seq shape (seq, batch, in_channels * rows * cols). Retourne (q, hidden)."""
        seq_len, batch_size, _ = obs_seq.shape
        x = obs_seq.reshape(seq_len * batch_size, self.in_channels, self.rows, self.cols)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = x.reshape(seq_len, batch_size, self._conv_out_features)
        x, new_hidden = self.lstm(x, hidden)
        q = self.fc_out(x)
        return q, new_hidden
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_conv_recurrent.py -v 2>&1 | tail -15
```

Attendu : `5 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `236 passed` (231 + 5).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/neural/conv_recurrent.py tests/neural/test_conv_recurrent.py
git commit -m "feat(v2-zy): add ConvRecurrentQNetwork (Conv block + LSTM + FC, ~3.3M params)"
```

---

## Phase 3 — `RecurrentDQNTrainer` extension `double_dqn`

### Task 3 : Ajouter flag `double_dqn` + branche conditionnelle

**Files :**
- Modify : `mw_ia/neural/recurrent_trainer.py` (`RecurrentDQNTrainer.__init__` + `step()`)
- Test : `tests/neural/test_recurrent_trainer.py` (existant V2-Y, +1 test)

- [ ] **Step 1 — Write the 1 failing test**

Ajouter à la fin de `tests/neural/test_recurrent_trainer.py` :

```python
def test_double_dqn_branch_differs_from_standard(cpu_device: torch.device) -> None:
    """V2-ZY/V2-W : avec online ≠ target, les formules DQN et Double DQN divergent."""
    from mw_ia.neural.recurrent import RecurrentQNetwork

    online = RecurrentQNetwork(input_dim=300, n_actions=4, fc_hidden=64, lstm_hidden=32).to(cpu_device)
    target = RecurrentQNetwork(input_dim=300, n_actions=4, fc_hidden=64, lstm_hidden=32).to(cpu_device)
    target.load_state_dict(online.state_dict())
    with torch.no_grad():
        for p in online.parameters():
            p.add_(0.5)

    torch.manual_seed(42)
    next_states = torch.randn(8, 4, 300, device=cpu_device)

    with torch.no_grad():
        q_target_all, _ = target(next_states, None)
        q_next_dqn = q_target_all.max(dim=-1).values
        q_online_all, _ = online(next_states, None)
        next_actions = q_online_all.argmax(dim=-1)
        q_next_double = q_target_all.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)

    assert not torch.allclose(q_next_dqn, q_next_double), (
        "Double DQN doit différer de DQN classique quand online ≠ target"
    )
```

- [ ] **Step 2 — Run test, verify it passes already (math invariant)**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_recurrent_trainer.py::test_double_dqn_branch_differs_from_standard -v 2>&1 | tail -10
```

Attendu : `1 passed`. Ce test vérifie la propriété mathématique des 2 formules, anti-régression.

- [ ] **Step 3 — Modify `RecurrentDQNTrainer.__init__` to accept `double_dqn` parameter**

Dans `mw_ia/neural/recurrent_trainer.py`, modifier la signature de `__init__`. Remplacer :

```python
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
```

par :

```python
    def __init__(
        self,
        online: RecurrentQNetwork,
        target: RecurrentQNetwork,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cuda",
        use_amp: bool = True,
        double_dqn: bool = False,
    ) -> None:
```

Et dans le corps de `__init__`, ajouter `self.double_dqn = double_dqn` juste après `self.gamma = gamma`.

- [ ] **Step 4 — Modify `RecurrentDQNTrainer.step()` to use the flag**

Dans `step()`, localiser le bloc :

```python
            with torch.no_grad():
                q_next_all, _ = self.target(next_states, None)
                q_next = q_next_all.max(dim=-1).values
                target_q = rewards + self.gamma * q_next * (1.0 - dones)
```

Remplacer par :

```python
            with torch.no_grad():
                if self.double_dqn:
                    # V2-W branche appliquée au BPTT recurrent :
                    # online sélectionne, target évalue (Hasselt 2015)
                    q_online_next_all, _ = self.online(next_states, None)
                    next_actions = q_online_next_all.argmax(dim=-1)
                    q_target_all, _ = self.target(next_states, None)
                    q_next = q_target_all.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
                else:
                    # V2-Y baseline DQN classique
                    q_next_all, _ = self.target(next_states, None)
                    q_next = q_next_all.max(dim=-1).values
                target_q = rewards + self.gamma * q_next * (1.0 - dones)
```

- [ ] **Step 5 — Relax type annotation on online/target params**

Le `RecurrentDQNTrainer` doit accepter `ConvRecurrentQNetwork` aussi (V2-ZY). En haut de `mw_ia/neural/recurrent_trainer.py`, remplacer :

```python
from mw_ia.neural.recurrent import RecurrentQNetwork
from mw_ia.neural.sequence_buffer import BatchSeq
```

par :

```python
from typing import TYPE_CHECKING

from torch import nn

from mw_ia.neural.sequence_buffer import BatchSeq

if TYPE_CHECKING:
    from mw_ia.neural.recurrent import RecurrentQNetwork
```

Et dans la signature `__init__`, remplacer les annotations :

```python
        online: RecurrentQNetwork,
        target: RecurrentQNetwork,
```

par :

```python
        online: nn.Module,  # RecurrentQNetwork ou ConvRecurrentQNetwork
        target: nn.Module,
```

Duck-typing sur le contrat `forward(obs_seq, hidden) → (q, hidden)`.

- [ ] **Step 6 — Run V2-Y existing tests to verify zero regression**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_recurrent_trainer.py tests/agents/test_recurrent_dqn.py tests/training/test_recurrent_procedural_runner.py -v 2>&1 | tail -10
```

Attendu : tous V2-Y tests existants passent + le nouveau test V2-ZY. Default `double_dqn=False` préserve V2-Y baseline.

- [ ] **Step 7 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `237 passed` (236 + 1).

- [ ] **Step 8 — Commit**

```bash
git add mw_ia/neural/recurrent_trainer.py tests/neural/test_recurrent_trainer.py
git commit -m "feat(v2-zy): extend RecurrentDQNTrainer with double_dqn flag (default False, V2-Y compat)"
```

---

## Phase 4 — `ConvRecurrentDQNConfig`

### Task 4 : Frozen dataclass combo + validation + Aether compat

**Files :**
- Modify : `mw_ia/config.py` (ajout `ConvRecurrentDQNConfig` en fin de fichier)
- Test : `tests/test_conv_recurrent_dqn_config.py`

- [ ] **Step 1 — Write the 5 failing tests**

Contenu de `tests/test_conv_recurrent_dqn_config.py` :

```python
"""Tests V2-ZY de ConvRecurrentDQNConfig (combo CNN + LSTM + Double DQN + V2-V)."""
from __future__ import annotations

import pytest

from mw_ia.config import ConvRecurrentDQNConfig
from mw_ia.guardrails import VariantSpec, verify_formal


def test_defaults() -> None:
    """V2-ZY defaults : combo Conv + LSTM + Double DQN + V2-V activé."""
    cfg = ConvRecurrentDQNConfig()
    assert cfg.conv_channels == (32, 64)
    assert cfg.kernel_size == 3
    assert cfg.padding == 1
    assert cfg.lstm_hidden == 128
    assert cfg.sequence_length == 32
    assert cfg.double_dqn is True
    assert cfg.eval_enabled is True
    assert cfg.eval_every_episodes == 100
    assert cfg.eval_seeds == tuple(range(10_000, 10_010))
    assert cfg.eval_target_difficulty == 0.30
    assert cfg.best_checkpoint_path is None


def test_validation_conv_channels_positive() -> None:
    """conv_channels doivent être > 0 et non-vide."""
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(conv_channels=())
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(conv_channels=(0, 64))
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(conv_channels=(-1, 64))


def test_validation_lstm_hidden_positive() -> None:
    """lstm_hidden doit être > 0."""
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(lstm_hidden=0)
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(lstm_hidden=-1)


def test_validation_eval_target_difficulty_bounds() -> None:
    """eval_target_difficulty dans [0, 1]."""
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(eval_target_difficulty=-0.1)
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(eval_target_difficulty=1.1)
    cfg = ConvRecurrentDQNConfig(eval_target_difficulty=0.50)
    assert cfg.eval_target_difficulty == 0.50


def test_aether_compat() -> None:
    """VariantSpec dérivé du ConvRecurrentDQNConfig passe les invariants Aether I1-I8."""
    cfg = ConvRecurrentDQNConfig()
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
source .venv/Scripts/activate && pytest tests/test_conv_recurrent_dqn_config.py -v 2>&1 | tail -10
```

Attendu : 5 fails (`ImportError` ou `AttributeError`).

- [ ] **Step 3 — Add `ConvRecurrentDQNConfig` à la fin de `mw_ia/config.py`**

Ajouter à la fin du fichier `mw_ia/config.py` :

```python
@dataclass(frozen=True)
class ConvRecurrentDQNConfig:
    """V2-ZY : Conv2D + LSTM + Double DQN combiné, avec V2-V eval activé.

    Combo des 3 leviers livrés V2-Z (perception spatiale), V2-Y (mémoire),
    V2-W (Double DQN). Réseau ConvRecurrentQNetwork, buffer SequenceReplayBuffer
    V2-Y, trainer RecurrentDQNTrainer V2-Y étendu avec flag double_dqn.

    Champs combinés V2-Y DRQNConfig + V2-Z ConvDQNConfig + V2-W double_dqn + V2-V eval_*.
    """

    # Conv-spécifique (V2-Z pattern)
    conv_channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    padding: int = 1

    # LSTM (V2-Y pattern)
    lstm_hidden: int = 128
    sequence_length: int = 32

    # Replay (TRAJECTOIRES, pas transitions — V2-Y pattern)
    replay_capacity: int = 5_000
    min_episodes_to_learn: int = 100
    train_steps_per_episode: int = 4

    # Optimisation
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    target_sync_steps: int = 1_000
    use_amp: bool = True

    # V2-W : Double DQN activé par défaut V2-ZY (combo des 3 leviers)
    double_dqn: bool = True

    # V2-V : Training Protocol Stabilization (activé par défaut V2-ZY)
    eval_enabled: bool = True
    eval_every_episodes: int = 100
    eval_seeds: tuple[int, ...] = tuple(range(10_000, 10_010))
    eval_max_steps: int = 200
    eval_target_difficulty: float = 0.30
    best_checkpoint_path: str | None = None

    # Training
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
        if self.lstm_hidden <= 0:
            raise ValueError(f"lstm_hidden doit être > 0, reçu {self.lstm_hidden}")
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
        if self.eval_every_episodes <= 0:
            raise ValueError(
                f"eval_every_episodes doit être > 0, reçu {self.eval_every_episodes}"
            )
        if len(self.eval_seeds) == 0:
            raise ValueError("eval_seeds ne peut pas être vide")
        if self.eval_max_steps <= 0:
            raise ValueError(f"eval_max_steps doit être > 0, reçu {self.eval_max_steps}")
        if not (0.0 <= self.eval_target_difficulty <= 1.0):
            raise ValueError(
                f"eval_target_difficulty doit être ∈ [0,1], reçu {self.eval_target_difficulty}"
            )
        if self.episodes <= 0:
            raise ValueError(f"episodes doit être > 0, reçu {self.episodes}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(
                f"max_steps_per_episode doit être > 0, reçu {self.max_steps_per_episode}"
            )
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/test_conv_recurrent_dqn_config.py -v 2>&1 | tail -10
```

Attendu : `5 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `242 passed` (237 + 5).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/config.py tests/test_conv_recurrent_dqn_config.py
git commit -m "feat(v2-zy): add ConvRecurrentDQNConfig (combo V2-Z + V2-Y + V2-W + V2-V defaults)"
```

---

## Phase 5 — `ConvRecurrentDQNAgent`

### Task 5 : Agent combiné (act + observe + end_episode + reset_hidden + begin_episode)

**Files :**
- Modify : `mw_ia/agents/conv_recurrent_dqn.py` (impl)
- Test : `tests/agents/test_conv_recurrent_dqn.py`

- [ ] **Step 1 — Write the 7 failing tests**

Contenu de `tests/agents/test_conv_recurrent_dqn.py` :

```python
"""Tests V2-ZY de ConvRecurrentDQNAgent (combo CNN + LSTM + Double DQN)."""
from __future__ import annotations

import numpy as np
import torch

from mw_ia.agents.conv_recurrent_dqn import ConvRecurrentDQNAgent
from mw_ia.config import ConvRecurrentDQNConfig
from mw_ia.guardrails import VariantSpec, verify_formal


def _make_agent(cfg: ConvRecurrentDQNConfig | None = None, seed: int = 0) -> ConvRecurrentDQNAgent:
    cfg = cfg or ConvRecurrentDQNConfig(
        min_episodes_to_learn=2, batch_size=2, train_steps_per_episode=1,
        sequence_length=4, max_steps_per_episode=8,
        replay_capacity=10, use_amp=False,
    )
    return ConvRecurrentDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=seed,
    )


def _obs() -> np.ndarray:
    return np.zeros((3, 10, 10), dtype=np.float32)


def test_init() -> None:
    """Online + target nets, sequence buffer empty, global_step=0, hidden None."""
    agent = _make_agent()
    assert agent.global_step == 0
    assert len(agent.buffer) == 0
    assert agent._hidden_state is None
    for p_o, p_t in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.allclose(p_o, p_t)


def test_reset_hidden() -> None:
    """reset_hidden() remet _hidden_state à None."""
    agent = _make_agent()
    agent.act(_obs())
    assert agent._hidden_state is not None
    agent.reset_hidden()
    assert agent._hidden_state is None


def test_begin_episode_resets_hidden_and_starts_trajectory() -> None:
    """begin_episode() reset hidden + vide la trajectoire en cours."""
    agent = _make_agent()
    agent.act(_obs())
    agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    assert len(agent._episode_trajectory) == 1
    agent.begin_episode()
    assert agent._hidden_state is None
    assert len(agent._episode_trajectory) == 0


def test_act_maintains_hidden_state_even_in_exploration() -> None:
    """eps=1.0 → action random MAIS hidden state updated par forward LSTM."""
    cfg = ConvRecurrentDQNConfig(epsilon_start=1.0, epsilon_end=1.0)
    agent = ConvRecurrentDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=42,
    )
    assert agent._hidden_state is None
    agent.act(_obs())
    assert agent._hidden_state is not None


def test_act_greedy_deterministic() -> None:
    """Eps=0 + même hidden + même obs → mêmes actions."""
    cfg = ConvRecurrentDQNConfig(epsilon_start=0.0, epsilon_end=0.0)
    agent = ConvRecurrentDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=42,
    )
    obs = _obs()
    agent.reset_hidden()
    a1 = agent.act(obs)
    agent.reset_hidden()
    a2 = agent.act(obs)
    assert a1 == a2


def test_end_episode_trains_when_buffer_full() -> None:
    """end_episode() train_steps quand buffer >= max(min_episodes_to_learn, batch_size)."""
    cfg = ConvRecurrentDQNConfig(
        min_episodes_to_learn=2, batch_size=2, train_steps_per_episode=1,
        sequence_length=4, max_steps_per_episode=8,
        replay_capacity=10, use_amp=False,
    )
    agent = ConvRecurrentDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    agent.begin_episode()
    for _ in range(4):
        agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    m1 = agent.end_episode()
    assert "loss" not in m1
    agent.begin_episode()
    for _ in range(4):
        agent.observe(_obs(), action=0, reward=0.0, next_state=_obs(), done=False)
    m2 = agent.end_episode()
    assert "loss" in m2
    assert np.isfinite(m2["loss"])


def test_aether_smoke() -> None:
    """Smoke E2E : VariantSpec dérivé d'un agent V2-ZY passe Aether I1-I8."""
    cfg = ConvRecurrentDQNConfig()
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
source .venv/Scripts/activate && pytest tests/agents/test_conv_recurrent_dqn.py -v 2>&1 | tail -10
```

Attendu : `ImportError` (`ConvRecurrentDQNAgent` n'existe pas).

- [ ] **Step 3 — Implement `ConvRecurrentDQNAgent`**

Remplacer le contenu de `mw_ia/agents/conv_recurrent_dqn.py` par :

```python
"""ConvRecurrentDQNAgent — V2-ZY combo CNN + LSTM + Double DQN.

Hidden state runtime maintenu entre act() consécutifs (pattern V2-Y).
Forward LSTM appliqué AUSSI en eps-greedy random.

Observations 3D `(in_channels, rows, cols)` flatten 1D pour storage dans
SequenceReplayBuffer V2-Y. Network reshape interne 1D → 3D pour Conv block.

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-cnn-lstm-double-dqn-design.md §2
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mw_ia.agents.base import Agent
from mw_ia.config import ConvRecurrentDQNConfig
from mw_ia.neural.conv_recurrent import ConvRecurrentQNetwork
from mw_ia.neural.recurrent_trainer import RecurrentDQNTrainer
from mw_ia.neural.sequence_buffer import SequenceReplayBuffer


class ConvRecurrentDQNAgent(Agent):
    """Agent V2-ZY combinant perception spatiale (Conv) + mémoire (LSTM) + Double DQN."""

    def __init__(
        self,
        *,
        in_channels: int,
        rows: int,
        cols: int,
        n_actions: int,
        cfg: ConvRecurrentDQNConfig,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        self.n_actions = n_actions
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        wants_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if wants_cuda else "cpu")
        self.online = ConvRecurrentQNetwork(
            in_channels=in_channels, rows=rows, cols=cols, n_actions=n_actions,
            conv_channels=cfg.conv_channels, kernel_size=cfg.kernel_size,
            padding=cfg.padding, lstm_hidden=cfg.lstm_hidden,
        ).to(self.device)
        self.target = ConvRecurrentQNetwork(
            in_channels=in_channels, rows=rows, cols=cols, n_actions=n_actions,
            conv_channels=cfg.conv_channels, kernel_size=cfg.kernel_size,
            padding=cfg.padding, lstm_hidden=cfg.lstm_hidden,
        ).to(self.device)
        self.trainer = RecurrentDQNTrainer(
            self.online, self.target,
            lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
            double_dqn=cfg.double_dqn,
        )
        obs_dim_flat = in_channels * rows * cols
        self.buffer = SequenceReplayBuffer(
            cfg.replay_capacity, obs_dim_flat, cfg.max_steps_per_episode, seed=seed,
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
        """Reset le hidden state LSTM. Appelé par le runner au début de chaque épisode."""
        self._hidden_state = None

    def begin_episode(self) -> None:
        """Reset hidden + vide la trajectoire courante. Compat V2-V duck-typing."""
        self._hidden_state = None
        self._episode_trajectory = []

    def act(self, state: np.ndarray, *, greedy: bool = False) -> int:
        """Forward LSTM toujours appliqué (maintient hidden state runtime).

        Pattern V2-Y : la mémoire LSTM suit la trajectoire complète indépendamment
        des choix d'action (eps-greedy random ne skip pas le forward).
        """
        assert state.shape == (self.in_channels, self.rows, self.cols), (
            f"state {state.shape} != ({self.in_channels}, {self.rows}, {self.cols})"
        )
        with torch.no_grad():
            x = torch.from_numpy(state.flatten()).float().to(self.device)
            x = x.unsqueeze(0).unsqueeze(0)
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
        assert state.shape == (self.in_channels, self.rows, self.cols)
        assert next_state.shape == (self.in_channels, self.rows, self.cols)
        self._episode_trajectory.append(
            (state.flatten(), action, reward, next_state.flatten(), done)
        )
        self.global_step += 1
        return {"epsilon": self.epsilon}

    def end_episode(self) -> dict[str, float]:
        """Push trajectoire dans buffer + train_steps_per_episode batches BPTT."""
        if self._episode_trajectory:
            self.buffer.push_trajectory(self._episode_trajectory)
        metrics: dict[str, float] = {"epsilon": self.epsilon}
        train_threshold = max(self.cfg.min_episodes_to_learn, self.cfg.batch_size)
        if len(self.buffer) >= train_threshold:
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

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_conv_recurrent_dqn.py -v 2>&1 | tail -15
```

Attendu : `7 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `249 passed` (242 + 7).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/agents/conv_recurrent_dqn.py tests/agents/test_conv_recurrent_dqn.py
git commit -m "feat(v2-zy): add ConvRecurrentDQNAgent (combo CNN + LSTM + Double DQN, V2-Y pattern)"
```

---

## Phase 6 — V2-V `PeriodicEvaluator` extension `begin_episode` duck-typing

### Task 6 : Reset hidden state entre rollouts eval (V2-V compat V2-ZY)

**Files :**
- Modify : `mw_ia/training/evaluator.py` (`PeriodicEvaluator.evaluate()`)
- Test : `tests/training/test_evaluator.py` (existant V2-V, +1 test)

- [ ] **Step 1 — Write the 1 failing test**

Ajouter à la fin de `tests/training/test_evaluator.py` :

```python
def test_evaluator_calls_begin_episode_if_exists() -> None:
    """V2-ZY duck-typing : si agent expose begin_episode(), evaluator l'appelle.

    Pour les agents recurrent (LSTM hidden state), reset entre rollouts eval
    est nécessaire pour empêcher la contamination du hidden state entre seeds eval.
    """
    evaluator = _build_evaluator(eval_seeds=(10_000, 10_001, 10_002))
    agent = _build_agent()

    call_count = [0]

    def tracked_begin_episode() -> None:
        call_count[0] += 1

    agent.begin_episode = tracked_begin_episode

    evaluator.evaluate(agent, difficulty=0.10)
    assert call_count[0] == 3
```

- [ ] **Step 2 — Run test, verify it fails**

```bash
source .venv/Scripts/activate && pytest tests/training/test_evaluator.py::test_evaluator_calls_begin_episode_if_exists -v 2>&1 | tail -10
```

Attendu : `assert call_count[0] == 3` échoue avec `0 == 3`.

- [ ] **Step 3 — Modify `PeriodicEvaluator.evaluate()` to duck-type `begin_episode`**

Dans `mw_ia/training/evaluator.py`, dans la méthode `evaluate()`, localiser la boucle `for seed in self.eval_seeds:`. Ajouter l'appel `begin_episode` AVANT le `reset(seed=seed)` :

Remplacer :

```python
        for seed in self.eval_seeds:
            state, info = self.eval_env.reset(seed=seed)
            maze = info["maze"]
```

par :

```python
        for seed in self.eval_seeds:
            # V2-ZY duck-typing : reset hidden state pour agents LSTM
            begin = getattr(agent, "begin_episode", None)
            if begin is not None:
                begin()
            state, info = self.eval_env.reset(seed=seed)
            maze = info["maze"]
```

- [ ] **Step 4 — Run test, verify it passes**

```bash
source .venv/Scripts/activate && pytest tests/training/test_evaluator.py::test_evaluator_calls_begin_episode_if_exists -v 2>&1 | tail -10
```

Attendu : `1 passed`.

- [ ] **Step 5 — Run V2-V existing tests to verify zero regression**

```bash
source .venv/Scripts/activate && pytest tests/training/test_evaluator.py -v 2>&1 | tail -15
```

Attendu : 9 passed (8 V2-V existants + 1 V2-ZY).

- [ ] **Step 6 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `250 passed` (249 + 1).

- [ ] **Step 7 — Commit**

```bash
git add mw_ia/training/evaluator.py tests/training/test_evaluator.py
git commit -m "feat(v2-zy): extend PeriodicEvaluator.evaluate() with begin_episode duck-typing"
```

---

## Phase 7 — `ConvRecurrentProceduralDQNRunner`

### Task 7 : Runner V2-ZY intégré avec V2-V eval+best-checkpoint

**Files :**
- Modify : `mw_ia/training/runner.py` (ajout `ConvRecurrentProceduralDQNRunner` en fin de fichier)
- Test : `tests/training/test_conv_recurrent_procedural_runner.py`

- [ ] **Step 1 — Write the 2 failing tests**

Contenu de `tests/training/test_conv_recurrent_procedural_runner.py` :

```python
"""Tests V2-ZY d'intégration de ConvRecurrentProceduralDQNRunner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mw_ia.config import (
    ConvRecurrentDQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ConvRecurrentProceduralDQNRunner, RunnerCallbacks


def _build_env() -> ProceduralGridWorld:
    cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                              min_density=0.0, max_density=0.20)
    gen = RandomObstaclesGenerator(
        rows=10, cols=10, start=(0, 0), goal=(9, 9),
        min_density=cfg.min_density, max_density=cfg.max_density,
    )
    return ProceduralGridWorld(cfg=cfg, generator=gen)


def test_runner_full_episode_with_eval_and_best_checkpoint(tmp_path: Path) -> None:
    """V2-ZY runner avec V2-V activé sauvegarde best-checkpoint."""
    env = _build_env()
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.0, max_density=0.20)
    best_path = tmp_path / "best.pt"
    dqn_cfg = ConvRecurrentDQNConfig(
        episodes=200, max_steps_per_episode=20,
        batch_size=2, min_episodes_to_learn=2, train_steps_per_episode=1,
        sequence_length=8, replay_capacity=20,
        epsilon_decay_steps=200, target_sync_steps=50,
        use_amp=False,
        eval_enabled=True,
        eval_every_episodes=50,
        eval_seeds=(10_000, 10_001, 10_002),
        eval_max_steps=10,
        eval_target_difficulty=0.10,
        best_checkpoint_path=str(best_path),
    )
    sched_cfg = SchedulerConfig(initial_difficulty=0.0)
    train_cfg = TrainingConfig(log_every_episodes=100)

    eval_count = [0]

    def evaluation_callback(**kw: Any) -> None:
        eval_count[0] += 1

    cb = RunnerCallbacks(on_eval=evaluation_callback)
    runner = ConvRecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=cb, device="cpu", seed=0,
    )
    runner.run()

    assert eval_count[0] >= 3, f"expected >=3 evals, got {eval_count[0]}"
    assert best_path.exists(), "best_checkpoint .pt manquant sur disque"
    assert runner.best_tracker is not None
    assert runner.best_tracker.best_winrate >= 0.0


def test_runner_eval_disabled_no_evaluator() -> None:
    """V2-ZY runner avec eval_enabled=False n'instancie pas evaluator."""
    env = _build_env()
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.0, max_density=0.20)
    dqn_cfg = ConvRecurrentDQNConfig(
        episodes=10, max_steps_per_episode=10,
        batch_size=2, min_episodes_to_learn=2, train_steps_per_episode=1,
        sequence_length=4, replay_capacity=10,
        epsilon_decay_steps=200, target_sync_steps=50,
        use_amp=False,
        eval_enabled=False,
    )
    sched_cfg = SchedulerConfig(initial_difficulty=0.0)
    train_cfg = TrainingConfig(log_every_episodes=100)

    runner = ConvRecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=RunnerCallbacks(), device="cpu", seed=0,
    )
    assert runner.evaluator is None
    assert runner.best_tracker is None
    runner.run()
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/training/test_conv_recurrent_procedural_runner.py -v 2>&1 | tail -15
```

Attendu : `ImportError` (`ConvRecurrentProceduralDQNRunner` n'existe pas).

- [ ] **Step 3 — Add imports + Implement `ConvRecurrentProceduralDQNRunner`**

Dans `mw_ia/training/runner.py`, étendre les imports en haut. Localiser la ligne d'imports config :

```python
from mw_ia.config import ConvDQNConfig, DQNConfig, DRQNConfig, ProceduralEnvConfig, QLearningConfig, SchedulerConfig, TrainingConfig
```

Remplacer par :

```python
from mw_ia.config import ConvDQNConfig, ConvRecurrentDQNConfig, DQNConfig, DRQNConfig, ProceduralEnvConfig, QLearningConfig, SchedulerConfig, TrainingConfig
```

Localiser la ligne d'import agent :

```python
from mw_ia.agents.conv_dqn import ConvDQNAgent
```

Ajouter après :

```python
from mw_ia.agents.conv_recurrent_dqn import ConvRecurrentDQNAgent
```

Ajouter à la fin de `mw_ia/training/runner.py` (après `ConvProceduralDQNRunner`) :

```python
class ConvRecurrentProceduralDQNRunner(_BaseRunner):
    """V2-ZY runner : Conv + LSTM + Double DQN + V2-V eval+best-checkpoint."""

    def __init__(
        self,
        *,
        env: ProceduralGridWorld,
        proc_cfg: ProceduralEnvConfig,
        dqn_cfg: ConvRecurrentDQNConfig,
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
        self.agent = ConvRecurrentDQNAgent(
            in_channels=3, rows=proc_cfg.max_rows, cols=proc_cfg.max_cols,
            n_actions=4, cfg=dqn_cfg, device=device, seed=seed,
        )

        if dqn_cfg.eval_enabled:
            eval_gen = type(env.generator).__new__(type(env.generator))
            eval_gen.__dict__.update(env.generator.__dict__)
            eval_env = ProceduralGridWorld(cfg=proc_cfg, generator=eval_gen)
            self.evaluator: PeriodicEvaluator | None = PeriodicEvaluator(
                eval_env=eval_env,
                eval_seeds=dqn_cfg.eval_seeds,
                max_steps=dqn_cfg.eval_max_steps,
                observation_encoder=encode_procedural_observation_2d,
                proc_cfg=proc_cfg,
            )
            self.best_tracker: BestCheckpointTracker | None = BestCheckpointTracker(
                path=dqn_cfg.best_checkpoint_path,
            )
        else:
            self.evaluator = None
            self.best_tracker = None

    def run(self) -> None:
        self.callbacks.fire_log(
            "info",
            f"V2-ZY Conv+LSTM+DoubleDQN ({self.proc_cfg.mode}) sur {self.agent.device}"
        )
        self.callbacks.fire_log(
            "info",
            f"Config: conv_channels={self.dqn_cfg.conv_channels} "
            f"lstm_hidden={self.dqn_cfg.lstm_hidden} "
            f"double_dqn={self.dqn_cfg.double_dqn} "
            f"eval_enabled={self.dqn_cfg.eval_enabled} "
            f"eval_target_difficulty={self.dqn_cfg.eval_target_difficulty}"
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

            self.agent.reset_hidden()
            self.agent.begin_episode()

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

            if (
                self.evaluator is not None
                and (ep + 1) % self.dqn_cfg.eval_every_episodes == 0
            ):
                eval_metrics = self.evaluator.evaluate(
                    self.agent, self.dqn_cfg.eval_target_difficulty,
                )
                improved = self.best_tracker.update(eval_metrics, self.agent, episode=ep)
                self.callbacks.fire_evaluation(
                    ep=ep,
                    eval_winrate=eval_metrics["winrate"],
                    eval_diff=eval_metrics["difficulty"],
                    best_winrate=self.best_tracker.best_winrate,
                    best_episode=self.best_tracker.best_episode,
                    improved=improved,
                )
                self.callbacks.fire_log(
                    "info",
                    f"eval ep {ep:>4} : winrate={eval_metrics['winrate']:.2%} "
                    f"@ diff={eval_metrics['difficulty']:.2f}  "
                    f"best={self.best_tracker.best_winrate:.2%} "
                    f"@ ep {self.best_tracker.best_episode}"
                    + ("  NEW BEST" if improved else "")
                )
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/training/test_conv_recurrent_procedural_runner.py -v 2>&1 | tail -10
```

Attendu : `2 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `252 passed` (250 + 2).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/training/runner.py tests/training/test_conv_recurrent_procedural_runner.py
git commit -m "feat(v2-zy): add ConvRecurrentProceduralDQNRunner (combo + V2-V integrated)"
```

---

## Phase 8 — CLI + CI smoke

### Task 8 : Script CLI V2-ZY

**Files :**
- Create : `scripts/train_cnn_lstm_dqn_procedural.py`

- [ ] **Step 1 — Create CLI script**

Créer `scripts/train_cnn_lstm_dqn_procedural.py` :

```python
"""Entraînement V2-ZY headless (CLI) : Conv + LSTM + Double DQN combiné.

Usage :
    python scripts/train_cnn_lstm_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda
"""
from __future__ import annotations

import argparse
import sys

from mw_ia.config import (
    ConvRecurrentDQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import PerfectMazeGenerator, RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ConvRecurrentProceduralDQNRunner, RunnerCallbacks


def _print_log(level: str, msg: str) -> None:
    print(f"[{level:7s}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-ZY CNN+LSTM+Double DQN combiné")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--mode", choices=("obstacles", "maze"), default="obstacles")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conv-channels", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--epsilon-decay-steps", type=int, default=200_000)
    parser.add_argument("--scheduler-update-interval", type=int, default=50)
    parser.add_argument("--scheduler-step", type=float, default=0.05)
    parser.add_argument(
        "--double-dqn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Double DQN (V2-W). Default activé pour V2-ZY.",
    )
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Évaluation périodique greedy (V2-V). Default activé.",
    )
    parser.add_argument("--eval-every-episodes", type=int, default=100)
    parser.add_argument("--eval-target-difficulty", type=float, default=0.30)
    parser.add_argument("--best-checkpoint-path", type=str, default=None)
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
    dqn_cfg = ConvRecurrentDQNConfig(
        episodes=args.episodes,
        conv_channels=tuple(args.conv_channels),
        lstm_hidden=args.lstm_hidden,
        sequence_length=args.sequence_length,
        epsilon_decay_steps=args.epsilon_decay_steps,
        double_dqn=args.double_dqn,
        eval_enabled=args.eval,
        eval_every_episodes=args.eval_every_episodes,
        eval_target_difficulty=args.eval_target_difficulty,
        best_checkpoint_path=args.best_checkpoint_path,
    )
    sched_cfg = SchedulerConfig(
        update_interval=args.scheduler_update_interval,
        step=args.scheduler_step,
    )
    train_cfg = TrainingConfig()

    cb = RunnerCallbacks(on_log=_print_log)
    runner = ConvRecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=cb, device=args.device, seed=args.seed,
    )
    runner.run()

    final_wr = runner.metrics.winrate()
    final_diff = runner.scheduler.current
    print(f"\nFinal : winrate={final_wr:.2%}, difficulty={final_diff:.2f}")
    if runner.best_tracker is not None:
        print(
            f"Best @ diff={dqn_cfg.eval_target_difficulty:.2f} : "
            f"winrate={runner.best_tracker.best_winrate:.2%} "
            f"@ ep {runner.best_tracker.best_episode}"
        )
    print("Per-bucket winrate :")
    for i, wr in enumerate(runner.bucket_tracker.winrate_per_bucket()):
        wr_str = f"{wr:.2%}" if wr is not None else "-"
        print(f"  bucket {i} ({i*0.2:.1f}-{(i+1)*0.2:.1f}) : {wr_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2 — Smoke test CPU 10 ép**

```bash
mkdir -p checkpoints && \
source .venv/Scripts/activate && \
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles --device cpu --eval-every-episodes 5 --best-checkpoint-path checkpoints/v2zy_smoke.pt 2>&1 | tail -15
```

Attendu : pas de crash, ≥ 1 ligne `eval ep`, fichier `.pt` créé.

Vérification :

```bash
ls -la checkpoints/v2zy_smoke.pt && rm checkpoints/v2zy_smoke.pt
```

- [ ] **Step 3 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `252 passed`.

- [ ] **Step 4 — Commit**

```bash
git add scripts/train_cnn_lstm_dqn_procedural.py
git commit -m "feat(v2-zy): add scripts/train_cnn_lstm_dqn_procedural.py CLI"
```

---

### Task 9 : CI workflow smoke V2-ZY

**Files :**
- Modify : `.github/workflows/aether_verify.yml`

- [ ] **Step 1 — Add V2-ZY smoke step**

Localiser le step "Smoke test V2-V eval + best-checkpoint" dans `.github/workflows/aether_verify.yml`. Ajouter juste après (pattern identique) :

```yaml
      - name: Smoke test V2-ZY CNN + LSTM + Double DQN
        run: |
          mkdir -p checkpoints
          python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles --device cpu --eval-every-episodes 5 --best-checkpoint-path checkpoints/ci_v2zy_best.pt
          test -f checkpoints/ci_v2zy_best.pt
```

- [ ] **Step 2 — Validate YAML syntax**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/aether_verify.yml'))" && echo "YAML OK"
```

Attendu : `YAML OK`.

- [ ] **Step 3 — Smoke local CPU 10 ép pour valider l'invocation CI**

```bash
mkdir -p checkpoints && \
source .venv/Scripts/activate && \
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles --device cpu --eval-every-episodes 5 --best-checkpoint-path checkpoints/ci_v2zy_test.pt && \
test -f checkpoints/ci_v2zy_test.pt && echo "FILE OK" && rm checkpoints/ci_v2zy_test.pt
```

Attendu : `FILE OK`.

- [ ] **Step 4 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `252 passed`.

- [ ] **Step 5 — Commit**

```bash
git add .github/workflows/aether_verify.yml
git commit -m "ci(v2-zy): add smoke test V2-ZY CNN + LSTM + Double DQN to aether_verify.yml"
```

---

## Phase 9 — README + CLAUDE.md + smoke E2E GPU + tag

### Task 10 : Documentation V2-ZY + smoke GPU + tag `v0.2.0-zy`

**Files :**
- Modify : `README.md`
- Modify : `CLAUDE.md`

- [ ] **Step 1 — Smoke E2E manuel GPU 500 ép V2-ZY**

```bash
mkdir -p checkpoints && \
source .venv/Scripts/activate && \
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 500 --mode obstacles --device cuda \
    --eval-every-episodes 100 \
    --best-checkpoint-path checkpoints/v2zy_smoke_gpu.pt 2>&1 | tail -15
```

Attendu : pas de crash sur 500 ép GPU, ≥ 5 lignes `eval ep`, fichier `.pt` créé.

Vérification :

```bash
ls -la checkpoints/v2zy_smoke_gpu.pt && rm checkpoints/v2zy_smoke_gpu.pt
```

- [ ] **Step 2 — Aether re-verify**

```bash
bash aether/verify_all.sh
```

Attendu : `8 OK`.

- [ ] **Step 3 — Add V2-ZY section to README.md**

Localiser la section `## V2-V` dans `README.md`. Insérer une nouvelle section V2-ZY juste après la fin de V2-V et avant `## Roadmap (V2+)`.

Contenu exact à insérer :

````markdown
## V2-ZY — CNN + LSTM + Double DQN combiné (sous-projet livré)

**Tag** : `v0.2.0-zy` — **Tests** : 252 verts (231 baseline + 21 V2-ZY)

Combo des 3 leviers livrés : perception spatiale (V2-Z Conv2D), mémoire temporelle (V2-Y LSTM), stabilité Q-target (V2-W Double DQN). Validé via V2-V eval rigoureux.

### Usage CLI

```bash
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda \
    --best-checkpoint-path checkpoints/v2zy_best_seed0.pt
```

### Architecture

- `mw_ia/neural/conv_recurrent.py::ConvRecurrentQNetwork` — `Conv → Flatten → LSTM → FC` (Hausknecht-style)
- `mw_ia/agents/conv_recurrent_dqn.py::ConvRecurrentDQNAgent` — hidden state runtime maintenu, pattern V2-Y
- `mw_ia/neural/recurrent_trainer.py::RecurrentDQNTrainer` (V2-Y) étendu avec flag `double_dqn`
- `mw_ia/training/runner.py::ConvRecurrentProceduralDQNRunner` — intègre V2-V eval+best-checkpoint
- `mw_ia/training/evaluator.py::PeriodicEvaluator` (V2-V) étendu avec duck-typing `agent.begin_episode()` pour reset hidden state entre rollouts eval

### Critère succès cible

4/5 seeds avec `best_checkpoint @ diff=0.30 ≥ 70 %` en eval rigoureux V2-V.
````

- [ ] **Step 4 — Add V2-ZY row to CLAUDE.md sub-projects table**

Dans `CLAUDE.md`, localiser le tableau "Sous-projets — décomposition". Ajouter une ligne ZY après la ligne V. Remplacer :

```markdown
| **V** | Training Protocol Stabilization (eval + best-checkpoint) | ✅ Livré (tag `v0.2.0-v`) |
| **B** | Mémoire persistante cross-session | ⏳ **Prochain par défaut** |
```

par :

```markdown
| **V** | Training Protocol Stabilization (eval + best-checkpoint) | ✅ Livré (tag `v0.2.0-v`) |
| **ZY** | CNN + LSTM + Double DQN combiné | ✅ Livré (tag `v0.2.0-zy`) |
| **B** | Mémoire persistante cross-session | ⏳ **Prochain par défaut** |
```

- [ ] **Step 5 — Add V2-ZY phases section to CLAUDE.md**

Dans `CLAUDE.md`, localiser une bonne position pour insérer (typiquement après la section "V2-V — benchmark complémentaire @ diff=0.20"). Insérer la section V2-ZY :

```markdown
### V2-ZY — état final des phases (livraison 2026-05-23)

| Phase | Tâches | Statut | Tests | Commits |
|---|---|---|---|---|
| 1 — Setup scaffold | T1 | ✅ | 0 | 1 |
| 2 — `ConvRecurrentQNetwork` | T2 | ✅ | 5 | 1 |
| 3 — `RecurrentDQNTrainer` extension `double_dqn` | T3 | ✅ | 1 | 1 |
| 4 — `ConvRecurrentDQNConfig` | T4 | ✅ | 5 | 1 |
| 5 — `ConvRecurrentDQNAgent` | T5 | ✅ | 7 | 1 |
| 6 — V2-V `PeriodicEvaluator` extension `begin_episode` duck-typing | T6 | ✅ | 1 | 1 |
| 7 — `ConvRecurrentProceduralDQNRunner` | T7 | ✅ | 2 | 1 |
| 8 — CLI + CI smoke | T8-T9 | ✅ | 0 | 2 |
| 9 — README + CLAUDE.md + tag `v0.2.0-zy` | T10 | ✅ | 0 | 1 + tag |

### Composants V2-ZY livrés

| Composant | Fichier | Rôle |
|---|---|---|
| `ConvRecurrentQNetwork` | `mw_ia/neural/conv_recurrent.py` | `Conv → Flatten → LSTM → FC`. Accepte obs flat 1D (compat SequenceReplayBuffer V2-Y), reshape interne 3D pour Conv. ~3.3 M params. |
| `ConvRecurrentDQNAgent` | `mw_ia/agents/conv_recurrent_dqn.py` | Combo CNN + LSTM + Double DQN. Pattern V2-Y : hidden state runtime, reset_hidden/begin_episode par épisode, train à end_episode. |
| `ConvRecurrentDQNConfig` | `mw_ia/config.py` | Combo défauts V2-Z + V2-Y + V2-W + V2-V activés. |
| `RecurrentDQNTrainer` extension | `mw_ia/neural/recurrent_trainer.py` | + kwarg `double_dqn: bool = False` (default préserve V2-Y baseline). Branche conditionnelle dans step(). |
| `PeriodicEvaluator` extension | `mw_ia/training/evaluator.py` | Duck-typing : appelle `agent.begin_episode()` au début de chaque rollout eval si la méthode existe. |
| `ConvRecurrentProceduralDQNRunner` | `mw_ia/training/runner.py` | Runner V2-ZY intégré avec V2-V eval+best-checkpoint dès l'origine. |
| CLI | `scripts/train_cnn_lstm_dqn_procedural.py` | Flags combinés V2-Z + V2-Y + V2-W + V2-V. |

### Décisions techniques V2-ZY

- **Architecture `Conv → Flatten → LSTM → FC`** (Hausknecht-style) : conv extrait features par frame, LSTM intègre temporellement, FC produit Q-values.
- **Réutilisation `SequenceReplayBuffer` V2-Y** : obs flatten 1D pour storage, network reshape interne 1D → 3D pour conv. Zéro duplication.
- **Réutilisation `RecurrentDQNTrainer` V2-Y** : étendu avec flag `double_dqn`. Default `False` préserve V2-Y baseline livré.
- **Hidden state forward maintenu même en eps-random** : pattern V2-Y. La mémoire LSTM doit suivre la trajectoire indépendamment des choix d'action.
- **V2-V `begin_episode` duck-typing** : `getattr(agent, 'begin_episode', None)` permet à V2-V de fonctionner avec V2-Z (no-op) ET V2-ZY (reset hidden).

### V2-ZY — pièges connus

1. **`SequenceReplayBuffer` stocke obs en flat** : agent flatten 3D→1D avant push, network reshape interne au forward.
2. **LSTM forward avec `hidden=None` au début de chaque séquence training** : DRQN simple (V2-Y), pas de burn-in R2D2 (hors-scope MVP).
3. **V2-Y trainer modification** : `double_dqn=False` par défaut préserve les 35 tests V2-Y existants.
4. **V2-V duck-typing** : ConvDQNAgent V2-Z n'a pas `begin_episode`, donc no-op pour V2-Z. ConvRecurrentDQNAgent V2-ZY a `begin_episode`, hidden reset entre seeds eval.
5. **Replay buffer 2.4 GB** : trajectoires complètes V2-Y pattern. Si OOM, descendre `replay_capacity`.
```

- [ ] **Step 6 — Update "Prochaines étapes prioritaires" section dans CLAUDE.md**

Localiser la section "Prochaines étapes prioritaires" et mettre à jour. Remplacer le contenu par :

```markdown
**Prochaines étapes prioritaires (post V2-ZY livré 2026-05-23)** :

1. ✅ **V2-ZY CNN + LSTM + Double DQN combiné** — **LIVRÉ** (tag `v0.2.0-zy`) : combo des 3 leviers + V2-V eval.

2. **Benchmark V2-ZY n=5 ep=5000 same-seed** — validation scientifique non-bloquante :
   - Lancer 5 runs V2-ZY ep=5000 avec `--eval-target-difficulty 0.30 --best-checkpoint-path checkpoints/v2zy_best_seed{N}.pt`
   - Comparer best @ diff=0.30 vs V2-W n=5 (mean 58 %, 2/5 ≥ 70 %)
   - **Critère succès** : 4/5 seeds avec best ≥ 70 % @ diff=0.30
   - Si critère atteint → benchmark bonus @ diff=0.40

3. **Sous-projets V3+ déblocables** :
   - **Soft target Polyak τ=0.005** : élimine le résidu de collapse tardif
   - **R2D2 burn-in** : remplace DRQN simple par burn-in pour stabilité LSTM
   - **Mazes larges (max_size=15/20)** : test translation equivariance CNN
   - **Sous-projet B (mémoire persistante cross-session)**
```

- [ ] **Step 7 — Run full suite + Aether final**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
bash aether/verify_all.sh
```

Attendu : `252 passed` + `8 OK`.

- [ ] **Step 8 — Commit doc**

```bash
git add README.md CLAUDE.md
git commit -m "docs(v2-zy): add V2-ZY section (CNN + LSTM + Double DQN combiné) to README + CLAUDE.md"
```

- [ ] **Step 9 — Tag**

```bash
git tag v0.2.0-zy
git tag --list | tail -8
```

Attendu : `v0.1.0`, `v0.2.0-a`, `v0.2.0-x`, `v0.2.0-y`, `v0.2.0-z`, `v0.2.0-w`, `v0.2.0-v`, `v0.2.0-zy`.

- [ ] **Step 10 — DoD final récap**

Print to stdout :

```
=== V2-ZY DoD CHECKLIST (livraison code) ===
[ ] pytest -q → 252 passed (231 + 21)
[ ] bash aether/verify_all.sh → 8 OK
[ ] smoke train_cnn_lstm_dqn_procedural.py --episodes 500 --device cuda OK
[ ] best_checkpoint .pt présent sur disque
[ ] V2-ZY section dans README.md + CLAUDE.md
[ ] Tag v0.2.0-zy posé
[ ] Tags antérieurs intacts
[ ] V2-Y baseline (35 tests existants) reste vert
==> Phase post-livraison : benchmark V2-ZY n=5 ep=5000 same-seed
    cible : 4/5 seeds best ≥ 70 % @ diff=0.30 en eval rigoureux V2-V.
```

---

## Récapitulatif

- **10 tasks** réparties sur **9 phases**
- **21 nouveaux tests** (231 → 252)
- **~12 commits** sur `main`
- **Tag livraison** : `v0.2.0-zy`
- **2 nouveaux fichiers code** : `conv_recurrent.py`, `conv_recurrent_dqn.py`
- **5 fichiers code modifiés** : `config.py`, `recurrent_trainer.py`, `evaluator.py`, `runner.py`, + 1 CLI script
- **4 fichiers tests nouveaux** : `test_conv_recurrent.py`, `test_conv_recurrent_dqn.py`, `test_conv_recurrent_dqn_config.py`, `test_conv_recurrent_procedural_runner.py`
- **2 fichiers tests étendus** : `test_recurrent_trainer.py` (+1 test), `test_evaluator.py` (+1 test)
- **DoD bloquante** : pytest 252 + Aether 8 OK + smoke E2E GPU + best.pt + tag + V2-Y régression zéro
- **DoD non-bloquante (objectif scientifique)** : benchmark V2-ZY n=5 ep=5000, cible 4/5 seeds best ≥ 70 % @ diff=0.30
