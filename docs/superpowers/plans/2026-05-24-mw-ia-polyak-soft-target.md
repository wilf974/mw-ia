# MW_IA V2-U Polyak soft target — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter un soft target update Polyak (`target ← τ × online + (1−τ) × target`) opt-in via `polyak_tau: float = 0.0` dans les 3 configs DQN. Hypothèse : réduire la variance inter-seed V2-ZY de 38 pp → < 20 pp sans dégrader la capacité maximale (seed 4 = 100 % @ diff=0.30).

**Architecture:** Champ `polyak_tau` dans `ConvDQNConfig` (V2-Z/W), `DRQNConfig` (V2-Y), `ConvRecurrentDQNConfig` (V2-ZY). Méthode `polyak_update(tau)` ajoutée aux 2 trainers (`_ConvDQNTrainer`, `RecurrentDQNTrainer`) qui mix in-place les paramètres. Branche conditionnelle dans `step()` post-optimizer. Agent skip le hard sync périodique si `polyak_tau > 0`. Default `0.0` partout → backwards compat strict.

**Tech Stack:** Python 3.13, PyTorch (cu128, in-place `mul_().add_()` sur params). Réutilise infrastructure V2-W/V2-Y/V2-ZY.

**Spec source:** `docs/superpowers/specs/2026-05-24-mw-ia-polyak-soft-target-design.md`

**État initial:** Branche `main`, 8 tags posés (jusqu'à `v0.2.0-zy`). **252 tests pytest verts**. Dernier commit avant V2-U : `a1132da` (spec V2-U).

---

## Phase 1 — Scaffold

### Task 1 : Créer le fichier de tests Polyak vide

**Files :**
- Create : `tests/neural/test_polyak_update.py` (vide)

- [ ] **Step 1 — Verify initial state**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `252 passed`.

- [ ] **Step 2 — Create empty test file**

Créer `tests/neural/test_polyak_update.py` avec contenu vide (0 byte).

- [ ] **Step 3 — Verify pytest still passes**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `252 passed`.

- [ ] **Step 4 — Commit**

```bash
git add tests/neural/test_polyak_update.py
git commit -m "chore(v2-u): scaffold tests/neural/test_polyak_update.py"
```

---

## Phase 2 — Méthode Polyak pure (sur `_ConvDQNTrainer` en premier)

### Task 2 : `polyak_update()` dans `_ConvDQNTrainer` + 5 tests TDD

**Files :**
- Modify : `mw_ia/agents/conv_dqn.py` (ajout méthode `polyak_update` dans `_ConvDQNTrainer`)
- Test : `tests/neural/test_polyak_update.py`

- [ ] **Step 1 — Write the 5 failing tests**

Contenu de `tests/neural/test_polyak_update.py` :

```python
"""Tests V2-U de polyak_update sur _ConvDQNTrainer (formule pure)."""
from __future__ import annotations

import torch

from mw_ia.agents.conv_dqn import ConvDQNAgent
from mw_ia.config import ConvDQNConfig


def _build_agent_pair() -> tuple[ConvDQNAgent, dict[str, torch.Tensor]]:
    """Construit un agent et capture une snapshot des params target initiaux."""
    cfg = ConvDQNConfig(min_replay_to_learn=10_000, use_amp=False)
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    # Désynchroniser online pour avoir online != target
    with torch.no_grad():
        for p in agent.online.parameters():
            p.add_(0.5)
    target_snapshot = {
        name: p.clone()
        for name, p in agent.target.named_parameters()
    }
    return agent, target_snapshot


def test_polyak_tau_zero_is_noop() -> None:
    """polyak_update(0.0) ne modifie pas target."""
    agent, snapshot = _build_agent_pair()
    agent.trainer.polyak_update(0.0)
    for name, p in agent.target.named_parameters():
        assert torch.allclose(p, snapshot[name]), f"{name} modified by tau=0.0"


def test_polyak_tau_one_copies_online_to_target() -> None:
    """polyak_update(1.0) rend target identique à online."""
    agent, _ = _build_agent_pair()
    agent.trainer.polyak_update(1.0)
    for p_target, p_online in zip(
        agent.target.parameters(), agent.online.parameters()
    ):
        assert torch.allclose(p_target, p_online)


def test_polyak_intermediate_tau() -> None:
    """polyak_update(0.5) produit target = 0.5 × old_target + 0.5 × online."""
    agent, snapshot = _build_agent_pair()
    # Capture online snapshot before update (in case Polyak modifies online by mistake)
    online_snapshot = {
        name: p.clone() for name, p in agent.online.named_parameters()
    }
    agent.trainer.polyak_update(0.5)
    for name, p_target_new in agent.target.named_parameters():
        expected = 0.5 * snapshot[name] + 0.5 * online_snapshot[name]
        assert torch.allclose(p_target_new, expected, atol=1e-6), (
            f"{name} mismatch with formula 0.5 × old_target + 0.5 × online"
        )


def test_polyak_idempotent_when_online_equals_target() -> None:
    """Si online == target, polyak_update(τ) ne change rien quel que soit τ."""
    cfg = ConvDQNConfig(min_replay_to_learn=10_000, use_amp=False)
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    # online == target dès l'init (sync_target dans Trainer.__init__)
    snapshot = {
        name: p.clone() for name, p in agent.target.named_parameters()
    }
    for tau in [0.1, 0.5, 0.9, 1.0]:
        agent.trainer.polyak_update(tau)
        for name, p in agent.target.named_parameters():
            assert torch.allclose(p, snapshot[name]), (
                f"tau={tau}: {name} modifié alors que online == target"
            )


def test_polyak_does_not_modify_online() -> None:
    """polyak_update ne touche qu'aux paramètres de target, online inchangé."""
    agent, _ = _build_agent_pair()
    online_snapshot = {
        name: p.clone() for name, p in agent.online.named_parameters()
    }
    agent.trainer.polyak_update(0.5)
    for name, p in agent.online.named_parameters():
        assert torch.allclose(p, online_snapshot[name]), (
            f"{name} (online) modifié par polyak_update"
        )
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_polyak_update.py -v 2>&1 | tail -15
```

Attendu : 5 fails (`AttributeError: '_ConvDQNTrainer' object has no attribute 'polyak_update'`).

- [ ] **Step 3 — Add `polyak_update` method to `_ConvDQNTrainer`**

Dans `mw_ia/agents/conv_dqn.py`, localiser la classe `_ConvDQNTrainer`. Ajouter la méthode juste après `sync_target()` (avant `step()`) :

```python
    def polyak_update(self, tau: float) -> None:
        """Soft update target ← τ × online + (1−τ) × target, in-place.

        Voir spec V2-U : docs/superpowers/specs/2026-05-24-mw-ia-polyak-soft-target-design.md
        """
        with torch.no_grad():
            for p_target, p_online in zip(
                self.target.parameters(), self.online.parameters()
            ):
                p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_polyak_update.py -v 2>&1 | tail -10
```

Attendu : `5 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `257 passed` (252 + 5).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/agents/conv_dqn.py tests/neural/test_polyak_update.py
git commit -m "feat(v2-u): add polyak_update() method to _ConvDQNTrainer (5 tests TDD)"
```

---

## Phase 3 — Branche conditionnelle dans `_ConvDQNTrainer.step()`

### Task 3 : Param `polyak_tau` + branche conditionnelle dans `step()`

**Files :**
- Modify : `mw_ia/agents/conv_dqn.py` (`_ConvDQNTrainer.__init__` + `_ConvDQNTrainer.step` + `ConvDQNAgent.__init__`)

- [ ] **Step 1 — Modify `_ConvDQNTrainer.__init__` to accept `polyak_tau`**

Dans `mw_ia/agents/conv_dqn.py`, localiser la signature `_ConvDQNTrainer.__init__` (qui se termine par `double_dqn: bool = True`). Remplacer la signature :

```python
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
        double_dqn: bool = True,
    ) -> None:
```

par :

```python
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
        double_dqn: bool = True,
        polyak_tau: float = 0.0,
    ) -> None:
```

Dans le corps de `__init__`, ajouter `self.polyak_tau = polyak_tau` juste après `self.double_dqn = double_dqn`.

- [ ] **Step 2 — Modify `_ConvDQNTrainer.step()` to call polyak_update**

Localiser la fin de `_ConvDQNTrainer.step()`. Juste avant le `return float(loss.detach().item())`, ajouter :

```python
        # V2-U : soft Polyak update à chaque train_step si tau > 0
        if self.polyak_tau > 0.0:
            self.polyak_update(self.polyak_tau)
```

- [ ] **Step 3 — Modify `ConvDQNAgent.__init__` to pass `cfg.polyak_tau` to trainer**

Localiser la construction du trainer dans `ConvDQNAgent.__init__`. Remplacer :

```python
        self.trainer = _ConvDQNTrainer(
            self.online, self.target,
            in_channels=in_channels, rows=rows, cols=cols,
            lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
            double_dqn=cfg.double_dqn,
        )
```

par :

```python
        self.trainer = _ConvDQNTrainer(
            self.online, self.target,
            in_channels=in_channels, rows=rows, cols=cols,
            lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
            double_dqn=cfg.double_dqn,
            polyak_tau=cfg.polyak_tau,
        )
```

**Note** : `cfg.polyak_tau` n'existe pas encore dans `ConvDQNConfig`. C'est Task 4. À ce stade, **les tests existants vont casser** (`AttributeError: 'ConvDQNConfig' object has no attribute 'polyak_tau'`). C'est OK : Task 3 + Task 4 sont enchainées ; le commit final passera après Task 4.

**ATTENDU À LA FIN DE TASK 3** : les tests V2-W/V2-Z existants cassent. **NE PAS COMMITER ICI**. Continuer directement Task 4 pour rétablir.

- [ ] **Step 4 — Verify tests are broken (expected mid-state)**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_conv_dqn.py -q 2>&1 | tail -3
```

Attendu : Erreurs `AttributeError: 'ConvDQNConfig' object has no attribute 'polyak_tau'`. Ne PAS commiter, continuer Task 4.

---

### Task 4 : Champ `polyak_tau` dans `ConvDQNConfig` + 1 test

**Files :**
- Modify : `mw_ia/config.py` (`ConvDQNConfig`)
- Test : `tests/test_conv_dqn_config.py`

- [ ] **Step 1 — Write the failing test**

Ajouter à la fin de `tests/test_conv_dqn_config.py` :

```python
def test_polyak_tau_default_and_validation() -> None:
    """V2-U : default polyak_tau=0.0, validation dans [0, 1]."""
    cfg = ConvDQNConfig()
    assert cfg.polyak_tau == 0.0
    cfg2 = ConvDQNConfig(polyak_tau=0.005)
    assert cfg2.polyak_tau == 0.005
    cfg3 = ConvDQNConfig(polyak_tau=1.0)
    assert cfg3.polyak_tau == 1.0
    with pytest.raises(ValueError):
        ConvDQNConfig(polyak_tau=-0.001)
    with pytest.raises(ValueError):
        ConvDQNConfig(polyak_tau=1.001)
```

- [ ] **Step 2 — Run test, verify it fails**

```bash
source .venv/Scripts/activate && pytest tests/test_conv_dqn_config.py::test_polyak_tau_default_and_validation -v 2>&1 | tail -10
```

Attendu : fail (`AttributeError`).

- [ ] **Step 3 — Add `polyak_tau` field + validation to `ConvDQNConfig`**

Dans `mw_ia/config.py`, localiser la dataclass `ConvDQNConfig`. Ajouter le champ `polyak_tau` juste après `double_dqn` (regrouper avec V2-W) :

Remplacer :

```python
    double_dqn: bool = True   # V2-W : Hasselt 2015. False = V2-Z baseline DQN classique.
```

par :

```python
    double_dqn: bool = True   # V2-W : Hasselt 2015. False = V2-Z baseline DQN classique.
    polyak_tau: float = 0.0   # V2-U : 0.0 = hard sync, >0 = soft Polyak (Lillicrap 2015).
```

Puis ajouter à la fin de `__post_init__` de `ConvDQNConfig` (après la dernière validation existante) :

```python
        if not (0.0 <= self.polyak_tau <= 1.0):
            raise ValueError(
                f"polyak_tau doit être ∈ [0, 1], reçu {self.polyak_tau}"
            )
```

- [ ] **Step 4 — Run tests, verify they pass**

```bash
source .venv/Scripts/activate && pytest tests/test_conv_dqn_config.py -v 2>&1 | tail -10
```

Attendu : tous les tests du fichier passent (incluant le nouveau `test_polyak_tau_default_and_validation`).

- [ ] **Step 5 — Run V2-W/V2-Z agent tests to verify regression resolved**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_conv_dqn.py -q 2>&1 | tail -3
```

Attendu : tous les tests V2-Z/V2-W passent à nouveau (default `polyak_tau=0.0` préserve baseline).

- [ ] **Step 6 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `258 passed` (257 + 1 nouveau test config).

- [ ] **Step 7 — Commit Task 3 + Task 4 together**

```bash
git add mw_ia/agents/conv_dqn.py mw_ia/config.py tests/test_conv_dqn_config.py
git commit -m "feat(v2-u): wire polyak_tau in ConvDQNConfig + _ConvDQNTrainer.step()"
```

---

## Phase 4 — Skip hard sync conditionnel dans `ConvDQNAgent`

### Task 5 : `ConvDQNAgent.observe()` skip hard sync si Polyak + 2 tests

**Files :**
- Modify : `mw_ia/agents/conv_dqn.py` (`ConvDQNAgent.observe`)
- Test : `tests/agents/test_conv_dqn.py`

- [ ] **Step 1 — Write the 2 failing tests**

Ajouter à la fin de `tests/agents/test_conv_dqn.py` :

```python
def test_polyak_tau_skips_hard_sync() -> None:
    """V2-U : avec polyak_tau > 0, hard sync périodique skip."""
    cfg = ConvDQNConfig(
        polyak_tau=0.005,
        target_sync_steps=2,
        min_replay_to_learn=10_000,  # disable train_step
        use_amp=False,
    )
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    obs = np.zeros((3, 10, 10), dtype=np.float32)
    for _ in range(5):
        agent.observe(obs, action=0, reward=0.0, next_state=obs, done=False)
    # target_syncs n'a jamais été incrémenté car Polyak prend le relais
    assert agent.target_syncs == 0


def test_polyak_tau_zero_preserves_hard_sync() -> None:
    """V2-U : avec polyak_tau=0.0 (default), hard sync s'active comme baseline V2-W."""
    cfg = ConvDQNConfig(
        polyak_tau=0.0,
        target_sync_steps=2,
        min_replay_to_learn=10_000,
        use_amp=False,
    )
    agent = ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    obs = np.zeros((3, 10, 10), dtype=np.float32)
    for _ in range(5):
        agent.observe(obs, action=0, reward=0.0, next_state=obs, done=False)
    # target_syncs doit avoir été incrémenté (ép 2, 4 → 2 syncs)
    assert agent.target_syncs >= 2
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_conv_dqn.py::test_polyak_tau_skips_hard_sync -v 2>&1 | tail -10
```

Attendu : fail (`assert agent.target_syncs == 0` — actuellement hard sync s'active toujours).

- [ ] **Step 3 — Modify `ConvDQNAgent.observe()` to skip hard sync if Polyak**

Dans `mw_ia/agents/conv_dqn.py`, localiser `ConvDQNAgent.observe()`. Remplacer le bloc :

```python
        if self.global_step % self.cfg.target_sync_steps == 0:
            self.trainer.sync_target()
            self.target_syncs += 1
        return metrics
```

par :

```python
        # V2-U : skip hard sync périodique si Polyak activé (le trainer.step()
        # appelle déjà polyak_update à chaque train_step).
        if self.cfg.polyak_tau == 0.0:
            if self.global_step % self.cfg.target_sync_steps == 0:
                self.trainer.sync_target()
                self.target_syncs += 1
        return metrics
```

- [ ] **Step 4 — Run tests, verify they pass**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_conv_dqn.py -v 2>&1 | tail -10
```

Attendu : tous les tests V2-Z/V2-W/V2-U passent.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `260 passed` (258 + 2).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/agents/conv_dqn.py tests/agents/test_conv_dqn.py
git commit -m "feat(v2-u): ConvDQNAgent skip periodic hard sync when polyak_tau > 0"
```

---

## Phase 5 — `RecurrentDQNTrainer` extension Polyak (V2-Y/V2-ZY)

### Task 6 : `polyak_update()` + param + branche conditionnelle dans `RecurrentDQNTrainer`

**Files :**
- Modify : `mw_ia/neural/recurrent_trainer.py`
- Test : `tests/neural/test_recurrent_trainer.py`

- [ ] **Step 1 — Write the failing test**

Ajouter à la fin de `tests/neural/test_recurrent_trainer.py` :

```python
def test_polyak_update_changes_target_in_step(cpu_device: torch.device) -> None:
    """V2-U : avec polyak_tau > 0, target params changent après step()."""
    import numpy as np

    from mw_ia.neural.recurrent import RecurrentQNetwork
    from mw_ia.neural.recurrent_trainer import RecurrentDQNTrainer
    from mw_ia.neural.sequence_buffer import SequenceReplayBuffer

    online = RecurrentQNetwork(input_dim=300, n_actions=4, fc_hidden=64, lstm_hidden=32).to(cpu_device)
    target = RecurrentQNetwork(input_dim=300, n_actions=4, fc_hidden=64, lstm_hidden=32).to(cpu_device)
    target.load_state_dict(online.state_dict())
    # Désynchroniser online pour que Polyak ait un effet
    with torch.no_grad():
        for p in online.parameters():
            p.add_(0.5)

    trainer = RecurrentDQNTrainer(
        online, target,
        lr=1e-3, gamma=0.99, device="cpu", use_amp=False,
        polyak_tau=0.5,  # τ=0.5 pour effet maximal sur 1 step
    )

    # Snapshot target avant step
    target_before = {name: p.clone() for name, p in target.named_parameters()}

    # Construire un batch minimal
    buffer = SequenceReplayBuffer(capacity=10, obs_dim=300, max_steps=8, seed=0)
    traj = [(np.zeros(300, np.float32), 0, 0.0, np.zeros(300, np.float32), False)] * 4
    buffer.push_trajectory(traj)
    buffer.push_trajectory(traj)
    batch = buffer.sample(batch_size=2, seq_len=4)
    trainer.step(batch)

    # Target params ont changé (Polyak update appliqué)
    target_changed = False
    for name, p_after in target.named_parameters():
        if not torch.allclose(p_after, target_before[name]):
            target_changed = True
            break
    assert target_changed, "Polyak update n'a pas modifié target après step()"
```

- [ ] **Step 2 — Run test, verify it fails**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_recurrent_trainer.py::test_polyak_update_changes_target_in_step -v 2>&1 | tail -10
```

Attendu : fail (`TypeError: RecurrentDQNTrainer.__init__() got an unexpected keyword argument 'polyak_tau'`).

- [ ] **Step 3 — Add `polyak_tau` param + method + branch in `RecurrentDQNTrainer`**

Dans `mw_ia/neural/recurrent_trainer.py`, modifier la signature de `__init__`. Remplacer :

```python
    def __init__(
        self,
        online: nn.Module,  # RecurrentQNetwork ou ConvRecurrentQNetwork
        target: nn.Module,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cuda",
        use_amp: bool = True,
        double_dqn: bool = False,
    ) -> None:
```

par :

```python
    def __init__(
        self,
        online: nn.Module,  # RecurrentQNetwork ou ConvRecurrentQNetwork
        target: nn.Module,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cuda",
        use_amp: bool = True,
        double_dqn: bool = False,
        polyak_tau: float = 0.0,
    ) -> None:
```

Dans le corps de `__init__`, ajouter `self.polyak_tau = polyak_tau` juste après `self.double_dqn = double_dqn`.

Ajouter la méthode `polyak_update` juste après `sync_target()` :

```python
    def polyak_update(self, tau: float) -> None:
        """Soft update target ← τ × online + (1−τ) × target, in-place.

        Voir spec V2-U : docs/superpowers/specs/2026-05-24-mw-ia-polyak-soft-target-design.md
        """
        with torch.no_grad():
            for p_target, p_online in zip(
                self.target.parameters(), self.online.parameters()
            ):
                p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)
```

Dans `step()`, juste avant `return float(loss.detach().item())`, ajouter :

```python
        # V2-U : soft Polyak update à chaque train_step si tau > 0
        if self.polyak_tau > 0.0:
            self.polyak_update(self.polyak_tau)
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/neural/test_recurrent_trainer.py -v 2>&1 | tail -10
```

Attendu : tous les tests V2-Y existants + 1 nouveau V2-U passent.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `261 passed` (260 + 1).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/neural/recurrent_trainer.py tests/neural/test_recurrent_trainer.py
git commit -m "feat(v2-u): add polyak_tau param + polyak_update() to RecurrentDQNTrainer"
```

---

## Phase 6 — `DRQNConfig` + `ConvRecurrentDQNConfig` extensions

### Task 7 : Champ `polyak_tau` dans `DRQNConfig` (V2-Y) + 1 test

**Files :**
- Modify : `mw_ia/config.py` (`DRQNConfig`)
- Test : `tests/test_drqn_config.py`

- [ ] **Step 1 — Write the failing test**

Ajouter à la fin de `tests/test_drqn_config.py` :

```python
def test_drqn_polyak_tau_default_and_validation() -> None:
    """V2-U : default polyak_tau=0.0, validation dans [0, 1]."""
    cfg = DRQNConfig()
    assert cfg.polyak_tau == 0.0
    cfg2 = DRQNConfig(polyak_tau=0.005)
    assert cfg2.polyak_tau == 0.005
    with pytest.raises(ValueError):
        DRQNConfig(polyak_tau=-0.001)
    with pytest.raises(ValueError):
        DRQNConfig(polyak_tau=1.001)
```

- [ ] **Step 2 — Run test, verify it fails**

```bash
source .venv/Scripts/activate && pytest tests/test_drqn_config.py::test_drqn_polyak_tau_default_and_validation -v 2>&1 | tail -10
```

Attendu : fail (`AttributeError` ou `TypeError`).

- [ ] **Step 3 — Add `polyak_tau` field + validation to `DRQNConfig`**

Dans `mw_ia/config.py`, localiser la dataclass `DRQNConfig`. Ajouter le champ `polyak_tau` juste après `use_amp: bool = True` (juste avant `# Training`) :

```python
    use_amp: bool = True
    polyak_tau: float = 0.0   # V2-U : 0.0 = hard sync, >0 = soft Polyak (Lillicrap 2015).
```

Puis ajouter à la fin de `__post_init__` de `DRQNConfig` (après la dernière validation existante) :

```python
        if not (0.0 <= self.polyak_tau <= 1.0):
            raise ValueError(
                f"polyak_tau doit être ∈ [0, 1], reçu {self.polyak_tau}"
            )
```

- [ ] **Step 4 — Run tests, verify they pass**

```bash
source .venv/Scripts/activate && pytest tests/test_drqn_config.py -v 2>&1 | tail -10
```

Attendu : tous les tests V2-Y existants + 1 nouveau V2-U passent.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `262 passed` (261 + 1).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/config.py tests/test_drqn_config.py
git commit -m "feat(v2-u): add polyak_tau field to DRQNConfig (V2-Y)"
```

---

### Task 8 : Champ `polyak_tau` dans `ConvRecurrentDQNConfig` (V2-ZY) + 1 test

**Files :**
- Modify : `mw_ia/config.py` (`ConvRecurrentDQNConfig`)
- Test : `tests/test_conv_recurrent_dqn_config.py`

- [ ] **Step 1 — Write the failing test**

Ajouter à la fin de `tests/test_conv_recurrent_dqn_config.py` :

```python
def test_v2zy_polyak_tau_default_and_validation() -> None:
    """V2-U : default polyak_tau=0.0, validation dans [0, 1]."""
    cfg = ConvRecurrentDQNConfig()
    assert cfg.polyak_tau == 0.0
    cfg2 = ConvRecurrentDQNConfig(polyak_tau=0.005)
    assert cfg2.polyak_tau == 0.005
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(polyak_tau=-0.001)
    with pytest.raises(ValueError):
        ConvRecurrentDQNConfig(polyak_tau=1.001)
```

- [ ] **Step 2 — Run test, verify it fails**

```bash
source .venv/Scripts/activate && pytest tests/test_conv_recurrent_dqn_config.py::test_v2zy_polyak_tau_default_and_validation -v 2>&1 | tail -10
```

Attendu : fail.

- [ ] **Step 3 — Add `polyak_tau` field + validation to `ConvRecurrentDQNConfig`**

Dans `mw_ia/config.py`, localiser la dataclass `ConvRecurrentDQNConfig`. Ajouter le champ `polyak_tau` juste après `double_dqn: bool = True` (regrouper V2-W et V2-U) :

```python
    # V2-W : Double DQN activé par défaut V2-ZY (combo des 3 leviers)
    double_dqn: bool = True
    polyak_tau: float = 0.0   # V2-U : 0.0 = hard sync, >0 = soft Polyak.
```

Puis ajouter à la fin de `__post_init__` de `ConvRecurrentDQNConfig` (après la dernière validation existante) :

```python
        if not (0.0 <= self.polyak_tau <= 1.0):
            raise ValueError(
                f"polyak_tau doit être ∈ [0, 1], reçu {self.polyak_tau}"
            )
```

- [ ] **Step 4 — Run tests, verify they pass**

```bash
source .venv/Scripts/activate && pytest tests/test_conv_recurrent_dqn_config.py -v 2>&1 | tail -10
```

Attendu : tous les tests V2-ZY existants + 1 nouveau V2-U passent.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `263 passed` (262 + 1).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/config.py tests/test_conv_recurrent_dqn_config.py
git commit -m "feat(v2-u): add polyak_tau field to ConvRecurrentDQNConfig (V2-ZY)"
```

---

## Phase 7 — Skip hard sync conditionnel dans `RecurrentDQNAgent` (V2-Y) + `ConvRecurrentDQNAgent` (V2-ZY)

### Task 9 : `RecurrentDQNAgent` (V2-Y) skip hard sync si Polyak + 1 test

**Files :**
- Modify : `mw_ia/agents/recurrent_dqn.py`
- Test : `tests/agents/test_recurrent_dqn.py`

- [ ] **Step 1 — Write the failing test**

Ajouter à la fin de `tests/agents/test_recurrent_dqn.py` :

```python
def test_v2y_polyak_tau_skips_hard_sync() -> None:
    """V2-U : V2-Y agent avec polyak_tau > 0 skip hard sync périodique."""
    cfg = DRQNConfig(
        polyak_tau=0.005,
        target_sync_steps=2,
        min_episodes_to_learn=10_000,  # disable train_step
        use_amp=False,
        sequence_length=4,
        max_steps_per_episode=8,
    )
    agent = RecurrentDQNAgent(
        obs_dim=200, n_actions=4, cfg=cfg, device="cpu", seed=0,
    )
    obs = np.zeros(200, dtype=np.float32)
    # Simuler 5 épisodes courts
    for _ in range(5):
        agent.reset_hidden()
        agent.begin_episode()
        for _ in range(4):
            agent.observe(obs, action=0, reward=0.0, next_state=obs, done=False)
        agent.end_episode()
    # target_syncs jamais incrémenté
    assert agent.target_syncs == 0
```

- [ ] **Step 2 — Run test, verify it fails**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_recurrent_dqn.py::test_v2y_polyak_tau_skips_hard_sync -v 2>&1 | tail -10
```

Attendu : fail (`AttributeError` ou `assert 0 == X`).

- [ ] **Step 3 — Modify `RecurrentDQNAgent.__init__` to pass `cfg.polyak_tau` to trainer**

Dans `mw_ia/agents/recurrent_dqn.py`, localiser la construction du trainer dans `__init__`. Remplacer :

```python
        self.trainer = RecurrentDQNTrainer(
            self.online, self.target, lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
        )
```

par :

```python
        self.trainer = RecurrentDQNTrainer(
            self.online, self.target, lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
            polyak_tau=cfg.polyak_tau,
        )
```

- [ ] **Step 4 — Modify `RecurrentDQNAgent.end_episode()` to skip hard sync if Polyak**

Localiser le bloc dans `end_episode()` :

```python
        if self.global_step // self.cfg.target_sync_steps > self.target_syncs:
            self.trainer.sync_target()
            self.target_syncs += 1
        return metrics
```

Remplacer par :

```python
        # V2-U : skip hard sync périodique si Polyak activé (le trainer.step()
        # appelle déjà polyak_update à chaque train_step).
        if self.cfg.polyak_tau == 0.0:
            if self.global_step // self.cfg.target_sync_steps > self.target_syncs:
                self.trainer.sync_target()
                self.target_syncs += 1
        return metrics
```

- [ ] **Step 5 — Run tests, verify they pass**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_recurrent_dqn.py -v 2>&1 | tail -10
```

Attendu : tous les tests V2-Y existants + 1 nouveau V2-U passent.

- [ ] **Step 6 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `264 passed` (263 + 1).

- [ ] **Step 7 — Commit**

```bash
git add mw_ia/agents/recurrent_dqn.py tests/agents/test_recurrent_dqn.py
git commit -m "feat(v2-u): RecurrentDQNAgent (V2-Y) skip hard sync when polyak_tau > 0"
```

---

### Task 10 : `ConvRecurrentDQNAgent` (V2-ZY) skip hard sync si Polyak + 1 test

**Files :**
- Modify : `mw_ia/agents/conv_recurrent_dqn.py`
- Test : `tests/agents/test_conv_recurrent_dqn.py`

- [ ] **Step 1 — Write the failing test**

Ajouter à la fin de `tests/agents/test_conv_recurrent_dqn.py` :

```python
def test_v2zy_polyak_tau_skips_hard_sync() -> None:
    """V2-U : V2-ZY agent avec polyak_tau > 0 skip hard sync périodique."""
    cfg = ConvRecurrentDQNConfig(
        polyak_tau=0.005,
        target_sync_steps=2,
        min_episodes_to_learn=10_000,
        use_amp=False,
        sequence_length=4,
        max_steps_per_episode=8,
    )
    agent = ConvRecurrentDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    obs = np.zeros((3, 10, 10), dtype=np.float32)
    for _ in range(5):
        agent.reset_hidden()
        agent.begin_episode()
        for _ in range(4):
            agent.observe(obs, action=0, reward=0.0, next_state=obs, done=False)
        agent.end_episode()
    assert agent.target_syncs == 0
```

- [ ] **Step 2 — Run test, verify it fails**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_conv_recurrent_dqn.py::test_v2zy_polyak_tau_skips_hard_sync -v 2>&1 | tail -10
```

Attendu : fail.

- [ ] **Step 3 — Modify `ConvRecurrentDQNAgent.__init__` to pass `cfg.polyak_tau`**

Dans `mw_ia/agents/conv_recurrent_dqn.py`, localiser la construction du trainer dans `__init__`. Remplacer :

```python
        self.trainer = RecurrentDQNTrainer(
            self.online, self.target,
            lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
            double_dqn=cfg.double_dqn,
        )
```

par :

```python
        self.trainer = RecurrentDQNTrainer(
            self.online, self.target,
            lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
            double_dqn=cfg.double_dqn,
            polyak_tau=cfg.polyak_tau,
        )
```

- [ ] **Step 4 — Modify `ConvRecurrentDQNAgent.end_episode()` to skip hard sync if Polyak**

Localiser le bloc dans `end_episode()` :

```python
        if self.global_step // self.cfg.target_sync_steps > self.target_syncs:
            self.trainer.sync_target()
            self.target_syncs += 1
        return metrics
```

Remplacer par :

```python
        # V2-U : skip hard sync périodique si Polyak activé.
        if self.cfg.polyak_tau == 0.0:
            if self.global_step // self.cfg.target_sync_steps > self.target_syncs:
                self.trainer.sync_target()
                self.target_syncs += 1
        return metrics
```

- [ ] **Step 5 — Run tests, verify they pass**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_conv_recurrent_dqn.py -v 2>&1 | tail -10
```

Attendu : tous les tests V2-ZY existants + 1 nouveau V2-U passent.

- [ ] **Step 6 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `265 passed` (264 + 1).

- [ ] **Step 7 — Commit**

```bash
git add mw_ia/agents/conv_recurrent_dqn.py tests/agents/test_conv_recurrent_dqn.py
git commit -m "feat(v2-u): ConvRecurrentDQNAgent (V2-ZY) skip hard sync when polyak_tau > 0"
```

---

## Phase 8 — CLI flags `--polyak-tau` (3 scripts)

### Task 11 : Ajouter `--polyak-tau` aux 3 scripts CLI

**Files :**
- Modify : `scripts/train_cnn_dqn_procedural.py` (V2-Z/W)
- Modify : `scripts/train_drqn_procedural.py` (V2-Y)
- Modify : `scripts/train_cnn_lstm_dqn_procedural.py` (V2-ZY)

- [ ] **Step 1 — Modify `scripts/train_cnn_dqn_procedural.py`**

Localiser la section argparse. Ajouter le flag juste avant `args = parser.parse_args()` :

```python
    parser.add_argument(
        "--polyak-tau",
        type=float,
        default=0.0,
        help="V2-U : soft Polyak target update τ. Default 0.0 = hard sync. "
             "Recommandé : 0.005 pour activer Polyak.",
    )
```

Localiser la construction de `ConvDQNConfig`. Ajouter `polyak_tau=args.polyak_tau` aux kwargs :

```python
    dqn_cfg = ConvDQNConfig(
        episodes=args.episodes,
        conv_channels=tuple(args.conv_channels),
        fc_hidden=args.fc_hidden,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_sync_steps=args.target_sync_steps,
        double_dqn=args.double_dqn,
        eval_enabled=args.eval,
        eval_every_episodes=args.eval_every_episodes,
        eval_target_difficulty=args.eval_target_difficulty,
        best_checkpoint_path=args.best_checkpoint_path,
        polyak_tau=args.polyak_tau,
    )
```

- [ ] **Step 2 — Modify `scripts/train_drqn_procedural.py`**

Ajouter le même flag CLI juste avant `args = parser.parse_args()` :

```python
    parser.add_argument(
        "--polyak-tau",
        type=float,
        default=0.0,
        help="V2-U : soft Polyak target update τ. Default 0.0 = hard sync.",
    )
```

Localiser la construction de `DRQNConfig`. Ajouter `polyak_tau=args.polyak_tau`.

- [ ] **Step 3 — Modify `scripts/train_cnn_lstm_dqn_procedural.py`**

Ajouter le même flag CLI juste avant `args = parser.parse_args()` :

```python
    parser.add_argument(
        "--polyak-tau",
        type=float,
        default=0.0,
        help="V2-U : soft Polyak target update τ. Default 0.0 = hard sync. "
             "Recommandé V2-ZY : 0.005 pour réduire variance inter-seed.",
    )
```

Localiser la construction de `ConvRecurrentDQNConfig`. Ajouter `polyak_tau=args.polyak_tau`.

- [ ] **Step 4 — Smoke CPU V2-ZY+Polyak**

```bash
mkdir -p checkpoints && \
source .venv/Scripts/activate && \
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles --device cpu \
    --polyak-tau 0.005 \
    --eval-every-episodes 5 \
    --best-checkpoint-path checkpoints/v2u_smoke.pt 2>&1 | tail -15
```

Attendu : pas de crash, eval logs présents, fichier .pt créé.

Vérification :

```bash
ls -la checkpoints/v2u_smoke.pt && rm checkpoints/v2u_smoke.pt
```

- [ ] **Step 5 — Smoke CPU V2-ZY baseline (polyak_tau=0.0)**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles --device cpu --polyak-tau 0.0 --eval-every-episodes 5 2>&1 | tail -10
```

Attendu : pas de crash, comportement identique à V2-ZY baseline.

- [ ] **Step 6 — Verify --help shows new flag**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_lstm_dqn_procedural.py --help 2>&1 | grep polyak
```

Attendu : `--polyak-tau POLYAK_TAU` + help text contenant "V2-U" et "0.005".

- [ ] **Step 7 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `265 passed`.

- [ ] **Step 8 — Commit**

```bash
git add scripts/train_cnn_dqn_procedural.py scripts/train_drqn_procedural.py scripts/train_cnn_lstm_dqn_procedural.py
git commit -m "feat(v2-u): add --polyak-tau CLI flag to 3 training scripts"
```

---

## Phase 9 — CI workflow smoke V2-U

### Task 12 : CI workflow smoke V2-U avec `--polyak-tau 0.005`

**Files :**
- Modify : `.github/workflows/aether_verify.yml`

- [ ] **Step 1 — Add V2-U smoke step**

Localiser le step "Smoke test V2-ZY CNN + LSTM + Double DQN" dans `.github/workflows/aether_verify.yml`. Ajouter juste après (pattern identique) :

```yaml
      - name: Smoke test V2-U Polyak soft target on V2-ZY
        run: |
          mkdir -p checkpoints
          python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles --device cpu --polyak-tau 0.005 --eval-every-episodes 5 --best-checkpoint-path checkpoints/ci_v2u_best.pt
          test -f checkpoints/ci_v2u_best.pt
```

- [ ] **Step 2 — Validate YAML syntax**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/aether_verify.yml'))" && echo "YAML OK"
```

Attendu : `YAML OK`.

- [ ] **Step 3 — Smoke local pour valider l'invocation CI**

```bash
mkdir -p checkpoints && \
source .venv/Scripts/activate && \
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles --device cpu --polyak-tau 0.005 --eval-every-episodes 5 --best-checkpoint-path checkpoints/ci_v2u_test.pt && \
test -f checkpoints/ci_v2u_test.pt && echo "FILE OK" && rm checkpoints/ci_v2u_test.pt
```

Attendu : `FILE OK`.

- [ ] **Step 4 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `265 passed`.

- [ ] **Step 5 — Commit**

```bash
git add .github/workflows/aether_verify.yml
git commit -m "ci(v2-u): add smoke test V2-U Polyak on V2-ZY to aether_verify.yml"
```

---

## Phase 10 — README + CLAUDE.md + smoke E2E GPU + tag

### Task 13 : Documentation V2-U + smoke GPU + tag `v0.2.0-u`

**Files :**
- Modify : `README.md`
- Modify : `CLAUDE.md`

- [ ] **Step 1 — Smoke E2E manuel GPU 500 ép V2-ZY+Polyak**

```bash
mkdir -p checkpoints && \
source .venv/Scripts/activate && \
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 500 --mode obstacles --device cuda \
    --polyak-tau 0.005 \
    --eval-every-episodes 100 \
    --best-checkpoint-path checkpoints/v2u_smoke_gpu.pt 2>&1 | tail -15
```

Attendu : pas de crash sur 500 ép GPU, ≥ 5 lignes `eval ep`, fichier .pt créé.

Vérification :

```bash
ls -la checkpoints/v2u_smoke_gpu.pt && rm checkpoints/v2u_smoke_gpu.pt
```

- [ ] **Step 2 — Smoke E2E V2-ZY baseline (polyak_tau=0.0, regression check)**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_lstm_dqn_procedural.py --episodes 100 --mode obstacles --device cuda --polyak-tau 0.0 --eval-every-episodes 50 2>&1 | tail -10
```

Attendu : comportement identique à V2-ZY baseline (target_syncs s'incrémente, logs identiques).

- [ ] **Step 3 — Aether re-verify**

```bash
bash aether/verify_all.sh
```

Attendu : `8 OK`.

- [ ] **Step 4 — Add V2-U section to README.md**

Localiser la section `## V2-ZY` dans `README.md`. Insérer une nouvelle section V2-U juste après la fin de V2-ZY et avant `## Roadmap (V2+)`.

Contenu exact à insérer :

````markdown
## V2-U — Polyak soft target update (sous-projet livré)

**Tag** : `v0.2.0-u` — **Tests** : 265 verts (252 baseline + 13 V2-U)

Sous-projet pour stabiliser V2-ZY (variance inter-seed 38 pp → cible < 20 pp) sans dégrader la capacité maximale.

### Hypothèse

Remplacer hard sync target tous les N steps par soft Polyak update `target ← τ × online + (1−τ) × target` à chaque train_step avec τ ≈ 0.005. Devrait réduire les oscillations target Q et stabiliser Conv+LSTM+BPTT.

### Usage CLI (opt-in via `--polyak-tau`)

```bash
# V2-ZY + Polyak (recommandé)
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda \
    --polyak-tau 0.005 \
    --best-checkpoint-path checkpoints/v2u_best_seed0.pt

# V2-W + Polyak (validation transverse)
python scripts/train_cnn_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda \
    --polyak-tau 0.005

# Baseline (V2-W/V2-Y/V2-ZY actuel : hard sync)
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda
```

### Architecture

- Champ `polyak_tau: float = 0.0` ajouté à `ConvDQNConfig`, `DRQNConfig`, `ConvRecurrentDQNConfig`. Default 0.0 = hard sync (backwards compat strict).
- Méthode `polyak_update(tau)` ajoutée à `_ConvDQNTrainer` (V2-Z/W) et `RecurrentDQNTrainer` (V2-Y/ZY). Mix in-place via `mul_().add_()` sur `parameters()` itération.
- Branche conditionnelle dans `step()` post-optimizer : `if self.polyak_tau > 0.0: self.polyak_update(self.polyak_tau)`.
- Agents skip le hard sync périodique si `cfg.polyak_tau > 0` (évite double-update).

### Critère succès

Variance inter-seed best @ diff=0.30 sur V2-ZY+Polyak n=5 **< 20 pp** (vs 38 pp baseline V2-ZY).
````

- [ ] **Step 5 — Add V2-U row to CLAUDE.md sub-projects table**

Dans `CLAUDE.md`, localiser le tableau "Sous-projets — décomposition". Ajouter une ligne U après la ligne ZY. Remplacer :

```markdown
| **ZY** | CNN + LSTM + Double DQN combiné | ✅ Livré (tag `v0.2.0-zy`) |
| **B** | Mémoire persistante cross-session | ⏳ **Prochain par défaut** |
```

par :

```markdown
| **ZY** | CNN + LSTM + Double DQN combiné | ✅ Livré (tag `v0.2.0-zy`) |
| **U** | Polyak soft target update | ✅ Livré (tag `v0.2.0-u`) |
| **B** | Mémoire persistante cross-session | ⏳ **Prochain par défaut** |
```

- [ ] **Step 6 — Add V2-U phases section to CLAUDE.md**

Dans `CLAUDE.md`, localiser une bonne position pour insérer (typiquement après la section "V2-ZY — benchmark n=5"). Insérer la section V2-U :

```markdown
### V2-U — état final des phases (livraison 2026-05-24)

| Phase | Tâches | Statut | Tests | Commits |
|---|---|---|---|---|
| 1 — Scaffold | T1 | ✅ | 0 | 1 |
| 2 — `polyak_update()` dans `_ConvDQNTrainer` | T2 | ✅ | 5 | 1 |
| 3+4 — `polyak_tau` flag + branche `_ConvDQNTrainer.step()` + `ConvDQNConfig` | T3-T4 | ✅ | 1 | 1 |
| 5 — `ConvDQNAgent` skip hard sync si Polyak | T5 | ✅ | 2 | 1 |
| 6 — `RecurrentDQNTrainer` extension Polyak | T6 | ✅ | 1 | 1 |
| 7 — `DRQNConfig` polyak_tau field | T7 | ✅ | 1 | 1 |
| 8 — `ConvRecurrentDQNConfig` polyak_tau field | T8 | ✅ | 1 | 1 |
| 9 — `RecurrentDQNAgent` skip hard sync | T9 | ✅ | 1 | 1 |
| 10 — `ConvRecurrentDQNAgent` skip hard sync | T10 | ✅ | 1 | 1 |
| 11 — CLI flags `--polyak-tau` × 3 scripts | T11 | ✅ | 0 | 1 |
| 12 — CI smoke V2-U | T12 | ✅ | 0 | 1 |
| 13 — README + CLAUDE.md + tag `v0.2.0-u` | T13 | ✅ | 0 | 1 + tag |

### Composants V2-U livrés

| Composant | Fichier | Rôle |
|---|---|---|
| `polyak_update(tau)` dans `_ConvDQNTrainer` | `mw_ia/agents/conv_dqn.py` | Soft update target ← τ × online + (1−τ) × target, in-place via `mul_().add_()`. |
| `polyak_update(tau)` dans `RecurrentDQNTrainer` | `mw_ia/neural/recurrent_trainer.py` | Idem pattern. Réutilisé par V2-Y et V2-ZY. |
| Champ `polyak_tau: float = 0.0` | `ConvDQNConfig`, `DRQNConfig`, `ConvRecurrentDQNConfig` (`mw_ia/config.py`) | Default 0.0 = hard sync (backwards compat). Validation [0, 1]. |
| Skip hard sync conditionnel | 3 agents (`ConvDQNAgent.observe`, `RecurrentDQNAgent.end_episode`, `ConvRecurrentDQNAgent.end_episode`) | Si `cfg.polyak_tau > 0`, skip `target_sync_steps` periodic hard sync. |
| CLI flag `--polyak-tau` | 3 scripts (`train_cnn_dqn_procedural.py`, `train_drqn_procedural.py`, `train_cnn_lstm_dqn_procedural.py`) | Default 0.0. Activation V2-U via `--polyak-tau 0.005`. |

### Décisions techniques V2-U

- **Formule Polyak** : `target ← τ × online + (1−τ) × target`, in-place via `p_target.data.mul_(1-tau).add_(p_online.data, alpha=tau)`. Standard Lillicrap 2015 DDPG.
- **`with torch.no_grad()`** autour de la formule (pas de grad accumulation).
- **Activation par train_step, pas par step env** : appliqué dans `trainer.step()` post-optimizer. ~4 train_steps/episode × 5000 ép = ~20k updates, smoothing exponentiel ~200 train_steps.
- **Skip hard sync si Polyak** : évite double-update. Logique dans `agent.observe()/end_episode()` : `if cfg.polyak_tau == 0.0: hard_sync(every target_sync_steps)`.
- **Default 0.0 partout** : backwards compat strict. V2-W/V2-Y/V2-ZY baselines n=5 reproductibles sans modif. Strict opt-in via CLI.
- **τ = 0.005 recommandé** : standard DDPG/SAC. Smoothing constant temporel ~200 train_steps.

### V2-U — pièges connus

1. **Double-update target si Polyak ET hard sync pas skip** : logique de skip dans agent (`if cfg.polyak_tau == 0.0: hard_sync`). Tests vérifient `target_syncs == 0` quand Polyak activé.
2. **Polyak n'inclut pas les buffers BN/LN** : réseaux actuels (Conv2d/ReLU/Linear/LSTM) n'ont pas de running stats. `parameters()` suffit. À noter pour évolution future (R2D2 LayerNorm).
3. **AMP + Polyak** : `polyak_update` en `torch.no_grad()` mais PAS sous autocast. Storage float32 → safe.
4. **τ trop conservateur ou trop agressif ?** : 0.005 default littéraire. Si V2-U n'atteint pas critère, grid search τ ∈ {0.001, 0.01, 0.05} en suite.
5. **save/load** : `cfg.__dict__` inclut `polyak_tau` automatiquement (frozen dataclass). Charger un checkpoint avec `polyak_tau` différent : safe (juste config change post-load).
```

- [ ] **Step 7 — Update "Prochaines étapes prioritaires" section dans CLAUDE.md**

Localiser la section "Prochaines étapes prioritaires" et mettre à jour. Remplacer le contenu par :

```markdown
**Prochaines étapes prioritaires (post V2-U livré 2026-05-24)** :

1. ✅ **V2-U Polyak soft target** — **LIVRÉ** (tag `v0.2.0-u`) : `polyak_tau` opt-in dans 3 configs DQN + 2 trainers + 3 agents + 3 CLI scripts.

2. **Benchmark V2-ZY+Polyak n=5 ep=5000 same-seed** — validation scientifique non-bloquante :
   - Lancer 5 runs V2-ZY ep=5000 avec `--polyak-tau 0.005 --best-checkpoint-path checkpoints/v2u_zy_best_seed{N}.pt`
   - Comparer best @ diff=0.30 vs V2-ZY baseline n=5 (mean 42 %, std 38 pp, seed 4 = 100 %)
   - **Critère succès primaire** : std < 20 pp (vs 38 pp baseline)
   - **Pas de cible sur le mean** — finding est pur gain de robustesse
   - Si critère atteint → re-benchmark V2-W+Polyak (consolidation transverse)
   - Si critère non atteint → grid search τ ∈ {0.001, 0.01, 0.05} ou R2D2 burn-in

3. **Sous-projets V3+ déblocables** (post V2-U benchmark) :
   - **R2D2 burn-in** : stabiliser LSTM directement
   - **Mazes larges (max_size=15/20)** : test translation equivariance CNN
   - **Sous-projet B (mémoire persistante cross-session)**
```

- [ ] **Step 8 — Run full suite + Aether final**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
bash aether/verify_all.sh
```

Attendu : `265 passed` + `8 OK`.

- [ ] **Step 9 — Commit doc**

```bash
git add README.md CLAUDE.md
git commit -m "docs(v2-u): add V2-U section (Polyak soft target) to README + CLAUDE.md"
```

- [ ] **Step 10 — Tag**

```bash
git tag v0.2.0-u
git tag --list | tail -9
```

Attendu : `v0.1.0`, `v0.2.0-a`, `v0.2.0-x`, `v0.2.0-y`, `v0.2.0-z`, `v0.2.0-w`, `v0.2.0-v`, `v0.2.0-zy`, `v0.2.0-u`.

- [ ] **Step 11 — DoD final récap**

Print to stdout :

```
=== V2-U DoD CHECKLIST (livraison code) ===
[ ] pytest -q → 265 passed (252 + 13)
[ ] bash aether/verify_all.sh → 8 OK
[ ] smoke train_cnn_lstm_dqn_procedural.py --episodes 500 --device cuda --polyak-tau 0.005 OK
[ ] smoke V2-ZY baseline --polyak-tau 0.0 OK (regression check)
[ ] best_checkpoint .pt présent sur disque
[ ] V2-U section dans README.md + CLAUDE.md
[ ] Tag v0.2.0-u posé
[ ] Tags antérieurs intacts (v0.1.0, v0.2.0-a/x/y/z/w/v/zy)
[ ] V2-W/V2-Y/V2-ZY baselines restent verts (polyak_tau=0.0 default)
==> Phase post-livraison : benchmark V2-ZY+Polyak n=5 ep=5000 same-seed
    cible : std inter-seed < 20pp (vs 38pp baseline V2-ZY).
```

---

## Récapitulatif

- **13 tasks** réparties sur **10 phases**
- **13 nouveaux tests** (252 → 265)
- **~13 commits** sur `main`
- **Tag livraison** : `v0.2.0-u`
- **0 nouveaux fichiers code** créés
- **8 fichiers code modifiés** : 2 trainers + 3 agents + 1 config + 3 CLI scripts
- **1 nouveau fichier tests** : `tests/neural/test_polyak_update.py`
- **5 fichiers tests étendus**
- **DoD bloquante** : pytest 265 + Aether 8 OK + smoke E2E GPU + best.pt + tag + zéro régression V2-W/V2-Y/V2-ZY
- **DoD non-bloquante (objectif scientifique)** : benchmark V2-ZY+Polyak n=5, cible **std < 20 pp** (vs 38 pp baseline)
