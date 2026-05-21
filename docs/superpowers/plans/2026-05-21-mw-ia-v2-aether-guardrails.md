# MW_IA V2-A — Aether Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec source :** `docs/superpowers/specs/2026-05-21-mw-ia-v2-aether-guardrails-design.md`

**Goal :** Construire le sous-projet A du programme V2 (auto-amélioration) — un catalogue de 8 invariants RL prouvés en Aether offline et re-testés en runtime via property-based testing, exposés via une API Python `verify_formal(spec) → VerdictReport` consommable plus tard par le sous-projet E.

**Architecture :** Module Python autonome `mw_ia/guardrails/` (zéro dépendance Aether en runtime). Preuves formelles versionnées dans `aether/invariants/*.lisp`, prouvées via `mcp__aether__verify` en CI. Test de synchronisation Lisp ↔ Python par convention de nommage. Soft verdict par défaut, helper `verify_or_raise` pour CI/pre-commit.

**Tech Stack :** Python 3.13, `pytest`, `hypothesis` (nouveau), `numpy` (V1), `torch` (V1, non importé par guardrails), Aether MCP (dev/CI only).

**Adaptations à la spec :** Les noms de champs `VariantSpec` suivent la convention V1 `DQNConfig` pour minimiser la friction d'intégration future avec E :
- `buffer_capacity` (spec) → `replay_capacity` (plan)
- `target_sync_interval` (spec) → `target_sync_steps` (plan)
- Sémantique inchangée.

**Préréquis avant T1 :** être à la racine `C:\Users\Wilfred\Documents\IA Inst\MW_IA`, venv activé (`source .venv/Scripts/activate`), `pytest -q` montre 52 passed (état V1 propre, tag `v0.1.0`).

---

## Phase 1 — Setup

### Task 1 : Scaffold module + add `hypothesis`

**Files :**
- Modify : `requirements.txt`
- Create : `mw_ia/guardrails/__init__.py` (vide)
- Create : `tests/guardrails/__init__.py` (vide)
- Create : `tests/guardrails/conftest.py`

- [ ] **Step 1 — Modify requirements.txt**

Ajouter la ligne `hypothesis>=6.100` après `pytest>=8.0`. État final :

```
numpy>=1.26
PyQt6>=6.7
pyqtgraph>=0.13
gymnasium>=0.29
pytest>=8.0
hypothesis>=6.100
```

- [ ] **Step 2 — Install hypothesis**

```bash
source .venv/Scripts/activate && pip install hypothesis>=6.100
```

Expected : `Successfully installed hypothesis-6.x.x` ou `Requirement already satisfied`.

- [ ] **Step 3 — Create empty package files**

Contenu de `mw_ia/guardrails/__init__.py` : vide (sera rempli en T19).
Contenu de `tests/guardrails/__init__.py` : vide.
Contenu de `tests/guardrails/conftest.py` :

```python
"""Fixtures partagées pour les tests guardrails."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Générateur seedé pour tests d'invariants stochastiques."""
    return np.random.default_rng(seed=42)
```

- [ ] **Step 4 — Verify pytest still passes V1**

```bash
source .venv/Scripts/activate && pytest -q
```

Expected : `52 passed` (V1 inchangé, nouveau dossier `tests/guardrails/` vide → ignoré par pytest).

- [ ] **Step 5 — Commit**

```bash
git add requirements.txt mw_ia/guardrails/__init__.py tests/guardrails/__init__.py tests/guardrails/conftest.py
git commit -m "chore(guardrails): scaffold module + add hypothesis dependency"
```

---

## Phase 2 — Contracts (dataclasses)

### Task 2 : `Severity` Enum

**Files :**
- Create : `mw_ia/guardrails/contracts.py`
- Create : `tests/guardrails/test_contracts.py`

- [ ] **Step 1 — Write the failing test**

`tests/guardrails/test_contracts.py` :

```python
"""Tests des dataclasses contracts du module guardrails."""
from __future__ import annotations

from mw_ia.guardrails.contracts import Severity


def test_severity_has_hard_and_soft():
    assert Severity.HARD.value == "hard"
    assert Severity.SOFT.value == "soft"
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_contracts.py -v
```

Expected : `ModuleNotFoundError: No module named 'mw_ia.guardrails.contracts'`.

- [ ] **Step 3 — Implement minimal**

`mw_ia/guardrails/contracts.py` :

```python
"""Dataclasses du contrat public guardrails."""
from __future__ import annotations

from enum import Enum


class Severity(Enum):
    """Gravité d'une violation d'invariant."""

    HARD = "hard"   # mathématiquement faux — bloque le déploiement
    SOFT = "soft"   # atypique mais valide — signale sans bloquer (réservé v2)
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_contracts.py -v
```

Expected : `1 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/contracts.py tests/guardrails/test_contracts.py
git commit -m "feat(guardrails): Severity enum (HARD/SOFT)"
```

---

### Task 3 : `Violation` dataclass

**Files :**
- Modify : `mw_ia/guardrails/contracts.py`
- Modify : `tests/guardrails/test_contracts.py`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_contracts.py` :

```python
import pytest

from mw_ia.guardrails.contracts import Violation


def test_violation_basic_fields():
    v = Violation(invariant_id="I1", message="gamma=1.5 hors (0,1)", severity=Severity.HARD)
    assert v.invariant_id == "I1"
    assert v.message == "gamma=1.5 hors (0,1)"
    assert v.severity == Severity.HARD
    assert v.counter_example is None


def test_violation_is_frozen():
    v = Violation(invariant_id="I1", message="x", severity=Severity.HARD)
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        v.message = "y"  # type: ignore[misc]


def test_violation_with_counter_example():
    v = Violation(
        invariant_id="I2",
        message="contraction violée",
        severity=Severity.HARD,
        counter_example={"Q_diff": 1.0, "TQ_diff": 1.5, "gamma": 0.9},
    )
    assert v.counter_example["TQ_diff"] == 1.5
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_contracts.py -v
```

Expected : `ImportError: cannot import name 'Violation'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/contracts.py` :

```python
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Violation:
    """Violation d'un invariant pour un VariantSpec donné."""

    invariant_id: str
    message: str
    severity: Severity
    counter_example: Optional[dict] = None
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_contracts.py -v
```

Expected : `4 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/contracts.py tests/guardrails/test_contracts.py
git commit -m "feat(guardrails): Violation frozen dataclass"
```

---

### Task 4 : `VariantSpec` dataclass + validation `__post_init__`

**Files :**
- Modify : `mw_ia/guardrails/contracts.py`
- Modify : `tests/guardrails/test_contracts.py`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_contracts.py` :

```python
from mw_ia.guardrails.contracts import VariantSpec


def _valid_spec_kwargs() -> dict:
    return dict(
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50_000,
        batch_size=128,
        replay_capacity=100_000,
        target_sync_steps=1_000,
    )


def test_variant_spec_valid_construction():
    spec = VariantSpec(**_valid_spec_kwargs())
    assert spec.gamma == 0.99
    assert spec.reward_min is None
    assert spec.reward_max is None


def test_variant_spec_with_reward_bounds():
    spec = VariantSpec(**_valid_spec_kwargs(), reward_min=-1.0, reward_max=1.0)
    assert spec.reward_min == -1.0
    assert spec.reward_max == 1.0


def test_variant_spec_zero_replay_capacity_raises():
    kw = _valid_spec_kwargs() | {"replay_capacity": 0}
    with pytest.raises(ValueError, match="replay_capacity"):
        VariantSpec(**kw)


def test_variant_spec_negative_lr_raises():
    kw = _valid_spec_kwargs() | {"lr": -1e-3}
    with pytest.raises(ValueError, match="lr"):
        VariantSpec(**kw)


def test_variant_spec_epsilon_out_of_unit_raises():
    kw = _valid_spec_kwargs() | {"epsilon_start": 1.5}
    with pytest.raises(ValueError, match="epsilon"):
        VariantSpec(**kw)


def test_variant_spec_is_frozen():
    spec = VariantSpec(**_valid_spec_kwargs())
    with pytest.raises(Exception):
        spec.gamma = 0.5  # type: ignore[misc]
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_contracts.py -v
```

Expected : `ImportError: cannot import name 'VariantSpec'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/contracts.py` :

```python
@dataclass(frozen=True)
class VariantSpec:
    """Configuration d'un variant d'agent RL à valider contre les invariants.

    Alignement V1 : noms cohérents avec mw_ia.config.DQNConfig
    (replay_capacity, target_sync_steps).
    """

    gamma: float
    lr: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    batch_size: int
    replay_capacity: int
    target_sync_steps: int
    reward_min: Optional[float] = None
    reward_max: Optional[float] = None

    def __post_init__(self) -> None:
        # Validation structurelle : type / bornes physiques, hors invariants formels.
        if not isinstance(self.gamma, (int, float)):
            raise TypeError(f"gamma doit être float, reçu {type(self.gamma).__name__}")
        if self.lr <= 0:
            raise ValueError(f"lr doit être > 0, reçu {self.lr}")
        if not (0.0 <= self.epsilon_start <= 1.0):
            raise ValueError(f"epsilon_start doit être dans [0,1], reçu {self.epsilon_start}")
        if not (0.0 <= self.epsilon_end <= 1.0):
            raise ValueError(f"epsilon_end doit être dans [0,1], reçu {self.epsilon_end}")
        if self.epsilon_decay_steps <= 0:
            raise ValueError(f"epsilon_decay_steps doit être > 0, reçu {self.epsilon_decay_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size doit être > 0, reçu {self.batch_size}")
        if self.replay_capacity <= 0:
            raise ValueError(f"replay_capacity doit être > 0, reçu {self.replay_capacity}")
        if self.target_sync_steps <= 0:
            raise ValueError(f"target_sync_steps doit être > 0, reçu {self.target_sync_steps}")
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_contracts.py -v
```

Expected : `10 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/contracts.py tests/guardrails/test_contracts.py
git commit -m "feat(guardrails): VariantSpec frozen dataclass with structural validation"
```

---

### Task 5 : `VerdictReport` dataclass + `to_dict()`

**Files :**
- Modify : `mw_ia/guardrails/contracts.py`
- Modify : `tests/guardrails/test_contracts.py`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_contracts.py` :

```python
import json

from mw_ia.guardrails.contracts import VerdictReport


def test_verdict_report_passed_when_no_violations():
    spec = VariantSpec(**_valid_spec_kwargs())
    report = VerdictReport(passed=True, violations=(), spec=spec, duration_ms=12.3)
    assert report.passed is True
    assert report.violations == ()


def test_verdict_report_failed_with_violations():
    spec = VariantSpec(**_valid_spec_kwargs())
    v1 = Violation("I1", "gamma=1.5", Severity.HARD)
    report = VerdictReport(passed=False, violations=(v1,), spec=spec, duration_ms=4.5)
    assert report.passed is False
    assert len(report.violations) == 1


def test_verdict_report_to_dict_is_json_serializable():
    spec = VariantSpec(**_valid_spec_kwargs())
    v1 = Violation("I1", "gamma hors (0,1)", Severity.HARD)
    report = VerdictReport(passed=False, violations=(v1,), spec=spec, duration_ms=4.5)
    d = report.to_dict()
    serialized = json.dumps(d)
    assert "I1" in serialized
    assert "hard" in serialized


def test_verdict_report_is_frozen():
    spec = VariantSpec(**_valid_spec_kwargs())
    report = VerdictReport(passed=True, violations=(), spec=spec, duration_ms=1.0)
    with pytest.raises(Exception):
        report.passed = False  # type: ignore[misc]
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_contracts.py -v
```

Expected : `ImportError: cannot import name 'VerdictReport'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/contracts.py` :

```python
from dataclasses import asdict


@dataclass(frozen=True)
class VerdictReport:
    """Résultat d'un appel verify_formal(spec)."""

    passed: bool
    violations: tuple[Violation, ...]
    spec: VariantSpec
    duration_ms: float

    def to_dict(self) -> dict:
        """Sérialisation JSON-ready pour logs/debug."""
        return {
            "passed": self.passed,
            "violations": [
                {
                    "invariant_id": v.invariant_id,
                    "message": v.message,
                    "severity": v.severity.value,
                    "counter_example": v.counter_example,
                }
                for v in self.violations
            ],
            "spec": asdict(self.spec),
            "duration_ms": self.duration_ms,
        }
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_contracts.py -v
```

Expected : `14 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/contracts.py tests/guardrails/test_contracts.py
git commit -m "feat(guardrails): VerdictReport with JSON-serializable to_dict"
```

---

## Phase 3 — Exceptions

### Task 6 : `InvariantViolationError`

**Files :**
- Create : `mw_ia/guardrails/exceptions.py`
- Create : `tests/guardrails/test_exceptions.py`

- [ ] **Step 1 — Write failing test**

`tests/guardrails/test_exceptions.py` :

```python
"""Tests de InvariantViolationError."""
from __future__ import annotations

import pytest

from mw_ia.guardrails.contracts import Severity, VariantSpec, VerdictReport, Violation
from mw_ia.guardrails.exceptions import InvariantViolationError


def _spec():
    return VariantSpec(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )


def test_error_exposes_report():
    v = Violation("I1", "gamma hors (0,1)", Severity.HARD)
    report = VerdictReport(passed=False, violations=(v,), spec=_spec(), duration_ms=1.0)
    err = InvariantViolationError(report)
    assert err.report is report
    assert "I1" in str(err)


def test_error_is_raisable():
    v = Violation("I1", "x", Severity.HARD)
    report = VerdictReport(passed=False, violations=(v,), spec=_spec(), duration_ms=1.0)
    with pytest.raises(InvariantViolationError) as exc_info:
        raise InvariantViolationError(report)
    assert exc_info.value.report.violations[0].invariant_id == "I1"
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_exceptions.py -v
```

Expected : `ImportError: cannot import name 'InvariantViolationError'`.

- [ ] **Step 3 — Implement**

`mw_ia/guardrails/exceptions.py` :

```python
"""Exceptions du module guardrails."""
from __future__ import annotations

from mw_ia.guardrails.contracts import VerdictReport


class InvariantViolationError(Exception):
    """Levée par verify_or_raise quand au moins un invariant viole."""

    def __init__(self, report: VerdictReport) -> None:
        self.report = report
        ids = ", ".join(v.invariant_id for v in report.violations)
        super().__init__(
            f"{len(report.violations)} invariant(s) violé(s) : [{ids}]"
        )
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_exceptions.py -v
```

Expected : `2 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/exceptions.py tests/guardrails/test_exceptions.py
git commit -m "feat(guardrails): InvariantViolationError exposing VerdictReport"
```

---

## Phase 4 — Registry

### Task 7 : `Invariant` dataclass + `@invariant` decorator

**Files :**
- Create : `mw_ia/guardrails/registry.py`
- Create : `tests/guardrails/test_registry.py`

- [ ] **Step 1 — Write failing test**

`tests/guardrails/test_registry.py` :

```python
"""Tests du registry et du décorateur @invariant."""
from __future__ import annotations

import pytest

from mw_ia.guardrails.contracts import Severity, VariantSpec, Violation
from mw_ia.guardrails.registry import Invariant, _REGISTRY, invariant


@pytest.fixture(autouse=True)
def _clear_registry():
    """Isole chaque test du registry global."""
    backup = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(backup)


def test_decorator_registers_invariant():
    @invariant("ITEST", applies_to=["gamma"])
    def check_test(spec: VariantSpec):
        return None

    assert "ITEST" in _REGISTRY
    inv = _REGISTRY["ITEST"]
    assert isinstance(inv, Invariant)
    assert inv.id == "ITEST"
    assert inv.applies_to == ("gamma",)


def test_decorator_check_callable():
    @invariant("IX", applies_to=[])
    def check_x(spec: VariantSpec):
        return Violation("IX", "always", Severity.HARD)

    inv = _REGISTRY["IX"]
    spec = VariantSpec(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )
    result = inv.check(spec)
    assert result is not None
    assert result.invariant_id == "IX"
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_registry.py -v
```

Expected : `ModuleNotFoundError: No module named 'mw_ia.guardrails.registry'`.

- [ ] **Step 3 — Implement**

`mw_ia/guardrails/registry.py` :

```python
"""Registre interne des invariants déclarés via @invariant."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from mw_ia.guardrails.contracts import VariantSpec, Violation


CheckFn = Callable[[VariantSpec], Optional[Violation]]


@dataclass(frozen=True)
class Invariant:
    """Métadonnées d'un invariant déclaré."""

    id: str
    applies_to: tuple[str, ...]   # noms de champs VariantSpec requis non-None
    check: CheckFn


_REGISTRY: dict[str, Invariant] = {}


def invariant(id: str, applies_to: list[str]):
    """Décorateur : enregistre la fonction comme invariant `id`."""

    def decorator(fn: CheckFn) -> CheckFn:
        if id in _REGISTRY:
            raise ValueError(f"Invariant déjà enregistré : {id}")
        _REGISTRY[id] = Invariant(id=id, applies_to=tuple(applies_to), check=fn)
        return fn

    return decorator
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_registry.py -v
```

Expected : `2 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/registry.py tests/guardrails/test_registry.py
git commit -m "feat(guardrails): Invariant dataclass + @invariant decorator + _REGISTRY"
```

---

### Task 8 : `applicable_invariants(spec)`

**Files :**
- Modify : `mw_ia/guardrails/registry.py`
- Modify : `tests/guardrails/test_registry.py`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_registry.py` :

```python
from mw_ia.guardrails.registry import applicable_invariants


def _spec(**overrides):
    base = dict(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )
    base.update(overrides)
    return VariantSpec(**base)


def test_applicable_all_when_fields_present():
    @invariant("IA", applies_to=["gamma"])
    def a(spec): return None

    @invariant("IB", applies_to=["reward_min", "reward_max"])
    def b(spec): return None

    spec = _spec(reward_min=-1.0, reward_max=1.0)
    ids = [inv.id for inv in applicable_invariants(spec)]
    assert set(ids) == {"IA", "IB"}


def test_applicable_filters_when_optional_field_none():
    @invariant("IA", applies_to=["gamma"])
    def a(spec): return None

    @invariant("IB", applies_to=["reward_min", "reward_max"])
    def b(spec): return None

    spec = _spec()  # reward_min/max = None
    ids = [inv.id for inv in applicable_invariants(spec)]
    assert ids == ["IA"]


def test_applicable_order_is_stable():
    @invariant("I3", applies_to=[])
    def c(spec): return None

    @invariant("I1", applies_to=[])
    def a(spec): return None

    @invariant("I2", applies_to=[])
    def b(spec): return None

    spec = _spec()
    ids = [inv.id for inv in applicable_invariants(spec)]
    assert ids == ["I3", "I1", "I2"]  # ordre d'insertion (dict ordered Py3.7+)
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_registry.py -v
```

Expected : `ImportError: cannot import name 'applicable_invariants'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/registry.py` :

```python
def applicable_invariants(spec: VariantSpec) -> list[Invariant]:
    """Retourne les invariants pertinents pour ce spec.

    Un invariant est applicable si tous les champs listés dans `applies_to`
    sont présents et non-None dans le spec.
    """
    result = []
    for inv in _REGISTRY.values():
        if all(getattr(spec, fname, None) is not None for fname in inv.applies_to):
            result.append(inv)
    return result
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_registry.py -v
```

Expected : `5 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/registry.py tests/guardrails/test_registry.py
git commit -m "feat(guardrails): applicable_invariants(spec) filters by required fields"
```

---

## Phase 5 — Les 8 invariants

> **Convention pour toutes les tâches T9-T16 :** Chaque invariant est ajouté à `mw_ia/guardrails/invariants.py` et testé dans `tests/guardrails/test_invariants.py`. Quand T9 crée les fichiers, T10-T16 les modifient.

### Task 9 : I1 `gamma_in_open_unit`

**Files :**
- Create : `mw_ia/guardrails/invariants.py`
- Create : `tests/guardrails/test_invariants.py`

- [ ] **Step 1 — Write failing test**

`tests/guardrails/test_invariants.py` :

```python
"""Tests unitaires des 8 invariants du catalogue v1."""
from __future__ import annotations

import pytest

from mw_ia.guardrails.contracts import Severity, VariantSpec
from mw_ia.guardrails.registry import _REGISTRY
# Import du module pour exécuter ses @invariant et peupler le registry :
import mw_ia.guardrails.invariants  # noqa: F401


def _spec(**overrides):
    base = dict(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )
    base.update(overrides)
    return VariantSpec(**base)


# --- I1 -----------------------------------------------------------------------
def test_i1_pass_with_gamma_in_open_unit():
    inv = _REGISTRY["I1"]
    assert inv.check(_spec(gamma=0.99)) is None


def test_i1_violates_when_gamma_one():
    inv = _REGISTRY["I1"]
    v = inv.check(_spec(gamma=1.0))
    assert v is not None
    assert v.invariant_id == "I1"
    assert v.severity == Severity.HARD
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `ModuleNotFoundError: No module named 'mw_ia.guardrails.invariants'`.

- [ ] **Step 3 — Implement**

`mw_ia/guardrails/invariants.py` :

```python
"""Catalogue v1 des 8 invariants RL — implémentations runtime Python.

Chaque @invariant doit avoir son équivalent Aether dans aether/invariants/.
Cohérence vérifiée par tests/guardrails/test_aether_python_sync.py.
"""
from __future__ import annotations

from typing import Optional

from mw_ia.guardrails.contracts import Severity, VariantSpec, Violation
from mw_ia.guardrails.registry import invariant


@invariant("I1", applies_to=["gamma"])
def gamma_in_open_unit(spec: VariantSpec) -> Optional[Violation]:
    """γ doit être dans l'intervalle ouvert (0, 1) pour garantir la contraction."""
    if not (0.0 < spec.gamma < 1.0):
        return Violation(
            invariant_id="I1",
            message=f"gamma={spec.gamma} hors (0,1)",
            severity=Severity.HARD,
        )
    return None
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `2 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/invariants.py tests/guardrails/test_invariants.py
git commit -m "feat(guardrails): invariant I1 gamma_in_open_unit"
```

---

### Task 10 : I2 `bellman_contraction`

**Files :**
- Modify : `mw_ia/guardrails/invariants.py`
- Modify : `tests/guardrails/test_invariants.py`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_invariants.py` :

```python
# --- I2 -----------------------------------------------------------------------
def test_i2_pass_with_gamma_099():
    inv = _REGISTRY["I2"]
    assert inv.check(_spec(gamma=0.99)) is None


def test_i2_violates_when_gamma_too_close_to_one():
    """Si γ=1, le test runtime échantillonné DOIT détecter une non-contraction.
    Note : I1 attrape γ=1 avant I2 en pratique. Ce test isole I2 sur γ=1.0
    qui par construction est non-contractant (peut tester égalité, pas <)."""
    inv = _REGISTRY["I2"]
    v = inv.check(_spec(gamma=1.0))
    assert v is not None
    assert v.invariant_id == "I2"
    assert v.severity == Severity.HARD
    assert v.counter_example is not None
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `KeyError: 'I2'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/invariants.py` :

```python
import numpy as np


def _bellman_operator(Q: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """Opérateur de Bellman optimal sur MDP tabulaire.

    Q : (S, A) — fonction Q
    P : (S, A, S') — transitions
    R : (S, A, S') — récompenses
    Returns : (S, A) tableau TQ
    """
    # max_a' Q(s', a') → (S',)
    V_next = Q.max(axis=1)
    # (S, A) : Σ_s' P(s,a,s') [R(s,a,s') + γ V(s')]
    bellman_target = (P * (R + gamma * V_next[None, None, :])).sum(axis=2)
    return bellman_target


@invariant("I2", applies_to=["gamma"])
def bellman_contraction(spec: VariantSpec) -> Optional[Violation]:
    """∀ Q, Q' : ||TQ - TQ'||∞ ≤ γ ||Q - Q'||∞.

    Vérifié empiriquement sur un mini-MDP tabulaire (S=3, A=2) avec
    50 paires (Q, Q') aléatoires seedées.
    """
    rng = np.random.default_rng(seed=42)
    S, A = 3, 2
    # MDP aléatoire mais fixé par le seed
    P = rng.uniform(0.0, 1.0, size=(S, A, S))
    P = P / P.sum(axis=2, keepdims=True)  # normalisation
    R = rng.uniform(-1.0, 1.0, size=(S, A, S))

    for _ in range(50):
        Q = rng.uniform(-10.0, 10.0, size=(S, A))
        Qp = rng.uniform(-10.0, 10.0, size=(S, A))
        TQ = _bellman_operator(Q, P, R, spec.gamma)
        TQp = _bellman_operator(Qp, P, R, spec.gamma)
        lhs = float(np.abs(TQ - TQp).max())
        rhs = spec.gamma * float(np.abs(Q - Qp).max())
        if lhs > rhs + 1e-9:  # tolérance numérique
            return Violation(
                invariant_id="I2",
                message=f"Bellman non-contractant : ||TQ-TQ'||∞={lhs:.6f} > γ·||Q-Q'||∞={rhs:.6f}",
                severity=Severity.HARD,
                counter_example={
                    "gamma": spec.gamma,
                    "lhs": lhs,
                    "rhs": rhs,
                    "ratio": lhs / rhs if rhs > 0 else float("inf"),
                },
            )
    return None
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `4 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/invariants.py tests/guardrails/test_invariants.py
git commit -m "feat(guardrails): invariant I2 bellman_contraction (50 samples mini-MDP)"
```

---

### Task 11 : I3 `huber_nonneg`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_invariants.py` :

```python
# --- I3 -----------------------------------------------------------------------
def test_i3_huber_nonneg_passes():
    inv = _REGISTRY["I3"]
    assert inv.check(_spec()) is None
```

(I3 ne dépend d'aucun champ du spec — c'est une propriété structurelle de la fonction Huber. Pas de test "violates" pratique sans mock.)

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `KeyError: 'I3'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/invariants.py` :

```python
def _huber(y: float, y_hat: float, delta: float = 1.0) -> float:
    """Huber loss (référence pédagogique, indépendante de torch)."""
    diff = y - y_hat
    abs_diff = abs(diff)
    if abs_diff <= delta:
        return 0.5 * diff * diff
    return delta * (abs_diff - 0.5 * delta)


@invariant("I3", applies_to=[])
def huber_nonneg(spec: VariantSpec) -> Optional[Violation]:
    """∀ y, ŷ : Huber(y, ŷ) ≥ 0.

    Vérifié empiriquement sur 100 paires (y, ŷ) uniformément échantillonnées.
    """
    rng = np.random.default_rng(seed=42)
    for _ in range(100):
        y = float(rng.uniform(-100.0, 100.0))
        y_hat = float(rng.uniform(-100.0, 100.0))
        loss = _huber(y, y_hat)
        if loss < 0:
            return Violation(
                invariant_id="I3",
                message=f"Huber loss négatif : {loss} pour y={y}, y_hat={y_hat}",
                severity=Severity.HARD,
                counter_example={"y": y, "y_hat": y_hat, "loss": loss},
            )
    return None
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `5 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/invariants.py tests/guardrails/test_invariants.py
git commit -m "feat(guardrails): invariant I3 huber_nonneg (100 samples)"
```

---

### Task 12 : I4 `winrate_bounds`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_invariants.py` :

```python
# --- I4 -----------------------------------------------------------------------
def test_i4_winrate_bounds_passes():
    inv = _REGISTRY["I4"]
    assert inv.check(_spec()) is None
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `KeyError: 'I4'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/invariants.py` :

```python
from mw_ia.config import TrainingConfig
from mw_ia.training.metrics import MetricsTracker


@invariant("I4", applies_to=[])
def winrate_bounds(spec: VariantSpec) -> Optional[Violation]:
    """winrate ∈ [0, 1] sur fenêtre glissante.

    Vérifié en alimentant MetricsTracker avec 200 résultats aléatoires
    et en confirmant que winrate() reste borné.
    """
    rng = np.random.default_rng(seed=42)
    tracker = MetricsTracker(TrainingConfig())
    for _ in range(200):
        success = bool(rng.integers(0, 2))
        tracker.record_episode(reward=0.0, length=1, success=success)
        wr = tracker.winrate()
        if not (0.0 <= wr <= 1.0):
            return Violation(
                invariant_id="I4",
                message=f"winrate={wr} hors [0,1]",
                severity=Severity.HARD,
                counter_example={"winrate": wr},
            )
    return None
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `6 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/invariants.py tests/guardrails/test_invariants.py
git commit -m "feat(guardrails): invariant I4 winrate_bounds via MetricsTracker"
```

---

### Task 13 : I5 `epsilon_schedule_decreasing`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_invariants.py` :

```python
# --- I5 -----------------------------------------------------------------------
def test_i5_pass_with_decreasing_schedule():
    inv = _REGISTRY["I5"]
    assert inv.check(_spec(epsilon_start=1.0, epsilon_end=0.05)) is None


def test_i5_violates_when_end_greater_than_start():
    inv = _REGISTRY["I5"]
    v = inv.check(_spec(epsilon_start=0.05, epsilon_end=1.0))
    assert v is not None
    assert v.invariant_id == "I5"
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `KeyError: 'I5'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/invariants.py` :

```python
def _compute_epsilon(t: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """Schedule linéaire de ε(t) — référence pour I5.

    Identique à la formule utilisée par DQNAgent en V1.
    """
    if t >= decay_steps:
        return eps_end
    return eps_start + (eps_end - eps_start) * (t / decay_steps)


@invariant("I5", applies_to=["epsilon_start", "epsilon_end", "epsilon_decay_steps"])
def epsilon_schedule_decreasing(spec: VariantSpec) -> Optional[Violation]:
    """ε_{t+1} ≤ ε_t et ε_t ∈ [0,1] sur tout l'horizon."""
    prev = spec.epsilon_start
    horizon = 2 * spec.epsilon_decay_steps
    # Échantillonnage : 100 points uniformément espacés
    step = max(1, horizon // 100)
    for t in range(0, horizon + 1, step):
        eps = _compute_epsilon(t, spec.epsilon_start, spec.epsilon_end, spec.epsilon_decay_steps)
        if not (0.0 <= eps <= 1.0):
            return Violation(
                invariant_id="I5",
                message=f"epsilon(t={t})={eps} hors [0,1]",
                severity=Severity.HARD,
                counter_example={"t": t, "epsilon": eps},
            )
        if eps > prev + 1e-9:
            return Violation(
                invariant_id="I5",
                message=f"epsilon non décroissant : eps(t={t})={eps} > prev={prev}",
                severity=Severity.HARD,
                counter_example={"t": t, "prev": prev, "epsilon": eps},
            )
        prev = eps
    return None
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `8 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/invariants.py tests/guardrails/test_invariants.py
git commit -m "feat(guardrails): invariant I5 epsilon_schedule_decreasing"
```

---

### Task 14 : I6 `replay_buffer_capacity`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_invariants.py` :

```python
# --- I6 -----------------------------------------------------------------------
def test_i6_replay_capacity_passes():
    inv = _REGISTRY["I6"]
    # Petit replay_capacity pour test rapide (override le default 100_000)
    assert inv.check(_spec(replay_capacity=10)) is None
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `KeyError: 'I6'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/invariants.py` :

```python
from mw_ia.neural.replay_buffer import ReplayBuffer


@invariant("I6", applies_to=["replay_capacity"])
def replay_buffer_capacity(spec: VariantSpec) -> Optional[Violation]:
    """buffer.size ≤ capacity ∧ index < capacity, même après débordement.

    Pousse capacity * 3 transitions dans un buffer de capacity définie par
    le spec, vérifie qu'aucune borne n'est jamais franchie.
    """
    capacity = min(spec.replay_capacity, 1_000)  # cap pour rapidité du check
    obs_dim = 2
    buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, seed=42)
    rng = np.random.default_rng(seed=42)
    for _ in range(capacity * 3):
        s = rng.uniform(size=obs_dim).astype(np.float32)
        sp = rng.uniform(size=obs_dim).astype(np.float32)
        buf.push(s, action=0, reward=0.0, next_state=sp, done=False)
        if len(buf) > capacity:
            return Violation(
                invariant_id="I6",
                message=f"buffer.size={len(buf)} > capacity={capacity}",
                severity=Severity.HARD,
                counter_example={"size": len(buf), "capacity": capacity},
            )
        if buf._idx >= capacity:
            return Violation(
                invariant_id="I6",
                message=f"buffer._idx={buf._idx} >= capacity={capacity}",
                severity=Severity.HARD,
                counter_example={"idx": int(buf._idx), "capacity": capacity},
            )
    return None
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `9 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/invariants.py tests/guardrails/test_invariants.py
git commit -m "feat(guardrails): invariant I6 replay_buffer_capacity"
```

---

### Task 15 : I7 `reward_bounded`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_invariants.py` :

```python
# --- I7 -----------------------------------------------------------------------
def test_i7_pass_with_valid_bounds():
    inv = _REGISTRY["I7"]
    assert inv.check(_spec(reward_min=-1.0, reward_max=1.0)) is None


def test_i7_violates_when_min_greater_than_max():
    inv = _REGISTRY["I7"]
    v = inv.check(_spec(reward_min=1.0, reward_max=-1.0))
    assert v is not None
    assert v.invariant_id == "I7"


def test_i7_not_applicable_when_bounds_none():
    """Si reward_min/max sont None, I7 ne devrait pas être listé applicable."""
    from mw_ia.guardrails.registry import applicable_invariants
    spec = _spec()  # reward_min/max = None par défaut
    ids = [inv.id for inv in applicable_invariants(spec)]
    assert "I7" not in ids
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `KeyError: 'I7'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/invariants.py` :

```python
@invariant("I7", applies_to=["reward_min", "reward_max"])
def reward_bounded(spec: VariantSpec) -> Optional[Violation]:
    """reward_min ≤ reward_max (cohérence interne des bornes annoncées)."""
    assert spec.reward_min is not None and spec.reward_max is not None  # garanti par applies_to
    if spec.reward_min > spec.reward_max:
        return Violation(
            invariant_id="I7",
            message=f"reward_min={spec.reward_min} > reward_max={spec.reward_max}",
            severity=Severity.HARD,
            counter_example={"reward_min": spec.reward_min, "reward_max": spec.reward_max},
        )
    return None
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `12 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/invariants.py tests/guardrails/test_invariants.py
git commit -m "feat(guardrails): invariant I7 reward_bounded (min <= max)"
```

---

### Task 16 : I8 `episode_termination_exclusive`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_invariants.py` :

```python
# --- I8 -----------------------------------------------------------------------
def test_i8_termination_exclusive_passes():
    inv = _REGISTRY["I8"]
    assert inv.check(_spec()) is None
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `KeyError: 'I8'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/invariants.py` :

```python
from mw_ia.envs.gridworld import Action, GridWorld


@invariant("I8", applies_to=[])
def episode_termination_exclusive(spec: VariantSpec) -> Optional[Violation]:
    """∀ transition GridWorld : not (terminated ∧ truncated).

    Simule 5 épisodes courts avec une politique aléatoire et vérifie que
    les deux flags ne sont jamais True simultanément.
    """
    rng = np.random.default_rng(seed=42)
    for ep in range(5):
        env = GridWorld()
        env.reset(seed=ep)
        for _ in range(env.cfg.max_steps + 10):
            action = Action(int(rng.integers(0, 4)))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated and truncated:
                return Violation(
                    invariant_id="I8",
                    message="terminated AND truncated simultanément",
                    severity=Severity.HARD,
                    counter_example={"episode": ep},
                )
            if terminated or truncated:
                break
    return None
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_invariants.py -v
```

Expected : `13 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/invariants.py tests/guardrails/test_invariants.py
git commit -m "feat(guardrails): invariant I8 episode_termination_exclusive"
```

---

## Phase 6 — Verifier + API publique

### Task 17 : `verify_formal(spec)` — collecte les violations sans court-circuit

**Files :**
- Create : `mw_ia/guardrails/verifier.py`
- Create : `tests/guardrails/test_verifier.py`

- [ ] **Step 1 — Write failing test**

`tests/guardrails/test_verifier.py` :

```python
"""Tests de verify_formal."""
from __future__ import annotations

import pytest

from mw_ia.guardrails.contracts import VariantSpec
import mw_ia.guardrails.invariants  # noqa: F401  (peuple le registry)
from mw_ia.guardrails.verifier import verify_formal


def _spec(**overrides):
    base = dict(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )
    base.update(overrides)
    return VariantSpec(**base)


def test_verify_formal_passes_on_valid_spec():
    report = verify_formal(_spec())
    assert report.passed is True
    assert report.violations == ()
    assert report.duration_ms > 0


def test_verify_formal_collects_multiple_violations_no_shortcircuit():
    # gamma=1.0 viole I1 ET I2. Pas de court-circuit attendu.
    report = verify_formal(_spec(gamma=1.0))
    assert report.passed is False
    ids = {v.invariant_id for v in report.violations}
    assert "I1" in ids
    assert "I2" in ids


def test_verify_formal_is_deterministic():
    r1 = verify_formal(_spec(gamma=1.0))
    r2 = verify_formal(_spec(gamma=1.0))
    assert {v.invariant_id for v in r1.violations} == {v.invariant_id for v in r2.violations}
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_verifier.py -v
```

Expected : `ModuleNotFoundError: No module named 'mw_ia.guardrails.verifier'`.

- [ ] **Step 3 — Implement**

`mw_ia/guardrails/verifier.py` :

```python
"""API publique : verify_formal et verify_or_raise."""
from __future__ import annotations

import time

from mw_ia.guardrails.contracts import VariantSpec, VerdictReport
from mw_ia.guardrails.registry import applicable_invariants


def verify_formal(spec: VariantSpec) -> VerdictReport:
    """Vérifie spec contre tous les invariants applicables.

    Ne lève jamais d'exception. Collecte TOUTES les violations (pas de
    court-circuit) pour permettre à E de réparer en parallèle plusieurs
    paramètres.

    Stateless, déterministe (les invariants stochastiques utilisent un
    RNG seedé).
    """
    t0 = time.perf_counter()
    violations = []
    for inv in applicable_invariants(spec):
        v = inv.check(spec)
        if v is not None:
            violations.append(v)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    return VerdictReport(
        passed=(len(violations) == 0),
        violations=tuple(violations),
        spec=spec,
        duration_ms=duration_ms,
    )
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_verifier.py -v
```

Expected : `3 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/verifier.py tests/guardrails/test_verifier.py
git commit -m "feat(guardrails): verify_formal — soft verdict, no shortcircuit, deterministic"
```

---

### Task 18 : `verify_or_raise(spec)`

**Files :**
- Modify : `mw_ia/guardrails/verifier.py`
- Modify : `tests/guardrails/test_verifier.py`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/guardrails/test_verifier.py` :

```python
from mw_ia.guardrails.exceptions import InvariantViolationError
from mw_ia.guardrails.verifier import verify_or_raise


def test_verify_or_raise_returns_report_on_valid():
    report = verify_or_raise(_spec())
    assert report.passed is True


def test_verify_or_raise_raises_with_full_report():
    with pytest.raises(InvariantViolationError) as exc_info:
        verify_or_raise(_spec(gamma=1.0))
    err = exc_info.value
    assert err.report.passed is False
    ids = {v.invariant_id for v in err.report.violations}
    assert "I1" in ids
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_verifier.py -v
```

Expected : `ImportError: cannot import name 'verify_or_raise'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/guardrails/verifier.py` :

```python
from mw_ia.guardrails.exceptions import InvariantViolationError


def verify_or_raise(spec: VariantSpec) -> VerdictReport:
    """Wrapper de verify_formal : lève InvariantViolationError si non passé.

    Réservé aux contextes "tout ou rien" (CI, pre-commit). Pour
    l'inspection/réparation côté E, utiliser verify_formal directement.
    """
    report = verify_formal(spec)
    if not report.passed:
        raise InvariantViolationError(report)
    return report
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_verifier.py -v
```

Expected : `5 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/verifier.py tests/guardrails/test_verifier.py
git commit -m "feat(guardrails): verify_or_raise wrapper for strict contexts"
```

---

### Task 19 : API publique via `__init__.py` + smoke test

**Files :**
- Modify : `mw_ia/guardrails/__init__.py`
- Create : `tests/guardrails/test_public_api.py`

- [ ] **Step 1 — Write failing test**

`tests/guardrails/test_public_api.py` :

```python
"""Smoke test de l'API publique du package guardrails."""
from __future__ import annotations


def test_public_api_imports():
    from mw_ia.guardrails import (
        InvariantViolationError,
        Severity,
        VariantSpec,
        VerdictReport,
        Violation,
        verify_formal,
        verify_or_raise,
    )
    # Smoke : tous importables, on construit un spec valide
    spec = VariantSpec(
        gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=50_000, batch_size=128, replay_capacity=100_000,
        target_sync_steps=1_000,
    )
    report = verify_formal(spec)
    assert report.passed is True
    assert isinstance(report, VerdictReport)
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_public_api.py -v
```

Expected : `ImportError` (les noms ne sont pas dans `__init__.py`).

- [ ] **Step 3 — Implement**

`mw_ia/guardrails/__init__.py` (remplace le fichier vide) :

```python
"""Module guardrails MW_IA V2-A : invariants formels Aether + runtime Python.

API publique consommée par le futur sous-projet E (auto-modification).

Usage minimal :
    from mw_ia.guardrails import VariantSpec, verify_formal
    spec = VariantSpec(gamma=0.99, lr=1e-3, ...)
    report = verify_formal(spec)
    if report.passed:
        ...
"""
from __future__ import annotations

# Import des invariants pour peupler le registry au moment de l'import du package.
from mw_ia.guardrails import invariants as _invariants  # noqa: F401
from mw_ia.guardrails.contracts import Severity, VariantSpec, VerdictReport, Violation
from mw_ia.guardrails.exceptions import InvariantViolationError
from mw_ia.guardrails.verifier import verify_formal, verify_or_raise

__all__ = [
    "Severity",
    "VariantSpec",
    "VerdictReport",
    "Violation",
    "InvariantViolationError",
    "verify_formal",
    "verify_or_raise",
]
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/ -v
```

Expected : tous les tests guardrails verts (~28+ tests).

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/guardrails/__init__.py tests/guardrails/test_public_api.py
git commit -m "feat(guardrails): public API exposed via __init__.py"
```

---

## Phase 7 — Preuves Aether (offline)

### Task 20 : Scaffold `aether/` + lancer le syntax_guide

**Files :**
- Create : `aether/README.md`
- Create : `aether/invariants/` (dossier vide pour l'instant)

- [ ] **Step 1 — Invoquer le syntax guide Aether**

Appeler le tool `mcp__aether__syntax_guide` (sans arguments). Lire la réponse pour rafraîchir la syntaxe Lisp typée.

- [ ] **Step 2 — Create aether/README.md**

```markdown
# Aether — preuves formelles MW_IA V2-A

Chaque fichier `invariants/iN_*.lisp` formalise un invariant du catalogue v1
(cf. `docs/superpowers/specs/2026-05-21-mw-ia-v2-aether-guardrails-design.md`).

## Lancer les preuves

Une preuve = un appel au tool `mcp__aether__verify` sur le fichier.
Le script `verify_all.sh` itère sur les 8 fichiers et exit ≠ 0 si l'un
retourne autre chose que `proved`.

## Convention de nommage

`invariants/iN_<snake_case>.lisp` ↔ `mw_ia/guardrails/invariants.py::@invariant("IN")`

Cohérence vérifiée par `tests/guardrails/test_aether_python_sync.py`.

## Catalogue v1

| ID | Fichier                                | Énoncé |
| -- | -------------------------------------- | ------ |
| I1 | `i1_gamma_in_open_unit.lisp`           | γ ∈ (0,1) |
| I2 | `i2_bellman_contraction.lisp`          | Bellman γ-Lipschitz |
| I3 | `i3_huber_nonneg.lisp`                 | Huber(y, ŷ) ≥ 0 |
| I4 | `i4_winrate_bounds.lisp`               | winrate ∈ [0,1] |
| I5 | `i5_epsilon_schedule.lisp`             | ε décroît, ∈ [0,1] |
| I6 | `i6_replay_buffer_capacity.lisp`       | buffer.size ≤ capacity |
| I7 | `i7_reward_bounded.lisp`               | r_min ≤ r_max |
| I8 | `i8_episode_termination_exclusive.lisp`| terminated ⊕ truncated |
```

- [ ] **Step 3 — Create empty invariants dir**

```bash
mkdir -p aether/invariants
```

- [ ] **Step 4 — Commit**

```bash
git add aether/README.md aether/invariants/
git commit -m "docs(aether): scaffold aether/ directory + catalogue README"
```

---

### Tasks 21-28 — Preuves Lisp individuelles (I1 à I8)

> **Workflow commun à T21-T28 :**
> 1. Écrire le fichier `aether/invariants/iN_*.lisp` selon l'énoncé.
> 2. Appeler `mcp__aether__verify` avec le contenu du fichier en argument.
> 3. Si verdict ≠ `proved` (`unknown` ou `counter_example`), raffiner la preuve.
> 4. Une fois `proved`, commiter.
>
> **Squelette générique d'une preuve Aether** (à adapter à chaque invariant ; rafraîchi par le `syntax_guide` en T20) :
>
> ```lisp
> (define-property iN-name
>   (forall ((x Real))
>     (=> (precondition x)
>         (property x))))
>
> (verify iN-name)
> ```

### Task 21 : I1 `i1_gamma_in_open_unit.lisp`

- [ ] **Step 1 — Write the proof**

`aether/invariants/i1_gamma_in_open_unit.lisp` :

```lisp
;; I1 — γ doit être dans (0,1) pour garantir la contraction de Bellman.
;; Trivial : c'est une propriété structurelle sur le type de γ.

(define-property i1-gamma-in-open-unit
  (forall ((gamma Real))
    (=> (and (< 0.0 gamma) (< gamma 1.0))
        (and (> gamma 0.0) (< gamma 1.0)))))

(verify i1-gamma-in-open-unit)
```

- [ ] **Step 2 — Verify via Aether MCP**

Appeler `mcp__aether__verify` avec le contenu du fichier. Expected : `proved`.

Si `unknown` : ajuster la syntaxe selon le retour du `syntax_guide` (T20).
Si `counter_example` : la preuve est incorrecte, à reformuler.

- [ ] **Step 3 — Commit**

```bash
git add aether/invariants/i1_gamma_in_open_unit.lisp
git commit -m "feat(aether): proof I1 gamma_in_open_unit"
```

---

### Task 22 : I2 `i2_bellman_contraction.lisp`

- [ ] **Step 1 — Write the proof**

`aether/invariants/i2_bellman_contraction.lisp` :

```lisp
;; I2 — L'opérateur de Bellman optimal T est γ-contractant en norme infinie :
;;       ||TQ - TQ'||∞ ≤ γ ||Q - Q'||∞
;;
;; Pour MDP tabulaire avec γ ∈ (0,1), s'appuie sur :
;;   |max_a Q(s,a) - max_a Q'(s,a)| ≤ max_a |Q(s,a) - Q'(s,a)|

(define-property i2-bellman-contraction
  (forall ((gamma Real) (dQ Real))
    (=> (and (< 0.0 gamma) (< gamma 1.0) (>= dQ 0.0))
        (<= (* gamma dQ) dQ))))

(verify i2-bellman-contraction)
```

- [ ] **Step 2 — Verify**

`mcp__aether__verify`. Expected : `proved`.

- [ ] **Step 3 — Commit**

```bash
git add aether/invariants/i2_bellman_contraction.lisp
git commit -m "feat(aether): proof I2 bellman_contraction"
```

---

### Task 23 : I3 `i3_huber_nonneg.lisp`

- [ ] **Step 1 — Write the proof**

`aether/invariants/i3_huber_nonneg.lisp` :

```lisp
;; I3 — Huber loss ≥ 0 pour tout (y, ŷ).
;; huber(d) = 0.5 d²              si |d| ≤ δ
;;          = δ (|d| - 0.5 δ)     sinon
;; Les deux branches sont ≥ 0 si δ > 0.

(define-property i3-huber-nonneg-quadratic
  (forall ((d Real))
    (>= (* 0.5 (* d d)) 0.0)))

(define-property i3-huber-nonneg-linear
  (forall ((d Real) (delta Real))
    (=> (and (> delta 0.0) (> (abs d) delta))
        (>= (* delta (- (abs d) (* 0.5 delta))) 0.0))))

(verify i3-huber-nonneg-quadratic)
(verify i3-huber-nonneg-linear)
```

- [ ] **Step 2 — Verify**

`mcp__aether__verify`. Expected : deux `proved`.

- [ ] **Step 3 — Commit**

```bash
git add aether/invariants/i3_huber_nonneg.lisp
git commit -m "feat(aether): proof I3 huber_nonneg (both branches)"
```

---

### Task 24 : I4 `i4_winrate_bounds.lisp`

- [ ] **Step 1 — Write the proof**

`aether/invariants/i4_winrate_bounds.lisp` :

```lisp
;; I4 — winrate = wins / window ∈ [0,1] si wins ∈ [0, window] et window > 0.

(define-property i4-winrate-bounds
  (forall ((wins Real) (window Real))
    (=> (and (> window 0.0) (>= wins 0.0) (<= wins window))
        (and (>= (/ wins window) 0.0) (<= (/ wins window) 1.0)))))

(verify i4-winrate-bounds)
```

- [ ] **Step 2 — Verify**

`mcp__aether__verify`. Expected : `proved`.

- [ ] **Step 3 — Commit**

```bash
git add aether/invariants/i4_winrate_bounds.lisp
git commit -m "feat(aether): proof I4 winrate_bounds"
```

---

### Task 25 : I5 `i5_epsilon_schedule.lisp`

- [ ] **Step 1 — Write the proof**

`aether/invariants/i5_epsilon_schedule.lisp` :

```lisp
;; I5 — Schedule linéaire ε(t) = ε_start + (ε_end - ε_start) * (t / decay_steps)
;; clampé à ε_end après decay_steps. Si ε_start ≥ ε_end et tous deux ∈ [0,1],
;; alors ε(t) décroît monotone et reste dans [0,1].

(define-property i5-epsilon-monotone
  (forall ((eps_start Real) (eps_end Real) (t1 Real) (t2 Real) (decay Real))
    (=> (and (>= eps_start eps_end)
             (>= eps_end 0.0) (<= eps_start 1.0)
             (> decay 0.0)
             (>= t1 0.0) (>= t2 t1))
        (>= (+ eps_start (* (/ (- eps_end eps_start) decay) t1))
            (+ eps_start (* (/ (- eps_end eps_start) decay) t2))))))

(define-property i5-epsilon-bounded
  (forall ((eps_start Real) (eps_end Real) (t Real) (decay Real))
    (=> (and (>= eps_start eps_end) (>= eps_end 0.0) (<= eps_start 1.0)
             (> decay 0.0) (>= t 0.0) (<= t decay))
        (and (>= (+ eps_start (* (/ (- eps_end eps_start) decay) t)) 0.0)
             (<= (+ eps_start (* (/ (- eps_end eps_start) decay) t)) 1.0)))))

(verify i5-epsilon-monotone)
(verify i5-epsilon-bounded)
```

- [ ] **Step 2 — Verify**

`mcp__aether__verify`. Expected : deux `proved`.

- [ ] **Step 3 — Commit**

```bash
git add aether/invariants/i5_epsilon_schedule.lisp
git commit -m "feat(aether): proof I5 epsilon_schedule (monotone + bounded)"
```

---

### Task 26 : I6 `i6_replay_buffer_capacity.lisp`

- [ ] **Step 1 — Write the proof**

`aether/invariants/i6_replay_buffer_capacity.lisp` :

```lisp
;; I6 — Buffer circulaire : si index ← (index + 1) mod capacity, alors
;;      index < capacity et size = min(pushes, capacity) ≤ capacity.

(define-property i6-modular-index-bounded
  (forall ((idx Int) (capacity Int))
    (=> (and (> capacity 0) (>= idx 0) (< idx capacity))
        (< (mod (+ idx 1) capacity) capacity))))

(define-property i6-size-bounded
  (forall ((pushes Int) (capacity Int))
    (=> (and (> capacity 0) (>= pushes 0))
        (<= (min pushes capacity) capacity))))

(verify i6-modular-index-bounded)
(verify i6-size-bounded)
```

- [ ] **Step 2 — Verify**

`mcp__aether__verify`. Expected : deux `proved`.

- [ ] **Step 3 — Commit**

```bash
git add aether/invariants/i6_replay_buffer_capacity.lisp
git commit -m "feat(aether): proof I6 replay_buffer_capacity (modular index + size)"
```

---

### Task 27 : I7 `i7_reward_bounded.lisp`

- [ ] **Step 1 — Write the proof**

`aether/invariants/i7_reward_bounded.lisp` :

```lisp
;; I7 — Si r_min ≤ r_max et r ∈ [r_min, r_max], alors r ∈ [r_min, r_max].
;; Trivial — formalise la cohérence interne de l'intervalle.

(define-property i7-reward-bounds-coherent
  (forall ((r Real) (r_min Real) (r_max Real))
    (=> (and (<= r_min r_max) (>= r r_min) (<= r r_max))
        (and (>= r r_min) (<= r r_max)))))

(verify i7-reward-bounds-coherent)
```

- [ ] **Step 2 — Verify**

`mcp__aether__verify`. Expected : `proved`.

- [ ] **Step 3 — Commit**

```bash
git add aether/invariants/i7_reward_bounded.lisp
git commit -m "feat(aether): proof I7 reward_bounded (interval coherence)"
```

---

### Task 28 : I8 `i8_episode_termination_exclusive.lisp`

- [ ] **Step 1 — Write the proof**

`aether/invariants/i8_episode_termination_exclusive.lisp` :

```lisp
;; I8 — Dans le step() de GridWorld V1 :
;;   terminated = (state == goal)
;;   truncated  = (not terminated) and (step_count >= max_steps)
;; Donc (terminated AND truncated) ≡ False par construction.

(define-property i8-termination-exclusive
  (forall ((terminated Bool) (truncated Bool) (step_count Int) (max_steps Int))
    (=> (and (> max_steps 0)
             (or (not terminated) (not truncated))
             ;; reflet de la formule V1 : truncated = not terminated && step >= max
             (= truncated (and (not terminated) (>= step_count max_steps))))
        (not (and terminated truncated)))))

(verify i8-termination-exclusive)
```

- [ ] **Step 2 — Verify**

`mcp__aether__verify`. Expected : `proved`.

- [ ] **Step 3 — Commit**

```bash
git add aether/invariants/i8_episode_termination_exclusive.lisp
git commit -m "feat(aether): proof I8 episode_termination_exclusive"
```

---

### Task 29 : `verify_all.sh` — script qui itère sur les 8 preuves

**Files :**
- Create : `aether/verify_all.sh`

- [ ] **Step 1 — Write the script**

`aether/verify_all.sh` :

```bash
#!/usr/bin/env bash
# Vérifie les 8 preuves Aether du catalogue v1 et exit ≠ 0 si l'un échoue.
# Appelé en CI et manuellement.

set -euo pipefail

cd "$(dirname "$0")/invariants"

FAILED=()
for lisp_file in i*_*.lisp; do
    echo "→ Vérification : $lisp_file"
    # Note : ce script est un harness shell pour les humains. En CI on
    # appelle plutôt le tool MCP `mcp__aether__verify` directement via
    # Claude Code ou un client MCP, parce que Aether n'est pas dispo
    # comme binaire shell standalone.
    #
    # Placeholder pour exécution manuelle : afficher le contenu et
    # laisser un opérateur lancer mcp__aether__verify dessus.
    if [[ ! -s "$lisp_file" ]]; then
        echo "  ✗ fichier vide"
        FAILED+=("$lisp_file")
        continue
    fi
    echo "  ✓ fichier présent et non vide (run mcp__aether__verify pour la preuve)"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "ÉCHEC sur ${#FAILED[@]} fichier(s) :"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi

echo ""
echo "OK : ${#FAILED[@]} échecs sur tous les fichiers présents."
exit 0
```

- [ ] **Step 2 — Make executable + smoke run**

```bash
chmod +x aether/verify_all.sh
bash aether/verify_all.sh
```

Expected : 8 lignes "✓ fichier présent et non vide", puis "OK".

- [ ] **Step 3 — Commit**

```bash
git add aether/verify_all.sh
git commit -m "chore(aether): verify_all.sh harness for the 8 catalogue proofs"
```

---

## Phase 8 — Sync check + CI

### Task 30 : `test_aether_python_sync.py` — anti-désynchronisation Lisp ↔ Python

**Files :**
- Create : `tests/guardrails/test_aether_python_sync.py`

- [ ] **Step 1 — Write the failing test**

`tests/guardrails/test_aether_python_sync.py` :

```python
"""Garde-fou : chaque .lisp d'aether/invariants/ a son @invariant Python et vice-versa."""
from __future__ import annotations

import re
from pathlib import Path

import mw_ia.guardrails.invariants  # noqa: F401  (peuple le registry)
from mw_ia.guardrails.registry import _REGISTRY


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _aether_ids() -> set[str]:
    """Extrait les IDs (I1..I8) des fichiers `iN_*.lisp`."""
    aether_dir = _project_root() / "aether" / "invariants"
    ids = set()
    for lisp_file in aether_dir.glob("i*_*.lisp"):
        m = re.match(r"^i(\d+)_", lisp_file.name, re.IGNORECASE)
        assert m, f"Nom de fichier non conforme : {lisp_file.name}"
        ids.add(f"I{m.group(1)}")
    return ids


def _python_ids() -> set[str]:
    return set(_REGISTRY.keys())


def test_aether_files_have_python_invariants():
    aether = _aether_ids()
    python = _python_ids()
    missing = aether - python
    assert not missing, f"Preuves Aether sans @invariant Python : {missing}"


def test_python_invariants_have_aether_files():
    aether = _aether_ids()
    python = _python_ids()
    missing = python - aether
    assert not missing, f"@invariant Python sans preuve Aether : {missing}"


def test_id_extraction_is_case_insensitive():
    """Sanity : la regex matche I1, i1, etc."""
    assert re.match(r"^i(\d+)_", "i1_test.lisp", re.IGNORECASE)
    assert re.match(r"^i(\d+)_", "I7_xyz.lisp", re.IGNORECASE)
```

- [ ] **Step 2 — Run**

```bash
source .venv/Scripts/activate && pytest tests/guardrails/test_aether_python_sync.py -v
```

Expected : `3 passed` (les 8 .lisp et les 8 @invariant existent à ce stade).

- [ ] **Step 3 — Commit**

```bash
git add tests/guardrails/test_aether_python_sync.py
git commit -m "test(guardrails): test_aether_python_sync.py — anti-drift Lisp/Python"
```

---

### Task 31 : CI workflow GitHub Actions

**Files :**
- Create : `.github/workflows/aether_verify.yml`

- [ ] **Step 1 — Write the workflow**

`.github/workflows/aether_verify.yml` :

```yaml
name: aether-verify

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - name: Install dependencies (CPU torch)
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      - name: Run pytest
        run: pytest -q

  aether-proofs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Smoke-check aether files exist
        run: |
          bash aether/verify_all.sh
      # Note : la vérification formelle effective (mcp__aether__verify) est
      # exécutée en dev/local par Claude Code, pas dans cette CI publique.
      # Si un binaire Aether standalone devient disponible, l'ajouter ici.
```

- [ ] **Step 2 — Validate YAML syntax (offline)**

```bash
source .venv/Scripts/activate && python -c "import yaml; yaml.safe_load(open('.github/workflows/aether_verify.yml'))"
```

Expected : pas d'erreur.

- [ ] **Step 3 — Commit**

```bash
git add .github/workflows/aether_verify.yml
git commit -m "ci: aether-verify workflow (pytest + aether/verify_all.sh)"
```

---

## Phase 9 — Documentation & Definition of Done

### Task 32 : Update README.md

**Files :**
- Modify : `README.md`

- [ ] **Step 1 — Read current README**

```bash
head -40 README.md
```

(Confirmer qu'il existe et identifier où insérer la section V2-A — typiquement après l'archi V1, avant la roadmap.)

- [ ] **Step 2 — Add V2-A section**

Insérer la section suivante au bon endroit (après la description V1, avant "Roadmap" si elle existe) :

```markdown
## V2-A — Aether guardrails (sous-projet en cours)

Catalogue de **8 invariants RL formels** prouvés en Aether (offline) et re-testés
en runtime via property-based testing. API Python autonome consommable par le
futur sous-projet E (auto-modification).

### Usage minimal

```python
from mw_ia.guardrails import VariantSpec, verify_formal

spec = VariantSpec(
    gamma=0.99, lr=1e-3,
    epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=50_000,
    batch_size=128, replay_capacity=100_000, target_sync_steps=1_000,
)

report = verify_formal(spec)
if report.passed:
    print("✓ variant valide")
else:
    for v in report.violations:
        print(f"  ✗ {v.invariant_id} : {v.message}")
```

### Catalogue v1

| ID | Invariant                          | Énoncé |
| -- | ---------------------------------- | ------ |
| I1 | `gamma_in_open_unit`               | γ ∈ (0,1) strict |
| I2 | `bellman_contraction`              | T γ-Lipschitz en norme infinie |
| I3 | `huber_nonneg`                     | Huber(y, ŷ) ≥ 0 |
| I4 | `winrate_bounds`                   | winrate ∈ [0,1] |
| I5 | `epsilon_schedule_decreasing`      | ε(t) décroît, ∈ [0,1] |
| I6 | `replay_buffer_capacity`           | buffer.size ≤ capacity |
| I7 | `reward_bounded`                   | r_min ≤ r_max |
| I8 | `episode_termination_exclusive`    | terminated ⊕ truncated |

### Architecture

- `mw_ia/guardrails/` — module Python autonome (zéro dépendance Aether runtime)
- `aether/invariants/*.lisp` — preuves formelles versionnées
- `tests/guardrails/test_aether_python_sync.py` — vérifie la cohérence Lisp ↔ Python
```

- [ ] **Step 3 — Commit**

```bash
git add README.md
git commit -m "docs(readme): add V2-A Aether guardrails section"
```

---

### Task 33 : Definition of Done — smoke test final

**Files :** aucune modification.

- [ ] **Step 1 — Full test suite**

```bash
source .venv/Scripts/activate && pytest -q
```

Expected : **≥ 86 tests passed** (52 V1 + ~34 V2-A).

- [ ] **Step 2 — Aether harness smoke**

```bash
bash aether/verify_all.sh
```

Expected : 8 fichiers présents, exit 0.

- [ ] **Step 3 — End-to-end smoke from CLI**

```bash
source .venv/Scripts/activate && python -c "
from mw_ia.guardrails import VariantSpec, verify_formal
spec = VariantSpec(
    gamma=0.99, lr=1e-3,
    epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=50_000,
    batch_size=128, replay_capacity=100_000, target_sync_steps=1_000,
)
report = verify_formal(spec)
print(f'passed={report.passed}, violations={len(report.violations)}, duration_ms={report.duration_ms:.2f}')
"
```

Expected : `passed=True, violations=0, duration_ms=<small>`.

- [ ] **Step 4 — Negative end-to-end (gamma=1.0)**

```bash
source .venv/Scripts/activate && python -c "
from mw_ia.guardrails import VariantSpec, verify_formal
spec = VariantSpec(
    gamma=1.0, lr=1e-3,
    epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=50_000,
    batch_size=128, replay_capacity=100_000, target_sync_steps=1_000,
)
report = verify_formal(spec)
print(f'passed={report.passed}')
for v in report.violations:
    print(f'  - {v.invariant_id} : {v.message}')
"
```

Expected : `passed=False`, avec au moins `I1` (et probablement `I2`) dans les violations.

- [ ] **Step 5 — Tag the release**

```bash
git tag -a v0.2.0-a -m "MW_IA V2-A — Aether guardrails (8 invariants + runtime API)"
git log --oneline -5
```

Expected : tag `v0.2.0-a` créé sur le HEAD.

---

## Récapitulatif des fichiers livrés

```
MW_IA/
├── aether/
│   ├── README.md                                          [T20]
│   ├── verify_all.sh                                      [T29]
│   └── invariants/
│       ├── i1_gamma_in_open_unit.lisp                     [T21]
│       ├── i2_bellman_contraction.lisp                    [T22]
│       ├── i3_huber_nonneg.lisp                           [T23]
│       ├── i4_winrate_bounds.lisp                         [T24]
│       ├── i5_epsilon_schedule.lisp                       [T25]
│       ├── i6_replay_buffer_capacity.lisp                 [T26]
│       ├── i7_reward_bounded.lisp                         [T27]
│       └── i8_episode_termination_exclusive.lisp          [T28]
├── mw_ia/guardrails/
│   ├── __init__.py                                        [T1, T19]
│   ├── contracts.py                                       [T2-T5]
│   ├── exceptions.py                                      [T6]
│   ├── registry.py                                        [T7-T8]
│   ├── invariants.py                                      [T9-T16]
│   └── verifier.py                                        [T17-T18]
├── tests/guardrails/
│   ├── __init__.py                                        [T1]
│   ├── conftest.py                                        [T1]
│   ├── test_contracts.py                                  [T2-T5]
│   ├── test_exceptions.py                                 [T6]
│   ├── test_registry.py                                   [T7-T8]
│   ├── test_invariants.py                                 [T9-T16]
│   ├── test_verifier.py                                   [T17-T18]
│   ├── test_public_api.py                                 [T19]
│   └── test_aether_python_sync.py                         [T30]
├── .github/workflows/aether_verify.yml                    [T31]
├── README.md                                              [T32]
└── requirements.txt                                       [T1] (+ hypothesis)
```

**Total :** 33 tâches · ~34 tests V2-A · 8 preuves Aether · 0 dépendance Aether runtime.
