# MW_IA V2-A — Aether Guardrails Design Document

**Date :** 2026-05-21
**Statut :** APPROUVÉ (brainstorming conversationnel, sous-projet **A** du programme V2 auto-amélioration)
**Auteur conversation :** Wilfred + Claude (Opus 4.7)
**Spec parente :** [`2026-05-21-mw-ia-rl-design.md`](./2026-05-21-mw-ia-rl-design.md) (V1, tag `v0.1.0`)

---

## 0. Décisions verrouillées en brainstorming

1. **Décomposition V2 en 6 sous-projets** (A → F), exécutés séquentiellement avec leur propre cycle spec → plan → impl :
   - **A** — Aether guardrails formels (ce document)
   - **B** — Mémoire persistante cross-session
   - **C** — Évaluateur de politique self-supervisé
   - **D** — Continual learning (EWC, rehearsal)
   - **E** — Auto-modification (proposer / tester / déployer variants)
   - **F** — Meta-RL (MAML / RL² / context-based), reportable en V3
2. **Ambition A = Niveau 3** : catalogue Aether + CI gate + module Python runtime exposant une API `verify_formal(spec) → VerdictReport` consommable par E.
3. **Périmètre A = formel uniquement.** Les garde-fous empiriques (non-régression sur suite de benchs) sont reportés au sous-projet C.
4. **Catalogue v1 = 8 invariants (I1-I8).** Voir §3.
5. **Architecture = option β.** Aether prouve une fois pour toutes en CI (`aether/*.lisp` versionnés) ; le runtime Python re-teste les mêmes énoncés via `hypothesis`. **Aether n'est jamais une dépendance runtime.**
6. **API = soft verdict.** `verify_formal(spec)` retourne toujours un `VerdictReport` (jamais d'exception sur violation). Helper `verify_or_raise(spec)` pour les contextes "tout ou rien" (CI, pre-commit).

---

## 1. Objectif

Doter MW_IA V2 d'une couche de **garde-fous formels** transverse, capable de juger si une configuration RL (hyperparamètres + référence d'architecture + reward shaping) respecte un ensemble d'invariants mathématiques prouvés en Aether.

Cette couche sera consommée plus tard par le sous-projet E (auto-modification), qui appellera `guardrails.verify_formal(variant)` **avant** chaque déploiement d'un variant proposé. E recevra un verdict structuré qu'il pourra inspecter, utiliser pour réparer un variant (clamp γ, etc.), ou rejeter.

**Hors objectif** : juger la qualité empirique d'un variant (→ C), modifier l'architecture du QNetwork de la V1 (rien à toucher), interfacer avec la GUI (V2-A reste headless).

---

## 2. Principe en deux temps

### 2.1 Au dev / CI — preuve formelle (offline)

Chaque invariant `iN` est écrit en Lisp typé Aether dans `aether/invariants/iN_xxx.lisp` et **prouvé une fois pour toutes** via `mcp__aether__verify`. La preuve est rejouée en CI par `aether/verify_all.sh`. Un invariant qui passe de `proved` à `unknown` ou `counter_example` bloque le merge.

### 2.2 Au runtime — re-test via property-based (online)

Le module Python `mw_ia/guardrails/` réimplémente les **mêmes énoncés** sous forme de fonctions Python décorées `@invariant("I1")`. Quand un invariant doit échantillonner (ex : Q, Q' aléatoires pour vérifier la contraction Bellman), il embarque sa propre stratégie `hypothesis`. L'appel `verify_formal(spec)` lance la batterie sur les valeurs concrètes du `VariantSpec` et agrège les résultats.

### 2.3 Pourquoi cette dualité

| Critère | Aether (offline) | Python + Hypothesis (runtime) |
|---|---|---|
| Garantie | Preuve formelle (mathématique) | Échantillonnage statistique |
| Vitesse | Lent (subprocess Lisp) | ≤ 50 ms |
| Dépendance | Outil de dev externe | Python pur (`hypothesis` dans `requirements.txt`) |
| Quand | Au dev / CI / une fois par invariant | À chaque appel d'E |
| Détecte | Erreurs structurelles | Variants paramétrés concrets qui violent |

Aether garantit que **l'énoncé est correct** ; Hypothesis garantit que **le variant concret le respecte**. Le test `test_aether_python_sync.py` garantit qu'on ne dérive pas entre les deux.

---

## 3. Catalogue v1 — Les 8 invariants

| ID | Nom | Énoncé informel | Catégorie CLAUDE.md | Stochastique ? |
|---|---|---|---|---|
| **I1** | `gamma_in_open_unit` | γ ∈ (0, 1) strict | (c) contraction | Non |
| **I2** | `bellman_contraction` | ∀ Q, Q' : ‖TQ − TQ'‖∞ ≤ γ · ‖Q − Q'‖∞ | (c) contraction | Oui (Q, Q' échantillonnés) |
| **I3** | `huber_nonneg` | ∀ y, ŷ : `huber(y, ŷ) ≥ 0` | (b) positivité loss | Oui (y, ŷ échantillonnés) |
| **I4** | `winrate_bounds` | winrate ∈ [0, 1] sur fenêtre glissante | (a) bornes winrate | Non |
| **I5** | `epsilon_schedule_decreasing` | ε_{t+1} ≤ ε_t et ε_t ∈ [0, 1] | (d) domaine | Non |
| **I6** | `replay_buffer_capacity` | buffer.size ≤ capacity ∧ index < capacity | (d) domaine | Non |
| **I7** | `reward_bounded` | ∀ transition : r ∈ [r_min, r_max] connus | (d) domaine | Non |
| **I8** | `episode_termination_exclusive` | terminated ⊕ truncated (jamais les deux) | (d) domaine | Non |

**Promus depuis la V1** : I2 (depuis `bellman_update`), I4 (extrait de `level_of`).

**Explicitement hors v1 (roadmap)** :
- Robbins-Monro sur learning rate α : Σ α_t = ∞ ∧ Σ α_t² < ∞ (raisonnement sur séries, complexe en Aether)
- Convergence asymptotique Q-Learning (preuve trop élaborée)
- Reward shaping potential-based (Ng et al.) — préservation politique optimale (pertinent quand E mutera le reward shaping)
- Monotonie de Value Iteration sous initialisation optimiste (redondant avec I2)

---

## 4. Arborescence & composants

```
MW_IA/
├── aether/                                # NOUVEAU — preuves formelles versionnées
│   ├── README.md                          # Comment lancer les preuves, mapping vers Python
│   ├── invariants/
│   │   ├── i1_gamma_in_open_unit.lisp
│   │   ├── i2_bellman_contraction.lisp
│   │   ├── i3_huber_nonneg.lisp
│   │   ├── i4_winrate_bounds.lisp
│   │   ├── i5_epsilon_schedule.lisp
│   │   ├── i6_replay_buffer_capacity.lisp
│   │   ├── i7_reward_bounded.lisp
│   │   └── i8_episode_termination_exclusive.lisp
│   └── verify_all.sh                      # boucle mcp__aether__verify
│
├── mw_ia/
│   ├── guardrails/                        # NOUVEAU — module runtime autonome
│   │   ├── __init__.py                    # exporte API publique
│   │   ├── contracts.py                   # VariantSpec, Violation, VerdictReport, Severity
│   │   ├── invariants.py                  # 8 fonctions @invariant("I1")..."I8"
│   │   ├── registry.py                    # collecte + applicable_invariants(spec)
│   │   ├── verifier.py                    # verify_formal + verify_or_raise
│   │   └── exceptions.py                  # InvariantViolationError
│   └── ... (V1 inchangé)
│
├── tests/
│   ├── guardrails/                        # NOUVEAU — ~30-35 tests
│   │   ├── test_contracts.py              # 6 tests
│   │   ├── test_invariants.py             # 16 tests (2 par invariant : pass/fail)
│   │   ├── test_registry.py               # 4 tests
│   │   ├── test_verifier.py               # 5 tests
│   │   └── test_aether_python_sync.py     # 3 tests
│   └── ... (V1 inchangé)
│
└── .github/workflows/                     # NOUVEAU (alternative : .pre-commit-config.yaml)
    └── aether_verify.yml                  # CI : verify_all.sh + pytest tests/guardrails/
```

### 4.1 API publique (`mw_ia/guardrails/__init__.py`)

```python
from mw_ia.guardrails.contracts import VariantSpec, Violation, VerdictReport, Severity
from mw_ia.guardrails.verifier import verify_formal, verify_or_raise
from mw_ia.guardrails.exceptions import InvariantViolationError

__all__ = [
    "VariantSpec", "Violation", "VerdictReport", "Severity",
    "verify_formal", "verify_or_raise",
    "InvariantViolationError",
]
```

### 4.2 Dataclasses (`contracts.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class Severity(Enum):
    HARD = "hard"   # mathématiquement faux — bloque le déploiement
    SOFT = "soft"   # atypique mais valide — signale sans bloquer (réservé v2)

@dataclass(frozen=True)
class VariantSpec:
    # hyperparams DQN
    gamma: float
    lr: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    batch_size: int
    buffer_capacity: int
    target_sync_interval: int
    # reward shaping (optionnel — None = invariant I7 ignoré)
    reward_min: Optional[float] = None
    reward_max: Optional[float] = None
    # extension future : network: Optional[NetworkSpec] = None

@dataclass(frozen=True)
class Violation:
    invariant_id: str
    message: str
    severity: Severity
    counter_example: Optional[dict] = None  # pour I2, I3 stochastiques

@dataclass(frozen=True)
class VerdictReport:
    passed: bool
    violations: tuple[Violation, ...]
    spec: VariantSpec
    duration_ms: float

    def to_dict(self) -> dict: ...   # sérialisation JSON pour debug / logs E
```

### 4.3 Décorateur invariant (`invariants.py`)

```python
from typing import Callable, Optional
from mw_ia.guardrails.contracts import VariantSpec, Violation

# registre interne peuplé par le décorateur
_REGISTRY: dict[str, "Invariant"] = {}

def invariant(id: str, applies_to: list[str]):
    """Décorateur : enregistre la fonction comme invariant `id`,
    applicable seulement si tous les champs `applies_to` sont non-None."""
    def decorator(fn: Callable[[VariantSpec], Optional[Violation]]):
        _REGISTRY[id] = Invariant(id=id, applies_to=applies_to, check=fn)
        return fn
    return decorator

@invariant("I1", applies_to=["gamma"])
def gamma_in_open_unit(spec: VariantSpec) -> Optional[Violation]:
    if not (0.0 < spec.gamma < 1.0):
        return Violation("I1", f"gamma={spec.gamma} hors (0,1)", Severity.HARD)
    return None

# ... I2 à I8
```

### 4.4 Verifier (`verifier.py`)

```python
import time
from mw_ia.guardrails.contracts import VariantSpec, VerdictReport
from mw_ia.guardrails.registry import applicable_invariants
from mw_ia.guardrails.exceptions import InvariantViolationError

def verify_formal(spec: VariantSpec, seed: int = 42) -> VerdictReport:
    """Vérifie spec contre tous les invariants applicables. Ne lève jamais.
    Pas de court-circuit : collecte TOUTES les violations."""
    t0 = time.perf_counter()
    violations = []
    for inv in applicable_invariants(spec):
        v = inv.check(spec)  # avec seed pour les invariants stochastiques
        if v is not None:
            violations.append(v)
    duration_ms = (time.perf_counter() - t0) * 1000
    return VerdictReport(
        passed=(len(violations) == 0),
        violations=tuple(violations),
        spec=spec,
        duration_ms=duration_ms,
    )

def verify_or_raise(spec: VariantSpec, seed: int = 42) -> VerdictReport:
    """Wrapper : lève InvariantViolationError(report) si non passé."""
    report = verify_formal(spec, seed=seed)
    if not report.passed:
        raise InvariantViolationError(report)
    return report
```

---

## 5. Flux de données

### 5.1 Flux dev / CI — preuve formelle

```
dev modifie aether/invariants/iN_*.lisp
    │
    ▼
aether/verify_all.sh appelle mcp__aether__verify par fichier
    │
    ▼
verdict Aether : proved / counter_example / unknown
    │
    ▼
exit 0 si tous "proved", sinon exit 1 → CI bloque le merge
```

### 5.2 Flux dev / CI — sync check

```
pytest tests/guardrails/test_aether_python_sync.py
    │
    ▼
parse aether/invariants/*.lisp  ∪  inspect mw_ia/guardrails/invariants.py
    │
    ▼
assert ensemble des IDs Lisp == ensemble des IDs @invariant Python
    │
    ▼
fail-fast si désync
```

### 5.3 Flux runtime — appel par E

```
E.propose_variant() construit VariantSpec(gamma=0.99, lr=2e-3, ...)
    │
    ▼
guardrails.verify_formal(spec)
    │
    ▼
registry.applicable_invariants(spec) → [I1, ..., I8]
    │
    ▼ (séquentiel, pas de court-circuit)
pour chaque invariant : inv.check(spec) → None ou Violation
    │
    ▼
VerdictReport(passed, violations, spec, duration_ms)
    │
    ▼
E inspecte :
    - passed=True  → déploie le variant
    - passed=False → répare (ex : clamp γ → 0.99) et re-verify, ou rejette
```

**Caractéristiques** :
- **Stateless** : `verify_formal` est pure, aucun side-effect, ne touche pas le disque, ne log nulle part.
- **Déterministe** : par défaut `seed=42`, même `VariantSpec` → même verdict bit-à-bit.
- **Rapide** : cible ≤ 50 ms par appel sur les 8 invariants.

---

## 6. Gestion d'erreur

### 6.1 Famille 1 — Violation d'invariant (cas attendu)

**Pas une erreur Python.** Encapsulé dans `VerdictReport`. Chaque `Violation` porte :
- `invariant_id` : "I1"..."I8"
- `message` : texte humain
- `severity` : `HARD` par défaut en v1 ; `SOFT` réservé pour v2 du module
- `counter_example` : `dict | None` (rempli pour I2, I3 quand Hypothesis trouve un contre-exemple)

`verify_or_raise` lève `InvariantViolationError(report)` qui expose le rapport **complet** via `.report` — pas un seul message.

### 6.2 Famille 2 — Spec malformé (erreur d'usage)

Validation dans `VariantSpec.__post_init__` : type incorrect, clé manquante, valeurs structurellement invalides (`buffer_capacity ≤ 0`, etc.) → `ValueError` / `TypeError` direct. **Pas de soft verdict** sur un spec qui ne tient pas debout.

### 6.3 Famille 3 — Erreur Aether en CI

Si `mcp__aether__verify` retourne `unknown` : CI échoue, message log clair. Pas d'impact runtime. Convention : un invariant doit être soit `proved`, soit retiré, soit explicitement marqué `@invariant.unproven_yet` (skip CI). **v1 vise zéro `unproven_yet`.**

### 6.4 Cas limites couverts par tests

| Cas | Comportement attendu |
|---|---|
| `VariantSpec(gamma=1.0)` | I1 viole (intervalle ouvert) |
| `reward_min=None, reward_max=None` | I7 ignoré (non applicable), pas de violation |
| `buffer_capacity=0` | Levée `ValueError` à la construction (spec malformé, famille 2) |
| Tous invariants violent en même temps | `VerdictReport(passed=False, violations=[V1..V8])` — pas seulement la 1ère |
| 1000 appels `verify_formal` | Pas de fuite mémoire, durée moyenne ≤ 50 ms |
| `seed` identique deux fois | Verdicts bit-à-bit identiques |

---

## 7. Stratégie de tests (TDD strict)

5 fichiers, **~34 tests** au total.

### 7.1 `test_contracts.py` (6 tests)
- `VariantSpec` valide se construit
- `VariantSpec(buffer_capacity=0)` lève `ValueError`
- `Violation` est frozen, comparable
- `VerdictReport.passed = (len(violations) == 0)`
- `VerdictReport.to_dict()` produit JSON sérialisable
- `Severity.HARD` et `SOFT` exposés

### 7.2 `test_invariants.py` (16 tests, 2 par invariant)
Pour chaque `IN` : un cas qui passe, un cas qui viole. I2 et I3 utilisent `hypothesis` seedé.

### 7.3 `test_registry.py` (4 tests)
- Registry contient les 8 IDs au moment de l'import
- `applicable_invariants(spec)` filtre quand un `applies_to` est `None`
- IDs uniques
- Ordre stable (pour reports reproductibles)

### 7.4 `test_verifier.py` (5 tests)
- `verify_formal(valid)` → `passed=True`, `duration_ms > 0`
- `verify_formal(spec_violant_3)` → `len(violations) == 3` (pas de court-circuit)
- Déterminisme : deux appels identiques → reports identiques
- `verify_or_raise(valid)` ne lève rien
- `verify_or_raise(invalid)` lève `InvariantViolationError` avec `.report` complet

### 7.5 `test_aether_python_sync.py` (3 tests)
- Chaque `.lisp` a son `@invariant` Python
- Chaque `@invariant` Python a son `.lisp`
- Extraction d'ID depuis `iN_xxx.lisp` stable et case-insensitive

### 7.6 Côté Aether
Pas de pytest pour les `.lisp`. **La preuve est le test.** CI lance `verify_all.sh`.

---

## 8. Definition of Done

1. `pytest -q` ≥ 86 tests verts (52 V1 + ~34 V2-A)
2. `bash aether/verify_all.sh` → 8/8 invariants `proved`
3. Smoke test : `python -c "from mw_ia.guardrails import verify_formal, VariantSpec; r = verify_formal(VariantSpec(gamma=0.99, lr=1e-3, ...)); print(r.passed)"` → `True`
4. CI workflow `.github/workflows/aether_verify.yml` lance les deux et bloque le merge en cas d'échec
5. `README.md` mis à jour avec un paragraphe "V2-A — Aether guardrails : usage et API"
6. Commit annoté du tag `v0.2.0-a` (ou ce qui sera décidé au moment de la livraison)

---

## 9. Risques & mitigations

| # | Risque | Mitigation |
|---|---|---|
| R1 | Désynchronisation Lisp ↔ Python | `test_aether_python_sync.py` + CI bloque si `verify_all.sh` échoue. Convention de commit : toute modif de `invariants.py` doit toucher `aether/invariants/`. |
| R2 | Hypothesis trouve un contre-exemple que Aether avait prouvé | `counter_example` dans la `Violation` est le matériau de debug. Rejoué dans Aether : si confirmé, preuve trop forte (corriger Lisp) ; sinon, stratégie Hypothesis trop large (resserrer). |
| R3 | Perf 50 ms violée quand le catalogue grossit | Verifier embarrassingly parallel. v1 séquentiel ; v2 module pourra paralléliser via `concurrent.futures` si E itère beaucoup. |
| R4 | `VariantSpec` trop rigide pour variants futurs (archi à nombre de couches variable) | v1 vise hyperparams uniquement. Extension additive : ajouter `network: NetworkSpec` quand E le justifiera. Rétrocompat garantie par `Optional`. |
| R5 | Hook security flagge faussement la séquence `e​xec` + parenthèse | Le module Python n'invoque jamais cette fonction. Cf. CLAUDE.md §Garde-fous (1) pour le contournement standard du projet. |

---

## 10. Hors-scope explicite

- **Intégration concrète avec E** : E n'existe pas encore. L'API est conçue pour, mais l'intégration sera testée au cycle E.
- **Benchs empiriques de non-régression** : → sous-projet C.
- **Continual learning, EWC, rehearsal** : → sous-projet D.
- **Stockage des variants / historique des verdicts** : → sous-projet E.
- **Modification de la GUI V1** : la GUI ne change pas en V2-A. Un widget verdict-report sera ajouté quand E le justifiera.
- **Meta-RL, MAML, RL²** : → sous-projet F (ou V3).

---

## 11. Roadmap vers les sous-projets suivants

```
A (ici)   Guardrails formels                   ← brainstorm + spec + plan + impl
↓
B         Mémoire persistante cross-session    trajectoires + checkpoints + replay long-terme
↓
C         Évaluateur self-supervisé            compare(variant, baseline) → score, complète A
↓
D         Continual learning                   EWC + rehearsal sur suite de tâches
↓
E         Auto-modification (cœur)             propose_variant → A + C → deploy/reject
↓
F         Meta-RL (ou V3)                      MAML / RL² / context-based
```

Chaque sous-projet suivant aura son propre cycle `superpowers:brainstorming` → spec → `writing-plans` → impl, comme V1 et V2-A.
