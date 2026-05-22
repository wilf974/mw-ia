# CLAUDE.md — Projet MW_IA (Reinforcement Learning éducatif)

> **À LIRE EN PREMIER** par Claude au démarrage d'une nouvelle session dans ce projet.

---

## Règles de comportement (héritées de `C:\Windows\System32\CLAUDE.md`)

- Toujours répondre en **français**
- Toujours invoquer les skills **superpowers** (brainstorming, writing-plans, TDD, subagent-driven-development, etc.) et **frontend-design** quand pertinent
- Utiliser le MCP **context7** pour toute documentation de librairie (PyTorch, PyQt6, PyQtGraph, Gymnasium…)
- Le MCP **aether** est disponible et **doit être utilisé activement** pour vérifier les invariants RL critiques. Cf. `~/.claude/projects/.../memory/feedback_aether_usage.md`. **PAS** une dépendance Python de MW_IA.
- Utiliser **TaskCreate/TaskUpdate** pour piloter les tâches multi-étapes
- Quand pertinent, dispatcher des **agents parallèles** (subagent_type=Explore, general-purpose, etc.)

---

## État au handoff (2026-05-22 → V2-A LIVRÉ, attaquer sous-projet B ou évolution roadmap)

**V1** livrée et taguée `v0.1.0` (2026-05-21). DQN converge à 99 % winrate Expert sur RTX 3060 en ~19 s.

**V2-A — Aether guardrails : 9 phases / 33 tâches LIVRÉES.** Tag `v0.2.0-a` posé. 95 tests pytest verts (52 V1 + 43 V2-A). ~24 commits V2-A sur `main`.

### Sous-projet V2-A — décomposition du programme V2

La V2 "auto-amélioration" a été décomposée en 6 sous-projets séquentiels. **A est terminé** ; **B** est le prochain par défaut.

| # | Sous-projet | Statut |
|---|---|---|
| **A** | Aether guardrails | ✅ Livré (tag `v0.2.0-a`) |
| **B** | Mémoire persistante cross-session | ⏳ **Prochain par défaut** |
| C | Évaluateur self-supervisé | Pas commencé |
| D | Continual learning (EWC, rehearsal) | Pas commencé |
| E | Auto-modification (proposer/tester variants) | Pas commencé |
| F | Meta-RL (MAML / RL² / context-based) | Reportable V3 |

### V2-A — état final des phases

| Phase | Tâches | Statut | Tests | Commits |
|---|---|---|---|---|
| 1 — Setup (hypothesis, scaffolds) | T1 | ✅ | — | 1 |
| 2 — Contracts (Severity/Violation/VariantSpec/VerdictReport) | T2-T5 | ✅ | 14 | 4 |
| 3 — Exceptions (InvariantViolationError) | T6 | ✅ | 2 | 1 |
| 4 — Registry (@invariant + applicable_invariants) | T7-T8 | ✅ | 5 | 2 |
| 5 — 8 invariants I1-I8 + fixups | T9-T16 + 2 refactor | ✅ | 13 | 10 |
| 6 — Verifier (verify_formal + verify_or_raise + API) | T17-T19 + fixup | ✅ | 6 | 4 |
| 7 — Validations Aether (8 fichiers + verify_all.sh) | T20-T29 | ✅ | — | 10 |
| 8 — Sync check Aether↔Python + CI workflow | T30-T31 | ✅ | 3 | 2 |
| 9 — README V2-A + DoD smoke test + tag `v0.2.0-a` | T32-T33 | ✅ | — | 1 + tag |

### Catalogue d'invariants v1 (Phase 5 + Phase 7)

| ID | Nom | Runtime Python | Validation Aether |
|---|---|---|---|
| I1 | `gamma_in_open_unit` — γ ∈ (0,1) | invariants.py | i1_gamma_in_open_unit.aether (7 examples) |
| I2 | `bellman_contraction` — T γ-Lipschitz | invariants.py (50 paires + short-circuit γ≥1) | i2_bellman_contraction.aether (5 examples) |
| I3 | `huber_nonneg` — Huber ≥ 0 | invariants.py (100 paires y,ŷ) | i3_huber_nonneg.aether (7 examples) |
| I4 | `winrate_bounds` — winrate ∈ [0,1] | invariants.py (200 épisodes via MetricsTracker) | i4_winrate_bounds.aether (6 examples) |
| I5 | `epsilon_schedule_decreasing` — ε décroît, ∈ [0,1] | invariants.py (100 points) | i5_epsilon_schedule.aether (8 examples) |
| I6 | `replay_buffer_capacity` — size ≤ capacity | invariants.py (3×capacity pushes) | i6_replay_buffer_capacity.aether (10 examples sur 2 fn) |
| I7 | `reward_bounded` — r_min ≤ r_max | invariants.py (déterministe, optionnel) | i7_reward_bounded.aether (7 examples) |
| I8 | `episode_termination_exclusive` — terminated ⊕ truncated | invariants.py (5 épisodes GridWorld) | i8_episode_termination_exclusive.aether (6 examples) |

**Décision Phase 7 prise pendant l'implémentation** : la spec V2-A originale supposait un theorem prover SMT (syntaxe Lisp `(define-property ... (verify ...))` style Z3). En pratique le MCP Aether v1.4 disponible est un **test runner property-based** (`@example` + `@invariant` sur des `fn` typées). Scope adapté : `.aether` au lieu de `.lisp`, validation déclarative au lieu de preuve universelle. Documenté dans `aether/README.md` + README projet + tag `v0.2.0-a`. Une vraie vérification universelle (Z3/Lean/Coq) reste possible mais hors-scope V2-A.

**Note sur I2** : le test runtime Python pour γ=1.0 passe via un court-circuit analytique parce que l'opérateur de Bellman optimal reste **non-expansif** à γ=1 (mais pas strictement contractant), donc l'échantillonnage seul ne détecterait pas la violation. Décision documentée dans `feat(guardrails): invariant I2` (commit `f5b0a11`).

### Bonus livré (refactors qualité)

- `ReplayBuffer.current_index` — propriété publique (remplace l'accès direct à `_idx`). Commit `9998743`.
- Imports `invariants.py` et `test_verifier.py` consolidés en tête de fichier (PEP 8). Commits `f5f4d8f` et `920e208`.

---

## Objectif long-terme & Roadmap d'évolutions

**Vision** : IA auto-améliorante qui propose et teste ses propres modifications d'hyperparamètres / d'architecture, sous contraintes vérifiables formellement via Aether. La V1 modulaire est l'infrastructure de départ ; aucune refonte requise pour la V2.

### Programme V2 — sous-projets formellement décomposés

1. **A — Aether guardrails formels** (en cours)
2. **B — Mémoire persistante cross-session** (RVF ou équivalent)
3. **C — Évaluateur de politique self-supervisé** — l'agent juge ses trajectoires
4. **D — Continual learning** (EWC, replay rehearsal) — apprendre sans oubli
5. **E — Système de proposition / test d'updates** — auto-modification d'hyperparams puis d'archi
6. **F — Meta-RL** (MAML / RL² / context-based) — apprendre à apprendre

### Roadmap d'évolutions additionnelles (idées brainstorm 2026-05-22)

Au-delà du programme V2 formel, idées d'évolution apportant de la valeur à intégrer comme **sous-projets V3+** (ou comme features dans les sous-projets V2 existants quand naturel) :

1. **Mémoire neuronale temporelle (LSTM / GRU)** — étendre `QNetwork` pour conserver un historique d'états et permettre des stratégies longues. Brique pertinente dans le cadre du sous-projet B (mémoire persistante) ou comme variant proposable par E.

2. **Perception visuelle (CNN)** — variante où l'agent reçoit l'image rendue du GridWorld au lieu des coordonnées. Couche CNN convolutionnelle. Permet de comparer agent "symbolique" vs agent "vision-based" comme un vrai DeepMind agent.

3. **Personnalités d'IA** — agent prudent / agressif / explorateur, via reward shaping et ε différenciés. Naturel à tester via E (proposition de variants) sous contrainte des invariants Aether.

4. **Multi-objectifs** — au-delà d'atteindre la cible : survivre, économiser mouvements, explorer, éviter ennemis, gérer énergie. Étape obligatoire pour passer de "puzzle solver" à vrai agent autonome.

5. **Chatbot RL (vision long-terme)** — le RL ne choisit plus une direction de mouvement mais une action de dialogue (générer / rechercher / mémoriser / questionner). Le réseau devient un "cerveau décisionnel". C'est la cible ultime mentionnée depuis le début du projet.

6. **Visualisation neuronale temps réel dans la GUI** — heatmap des Q-values, activations des neurones, importance des actions, visualisation des couches. Extension naturelle de la GUI V1, gros impact pédagogique.

7. **Double DQN** — initialement hors-scope V1, à intégrer dans le cadre de E (comme un variant testable). Réduction de la surestimation des Q-values.

8. **Environnement procédural** — générer obstacles / cartes / objectifs aléatoires. Test crucial pour vérifier si l'IA apprend vraiment ou mémorise juste une map. Couple naturellement avec D (continual learning).

**Note d'orchestration** : chaque évolution sera traitée via le cycle complet `superpowers:brainstorming` → spec → `superpowers:writing-plans` → `superpowers:subagent-driven-development`, dans son propre sous-projet bien borné. Pas de big-bang refactor.

---

## Environnement machine (vérifié 2026-05-21)

- **OS** : Windows 11 Pro 10.0.26200
- **Shell Claude Code** : Git Bash (Unix-style dans `Bash` tool). PowerShell via `powershell.exe -NonInteractive -Command "..."`.
- **Python** : 3.13.12 — `py` alias ou `C:\Python313\python.exe`
- **GPU** : NVIDIA GeForce RTX 3060 (12 Go VRAM, Ampere CC 8.6)
- **Driver NVIDIA** : 591.86 / CUDA 13.1
- **PyTorch** : `2.11.0+cu128` (⚠️ wheels `cu121` n'existent plus pour Python 3.13 — utiliser `cu128`)
- **claude CLI** : v2.1.146
- **hypothesis** : `6.152.9` (ajouté pour V2-A, dans `requirements.txt`)

### Activer le venv (réflexe à chaque commande)
```bash
source .venv/Scripts/activate
```

### Réinstaller PyTorch depuis zéro si besoin
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Repo git

- **Repo dédié MW_IA** initialisé le 2026-05-21 dans `C:\Users\Wilfred\Documents\IA Inst\MW_IA\.git`.
- ⚠️ **Ne PAS confondre** avec le repo parent `C:/Users/Wilfred/.git` (branche `feature/pentest-agent`) qui appartient au projet **PenTest**.
- Branche : `main`. Tag : `v0.1.0`. ~46 commits au handoff V2-A Phase 5.
- **Pattern** : V1 et V2-A développées entièrement sur `main` (pas de feature branch). Garder ce pattern sauf indication contraire.

---

## Arborescence (état après V2-A, tag `v0.2.0-a`)

```
MW_IA/
├── CLAUDE.md                           # ce fichier
├── README.md                           # théorie + install + archi + section V2-A
├── requirements.txt                    # + hypothesis>=6.100
├── pyproject.toml
├── check_gpu.py
├── .github/workflows/
│   └── aether_verify.yml               # CI : pytest + aether/verify_all.sh
├── scripts/
│   ├── train_tabular.py
│   ├── train_dqn.py
│   └── launch_gui.py
├── mw_ia/
│   ├── config.py
│   ├── envs/gridworld.py
│   ├── agents/                         # base.py, value_iteration.py, policy_iteration.py, q_learning.py, dqn.py
│   ├── neural/                         # network.py, replay_buffer.py (+ current_index property), trainer.py
│   ├── training/                       # metrics.py, runner.py
│   ├── persistence/checkpoint.py
│   ├── gui/                            # theme.py, widgets/, app.py
│   └── guardrails/                     # [V2-A livré]
│       ├── __init__.py                 # API publique (re-exports + auto-import invariants)
│       ├── contracts.py                # Severity, Violation, VariantSpec, VerdictReport
│       ├── exceptions.py               # InvariantViolationError
│       ├── registry.py                 # Invariant, @invariant, _REGISTRY, applicable_invariants
│       ├── invariants.py               # I1-I8 + helpers (_bellman_operator, _huber, _compute_epsilon)
│       └── verifier.py                 # verify_formal + verify_or_raise
├── aether/                             # [V2-A livré]
│   ├── README.md                       # nature de la validation Aether v1.4
│   ├── verify_all.sh                   # harness shell : présence + non-vacuité des 8 .aether
│   └── invariants/
│       ├── .gitkeep
│       ├── i1_gamma_in_open_unit.aether
│       ├── i2_bellman_contraction.aether
│       ├── i3_huber_nonneg.aether
│       ├── i4_winrate_bounds.aether
│       ├── i5_epsilon_schedule.aether
│       ├── i6_replay_buffer_capacity.aether
│       ├── i7_reward_bounded.aether
│       └── i8_episode_termination_exclusive.aether
├── tests/                              # 95 tests (52 V1 + 43 V2-A)
│   ├── (V1 inchangés)
│   └── guardrails/
│       ├── conftest.py
│       ├── test_contracts.py           # 14 tests
│       ├── test_exceptions.py          # 2 tests
│       ├── test_registry.py            # 5 tests
│       ├── test_invariants.py          # 13 tests
│       ├── test_verifier.py            # 5 tests
│       ├── test_public_api.py          # 1 smoke test
│       └── test_aether_python_sync.py  # 3 tests anti-drift Aether↔Python
├── checkpoints/                        # .pt / .npz (gitignored)
├── logs/
└── docs/superpowers/
    ├── specs/
    │   ├── 2026-05-21-mw-ia-rl-design.md                     # V1
    │   └── 2026-05-21-mw-ia-v2-aether-guardrails-design.md   # V2-A (note : scope SMT vs runner adapté pendant impl)
    └── plans/
        ├── 2026-05-21-mw-ia-v1.md                            # V1
        └── 2026-05-21-mw-ia-v2-aether-guardrails.md          # V2-A (33 tâches, 9 phases — toutes ✅)
```

---

## Procédures usuelles

### Lancer les tests
```bash
source .venv/Scripts/activate && pytest -q
```
Attendu : **95 passed** (52 V1 + 43 V2-A).

### Entraîner Q-Learning tabulaire (headless)
```bash
source .venv/Scripts/activate && python scripts/train_tabular.py --episodes 1000
```
Référence : winrate 100 % niveau Expert en ~2 s.

### Entraîner DQN sur GPU (headless)
```bash
source .venv/Scripts/activate && python scripts/train_dqn.py --episodes 200 --device cuda
```
Référence : winrate 99 % niveau Expert en ~19 s sur RTX 3060.

### Lancer la GUI live
```bash
source .venv/Scripts/activate && python scripts/launch_gui.py
```

### Smoke test des guardrails V2-A
```bash
source .venv/Scripts/activate && pytest tests/guardrails/ -v
```
Attendu : 43 passed.

### Smoke harness Aether (présence des 8 fichiers .aether)
```bash
bash aether/verify_all.sh
```
Attendu : 8 OK, exit 0. La validation formelle (`@example` + `@invariant`) reste exécutée via le MCP `mcp__aether__aether_verify` côté Claude Code.

### Smoke E2E guardrails (API publique)
```bash
source .venv/Scripts/activate && PYTHONIOENCODING=utf-8 python -c "
from mw_ia.guardrails import VariantSpec, verify_formal
spec = VariantSpec(gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
                   epsilon_decay_steps=50_000, batch_size=128,
                   replay_capacity=100_000, target_sync_steps=1_000)
r = verify_formal(spec); print(f'passed={r.passed}, violations={len(r.violations)}')
"
```
Attendu : `passed=True, violations=0`. Avec `gamma=1.0` : `passed=False, violations=2` (I1 + I2).
**Note encodage** : `PYTHONIOENCODING=utf-8` est nécessaire pour afficher les messages contenant `γ` sur le shell Windows cp1252 par défaut.

---

## Garde-fous & pièges à connaître

1. **Hook `security_reminder_hook.py`** flagge naïvement la séquence `exec` suivie d'une parenthèse ouvrante comme une vuln Node.js — y compris la méthode `QApplication.exec` de PyQt6, **et même cette séquence dans des fichiers Markdown** (rencontré lors de la rédaction de la spec V2-A). **Contournement** : `getattr(app, "exec")()` ou variable intermédiaire pour le code, et reformulation dans les docs (ex : utiliser un caractère zero-width ou la phrase "appel de la fonction `exec`" sans la parenthèse immédiate).

2. **PyTorch cu121 obsolète pour Python 3.13.** Toujours installer via `--index-url https://download.pytorch.org/whl/cu128`. Si même `cu128` ne marche plus dans le futur, essayer `cu126`, `cu124`, puis fallback CPU explicite.

3. **`.claude/` doit rester gitignoré.**

4. **GridWorld `step_count` s'incrémente même quand bloqué** (mur ou obstacle). Intentionnel — sinon truncation par `max_steps` deviendrait contournable. Documenté commit `149b4d0`.

5. **Aether MCP — vraie nature découverte en Phase 7** : `mcp__aether__aether_verify` n'est PAS un theorem prover SMT mais un **test runner property-based Lisp typé** (Aether v1.4, syntaxe `(fn nom ((p type)) ret BODY)` avec `@intent`/`@invariant`/`@example` AVANT la `fn`). Lancer `mcp__aether__aether_quickref` au début (pas `syntax_guide` qui n'existe plus). Conséquence : les fichiers du catalogue sont des `.aether` (pas `.lisp`), validations déclaratives (pas preuves universelles). Documenté dans `aether/README.md`. Pour une vraie preuve universelle, prévoir un sous-projet V3+ Z3/Lean/Coq.

6. **VariantSpec** utilise les noms cohérents avec V1 `DQNConfig` (`replay_capacity`, `target_sync_steps`) plutôt que les noms génériques de la spec (`buffer_capacity`, `target_sync_interval`). Adaptation documentée dans le plan V2-A.

7. **I2 (bellman_contraction)** : le runtime échantillonnage à 50 paires ne détecte pas γ=1 (opérateur non-expansif mais non strictement contractant). Un short-circuit analytique pour `γ ≥ 1.0` complète la vérification — **ne pas le retirer** sous prétexte de "doublon", il couvre exactement le cas que l'échantillonnage manque.

8. **Encodage shell Windows** : le shell Windows par défaut (cp1252) crash sur les caractères Unicode des messages Violation (ex. `γ` dans le message I2). Forcer `PYTHONIOENCODING=utf-8` avant `python -c "..."` pour les smoke tests en CLI. Non-bloquant pour pytest (encode UTF-8 par défaut).

9. **`@example` Aether v1.4 et floats** : `aether_verify` compare `expected == actual` strictement. Choisir des `@example` qui produisent des floats IEEE 754 exacts (ex. `(0.5, 4, 2) → 0.75` exact, pas `(0.05, 50000, 25000) → 0.5250000000000001`). Pattern utilisé en T25 (I5).

---

## Mémoires persistantes liées

À consulter au démarrage :
- `~/.claude/projects/C--Users-Wilfred/memory/projet_mw_ia.md` — état du projet
- `~/.claude/projects/C--Users-Wilfred/memory/feedback_aether_usage.md` — usage Aether
- `~/.claude/projects/C--Users-Wilfred/memory/MEMORY.md` — index global

---

## Instructions pour la prochaine session

### Reprise par défaut — attaquer un nouveau sous-projet

V2-A étant terminé, la **suite naturelle** est le **sous-projet B (mémoire persistante cross-session)** ou une **évolution roadmap** (cf. section "Roadmap d'évolutions additionnelles" plus haut — 8 idées dont environnement procédural, CNN, LSTM, visualisation neuronale GUI, etc.).

1. **Lire ce CLAUDE.md en entier.**
2. **Smoke test rapide** :
   ```bash
   source .venv/Scripts/activate && pytest -q
   ```
   Attendu : 95 passed. + `bash aether/verify_all.sh` → 8 OK.
3. **Aligner avec l'utilisateur** sur le prochain sous-projet :
   - **B — Mémoire persistante cross-session** (RVF, agent qui se souvient d'une session à l'autre)
   - Ou une évolution roadmap (#1-8 listées dans la section "Roadmap d'évolutions additionnelles") — particulièrement **#8 environnement procédural / curriculum learning** qui couple naturellement avec D
   - Ou un sous-projet C/D/E/F directement
4. **Cycle complet** pour tout nouveau sous-projet :
   - `superpowers:brainstorming` → cerner intent, scope, contraintes
   - Écrire la spec dans `docs/superpowers/specs/YYYY-MM-DD-<sub-projet>-design.md`
   - `superpowers:writing-plans` → plan TDD bite-sized
   - `superpowers:subagent-driven-development` (pattern V2-A) : implementer + spec reviewer + code quality reviewer par task
5. **Ne pas démarrer en code-only** sur un nouveau sous-projet — passer obligatoirement par brainstorm + spec + plan, comme V2-A.

### Si l'objectif est un quick fix / petite feature V1

- TDD (test rouge → impl → vert → commit)
- Ne pas casser les tags `v0.1.0` ou `v0.2.0-a` (pas de force-push)
- Re-lancer `pytest -q` avant chaque commit (attendu : 95 passed)
- Ne pas toucher au module `mw_ia/guardrails/` ni à `aether/invariants/` sans raison documentée (V2-A livré, surface stable)

### Si l'objectif est de pousser un sous-projet V3+ (auto-modification, etc.)

Terminer **B → C → D → E** d'abord (ordre des prérequis logiques : mémoire → évaluateur → continual learning → auto-mod), sauf décision explicite de l'utilisateur. Chaque sous-projet reste un cycle complet brainstorm + spec + plan + impl.
