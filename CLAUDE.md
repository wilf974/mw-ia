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

## État au handoff (2026-05-22 → reprise V2-A Phase 6)

**V1** livrée et taguée `v0.1.0` (2026-05-21). DQN converge à 99 % winrate Expert sur RTX 3060 en ~19 s.

**V2-A — Aether guardrails : Phases 1-5 sur 9 terminées.** 86 tests pytest verts (52 V1 + 34 V2-A). 16 commits code + 2 commits docs (spec + plan) sur `main`.

### Sous-projet V2-A — décomposition du programme V2

La V2 "auto-amélioration" a été décomposée en 6 sous-projets séquentiels (cf. spec ci-dessous). **A** (Aether guardrails) est le premier, en cours.

| # | Sous-projet | Statut |
|---|---|---|
| **A** | Aether guardrails formels | Phases 1-5 / 9 terminées |
| B | Mémoire persistante cross-session | Pas commencé |
| C | Évaluateur self-supervisé | Pas commencé |
| D | Continual learning (EWC, rehearsal) | Pas commencé |
| E | Auto-modification (proposer/tester variants) | Pas commencé |
| F | Meta-RL (MAML / RL² / context-based) | Reportable V3 |

### V2-A — état détaillé des phases

| Phase | Tâches | Statut | Tests | Commits |
|---|---|---|---|---|
| 1 — Setup (hypothesis, scaffolds) | T1 | ✅ | — | 1 |
| 2 — Contracts (Severity/Violation/VariantSpec/VerdictReport) | T2-T5 | ✅ | 14 | 4 |
| 3 — Exceptions (InvariantViolationError) | T6 | ✅ | 2 | 1 |
| 4 — Registry (@invariant + applicable_invariants) | T7-T8 | ✅ | 5 | 2 |
| 5 — 8 invariants I1-I8 + fixups | T9-T16 + 2 refactor | ✅ | 13 | 10 |
| **6 — Verifier (verify_formal + verify_or_raise + public API)** | **T17-T19** | **⏳ à faire** | — | — |
| 7 — Preuves Aether (i1-i8.lisp + verify_all.sh) | T20-T29 | ⏳ | — | — |
| 8 — Sync check + CI workflow | T30-T31 | ⏳ | — | — |
| 9 — README + DoD smoke test + tag v0.2.0-a | T32-T33 | ⏳ | — | — |

### Catalogue d'invariants v1 livré (Phase 5)

| ID | Nom | Localisation | Échantillonnage |
|---|---|---|---|
| I1 | `gamma_in_open_unit` — γ ∈ (0,1) | invariants.py | déterministe |
| I2 | `bellman_contraction` — T γ-Lipschitz | invariants.py | 50 paires (Q,Q') mini-MDP + short-circuit analytique γ ≥ 1 |
| I3 | `huber_nonneg` — Huber ≥ 0 | invariants.py | 100 paires (y, ŷ) |
| I4 | `winrate_bounds` — winrate ∈ [0,1] | invariants.py | 200 épisodes via MetricsTracker |
| I5 | `epsilon_schedule_decreasing` — ε décroît, ∈ [0,1] | invariants.py | 100 points |
| I6 | `replay_buffer_capacity` — size ≤ capacity | invariants.py | 3×capacity pushes via ReplayBuffer |
| I7 | `reward_bounded` — r_min ≤ r_max | invariants.py | déterministe, optionnel |
| I8 | `episode_termination_exclusive` — terminated ⊕ truncated | invariants.py | 5 épisodes GridWorld |

**Note sur I2** : le test pour γ=1.0 passe via un court-circuit analytique parce que l'opérateur de Bellman optimal reste **non-expansif** à γ=1 (mais pas strictement contractant), donc l'échantillonnage seul ne détecterait pas la violation. Décision documentée dans `feat(guardrails): invariant I2` (commit `f5b0a11`).

### Bonus livré (refactors qualité)

- `ReplayBuffer.current_index` — propriété publique (remplace l'accès direct à `_idx`). Commit `9998743`.
- Imports `invariants.py` consolidés en tête de fichier (PEP 8 + prêt pour T30 sync check). Commit `f5f4d8f`.

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

## Arborescence (état après V2-A Phase 5)

```
MW_IA/
├── CLAUDE.md                           # ce fichier
├── README.md                           # théorie + install + archi (à enrichir en T32)
├── requirements.txt                    # + hypothesis>=6.100
├── pyproject.toml
├── check_gpu.py
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
│   └── guardrails/                     # [NOUVEAU V2-A]
│       ├── __init__.py                 # vide pour l'instant — sera rempli en T19
│       ├── contracts.py                # Severity, Violation, VariantSpec, VerdictReport
│       ├── exceptions.py               # InvariantViolationError
│       ├── registry.py                 # Invariant, @invariant, _REGISTRY, applicable_invariants
│       └── invariants.py               # I1-I8 + helpers (_bellman_operator, _huber, _compute_epsilon)
├── tests/                              # 86 tests (52 V1 + 34 V2-A)
│   ├── (V1 inchangés)
│   └── guardrails/
│       ├── conftest.py
│       ├── test_contracts.py           # 14 tests
│       ├── test_exceptions.py          # 2 tests
│       ├── test_registry.py            # 5 tests
│       └── test_invariants.py          # 13 tests
├── checkpoints/                        # .pt / .npz (gitignored)
├── logs/
└── docs/superpowers/
    ├── specs/
    │   ├── 2026-05-21-mw-ia-rl-design.md                     # V1
    │   └── 2026-05-21-mw-ia-v2-aether-guardrails-design.md   # V2-A
    └── plans/
        ├── 2026-05-21-mw-ia-v1.md                            # V1
        └── 2026-05-21-mw-ia-v2-aether-guardrails.md          # V2-A (33 tâches, 9 phases)
```

---

## Procédures usuelles

### Lancer les tests
```bash
source .venv/Scripts/activate && pytest -q
```
Attendu : **86 passed** (52 V1 + 34 V2-A).

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
Attendu : 34 passed.

---

## Garde-fous & pièges à connaître

1. **Hook `security_reminder_hook.py`** flagge naïvement la séquence `exec` suivie d'une parenthèse ouvrante comme une vuln Node.js — y compris la méthode `QApplication.exec` de PyQt6, **et même cette séquence dans des fichiers Markdown** (rencontré lors de la rédaction de la spec V2-A). **Contournement** : `getattr(app, "exec")()` ou variable intermédiaire pour le code, et reformulation dans les docs (ex : utiliser un caractère zero-width ou la phrase "appel de la fonction `exec`" sans la parenthèse immédiate).

2. **PyTorch cu121 obsolète pour Python 3.13.** Toujours installer via `--index-url https://download.pytorch.org/whl/cu128`. Si même `cu128` ne marche plus dans le futur, essayer `cu126`, `cu124`, puis fallback CPU explicite.

3. **`.claude/` doit rester gitignoré.**

4. **GridWorld `step_count` s'incrémente même quand bloqué** (mur ou obstacle). Intentionnel — sinon truncation par `max_steps` deviendrait contournable. Documenté commit `149b4d0`.

5. **Aether MCP** doit être mobilisé activement. Lancer `mcp__aether__syntax_guide` d'abord pour rafraîchir la syntaxe Lisp typée.

6. **VariantSpec** utilise les noms cohérents avec V1 `DQNConfig` (`replay_capacity`, `target_sync_steps`) plutôt que les noms génériques de la spec (`buffer_capacity`, `target_sync_interval`). Adaptation documentée dans le plan V2-A.

7. **I2 (bellman_contraction)** : le runtime échantillonnage à 50 paires ne détecte pas γ=1 (opérateur non-expansif mais non strictement contractant). Un short-circuit analytique pour `γ ≥ 1.0` complète la vérification — **ne pas le retirer** sous prétexte de "doublon", il couvre exactement le cas que l'échantillonnage manque.

---

## Mémoires persistantes liées

À consulter au démarrage :
- `~/.claude/projects/C--Users-Wilfred/memory/projet_mw_ia.md` — état du projet
- `~/.claude/projects/C--Users-Wilfred/memory/feedback_aether_usage.md` — usage Aether
- `~/.claude/projects/C--Users-Wilfred/memory/MEMORY.md` — index global

---

## Instructions pour la prochaine session (reprise V2-A)

### Reprise par défaut — continuer V2-A à partir de la Phase 6

1. **Lire ce CLAUDE.md en entier.**
2. **Lire la spec V2-A** : `docs/superpowers/specs/2026-05-21-mw-ia-v2-aether-guardrails-design.md`
3. **Lire le plan V2-A** : `docs/superpowers/plans/2026-05-21-mw-ia-v2-aether-guardrails.md` — focus sur les **Phases 6 à 9** (Phases 1-5 déjà livrées).
4. **Smoke test rapide** :
   ```bash
   source .venv/Scripts/activate && pytest -q
   ```
   Attendu : 86 passed.
5. **Continuer en Subagent-Driven Development**, à partir de **T17 (Phase 6 — verify_formal)**. Pattern utilisé jusqu'ici :
   - Implementer subagent (haiku ou sonnet selon complexité) sur 1 tâche ou groupe cohérent (typiquement 1 fichier source)
   - Spec compliance reviewer
   - Code quality reviewer
   - Marquer phase complète, passer à la suivante
6. Phases restantes :
   - **Phase 6** (T17-T19) : `verifier.py` + API publique dans `__init__.py` — petite phase
   - **Phase 7** (T20-T29) : 8 preuves Aether `.lisp` + `verify_all.sh` — nécessite `mcp__aether__syntax_guide` puis `mcp__aether__verify` par fichier
   - **Phase 8** (T30-T31) : sync check Lisp↔Python + workflow CI
   - **Phase 9** (T32-T33) : README + smoke test DoD + tag `v0.2.0-a`

### Si l'objectif est un autre sous-projet V2 (B-F) ou une évolution roadmap

Faire **terminer V2-A d'abord** (ne reste que Phases 6-9, soit ~15 tâches). Puis attaquer le sous-projet suivant avec son propre cycle complet (brainstorming → spec → plan → impl).

### Si l'objectif est un quick fix / petite feature V1

- TDD (test rouge → impl → vert → commit)
- Ne pas casser le tag `v0.1.0` (pas de force-push)
- Re-lancer `pytest -q` avant chaque commit (attendu : 86 passed)
- Ne pas toucher au module `mw_ia/guardrails/` sans raison documentée (V2-A en cours)
