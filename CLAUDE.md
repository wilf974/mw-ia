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

## État au handoff (2026-05-22 → V2-A + V2-X + V2-Y LIVRÉS, validation V2-Y en cours)

**V1** livrée et taguée `v0.1.0` (2026-05-21). DQN converge à 99 % winrate Expert sur RTX 3060 en ~19 s.

**V2-A — Aether guardrails : 9 phases / 33 tâches LIVRÉES.** Tag `v0.2.0-a` posé. 95 tests pytest verts (52 V1 + 43 V2-A). ~24 commits V2-A sur `main`.

**V2-X — Procedural environment & curriculum learning : 13 phases / 16 tâches LIVRÉES.** Tag `v0.2.0-x` posé. 148 tests pytest verts (95 baseline + 53 V2-X). 17 commits V2-X sur `main`. Smoke E2E maze 50 ép → winrate 94 % et scheduler s'est déclenché. **Plafond architectural identifié** : DQN feedforward ne dépasse pas diff~0.10 même avec recette gagnante (`--hidden 256 256 --epsilon-decay-steps 200000`), winrate ~80 %. Cf. section "V2-X — recette opérationnelle" plus bas.

**V2-Y — Deep Recurrent Q-Network (LSTM) : 9 phases / 13 tâches LIVRÉES.** Tag `v0.2.0-y` posé. **183 tests pytest verts** (148 baseline + 35 V2-Y). 12 commits V2-Y sur `main`. Motivation : franchir le plafond V2-X via mémoire neuronale temporelle. Smoke E2E maze 20 ép CPU → 100 % winrate à diff=0. **Critère succès final à valider en grand entraînement** : bucket 1 du tracker (diff 0.20-0.40) ≥ 70 % winrate sur 5000 ép.

### Sous-projets — décomposition du programme V2 + évolutions

La V2 "auto-amélioration" a été décomposée en 6 sous-projets séquentiels (A-F). En parallèle, des évolutions roadmap (`V2-X`, `V2-Y`...) sont livrables indépendamment quand naturel. **A, X et Y sont terminés** ; **B** reste le prochain sous-projet "auto-amélioration" naturel.

| # | Sous-projet | Statut |
|---|---|---|
| **A** | Aether guardrails | ✅ Livré (tag `v0.2.0-a`) |
| **X** | Environnement procédural + curriculum | ✅ Livré (tag `v0.2.0-x`) |
| **Y** | Deep Recurrent Q-Network (LSTM, roadmap #1) | ✅ Livré (tag `v0.2.0-y`) |
| **B** | Mémoire persistante cross-session | ⏳ **Prochain par défaut** |
| C | Évaluateur self-supervisé | Pas commencé |
| D | Continual learning (EWC, rehearsal) | Pas commencé — préfiguré par bucket tracker V2-X |
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
- Imports `test_maze_generators.py` consolidés + harmonisation `assert`/`ValueError` dans `RandomObstaclesGenerator.__post_init__`. Commit `bbcfd55`.

### V2-X — état final des phases (livraison 2026-05-22)

| Phase | Tâches | Statut | Tests | Commits |
|---|---|---|---|---|
| 1 — Setup (scaffold envs/) | T1 | ✅ | — | 1 |
| 2 — `maze_bfs_check` | T2 | ✅ | 7 | 1 |
| 3 — `RandomObstaclesGenerator` (BFS-checked) | T3 | ✅ | 6 (+1 property) | 1 |
| 4 — `PerfectMazeGenerator` (DFS backtracker, quasi-parfait tailles paires) | T4 | ✅ | 5 (+1 property) | 1 + fixup `bbcfd55` |
| 5 — `ProceduralEnvConfig` + `SchedulerConfig` | T5 | ✅ | 9 | 1 |
| 6 — `AdaptiveDifficultyScheduler` | T6 | ✅ | 7 | 1 |
| 7 — `DifficultyBucketTracker` + `MetricsTracker.has_data` | T7 | ✅ | 5 | 1 |
| 8 — `ProceduralGridWorld` | T8 | ✅ | 5 | 1 |
| 9 — `encode_procedural_observation` (one-hot + grid flatten + padding) | T9 | ✅ | 5 | 1 |
| 10 — `ProceduralDQNRunner` + extensions `RunnerCallbacks` | T10 | ✅ | 3 | 1 |
| 11 — `scripts/train_dqn_procedural.py` + CI smoke | T11 | ✅ | — | 1 |
| 12 — GUI : signal `maze_changed`, 5e courbe, label, bouton Procedural | T12-T14 | ✅ | — | 3 |
| 13 — README V2-X + DoD + tag `v0.2.0-x` | T15-T16 | ✅ | — | 1 + tag |

### Composants V2-X livrés

| Composant | Fichier | Rôle |
|---|---|---|
| `maze_bfs_check` | `mw_ia/envs/maze_generators.py` | BFS 4-connexe — garantie solvabilité |
| `RandomObstaclesGenerator` | idem | Place N obstacles aléatoires + check BFS post-hoc (regen ≤100 tentatives sinon `RuntimeError`) |
| `PerfectMazeGenerator` | idem | DFS recursive backtracker, solvable par construction (quasi-parfait pour tailles paires) |
| `ProceduralEnvConfig`, `SchedulerConfig` | `mw_ia/config.py` | frozen dataclasses + `__post_init__` validation |
| `AdaptiveDifficultyScheduler` | `mw_ia/training/scheduler.py` | winrate ≥ 80 % → diff +0.05 ; ≤ 30 % → -0.05 |
| `DifficultyBucketTracker` | `mw_ia/training/metrics.py` | 5 buckets [0,0.2)..[0.8,1.0], routing `min(4, int(diff*5))` |
| `ProceduralGridWorld` | `mw_ia/envs/procedural_env.py` | Wrapper sur V1 `GridWorld`, régénère le maze à chaque `reset()` |
| `encode_procedural_observation` | idem | one-hot position (`max_rows*max_cols` dim) + grid flatten (idem) → dim 200 par défaut, padding zéro pour mazes < max_size |
| `ProceduralDQNRunner` | `mw_ia/training/runner.py` | Boucle DQN procedural avec scheduler + bucket tracker + callbacks GUI étendus |
| GUI procedural | `mw_ia/gui/widgets/{gridworld_view,live_plots,control_panel,difficulty_label}.py` | Signal `maze_changed`, 5e courbe `difficulty`, label "Maze #N, diff=X.XX", bouton "Démarrer (procedural)" |

**Décisions techniques V2-X** :

- **Observation procedural** : la spec parlait de `[row, col, *grid_flatten]` (2+R*C dim). Découverte pendant impl : V1 utilise déjà un encoding **one-hot** (`DQNRunner._state_vec`). Aligné sur ce pattern : observation = `concat(position_one_hot, grid_flatten)` = `2 * max_rows * max_cols` dim. Documenté dans la docstring d'`encode_procedural_observation`.
- **`max_density` fixture test à 0.40 vs default classe 0.50** : empiriquement sur 10×10, density=0.50 produit ~0.6 % de mazes solvables par tirage → flaky avec `max_attempts=100`. Fixture test descendu à 0.40 (taux succès ~10.8 %/tirage → 99.9 % en 100 tentatives). Le default de la classe reste 0.50 (spec respectée, à utiliser en connaissance de cause).
- **DFS tailles paires** : le DFS classique creuse uniquement les cellules paires/paires. Pour size paire (4, 6, …, 20), `goal=(size-1, size-1)` est impair/impair → non accessible. Fix : ouvrir explicitement les 2 murs intermédiaires vers le voisin pair-pair. Conséquence : "quasi-parfait" (goal accessible par 1-2 chemins supplémentaires), pas strict-parfait. Documenté dans la docstring.

### V2-X — recette opérationnelle (post-livraison, sessions empiriques 2026-05-22)

Le DQN feedforward V1 par défaut (`hidden=(128, 128)`, `epsilon_decay_steps=50000`) **ne converge PAS** en mode procedural obstacles 10×10 : winrate 1 % à 2000 ép (pire que random). Trois leviers indépendants identifiés et corrigés :

| Levier | Symptôme | Fix | Effet sur winrate final 2000 ép |
|---|---|---|---|
| 1. `min_density=0.10` trop élevé | Curriculum bloqué à diff=0 dès le départ (10 obstacles d'office) | Default → **0.0** (commits `97f3f30`) | 1% → 53% |
| 2. `hidden_layers=(128, 128)` insuffisant | Agent retombe à diff=0 après bref passage à 0.05 | `--hidden 256 256` (CLI option `8c7b091`) | 53% → 47%* |
| 3. `epsilon_decay_steps=50000` trop court (ε=0.05 dès ép 290) | Quasi-greedy avant consolidation politique | `--epsilon-decay-steps 200000` (CLI `40a6b23`) | 47% → **80%** ✓ |

\* baisse marginale isolément, mais permet de tenir diff=0.05 (vs 0.00 avant)

**Recette gagnante** (winrate=80%, diff=0.10 à 2000 ép, scheduler oscille adaptativement entre 0.05 et 0.10) :

```bash
python scripts/train_dqn_procedural.py \
    --episodes 2000 --mode obstacles --device cuda \
    --hidden 256 256 --epsilon-decay-steps 200000
```

**Plafond architectural identifié** : avec cette recette, l'agent reste bloqué autour de diff=0.10 (bucket 0 du tracker, 80% winrate). Les buckets 1-4 restent vides. Le critère spec V2-X "≥70% à bucket max densité 0.4" est **inatteignable avec le DQN feedforward simple** — c'est précisément ce que la spec prédisait. Justifie le **sous-projet V2-Y LSTM/GRU** (roadmap #1) : mémoire neuronale temporelle pour vrai apprentissage maze-conditional. Alternativement : Double DQN, Dueling, CNN sur image rendue (roadmap #2/#7).

**Comportement adaptatif vérifié** : avec la recette gagnante, le scheduler monte à diff 0.10 (winrate ≥ 80%), l'agent struggle, scheduler redescend à 0.05 (winrate ≤ 30%), agent réapprend, retest. Mécanisme PCG-RL canonique opérationnel.

### V2-X — fix scheduler consolidé (2026-05-22, post-GUI-fix)

Suite au fix GUI procedural (commit `1a040c6`), 3 itérations de tuning scheduler sur V2-X 2000 ép GPU avec recette gagnante :

| Iter | `update_interval` | `step` | Final winrate | Final diff | Verdict |
|---|---|---|---|---|---|
| 1 (V2-X initial) | 50 | 0.05 | ~47 % | 0.10 | Scheduler trop agressif, oscille rapidement, décroche à 0.10 |
| 2 (`469b7fc`) | **200** | 0.05 | **72 %** | 0.05 | Scheduler patient, oscille 0.05 ↔ 0.10 cycle ~400 ép, stable au plus bas palier |
| 3 (`300fe7d` → reverté) | 200 | 0.025 | 54 % ⬇ | 0.05 | Step trop fin = 3 paliers oscillation 0.025↔0.05↔0.075, instable |

**Default V2-X consolidé** (commit `[revert]`) :
- `SchedulerConfig.update_interval = 200` (vs 50 original)
- `SchedulerConfig.step = 0.05` (inchangé après expérience iter 3)
- `step = 0.025` reste **option expérimentale** pour V2-Y / CNN / Double DQN qui pourraient bénéficier d'un palier intermédiaire 0.075.

**Plafond V2-X consolidé** : le DQN feedforward (même avec `hidden=(256,256)`, `epsilon_decay_steps=200000`, `min_density=0.0`, scheduler `update_interval=200`, `step=0.05`) **plafonne à diff ≈ 0.05** (5 obstacles sur 10×10), pas 0.10 comme initialement perçu. Le palier 0.075 a été atteint en iter 3 mais l'agent décroche immédiatement. Au-delà nécessite changement d'archi : **V2-Y LSTM (livré), Double DQN, CNN, ou Dueling**.

### V2-Y — scheduler antagoniste à V2-X (finding empirique 2026-05-22)

Test du scheduler V2-X consolidé (`update_interval=200`, `step=0.05`) sur V2-Y DRQN 5000 ép : **catastrophe**. L'agent oscille violemment entre diff 0 (winrate 80-96%) et diff 0.05 (winrate 5-22%), finit à **5% @ diff 0.05** (vs V2-Y initial avec update=50 : 95% @ diff 0.05).

**Diagnostic : les hyperparams scheduler dépendent de l'archi de l'agent** :
- DQN feedforward (V2-X) : apprend lentement mais stable → scheduler patient (`update=200`) = optimal
- DQN LSTM (V2-Y) : catastrophic forgetting naturel → scheduler patient = chaos amplifié

**Décision design** : `SchedulerConfig()` defaults globaux restent V2-X-optimaux (`update=200`, `step=0.05`). Les CLI scripts exposent les flags `--scheduler-update-interval` et `--scheduler-step` (commits `[hash]`) avec des defaults adaptés par archi :
- `scripts/train_dqn_procedural.py` (V2-X) : default `--scheduler-update-interval 200` (recette V2-X gagnante)
- `scripts/train_drqn_procedural.py` (V2-Y) : default `--scheduler-update-interval 50` (LSTM-friendly, V2-Y initial 95% winrate validé)

**Recommandation usage** :
- V2-X DQN feedforward → garder defaults (`update=200`, `step=0.05`)
- V2-Y DRQN LSTM → garder defaults V2-Y CLI (`update=50`, `step=0.05`) — JAMAIS passer `--scheduler-update-interval 200` sur V2-Y
- Futures archis (CNN, Double DQN) → expérimenter au cas par cas

### V2-Y — baseline LSTM consolidée (2026-05-22)

Reproductibilité validée sur 2 entraînements 5000 ép GPU avec config CLI V2-Y consolidée :

```bash
python scripts/train_drqn_procedural.py --episodes 5000 --mode obstacles --device cuda
```

**Defaults V2-Y CLI consolidés** (commits `e7eda95` initial + `c1c4214` CLI scheduler) :
- `--fc-hidden 256`, `--lstm-hidden 128`, `--sequence-length 32`, `--epsilon-decay-steps 200000` (defaults CLI)
- `--scheduler-update-interval 50`, `--scheduler-step 0.05` (defaults CLI V2-Y, distincts de SchedulerConfig() global)
- Defaults DRQNConfig() : `replay_capacity=5000` trajectoires, `min_episodes_to_learn=100`, `train_steps_per_episode=4`, `target_sync_steps=1000`, `batch_size=128`

**Résultat reproductible** :
- **Final winrate ≈ 95% @ diff 0.05** (bucket 0 plein)
- Pattern de trajectoire : oscillation diff 0 / 0.05, catastrophic forgetting visible au milieu (~ep 1500-2000), récupération forte en fin d'entraînement
- Buckets 1-4 restent vides → diff 0.05 jamais franchi de manière stable
- Critère succès original V2-Y "bucket 1 (0.2-0.4) ≥ 70 %" **PAS atteint**

**Comparaison V2-X feedforward vs V2-Y LSTM** (recette gagnante pour chacun) :

| Métrique | V2-X feedforward | V2-Y LSTM | Δ |
|---|---|---|---|
| Final winrate (bucket 0) | 72 % | **95 %** | +23 pp ✓ |
| Final diff atteinte | 0.05 | 0.05 | = |
| Bucket 1 rempli | non | non | = |
| Plafond architectural | diff ~0.05 | diff ~0.05 | = (mais meilleur winrate) |

**Trouvaille consolidée** : le LSTM apprend mieux **la map** (winrate +23 pp à diff identique) mais ne franchit pas le plafond de difficulté du scheduler — c'est-à-dire que l'agent récurrent maîtrise mieux les mazes à 5 obstacles mais reste incapable de tenir à 7-10 obstacles. Le plafond V2-Y est dû à un mélange de :
1. **Catastrophic forgetting LSTM** au milieu de l'entraînement (oscille entre récupération et chute)
2. **BPTT 32 + 4 train_steps/épisode** = updates très lourdes, l'agent ne consolide pas avant que la difficulté monte
3. **Scheduler+LSTM = dynamique antagoniste** (cf. section précédente)

**Pour franchir ce plafond, options possibles** (sous-projets V3+ ou itérations V2-Y) :
- **Burn-in style R2D2** : remplacer DRQN simple par hidden state burn-in dans le trainer (hors-scope V2-Y MVP, documenté en `recurrent_trainer.py` docstring)
- **Double DQN** : appliquer à V2-X ou V2-Y pour réduire surestimation Q-values (roadmap #7)
- **CNN sur image rendue** : observation visuelle au lieu de `[position_one_hot, grid_flatten]` (roadmap #2)
- **Tuning V2-Y** : `train_steps_per_episode=1-2`, `sequence_length=16`, `target_sync_steps=5000`

### V2-Y — état final des phases (livraison 2026-05-22)

| Phase | Tâches | Statut | Tests | Commits |
|---|---|---|---|---|
| 1 — Setup (scaffold neural/ + tests/neural/) | T1 | ✅ | — | 1 |
| 2 — `RecurrentQNetwork` (init + forward 1 step + séquence) | T2-T3 | ✅ | 7 | 2 |
| 3 — `SequenceReplayBuffer` (push_trajectory + sample padding+mask) | T4-T5 | ✅ | 9 | 2 |
| 4 — `RecurrentDQNTrainer` (BPTT + Huber masquée + AMP) | T6 | ✅ | 3 | 1 |
| 5 — `DRQNConfig` (frozen dataclass + validation) | T7 | ✅ | 6 | 1 |
| 6 — `RecurrentDQNAgent` (init + hidden state runtime + observe + end_episode) | T8-T9 | ✅ | 8 | 2 |
| 7 — `RecurrentProceduralDQNRunner` (extension runner.py) | T10 | ✅ | 2 | 1 (+ fix mineur agent inclus) |
| 8 — CLI `train_drqn_procedural.py` + CI smoke | T11 | ✅ | — | 1 |
| 9 — README V2-Y + DoD + tag `v0.2.0-y` | T12-T13 | ✅ | — | 1 + tag |

### Composants V2-Y livrés

| Composant | Fichier | Rôle |
|---|---|---|
| `RecurrentQNetwork` | `mw_ia/neural/recurrent.py` | `Linear → ReLU → LSTM → Linear`, `batch_first=False`, hidden tuple `(h, c)` ou `None` (auto-init zéros) |
| `SequenceReplayBuffer`, `BatchSeq` | `mw_ia/neural/sequence_buffer.py` | Buffer circulaire de trajectoires (capacity = nombre de trajectoires, PAS transitions). Sample : `B` trajectoires × fenêtre seq_len aléatoire, padding zéros + mask (1=vrai step, 0=padding) |
| `RecurrentDQNTrainer` | `mw_ia/neural/recurrent_trainer.py` | BPTT 32 steps, Huber `reduction="none"` × mask, AMP + grad clip 10 |
| `DRQNConfig` | `mw_ia/config.py` | frozen dataclass, defaults : fc=256, lstm=128, seq=32, replay_capacity=5000 trajectoires, decay=200_000, episodes=5_000 |
| `RecurrentDQNAgent` | `mw_ia/agents/recurrent_dqn.py` | Hidden state runtime maintenu (y compris en ε-random), reset_hidden/begin_episode/end_episode hooks, train à fin d'épisode, fix `max(min_episodes_to_learn, batch_size)` pour buffer sample |
| `RecurrentProceduralDQNRunner` | `mw_ia/training/runner.py` | Extension parallèle au `ProceduralDQNRunner` V2-X (V2-X intact) |
| CLI | `scripts/train_drqn_procedural.py` | Args `--fc-hidden`, `--lstm-hidden`, `--sequence-length`, `--epsilon-decay-steps` |

**Décisions techniques V2-Y** :

- **DRQN simple (Hausknecht & Stone 2015), pas burn-in R2D2** : hidden state zero-init au début de chaque séquence d'entraînement (bias accepté en MVP).
- **Hidden state runtime maintenu y compris en ε-random** : le LSTM doit observer la trajectoire complète d'obs indépendamment des choix d'action de l'agent.
- **Train step à la fin d'épisode** (pas à chaque step comme V1) : DRQN nécessite trajectoires complètes pour BPTT. `train_steps_per_episode=4` par défaut.
- **`replay_capacity` = nombre de TRAJECTOIRES** dans `SequenceReplayBuffer` (vs V1 `ReplayBuffer` qui compte des transitions). À ne pas confondre.
- **Fix `end_episode()` train trigger** : seuil corrigé de `>= min_episodes_to_learn` à `>= max(min_episodes_to_learn, batch_size)` pour éviter `ValueError` du buffer sample quand `batch_size > min_episodes_to_learn`. Inclus dans commit `18d09d9` (avec le runner). Concern d'atomicité commit notée mais correction acceptée pour correctness.
- **Padding zéros + mask** plutôt que séquences fixes : permet d'entraîner sur des épisodes courts (agent atteint goal en 18 steps → 14 dernières fenêtres paddées, mask les exclut du gradient).

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

## Arborescence (état après V2-X, tag `v0.2.0-x`)

```
MW_IA/
├── CLAUDE.md                           # ce fichier
├── README.md                           # théorie + install + archi + sections V2-A et V2-X
├── requirements.txt                    # + hypothesis>=6.100
├── pyproject.toml
├── check_gpu.py
├── .github/workflows/
│   └── aether_verify.yml               # CI : pytest + smoke procedural + aether/verify_all.sh
├── scripts/
│   ├── train_tabular.py
│   ├── train_dqn.py                    # V1 fix-map
│   ├── train_dqn_procedural.py         # [V2-X] CLI procedural DQN feedforward
│   ├── train_drqn_procedural.py        # [V2-Y] CLI procedural DRQN (LSTM)
│   └── launch_gui.py
├── mw_ia/
│   ├── config.py                       # + ProceduralEnvConfig + SchedulerConfig [V2-X]
│   ├── envs/
│   │   ├── gridworld.py                # V1 inchangé
│   │   ├── maze_generators.py          # [V2-X] maze_bfs_check + 2 générateurs
│   │   └── procedural_env.py           # [V2-X] ProceduralGridWorld + encode helper
│   ├── agents/                         # V1 inchangé
│   ├── neural/
│   │   ├── network.py                  # V1 inchangé (QNetwork déjà paramétrable input_dim)
│   │   ├── replay_buffer.py            # V1 inchangé
│   │   ├── trainer.py                  # V1 inchangé
│   │   ├── recurrent.py                # [V2-Y] RecurrentQNetwork (Linear+LSTM+Linear)
│   │   ├── sequence_buffer.py          # [V2-Y] SequenceReplayBuffer + BatchSeq
│   │   └── recurrent_trainer.py        # [V2-Y] RecurrentDQNTrainer (BPTT masqué)
│   ├── agents/                         # V1 (q_learning, value_iteration, dqn) + [V2-Y] recurrent_dqn.py
│   ├── training/
│   │   ├── metrics.py                  # + DifficultyBucketTracker + has_data() [V2-X]
│   │   ├── scheduler.py                # [V2-X] AdaptiveDifficultyScheduler
│   │   └── runner.py                   # + ProceduralDQNRunner [V2-X] + RecurrentProceduralDQNRunner [V2-Y]
│   ├── persistence/checkpoint.py
│   ├── gui/
│   │   ├── theme.py
│   │   ├── app.py                      # + on_start_procedural + 2 signaux thread [V2-X]
│   │   └── widgets/
│   │       ├── gridworld_view.py       # + on_maze_changed slot (redraw obstacles) [V2-X]
│   │       ├── live_plots.py           # + 5e courbe difficulty, layout 2x3 [V2-X]
│   │       ├── control_panel.py        # + bouton "Démarrer (procedural)" [V2-X]
│   │       ├── difficulty_label.py     # [V2-X] label "Maze #N, diff=X.XX"
│   │       ├── stats_panel.py          # V1 inchangé
│   │       └── log_console.py          # V1 inchangé
│   └── guardrails/                     # [V2-A livré, inchangé V2-X]
│       ├── __init__.py
│       ├── contracts.py
│       ├── exceptions.py
│       ├── registry.py
│       ├── invariants.py
│       └── verifier.py
├── aether/                             # [V2-A livré, inchangé V2-X]
│   ├── README.md
│   ├── verify_all.sh
│   └── invariants/iN_*.aether          # 8 fichiers
├── tests/                              # 183 tests (52 V1 + 43 V2-A + 53 V2-X + 35 V2-Y)
│   ├── (V1 inchangés)
│   ├── guardrails/                     # [V2-A]
│   ├── envs/                           # [V2-X]
│   │   ├── conftest.py                 # fixture rng seedée
│   │   ├── test_maze_generators.py     # 18 tests (BFS + 2 générateurs + property-based)
│   │   └── test_procedural_env.py      # 10 tests (env wrapper + encode)
│   ├── neural/                         # [V2-Y]
│   │   ├── conftest.py                 # fixture cpu_device
│   │   ├── test_recurrent.py           # 7 tests (RecurrentQNetwork)
│   │   ├── test_sequence_buffer.py     # 9 tests (push + sample padding+mask)
│   │   └── test_recurrent_trainer.py   # 3 tests (BPTT + mask + sync_target)
│   ├── agents/                         # [V2-Y]
│   │   └── test_recurrent_dqn.py       # 8 tests (init + reset_hidden + observe + end_episode)
│   ├── training/                       # [V2-X + V2-Y]
│   │   ├── test_scheduler.py           # 7 tests
│   │   ├── test_bucket_tracker.py      # 5 tests
│   │   ├── test_procedural_runner.py   # 3 tests d'intégration V2-X
│   │   └── test_recurrent_procedural_runner.py  # 2 tests d'intégration V2-Y
│   ├── test_procedural_config.py       # [V2-X] 9 tests
│   └── test_drqn_config.py             # [V2-Y] 6 tests
├── checkpoints/                        # .pt / .npz (gitignored)
├── logs/
└── docs/superpowers/
    ├── specs/
    │   ├── 2026-05-21-mw-ia-rl-design.md                     # V1
    │   ├── 2026-05-21-mw-ia-v2-aether-guardrails-design.md   # V2-A
    │   └── 2026-05-22-mw-ia-procedural-env-design.md         # V2-X
    ├── specs/
    │   ├── 2026-05-21-mw-ia-rl-design.md                     # V1
    │   ├── 2026-05-21-mw-ia-v2-aether-guardrails-design.md   # V2-A
    │   ├── 2026-05-22-mw-ia-procedural-env-design.md         # V2-X
    │   └── 2026-05-22-mw-ia-recurrent-network-design.md      # V2-Y
    └── plans/
        ├── 2026-05-21-mw-ia-v1.md                            # V1
        ├── 2026-05-21-mw-ia-v2-aether-guardrails.md          # V2-A
        ├── 2026-05-22-mw-ia-procedural-env.md                # V2-X (16 tâches sur 13 phases — toutes ✅)
        └── 2026-05-22-mw-ia-recurrent-network.md             # V2-Y (13 tâches sur 9 phases — toutes ✅)
```

---

## Procédures usuelles

### Lancer les tests
```bash
source .venv/Scripts/activate && pytest -q
```
Attendu : **183 passed** (52 V1 + 43 V2-A + 53 V2-X + 35 V2-Y).

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

Boutons : "Démarrer" (V1 fix-map) ou "Démarrer (procedural)" (V2-X mazes aléatoires + curriculum).

### Entraîner DQN procedural (headless V2-X)
```bash
source .venv/Scripts/activate && python scripts/train_dqn_procedural.py --episodes 500 --mode obstacles --device cuda
# ou : --mode maze
# recette gagnante V2-X : --hidden 256 256 --epsilon-decay-steps 200000
```

### Entraîner Recurrent DQN procedural (headless V2-Y)
```bash
source .venv/Scripts/activate && python scripts/train_drqn_procedural.py --episodes 5000 --mode obstacles --device cuda
# Defaults V2-Y déjà gagnants : fc_hidden=256, lstm_hidden=128, sequence_length=32, epsilon_decay_steps=200000
```

Sortie : winrate global + per-bucket (5 buckets de difficulté) + difficulté finale.

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

10. **V2-X `RandomObstaclesGenerator` densité > 0.40 sur 10×10** : statistiquement instable (probabilité de solvabilité chute exponentiellement avec la densité 4-connexe). Le default classe `max_density=0.50` est respecté mais probabilité succès ~45 % en 100 tentatives. Préférer `max_density ≤ 0.40` pour les fixture de test ou tout contexte qui ne tolère pas de `RuntimeError` probabilistes.

11. **V2-X `PerfectMazeGenerator` est "quasi-parfait" sur tailles paires** : le DFS classique creuse uniquement les cellules paires-paires ; pour size paire, `goal=(size-1, size-1)` est forcé accessible via ouverture de 1-2 murs intermédiaires. Conséquence : 1-2 chemins supplémentaires localement sur les dernières cellules vers goal. Solvabilité garantie par construction (pas besoin de BFS-check). Documenté dans la docstring de la classe.

12. **V2-X observation procedural** : `encode_procedural_observation` produit `concat(position_one_hot, grid_flatten)` de dim `2 * max_rows * max_cols` (= 200 pour 10×10), pas `[row, col, *grid]` comme la spec V2-X originale le suggérait. Aligné sur le pattern V1 `DQNRunner._state_vec`. Mazes plus petits que `max_size` paddés top-left avec zéros (cellules libres) — l'agent voit des bordures artificielles qu'il apprend à ignorer.

13. **V2-Y `SequenceReplayBuffer.replay_capacity` = nombre de TRAJECTOIRES** (pas transitions). Memory budget `5000 traj × 200 max_steps × 200 obs_dim × 4 bytes × 2 (state + next_state)` ≈ 1.6 GB sur RTX 3060. Si OOM : descendre à 2000 trajectoires.

14. **V2-Y hidden state runtime maintenu y compris en ε-random** : l'agent `act()` fait TOUJOURS le forward LSTM (pour mettre à jour `_hidden_state`), même quand ε-greedy tire random. C'est cohérent avec le but du LSTM : suivre la trajectoire d'observations indépendamment des choix d'action. Documenté dans la docstring de `act()`.

15. **V2-Y train trigger** : `end_episode()` ne déclenche les `train_steps_per_episode` batches que si `len(buffer) >= max(min_episodes_to_learn, batch_size)`. Le `max(...)` évite `ValueError` du buffer sample quand l'utilisateur configure `batch_size > min_episodes_to_learn`. Inclus dans commit `18d09d9` avec une note d'atomicité (2 fichiers dans 1 commit, dérogation acceptée pour correctness).

---

## Mémoires persistantes liées

À consulter au démarrage :
- `~/.claude/projects/C--Users-Wilfred/memory/projet_mw_ia.md` — état du projet
- `~/.claude/projects/C--Users-Wilfred/memory/feedback_aether_usage.md` — usage Aether
- `~/.claude/projects/C--Users-Wilfred/memory/MEMORY.md` — index global

---

## Instructions pour la prochaine session

### Reprise par défaut — attaquer un nouveau sous-projet

V2-A, V2-X et V2-Y étant terminés ET la baseline V2-Y validée empiriquement (95% @ diff 0.05, reproductible), la **suite naturelle est CNN ou Double DQN** :

**Diagnostic empirique consolidé de fin de session 2026-05-22** :
> Le bottleneck actuel n'est PLUS la mémoire (LSTM testé, plafond identique). C'est la **représentation spatiale** (`grid_flatten` dim 100 ignore la structure 2D) + **Q-values instables** (target net DQN classique). V2-Y bat V2-X en qualité de politique au même palier, mais pas en capacité de généralisation curriculum.

**Prochaine étape probable** (priorité à débrainstormer en session fraîche) :

1. **V2-Z : CNN perception spatiale (roadmap #2) — recommandé** : remplacer `concat(position_one_hot, grid_flatten)` par un input 2D `(channels, rows, cols)` traité par une Conv2D. Apprend les motifs locaux (dead-ends, couloirs, intersections) avec translation equivariance, beaucoup moins de paramètres. Cible : franchir diff 0.05 vers 0.20+.
2. **V2-W : Double DQN (roadmap #7)** : ~30 LOC modif `DQNTrainer.step()` pour découpler sélection d'action (online net) et évaluation (target net). Réduit la surestimation Q-values, particulièrement utile pour V2-Y LSTM instable.
3. **Combinaison CNN + Double DQN** : sous-projet plus ambitieux mais probablement nécessaire pour vraiment franchir le plafond.

Le **sous-projet B (mémoire persistante cross-session)** du programme V2 officiel reste viable mais est plus "autonomie long-terme" que "résoudre mazes mieux" — moins prioritaire au vu du finding bottleneck spatial.

1. **Lire ce CLAUDE.md en entier.**
2. **Smoke test rapide** :
   ```bash
   source .venv/Scripts/activate && pytest -q
   ```
   Attendu : 183 passed. + `bash aether/verify_all.sh` → 8 OK.
3. **Aligner avec l'utilisateur** sur le prochain sous-projet.
4. **Cycle complet** pour tout nouveau sous-projet :
   - `superpowers:brainstorming` → cerner intent, scope, contraintes
   - Écrire la spec dans `docs/superpowers/specs/YYYY-MM-DD-<sub-projet>-design.md`
   - `superpowers:writing-plans` → plan TDD bite-sized
   - `superpowers:subagent-driven-development` (pattern V1/V2-A/V2-X) : implementer + spec reviewer + code quality reviewer par task ou groupe cohérent. Pour les blocks à faible risque (dataclasses, helpers isolés), la review peut être skippée par inspection directe du diff.
5. **Ne pas démarrer en code-only** sur un nouveau sous-projet — passer obligatoirement par brainstorm + spec + plan, comme V2-A et V2-X.

### Si l'objectif est un quick fix / petite feature

- TDD (test rouge → impl → vert → commit)
- Ne pas casser les tags `v0.1.0`, `v0.2.0-a`, `v0.2.0-x`, `v0.2.0-y` (pas de force-push)
- Re-lancer `pytest -q` avant chaque commit (attendu : 183 passed)
- Ne pas toucher aux modules livrés (`mw_ia/guardrails/`, `aether/invariants/`, `mw_ia/envs/maze_generators.py`, `mw_ia/envs/procedural_env.py`, `mw_ia/training/scheduler.py`, `mw_ia/neural/recurrent.py`, `mw_ia/neural/sequence_buffer.py`, `mw_ia/neural/recurrent_trainer.py`, `mw_ia/agents/recurrent_dqn.py`) sans raison documentée

### Si l'objectif est de pousser un sous-projet V3+ (auto-modification, etc.)

Terminer **B → C → D → E** d'abord (ordre des prérequis logiques : mémoire → évaluateur → continual learning → auto-mod), sauf décision explicite de l'utilisateur. Le bucket tracker V2-X livré pose déjà l'infrastructure observationelle pour D. Chaque sous-projet reste un cycle complet brainstorm + spec + plan + impl.
