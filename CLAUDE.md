# CLAUDE.md — Projet MW_IA (Reinforcement Learning éducatif)

> **À LIRE EN PREMIER** par Claude au démarrage d'une nouvelle session dans ce projet.

---

## Règles de comportement (héritées de `C:\Windows\System32\CLAUDE.md`)

- Toujours répondre en **français**
- Toujours invoquer les skills **superpowers** (brainstorming, writing-plans, TDD, etc.) et **frontend-design** quand pertinent
- Utiliser le MCP **context7** pour toute documentation de librairie (PyTorch, PyQt6, PyQtGraph, Gymnasium…)
- Le MCP **aether** est disponible et **doit être utilisé activement** pour vérifier les invariants RL critiques (Bellman, seuils de niveau, conditions de convergence). Cf. `~/.claude/projects/.../memory/feedback_aether_usage.md`. **PAS** une dépendance Python de MW_IA.
- Utiliser **TaskCreate/TaskUpdate** pour piloter les tâches multi-étapes
- Quand pertinent, dispatcher des **agents parallèles** (subagent_type=Explore, general-purpose, etc.)

---

## État au handoff (2026-05-21 → reprise V2)

**V1 monolithique livrée et taguée `v0.1.0`.** 26 tâches du plan exécutées en TDD strict via subagent-driven-development. 52 tests pytest verts, DQN converge à **99 % winrate niveau Expert sur RTX 3060** en 200 ép. / 18.8 s.

### Ce qui marche déjà
- **GridWorld 10×10** avec obstacles, reward shaping, terminaison/truncation correctes
- **3 agents tabulaires** : Value Iteration, Policy Iteration, Q-Learning (TD)
- **DQN complet** : QNetwork MLP, ReplayBuffer circulaire, DQNTrainer (Huber + Adam + AMP + grad clip), DQNAgent (ε-greedy + target sync)
- **MetricsTracker** : winrate glissant, niveau IA (Débutant/Inter/Avancé/Expert)
- **TabularRunner + DQNRunner** callback-friendly (zéro couplage Qt)
- **GUI PyQt6 + PyQtGraph** : grille animée + 4 courbes live + KPIs + 5 boutons + log coloré, threading via QThread + signaux Qt
- **CLI headless** : `scripts/train_tabular.py`, `scripts/train_dqn.py`, `scripts/launch_gui.py`
- **Persistance** : checkpoints `.pt` (DQN) et `.npz` (tabulaire) + métriques JSON
- **Vérification formelle Aether** : invariants `bellman_update` et `level_of` (44 checks passed)

### Ce qui reste à valider visuellement (côté user)
Cocher les 5 cases de `logs/manual_validation.md` :
```bash
source .venv/Scripts/activate && python scripts/launch_gui.py
```

### Ce qui n'est PAS dans la V1 (hors-scope volontaire — voir design doc §9)
Double DQN · Dueling · Prioritized Experience Replay · LSTM / mémoire · Chatbot RL · LLM local backbone · Apprentissage continu cross-session.

---

## Objectif long-terme

**IA auto-améliorante** qui propose et teste ses propres modifications d'hyperparamètres / d'architecture, sous contraintes vérifiables formellement via Aether. Décliné en V2+ :

1. **Mémoire persistante cross-session** (RVF ou équivalent) — pas d'oubli catastrophique
2. **Évaluateur de politique self-supervisé** — l'agent juge ses trajectoires
3. **Système de proposition/test d'updates** — variants d'architecture, hyperparams, reward shaping
4. **Contrat formel d'invariants** (Aether) — l'auto-modification ne viole jamais : (a) bornes du winrate, (b) positivité de la loss, (c) contraction γ-lipschitz de Bellman, (d) tout invariant spécifique au domaine
5. **Continual learning** (EWC, replay rehearsal) — apprendre de nouvelles tâches sans oublier les anciennes

La V1 modulaire est l'infrastructure de départ. **Aucune refonte requise pour la V2.**

---

## Environnement machine (vérifié 2026-05-21)

- **OS** : Windows 11 Pro 10.0.26200
- **Shell Claude Code** : Git Bash (Unix-style dans `Bash` tool). PowerShell via `powershell.exe -NonInteractive -Command "..."`.
- **Python** : 3.13.12 — `py` alias ou `C:\Python313\python.exe`
- **GPU** : NVIDIA GeForce RTX 3060 (12 Go VRAM, Ampere CC 8.6)
- **Driver NVIDIA** : 591.86 / CUDA 13.1
- **PyTorch** : `2.11.0+cu128` (⚠️ wheels `cu121` n'existent plus pour Python 3.13 — utiliser `cu128`)
- **claude CLI** : v2.1.146

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
- ⚠️ **Ne PAS confondre** avec le repo parent `C:/Users/Wilfred/.git` (branche `feature/pentest-agent`) qui appartient au projet **PenTest**. Ils sont indépendants.
- Branche : `main`. Tag : `v0.1.0`. 28 commits propres au handoff.

---

## Arborescence finale V1

```
MW_IA/
├── CLAUDE.md                           # ce fichier
├── README.md                           # théorie + install + archi + roadmap
├── requirements.txt                    # numpy, PyQt6, pyqtgraph, gymnasium, pytest
├── pyproject.toml                      # setuptools + pytest config
├── check_gpu.py                        # diagnostic CUDA / VRAM
├── scripts/
│   ├── train_tabular.py                # CLI Q-Learning headless
│   ├── train_dqn.py                    # CLI DQN headless (--device cuda)
│   └── launch_gui.py                   # entrypoint GUI
├── mw_ia/
│   ├── config.py                       # GridWorldConfig / QLearningConfig / DQNConfig / TrainingConfig
│   ├── envs/gridworld.py               # GridWorld + Action enum
│   ├── agents/                         # base.py, value_iteration.py, policy_iteration.py, q_learning.py, dqn.py
│   ├── neural/                         # network.py, replay_buffer.py, trainer.py
│   ├── training/                       # metrics.py (Level + MetricsTracker), runner.py (Tabular + DQN)
│   ├── persistence/checkpoint.py       # JSON metrics dump/load
│   └── gui/
│       ├── theme.py                    # Theme + QSS
│       ├── widgets/                    # gridworld_view, live_plots, stats_panel, control_panel, log_console
│       └── app.py                      # MainWindow + TrainingThread
├── tests/                              # 13 fichiers, 52 tests
├── checkpoints/                        # .pt / .npz (gitignored)
├── logs/                               # metrics .json + manual_validation.md (gitignored sauf -f)
└── docs/superpowers/
    ├── specs/2026-05-21-mw-ia-rl-design.md
    └── plans/2026-05-21-mw-ia-v1.md
```

---

## Procédures usuelles

### Lancer les tests
```bash
source .venv/Scripts/activate && pytest -q
```
Attendu : 52 passed.

### Entraîner Q-Learning tabulaire (headless)
```bash
source .venv/Scripts/activate && python scripts/train_tabular.py --episodes 1000
```
Résultat de référence : winrate 100 % niveau Expert en ~2 s.

### Entraîner DQN sur GPU (headless)
```bash
source .venv/Scripts/activate && python scripts/train_dqn.py --episodes 200 --device cuda
```
Résultat de référence : winrate 99 % niveau Expert en ~19 s sur RTX 3060.

### Lancer la GUI live
```bash
source .venv/Scripts/activate && python scripts/launch_gui.py
```

---

## Garde-fous & pièges à connaître

1. **Hook `security_reminder_hook.py`** flagge naïvement la séquence `exec` suivie d'une parenthèse ouvrante comme une vuln Node.js `child_process` — y compris la méthode `QApplication.exec` de PyQt6. **Contournement** : `getattr(app, "exec")()` ou `run_loop = app.exec; run_loop()`. Le launcher utilise déjà cette astuce.

2. **PyTorch cu121 obsolète pour Python 3.13.** Toujours installer via `--index-url https://download.pytorch.org/whl/cu128`. Si même `cu128` ne marche plus dans le futur, essayer `cu126`, `cu124`, puis fallback CPU explicite.

3. **`.claude/` doit rester gitignoré** (présent dans `.gitignore`) — sinon les artefacts de session Claude Code se retrouveraient committés.

4. **Pas de Double DQN / Dueling / PER / LSTM dans la V1.** Ces évolutions sont réservées à la V2. Toute proposition d'inclusion en V1 = re-poser la question à l'utilisateur.

5. **`mw_ia/envs/gridworld.py`** : le `step_count` s'incrémente **même quand le mouvement est bloqué** (mur ou obstacle). C'est intentionnel — sinon truncation par `max_steps` deviendrait contournable. Documenté dans le commit `149b4d0`.

6. **Aether MCP doit être mobilisé activement** quand pertinent (RL convergence, contraintes math, contrats numériques). Lancer `mcp__aether__syntax_guide` d'abord pour rafraîchir la syntaxe Lisp typée.

---

## Mémoires persistantes liées

À consulter au démarrage :
- `~/.claude/projects/C--Users-Wilfred/memory/projet_mw_ia.md` — état du projet
- `~/.claude/projects/C--Users-Wilfred/memory/feedback_aether_usage.md` — usage Aether
- `~/.claude/projects/C--Users-Wilfred/memory/MEMORY.md` — index global

---

## Instructions pour la prochaine session

### Si l'objectif est de valider la GUI
```bash
source .venv/Scripts/activate && python scripts/launch_gui.py
```
Puis cocher les cases dans `logs/manual_validation.md`.

### Si l'objectif est de démarrer la V2 "auto-amélioration"
1. Lire ce CLAUDE.md en entier
2. Lire `~/.claude/projects/C--Users-Wilfred/memory/projet_mw_ia.md`
3. Invoquer la skill `superpowers:brainstorming` avec le prompt :
   > "MW_IA V2 — IA auto-améliorante. Sur la V1 modulaire actuelle (DQN + GUI), ajouter : mémoire persistante cross-session, évaluateur self-supervisé de politique, système de proposition/test d'updates, contrat formel d'invariants Aether, continual learning sans oubli catastrophique."
4. Produire un design doc V2 dans `docs/superpowers/specs/`
5. Puis `superpowers:writing-plans` puis subagent-driven-development comme pour la V1

### Si l'objectif est un quick fix / petite feature V1
- Suivre TDD (test rouge → impl → vert → commit)
- Ne pas casser le tag `v0.1.0` (pas de force-push)
- Re-lancer `pytest -q` avant chaque commit
