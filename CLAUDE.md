# CLAUDE.md — Projet MW_IA (Reinforcement Learning éducatif)

> **À LIRE EN PREMIER** par Claude au démarrage d'une nouvelle session dans ce projet.

---

## Règles de comportement (héritées de C:\Windows\System32\CLAUDE.md)

- Toujours répondre en **français**
- Toujours invoquer les skills **superpowers** (brainstorming, writing-plans, TDD, etc.) et **frontend-design** quand pertinent
- Utiliser le MCP **context7** pour toute documentation de librairie (PyTorch, PyQt6, PyQtGraph, Gymnasium...)
- Le MCP **aether** est désormais disponible (installé au user scope) — outil de dev, **PAS** une dépendance de MW_IA
- Utiliser **TaskCreate/TaskUpdate** pour piloter les tâches multi-étapes
- Quand pertinent, dispatcher des **agents parallèles** (subagent_type=Explore, general-purpose, etc.)

## État au moment du handoff (2026-05-21)

Phase atteinte : **brainstorming terminé + design doc validé**. Reste à faire :
1. Invoquer la skill `superpowers:writing-plans` pour générer le plan d'implémentation détaillé
2. Implémenter la V1 monolithique complète (voir checklist plus bas)
3. Lancer un premier entraînement Q-Learning sur GridWorld pour valider la chaîne

**Le design doc à lire absolument** : [`docs/superpowers/specs/2026-05-21-mw-ia-rl-design.md`](docs/superpowers/specs/2026-05-21-mw-ia-rl-design.md)

## Objectif du projet (résumé)

Construire de zéro une IA RL **pédagogique et évolutive** :
- Théorie : MDP, V(s), Q(s,a), Bellman, Value Iteration, Policy Iteration, Q-Learning, DQN
- Stack : Python 3.13 + PyTorch CUDA 12.1 + PyQt6 + PyQtGraph + NumPy + Gymnasium
- GPU cible : **NVIDIA GeForce RTX 3060** (Ampere, ~12 Go VRAM)
- GUI moderne temps réel : grille animée + 4 courbes (reward/loss/epsilon/winrate) + stats + niveau Débutant→Expert + boutons Start/Pause/Reset/Save/Load + log
- Évolution future préparée : Double DQN, Dueling, Prioritized ER, LSTM, chatbot RL avec mémoire, LLM local

## Décisions verrouillées

| # | Décision | Détail |
|---|---|---|
| 1 | **Architecture Option A** | Modulaire en couches (envs/agents/neural/training/gui/persistence) — voir design doc §4 |
| 2 | **GUI = PyQt6 + PyQtGraph** | Rendu OpenGL, 60+ FPS, courbes fluides temps réel |
| 3 | **Phasage = V1 monolithique complet** | Tout livré d'un coup : tabulaire + DQN + GUI + tests |
| 4 | **Threading = QThread + signaux Qt** | Pas de polling, pas de freeze UI |
| 5 | **Mixed precision (AMP) activée** sur DQN | `torch.amp` côté CUDA |
| 6 | **Batch DQN = 128, lr Adam = 1e-3, Huber Loss** | Defaults configurables dans `mw_ia/config.py` |
| 7 | **GridWorld 10×10** par défaut | Visualisable + tractable tabulairement |
| 8 | **Aether MCP = outil dev externe** | N'est PAS embarqué dans MW_IA, pas de dépendance |
| 9 | **Installation** | Le user veut : créer venv + installer torch CUDA + lancer un premier entraînement de validation |
| 10 | **Niveau IA** | Calculé sur winrate glissant 100 ép. — seuils 30/60/85% |

## Environnement machine

- **OS :** Windows 11 Pro 10.0.26200
- **Shell Claude Code :** bash (syntaxe Unix dans Bash tool, pas PowerShell)
- **Python :** 3.13.12 — `py` (alias) ou `C:\Python313\python.exe`
- **GPU :** NVIDIA GeForce RTX 3060
- **claude CLI :** v2.1.146
- **Pour PowerShell depuis bash :** `powershell.exe -NonInteractive -Command "..."`

## MCPs disponibles (vérifiés au handoff)

```
aether                          ✓ Connected (user scope) — Aether language, 14 tools
plugin:context7:context7        ✓ Connected — docs librairies
plugin:playwright:playwright    ✓ Connected — automation browser
claude.ai Google Drive/Cal/Gmail — auth requise si besoin
```

Aether MCP path : `C:\Users\Wilfred\Tools\aether\aether_mcp.py`

## Checklist V1 (ordre suggéré d'implémentation)

> Cette liste sera reprise par `writing-plans` pour devenir un vrai plan détaillé.
> **Important :** invoquer `superpowers:writing-plans` AVANT de toucher au code.

### Étape 0 — Plan
- [ ] Invoquer `superpowers:writing-plans` avec le design doc comme input

### Étape 1 — Scaffolding
- [ ] Créer venv : `py -m venv .venv` puis activer
- [ ] Écrire `requirements.txt` (torch cu121, pyqt6, pyqtgraph, numpy, gymnasium, pytest)
- [ ] `pip install -r requirements.txt` (≈3 Go DL pour torch CUDA, prévoir ~5-10 min)
- [ ] Écrire `check_gpu.py` et l'exécuter → doit afficher "RTX 3060 + CUDA OK"
- [ ] Créer arborescence `mw_ia/` (envs/agents/neural/training/gui/persistence) avec `__init__.py`
- [ ] Écrire `mw_ia/config.py` (dataclass des hyperparamètres)
- [ ] `.gitignore` : `.venv/`, `checkpoints/`, `logs/`, `__pycache__/`, `*.pt`

### Étape 2 — Environment
- [ ] `mw_ia/envs/gridworld.py` : GridWorld 10×10, obstacles configurables, reward shaping, API `reset()` / `step(action)` style Gymnasium
- [ ] `tests/test_gridworld.py` : reset déterministe, step retourne tuple correct, terminal sur goal

### Étape 3 — Agents tabulaires
- [ ] `mw_ia/agents/base.py` : interface `Agent` (act, learn, save, load)
- [ ] `mw_ia/agents/value_iteration.py` : converge sur GridWorld connu
- [ ] `mw_ia/agents/policy_iteration.py`
- [ ] `mw_ia/agents/q_learning.py` : Q-table, ε-greedy, decay
- [ ] `tests/test_q_learning.py` : converge en <1000 ép. sur GridWorld 5×5

### Étape 4 — DQN neural
- [ ] `mw_ia/neural/network.py` : QNetwork PyTorch (FC + ReLU, archi configurable nb couches/tailles)
- [ ] `mw_ia/neural/replay_buffer.py` : buffer circulaire, sample uniforme
- [ ] `mw_ia/neural/trainer.py` : backprop, Huber Loss, Adam, AMP `torch.amp.autocast`
- [ ] `mw_ia/agents/dqn.py` : orchestrateur (ε-greedy, target network sync, train step)
- [ ] `mw_ia/persistence/checkpoint.py` : save/load `.pt` + métriques JSON
- [ ] `tests/test_dqn_smoke.py` : 100 steps sans crash, save/load round-trip

### Étape 5 — Training pipeline
- [ ] `mw_ia/training/metrics.py` : MetricsTracker (winrate glissant, moyenne, best, FPS)
- [ ] `mw_ia/training/runner.py` : boucle d'entraînement compatible QThread (émet signaux)
- [ ] `scripts/train_tabular.py` : CLI headless Q-Learning
- [ ] `scripts/train_dqn.py` : CLI headless DQN (avec `--episodes`, `--device`)

### Étape 6 — GUI PyQt6
- [ ] `mw_ia/gui/theme.py` : palette dark moderne
- [ ] `mw_ia/gui/widgets/gridworld_view.py` : QGraphicsScene animée (agent, obstacles, goal, trail optionnel)
- [ ] `mw_ia/gui/widgets/live_plots.py` : 4 plots PyQtGraph (reward, loss, epsilon, winrate)
- [ ] `mw_ia/gui/widgets/stats_panel.py` : KPIs + label niveau (Débutant→Expert)
- [ ] `mw_ia/gui/widgets/control_panel.py` : 5 boutons (Start/Pause/Reset/Save/Load)
- [ ] `mw_ia/gui/widgets/log_console.py` : QPlainTextEdit append-only
- [ ] `mw_ia/gui/app.py` : MainWindow, layout, connexion signaux/slots, TrainingThread
- [ ] `scripts/launch_gui.py` : entrypoint

### Étape 7 — Documentation
- [ ] `README.md` : présentation, install, théorie (formules Bellman + Q-Learning), screenshots, roadmap
- [ ] Commenter le code en français, surtout les passages pédagogiques

### Étape 8 — Validation finale
- [ ] `pytest tests/` → tout vert
- [ ] `py scripts/train_tabular.py` → converge GridWorld 10×10 en <1 min
- [ ] `py scripts/train_dqn.py --episodes 200` → loss décroît, pas de NaN
- [ ] `py scripts/launch_gui.py` → fenêtre s'ouvre, entraînement live visible, boutons fonctionnels

## Tâches en cours (TaskList au moment du handoff)

```
#1. [completed] Présenter architecture proposée + obtenir approbation
#2. [completed] Écrire et committer le design doc
#3. [pending]   Invoquer writing-plans pour le plan d'implémentation   ← REPRENDRE ICI
#4. [pending]   Implémenter V1 monolithique complète
#5. [completed] Installer le MCP Aether dans Claude Code
```

## Instructions précises pour la prochaine session

1. **Lis ce CLAUDE.md en entier**
2. **Lis le design doc** `docs/superpowers/specs/2026-05-21-mw-ia-rl-design.md`
3. Vérifie l'état des tâches via `TaskList` (si elles persistent) ou recrée-les depuis la checklist ci-dessus
4. Invoque la skill `superpowers:writing-plans` avec le design doc comme spec d'entrée
5. Présente le plan d'implémentation au user, attends son feu vert
6. Exécute le plan en suivant la checklist Étape 1 → 8
7. Quand `pip install torch` est lancé, prévoir un timeout généreux (5-10 min, ~3 Go)
8. Avant de claim "fait", lance vraiment `py check_gpu.py` puis `py scripts/launch_gui.py` pour vérifier visuellement
9. Si CUDA n'est pas dispo sur cette install Python, dire le user et proposer fallback CPU explicite

## Théorie à inclure dans le README (rappel)

```
Équation de Bellman (optimalité) :
  V*(s) = max_a Σ_s' P(s'|s,a) [R(s,a,s') + γ V*(s')]

Mise à jour Q-Learning :
  Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

DQN :
  L(θ) = E[(r + γ max_a' Q(s',a'; θ⁻) - Q(s,a; θ))²]  (Huber au lieu de carré en pratique)
```

## Garde-fous

- ❌ Ne PAS modifier `C:\Windows\System32\CLAUDE.md`
- ❌ Ne PAS ajouter Aether comme dépendance Python dans `requirements.txt`
- ❌ Ne PAS implémenter Double DQN / Dueling / PER / LSTM dans la V1 (hors-scope)
- ✅ Le design doc est la **source de vérité** pour l'archi. Pour s'en écarter : reposer la question au user.
- ✅ Si une lib pose problème (PyQt6 install, torch CUDA), utiliser `context7` AVANT de proposer un workaround
