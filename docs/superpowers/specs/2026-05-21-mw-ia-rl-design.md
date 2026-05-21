# MW_IA — Design Document V1

**Date :** 2026-05-21
**Statut :** APPROUVÉ (Option A — architecture modulaire en couches)
**Auteur conversation :** Wilfred + Claude (Opus 4.7)

---

## 1. Objectif

Construire **de zéro** une IA d'apprentissage par renforcement, pédagogique et évolutive, partant de l'équation de Bellman et allant jusqu'au Deep Q-Network sur GPU, avec une **interface graphique moderne** visualisant l'apprentissage en temps réel.

Le projet doit servir de base pour évoluer ensuite vers un **agent autonome / chatbot RL avec mémoire conversationnelle**.

## 2. Matériel cible

- **GPU :** NVIDIA GeForce RTX 3060 (Ampere, Compute Capability 8.6, 12 Go VRAM nominal)
- **OS :** Windows 11 Pro 10.0.26200
- **Python :** 3.13.12 (`C:\Python313\python.exe`, alias `py`)
- **CUDA wheel cible :** PyTorch cu121 (compatible Ampere)
- **Fallback CPU obligatoire** si CUDA indisponible

## 3. Théorie obligatoire (à expliquer dans le README)

- Processus de décision markovien (MDP) — états, actions, récompenses, transitions
- Fonction de valeur **V(s)** et fonction de valeur d'action **Q(s,a)**
- **Équation de Bellman** :
  ```
  V(s) = max_a Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s')]
  ```
- **Value Iteration** (planning exact si MDP connu)
- **Policy Iteration** (évaluation + amélioration)
- **Q-Learning** tabulaire (model-free, off-policy) :
  ```
  Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
  ```
- **Deep Q-Network (DQN)** — approximation de Q par réseau neuronal, replay buffer, target network, ε-greedy

## 4. Architecture (Option A — validée)

```
MW_IA/
├── README.md                    # Théorie + formules + roadmap
├── requirements.txt             # torch (CUDA 12.1), PyQt6, pyqtgraph, numpy, gymnasium
├── check_gpu.py                 # Diagnostic CUDA / VRAM / device
├── CLAUDE.md                    # Handoff pour reprise de session
│
├── mw_ia/                       # Package principal
│   ├── __init__.py
│   ├── config.py                # Hyperparamètres centralisés (dataclass)
│   │
│   ├── envs/                    # Environnements RL
│   │   ├── __init__.py
│   │   └── gridworld.py         # GridWorld + obstacles + reward shaping
│   │
│   ├── agents/                  # Algorithmes RL
│   │   ├── __init__.py
│   │   ├── base.py              # Interface Agent commune (act, learn, save, load)
│   │   ├── value_iteration.py   # Bellman exact (planning, MDP connu)
│   │   ├── policy_iteration.py  # Évaluation + amélioration
│   │   ├── q_learning.py        # Q-table tabulaire (model-free)
│   │   └── dqn.py               # DQN (orchestrateur replay/target/epsilon)
│   │
│   ├── neural/                  # Brique neuronale isolée (PyTorch)
│   │   ├── __init__.py
│   │   ├── network.py           # QNetwork PyTorch configurable (FC + ReLU)
│   │   ├── replay_buffer.py     # ReplayBuffer (préparé pour Prioritized ER)
│   │   └── trainer.py           # Backprop + Huber Loss + Adam + AMP (mixed prec)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── runner.py            # Boucle d'entraînement (QThread-friendly)
│   │   └── metrics.py           # MetricsTracker (signaux Qt vers GUI)
│   │
│   ├── gui/                     # PyQt6 + PyQtGraph (moderne, dark theme)
│   │   ├── __init__.py
│   │   ├── app.py               # MainWindow + QThread training
│   │   ├── theme.py             # Couleurs / fonts modernes
│   │   └── widgets/
│   │       ├── __init__.py
│   │       ├── gridworld_view.py   # QGraphicsScene animée (agent, obstacles, goal)
│   │       ├── live_plots.py       # PyQtGraph : reward/loss/epsilon/winrate
│   │       ├── stats_panel.py      # KPIs + niveau (Débutant→Expert)
│   │       ├── control_panel.py    # Start/Pause/Reset/Save/Load
│   │       └── log_console.py      # Journal actions/erreurs
│   │
│   └── persistence/
│       ├── __init__.py
│       └── checkpoint.py        # Save/load poids (.pt) + métriques (.json)
│
├── scripts/
│   ├── train_tabular.py         # CLI : Value Iteration + Q-Learning headless
│   ├── train_dqn.py             # CLI : entraînement DQN headless
│   └── launch_gui.py            # Point d'entrée GUI principal
│
├── checkpoints/                 # Poids sauvegardés (gitignore)
├── logs/                        # Métriques JSON exportées (gitignore)
├── docs/
│   └── superpowers/specs/       # Ce document
└── tests/
    ├── __init__.py
    ├── test_gridworld.py
    ├── test_q_learning.py
    └── test_dqn_smoke.py        # 100 steps pour vérifier la chaîne
```

## 5. Choix techniques verrouillés

| Aspect | Choix | Justification |
|---|---|---|
| GUI framework | **PyQt6 + PyQtGraph** | Rendu OpenGL accéléré, 60+ FPS, courbes temps réel fluides |
| Threading | `QThread` pour l'entraînement, **signaux Qt** vers GUI | Pas de freeze UI, animation fluide |
| Mixed precision | `torch.amp.autocast(device_type='cuda')` sur DQN | Gain VRAM/vitesse sur Ampere |
| Batch size DQN | **128** (configurable dans `config.py`) | Sweet spot RTX 3060 |
| Optimizer | **Adam** (lr=1e-3 initial, configurable) | Spec utilisateur |
| Loss | **Huber Loss** (`SmoothL1Loss`) | Plus stable que MSE sur RL |
| GridWorld | **10×10 par défaut**, configurable | Visualisable + tractable en tabulaire |
| Replay buffer | 100k transitions, échantillonnage uniforme V1 | Préparé pour Prioritized ER V2 |
| ε-greedy | linear decay 1.0 → 0.05 sur 50k steps | Standard DQN |
| Target network | sync hard tous les 1000 steps | Standard DQN |
| Niveau IA | calculé sur **winrate glissant 100 ép.** : Débutant <30%, Inter 30-60%, Avancé 60-85%, Expert >85% | Critère mesurable et stable |
| PyTorch install | `torch --index-url https://download.pytorch.org/whl/cu121` | Ampere compatible |

## 6. Communication GUI ↔ Training

Le `TrainingRunner` tourne dans un `QThread`. Il émet des signaux Qt à fréquence contrôlée :

- `step_completed(state, action, reward, next_state)` → GridWorldView anime
- `episode_completed(ep_num, reward, length, success)` → Plots + Stats + Niveau
- `loss_updated(step, loss_value)` → LivePlots (loss DQN)
- `epsilon_updated(step, epsilon)` → LivePlots (epsilon decay)
- `log_message(level, text)` → LogConsole

Pas de polling. Pas de partage d'état mutable entre threads. La GUI consomme passivement.

## 7. Évolution préparée (pas dans V1, mais l'archi le supporte)

- **Double DQN** → modifier uniquement `mw_ia/agents/dqn.py` (calcul du target)
- **Dueling DQN** → modifier uniquement `mw_ia/neural/network.py` (2 têtes V + A)
- **Prioritized Experience Replay** → swap `mw_ia/neural/replay_buffer.py` (déjà isolé)
- **LSTM / mémoire** → nouveau `mw_ia/neural/recurrent.py`, importable depuis dqn.py
- **Chatbot RL** → nouveau `mw_ia/envs/dialog_env.py` + `mw_ia/agents/chatbot_agent.py`
- **LLM local** → nouveau `mw_ia/neural/llm_backbone.py` (Ollama / llama-cpp)

## 8. Critères d'acceptation V1

- [ ] `py check_gpu.py` affiche CUDA disponible + RTX 3060 + VRAM
- [ ] `py scripts/train_tabular.py` converge sur GridWorld 10×10 en <1 min
- [ ] `py scripts/train_dqn.py --episodes 200` tourne sans crash, loss décroît
- [ ] `py scripts/launch_gui.py` ouvre la fenêtre, démarrer/pause/reset fonctionnent
- [ ] Les 4 courbes (reward, loss, epsilon, winrate) se mettent à jour pendant l'entraînement
- [ ] Save/Load d'un checkpoint round-trip identique
- [ ] `pytest tests/` passe en vert (au moins le smoke DQN)

## 9. Hors-scope V1 (à venir V2+)

- Double DQN / Dueling / Prioritized ER
- LSTM / mémoire conversationnelle
- LLM local
- Chatbot
- Apprentissage continu inter-sessions
- Gymnasium au-delà du wrapper GridWorld minimal

## 10. Lien avec Aether MCP

Décision actée : Aether reste un **outil de dev externe** pour Claude, **n'est pas embarqué dans MW_IA**. Le MCP est installé au user scope et disponible dans les futures sessions Claude Code. Aucune dépendance Aether dans `requirements.txt`.
