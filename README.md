# MW_IA — Reinforcement Learning éducatif

MW_IA est un projet pédagogique de Reinforcement Learning **construit de zéro**, allant de l'équation de Bellman au Deep Q-Network sur GPU, avec une **interface graphique PyQt6** qui visualise l'apprentissage en temps réel.

## Pourquoi ce projet ?

Comprendre RL en *voyant* un agent apprendre est mille fois plus parlant qu'un papier. MW_IA propose :

- Le **MDP minimal viable** (GridWorld 10×10 avec obstacles)
- Trois agents tabulaires : **Value Iteration**, **Policy Iteration**, **Q-Learning**
- Le passage au **Deep Q-Network** (PyTorch + GPU + mixed precision)
- Une **GUI moderne dark** avec grille animée, 4 courbes live (reward, loss, ε, winrate), KPIs, niveau IA (Débutant → Expert) et boutons Start / Pause / Reset / Save / Load

## Théorie

### Équation de Bellman (optimalité)

```
V*(s) = max_a Σ_s' P(s'|s,a) [R(s,a,s') + γ V*(s')]
```

### Mise à jour Q-Learning (sans modèle, off-policy)

```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
```

### Loss DQN (Huber)

```
L(θ) = E[ smooth_L1( r + γ max_a' Q(s',a'; θ⁻) , Q(s,a; θ) ) ]
```

Où **θ⁻** est un *target network* synchronisé toutes les `target_sync_steps`.

## Installation

```bash
py -m venv .venv
source .venv/Scripts/activate          # Git Bash sous Windows
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
python check_gpu.py
```

> **Note Python 3.13 + CUDA** : les wheels `cu121` ne sont plus publiés pour Python 3.13. Utiliser **`cu128`** (CUDA 12.8) qui supporte les drivers NVIDIA récents (591+) et reste compatible Ampere (RTX 3060). Si CUDA indisponible :
>
> ```bash
> pip install torch torchvision           # wheels CPU PyPI
> ```
>
> Tous les entraînements supportent `--device cpu`.

## Lancement

```bash
# Tests
pytest -q

# Entraînement headless tabulaire
python scripts/train_tabular.py --episodes 1000

# Entraînement headless DQN
python scripts/train_dqn.py --episodes 500 --device cuda

# GUI complète (live)
python scripts/launch_gui.py
```

## Architecture

```
mw_ia/
├── config.py            # hyperparamètres centralisés (dataclasses frozen)
├── envs/gridworld.py    # GridWorld + obstacles + reward shaping
├── agents/              # base, value_iteration, policy_iteration, q_learning, dqn
├── neural/              # QNetwork, ReplayBuffer, DQNTrainer (AMP, Huber, grad clip)
├── training/            # MetricsTracker (winrate + Level), TabularRunner, DQNRunner
├── persistence/         # checkpoint JSON helpers
└── gui/                 # PyQt6 + PyQtGraph (theme + widgets + MainWindow + TrainingThread)
```

Le `TrainingRunner` est **callback-friendly** (pas de dépendance Qt) ; un `QThread` mince fait le pont vers la GUI via `pyqtSignal`. Pas de polling, pas de partage d'état mutable entre threads.

## Critères de succès V1

- `check_gpu.py` détecte le GPU et la VRAM
- `train_tabular.py` converge sur GridWorld 10×10 (winrate ≈ 100 % niveau Expert en ~200 ép.)
- `train_dqn.py` tourne sans NaN, la loss décroît
- GUI : grille animée, 4 courbes live, Start/Pause/Reset/Save/Load fonctionnels
- `pytest tests/` au vert

## Roadmap (V2+)

Architecture pensée pour ajouter sans refonte :

- **Double DQN** → modifier `agents/dqn.py` (calcul du target)
- **Dueling DQN** → modifier `neural/network.py` (têtes V + A)
- **Prioritized Experience Replay** → swap `neural/replay_buffer.py`
- **LSTM / mémoire** → nouveau `neural/recurrent.py`
- **Chatbot RL** → `envs/dialog_env.py` + `agents/chatbot_agent.py`
- **LLM local backbone** → `neural/llm_backbone.py` (Ollama / llama-cpp)

Objectif long-terme : IA **auto-améliorante** qui propose et teste ses propres modifications d'hyperparamètres / d'architecture, sous contraintes vérifiables formellement (Aether MCP).

## Licence

À définir avant publication.
