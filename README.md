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

## V2-A — Aether guardrails (sous-projet livré)

Catalogue de **8 invariants RL** formalises en Aether v1.4 (validation par
`@example` + `@invariant` via le test runner Aether) et re-testes en runtime
via property-based testing Python. API publique consommable par le futur
sous-projet E (auto-modification).

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
    print("OK variant valide")
else:
    for v in report.violations:
        print(f"  X {v.invariant_id} : {v.message}")
```

Pour les contextes "tout ou rien" (CI, pre-commit) :

```python
from mw_ia.guardrails import verify_or_raise, InvariantViolationError

try:
    verify_or_raise(spec)
except InvariantViolationError as e:
    # e.report contient la liste complete des violations
    raise
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
- `aether/invariants/*.aether` — validations Aether v1.4 versionnées
- `tests/guardrails/test_aether_python_sync.py` — vérifie la cohérence
  `aether/invariants/iN_*.aether` ↔ `@invariant("IN")` Python

### Nature de la validation Aether

Aether v1.4 est un **interpréteur Lisp typé avec test runner property-based**,
pas un theorem prover SMT. Les fichiers `.aether` ne sont donc pas des
*preuves formelles universelles* mais des validations déclaratives (`@example`
+ `@invariant`) exécutées par `mcp__aether__aether_verify`. Une vraie
vérification universelle (Z3 / Lean / Coq) reste possible mais hors-scope V2-A.

## V2-X — Environnement procédural & curriculum learning (sous-projet livré)

Génération de labyrinthes solvables à chaque épisode + scheduler adaptatif de
difficulté piloté par le winrate. Fait passer l'agent d'un "résolveur de map
fixe" à un "résolveur de labyrinthes en général".

### Usage CLI

```bash
# Mode obstacles aléatoires (densité variable avec la difficulté)
python scripts/train_dqn_procedural.py --episodes 500 --mode obstacles --device cuda

# Mode maze parfait (taille variable avec la difficulté)
python scripts/train_dqn_procedural.py --episodes 500 --mode maze --device cuda
```

### Garantie de solvabilité

Tout maze généré est **garanti solvable** :
- Mode `obstacles` : check BFS post-hoc, regénère jusqu'à 100 tentatives avant `RuntimeError`.
- Mode `maze` : DFS recursive backtracker, solvable par construction (quasi-parfait pour tailles paires : goal forcé accessible).

### Curriculum adaptatif

Le scheduler ajuste la difficulté toutes les 50 épisodes :
- winrate >= 80 % → difficulté monte de 0.05
- winrate <= 30 % → difficulté descend de 0.05
- sinon : inchangé

Métriques par bucket de difficulté (5 buckets [0,0.2)..[0.8,1.0]) pour détecter
l'oubli catastrophique des niveaux faciles (préfigure le sous-projet D).

### GUI

`python scripts/launch_gui.py` puis bouton "Démarrer (procedural)" : la grille
redessine à chaque épisode, une 5e courbe `difficulty(t)` s'ajoute aux 4 V1, et
un label "Maze #N, diff=X.XX" suit l'évolution.

### Architecture

- `mw_ia/envs/maze_generators.py` — `maze_bfs_check`, `RandomObstaclesGenerator`, `PerfectMazeGenerator`
- `mw_ia/envs/procedural_env.py` — `ProceduralGridWorld` + `encode_procedural_observation`
- `mw_ia/training/scheduler.py` — `AdaptiveDifficultyScheduler`
- `mw_ia/training/metrics.py` — `DifficultyBucketTracker`
- `mw_ia/training/runner.py` — `ProceduralDQNRunner`
- `mw_ia/gui/widgets/difficulty_label.py` — label "Maze #N, diff=X.XX"

## V2-Y — Deep Recurrent Q-Network (LSTM) (sous-projet livré)

Mémoire neuronale temporelle pour franchir le plafond architectural V2-X
(DQN feedforward plafonne à ~80% winrate à diff=0.10). Le LSTM permet à
l'agent de se souvenir des dead-ends récents dans le maze courant.

### Usage CLI

```bash
# Mode obstacles, recette V2-X gagnante héritée par défaut
python scripts/train_drqn_procedural.py --episodes 5000 --mode obstacles --device cuda

# Mode maze parfait
python scripts/train_drqn_procedural.py --episodes 5000 --mode maze --device cuda
```

### Critère de succès

Bucket 1 du tracker (difficulté 0.20-0.40) doit afficher **winrate ≥ 70%** en
fin d'entraînement. Comparaison directe avec V2-X (DQN feedforward) qui plafonne
au bucket 0 (0.0-0.20).

### Architecture

- `mw_ia/neural/recurrent.py` — `RecurrentQNetwork` (Linear → ReLU → LSTM → Linear)
- `mw_ia/neural/sequence_buffer.py` — `SequenceReplayBuffer` (buffer de trajectoires complètes, sample seq_len avec padding+mask)
- `mw_ia/neural/recurrent_trainer.py` — `RecurrentDQNTrainer` (BPTT 32 steps, Huber masquée, AMP + grad clip)
- `mw_ia/agents/recurrent_dqn.py` — `RecurrentDQNAgent` (hidden state runtime maintenu, reset par épisode)
- `mw_ia/training/runner.py::RecurrentProceduralDQNRunner` — extension V2-Y

### Algorithme

DRQN simple (Hausknecht & Stone 2015) : hidden state zero-init au début de
chaque séquence d'entraînement (pas de burn-in en V2-Y MVP). Hidden state
runtime maintenu entre `act()` consécutifs dans un épisode, reset à chaque
nouvel épisode (pas de mémoire cross-épisodes — cohérent avec le but
"résoudre le maze courant", pas "se souvenir des mazes précédents").

## V2-Z — CNN perception spatiale (sous-projet livré)

**Tag** : `v0.2.0-z` — **Tests** : 208 verts (183 baseline + 25 V2-Z)

Motivation : V2-X (MLP) et V2-Y (LSTM) plafonnent tous deux à `diff ≈ 0.05`.
Le bottleneck est la représentation spatiale 1D — un encoding
`concat(position_one_hot, grid_flatten)` détruit la structure 2D du maze.
V2-Z remplace l'encoder par un tensor 3-canaux (agent + obstacles + goal)
et le réseau par un Conv2D.

### Usage CLI

```bash
python scripts/train_cnn_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda
```

Defaults gagnants V2-Z (recette consolidée à itérer empiriquement) :
- `--conv-channels 32 64` (default)
- `--fc-hidden 256` (default)
- `--epsilon-decay-steps 200000` (default V2-Z, hérité V2-X)
- `--scheduler-update-interval 200` (default V2-X)
- `--scheduler-step 0.05` (default V2-X)

### Architecture

Architecture : `Conv(3→32, k=3, pad=1) → ReLU → Conv(32→64, k=3, pad=1) → ReLU →
Flatten → Linear(64·R·C → 256) → ReLU → Linear(256 → 4)`. Pas de pooling
pour préserver l'info spatiale sur 10×10. ~1.66M params.

- `mw_ia/envs/procedural_env.py` — `encode_procedural_observation_2d` (3 canaux)
- `mw_ia/neural/conv_network.py` — `ConvQNetwork` (Conv2d + FC)
- `mw_ia/agents/conv_dqn.py` — `ConvDQNAgent` + `_ConvDQNTrainer` interne
- `mw_ia/training/runner.py::ConvProceduralDQNRunner` — extension V2-Z

### GUI

Bouton "Démarrer (procedural CNN)" disponible dans `python scripts/launch_gui.py`.

## V2-W — Double DQN sur ConvDQN (sous-projet livré)

**Tag** : `v0.2.0-w` — **Tests** : 211 verts (208 baseline + 3 V2-W)

Motivation : V2-Z (CNN seul) franchit le plafond V2-X/V2-Y (atteint diff=0.25-0.35
sur 2/3 seeds) mais avec **variance inter-seeds élevée** (écart-type ±0.13).
Symptôme classique de **surestimation Q-values DQN** (Hasselt 2015) + sensibilité
aux conditions initiales.

V2-W ajoute Double DQN : on découple la sélection d'action (online net) et son
évaluation (target net) pour stabiliser l'apprentissage.

### Usage CLI

```bash
# V2-W par défaut (Double DQN activé)
python scripts/train_cnn_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda

# Reproduire la baseline V2-Z (DQN classique)
python scripts/train_cnn_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda --no-double-dqn
```

### Diff algorithmique (~10 LOC dans `_ConvDQNTrainer.step()`)

Avant (V2-Z DQN classique) :
```python
q_next = self.target(next_states).max(dim=1).values
```

Après (V2-W Double DQN) :
```python
next_actions = self.online(next_states).argmax(dim=1)   # sélection : online
q_next = self.target(next_states).gather(1, next_actions.view(-1, 1)).squeeze(1)
```

### Architecture

Aucun nouveau fichier code. Flag `double_dqn: bool = True` dans `ConvDQNConfig`,
branche conditionnelle dans `_ConvDQNTrainer.step()`, exposition CLI via
`argparse.BooleanOptionalAction`. Réutilise intégralement l'infrastructure V2-Z
(ConvQNetwork, ConvDQNAgent, ConvProceduralDQNRunner, ReplayBuffer).

### GUI

Le bouton "Démarrer (procedural CNN)" utilise `ConvDQNConfig()` par défaut, donc
V2-W (Double DQN) est activé automatiquement. Pas de nouveau bouton.

## V2-V — Training Protocol Stabilization (sous-projet livré)

**Tag** : `v0.2.0-v` — **Tests** : 230 verts (211 baseline + 19 V2-V)

Motivation : H1 confirmée 2026-05-23 — sur V2-W seed 4, le winrate passe de 1 % (ep=5000) à 71 % (ep=3000). Le pipeline "train until end" est cassé : on jette littéralement le meilleur agent entraîné.

V2-V ajoute :
- **Évaluation périodique greedy** sur 10 seeds eval séparés du training (seeds 10000-10009)
- **Best-checkpoint tracking** : sauvegarde automatique du modèle au pic d'eval_winrate

### Usage CLI

```bash
# V2-V par défaut (eval activé, sans sauvegarde disque)
python scripts/train_cnn_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda

# Avec sauvegarde du best-checkpoint
python scripts/train_cnn_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda \
    --best-checkpoint-path checkpoints/v2v_best_seed0.pt

# Reproduire baseline pre-V2-V (sans eval)
python scripts/train_cnn_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda --no-eval
```

### Architecture

- `mw_ia/training/evaluator.py::PeriodicEvaluator` — env eval séparé, méthode `evaluate(agent, difficulty)` qui ne pollue ni le buffer ni le scheduler
- `mw_ia/training/checkpoint_tracker.py::BestCheckpointTracker` — sauvegarde au pic d'eval_winrate (idempotent, path=None = tracking en mémoire)
- `ConvDQNConfig` étendu : `eval_enabled`, `eval_every_episodes`, `eval_seeds`, `eval_max_steps`, `best_checkpoint_path`
- `ConvProceduralDQNRunner` intègre evaluator + tracker via `eval_enabled=True`

### Overhead

~10 % de temps d'entraînement en plus (10 seeds eval × 200 max_steps / 100 ép training = ~100 ms / ép). Compensation : récupération du best-model qui aurait été détruit par le late-stage collapse.

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
