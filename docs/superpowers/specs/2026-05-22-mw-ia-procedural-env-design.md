# MW_IA — Procedural Environment & Curriculum Learning — Design Spec

> **Sous-projet V2-X** (parallèle au programme V2 A-F). À implémenter via le pattern Subagent-Driven Development utilisé pour V1 et V2-A.

---

## 1. Objectif

Faire passer l'agent DQN MW_IA d'un **résolveur d'une seule map mémorisée** (V1) à un **résolveur de labyrinthes en général** via :

- un **environnement procédural** qui génère un nouveau maze à chaque épisode ;
- un **scheduler de difficulté adaptatif** piloté par le winrate ;
- des **métriques par bucket de difficulté** pour observer la généralisation et l'oubli ;
- une **intégration GUI** affichant le maze courant + une 5e courbe `difficulty(t)`.

### Critères de succès vérifiables

1. L'agent atteint **≥ 70 % de winrate** sur le bucket de difficulté max (densité 0.4 / taille ≥ 15) au bout d'un entraînement suffisamment long.
2. Le winrate du bucket 0 (faciles) reste **≥ 80 %** en fin d'entraînement (pas d'oubli catastrophique total — sinon ça déclenche le sous-projet D continual learning).
3. La courbe de difficulté monte de façon adaptative (visible dans la GUI).
4. **120+ tests pytest verts** à la livraison (95 actuels + ~28 nouveaux).

### Hors-scope explicites

- **Continual learning** (EWC, rehearsal explicite) → sous-projet D
- **LSTM / mémoire neuronale temporelle** → roadmap #1
- **Randomisation start/goal** → V2 de ce sous-projet
- **Vue locale (window) sur l'agent** → couplé avec LSTM (roadmap)
- **Multi-objectifs** (survival, énergie) → roadmap #4
- **Modification rewards** (step_penalty, goal_reward) → conservés V1
- **Benchmark de performance automatisé** → manuel via GUI
- **Casser la rétro-compat V1** : V1 fix-map continue de marcher inchangé

### Contrainte de design non-négociable

**Tout maze généré DOIT être solvable.** Pattern obligatoire validé par l'utilisateur (2026-05-22) et capturé en mémoire persistante :

```
1. generate maze
2. check path exists from start to goal (BFS)
3. if no path → regenerate (max 100 tentatives, sinon RuntimeError)
4. else → use this maze
```

Le mode **maze parfait** (DFS recursive backtracker) est solvable par construction et n'a pas besoin du check post-hoc. Le mode **obstacles** (placement aléatoire) DOIT le faire systématiquement.

---

## 2. Architecture

V1 reste intact. Le procedural est une **couche au-dessus** de `GridWorld`, pas une mutation invasive. La rétro-compat est garantie par un flag de config.

```
mw_ia/
├── envs/
│   ├── gridworld.py              # V1 inchangé (map fixe via cfg.obstacles)
│   ├── maze_generators.py        # [NOUVEAU] 2 générateurs + check BFS
│   └── procedural_env.py         # [NOUVEAU] wrapper sur GridWorld qui régénère à reset()
├── training/
│   ├── metrics.py                # [MODIFIÉ] + DifficultyBucketTracker (5 buckets)
│   ├── scheduler.py              # [NOUVEAU] AdaptiveDifficultyScheduler
│   └── runner.py                 # [MODIFIÉ] runner DQN procedural (callback maze→difficulty)
├── neural/
│   └── network.py                # [MODIFIÉ] QNetwork accepte input_dim variable
├── config.py                     # [MODIFIÉ] + ProceduralEnvConfig (frozen dataclass)
└── gui/
    ├── app.py                    # [MODIFIÉ] grille redessine à chaque reset
    └── widgets/
        ├── plots.py              # [MODIFIÉ] 5e courbe 'difficulty'
        └── difficulty_label.py   # [NOUVEAU] text "Maze #42, diff=0.45"
```

### 2.1 Nouveaux composants

#### `MazeGenerator` (Protocol)

Interface commune en `mw_ia/envs/maze_generators.py` :

```python
class MazeGenerator(Protocol):
    def generate(self, seed: int, difficulty: float) -> np.ndarray:
        """Retourne une grille (rows, cols) de booléens (True = obstacle).

        Args:
            seed: graine pour reproductibilité.
            difficulty: ∈ [0,1]. Sens dépend de l'implémentation concrète.

        Returns:
            np.ndarray[bool] de shape (rows, cols). Garantie : start et goal
            sont des cellules libres ; un chemin existe entre les deux.

        Raises:
            RuntimeError: si la solvabilité ne peut être obtenue après les
                tentatives maximales (cas RandomObstaclesGenerator avec densité
                pathologique).
        """
```

Deux implémentations :

**`RandomObstaclesGenerator`** : place `density * rows * cols` obstacles aléatoirement. Densité = difficulté (interpolation entre `min_density=0.1` et `max_density=0.5`). BFS-check post-hoc, regénère jusqu'à 100 tentatives. Au-delà → `RuntimeError("density=X.XX unreachable after 100 attempts")`. Grille fixe `rows × cols` (déterminé par la config, pas par la difficulté).

**`PerfectMazeGenerator`** : DFS recursive backtracker → solvable par construction. Difficulté = taille (interpolation entre `min_size=4` et `max_size=20`). Pas de check BFS nécessaire.

#### `ProceduralGridWorld`

Wrapper en `mw_ia/envs/procedural_env.py`. Accepte un `MazeGenerator` + une `difficulty` mutable. À chaque `reset()`, appelle `generator.generate(seed=episode_seed, difficulty=current_difficulty)`, reconstruit un `GridWorldConfig` neuf, ré-instancie le `GridWorld` interne. `step()` délègue intégralement à V1. `info` enrichi de `maze`, `difficulty`, `episode_id`.

#### `AdaptiveDifficultyScheduler`

Dans `mw_ia/training/scheduler.py`. État : `current_difficulty: float ∈ [0,1]`.

```python
def update(self, winrate: float) -> float:
    if winrate >= self.up_threshold:     # 0.8 par défaut
        self.current = min(1.0, self.current + self.step)  # step=0.05
    elif winrate <= self.down_threshold: # 0.3 par défaut
        self.current = max(0.0, self.current - self.step)
    return self.current
```

Appelée par le runner toutes les `update_interval` épisodes (50 par défaut). `up_threshold ≤ down_threshold` → `ValueError` à la construction.

#### `DifficultyBucketTracker`

Dans `mw_ia/training/metrics.py`. Wrapper sur 5 `MetricsTracker` (un par bucket : `[0,0.2)`, `[0.2,0.4)`, `[0.4,0.6)`, `[0.6,0.8)`, `[0.8,1.0]`).

```python
def record_episode(self, success: bool, reward: float, length: int, difficulty: float) -> None:
    bucket = min(4, int(difficulty * 5))   # difficulty=1.0 → bucket 4 (inclusive)
    self._trackers[bucket].record_episode(reward=reward, length=length, success=success)

def winrate_per_bucket(self) -> list[Optional[float]]:
    return [t.winrate() if t.has_data() else None for t in self._trackers]
```

**Note** : `record_episode` reçoit les vrais `reward` et `length` de l'épisode pour que chaque bucket ait des stats correctes (pas seulement le winrate). Le `min(4, ...)` traite le cas `difficulty=1.0` qui donnerait sinon `bucket=5` (out of bounds).

#### `ProceduralDQNRunner`

Variante de `DQNRunner` qui :
- Régénère le maze à chaque `reset()`.
- Encode l'observation : `[row/(rows-1), col/(cols-1), *grid.flatten().astype(float32)]`. Dim totale `2 + max_rows * max_cols`.
- Met à jour le scheduler toutes les `update_interval` épisodes.
- Route les épisodes vers le bon bucket.
- Émet 3 nouveaux callbacks GUI : `on_maze_changed(maze, episode_id, difficulty)`, `on_difficulty_updated(difficulty)`, et le `on_episode_done` V1 enrichi du `bucket_winrate`.

#### `maze_bfs_check(grid, start, goal) -> bool`

Helper standalone en `maze_generators.py`. BFS classique, retourne `True` si un chemin existe entre start et goal en évitant les cellules `grid[r, c] == True`. Complexité O(R*C). **Isolé exprès** pour devenir un futur invariant Aether I9 (cf. §6.4).

### 2.2 Composants modifiés

#### `QNetwork` (`mw_ia/neural/network.py`)

V1 prend `input_dim=2` hard-codé. On rend `input_dim` paramétrable au constructeur :

```python
def __init__(self, input_dim: int = 2, n_actions: int = 4, hidden: int = 128):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )
```

V1 continue de marcher (default `input_dim=2`). Mode procedural passe `input_dim=2 + max_rows*max_cols`.

#### Padding des mazes plus petits que `max_size`

Pour le mode `PerfectMazeGenerator` où la taille varie avec la difficulté, les mazes plus petits sont **placés top-left dans une grille `max_size × max_size` paddée de zéros** (= cellules libres). L'agent voit des bordures artificielles qu'il apprend à ignorer. Documenté en README V2-X et dans la docstring de `encode()`.

#### GUI

- Grille animée (`mw_ia/gui/widgets/grid.py`) : signal Qt `maze_changed(np.ndarray, int, float)` connecté pour redessiner les obstacles à chaque épisode.
- Plots (`mw_ia/gui/widgets/plots.py`) : ajout d'une 5e courbe `difficulty` à côté des 4 existantes (reward, loss, ε, winrate).
- Nouveau widget `difficulty_label.py` : text "Maze #42, diff=0.45" mis à jour à chaque épisode.
- Pattern `QueuedConnection` V1 réutilisé : aucune mutation concurrente entre thread d'entraînement et thread GUI.

---

## 3. Data flow

Flow d'un épisode dans le mode procedural :

```
[ProceduralDQNRunner.run_episode(episode_id)]
   │
   ├─► env.reset(seed=episode_id)
   │      │
   │      ├─► generator.generate(seed=episode_id, difficulty=scheduler.current)
   │      │      ├─► boucle (max 100 tentatives en mode obstacles) :
   │      │      │      ├─► place density*rows*cols obstacles aléatoires
   │      │      │      └─► maze_bfs_check(grid, start, goal)
   │      │      │             └─► True → return grid
   │      │      │             └─► False → retry
   │      │      └─► tentatives épuisées → RuntimeError
   │      ├─► new_cfg = GridWorldConfig(rows, cols, obstacles=where(grid))
   │      ├─► _inner = GridWorld(new_cfg)
   │      └─► return (state, info{"maze": grid, "difficulty": current, "episode_id": episode_id})
   │
   ├─► observation = encode(state, grid)  # [row, col, *grid.flatten()] → np.float32 dim 2+R*C
   │
   ├─► boucle step :
   │      ├─► action = agent.act(observation)
   │      ├─► next_state, reward, terminated, truncated, info = env.step(action)
   │      ├─► next_obs = encode(next_state, grid)  # grid inchangé dans l'épisode
   │      ├─► replay_buffer.push(observation, action, reward, next_obs, done)
   │      ├─► agent.train_step()    # batch tiré du buffer
   │      └─► observation = next_obs
   │
   ├─► bucket = min(4, int(scheduler.current * 5))
   ├─► bucket_tracker.record_episode(success=terminated, difficulty=scheduler.current)
   ├─► metrics_global.record_episode(success=terminated, reward=ep_reward, length=ep_len)
   │
   ├─► si episode_id % update_interval == 0 :
   │      └─► scheduler.update(winrate=metrics_global.winrate())
   │
   └─► callbacks GUI :
          ├─► on_maze_changed(grid, episode_id, scheduler.current)
          ├─► on_episode_done(reward, length, winrate, epsilon, loss)
          └─► on_difficulty_updated(scheduler.current)
```

### Points clés

- **Seed déterministe** : `episode_id` → mêmes mazes en replay. Permet de reproduire un bug ou tracer une session.
- **Grid figé dans l'épisode** : `generate()` une seule fois par épisode (au `reset()`). Aucune mutation pendant `step()`. Garantit la stationnarité.
- **Observation** : `[row, col, *grid.flatten()]` — `row,col` normalisés en `[0,1]`, grid en `{0,1}` booléen → float32. Dim totale `2 + R*C`.
- **Replay buffer mélange transitions inter-épisodes** : **pas un bug, c'est le but**. Le réseau apprend une politique générale, pas une politique map-spécifique.
- **Scheduler update** : toutes les `update_interval` épisodes (50 par défaut), pas à chaque épisode (bruit).

---

## 4. Error handling

| Cas | Source | Stratégie MVP | Visibilité |
|---|---|---|---|
| Densité ≥ 95 % → BFS échoue 100 fois | `RandomObstaclesGenerator.generate()` | `raise RuntimeError("density={density:.2f} unreachable after 100 attempts")` | Crash explicite |
| Start ou goal placés sur obstacle | Générateur | Garanti par construction (set difference avant le tirage) | Test unitaire |
| Difficulté hors `[0,1]` | Scheduler manipulé externalement | `__post_init__` valide ; `update()` clampe via `np.clip` | Validation à construction |
| `up_threshold ≤ down_threshold` | Mauvaise config scheduler | `__post_init__` raise `ValueError` | Crash à construction |
| Maze 1×1 (taille pathologique) | `PerfectMazeGenerator` avec difficulté = 0 | Floor minimum à `min_size=4` (codé dur, pas configurable) | Silent clamp documenté |
| DFS récurse trop profond | `PerfectMazeGenerator` sur grosse grille | MVP : max_size=20 dur dans le générateur. Au-delà → `ValueError` | Crash explicite |
| Replay buffer mélange transitions inter-épisodes | Conception | Pas une erreur — c'est le but. Documenté dans la docstring du runner | Doc only |
| GUI grid widget reçoit `maze_changed` pendant un step | Signaux Qt | `QueuedConnection` (pattern V1) | Pattern V1 réutilisé |
| QNetwork attend `input_dim=102` mais reçoit `input_dim=402` | Changement de taille pendant l'entraînement | `input_dim` figé à la construction ; padding zéro pour mazes plus petits que `max_size` | Doc + test |
| `encode()` reçoit grid de mauvaise taille | Bug runner | Assert `grid.shape == (max_rows, max_cols)` | AssertionError |
| Scheduler reste bloqué à 0.0 | Agent non-fonctionnel | Pas d'erreur, légitime ; visible dans courbe difficulté GUI | Observabilité only |

### Décisions de design

1. **Crashes explicites > silent fallbacks** : pattern projet (V1 utilise `assert` et `ValueError` libéralement).
2. **Padding à zéro pour mazes de taille variable** : `QNetwork` dimensionné pour `max_size²+2`. Conséquence acceptable pédagogiquement, à documenter.
3. **Validation à la construction**, pas à l'usage : `ProceduralEnvConfig` frozen, `__post_init__` lève `ValueError` pour combinaisons incohérentes (comme `VariantSpec` V2-A).
4. **Pas de mécanisme de récupération dans le runner** : si générateur lève `RuntimeError`, le runner remonte. L'utilisateur ajuste la config et relance.
5. **Pas d'auto-baisse de la densité** en V1 du sous-projet. Feature V2 si utile.

---

## 5. Configuration

### `ProceduralEnvConfig` (nouveau dataclass frozen)

```python
@dataclass(frozen=True)
class ProceduralEnvConfig:
    mode: Literal["obstacles", "maze"]    # générateur à utiliser
    max_rows: int = 10                    # taille max de la grille
    max_cols: int = 10
    min_density: float = 0.10             # mode obstacles uniquement
    max_density: float = 0.50
    min_size: int = 4                     # mode maze uniquement
    max_size: int = 20

    def __post_init__(self) -> None:
        if self.mode not in ("obstacles", "maze"):
            raise ValueError(...)
        if not (0.0 <= self.min_density < self.max_density <= 1.0):
            raise ValueError(...)
        if not (2 <= self.min_size < self.max_size):
            raise ValueError(...)
        # ... etc.
```

### `SchedulerConfig` (nouveau dataclass frozen)

```python
@dataclass(frozen=True)
class SchedulerConfig:
    initial_difficulty: float = 0.0
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    up_threshold: float = 0.80
    down_threshold: float = 0.30
    step: float = 0.05
    update_interval: int = 50           # épisodes
```

---

## 6. Testing strategy

### 6.1 Tests unitaires (~25 tests)

**`tests/envs/test_maze_generators.py`** — ~12 tests
- `maze_bfs_check` : 5 cas (trivial, avec obstacles, bloqué, start==goal, hors grille)
- `RandomObstaclesGenerator` : densité 0/0.5, seed déterministe, density 0.95 → RuntimeError, start/goal jamais sur obstacle (1000 tirages hypothesis), solvabilité (1000 tirages hypothesis)
- `PerfectMazeGenerator` : taille 4×4 et 20×20 solvable, seed déterministe, taille > 20 → ValueError

**`tests/envs/test_procedural_env.py`** — ~5 tests
- `reset()` régénère un maze différent
- `reset(seed=42)` déterministe
- `step()` délègue à V1 (mêmes rewards/terminations)
- `info["maze"|"difficulty"|"episode_id"]` présents
- Changement de difficulté → grille différente

**`tests/training/test_scheduler.py`** — ~6 tests
- update winrate haut → difficulté monte
- update winrate bas → difficulté descend
- update zone neutre → inchangé
- Clamp `[min,max]`
- `up_threshold ≤ down_threshold` → ValueError
- Stabilité (100 updates winrate=0.5 → inchangé)

**`tests/training/test_bucket_tracker.py`** — ~4 tests
- Routing difficulté → bucket
- `winrate_per_bucket()` retourne 5 valeurs
- Bucket vide → None (pas 0.0)
- Tous les buckets remplis → tous les winrates valides

### 6.2 Tests intégration (~3 tests)

**`tests/training/test_procedural_runner.py`**
- 20 épisodes mode obstacles, density=0.2, scheduler désactivé → pas d'erreur, mazes différents, observations cohérentes
- 20 épisodes mode maze, scheduler actif (winrate forcé 0.9) → difficulté monte
- Smoke DQN apprend (loss > 0 décroît sur 100 ép, winrate > epsilon)

### 6.3 Test E2E smoke (1)

**`scripts/train_dqn_procedural.py`** : `python scripts/train_dqn_procedural.py --episodes 50 --mode obstacles --device cpu`. Vérifie : exit 0, checkpoint produit, log contient "winrate" et "difficulty=X.XX". Lancé en CI dans `aether_verify.yml` (job pytest existant).

### 6.4 Tests Aether (préparation V2-A.2, **hors livraison MVP**)

Le helper `maze_bfs_check` est isolé pour devenir `i9_maze_solvability.aether` dans une future itération de V2-A. **Pas livré dans ce sous-projet** mais le découplage est prévu.

### 6.5 Couverture totale

- ~28 nouveaux tests unitaires + 3 intégration + 1 smoke E2E
- **120+ tests pytest verts** à la livraison
- Tous les générateurs : test **property-based** de solvabilité (hypothesis) — garantie machine-vérifiée de la contrainte non-négociable.

### 6.6 Hors-scope testing

- Pas de benchmark de performance automatisé
- Pas de test sur GUI visuelle (pattern projet)
- Pas de test du callback Qt cross-thread (déjà testé indirectement par V1)

---

## 7. Stratégie d'implémentation

Pattern **Subagent-Driven Development** utilisé pour V1 et V2-A. Phases prévues (à détailler dans le plan d'implémentation suivant) :

1. **Phase 1 — `maze_generators.py`** : `maze_bfs_check` + `RandomObstaclesGenerator` + `PerfectMazeGenerator`, tous property-based-tested.
2. **Phase 2 — `ProceduralGridWorld`** : wrapper sur `GridWorld`.
3. **Phase 3 — `AdaptiveDifficultyScheduler` + `DifficultyBucketTracker`**.
4. **Phase 4 — `QNetwork` paramétrable + `encode()` helper avec padding**.
5. **Phase 5 — `ProceduralDQNRunner`** : runner DQN avec callbacks GUI étendus.
6. **Phase 6 — GUI** : signal `maze_changed`, 5e courbe, widget difficulty_label.
7. **Phase 7 — `scripts/train_dqn_procedural.py`** : CLI smoke E2E + intégration CI.
8. **Phase 8 — README V2-X + tag `v0.2.0-x`** (ou nom approprié à décider).

Chaque phase suit le triple gate : implementer subagent → spec compliance review → code quality review.

---

## 8. Références

- Pattern de génération solvable validé : mémoire `projet_mw_ia_curriculum.md` (2026-05-22).
- État projet : `CLAUDE.md` (V1 livré `v0.1.0`, V2-A livré `v0.2.0-a`, 95 tests verts).
- Inspiration curriculum learning : Bengio et al. (2009), "Curriculum Learning" ICML.
- Inspiration PCG-RL : OpenAI Procgen Benchmark (2019), DeepMind XLand (2021).
