# V2-BX — Diagnostic causal du bottleneck 15×15 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Outiller trois sondes-oracles jetables (représentation / horizon / exploration) derrière des flags `default=off`, plus un logging structuré par sonde, pour identifier par la mesure (`diff_max`) quelle famille est le bottleneck du régime 15×15 de V2-ZY+Polyak.

**Architecture:** Sondes activées par flags sur `ConvRecurrentDQNConfig`, branchées dans le seul `ConvRecurrentProceduralDQNRunner` (substrat unique V2-ZY). Sonde C = 4ᵉ canal d'observation (`in_channels 3→4`, plan uniforme C1 / champ BFS C2) via une fonction `bfs_distance_field` réutilisant le pattern BFS existant. Sonde A = exposer le champ `gamma` existant en CLI. Sonde B = bonus de nouveauté count-based ajouté au reward dans la boucle runner. Tout est conditionnel et no-op quand les flags valent leur défaut, garantissant la reproductibilité bit-à-bit des baselines V2-U/B0/B1a.

**Tech Stack:** Python 3.13, PyTorch 2.11+cu128, numpy, pytest. Conventions projet : TDD test-first, dataclasses frozen-style avec `__post_init__`, flags opt-in (pattern `per_enabled`/`b1a_enabled`), messages/help-text ASCII (piège #8 Windows cp1252).

**Spec de référence :** `docs/superpowers/specs/2026-05-30-mw-ia-bx-diagnostic-design.md`

**Pré-requis environnement :** `source .venv/Scripts/activate` avant toute commande. Baseline tests : `pytest -q` doit afficher **356 passed** au départ.

---

## Task 1 : `bfs_distance_field` — champ de distance géodésique au goal

**Files:**
- Modify: `mw_ia/envs/maze_generators.py` (ajout fonction après `maze_bfs_check`, ~ligne 56)
- Test: `tests/envs/test_bfs_distance_field.py` (créer)

- [ ] **Step 1: Écrire le test qui échoue**

Créer `tests/envs/test_bfs_distance_field.py` :

```python
"""Tests du champ de distance BFS au goal (V2-BX Sonde C)."""
from __future__ import annotations

import numpy as np

from mw_ia.envs.maze_generators import bfs_distance_field


def test_goal_cell_is_zero():
    grid = np.zeros((3, 3), dtype=bool)
    dist = bfs_distance_field(grid, goal=(2, 2))
    assert dist[2, 2] == 0.0


def test_open_grid_manhattan_distances():
    # Grille vide 3x3, goal en (0,0) -> distance = Manhattan (pas d'obstacle).
    grid = np.zeros((3, 3), dtype=bool)
    dist = bfs_distance_field(grid, goal=(0, 0))
    assert dist[0, 1] == 1.0
    assert dist[1, 0] == 1.0
    assert dist[1, 1] == 2.0
    assert dist[2, 2] == 4.0


def test_obstacle_cells_are_inf():
    grid = np.zeros((3, 3), dtype=bool)
    grid[1, 1] = True  # obstacle au centre
    dist = bfs_distance_field(grid, goal=(0, 0))
    assert np.isinf(dist[1, 1])


def test_wall_detour_increases_distance():
    # Mur vertical en colonne 1 sur lignes 0 et 1 force un détour par le bas.
    grid = np.zeros((3, 3), dtype=bool)
    grid[0, 1] = True
    grid[1, 1] = True
    dist = bfs_distance_field(grid, goal=(0, 0))
    # (0,2) atteignable seulement via (2,1) : 0,0->1,0->2,0->2,1->2,2->1,2->0,2 = 6
    assert dist[0, 2] == 6.0


def test_unreachable_region_is_inf():
    # Cellule (0,2) emmurée : obstacles en (0,1) et (1,2).
    grid = np.zeros((3, 3), dtype=bool)
    grid[0, 1] = True
    grid[1, 2] = True
    dist = bfs_distance_field(grid, goal=(0, 0))
    assert np.isinf(dist[0, 2])


def test_returns_float_array_of_grid_shape():
    grid = np.zeros((4, 5), dtype=bool)
    dist = bfs_distance_field(grid, goal=(3, 4))
    assert dist.shape == (4, 5)
    assert dist.dtype == np.float64
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `pytest tests/envs/test_bfs_distance_field.py -q`
Expected: FAIL avec `ImportError: cannot import name 'bfs_distance_field'`.

- [ ] **Step 3: Implémenter la fonction**

Dans `mw_ia/envs/maze_generators.py`, après `maze_bfs_check` (après la ligne 55, avant `@dataclass class RandomObstaclesGenerator`), insérer :

```python
def bfs_distance_field(
    grid: np.ndarray, *, goal: tuple[int, int]
) -> np.ndarray:
    """Distance géodésique BFS 4-connexe de chaque cellule au goal.

    Args:
        grid: tableau (rows, cols) de booléens. True = obstacle.
        goal: (row, col) du goal (doit être libre).

    Returns:
        np.ndarray[float64] (rows, cols) : nombre de pas du plus court chemin
        4-connexe de chaque cellule au goal en évitant les obstacles.
        goal = 0.0 ; cellules-obstacle et cellules non-atteignables = np.inf.

    Raises:
        AssertionError: si goal hors grille ou sur un obstacle.
    """
    rows, cols = grid.shape
    gr, gc = goal
    assert 0 <= gr < rows and 0 <= gc < cols, f"goal {goal} hors grille {grid.shape}"
    assert not grid[gr, gc], f"goal {goal} sur obstacle"

    dist = np.full((rows, cols), np.inf, dtype=np.float64)
    dist[gr, gc] = 0.0
    queue: deque[tuple[int, int]] = deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] or np.isfinite(dist[nr, nc]):
                continue
            dist[nr, nc] = dist[r, c] + 1.0
            queue.append((nr, nc))
    return dist
```

(`deque` et `np` sont déjà importés en tête de fichier.)

- [ ] **Step 4: Lancer le test pour vérifier qu'il passe**

Run: `pytest tests/envs/test_bfs_distance_field.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add mw_ia/envs/maze_generators.py tests/envs/test_bfs_distance_field.py
git commit -m "feat(v2-bx): bfs_distance_field geodesic distance map for Sonde C"
```

---

## Task 2 : Flags config `bx_repr_oracle` + `bx_novelty_beta`

**Files:**
- Modify: `mw_ia/config.py` (`ConvRecurrentDQNConfig` : champs ~après ligne 428 ; validation dans `__post_init__` ~après ligne 489)
- Test: `tests/test_bx_config.py` (créer)

- [ ] **Step 1: Écrire le test qui échoue**

Créer `tests/test_bx_config.py` :

```python
"""Tests des flags V2-BX sur ConvRecurrentDQNConfig."""
from __future__ import annotations

import pytest

from mw_ia.config import ConvRecurrentDQNConfig


def test_defaults_are_no_op():
    cfg = ConvRecurrentDQNConfig()
    assert cfg.bx_repr_oracle == "none"
    assert cfg.bx_novelty_beta == 0.0


def test_valid_oracle_modes_accepted():
    for mode in ("none", "scalar", "field"):
        cfg = ConvRecurrentDQNConfig(bx_repr_oracle=mode)
        assert cfg.bx_repr_oracle == mode


def test_invalid_oracle_mode_rejected():
    with pytest.raises(ValueError, match="bx_repr_oracle"):
        ConvRecurrentDQNConfig(bx_repr_oracle="bogus")


def test_negative_novelty_beta_rejected():
    with pytest.raises(ValueError, match="bx_novelty_beta"):
        ConvRecurrentDQNConfig(bx_novelty_beta=-0.1)


def test_positive_novelty_beta_accepted():
    cfg = ConvRecurrentDQNConfig(bx_novelty_beta=0.1)
    assert cfg.bx_novelty_beta == 0.1
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `pytest tests/test_bx_config.py -q`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'bx_repr_oracle'`).

- [ ] **Step 3: Ajouter les champs + validation**

Dans `mw_ia/config.py`, dans `ConvRecurrentDQNConfig`, juste après le bloc B1a (après la ligne `b1a_mix_ratio: float = 0.2`, ~ligne 428), ajouter :

```python
    # V2-BX : sondes-oracles diagnostiques (jetables, default = no-op)
    bx_repr_oracle: str = "none"   # "none" | "scalar" (C1) | "field" (C2)
    bx_novelty_beta: float = 0.0   # bonus exploration count-based (Sonde B)
```

Puis dans `__post_init__` de `ConvRecurrentDQNConfig`, à la fin (juste avant la fin de la méthode), ajouter :

```python
        if self.bx_repr_oracle not in ("none", "scalar", "field"):
            raise ValueError(
                f"bx_repr_oracle doit etre none|scalar|field, recu {self.bx_repr_oracle}"
            )
        if self.bx_novelty_beta < 0.0:
            raise ValueError(
                f"bx_novelty_beta doit etre >= 0, recu {self.bx_novelty_beta}"
            )
```

- [ ] **Step 4: Lancer le test pour vérifier qu'il passe**

Run: `pytest tests/test_bx_config.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Vérifier non-régression config**

Run: `pytest tests/ -q -k config`
Expected: tous les tests config passent (aucune régression sur les configs existantes).

- [ ] **Step 6: Commit**

```bash
git add mw_ia/config.py tests/test_bx_config.py
git commit -m "feat(v2-bx): bx_repr_oracle + bx_novelty_beta flags on ConvRecurrentDQNConfig"
```

---

## Task 3 : Encodeur d'observation à 4ᵉ canal oracle

**Files:**
- Modify: `mw_ia/envs/procedural_env.py` (`encode_procedural_observation_2d` ~ligne 122 : ajout param `oracle_mode` + canal)
- Test: `tests/envs/test_procedural_oracle_obs.py` (créer)

- [ ] **Step 1: Écrire le test qui échoue**

Créer `tests/envs/test_procedural_oracle_obs.py` :

```python
"""Tests de l'encodeur d'observation 4-canaux oracle (V2-BX Sonde C)."""
from __future__ import annotations

import numpy as np

from mw_ia.envs.procedural_env import encode_procedural_observation_2d


def _empty_grid(n: int = 4) -> np.ndarray:
    return np.zeros((n, n), dtype=bool)


def test_none_mode_is_three_channels():
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=_empty_grid(), goal=(3, 3),
        max_rows=4, max_cols=4, oracle_mode="none",
    )
    assert obs.shape == (3, 4, 4)


def test_scalar_mode_adds_uniform_fourth_channel():
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=_empty_grid(), goal=(3, 3),
        max_rows=4, max_cols=4, oracle_mode="scalar",
    )
    assert obs.shape == (4, 4, 4)
    chan = obs[3]
    # Plan uniforme : toutes les cellules egales.
    assert np.allclose(chan, chan.flat[0])
    # Valeur = BFS(agent->goal)/ (rows*cols) = 6 / 16 sur grille vide 4x4.
    assert np.isclose(chan.flat[0], 6.0 / 16.0)


def test_field_mode_per_cell_distances():
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=_empty_grid(), goal=(0, 0),
        max_rows=4, max_cols=4, oracle_mode="field",
    )
    assert obs.shape == (4, 4, 4)
    chan = obs[3]
    # goal en (0,0) -> distance 0 ; (3,3) -> 6/16.
    assert np.isclose(chan[0, 0], 0.0)
    assert np.isclose(chan[3, 3], 6.0 / 16.0)


def test_field_mode_obstacle_sentinel_is_one():
    grid = _empty_grid()
    grid[1, 1] = True
    obs = encode_procedural_observation_2d(
        state=(0, 0), grid=grid, goal=(3, 3),
        max_rows=4, max_cols=4, oracle_mode="field",
    )
    # Cellule-obstacle = sentinelle 1.0 (plus loin que tout).
    assert np.isclose(obs[3, 1, 1], 1.0)


def test_first_three_channels_unchanged_by_oracle():
    base = encode_procedural_observation_2d(
        state=(1, 2), grid=_empty_grid(), goal=(3, 3),
        max_rows=4, max_cols=4, oracle_mode="none",
    )
    withc = encode_procedural_observation_2d(
        state=(1, 2), grid=_empty_grid(), goal=(3, 3),
        max_rows=4, max_cols=4, oracle_mode="scalar",
    )
    assert np.array_equal(base, withc[:3])


def test_invalid_oracle_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="oracle_mode"):
        encode_procedural_observation_2d(
            state=(0, 0), grid=_empty_grid(), goal=(3, 3),
            max_rows=4, max_cols=4, oracle_mode="bogus",
        )
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `pytest tests/envs/test_procedural_oracle_obs.py -q`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'oracle_mode'`).

- [ ] **Step 3: Étendre l'encodeur**

Dans `mw_ia/envs/procedural_env.py` :

Ajouter en tête (après `from mw_ia.envs.gridworld import Action, GridWorld`, ~ligne 9) :

```python
from mw_ia.envs.maze_generators import bfs_distance_field
```

Modifier la signature de `encode_procedural_observation_2d` (ligne 122) pour ajouter `oracle_mode: str = "none"` après `max_cols: int,` :

```python
def encode_procedural_observation_2d(
    *,
    state: tuple[int, int],
    grid: np.ndarray,
    goal: tuple[int, int],
    max_rows: int,
    max_cols: int,
    oracle_mode: str = "none",
) -> np.ndarray:
```

Puis, juste avant le `return obs` final (actuellement ligne 167), remplacer `return obs` par :

```python
    if oracle_mode == "none":
        return obs
    if oracle_mode not in ("scalar", "field"):
        raise ValueError(
            f"oracle_mode doit etre none|scalar|field, recu {oracle_mode}"
        )

    rows, cols = grid.shape
    dist_norm = float(max_rows * max_cols)
    dist = bfs_distance_field(grid, goal=goal)  # (rows, cols), inf hors-atteignable
    # Normalisation + sentinelle 1.0 pour obstacle / non-atteignable.
    norm_field = np.where(np.isfinite(dist), dist / dist_norm, 1.0)
    norm_field = np.clip(norm_field, 0.0, 1.0).astype(np.float32)

    oracle_chan = np.ones((max_rows, max_cols), dtype=np.float32)  # padding = 1.0
    if oracle_mode == "field":
        oracle_chan[:rows, :cols] = norm_field
    else:  # scalar : plan uniforme = distance de l'agent au goal
        oracle_chan[:] = norm_field[state[0], state[1]]

    return np.concatenate([obs, oracle_chan[np.newaxis, :, :]], axis=0)
```

- [ ] **Step 4: Lancer le test pour vérifier qu'il passe**

Run: `pytest tests/envs/test_procedural_oracle_obs.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Vérifier non-régression env**

Run: `pytest tests/envs/ -q`
Expected: tous verts (l'encodeur `oracle_mode="none"` par défaut ne change rien aux appels existants).

- [ ] **Step 6: Commit**

```bash
git add mw_ia/envs/procedural_env.py tests/envs/test_procedural_oracle_obs.py
git commit -m "feat(v2-bx): oracle 4th-channel in encode_procedural_observation_2d (C1 scalar / C2 field)"
```

---

## Task 4 : Wiring Sonde C dans le runner (in_channels + oracle_mode)

**Files:**
- Modify: `mw_ia/training/runner.py` (`ConvRecurrentProceduralDQNRunner` : `__init__` ~ligne 628-643, `run()` encode calls ~ligne 686-695)
- Test: `tests/training/test_bx_runner.py` (créer)

- [ ] **Step 1: Écrire le test qui échoue**

Créer `tests/training/test_bx_runner.py` :

```python
"""Tests d'intégration V2-BX runner : wiring oracle + nouveauté + logging."""
from __future__ import annotations

from mw_ia.config import (
    ConvRecurrentDQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ConvRecurrentProceduralDQNRunner, RunnerCallbacks


def _make_runner(**cfg_overrides):
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=5, max_cols=5, max_steps=20)
    gen = RandomObstaclesGenerator(
        rows=5, cols=5, start=(0, 0), goal=(4, 4),
        min_density=proc_cfg.min_density, max_density=proc_cfg.max_density,
        max_attempts=100,
    )
    env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
    dqn_cfg = ConvRecurrentDQNConfig(
        episodes=3, conv_channels=(8,), lstm_hidden=16, sequence_length=4,
        replay_capacity=10, min_episodes_to_learn=1, batch_size=1,
        max_steps_per_episode=20, eval_max_steps=20, eval_enabled=False,
        **cfg_overrides,
    )
    sched_cfg = SchedulerConfig(update_interval=1, step=0.05)
    cb = RunnerCallbacks()
    return ConvRecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=TrainingConfig(), callbacks=cb, device="cpu", seed=0,
    )


def test_none_oracle_uses_three_channels():
    runner = _make_runner(bx_repr_oracle="none")
    assert runner.agent.in_channels == 3


def test_scalar_oracle_uses_four_channels():
    runner = _make_runner(bx_repr_oracle="scalar")
    assert runner.agent.in_channels == 4


def test_field_oracle_runs_end_to_end():
    runner = _make_runner(bx_repr_oracle="field")
    assert runner.agent.in_channels == 4
    runner.run()  # ne doit pas lever (obs 4-canaux cohérentes act/observe)


def test_novelty_bonus_runs_end_to_end():
    runner = _make_runner(bx_novelty_beta=0.1)
    runner.run()  # reward shaping ne casse pas la boucle
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `pytest tests/training/test_bx_runner.py -q`
Expected: FAIL — `test_scalar_oracle_uses_four_channels` échoue (`in_channels == 3` car hardcodé).

- [ ] **Step 3: Wirer `in_channels` + `oracle_mode` dans le runner**

Dans `mw_ia/training/runner.py` :

Ajouter en tête de fichier (zone imports, après les imports existants) :

```python
import functools
import math
```

Dans `ConvRecurrentProceduralDQNRunner.__init__`, remplacer le bloc de construction de l'agent (lignes 628-631) :

```python
        self.agent = ConvRecurrentDQNAgent(
            in_channels=3, rows=proc_cfg.max_rows, cols=proc_cfg.max_cols,
            n_actions=4, cfg=dqn_cfg, device=device, seed=seed,
        )
```

par :

```python
        in_channels = 4 if dqn_cfg.bx_repr_oracle != "none" else 3
        self.agent = ConvRecurrentDQNAgent(
            in_channels=in_channels, rows=proc_cfg.max_rows, cols=proc_cfg.max_cols,
            n_actions=4, cfg=dqn_cfg, device=device, seed=seed,
        )
```

Dans le même `__init__`, remplacer la construction de l'évaluateur (ligne 641) :

```python
                observation_encoder=encode_procedural_observation_2d,
```

par (l'évaluateur appelle l'encodeur avec state/grid/goal/max_rows/max_cols ; on lie `oracle_mode` via partial pour qu'il produise le même nombre de canaux que le training) :

```python
                observation_encoder=functools.partial(
                    encode_procedural_observation_2d,
                    oracle_mode=dqn_cfg.bx_repr_oracle,
                ),
```

Dans `run()`, remplacer les DEUX appels à `encode_procedural_observation_2d` (lignes 686-689 et 692-695) pour ajouter `oracle_mode=self.dqn_cfg.bx_repr_oracle` :

```python
                obs = encode_procedural_observation_2d(
                    state=state, grid=maze, goal=goal,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                    oracle_mode=self.dqn_cfg.bx_repr_oracle,
                )
```

et

```python
                next_obs = encode_procedural_observation_2d(
                    state=s2, grid=maze, goal=goal,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                    oracle_mode=self.dqn_cfg.bx_repr_oracle,
                )
```

- [ ] **Step 4: Lancer le test pour vérifier qu'il passe**

Run: `pytest tests/training/test_bx_runner.py -q`
Expected: PASS pour les 3 tests d'oracle (`test_novelty_bonus_runs_end_to_end` peut encore échouer → traité en Task 5).

Note : si `test_novelty_bonus_runs_end_to_end` échoue ici, c'est attendu (la nouveauté arrive en Task 5). Vérifier au minimum que les 3 tests oracle passent :
Run: `pytest tests/training/test_bx_runner.py -q -k oracle`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add mw_ia/training/runner.py tests/training/test_bx_runner.py
git commit -m "feat(v2-bx): wire in_channels=4 + oracle_mode threading in ConvRecurrent runner (Sonde C)"
```

---

## Task 5 : Sonde B — bonus de nouveauté count-based dans le runner

**Files:**
- Modify: `mw_ia/training/runner.py` (`ConvRecurrentProceduralDQNRunner` : `__init__` init état nouveauté ; `run()` reset par épisode + bonus après `env.step`)

- [ ] **Step 1: (test déjà écrit en Task 4)** `test_novelty_bonus_runs_end_to_end`

Run: `pytest tests/training/test_bx_runner.py::test_novelty_bonus_runs_end_to_end -q`
Expected: PASS si déjà couvert par Task 4 sans erreur ; sinon ce test guide l'implémentation ci-dessous. (Le bonus ne doit pas casser la boucle.)

- [ ] **Step 2: Ajouter un test ciblé sur le calcul du bonus**

Ajouter à `tests/training/test_bx_runner.py` :

```python
def test_novelty_bonus_value_formula():
    runner = _make_runner(bx_novelty_beta=0.2)
    runner._reset_novelty()
    b1 = runner._novelty_bonus((1, 1))  # 1re visite -> 0.2 / sqrt(1)
    b2 = runner._novelty_bonus((1, 1))  # 2e visite  -> 0.2 / sqrt(2)
    assert abs(b1 - 0.2) < 1e-9
    assert abs(b2 - 0.2 / (2 ** 0.5)) < 1e-9


def test_novelty_bonus_zero_when_beta_zero():
    runner = _make_runner(bx_novelty_beta=0.0)
    runner._reset_novelty()
    assert runner._novelty_bonus((0, 0)) == 0.0
```

Run: `pytest tests/training/test_bx_runner.py -q -k novelty`
Expected: FAIL (`AttributeError: ... '_reset_novelty'`).

- [ ] **Step 3: Implémenter `_reset_novelty` / `_novelty_bonus` + brancher dans `run()`**

Dans `ConvRecurrentProceduralDQNRunner.__init__`, à la fin (après la construction de l'évaluateur/best_tracker), ajouter :

```python
        self._visit_counts: dict[tuple[int, int], int] = {}
```

Ajouter ces deux méthodes à la classe (par ex. juste avant `run()`) :

```python
    def _reset_novelty(self) -> None:
        """Reset la table de comptes de visites (debut d'episode / maze)."""
        self._visit_counts = {}

    def _novelty_bonus(self, cell: tuple[int, int]) -> float:
        """Bonus count-based pur beta / sqrt(visits) (Sonde B). 0 si beta=0."""
        beta = self.dqn_cfg.bx_novelty_beta
        if beta <= 0.0:
            return 0.0
        self._visit_counts[cell] = self._visit_counts.get(cell, 0) + 1
        return beta / math.sqrt(self._visit_counts[cell])
```

Dans `run()`, juste après `self.agent.begin_episode()` (ligne 676), ajouter :

```python
            self._reset_novelty()
```

Dans la boucle `while`, après `s2, r, terminated, truncated, _ = self.env.step(a)` (ligne 691) et AVANT `next_obs = ...`, ajouter :

```python
                r = r + self._novelty_bonus(s2)
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `pytest tests/training/test_bx_runner.py -q`
Expected: PASS (tous, y compris novelty).

- [ ] **Step 5: Commit**

```bash
git add mw_ia/training/runner.py tests/training/test_bx_runner.py
git commit -m "feat(v2-bx): count-based novelty bonus in ConvRecurrent runner (Sonde B)"
```

---

## Task 6 : Logging structuré par sonde (`diff_max`, `ep_to_diff_0.30`, etc.)

**Files:**
- Modify: `mw_ia/training/runner.py` (`ConvRecurrentProceduralDQNRunner` : `__init__` init tracking ; `run()` update + ligne finale ; helper `_probe_descriptor`)

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `tests/training/test_bx_runner.py` :

```python
def test_probe_descriptor_representation():
    runner = _make_runner(bx_repr_oracle="scalar")
    assert runner._probe_descriptor() == ("representation", "scalar")


def test_probe_descriptor_exploration():
    runner = _make_runner(bx_novelty_beta=0.1)
    ptype, pstrength = runner._probe_descriptor()
    assert ptype == "exploration"
    assert "0.1" in pstrength


def test_probe_descriptor_horizon():
    runner = _make_runner(gamma=0.997)
    ptype, pstrength = runner._probe_descriptor()
    assert ptype == "horizon"
    assert "0.997" in pstrength


def test_probe_descriptor_baseline():
    runner = _make_runner()
    assert runner._probe_descriptor() == ("baseline", "none")


def test_run_emits_structured_probe_log():
    logs: list[str] = []
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=5, max_cols=5, max_steps=20)
    gen = RandomObstaclesGenerator(
        rows=5, cols=5, start=(0, 0), goal=(4, 4),
        min_density=proc_cfg.min_density, max_density=proc_cfg.max_density, max_attempts=100,
    )
    env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
    dqn_cfg = ConvRecurrentDQNConfig(
        episodes=2, conv_channels=(8,), lstm_hidden=16, sequence_length=4,
        replay_capacity=10, min_episodes_to_learn=1, batch_size=1,
        max_steps_per_episode=20, eval_max_steps=20, eval_enabled=False,
        bx_repr_oracle="scalar",
    )
    cb = RunnerCallbacks(on_log=lambda level, msg: logs.append(msg))
    runner = ConvRecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg,
        sched_cfg=SchedulerConfig(update_interval=1, step=0.05),
        train_cfg=TrainingConfig(), callbacks=cb, device="cpu", seed=0,
    )
    runner.run()
    assert any("BX_PROBE_RESULT" in m and "probe_type=representation" in m for m in logs)
```

Run: `pytest tests/training/test_bx_runner.py -q -k "probe_descriptor or structured"`
Expected: FAIL (`AttributeError: ... '_probe_descriptor'`).

- [ ] **Step 2: Implémenter le tracking + le descripteur + la ligne finale**

Dans `ConvRecurrentProceduralDQNRunner.__init__`, après `self._visit_counts = {}` (Task 5), ajouter :

```python
        self._diff_max: float = 0.0
        self._ep_to_diff_030: int | None = None
```

Ajouter le helper à la classe :

```python
    def _probe_descriptor(self) -> tuple[str, str]:
        """Identifie la sonde active pour le logging structure."""
        if self.dqn_cfg.bx_repr_oracle != "none":
            return "representation", self.dqn_cfg.bx_repr_oracle
        if self.dqn_cfg.bx_novelty_beta > 0.0:
            return "exploration", f"beta={self.dqn_cfg.bx_novelty_beta}"
        if abs(self.dqn_cfg.gamma - 0.99) > 1e-9:
            return "horizon", f"gamma={self.dqn_cfg.gamma}"
        return "baseline", "none"
```

Dans `run()`, juste après `self.env.set_difficulty(self.scheduler.current)` (ligne 668), ajouter le tracking du plafond :

```python
            self._diff_max = max(self._diff_max, self.scheduler.current)
            if self._ep_to_diff_030 is None and self.scheduler.current >= 0.30:
                self._ep_to_diff_030 = ep
```

À la toute fin de `run()`, APRÈS la fin de la boucle `for ep in range(...)`, ajouter le log structuré :

```python
        probe_type, probe_strength = self._probe_descriptor()
        best_eval = (
            self.best_tracker.best_winrate if self.best_tracker is not None else float("nan")
        )
        self.callbacks.fire_log(
            "info",
            f"BX_PROBE_RESULT probe_type={probe_type} probe_strength={probe_strength} "
            f"diff_max={self._diff_max:.2f} ep_to_diff_0.30={self._ep_to_diff_030} "
            f"best_eval_0.30={best_eval:.2%}"
        )
```

- [ ] **Step 3: Lancer les tests pour vérifier qu'ils passent**

Run: `pytest tests/training/test_bx_runner.py -q`
Expected: PASS (tous).

- [ ] **Step 4: Commit**

```bash
git add mw_ia/training/runner.py tests/training/test_bx_runner.py
git commit -m "feat(v2-bx): structured per-probe logging (diff_max, ep_to_diff_0.30, best_eval)"
```

---

## Task 7 : Flags CLI `--gamma`, `--bx-repr-oracle`, `--bx-novelty-beta`

**Files:**
- Modify: `scripts/train_cnn_lstm_dqn_procedural.py` (argparse ~ligne 106 ; construction `ConvRecurrentDQNConfig` ~ligne 125-150)

- [ ] **Step 1: Ajouter les arguments argparse**

Dans `scripts/train_cnn_lstm_dqn_procedural.py`, juste avant `args = parser.parse_args()` (ligne 107), ajouter :

```python
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="V2-BX Sonde A (horizon) : discount factor. Default 0.99. "
             "Sonde A recommande : 0.997 (horizon effectif ~333, coherent max-steps 400).",
    )
    parser.add_argument(
        "--bx-repr-oracle",
        choices=("none", "scalar", "field"),
        default="none",
        help="V2-BX Sonde C (representation) : 4e canal oracle BFS. "
             "none=baseline, scalar=C1 (plan uniforme distance agent), "
             "field=C2 (champ distance par cellule). Default none.",
    )
    parser.add_argument(
        "--bx-novelty-beta",
        type=float,
        default=0.0,
        help="V2-BX Sonde B (exploration) : poids du bonus count-based "
             "beta/sqrt(visits) par cellule/episode. Default 0.0 = desactive. "
             "Point de depart recommande : 0.05 a 0.1.",
    )
```

- [ ] **Step 2: Wirer les flags dans la config**

Dans la construction de `ConvRecurrentDQNConfig` (ligne 125-150), ajouter ces trois lignes (par ex. après `b1a_mix_ratio=args.b1a_mix_ratio,`) :

```python
        gamma=args.gamma,
        bx_repr_oracle=args.bx_repr_oracle,
        bx_novelty_beta=args.bx_novelty_beta,
```

- [ ] **Step 3: Smoke manuel CPU (2 épisodes par sonde)**

Run:
```bash
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 2 --mode obstacles \
    --device cpu --max-rows 5 --max-cols 5 --max-steps 20 --no-eval \
    --bx-repr-oracle scalar
```
Expected: se termine sans erreur, affiche une ligne contenant `BX_PROBE_RESULT probe_type=representation`.

Run:
```bash
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 2 --mode obstacles \
    --device cpu --max-rows 5 --max-cols 5 --max-steps 20 --no-eval \
    --bx-novelty-beta 0.1
```
Expected: se termine, ligne `BX_PROBE_RESULT probe_type=exploration`.

Run:
```bash
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 2 --mode obstacles \
    --device cpu --max-rows 5 --max-cols 5 --max-steps 20 --no-eval \
    --gamma 0.997
```
Expected: se termine, ligne `BX_PROBE_RESULT probe_type=horizon`.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_cnn_lstm_dqn_procedural.py
git commit -m "feat(v2-bx): CLI flags --gamma (Sonde A) + --bx-repr-oracle (C) + --bx-novelty-beta (B)"
```

---

## Task 8 : Smoke CI des sondes

**Files:**
- Modify: `.github/workflows/aether_verify.yml` (ajout d'un step smoke après les smokes B1a existants)

- [ ] **Step 1: Lire le workflow existant pour suivre le pattern**

Run: `grep -n "b1a\|train_cnn_lstm" .github/workflows/aether_verify.yml`
Expected: repérer les steps smoke B1a (`--b1a` seul + `--b1a --polyak-tau 0.005`) pour imiter leur structure exacte (nom de step, `run:` multi-lignes).

- [ ] **Step 2: Ajouter les steps smoke BX**

Dans `.github/workflows/aether_verify.yml`, juste après le dernier step smoke V2-B1a, ajouter (en respectant l'indentation YAML du fichier et le device CPU des autres smokes) :

```yaml
      - name: Smoke V2-BX Sonde C (repr oracle scalar)
        run: |
          python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles \
            --device cpu --max-rows 5 --max-cols 5 --max-steps 20 --no-eval \
            --bx-repr-oracle scalar

      - name: Smoke V2-BX Sonde B (novelty) + Polyak cohabit
        run: |
          python scripts/train_cnn_lstm_dqn_procedural.py --episodes 10 --mode obstacles \
            --device cpu --max-rows 5 --max-cols 5 --max-steps 20 --no-eval \
            --bx-novelty-beta 0.1 --polyak-tau 0.005
```

- [ ] **Step 3: Valider le YAML localement**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/aether_verify.yml')); print('YAML OK')"`
Expected: `YAML OK` (pas d'erreur de parsing / indentation).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/aether_verify.yml
git commit -m "ci(v2-bx): smoke jobs Sonde C (repr scalar) + Sonde B (novelty) + Polyak cohabit"
```

---

## Task 9 : Vérification end-to-end + non-régression complète

**Files:** aucun (vérification)

- [ ] **Step 1: Suite complète**

Run: `pytest -q`
Expected: **375 passed** (356 baseline + 6 bfs + 5 config + 6 obs + ~6 runner). Le compte exact peut varier de ±2 selon le regroupement ; l'essentiel : **0 failed**, et tous les nouveaux fichiers de test verts.

- [ ] **Step 2: Reproductibilité baseline (no-op des flags)**

Run:
```bash
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 20 --mode obstacles \
    --device cpu --max-rows 5 --max-cols 5 --max-steps 20 --no-eval --seed 0
```
puis le même avec `--bx-repr-oracle none --bx-novelty-beta 0.0 --gamma 0.99` explicites.
Expected: les deux runs produisent une ligne `BX_PROBE_RESULT probe_type=baseline` et un `Final : winrate=...` identique (flags à leur défaut = no-op strict).

- [ ] **Step 3: Smoke Aether inchangé**

Run: `bash aether/verify_all.sh`
Expected: `8 OK`, exit 0.

- [ ] **Step 4: Commit (si ajustements)**

Si des ajustements ont été nécessaires :
```bash
git add -A
git commit -m "test(v2-bx): end-to-end verification + baseline no-op reproducibility"
```

---

## Task 10 : Bench Sonde C1 (15×15, n=3, GPU) — GATE manuel

> **Note d'exécution :** Tasks 10-11 sont des **runs GPU manuels** sur RTX 3060 (pas du code). Elles suivent le protocole de la spec §6. Chaque run ≈ 0,75 h. Lancer, collecter les lignes `BX_PROBE_RESULT`, remplir la table.

**Files:**
- Create: `docs/findings/2026-05-30-v2bx-diagnostic-results.md` (table de résultats, créée et remplie au fil des runs)

- [ ] **Step 1: Lancer Sonde C1 (scalar) n=3 seeds**

```bash
for seed in 0 1 2; do
  python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 5000 --mode obstacles --device cuda --seed $seed \
    --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --max-attempts-bfs 500 \
    --bx-repr-oracle scalar \
    --best-checkpoint-path checkpoints/bx_c1_seed${seed}.pt \
    2>&1 | tee logs/bx_c1_seed${seed}.log
done
```

- [ ] **Step 2: Collecter `diff_max` et remplir la table**

Run: `grep "BX_PROBE_RESULT" logs/bx_c1_seed*.log`
Reporter `diff_max` par seed dans `docs/findings/2026-05-30-v2bx-diagnostic-results.md`.

- [ ] **Step 3: Appliquer l'arbre de décision (spec §6)**

```text
moyenne diff_max C1 (n=3) :
  >= 0.45  -> REPRESENTATION confirmee -> Task 11 (confirmation V2-V n=5)
  0.40-0.45 -> escalade C2 (Task 10bis : meme commande, --bx-repr-oracle field)
  < 0.40   -> representation simple insuffisante -> Sonde A (--gamma 0.997)
```

- [ ] **Step 4: Escalades conditionnelles selon le verdict**

Selon le résultat, lancer la sonde suivante (même boucle n=3, seuls les flags changent) :
- **Escalade C2** : `--bx-repr-oracle field` (au lieu de `scalar`).
- **Sonde A** : retirer `--bx-repr-oracle`, ajouter `--gamma 0.997`.
- **Sonde B** : retirer les flags repr/gamma, ajouter `--bx-novelty-beta 0.1` (re-essai `0.05` si frôle 0.45).

Reporter chaque `diff_max` dans la table. S'arrêter dès qu'une sonde atteint `diff_max ≥ 0.45` (→ Task 11), ou conclure « bottleneck plus profond » si B échoue aussi.

- [ ] **Step 5: Commit de la table**

```bash
git add docs/findings/2026-05-30-v2bx-diagnostic-results.md logs/bx_*.log
git commit -m "bench(v2-bx): Sonde C1 (+ escalades) n=3 15x15 diff_max results"
```

---

## Task 11 : Confirmation V2-V de la sonde gagnante + finding + tag

**Files:**
- Modify: `docs/findings/2026-05-30-v2bx-diagnostic-results.md` (confirmation n=5)
- Modify: `CLAUDE.md` (section V2-BX)
- Modify: mémoire (`~/.claude/.../memory/`)

- [ ] **Step 1: Confirmation n=5 de la famille gagnante (si une sonde ≥ 0.45)**

Relancer la sonde gagnante sur **seeds 0-4** (mêmes flags que la passe n=3 gagnante), best-checkpoint distinct par seed. Collecter `best_eval_0.30`.

```text
Critere de validation : mean best @ diff=0.30 (n=5) : 64 % -> >= 74 %
```

- [ ] **Step 2: Rédiger le finding consolidé**

Compléter `docs/findings/2026-05-30-v2bx-diagnostic-results.md` : table `diff_max` par sonde, famille identifiée (ou « bottleneck plus profond »), confirmation V2-V, lecture causale, et la **prochaine étape** (sous-projet de la vraie solution non-oracle : auxiliary distance head si C, n-step propre si A, ICM/curiosity si B).

- [ ] **Step 3: Mettre à jour CLAUDE.md + mémoire**

Ajouter une section « V2-BX — diagnostic » au CLAUDE.md (statut, verdict, sonde gagnante, suite). Mettre à jour `MEMORY.md` + un fichier mémoire d'état d'exécution V2-BX.

- [ ] **Step 4: Tag**

```bash
pytest -q   # confirmer vert avant tag
git add -A && git commit -m "docs(v2-bx): finding consolide + CLAUDE.md + memoire"
git tag v0.2.0-bx
```

---

## Self-Review (rempli par l'auteur du plan)

- **Couverture spec** : §5.1 Sonde C → Tasks 1,3,4 ; §5.2 Sonde A → Task 7 ; §5.3 Sonde B → Task 5 ; §6 protocole/métrique → Tasks 6,10,11 ; §7 composants/isolation → Tasks 2,4 ; §8 tests → toutes ; §9 livrable/tag → Task 11. ✅
- **No-op des flags** (reproductibilité baseline) : Task 9 Step 2. ✅
- **Cohérence des noms** : `bx_repr_oracle` ("none"/"scalar"/"field"), `bx_novelty_beta`, `bfs_distance_field`, `_probe_descriptor`, `_reset_novelty`/`_novelty_bonus`, ligne `BX_PROBE_RESULT` — identiques entre config, encodeur, runner, CLI et tests. ✅
- **Placeholders** : aucun ; tout step de code contient le code complet. Tasks 10-11 sont explicitement des runs GPU manuels (pas du code), avec commandes exactes. ✅
- **Pièges projet** : help-text/erreurs ASCII (pas de γ/β/τ littéraux) ; `--max-attempts-bfs 500` au bench ; modules livrés touchés (runner, encoder, config) sous flags conditionnels documentés par la spec. ✅
