# MW_IA Procedural Environment & Curriculum Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faire passer l'agent DQN MW_IA d'un résolveur d'une seule map (V1) à un résolveur de labyrinthes en général via un environnement procédural avec curriculum learning adaptatif, sans casser V1.

**Architecture:** Couche au-dessus de V1 (zéro modification invasive). 2 générateurs de mazes (obstacles aléatoires BFS-checkés + maze parfait DFS-backtracker), wrapper `ProceduralGridWorld` qui régénère à chaque reset, scheduler adaptatif piloté par winrate, tracker par bucket de difficulté, runner DQN procedural avec callbacks GUI étendus.

**Tech Stack:** Python 3.13, NumPy, PyTorch (cu128), pytest, hypothesis (déjà installé V2-A), PyQt6 + PyQtGraph (GUI V1).

**Spec source:** `docs/superpowers/specs/2026-05-22-mw-ia-procedural-env-design.md`

**État initial:** Branche `main`, tag `v0.2.0-a` posé, 95 tests pytest verts. Pattern : développement sur `main` (pas de feature branch, comme V1 et V2-A).

**Adaptation à la spec découverte pendant exploration codebase :**
- Spec parle d'observation `[row, col, *grid]` (dim 2+R*C). En réalité V1 utilise déjà un encoding **one-hot** sur la position (`DQNRunner._state_vec` → `np.zeros(n_states), [idx]=1`). On garde ce pattern pour cohérence : observation procédural = `concat(position_one_hot, grid_flatten)` → dim `2 * (max_rows * max_cols)`. Pour 10×10 : 200 dim. Adaptation documentée dans la docstring de `encode()` (Task 17).
- `QNetwork` (`mw_ia/neural/network.py`) est DÉJÀ paramétrable (`input_dim` argument constructeur). Pas besoin de modification — il suffit de l'appeler avec le bon `input_dim`.

---

## Phase 1 — Setup

### Task 1 : Scaffold + fixtures

**Files :**
- Create : `mw_ia/envs/maze_generators.py` (vide)
- Create : `tests/envs/__init__.py` (vide, si absent)
- Create : `tests/envs/test_maze_generators.py` (vide)
- Modify : `tests/envs/conftest.py` (créer si absent)

- [ ] **Step 1 — Verify state initial**

```bash
source .venv/Scripts/activate && pytest -q
```

Expected : `95 passed` (état post V2-A).

- [ ] **Step 2 — Create empty scaffolds**

```bash
mkdir -p tests/envs
```

Contenu de `tests/envs/__init__.py` : vide.
Contenu de `tests/envs/conftest.py` :

```python
"""Fixtures partagées pour les tests envs."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Générateur seedé pour tests stochastiques."""
    return np.random.default_rng(seed=42)
```

Contenu de `mw_ia/envs/maze_generators.py` :

```python
"""Générateurs de mazes procéduraux + helper de solvabilité BFS.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-procedural-env-design.md §2.1
"""
from __future__ import annotations
```

Contenu de `tests/envs/test_maze_generators.py` : vide.

- [ ] **Step 3 — Verify pytest still passes**

```bash
source .venv/Scripts/activate && pytest -q
```

Expected : `95 passed` (les nouveaux fichiers vides sont ignorés).

- [ ] **Step 4 — Commit**

```bash
git add mw_ia/envs/maze_generators.py tests/envs/__init__.py tests/envs/conftest.py tests/envs/test_maze_generators.py
git commit -m "chore(procedural): scaffold mw_ia/envs/maze_generators + tests/envs/"
```

---

## Phase 2 — `maze_bfs_check`

### Task 2 : Helper de solvabilité BFS

**Files :**
- Modify : `mw_ia/envs/maze_generators.py`
- Modify : `tests/envs/test_maze_generators.py`

- [ ] **Step 1 — Add failing tests**

Contenu de `tests/envs/test_maze_generators.py` :

```python
"""Tests de maze_generators."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.envs.maze_generators import maze_bfs_check


def test_bfs_trivial_empty_grid():
    grid = np.zeros((5, 5), dtype=bool)
    assert maze_bfs_check(grid, start=(0, 0), goal=(4, 4)) is True


def test_bfs_with_obstacles_solvable():
    grid = np.zeros((5, 5), dtype=bool)
    grid[1, 1] = True  # obstacle isolé
    grid[3, 3] = True
    assert maze_bfs_check(grid, start=(0, 0), goal=(4, 4)) is True


def test_bfs_blocked_full_wall():
    grid = np.zeros((5, 5), dtype=bool)
    grid[2, :] = True  # mur horizontal complet à la rangée 2
    assert maze_bfs_check(grid, start=(0, 0), goal=(4, 4)) is False


def test_bfs_start_equals_goal():
    grid = np.zeros((5, 5), dtype=bool)
    assert maze_bfs_check(grid, start=(2, 2), goal=(2, 2)) is True


def test_bfs_start_on_obstacle_raises():
    grid = np.zeros((5, 5), dtype=bool)
    grid[0, 0] = True
    with pytest.raises(AssertionError):
        maze_bfs_check(grid, start=(0, 0), goal=(4, 4))


def test_bfs_goal_on_obstacle_raises():
    grid = np.zeros((5, 5), dtype=bool)
    grid[4, 4] = True
    with pytest.raises(AssertionError):
        maze_bfs_check(grid, start=(0, 0), goal=(4, 4))


def test_bfs_start_out_of_grid_raises():
    grid = np.zeros((5, 5), dtype=bool)
    with pytest.raises(AssertionError):
        maze_bfs_check(grid, start=(-1, 0), goal=(4, 4))
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_maze_generators.py -v
```

Expected : `ImportError: cannot import name 'maze_bfs_check'`.

- [ ] **Step 3 — Implement**

Ajouter à `mw_ia/envs/maze_generators.py` :

```python
from collections import deque

import numpy as np


def maze_bfs_check(
    grid: np.ndarray, *, start: tuple[int, int], goal: tuple[int, int]
) -> bool:
    """Retourne True ssi un chemin existe entre start et goal en évitant les obstacles.

    Args:
        grid: tableau (rows, cols) de booléens. True = obstacle.
        start: (row, col) de la position de départ.
        goal: (row, col) de la position d'arrivée.

    Returns:
        True si un chemin 4-connexe existe ; False sinon.

    Raises:
        AssertionError: si start ou goal est hors grille ou sur un obstacle.
    """
    rows, cols = grid.shape
    sr, sc = start
    gr, gc = goal
    assert 0 <= sr < rows and 0 <= sc < cols, f"start {start} hors grille {grid.shape}"
    assert 0 <= gr < rows and 0 <= gc < cols, f"goal {goal} hors grille {grid.shape}"
    assert not grid[sr, sc], f"start {start} sur obstacle"
    assert not grid[gr, gc], f"goal {goal} sur obstacle"

    if start == goal:
        return True

    visited = np.zeros_like(grid, dtype=bool)
    visited[sr, sc] = True
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if visited[nr, nc] or grid[nr, nc]:
                continue
            if (nr, nc) == goal:
                return True
            visited[nr, nc] = True
            queue.append((nr, nc))
    return False
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_maze_generators.py -v
```

Expected : `7 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/envs/maze_generators.py tests/envs/test_maze_generators.py
git commit -m "feat(procedural): maze_bfs_check helper for solvability"
```

---

## Phase 3 — `RandomObstaclesGenerator`

### Task 3 : Generator par placement aléatoire d'obstacles

**Files :**
- Modify : `mw_ia/envs/maze_generators.py`
- Modify : `tests/envs/test_maze_generators.py`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/envs/test_maze_generators.py` :

```python
from hypothesis import given, settings
from hypothesis import strategies as st

from mw_ia.envs.maze_generators import RandomObstaclesGenerator


def _gen() -> RandomObstaclesGenerator:
    return RandomObstaclesGenerator(
        rows=10, cols=10, start=(0, 0), goal=(9, 9),
        min_density=0.10, max_density=0.50, max_attempts=100,
    )


def test_random_obstacles_density_zero():
    gen = _gen()
    grid = gen.generate(seed=42, difficulty=0.0)
    # density interpolated to min_density=0.10 (0.0 difficulty → 10% obstacles)
    # On 10x10 = 100 cells, ~10 obstacles. On vérifie présence non-nulle et borne haute.
    n_obstacles = int(grid.sum())
    assert 0 < n_obstacles <= 15


def test_random_obstacles_density_max():
    gen = _gen()
    grid = gen.generate(seed=42, difficulty=1.0)
    # difficulty=1.0 → max_density=0.50 → ~50 obstacles
    n_obstacles = int(grid.sum())
    assert 40 <= n_obstacles <= 50


def test_random_obstacles_seed_deterministic():
    gen = _gen()
    g1 = gen.generate(seed=42, difficulty=0.5)
    g2 = gen.generate(seed=42, difficulty=0.5)
    assert np.array_equal(g1, g2)


def test_random_obstacles_start_goal_never_obstacle():
    gen = _gen()
    for seed in range(50):
        grid = gen.generate(seed=seed, difficulty=0.5)
        assert grid[0, 0] == False, f"seed={seed}: start sur obstacle"
        assert grid[9, 9] == False, f"seed={seed}: goal sur obstacle"


def test_random_obstacles_always_solvable():
    gen = _gen()
    for seed in range(50):
        grid = gen.generate(seed=seed, difficulty=0.5)
        assert maze_bfs_check(grid, start=(0, 0), goal=(9, 9)), \
            f"seed={seed}: maze non solvable"


def test_random_obstacles_pathological_density_raises():
    # max_density=0.95 → quasi-aucun chemin possible avec start=(0,0), goal=(9,9)
    gen = RandomObstaclesGenerator(
        rows=10, cols=10, start=(0, 0), goal=(9, 9),
        min_density=0.95, max_density=0.95, max_attempts=10,
    )
    with pytest.raises(RuntimeError, match="unreachable after"):
        gen.generate(seed=42, difficulty=0.5)


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=50, deadline=None)
def test_random_obstacles_property_solvability(seed: int):
    """Property : tout maze généré est solvable."""
    gen = _gen()
    grid = gen.generate(seed=seed, difficulty=0.5)
    assert maze_bfs_check(grid, start=(0, 0), goal=(9, 9))
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_maze_generators.py -v
```

Expected : `ImportError: cannot import name 'RandomObstaclesGenerator'`.

- [ ] **Step 3 — Implement**

Ajouter à `mw_ia/envs/maze_generators.py` :

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RandomObstaclesGenerator:
    """Generator par placement aléatoire d'obstacles avec BFS-check.

    La difficulté ∈ [0,1] interpole linéairement entre min_density et max_density.

    Pattern de génération (contrainte non-négociable, cf. spec) :
        1. tire density*rows*cols obstacles aléatoires
        2. exclut start et goal du tirage
        3. maze_bfs_check → True : retourner, sinon retry
        4. après max_attempts tentatives : RuntimeError
    """

    rows: int
    cols: int
    start: tuple[int, int]
    goal: tuple[int, int]
    min_density: float = 0.10
    max_density: float = 0.50
    max_attempts: int = 100

    def __post_init__(self) -> None:
        assert self.rows > 0 and self.cols > 0
        assert 0 <= self.start[0] < self.rows and 0 <= self.start[1] < self.cols
        assert 0 <= self.goal[0] < self.rows and 0 <= self.goal[1] < self.cols
        if not (0.0 <= self.min_density <= self.max_density <= 1.0):
            raise ValueError(
                f"densités invalides : min={self.min_density}, max={self.max_density}"
            )
        if self.max_attempts <= 0:
            raise ValueError(f"max_attempts doit être > 0, reçu {self.max_attempts}")

    def generate(self, *, seed: int, difficulty: float) -> np.ndarray:
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        density = self.min_density + (self.max_density - self.min_density) * difficulty
        n_cells = self.rows * self.cols
        n_obstacles = int(round(density * n_cells))

        rng = np.random.default_rng(seed=seed)
        sr, sc = self.start
        gr, gc = self.goal
        # Cellules candidates : toutes sauf start et goal
        all_indices = np.arange(n_cells)
        forbidden = {sr * self.cols + sc, gr * self.cols + gc}
        candidates = np.array([i for i in all_indices if i not in forbidden])

        for attempt in range(self.max_attempts):
            picks = rng.choice(candidates, size=min(n_obstacles, len(candidates)), replace=False)
            grid = np.zeros((self.rows, self.cols), dtype=bool)
            grid.flat[picks] = True
            if maze_bfs_check(grid, start=self.start, goal=self.goal):
                return grid

        raise RuntimeError(
            f"density={density:.2f} unreachable after {self.max_attempts} attempts "
            f"(rows={self.rows}, cols={self.cols}, start={self.start}, goal={self.goal})"
        )
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_maze_generators.py -v
```

Expected : `13 passed` (7 + 6 nouveaux ; le property-based test compte pour 1).

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/envs/maze_generators.py tests/envs/test_maze_generators.py
git commit -m "feat(procedural): RandomObstaclesGenerator with BFS-checked solvability"
```

---

## Phase 4 — `PerfectMazeGenerator`

### Task 4 : Generator DFS recursive backtracker

**Files :**
- Modify : `mw_ia/envs/maze_generators.py`
- Modify : `tests/envs/test_maze_generators.py`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/envs/test_maze_generators.py` :

```python
from mw_ia.envs.maze_generators import PerfectMazeGenerator


def test_perfect_maze_size_4_solvable():
    gen = PerfectMazeGenerator(min_size=4, max_size=20)
    grid = gen.generate(seed=42, difficulty=0.0)
    rows, cols = grid.shape
    assert rows == cols == 4
    assert maze_bfs_check(grid, start=(0, 0), goal=(rows - 1, cols - 1))


def test_perfect_maze_size_20_solvable():
    gen = PerfectMazeGenerator(min_size=4, max_size=20)
    grid = gen.generate(seed=42, difficulty=1.0)
    rows, cols = grid.shape
    assert rows == cols == 20
    assert maze_bfs_check(grid, start=(0, 0), goal=(rows - 1, cols - 1))


def test_perfect_maze_seed_deterministic():
    gen = PerfectMazeGenerator(min_size=4, max_size=20)
    g1 = gen.generate(seed=42, difficulty=0.5)
    g2 = gen.generate(seed=42, difficulty=0.5)
    assert np.array_equal(g1, g2)


def test_perfect_maze_invalid_size_raises():
    with pytest.raises(ValueError):
        PerfectMazeGenerator(min_size=20, max_size=4)  # min > max
    with pytest.raises(ValueError):
        PerfectMazeGenerator(min_size=1, max_size=10)  # min < 2


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=30, deadline=None)
def test_perfect_maze_property_solvability(seed: int):
    """Property : tout maze parfait est solvable par construction."""
    gen = PerfectMazeGenerator(min_size=4, max_size=20)
    grid = gen.generate(seed=seed, difficulty=0.5)
    rows, cols = grid.shape
    assert maze_bfs_check(grid, start=(0, 0), goal=(rows - 1, cols - 1))
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_maze_generators.py -v
```

Expected : `ImportError: cannot import name 'PerfectMazeGenerator'`.

- [ ] **Step 3 — Implement**

Ajouter à `mw_ia/envs/maze_generators.py` :

```python
@dataclass(frozen=True)
class PerfectMazeGenerator:
    """Generator de maze parfait via DFS recursive backtracker.

    Maze parfait = un et un seul chemin entre deux cellules. Solvable par
    construction (pas besoin de BFS-check post-hoc).

    La difficulté ∈ [0,1] interpole la TAILLE entre min_size et max_size.
    """

    min_size: int = 4
    max_size: int = 20

    def __post_init__(self) -> None:
        if self.min_size < 2:
            raise ValueError(f"min_size doit être >= 2, reçu {self.min_size}")
        if self.min_size >= self.max_size:
            raise ValueError(
                f"min_size ({self.min_size}) doit être < max_size ({self.max_size})"
            )

    def generate(self, *, seed: int, difficulty: float) -> np.ndarray:
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        size = self.min_size + int(round((self.max_size - self.min_size) * difficulty))

        # Initialement, toutes les cellules sont des obstacles. Le DFS creuse
        # des couloirs en marquant False (cellule libre).
        grid = np.ones((size, size), dtype=bool)
        rng = np.random.default_rng(seed=seed)

        # Carve depuis (0,0). Pour garantir start (0,0) et goal (size-1, size-1)
        # libres, le DFS classique sur maze sur grid creuse en partant de start.
        # On ouvre start et goal explicitement à la fin pour cohérence.
        stack: list[tuple[int, int]] = [(0, 0)]
        grid[0, 0] = False
        while stack:
            r, c = stack[-1]
            # Voisins non visités à distance 2 (cellule + mur)
            neighbors: list[tuple[int, int, int, int]] = []
            for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and grid[nr, nc]:
                    # mur entre (r,c) et (nr,nc) : (r+dr/2, c+dc/2)
                    wr, wc = r + dr // 2, c + dc // 2
                    neighbors.append((nr, nc, wr, wc))
            if not neighbors:
                stack.pop()
                continue
            pick = neighbors[rng.integers(0, len(neighbors))]
            nr, nc, wr, wc = pick
            grid[wr, wc] = False  # casser le mur
            grid[nr, nc] = False  # ouvrir la cellule
            stack.append((nr, nc))

        # Garantir goal libre (le DFS sur taille paire peut le laisser fermé)
        grid[size - 1, size - 1] = False
        return grid
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_maze_generators.py -v
```

Expected : `18 passed` (13 + 5 nouveaux).

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/envs/maze_generators.py tests/envs/test_maze_generators.py
git commit -m "feat(procedural): PerfectMazeGenerator (DFS recursive backtracker)"
```

---

## Phase 5 — Configs

### Task 5 : `ProceduralEnvConfig` + `SchedulerConfig`

**Files :**
- Modify : `mw_ia/config.py`
- Create : `tests/test_procedural_config.py`

- [ ] **Step 1 — Write failing tests**

`tests/test_procedural_config.py` :

```python
"""Tests des nouveaux dataclasses de configuration procedural."""
from __future__ import annotations

import pytest

from mw_ia.config import ProceduralEnvConfig, SchedulerConfig


def test_procedural_config_default_valid():
    cfg = ProceduralEnvConfig(mode="obstacles")
    assert cfg.mode == "obstacles"
    assert cfg.max_rows == 10
    assert cfg.max_cols == 10


def test_procedural_config_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        ProceduralEnvConfig(mode="invalid")  # type: ignore[arg-type]


def test_procedural_config_density_inverted_raises():
    with pytest.raises(ValueError, match="density"):
        ProceduralEnvConfig(mode="obstacles", min_density=0.5, max_density=0.1)


def test_procedural_config_size_inverted_raises():
    with pytest.raises(ValueError, match="size"):
        ProceduralEnvConfig(mode="maze", min_size=20, max_size=4)


def test_procedural_config_is_frozen():
    cfg = ProceduralEnvConfig(mode="obstacles")
    with pytest.raises(Exception):
        cfg.mode = "maze"  # type: ignore[misc]


def test_scheduler_config_default_valid():
    cfg = SchedulerConfig()
    assert cfg.initial_difficulty == 0.0
    assert cfg.up_threshold == 0.80
    assert cfg.down_threshold == 0.30


def test_scheduler_config_thresholds_inverted_raises():
    with pytest.raises(ValueError, match="threshold"):
        SchedulerConfig(up_threshold=0.30, down_threshold=0.80)


def test_scheduler_config_initial_out_of_range_raises():
    with pytest.raises(ValueError, match="initial"):
        SchedulerConfig(initial_difficulty=1.5)


def test_scheduler_config_is_frozen():
    cfg = SchedulerConfig()
    with pytest.raises(Exception):
        cfg.step = 0.10  # type: ignore[misc]
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/test_procedural_config.py -v
```

Expected : `ImportError: cannot import name 'ProceduralEnvConfig'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/config.py` :

```python
from typing import Literal


@dataclass(frozen=True)
class ProceduralEnvConfig:
    """Configuration de l'environnement procédural V2-X."""

    mode: Literal["obstacles", "maze"]
    max_rows: int = 10
    max_cols: int = 10
    min_density: float = 0.10           # mode obstacles uniquement
    max_density: float = 0.50
    min_size: int = 4                   # mode maze uniquement
    max_size: int = 20
    max_attempts_bfs: int = 100         # mode obstacles : tentatives avant RuntimeError

    def __post_init__(self) -> None:
        if self.mode not in ("obstacles", "maze"):
            raise ValueError(f"mode doit être 'obstacles' ou 'maze', reçu {self.mode!r}")
        if not (0.0 <= self.min_density <= self.max_density <= 1.0):
            raise ValueError(
                f"densités invalides : min_density={self.min_density}, "
                f"max_density={self.max_density}"
            )
        if not (2 <= self.min_size < self.max_size):
            raise ValueError(
                f"tailles invalides : min_size={self.min_size}, max_size={self.max_size}"
            )
        if self.max_attempts_bfs <= 0:
            raise ValueError(f"max_attempts_bfs doit être > 0, reçu {self.max_attempts_bfs}")


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration du scheduler adaptatif de difficulté."""

    initial_difficulty: float = 0.0
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    up_threshold: float = 0.80
    down_threshold: float = 0.30
    step: float = 0.05
    update_interval: int = 50           # épisodes

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_difficulty <= self.max_difficulty <= 1.0):
            raise ValueError(
                f"difficultés invalides : min={self.min_difficulty}, max={self.max_difficulty}"
            )
        if not (self.min_difficulty <= self.initial_difficulty <= self.max_difficulty):
            raise ValueError(
                f"initial_difficulty={self.initial_difficulty} hors "
                f"[{self.min_difficulty}, {self.max_difficulty}]"
            )
        if self.up_threshold <= self.down_threshold:
            raise ValueError(
                f"up_threshold ({self.up_threshold}) doit être > "
                f"down_threshold ({self.down_threshold})"
            )
        if not (0.0 < self.step <= 1.0):
            raise ValueError(f"step doit être ∈ (0,1], reçu {self.step}")
        if self.update_interval <= 0:
            raise ValueError(f"update_interval doit être > 0, reçu {self.update_interval}")
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/test_procedural_config.py -v
```

Expected : `9 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/config.py tests/test_procedural_config.py
git commit -m "feat(procedural): ProceduralEnvConfig + SchedulerConfig dataclasses"
```

---

## Phase 6 — `AdaptiveDifficultyScheduler`

### Task 6 : Scheduler adaptatif piloté par winrate

**Files :**
- Create : `mw_ia/training/scheduler.py`
- Create : `tests/training/test_scheduler.py`

- [ ] **Step 1 — Write failing tests**

`tests/training/test_scheduler.py` :

```python
"""Tests de AdaptiveDifficultyScheduler."""
from __future__ import annotations

import pytest

from mw_ia.config import SchedulerConfig
from mw_ia.training.scheduler import AdaptiveDifficultyScheduler


def test_scheduler_starts_at_initial():
    s = AdaptiveDifficultyScheduler(SchedulerConfig(initial_difficulty=0.2))
    assert s.current == 0.2


def test_scheduler_winrate_above_up_threshold_increases():
    s = AdaptiveDifficultyScheduler(
        SchedulerConfig(initial_difficulty=0.5, up_threshold=0.80, step=0.05)
    )
    new = s.update(winrate=0.9)
    assert new == pytest.approx(0.55)
    assert s.current == pytest.approx(0.55)


def test_scheduler_winrate_below_down_threshold_decreases():
    s = AdaptiveDifficultyScheduler(
        SchedulerConfig(initial_difficulty=0.5, down_threshold=0.30, step=0.05)
    )
    new = s.update(winrate=0.2)
    assert new == pytest.approx(0.45)


def test_scheduler_winrate_in_neutral_zone_unchanged():
    s = AdaptiveDifficultyScheduler(
        SchedulerConfig(initial_difficulty=0.5,
                        up_threshold=0.80, down_threshold=0.30, step=0.05)
    )
    new = s.update(winrate=0.5)
    assert new == 0.5
    assert s.current == 0.5


def test_scheduler_clamps_to_max():
    s = AdaptiveDifficultyScheduler(
        SchedulerConfig(initial_difficulty=0.95, up_threshold=0.80,
                        max_difficulty=1.0, step=0.10)
    )
    new = s.update(winrate=0.9)
    assert new == 1.0  # 0.95 + 0.10 = 1.05 → clamp à 1.0


def test_scheduler_clamps_to_min():
    s = AdaptiveDifficultyScheduler(
        SchedulerConfig(initial_difficulty=0.05, down_threshold=0.30,
                        min_difficulty=0.0, step=0.10)
    )
    new = s.update(winrate=0.1)
    assert new == 0.0  # 0.05 - 0.10 = -0.05 → clamp à 0.0


def test_scheduler_stable_at_neutral_winrate():
    s = AdaptiveDifficultyScheduler(SchedulerConfig(initial_difficulty=0.5))
    for _ in range(100):
        s.update(winrate=0.5)
    assert s.current == 0.5
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/training/test_scheduler.py -v
```

Expected : `ModuleNotFoundError: No module named 'mw_ia.training.scheduler'`.

- [ ] **Step 3 — Implement**

`mw_ia/training/scheduler.py` :

```python
"""Scheduler adaptatif de difficulté piloté par winrate."""
from __future__ import annotations

import numpy as np

from mw_ia.config import SchedulerConfig


class AdaptiveDifficultyScheduler:
    """Monte/descend la difficulté selon le winrate observé.

    Règle :
        winrate >= up_threshold   → current += step (cap max_difficulty)
        winrate <= down_threshold → current -= step (floor min_difficulty)
        sinon                     → inchangé

    L'instance est mutable (self.current change), mais la config est frozen.
    """

    def __init__(self, cfg: SchedulerConfig) -> None:
        self.cfg = cfg
        self.current: float = cfg.initial_difficulty

    def update(self, *, winrate: float) -> float:
        """Met à jour current selon winrate et retourne la nouvelle valeur."""
        if winrate >= self.cfg.up_threshold:
            self.current = float(np.clip(
                self.current + self.cfg.step, self.cfg.min_difficulty, self.cfg.max_difficulty
            ))
        elif winrate <= self.cfg.down_threshold:
            self.current = float(np.clip(
                self.current - self.cfg.step, self.cfg.min_difficulty, self.cfg.max_difficulty
            ))
        return self.current
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/training/test_scheduler.py -v
```

Expected : `7 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/training/scheduler.py tests/training/test_scheduler.py
git commit -m "feat(procedural): AdaptiveDifficultyScheduler driven by winrate"
```

---

## Phase 7 — `DifficultyBucketTracker`

### Task 7 : Tracker par bucket de difficulté

**Files :**
- Modify : `mw_ia/training/metrics.py`
- Create : `tests/training/test_bucket_tracker.py`

- [ ] **Step 1 — Add `has_data()` to MetricsTracker first**

Le `DifficultyBucketTracker` utilise `has_data()` ; on l'ajoute à `MetricsTracker` d'abord. Pas de test isolé — testé indirectement via le bucket tracker.

Modifier `mw_ia/training/metrics.py`, ajouter à la classe `MetricsTracker` après la méthode `winrate()` :

```python
    def has_data(self) -> bool:
        """True si au moins un épisode a été enregistré."""
        return bool(self._success_window)
```

- [ ] **Step 2 — Write failing tests**

`tests/training/test_bucket_tracker.py` :

```python
"""Tests de DifficultyBucketTracker."""
from __future__ import annotations

from mw_ia.config import TrainingConfig
from mw_ia.training.metrics import DifficultyBucketTracker


def _tracker() -> DifficultyBucketTracker:
    return DifficultyBucketTracker(TrainingConfig())


def test_bucket_routing_low():
    t = _tracker()
    t.record_episode(success=True, reward=1.0, length=10, difficulty=0.15)
    wr = t.winrate_per_bucket()
    assert wr[0] == 1.0
    for i in range(1, 5):
        assert wr[i] is None


def test_bucket_routing_high():
    t = _tracker()
    t.record_episode(success=True, reward=1.0, length=10, difficulty=0.85)
    wr = t.winrate_per_bucket()
    assert wr[4] == 1.0
    for i in range(4):
        assert wr[i] is None


def test_bucket_routing_max_difficulty_inclusive():
    """difficulty=1.0 doit aller dans le bucket 4, pas 5 (out of bounds)."""
    t = _tracker()
    t.record_episode(success=True, reward=1.0, length=10, difficulty=1.0)
    wr = t.winrate_per_bucket()
    assert wr[4] == 1.0


def test_bucket_empty_returns_none():
    t = _tracker()
    wr = t.winrate_per_bucket()
    assert wr == [None, None, None, None, None]


def test_bucket_winrate_per_bucket_returns_5_values():
    t = _tracker()
    for d, s in [(0.1, True), (0.3, False), (0.5, True), (0.7, True), (0.9, False)]:
        t.record_episode(success=s, reward=1.0 if s else 0.0, length=10, difficulty=d)
    wr = t.winrate_per_bucket()
    assert len(wr) == 5
    assert all(w is not None for w in wr)
```

- [ ] **Step 3 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/training/test_bucket_tracker.py -v
```

Expected : `ImportError: cannot import name 'DifficultyBucketTracker'`.

- [ ] **Step 4 — Implement**

Ajouter en bas de `mw_ia/training/metrics.py` :

```python
class DifficultyBucketTracker:
    """5 MetricsTracker, un par bucket de difficulté [0,0.2), [0.2,0.4), ...

    Routing : bucket = min(4, int(difficulty * 5)). Le min(4, ...) gère
    difficulty=1.0 qui donnerait sinon bucket=5 (out of bounds).
    """

    N_BUCKETS = 5

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self._trackers: list[MetricsTracker] = [
            MetricsTracker(cfg) for _ in range(self.N_BUCKETS)
        ]

    def record_episode(
        self, *, success: bool, reward: float, length: int, difficulty: float
    ) -> None:
        bucket = min(self.N_BUCKETS - 1, int(difficulty * self.N_BUCKETS))
        self._trackers[bucket].record_episode(reward, length, success=success)

    def winrate_per_bucket(self) -> list[float | None]:
        return [t.winrate() if t.has_data() else None for t in self._trackers]
```

- [ ] **Step 5 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/training/test_bucket_tracker.py -v
```

Expected : `5 passed`.

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/training/metrics.py tests/training/test_bucket_tracker.py
git commit -m "feat(procedural): DifficultyBucketTracker (5 buckets) + MetricsTracker.has_data"
```

---

## Phase 8 — `ProceduralGridWorld`

### Task 8 : Wrapper env qui régénère à chaque reset

**Files :**
- Create : `mw_ia/envs/procedural_env.py`
- Create : `tests/envs/test_procedural_env.py`

- [ ] **Step 1 — Write failing tests**

`tests/envs/test_procedural_env.py` :

```python
"""Tests de ProceduralGridWorld."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.config import ProceduralEnvConfig
from mw_ia.envs.gridworld import Action
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld


def _env() -> ProceduralGridWorld:
    cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                              min_density=0.10, max_density=0.30)
    gen = RandomObstaclesGenerator(rows=10, cols=10, start=(0, 0), goal=(9, 9),
                                   min_density=cfg.min_density, max_density=cfg.max_density)
    return ProceduralGridWorld(cfg=cfg, generator=gen)


def test_reset_returns_state_and_info_with_maze():
    env = _env()
    env.set_difficulty(0.5)
    state, info = env.reset(seed=42)
    assert state == (0, 0)
    assert "maze" in info
    assert info["maze"].shape == (10, 10)
    assert info["difficulty"] == 0.5
    assert info["episode_id"] == 42


def test_reset_with_different_seeds_gives_different_mazes():
    env = _env()
    env.set_difficulty(0.5)
    _, info1 = env.reset(seed=1)
    _, info2 = env.reset(seed=2)
    assert not np.array_equal(info1["maze"], info2["maze"])


def test_reset_with_same_seed_is_deterministic():
    env = _env()
    env.set_difficulty(0.5)
    _, info1 = env.reset(seed=42)
    _, info2 = env.reset(seed=42)
    assert np.array_equal(info1["maze"], info2["maze"])


def test_step_delegates_to_v1_gridworld():
    """Vérifie que step() respecte les rewards V1."""
    env = _env()
    env.set_difficulty(0.0)  # densité min, moins de chance d'obstacle hit
    env.reset(seed=42)
    _, reward, terminated, truncated, info = env.step(Action.RIGHT)
    # Au moins le step_penalty doit être appliqué
    assert reward <= 0.0 or terminated  # négatif sauf si on tombe sur goal directement
    assert "step" in info


def test_changing_difficulty_changes_maze():
    env = _env()
    env.set_difficulty(0.0)
    _, info_easy = env.reset(seed=42)
    env.set_difficulty(1.0)
    _, info_hard = env.reset(seed=42)
    assert not np.array_equal(info_easy["maze"], info_hard["maze"]) \
        or info_easy["maze"].sum() != info_hard["maze"].sum()
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_procedural_env.py -v
```

Expected : `ModuleNotFoundError: No module named 'mw_ia.envs.procedural_env'`.

- [ ] **Step 3 — Implement**

`mw_ia/envs/procedural_env.py` :

```python
"""Wrapper procédural sur GridWorld V1 qui régénère le maze à chaque reset."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from mw_ia.config import GridWorldConfig, ProceduralEnvConfig
from mw_ia.envs.gridworld import Action, GridWorld


class MazeGenerator(Protocol):
    """Interface commune des générateurs de mazes."""

    def generate(self, *, seed: int, difficulty: float) -> np.ndarray:
        ...


class ProceduralGridWorld:
    """Régénère un nouveau maze à chaque reset() via un MazeGenerator.

    Délègue step() à un GridWorld V1 interne reconstruit à chaque reset.
    """

    def __init__(
        self,
        *,
        cfg: ProceduralEnvConfig,
        generator: MazeGenerator,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
    ) -> None:
        self.cfg = cfg
        self.generator = generator
        self.start = start
        # Goal par défaut : coin opposé. Si maze de taille < max_*, sera ajusté
        # à goal réel au reset() (mais GridWorld interne aura ses propres coords).
        self.goal = goal if goal is not None else (cfg.max_rows - 1, cfg.max_cols - 1)
        self._difficulty: float = 0.0
        self._inner: GridWorld | None = None

    def set_difficulty(self, difficulty: float) -> None:
        self._difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def reset(self, *, seed: int) -> tuple[tuple[int, int], dict[str, Any]]:
        maze = self.generator.generate(seed=seed, difficulty=self._difficulty)
        rows, cols = maze.shape
        goal = (rows - 1, cols - 1)
        obstacles = tuple(
            (int(r), int(c))
            for r, c in zip(*np.where(maze))
        )
        gw_cfg = GridWorldConfig(
            rows=rows, cols=cols,
            start=self.start, goal=goal,
            obstacles=obstacles,
        )
        self._inner = GridWorld(gw_cfg)
        state, _ = self._inner.reset()
        info = {
            "maze": maze,
            "difficulty": self._difficulty,
            "episode_id": seed,
            "step": 0,
        }
        return state, info

    def step(
        self, action: Action | int
    ) -> tuple[tuple[int, int], float, bool, bool, dict[str, Any]]:
        assert self._inner is not None, "reset() doit être appelé avant step()"
        return self._inner.step(action)

    @property
    def inner(self) -> GridWorld:
        assert self._inner is not None, "reset() doit être appelé avant inner"
        return self._inner
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_procedural_env.py -v
```

Expected : `5 passed`.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/envs/procedural_env.py tests/envs/test_procedural_env.py
git commit -m "feat(procedural): ProceduralGridWorld wrapper (regenerates maze per reset)"
```

---

## Phase 9 — Observation encoder

### Task 9 : `encode()` helper avec padding

**Files :**
- Modify : `mw_ia/envs/procedural_env.py`
- Modify : `tests/envs/test_procedural_env.py`

- [ ] **Step 1 — Add failing tests**

Ajouter en bas de `tests/envs/test_procedural_env.py` :

```python
from mw_ia.envs.procedural_env import encode_procedural_observation


def test_encode_dim_matches_2x_max_size():
    """Observation = position_one_hot (R*C dim) + grid_flatten (R*C dim) = 2*R*C."""
    grid = np.zeros((10, 10), dtype=bool)
    obs = encode_procedural_observation(state=(0, 0), grid=grid, max_rows=10, max_cols=10)
    assert obs.shape == (200,)  # 2 * 10 * 10
    assert obs.dtype == np.float32


def test_encode_position_one_hot():
    grid = np.zeros((10, 10), dtype=bool)
    obs = encode_procedural_observation(state=(0, 0), grid=grid, max_rows=10, max_cols=10)
    # Première moitié = position one-hot. State (0,0) = index 0.
    assert obs[0] == 1.0
    assert obs[1:100].sum() == 0.0


def test_encode_grid_in_second_half():
    grid = np.zeros((10, 10), dtype=bool)
    grid[2, 2] = True  # obstacle à (2,2) = index 22
    obs = encode_procedural_observation(state=(0, 0), grid=grid, max_rows=10, max_cols=10)
    # Deuxième moitié contient le grid. obstacle à index 22 dans la deuxième moitié.
    assert obs[100 + 22] == 1.0


def test_encode_padding_when_grid_smaller_than_max():
    """Maze 4x4 dans grille max 10x10 → padding top-left avec zéros."""
    grid = np.ones((4, 4), dtype=bool)
    grid[0, 0] = False  # start libre
    grid[3, 3] = False  # goal libre
    obs = encode_procedural_observation(state=(0, 0), grid=grid, max_rows=10, max_cols=10)
    assert obs.shape == (200,)
    # Le grid 4x4 est placé top-left dans une 10x10. Les obstacles sont aux
    # positions (r,c) avec r,c ∈ [0,3]. La cellule (3,3) (goal) doit être 0,
    # et la cellule (4,4) (hors maze original) doit être 0 aussi.
    assert obs[100 + 33] == 0.0  # goal libre
    assert obs[100 + 44] == 0.0  # padding


def test_encode_grid_too_large_raises():
    grid = np.zeros((11, 11), dtype=bool)
    with pytest.raises(AssertionError):
        encode_procedural_observation(state=(0, 0), grid=grid, max_rows=10, max_cols=10)
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_procedural_env.py -v
```

Expected : `ImportError: cannot import name 'encode_procedural_observation'`.

- [ ] **Step 3 — Implement**

Ajouter en bas de `mw_ia/envs/procedural_env.py` :

```python
def encode_procedural_observation(
    *, state: tuple[int, int], grid: np.ndarray, max_rows: int, max_cols: int
) -> np.ndarray:
    """Encode l'observation procédural pour QNetwork.

    Format : concat(position_one_hot, grid_flatten) → np.float32 de dim 2*max_rows*max_cols.

    Pour les mazes plus petits que max_rows × max_cols (mode maze parfait avec
    difficulté variable), la grille est placée top-left dans une zone paddée
    de zéros (= cellules libres). Conséquence : l'agent voit des bordures
    artificielles qu'il apprend à ignorer.

    Args:
        state: position (row, col) de l'agent.
        grid: maze actuel (rows ≤ max_rows, cols ≤ max_cols), True = obstacle.
        max_rows: nombre de rangées max (dim du QNetwork).
        max_cols: nombre de colonnes max.

    Returns:
        np.ndarray[float32] de shape (2 * max_rows * max_cols,).
    """
    rows, cols = grid.shape
    assert rows <= max_rows and cols <= max_cols, (
        f"grid {grid.shape} > max ({max_rows}, {max_cols})"
    )

    n_cells = max_rows * max_cols
    obs = np.zeros(2 * n_cells, dtype=np.float32)

    # Position one-hot dans la grille max_rows × max_cols
    r, c = state
    obs[r * max_cols + c] = 1.0

    # Grid paddé top-left dans la deuxième moitié
    padded = np.zeros((max_rows, max_cols), dtype=np.float32)
    padded[:rows, :cols] = grid.astype(np.float32)
    obs[n_cells:] = padded.flatten()

    return obs
```

- [ ] **Step 4 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/envs/test_procedural_env.py -v
```

Expected : `10 passed` (5 + 5 nouveaux).

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/envs/procedural_env.py tests/envs/test_procedural_env.py
git commit -m "feat(procedural): encode_procedural_observation with top-left zero padding"
```

---

## Phase 10 — `ProceduralDQNRunner`

### Task 10 : Runner DQN procedural avec callbacks GUI étendus

**Files :**
- Modify : `mw_ia/training/runner.py`
- Create : `tests/training/test_procedural_runner.py`

- [ ] **Step 1 — Write failing tests**

`tests/training/test_procedural_runner.py` :

```python
"""Tests d'intégration de ProceduralDQNRunner."""
from __future__ import annotations

import numpy as np

from mw_ia.config import (
    DQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ProceduralDQNRunner, RunnerCallbacks


def _make_runner(episodes: int = 20, force_winrate: float | None = None) -> ProceduralDQNRunner:
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.10, max_density=0.30)
    gen = RandomObstaclesGenerator(rows=10, cols=10, start=(0, 0), goal=(9, 9),
                                   min_density=proc_cfg.min_density,
                                   max_density=proc_cfg.max_density)
    env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
    dqn_cfg = DQNConfig(episodes=episodes, min_replay_to_learn=10,
                        max_steps_per_episode=30, use_amp=False,
                        replay_capacity=1_000)
    sched_cfg = SchedulerConfig(initial_difficulty=0.0, step=0.05, update_interval=5)
    train_cfg = TrainingConfig()
    return ProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=RunnerCallbacks(),
        device="cpu", seed=0,
    )


def test_procedural_runner_20_episodes_no_error():
    runner = _make_runner(episodes=20)
    runner.run()
    assert runner.metrics.total_episodes == 20


def test_procedural_runner_emits_maze_changed_callback():
    captured: list[tuple[np.ndarray, int, float]] = []
    cb = RunnerCallbacks(on_maze_changed=lambda **kw: captured.append(
        (kw["maze"], kw["episode_id"], kw["difficulty"])
    ))
    runner = _make_runner(episodes=5)
    runner.callbacks = cb
    runner.run()
    assert len(captured) == 5
    assert captured[0][0].shape == (10, 10)


def test_procedural_runner_bucket_tracker_filled():
    runner = _make_runner(episodes=20)
    runner.run()
    bucket_wr = runner.bucket_tracker.winrate_per_bucket()
    # Au moins le bucket 0 (difficulté faible début) doit avoir des données
    assert bucket_wr[0] is not None
```

- [ ] **Step 2 — Run, expect fail**

```bash
source .venv/Scripts/activate && pytest tests/training/test_procedural_runner.py -v
```

Expected : `ImportError: cannot import name 'ProceduralDQNRunner'`.

- [ ] **Step 3 — Extend `RunnerCallbacks`**

Modifier `mw_ia/training/runner.py`, dans la dataclass `RunnerCallbacks`, ajouter après `on_log` :

```python
    on_maze_changed: Callable[..., None] | None = None
    on_difficulty_updated: Callable[..., None] | None = None

    def fire_maze_changed(self, **kw: object) -> None:
        if self.on_maze_changed:
            self.on_maze_changed(**kw)

    def fire_difficulty_updated(self, **kw: object) -> None:
        if self.on_difficulty_updated:
            self.on_difficulty_updated(**kw)
```

- [ ] **Step 4 — Implement `ProceduralDQNRunner`**

Ajouter en bas de `mw_ia/training/runner.py` :

```python
from mw_ia.config import ProceduralEnvConfig, SchedulerConfig
from mw_ia.envs.procedural_env import ProceduralGridWorld, encode_procedural_observation
from mw_ia.training.metrics import DifficultyBucketTracker
from mw_ia.training.scheduler import AdaptiveDifficultyScheduler


class ProceduralDQNRunner(_BaseRunner):
    """Boucle DQN sur environnement procédural avec curriculum adaptatif.

    Différences avec DQNRunner V1 :
    - env régénère le maze à chaque reset()
    - observation = position_one_hot + grid_flatten (dim 2*max_rows*max_cols)
    - scheduler adapte la difficulté toutes les update_interval épisodes
    - bucket_tracker route les épisodes par bucket de difficulté
    - callbacks GUI étendus : on_maze_changed, on_difficulty_updated

    NOTE : le replay buffer mélange volontairement transitions inter-épisodes.
    C'est précisément le but du curriculum : apprendre une politique générale,
    pas une politique map-spécifique.
    """

    def __init__(
        self,
        *,
        env: ProceduralGridWorld,
        proc_cfg: ProceduralEnvConfig,
        dqn_cfg: DQNConfig,
        sched_cfg: SchedulerConfig,
        train_cfg: TrainingConfig,
        callbacks: RunnerCallbacks,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        super().__init__(train_cfg, callbacks)
        self.env = env
        self.proc_cfg = proc_cfg
        self.dqn_cfg = dqn_cfg
        self.sched_cfg = sched_cfg
        self.scheduler = AdaptiveDifficultyScheduler(sched_cfg)
        self.bucket_tracker = DifficultyBucketTracker(train_cfg)
        # Observation dim = 2 * max_rows * max_cols (one-hot pos + grid flatten)
        obs_dim = 2 * proc_cfg.max_rows * proc_cfg.max_cols
        # DQNAgent attend un env avec n_states/n_actions ; on adapte
        # en injectant directement obs_dim via un objet léger.
        self.agent = self._make_agent(obs_dim=obs_dim, device=device, seed=seed)

    def _make_agent(self, *, obs_dim: int, device: str, seed: int) -> "DQNAgent":
        """Construit un DQNAgent qui consomme des observations de dim obs_dim.

        Hack léger : DQNAgent V1 dérive obs_dim de env.n_states. On bypass en
        wrappant l'env procedural dans un objet exposant n_states=obs_dim et
        n_actions=4.
        """
        class _ObsDimEnv:
            n_states = obs_dim
            n_actions = 4

        return DQNAgent(_ObsDimEnv(), self.dqn_cfg, device=device, seed=seed)

    def run(self) -> None:
        self.callbacks.fire_log(
            "info",
            f"Procedural DQN ({self.proc_cfg.mode}) sur {self.agent.device} démarrage"
        )
        for ep in range(self.dqn_cfg.episodes):
            if self._stop:
                return

            self.env.set_difficulty(self.scheduler.current)
            state, info = self.env.reset(seed=ep)
            maze = info["maze"]
            difficulty = info["difficulty"]
            self.callbacks.fire_maze_changed(maze=maze, episode_id=ep, difficulty=difficulty)

            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated) and ep_len < self.dqn_cfg.max_steps_per_episode:
                if self._stop:
                    return
                while self._paused and not self._stop:
                    pass
                obs = encode_procedural_observation(
                    state=state, grid=maze,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                a = self.agent.act(obs)
                s2, r, terminated, truncated, _ = self.env.step(a)
                next_obs = encode_procedural_observation(
                    state=s2, grid=maze,
                    max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
                )
                m = self.agent.observe(obs, a, r, next_obs, terminated or truncated)
                if "loss" in m:
                    self.metrics.record_loss(m["loss"])
                    self.callbacks.fire_loss(self.agent.global_step, m["loss"])
                self.callbacks.fire_epsilon(self.agent.global_step, m["epsilon"])
                self.metrics.record_epsilon(m["epsilon"])
                self.callbacks.fire_step(state=state, action=a, reward=r, next_state=s2)
                state = s2
                ep_reward += r
                ep_len += 1

            self.metrics.record_episode(ep_reward, ep_len, success=terminated)
            self.bucket_tracker.record_episode(
                success=terminated, reward=ep_reward, length=ep_len, difficulty=difficulty,
            )
            self.callbacks.fire_episode(ep=ep, reward=ep_reward, length=ep_len, success=terminated)

            if (ep + 1) % self.sched_cfg.update_interval == 0:
                new_diff = self.scheduler.update(winrate=self.metrics.winrate())
                self.callbacks.fire_difficulty_updated(difficulty=new_diff, episode_id=ep)

            if ep % self.train_cfg.log_every_episodes == 0:
                self.callbacks.fire_log(
                    "info",
                    f"ep {ep:>4}  R={ep_reward:+.2f}  L={ep_len:>3}  "
                    f"eps={self.agent.epsilon:.3f}  winrate={self.metrics.winrate():.2%}  "
                    f"diff={self.scheduler.current:.2f}"
                )
```

- [ ] **Step 5 — Run, expect pass**

```bash
source .venv/Scripts/activate && pytest tests/training/test_procedural_runner.py -v
```

Expected : `3 passed`.

- [ ] **Step 6 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q
```

Expected : ≥ **140 passed** (95 V1+V2-A + ~45 nouveaux jusqu'ici).

- [ ] **Step 7 — Commit**

```bash
git add mw_ia/training/runner.py tests/training/test_procedural_runner.py
git commit -m "feat(procedural): ProceduralDQNRunner with curriculum and GUI callbacks"
```

---

## Phase 11 — CLI script + CI

### Task 11 : `scripts/train_dqn_procedural.py`

**Files :**
- Create : `scripts/train_dqn_procedural.py`

- [ ] **Step 1 — Write the CLI script**

`scripts/train_dqn_procedural.py` :

```python
"""Entraînement DQN procedural headless (CLI).

Usage :
    python scripts/train_dqn_procedural.py --episodes 200 --mode obstacles --device cpu
"""
from __future__ import annotations

import argparse
import sys

from mw_ia.config import (
    DQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import PerfectMazeGenerator, RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ProceduralDQNRunner, RunnerCallbacks


def _print_log(level: str, msg: str) -> None:
    print(f"[{level:7s}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="DQN procedural training")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--mode", choices=("obstacles", "maze"), default="obstacles")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    proc_cfg = ProceduralEnvConfig(mode=args.mode)
    if args.mode == "obstacles":
        gen = RandomObstaclesGenerator(
            rows=proc_cfg.max_rows, cols=proc_cfg.max_cols,
            start=(0, 0), goal=(proc_cfg.max_rows - 1, proc_cfg.max_cols - 1),
            min_density=proc_cfg.min_density, max_density=proc_cfg.max_density,
            max_attempts=proc_cfg.max_attempts_bfs,
        )
    else:
        gen = PerfectMazeGenerator(min_size=proc_cfg.min_size, max_size=proc_cfg.max_size)

    env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
    dqn_cfg = DQNConfig(episodes=args.episodes)
    sched_cfg = SchedulerConfig()
    train_cfg = TrainingConfig()

    cb = RunnerCallbacks(on_log=_print_log)
    runner = ProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=cb, device=args.device, seed=args.seed,
    )
    runner.run()

    final_wr = runner.metrics.winrate()
    final_diff = runner.scheduler.current
    print(f"\nFinal : winrate={final_wr:.2%}, difficulty={final_diff:.2f}")
    print("Per-bucket winrate :")
    for i, wr in enumerate(runner.bucket_tracker.winrate_per_bucket()):
        wr_str = f"{wr:.2%}" if wr is not None else "—"
        print(f"  bucket {i} ({i*0.2:.1f}-{(i+1)*0.2:.1f}) : {wr_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2 — Smoke test on CPU**

```bash
source .venv/Scripts/activate && python scripts/train_dqn_procedural.py --episodes 20 --mode obstacles --device cpu
```

Expected : pas d'erreur, log toutes les 10 épisodes, ligne "Final : winrate=X.XX%, difficulty=Y.YY" à la fin, et "Per-bucket winrate :" affiche 5 lignes.

- [ ] **Step 3 — Smoke test mode maze**

```bash
source .venv/Scripts/activate && python scripts/train_dqn_procedural.py --episodes 20 --mode maze --device cpu
```

Expected : même pattern, taille de maze variable selon difficulté.

- [ ] **Step 4 — Update CI workflow to include procedural smoke**

Modifier `.github/workflows/aether_verify.yml`, ajouter un step dans le job `pytest` après `Run pytest` :

```yaml
      - name: Smoke test procedural training
        run: |
          python scripts/train_dqn_procedural.py --episodes 10 --mode obstacles --device cpu
          python scripts/train_dqn_procedural.py --episodes 10 --mode maze --device cpu
```

- [ ] **Step 5 — Commit**

```bash
git add scripts/train_dqn_procedural.py .github/workflows/aether_verify.yml
git commit -m "feat(procedural): scripts/train_dqn_procedural.py CLI + CI smoke"
```

---

## Phase 12 — GUI integration

### Task 12 : Signal `maze_changed` dans la GUI

**Files :**
- Modify : `mw_ia/gui/widgets/grid.py` (à identifier dans le code existant)
- Modify : `mw_ia/gui/app.py`

> **Note pour l'engineer** : la GUI V1 est dans `mw_ia/gui/`. Avant de coder, lire `mw_ia/gui/app.py` et identifier le widget Grid + le `TrainingThread` Qt pour brancher le signal `maze_changed`. Suivre **exactement** le pattern V1 (`pyqtSignal` + `QueuedConnection`).

- [ ] **Step 1 — Read existing GUI structure**

```bash
ls mw_ia/gui/widgets/
cat mw_ia/gui/app.py | head -80
```

Identifier : le widget qui dessine la grille (probablement `grid.py`), le `QThread` runner, où sont déclarés les `pyqtSignal`.

- [ ] **Step 2 — Add `maze_changed` signal to TrainingThread**

Dans le fichier `app.py` (ou wherever le `QThread` du runner est défini), ajouter un `pyqtSignal` :

```python
maze_changed = pyqtSignal(object, int, float)  # maze, episode_id, difficulty
```

Brancher `RunnerCallbacks.on_maze_changed` pour émettre ce signal :

```python
def _on_maze_changed(self, *, maze, episode_id, difficulty):
    self.maze_changed.emit(maze, episode_id, difficulty)
```

- [ ] **Step 3 — Grid widget redraws on signal**

Dans le widget grille (probablement `grid.py`), ajouter une slot :

```python
@pyqtSlot(object, int, float)
def on_maze_changed(self, maze, episode_id, difficulty):
    """Redessine les obstacles à partir du nouveau maze (numpy bool array)."""
    self._obstacles = {(int(r), int(c)) for r, c in zip(*maze.nonzero())}
    self.update()  # ou self.redraw() selon le widget
```

Connecter `TrainingThread.maze_changed.connect(grid_widget.on_maze_changed)` dans `MainWindow.__init__` ou équivalent. Utiliser `Qt.ConnectionType.QueuedConnection` pour cross-thread.

- [ ] **Step 4 — Manual smoke (GUI cannot be unit-tested headless)**

```bash
source .venv/Scripts/activate && python scripts/launch_gui.py
```

Cliquer "Start procedural" (à ajouter en Task 14). À défaut, smoke test en lançant à la main `python scripts/launch_gui.py` et observant que rien ne casse en mode V1 (fix-map). **Documenter dans le commit** que GUI procedural est testé en Task 14.

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/gui/
git commit -m "feat(procedural-gui): maze_changed signal redraws grid widget"
```

### Task 13 : 5e courbe `difficulty` dans les plots

**Files :**
- Modify : `mw_ia/gui/widgets/plots.py`
- Modify : `mw_ia/gui/app.py`

- [ ] **Step 1 — Add `difficulty` curve to plots widget**

Dans `plots.py`, identifier le pattern V1 des 4 courbes (reward, loss, ε, winrate). Ajouter une 5e courbe `difficulty` :

```python
self._difficulty_curve = self._plot_widget.plot(
    pen=pg.mkPen(color="#FFA500", width=2), name="Difficulty"
)
self._difficulty_x: list[int] = []
self._difficulty_y: list[float] = []

@pyqtSlot(float, int)
def on_difficulty_updated(self, difficulty: float, episode_id: int):
    self._difficulty_x.append(episode_id)
    self._difficulty_y.append(difficulty)
    self._difficulty_curve.setData(self._difficulty_x, self._difficulty_y)
```

- [ ] **Step 2 — Connect signal in TrainingThread**

Dans `app.py`, ajouter un signal :

```python
difficulty_updated = pyqtSignal(float, int)
```

Brancher au callback :

```python
def _on_difficulty_updated(self, *, difficulty, episode_id):
    self.difficulty_updated.emit(difficulty, episode_id)
```

Connecter à `plots_widget.on_difficulty_updated`.

- [ ] **Step 3 — Manual smoke**

Lancer `python scripts/launch_gui.py` ; vérifier que le widget plots n'est pas cassé en mode V1.

- [ ] **Step 4 — Commit**

```bash
git add mw_ia/gui/
git commit -m "feat(procedural-gui): 5th curve 'difficulty' in plots widget"
```

### Task 14 : Widget `difficulty_label` + mode switch

**Files :**
- Create : `mw_ia/gui/widgets/difficulty_label.py`
- Modify : `mw_ia/gui/app.py`

- [ ] **Step 1 — Create the label widget**

`mw_ia/gui/widgets/difficulty_label.py` :

```python
"""Widget affichant 'Maze #N, diff=X.XX' pour le mode procedural."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QLabel


class DifficultyLabel(QLabel):
    """Label texte mis à jour à chaque maze_changed."""

    def __init__(self) -> None:
        super().__init__("Mode V1 (map fixe)")
        self.setStyleSheet("color: #DDD; font-family: monospace; font-size: 11pt;")

    @pyqtSlot(object, int, float)
    def on_maze_changed(self, maze, episode_id: int, difficulty: float) -> None:
        self.setText(f"Maze #{episode_id}, diff={difficulty:.2f}")
```

- [ ] **Step 2 — Add to main window**

Dans `app.py`, instancier `DifficultyLabel` et l'insérer dans le layout (à côté du Level label V1 probablement). Connecter au signal `maze_changed`.

- [ ] **Step 3 — Add 'Start procedural' button**

Ajouter un bouton dans la toolbar / panel : "Start (procedural)". Quand cliqué, instancier `ProceduralDQNRunner` au lieu de `DQNRunner` V1.

```python
def _on_start_procedural(self):
    proc_cfg = ProceduralEnvConfig(mode="obstacles")  # ou exposer un combo "mode"
    gen = RandomObstaclesGenerator(rows=10, cols=10, start=(0, 0), goal=(9, 9))
    env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
    self._runner = ProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=DQNConfig(),
        sched_cfg=SchedulerConfig(), train_cfg=TrainingConfig(),
        callbacks=self._make_callbacks(), device=self._device,
    )
    self._training_thread.runner = self._runner
    self._training_thread.start()
```

- [ ] **Step 4 — Manual smoke**

```bash
source .venv/Scripts/activate && python scripts/launch_gui.py
```

- Cliquer "Start (procedural)"
- Vérifier : grille change toutes les épisodes, label affiche "Maze #X, diff=Y.YY", 5e courbe difficulty apparaît, pas d'erreur dans la console
- Cliquer "Start" (V1 fix-map) : doit toujours marcher inchangé

- [ ] **Step 5 — Commit**

```bash
git add mw_ia/gui/
git commit -m "feat(procedural-gui): DifficultyLabel + 'Start procedural' button"
```

---

## Phase 13 — Documentation + tag

### Task 15 : README V2-X section

**Files :**
- Modify : `README.md`

- [ ] **Step 1 — Add procedural section**

Insérer après la section "V2-A — Aether guardrails (sous-projet livré)" et avant "Roadmap (V2+)" :

```markdown
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
- Mode `maze` : DFS recursive backtracker, solvable par construction.

### Curriculum adaptatif

Le scheduler ajuste la difficulté toutes les 50 épisodes :
- winrate >= 80 % → difficulté monte de 0.05
- winrate <= 30 % → difficulté descend de 0.05
- sinon : inchangé

Métriques par bucket de difficulté (5 buckets [0,0.2)..[0.8,1.0]) pour détecter
l'oubli catastrophique des niveaux faciles (préfigure le sous-projet D).

### GUI

`python scripts/launch_gui.py` puis bouton "Start (procedural)" : la grille
change à chaque épisode, une 5e courbe `difficulty(t)` s'ajoute aux 4 V1, et
un label "Maze #N, diff=X.XX" suit l'évolution.

### Architecture

- `mw_ia/envs/maze_generators.py` — `maze_bfs_check`, `RandomObstaclesGenerator`, `PerfectMazeGenerator`
- `mw_ia/envs/procedural_env.py` — `ProceduralGridWorld` + `encode_procedural_observation`
- `mw_ia/training/scheduler.py` — `AdaptiveDifficultyScheduler`
- `mw_ia/training/metrics.py` — `DifficultyBucketTracker`
- `mw_ia/training/runner.py` — `ProceduralDQNRunner`
- `mw_ia/gui/widgets/difficulty_label.py` — label "Maze #N, diff=X.XX"
```

- [ ] **Step 2 — Commit**

```bash
git add README.md
git commit -m "docs(readme): add V2-X procedural environment section"
```

### Task 16 : Definition of Done + tag

**Files :** aucune modification.

- [ ] **Step 1 — Full pytest**

```bash
source .venv/Scripts/activate && pytest -q
```

Expected : **≥ 145 passed** (95 baseline + ~52 nouveaux).

- [ ] **Step 2 — Smoke E2E obstacles**

```bash
source .venv/Scripts/activate && python scripts/train_dqn_procedural.py --episodes 50 --mode obstacles --device cpu
```

Expected : pas d'erreur, ligne "Final : winrate=..." et "Per-bucket winrate :" affichées.

- [ ] **Step 3 — Smoke E2E maze**

```bash
source .venv/Scripts/activate && python scripts/train_dqn_procedural.py --episodes 50 --mode maze --device cpu
```

Expected : idem.

- [ ] **Step 4 — Smoke GUI manuel**

```bash
source .venv/Scripts/activate && python scripts/launch_gui.py
```

Vérifications manuelles :
- "Start" V1 fonctionne (rétro-compat)
- "Start (procedural)" : grille change, label "Maze #X, diff=Y.YY" s'affiche, 5e courbe difficulty apparaît
- Pas d'erreur Qt cross-thread

- [ ] **Step 5 — Tag the release**

```bash
git tag -a v0.2.0-x -m "MW_IA V2-X — Procedural environment & curriculum learning

Sous-projet V2-X (parallele au programme V2 A-F). Livraison :
- 2 generateurs (RandomObstacles BFS-checked + PerfectMaze DFS-backtracker)
- ProceduralGridWorld wrapper qui regenere a chaque reset
- AdaptiveDifficultyScheduler pilote par winrate
- DifficultyBucketTracker (5 buckets)
- ProceduralDQNRunner avec callbacks GUI etendus
- GUI : signal maze_changed, 5e courbe difficulty, widget DifficultyLabel
- scripts/train_dqn_procedural.py + CI smoke

Garantie non-negociable validee 2026-05-22 : tout maze genere est solvable
(BFS-check post-hoc en mode obstacles, par construction en mode maze).

~52 nouveaux tests pytest (>= 145 verts au total)."
git log --oneline -5
```

- [ ] **Step 6 — Final verification**

```bash
git tag | grep v0
git status
```

Expected : `v0.1.0`, `v0.2.0-a`, `v0.2.0-x` tous présents. Working tree clean.

---

## Récapitulatif des fichiers livrés

```
MW_IA/
├── mw_ia/
│   ├── config.py                          [Task 5] + ProceduralEnvConfig + SchedulerConfig
│   ├── envs/
│   │   ├── maze_generators.py             [Tasks 2-4] maze_bfs_check + 2 générateurs
│   │   └── procedural_env.py              [Tasks 8-9] ProceduralGridWorld + encode
│   ├── training/
│   │   ├── metrics.py                     [Task 7] + DifficultyBucketTracker + has_data
│   │   ├── scheduler.py                   [Task 6] AdaptiveDifficultyScheduler
│   │   └── runner.py                      [Task 10] + ProceduralDQNRunner + 2 callbacks
│   └── gui/
│       ├── app.py                         [Tasks 12-14] signals + Start procedural
│       ├── widgets/
│       │   ├── grid.py                    [Task 12] slot on_maze_changed
│       │   ├── plots.py                   [Task 13] 5e courbe difficulty
│       │   └── difficulty_label.py        [Task 14] nouveau widget
├── scripts/
│   └── train_dqn_procedural.py            [Task 11] CLI procedural
├── tests/
│   ├── envs/
│   │   ├── conftest.py                    [Task 1]
│   │   ├── test_maze_generators.py        [Tasks 2-4] 18 tests
│   │   └── test_procedural_env.py         [Tasks 8-9] 10 tests
│   ├── training/
│   │   ├── test_scheduler.py              [Task 6] 7 tests
│   │   ├── test_bucket_tracker.py         [Task 7] 5 tests
│   │   └── test_procedural_runner.py      [Task 10] 3 tests
│   └── test_procedural_config.py          [Task 5] 9 tests
├── .github/workflows/aether_verify.yml    [Task 11] + CI smoke
└── README.md                              [Task 15] section V2-X
```

**Total :** 16 tâches sur 13 phases · ~52 nouveaux tests · 2 générateurs · 1 scheduler · 1 bucket tracker · 1 runner · 3 widgets GUI · 1 CLI script · 0 dépendance ajoutée.
