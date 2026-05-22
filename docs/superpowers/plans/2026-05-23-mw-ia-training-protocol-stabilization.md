# MW_IA V2-V Training Protocol Stabilization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter une évaluation périodique greedy sur seeds eval séparés du training + sauvegarde automatique du meilleur modèle observé, pour récupérer le pic de performance avant le late-stage collapse identifié par H1.

**Architecture:** 2 nouveaux composants découplés — `PeriodicEvaluator` (env eval séparé, méthode `evaluate(agent, difficulty)`) + `BestCheckpointTracker` (sauvegarde au pic d'eval_winrate). Branchés sur `ConvProceduralDQNRunner` uniquement via 5 nouveaux champs `ConvDQNConfig`. Zéro pollution training (buffer/global_step/scheduler/rng intacts grâce à `agent.act(..., greedy=True)`).

**Tech Stack:** Python 3.13, PyTorch (cu128), pytest, argparse `BooleanOptionalAction`. Réutilise infrastructure V2-Z/W (`ConvDQNAgent.act(greedy=True)`, `agent.save(path)`, `ProceduralGridWorld`, `encode_procedural_observation_2d`).

**Spec source:** `docs/superpowers/specs/2026-05-23-mw-ia-training-protocol-stabilization-design.md`

**État initial:** Branche `main`, tags `v0.1.0` + `v0.2.0-a` + `v0.2.0-x` + `v0.2.0-y` + `v0.2.0-z` + `v0.2.0-w` posés. **211 tests pytest verts**. Dernier commit avant V2-V : `370aba7` (spec V2-V).

---

## Phase 1 — Setup scaffold

### Task 1 : Créer les fichiers code + tests vides

**Files :**
- Create : `mw_ia/training/evaluator.py` (docstring seulement)
- Create : `mw_ia/training/checkpoint_tracker.py` (docstring seulement)
- Create : `tests/training/test_evaluator.py` (vide)
- Create : `tests/training/test_checkpoint_tracker.py` (vide)

- [ ] **Step 1 — Verify initial state**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `211 passed`.

- [ ] **Step 2 — Create scaffold files**

Contenu de `mw_ia/training/evaluator.py` :

```python
"""PeriodicEvaluator — évaluation périodique greedy pour V2-V.

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-training-protocol-stabilization-design.md §2
"""
from __future__ import annotations
```

Contenu de `mw_ia/training/checkpoint_tracker.py` :

```python
"""BestCheckpointTracker — sauvegarde automatique du meilleur modèle observé (V2-V).

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-training-protocol-stabilization-design.md §2
"""
from __future__ import annotations
```

Les 2 fichiers de tests (`test_evaluator.py`, `test_checkpoint_tracker.py`) restent vides à ce stade.

- [ ] **Step 3 — Verify pytest still passes**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `211 passed`.

- [ ] **Step 4 — Commit scaffold**

```bash
git add mw_ia/training/evaluator.py mw_ia/training/checkpoint_tracker.py \
        tests/training/test_evaluator.py tests/training/test_checkpoint_tracker.py
git commit -m "chore(v2-v): scaffold evaluator + checkpoint_tracker modules + empty test files"
```

---

## Phase 2 — `PeriodicEvaluator`

### Task 2 : Évaluateur greedy sur seeds eval séparés

**Files :**
- Modify : `mw_ia/training/evaluator.py` (impl)
- Test : `tests/training/test_evaluator.py`

- [ ] **Step 1 — Write the 8 failing tests**

Contenu de `tests/training/test_evaluator.py` :

```python
"""Tests V2-V de PeriodicEvaluator (greedy eval sans pollution training)."""
from __future__ import annotations

import numpy as np

from mw_ia.agents.conv_dqn import ConvDQNAgent
from mw_ia.config import ConvDQNConfig, ProceduralEnvConfig
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld, encode_procedural_observation_2d
from mw_ia.training.evaluator import PeriodicEvaluator


def _build_eval_env() -> ProceduralGridWorld:
    cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                              min_density=0.0, max_density=0.20)
    gen = RandomObstaclesGenerator(
        rows=10, cols=10, start=(0, 0), goal=(9, 9),
        min_density=cfg.min_density, max_density=cfg.max_density,
    )
    return ProceduralGridWorld(cfg=cfg, generator=gen)


def _build_agent() -> ConvDQNAgent:
    cfg = ConvDQNConfig(min_replay_to_learn=4, batch_size=2, train_every=1)
    return ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )


def _build_evaluator(eval_seeds: tuple[int, ...] = (10_000, 10_001, 10_002)) -> PeriodicEvaluator:
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.0, max_density=0.20)
    return PeriodicEvaluator(
        eval_env=_build_eval_env(),
        eval_seeds=eval_seeds,
        max_steps=30,
        observation_encoder=encode_procedural_observation_2d,
        proc_cfg=proc_cfg,
    )


def test_init_builds_eval_env() -> None:
    """Eval env est une instance ProceduralGridWorld distincte."""
    evaluator = _build_evaluator()
    assert isinstance(evaluator.eval_env, ProceduralGridWorld)
    assert evaluator.eval_seeds == (10_000, 10_001, 10_002)
    assert evaluator.max_steps == 30


def test_evaluate_returns_proper_metrics() -> None:
    """Le dict retourné contient winrate, mean_reward, mean_length, n_episodes, difficulty."""
    evaluator = _build_evaluator()
    agent = _build_agent()
    metrics = evaluator.evaluate(agent, difficulty=0.10)
    assert set(metrics.keys()) == {
        "winrate", "mean_reward", "mean_length", "n_episodes", "difficulty",
    }
    assert metrics["n_episodes"] == 3
    assert metrics["difficulty"] == 0.10


def test_evaluate_does_not_pollute_buffer() -> None:
    """CRITIQUE : len(agent.buffer) inchangé avant/après évaluation."""
    evaluator = _build_evaluator()
    agent = _build_agent()
    buffer_before = len(agent.buffer)
    evaluator.evaluate(agent, difficulty=0.10)
    assert len(agent.buffer) == buffer_before


def test_evaluate_does_not_increment_global_step() -> None:
    """CRITIQUE : agent.global_step inchangé."""
    evaluator = _build_evaluator()
    agent = _build_agent()
    step_before = agent.global_step
    evaluator.evaluate(agent, difficulty=0.10)
    assert agent.global_step == step_before


def test_evaluate_uses_greedy() -> None:
    """Eval déterministe : 2 évaluations successives donnent les mêmes métriques.

    Avec greedy=True, agent.act() bypass l'eps-greedy et ne touche pas
    self._rng. Deux évaluations sur les mêmes seeds doivent donner le
    même winrate (déterminisme strict).
    """
    evaluator = _build_evaluator()
    agent = _build_agent()
    m1 = evaluator.evaluate(agent, difficulty=0.10)
    m2 = evaluator.evaluate(agent, difficulty=0.10)
    assert m1["winrate"] == m2["winrate"]
    assert m1["mean_reward"] == m2["mean_reward"]
    assert m1["mean_length"] == m2["mean_length"]


def test_evaluate_runs_all_seeds() -> None:
    """n_episodes retourné = len(eval_seeds)."""
    evaluator = _build_evaluator(eval_seeds=(10_000, 10_001, 10_002, 10_003, 10_004))
    agent = _build_agent()
    metrics = evaluator.evaluate(agent, difficulty=0.10)
    assert metrics["n_episodes"] == 5


def test_evaluate_winrate_bounds() -> None:
    """winrate dans [0, 1] (compatible Aether I4)."""
    evaluator = _build_evaluator()
    agent = _build_agent()
    metrics = evaluator.evaluate(agent, difficulty=0.10)
    assert 0.0 <= metrics["winrate"] <= 1.0


def test_evaluate_respects_difficulty() -> None:
    """eval_env.set_difficulty(diff) appelé → _difficulty interne synchro."""
    evaluator = _build_evaluator()
    agent = _build_agent()
    evaluator.evaluate(agent, difficulty=0.25)
    assert evaluator.eval_env._difficulty == 0.25
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/training/test_evaluator.py -v 2>&1 | tail -15
```

Attendu : `ImportError` (`PeriodicEvaluator` n'existe pas).

- [ ] **Step 3 — Implement `PeriodicEvaluator`**

Remplacer le contenu de `mw_ia/training/evaluator.py` par :

```python
"""PeriodicEvaluator — évaluation périodique greedy pour V2-V.

Garantit zéro pollution du training :
- env eval distinct de l'env training (créé à l'init du PeriodicEvaluator)
- agent.act(obs, greedy=True) bypass eps-greedy et le rng training
- agent.observe() JAMAIS appelé → pas de buffer push, pas de global_step,
  pas de scheduler update
- torch.no_grad() interne à agent.act() (déjà géré par V2-Z)

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-training-protocol-stabilization-design.md §2
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from mw_ia.config import ProceduralEnvConfig
from mw_ia.envs.procedural_env import ProceduralGridWorld


class _ActableAgent(Protocol):
    """Contrat minimal qu'un agent doit respecter pour être évalué."""

    def act(self, state: np.ndarray, *, greedy: bool = False) -> int: ...


class PeriodicEvaluator:
    """Évalue un agent en mode greedy sur un set de seeds eval fixes.

    L'env eval est strictement séparé de l'env training pour garantir
    qu'aucune trajectoire d'évaluation ne pollue le replay buffer ou le
    scheduler de difficulté.
    """

    def __init__(
        self,
        *,
        eval_env: ProceduralGridWorld,
        eval_seeds: tuple[int, ...],
        max_steps: int,
        observation_encoder: Callable[..., np.ndarray],
        proc_cfg: ProceduralEnvConfig,
    ) -> None:
        if len(eval_seeds) == 0:
            raise ValueError("eval_seeds ne peut pas être vide")
        if max_steps <= 0:
            raise ValueError(f"max_steps doit être > 0, reçu {max_steps}")
        self.eval_env = eval_env
        self.eval_seeds = tuple(eval_seeds)
        self.max_steps = int(max_steps)
        self.observation_encoder = observation_encoder
        self.proc_cfg = proc_cfg

    def evaluate(
        self, agent: _ActableAgent, difficulty: float,
    ) -> dict[str, float]:
        """Lance len(eval_seeds) rollouts greedy. Retourne metrics dict.

        Args:
            agent: agent à évaluer (doit exposer act(state, greedy=True) -> int).
            difficulty: difficulty à set sur l'env eval avant les rollouts.

        Returns:
            dict avec keys : winrate, mean_reward, mean_length, n_episodes, difficulty.
        """
        self.eval_env.set_difficulty(difficulty)
        n_success = 0
        total_reward = 0.0
        total_length = 0
        for seed in self.eval_seeds:
            state, info = self.eval_env.reset(seed=seed)
            maze = info["maze"]
            goal = self.eval_env.inner.cfg.goal
            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated) and ep_len < self.max_steps:
                obs = self.observation_encoder(
                    state=state, grid=maze, goal=goal,
                    max_rows=self.proc_cfg.max_rows,
                    max_cols=self.proc_cfg.max_cols,
                )
                action = agent.act(obs, greedy=True)
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                ep_reward += reward
                ep_len += 1
            if terminated:
                n_success += 1
            total_reward += ep_reward
            total_length += ep_len

        n = len(self.eval_seeds)
        return {
            "winrate": n_success / n,
            "mean_reward": total_reward / n,
            "mean_length": total_length / n,
            "n_episodes": n,
            "difficulty": float(difficulty),
        }
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/training/test_evaluator.py -v 2>&1 | tail -15
```

Attendu : `8 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `219 passed` (211 + 8).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/training/evaluator.py tests/training/test_evaluator.py
git commit -m "feat(v2-v): add PeriodicEvaluator (greedy eval on separate seeds, no buffer pollution)"
```

---

## Phase 3 — `BestCheckpointTracker`

### Task 3 : Sauvegarde auto du meilleur modèle observé

**Files :**
- Modify : `mw_ia/training/checkpoint_tracker.py` (impl)
- Test : `tests/training/test_checkpoint_tracker.py`

- [ ] **Step 1 — Write the 6 failing tests**

Contenu de `tests/training/test_checkpoint_tracker.py` :

```python
"""Tests V2-V de BestCheckpointTracker (sauvegarde au pic eval_winrate)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from mw_ia.agents.conv_dqn import ConvDQNAgent
from mw_ia.config import ConvDQNConfig
from mw_ia.training.checkpoint_tracker import BestCheckpointTracker


def _build_agent() -> ConvDQNAgent:
    cfg = ConvDQNConfig(min_replay_to_learn=4, batch_size=2, train_every=1)
    return ConvDQNAgent(
        in_channels=3, rows=10, cols=10, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )


def test_init_no_best() -> None:
    """best_winrate = -inf, best_episode = None à l'init."""
    tracker = BestCheckpointTracker(path=None)
    assert math.isinf(tracker.best_winrate) and tracker.best_winrate < 0
    assert tracker.best_episode is None
    assert tracker.best_difficulty is None


def test_first_update_always_saves(tmp_path: Path) -> None:
    """Premier eval triggers save (improvement vs -inf)."""
    path = tmp_path / "best.pt"
    tracker = BestCheckpointTracker(path=path)
    agent = _build_agent()
    metrics = {"winrate": 0.0, "difficulty": 0.05}
    improved = tracker.update(metrics, agent, episode=100)
    assert improved is True
    assert path.exists()
    assert tracker.best_winrate == 0.0
    assert tracker.best_episode == 100
    assert tracker.best_difficulty == 0.05


def test_lower_winrate_does_not_save(tmp_path: Path) -> None:
    """eval_winrate < best → skip, fichier inchangé."""
    path = tmp_path / "best.pt"
    tracker = BestCheckpointTracker(path=path)
    agent = _build_agent()
    tracker.update({"winrate": 0.5, "difficulty": 0.10}, agent, episode=100)
    mtime_before = path.stat().st_mtime_ns
    improved = tracker.update({"winrate": 0.3, "difficulty": 0.15}, agent, episode=200)
    assert improved is False
    assert tracker.best_winrate == 0.5
    assert tracker.best_episode == 100
    assert path.stat().st_mtime_ns == mtime_before


def test_higher_winrate_saves_and_updates_best(tmp_path: Path) -> None:
    """eval_winrate > best → save + update state."""
    path = tmp_path / "best.pt"
    tracker = BestCheckpointTracker(path=path)
    agent = _build_agent()
    tracker.update({"winrate": 0.5, "difficulty": 0.10}, agent, episode=100)
    improved = tracker.update({"winrate": 0.7, "difficulty": 0.20}, agent, episode=200)
    assert improved is True
    assert tracker.best_winrate == 0.7
    assert tracker.best_episode == 200
    assert tracker.best_difficulty == 0.20


def test_path_none_no_save_attempted() -> None:
    """path=None → tracking en mémoire, pas d'IO."""
    tracker = BestCheckpointTracker(path=None)
    agent = _build_agent()
    improved = tracker.update({"winrate": 0.5, "difficulty": 0.10}, agent, episode=100)
    assert improved is True
    assert tracker.best_winrate == 0.5
    assert tracker.best_episode == 100


def test_equal_winrate_does_not_save(tmp_path: Path) -> None:
    """Idempotence : eval_winrate == best → skip."""
    path = tmp_path / "best.pt"
    tracker = BestCheckpointTracker(path=path)
    agent = _build_agent()
    tracker.update({"winrate": 0.5, "difficulty": 0.10}, agent, episode=100)
    mtime_before = path.stat().st_mtime_ns
    improved = tracker.update({"winrate": 0.5, "difficulty": 0.15}, agent, episode=200)
    assert improved is False
    assert tracker.best_episode == 100
    assert path.stat().st_mtime_ns == mtime_before
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/training/test_checkpoint_tracker.py -v 2>&1 | tail -15
```

Attendu : `ImportError` (`BestCheckpointTracker` n'existe pas).

- [ ] **Step 3 — Implement `BestCheckpointTracker`**

Remplacer le contenu de `mw_ia/training/checkpoint_tracker.py` par :

```python
"""BestCheckpointTracker — sauvegarde automatique du meilleur modèle observé (V2-V).

Maintient best_eval_winrate en mémoire et sauvegarde le modèle au pic.
Découple "savoir quand sauver" de "où sauver" — path=None pour tracking
en mémoire pur (utile pour tests / dry-run).

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-training-protocol-stabilization-design.md §2
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol


class _SaveableAgent(Protocol):
    """Contrat minimal qu'un agent doit respecter pour être sauvegardé."""

    def save(self, path: str | Path) -> None: ...


class BestCheckpointTracker:
    """Sauvegarde le modèle au pic d'eval_winrate observé.

    Idempotent : un eval avec winrate égal ou inférieur au meilleur observé
    ne déclenche aucune sauvegarde.
    """

    def __init__(self, path: str | Path | None) -> None:
        self._path: Path | None = Path(path) if path is not None else None
        self._best_winrate: float = -math.inf
        self._best_episode: int | None = None
        self._best_difficulty: float | None = None

    @property
    def best_winrate(self) -> float:
        return self._best_winrate

    @property
    def best_episode(self) -> int | None:
        return self._best_episode

    @property
    def best_difficulty(self) -> float | None:
        return self._best_difficulty

    @property
    def path(self) -> Path | None:
        return self._path

    def update(
        self, eval_metrics: dict[str, float], agent: _SaveableAgent, episode: int,
    ) -> bool:
        """Update best si nouveau pic. Retourne True si save déclenché.

        Args:
            eval_metrics: dict avec keys 'winrate' et 'difficulty'.
            agent: objet exposant save(path).
            episode: épisode courant (pour traçabilité du pic).

        Returns:
            True si nouveau best sauvegardé, False sinon (idempotent sur égalité).
        """
        wr = float(eval_metrics["winrate"])
        if wr > self._best_winrate:
            self._best_winrate = wr
            self._best_episode = int(episode)
            self._best_difficulty = float(eval_metrics["difficulty"])
            if self._path is not None:
                agent.save(self._path)
            return True
        return False
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/training/test_checkpoint_tracker.py -v 2>&1 | tail -15
```

Attendu : `6 passed`.

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `225 passed` (219 + 6).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/training/checkpoint_tracker.py tests/training/test_checkpoint_tracker.py
git commit -m "feat(v2-v): add BestCheckpointTracker (save model at peak eval_winrate, idempotent)"
```

---

## Phase 4 — Extension `ConvDQNConfig`

### Task 4 : 5 nouveaux champs eval + validation

**Files :**
- Modify : `mw_ia/config.py` (`ConvDQNConfig`)
- Test : `tests/test_conv_dqn_config.py`

- [ ] **Step 1 — Write the 3 failing tests**

Ajouter à la fin de `tests/test_conv_dqn_config.py` :

```python
def test_eval_defaults() -> None:
    """V2-V defaults : eval activé, 100 ép, seeds 10000-10009, max 200 steps."""
    cfg = ConvDQNConfig()
    assert cfg.eval_enabled is True
    assert cfg.eval_every_episodes == 100
    assert cfg.eval_seeds == tuple(range(10_000, 10_010))
    assert cfg.eval_max_steps == 200
    assert cfg.best_checkpoint_path is None


def test_eval_can_be_disabled() -> None:
    """eval_enabled=False pour reproduire baseline pre-V2-V."""
    cfg = ConvDQNConfig(eval_enabled=False)
    assert cfg.eval_enabled is False


def test_eval_validation() -> None:
    """eval_every_episodes ≤ 0 et eval_seeds vide rejetés."""
    with pytest.raises(ValueError):
        ConvDQNConfig(eval_every_episodes=0)
    with pytest.raises(ValueError):
        ConvDQNConfig(eval_every_episodes=-1)
    with pytest.raises(ValueError):
        ConvDQNConfig(eval_seeds=())
    with pytest.raises(ValueError):
        ConvDQNConfig(eval_max_steps=0)
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/test_conv_dqn_config.py::test_eval_defaults tests/test_conv_dqn_config.py::test_eval_can_be_disabled tests/test_conv_dqn_config.py::test_eval_validation -v 2>&1 | tail -10
```

Attendu : 3 fails avec `AttributeError` ou `TypeError` (champs eval n'existent pas).

- [ ] **Step 3 — Add fields to `ConvDQNConfig`**

Dans `mw_ia/config.py`, localiser la dataclass `ConvDQNConfig`. Ajouter les 5 champs eval juste après `double_dqn` (regrouper les params V2-W/V2-V) et avant `episodes`. Remplacer :

```python
    train_every: int = 4
    use_amp: bool = True
    double_dqn: bool = True   # V2-W : Hasselt 2015. False = V2-Z baseline DQN classique.
    episodes: int = 5_000
    max_steps_per_episode: int = 200
```

par :

```python
    train_every: int = 4
    use_amp: bool = True
    double_dqn: bool = True   # V2-W : Hasselt 2015. False = V2-Z baseline DQN classique.
    # V2-V : Training Protocol Stabilization (eval périodique + best-checkpoint)
    eval_enabled: bool = True
    eval_every_episodes: int = 100
    eval_seeds: tuple[int, ...] = tuple(range(10_000, 10_010))
    eval_max_steps: int = 200
    best_checkpoint_path: str | None = None
    episodes: int = 5_000
    max_steps_per_episode: int = 200
```

Puis ajouter à la fin de `__post_init__` (avant la dernière `raise`) :

```python
        if self.eval_every_episodes <= 0:
            raise ValueError(
                f"eval_every_episodes doit être > 0, reçu {self.eval_every_episodes}"
            )
        if len(self.eval_seeds) == 0:
            raise ValueError("eval_seeds ne peut pas être vide")
        if self.eval_max_steps <= 0:
            raise ValueError(f"eval_max_steps doit être > 0, reçu {self.eval_max_steps}")
```

- [ ] **Step 4 — Run tests, verify all pass**

```bash
source .venv/Scripts/activate && pytest tests/test_conv_dqn_config.py -v 2>&1 | tail -15
```

Attendu : 9 passed (6 V2-W existants + 3 nouveaux V2-V).

- [ ] **Step 5 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `228 passed` (225 + 3).

- [ ] **Step 6 — Commit**

```bash
git add mw_ia/config.py tests/test_conv_dqn_config.py
git commit -m "feat(v2-v): add 5 eval fields to ConvDQNConfig (eval_enabled, eval_every_episodes, eval_seeds, eval_max_steps, best_checkpoint_path)"
```

---

## Phase 5 — Intégration `ConvProceduralDQNRunner`

### Task 5 : Brancher evaluator + tracker dans le runner

**Files :**
- Modify : `mw_ia/training/runner.py` (`RunnerCallbacks`, `ConvProceduralDQNRunner`)
- Test : `tests/training/test_conv_procedural_runner.py`

- [ ] **Step 1 — Write the failing integration tests**

Ajouter à la fin de `tests/training/test_conv_procedural_runner.py` :

```python
def test_runner_eval_enabled_saves_best(tmp_path) -> None:
    """V2-V : runner avec eval activé sauvegarde best-checkpoint."""
    from mw_ia.training.runner import ConvProceduralDQNRunner

    env = _build_env()
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.0, max_density=0.20)
    best_path = tmp_path / "best.pt"
    dqn_cfg = ConvDQNConfig(
        episodes=200, max_steps_per_episode=30,
        batch_size=8, min_replay_to_learn=8, train_every=1,
        epsilon_decay_steps=200, target_sync_steps=50,
        replay_capacity=500, use_amp=False,
        eval_enabled=True,
        eval_every_episodes=50,
        eval_seeds=(10_000, 10_001, 10_002),
        eval_max_steps=30,
        best_checkpoint_path=str(best_path),
    )
    sched_cfg = SchedulerConfig(initial_difficulty=0.0)
    train_cfg = TrainingConfig(log_every_episodes=100)

    eval_call_count = [0]

    def on_eval(**kw: object) -> None:
        eval_call_count[0] += 1

    cb = RunnerCallbacks(on_eval=on_eval)
    runner = ConvProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=cb, device="cpu", seed=0,
    )
    runner.run()

    assert eval_call_count[0] >= 3, f"expected ≥3 evals, got {eval_call_count[0]}"
    assert best_path.exists(), "best_checkpoint .pt manquant sur disque"
    assert runner.best_tracker is not None
    assert runner.best_tracker.best_winrate >= 0.0


def test_runner_eval_disabled_no_evaluator(tmp_path) -> None:
    """V2-V : runner avec eval_enabled=False n'instancie pas l'evaluator."""
    from mw_ia.training.runner import ConvProceduralDQNRunner

    env = _build_env()
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.0, max_density=0.20)
    dqn_cfg = ConvDQNConfig(
        episodes=10, max_steps_per_episode=30,
        batch_size=8, min_replay_to_learn=8, train_every=1,
        epsilon_decay_steps=200, target_sync_steps=50,
        replay_capacity=500, use_amp=False,
        eval_enabled=False,
    )
    sched_cfg = SchedulerConfig(initial_difficulty=0.0)
    train_cfg = TrainingConfig(log_every_episodes=100)

    runner = ConvProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=RunnerCallbacks(), device="cpu", seed=0,
    )
    assert runner.evaluator is None
    assert runner.best_tracker is None
    runner.run()
```

- [ ] **Step 2 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/training/test_conv_procedural_runner.py::test_runner_eval_enabled_saves_best tests/training/test_conv_procedural_runner.py::test_runner_eval_disabled_no_evaluator -v 2>&1 | tail -15
```

Attendu : fails — soit `AttributeError: on_eval` (callback manquant), soit `AttributeError: runner.evaluator`, soit `runner.best_tracker`.

- [ ] **Step 3 — Add `on_eval` callback + `fire_eval` method**

Dans `mw_ia/training/runner.py`, modifier `RunnerCallbacks`. Localiser la dataclass et ajouter le champ `on_eval`. Remplacer :

```python
@dataclass
class RunnerCallbacks:
    on_step: StepCb | None = None
    on_episode: EpisodeCb | None = None
    on_loss: LossCb | None = None
    on_epsilon: EpsilonCb | None = None
    on_log: LogCb | None = None
    on_maze_changed: Callable[..., None] | None = None
    on_difficulty_updated: Callable[..., None] | None = None
```

par :

```python
@dataclass
class RunnerCallbacks:
    on_step: StepCb | None = None
    on_episode: EpisodeCb | None = None
    on_loss: LossCb | None = None
    on_epsilon: EpsilonCb | None = None
    on_log: LogCb | None = None
    on_maze_changed: Callable[..., None] | None = None
    on_difficulty_updated: Callable[..., None] | None = None
    on_eval: Callable[..., None] | None = None
```

Et ajouter la méthode `fire_eval` après `fire_difficulty_updated` :

```python
    def fire_eval(self, **kw: object) -> None:
        if self.on_eval:
            self.on_eval(**kw)
```

- [ ] **Step 4 — Add imports for evaluator + tracker**

En haut de `mw_ia/training/runner.py`, à côté des imports existants `from mw_ia.training.metrics import ...` et `from mw_ia.training.scheduler import ...`, ajouter :

```python
from mw_ia.training.checkpoint_tracker import BestCheckpointTracker
from mw_ia.training.evaluator import PeriodicEvaluator
```

- [ ] **Step 5 — Modify `ConvProceduralDQNRunner.__init__`**

Dans `mw_ia/training/runner.py`, localiser `ConvProceduralDQNRunner.__init__`. À la fin du `__init__` (après la construction de `self.agent`), ajouter le bloc V2-V :

```python
        # V2-V : eval périodique + best-checkpoint
        if dqn_cfg.eval_enabled:
            # Eval env séparé avec MÊME proc_cfg que training mais générateur fresh
            eval_gen = type(env.generator).__new__(type(env.generator))
            eval_gen.__dict__.update(env.generator.__dict__)
            eval_env = ProceduralGridWorld(cfg=proc_cfg, generator=eval_gen)
            self.evaluator: PeriodicEvaluator | None = PeriodicEvaluator(
                eval_env=eval_env,
                eval_seeds=dqn_cfg.eval_seeds,
                max_steps=dqn_cfg.eval_max_steps,
                observation_encoder=encode_procedural_observation_2d,
                proc_cfg=proc_cfg,
            )
            self.best_tracker: BestCheckpointTracker | None = BestCheckpointTracker(
                path=dqn_cfg.best_checkpoint_path,
            )
        else:
            self.evaluator = None
            self.best_tracker = None
```

- [ ] **Step 6 — Modify `ConvProceduralDQNRunner.run()` to call evaluation periodically**

Localiser la fin de la boucle `for ep in range(self.dqn_cfg.episodes):` dans `ConvProceduralDQNRunner.run()`. Juste après le bloc `if ep % self.train_cfg.log_every_episodes == 0:` (qui fire le log d'épisode), ajouter le bloc V2-V :

```python
            # V2-V : eval périodique + best-checkpoint
            if (
                self.evaluator is not None
                and (ep + 1) % self.dqn_cfg.eval_every_episodes == 0
            ):
                eval_metrics = self.evaluator.evaluate(self.agent, self.scheduler.current)
                improved = self.best_tracker.update(eval_metrics, self.agent, episode=ep)
                self.callbacks.fire_eval(
                    ep=ep,
                    eval_winrate=eval_metrics["winrate"],
                    eval_diff=eval_metrics["difficulty"],
                    best_winrate=self.best_tracker.best_winrate,
                    best_episode=self.best_tracker.best_episode,
                    improved=improved,
                )
                self.callbacks.fire_log(
                    "info",
                    f"eval ep {ep:>4} : winrate={eval_metrics['winrate']:.2%} "
                    f"@ diff={eval_metrics['difficulty']:.2f}  "
                    f"best={self.best_tracker.best_winrate:.2%} "
                    f"@ ep {self.best_tracker.best_episode}"
                    + ("  NEW BEST" if improved else "")
                )
```

- [ ] **Step 7 — Run scoped tests, verify they pass**

```bash
source .venv/Scripts/activate && pytest tests/training/test_conv_procedural_runner.py -v 2>&1 | tail -15
```

Attendu : 5 passed (3 V2-Z existants + 2 nouveaux V2-V).

- [ ] **Step 8 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `230 passed` (228 + 2). Note : 1 test de plus que prévu dans le plan original (229) car ajout du test `test_runner_eval_disabled_no_evaluator` pour couvrir la branche `else`.

- [ ] **Step 9 — Commit**

```bash
git add mw_ia/training/runner.py tests/training/test_conv_procedural_runner.py
git commit -m "feat(v2-v): integrate PeriodicEvaluator + BestCheckpointTracker in ConvProceduralDQNRunner"
```

---

## Phase 6 — CLI flags

### Task 6 : Exposer eval + best-checkpoint via CLI

**Files :**
- Modify : `scripts/train_cnn_dqn_procedural.py`

- [ ] **Step 1 — Add 3 CLI flags to argparse**

Dans `scripts/train_cnn_dqn_procedural.py`, localiser la section argparse dans `main()`. Ajouter les 3 flags V2-V juste après le flag `--double-dqn` :

```python
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Évaluation périodique greedy sur seeds eval séparés (V2-V). "
             "Default activé. Utiliser --no-eval pour reproduire baseline pre-V2-V.",
    )
    parser.add_argument(
        "--eval-every-episodes",
        type=int,
        default=100,
        help="Périodicité (ép) de l'évaluation greedy (default V2-V : 100).",
    )
    parser.add_argument(
        "--best-checkpoint-path",
        type=str,
        default=None,
        help="Chemin .pt du best-checkpoint (default None = pas de sauvegarde disque). "
             "Suggestion : checkpoints/v2v_best_seed{N}.pt",
    )
```

- [ ] **Step 2 — Pass the flags to `ConvDQNConfig`**

Localiser la construction de `ConvDQNConfig`. Ajouter les 3 nouveaux args :

```python
    dqn_cfg = ConvDQNConfig(
        episodes=args.episodes,
        conv_channels=tuple(args.conv_channels),
        fc_hidden=args.fc_hidden,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_sync_steps=args.target_sync_steps,
        double_dqn=args.double_dqn,
        eval_enabled=args.eval,
        eval_every_episodes=args.eval_every_episodes,
        best_checkpoint_path=args.best_checkpoint_path,
    )
```

- [ ] **Step 3 — Smoke test V2-V CLI (avec eval activé + best-checkpoint)**

```bash
mkdir -p checkpoints && \
source .venv/Scripts/activate && \
python scripts/train_cnn_dqn_procedural.py --episodes 20 --mode obstacles --device cpu \
    --eval-every-episodes 10 \
    --best-checkpoint-path checkpoints/v2v_smoke_test.pt 2>&1 | tail -15
```

Attendu :
- Pas de crash
- Au moins 1 ligne `eval ep XXX : winrate=...` dans la sortie
- Fichier `checkpoints/v2v_smoke_test.pt` créé sur disque

Vérification finale :

```bash
ls -la checkpoints/v2v_smoke_test.pt && rm checkpoints/v2v_smoke_test.pt
```

Attendu : fichier listé (~7 MB), puis supprimé proprement.

- [ ] **Step 4 — Smoke V2-V désactivé (repro baseline pre-V2-V)**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_dqn_procedural.py --episodes 20 --mode obstacles --device cpu --no-eval 2>&1 | tail -10
```

Attendu : pas de crash, aucune ligne `eval ep` dans la sortie.

- [ ] **Step 5 — Verify --help shows the new flags**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_dqn_procedural.py --help 2>&1 | grep -E "eval|checkpoint"
```

Attendu : output contenant `--eval`, `--no-eval`, `--eval-every-episodes`, `--best-checkpoint-path`.

- [ ] **Step 6 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `230 passed`.

- [ ] **Step 7 — Commit**

```bash
git add scripts/train_cnn_dqn_procedural.py
git commit -m "feat(v2-v): add --eval / --no-eval / --eval-every-episodes / --best-checkpoint-path CLI flags"
```

---

## Phase 7 — CI smoke + README + CLAUDE.md + smoke E2E GPU + tag

### Task 7 : CI workflow smoke V2-V

**Files :**
- Modify : `.github/workflows/aether_verify.yml`

- [ ] **Step 1 — Add V2-V smoke step**

Localiser le step "Smoke test conv procedural training" dans `.github/workflows/aether_verify.yml`. Ajouter un nouveau step juste après (pattern identique) :

```yaml
      - name: Smoke test V2-V eval + best-checkpoint
        run: |
          mkdir -p checkpoints
          python scripts/train_cnn_dqn_procedural.py --episodes 10 --mode obstacles --device cpu --eval-every-episodes 5 --best-checkpoint-path checkpoints/ci_v2v_best.pt
          test -f checkpoints/ci_v2v_best.pt
```

- [ ] **Step 2 — Validate YAML syntax**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/aether_verify.yml'))" && echo "YAML OK"
```

Attendu : `YAML OK`.

- [ ] **Step 3 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `230 passed`.

- [ ] **Step 4 — Commit**

```bash
git add .github/workflows/aether_verify.yml
git commit -m "ci(v2-v): add smoke test V2-V eval + best-checkpoint to aether_verify.yml"
```

---

### Task 8 : README + CLAUDE.md V2-V section + smoke E2E GPU + tag

**Files :**
- Modify : `README.md`
- Modify : `CLAUDE.md`

- [ ] **Step 1 — Smoke E2E manuel GPU 500 ép V2-V**

```bash
mkdir -p checkpoints && \
source .venv/Scripts/activate && \
python scripts/train_cnn_dqn_procedural.py --episodes 500 --mode obstacles --device cuda \
    --eval-every-episodes 100 \
    --best-checkpoint-path checkpoints/v2v_smoke_gpu.pt 2>&1 | tail -15
```

Attendu :
- Pas de crash sur 500 ép GPU
- ≥ 5 lignes `eval ep` dans la sortie
- Fichier `checkpoints/v2v_smoke_gpu.pt` créé

Vérification :

```bash
ls -la checkpoints/v2v_smoke_gpu.pt && rm checkpoints/v2v_smoke_gpu.pt
```

- [ ] **Step 2 — Aether re-verify**

```bash
bash aether/verify_all.sh
```

Attendu : `8 OK`.

- [ ] **Step 3 — Add V2-V section to README.md**

Localiser la section `## V2-W` dans `README.md`. Insérer une nouvelle section V2-V juste après la fin de la section V2-W et avant `## Roadmap (V2+)`.

Contenu exact à insérer (préserver les triples-backticks) :

````markdown
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
````

- [ ] **Step 4 — Add V2-V row to CLAUDE.md sub-projects table**

Dans `CLAUDE.md`, localiser le tableau "Sous-projets — décomposition". Ajouter une ligne V après la ligne W. Remplacer :

```markdown
| **W** | Double DQN sur ConvDQN (roadmap #7) | ✅ Livré (tag `v0.2.0-w`) |
| **B** | Mémoire persistante cross-session | ⏳ **Prochain par défaut** |
```

par :

```markdown
| **W** | Double DQN sur ConvDQN (roadmap #7) | ✅ Livré (tag `v0.2.0-w`) |
| **V** | Training Protocol Stabilization (eval + best-checkpoint) | ✅ Livré (tag `v0.2.0-v`) |
| **B** | Mémoire persistante cross-session | ⏳ **Prochain par défaut** |
```

- [ ] **Step 5 — Add V2-V phases section to CLAUDE.md**

Dans `CLAUDE.md`, localiser la fin de la section "V2-W — H1 confirmée : best-before-collapse" (chercher la ligne "**Sous-projet V2-V ... devient la priorité absolue**"). Insérer juste après une nouvelle section :

```markdown
### V2-V — état final des phases (livraison 2026-05-23)

| Phase | Tâches | Statut | Tests | Commits |
|---|---|---|---|---|
| 1 — Setup scaffold | T1 | ✅ | 0 | 1 |
| 2 — `PeriodicEvaluator` | T2 | ✅ | 8 | 1 |
| 3 — `BestCheckpointTracker` | T3 | ✅ | 6 | 1 |
| 4 — `ConvDQNConfig` extension (5 champs eval) | T4 | ✅ | 3 | 1 |
| 5 — `ConvProceduralDQNRunner` intégration + `on_eval` callback | T5 | ✅ | 2 | 1 |
| 6 — CLI flags `--eval / --eval-every-episodes / --best-checkpoint-path` | T6 | ✅ | 0 | 1 |
| 7 — CI smoke + README + CLAUDE.md + tag `v0.2.0-v` | T7-T8 | ✅ | 0 | 2 + tag |

### Composants V2-V livrés

| Composant | Fichier | Rôle |
|---|---|---|
| `PeriodicEvaluator` | `mw_ia/training/evaluator.py` | Greedy eval sur env eval séparé. Méthode `evaluate(agent, difficulty)` retourne dict `{winrate, mean_reward, mean_length, n_episodes, difficulty}`. Zéro pollution training. |
| `BestCheckpointTracker` | `mw_ia/training/checkpoint_tracker.py` | Sauvegarde auto du modèle au pic eval_winrate. Idempotent (égalité ne triggers pas save). `path=None` = tracking en mémoire. |
| `ConvDQNConfig` extension | `mw_ia/config.py` | + 5 champs : `eval_enabled`, `eval_every_episodes`, `eval_seeds`, `eval_max_steps`, `best_checkpoint_path`. Defaults V2-V activé. |
| `ConvProceduralDQNRunner` extension | `mw_ia/training/runner.py` | Instancie evaluator + tracker si `eval_enabled`. Appelle eval tous les `eval_every_episodes`. `RunnerCallbacks.on_eval` ajouté. |
| CLI flags | `scripts/train_cnn_dqn_procedural.py` | `--eval / --no-eval`, `--eval-every-episodes`, `--best-checkpoint-path`. |

### Décisions techniques V2-V

- **Méthode `evaluate()` au lieu de `eval`** : nommage explicite, évite collision avec builtin Python (lisible et sans ambigüité).
- **Eval seeds 10000-10009 hors-training** : vraie mesure de généralisation. Training utilise seeds 0..episodes-1.
- **`agent.act(obs, greedy=True)`** : bypass de l'eps-greedy ET du rng training (vérifié dans code V2-Z : `if (not greedy) and self._rng.random() < ...`).
- **Eval env construit avec générateur fresh** : `eval_gen = type(env.generator).__new__(type(env.generator)); eval_gen.__dict__.update(env.generator.__dict__)`. Évite le partage du rng generator entre training et eval.
- **`best_checkpoint_path=None` par défaut** : tracking en mémoire sans IO disque. L'utilisateur DOIT passer un chemin pour persister le best-model.
- **Pas de modification de l'agent** : `act(greedy=True)` et `save()` existaient déjà en V1. V2-V est pur orchestrateur externe.

### V2-V — pièges connus

1. **Eval env partage le rng generator si on passe l'instance training** : utiliser `__new__` + `__dict__.update` pour cloner sans partage de state. Documenter en commentaire.
2. **`tmp_path` fixture pytest sur Windows : chemins avec espaces** : utiliser `pathlib.Path` partout, pas de string concat. PyTorch accepte `Path` en argument de `torch.save`.
3. **Best-checkpoint écrasé entre runs** si même `--best-checkpoint-path` : suggérer le pattern `checkpoints/v2v_best_seed{N}.pt` dans le help CLI pour éviter collision.
4. **`eval_seeds=10000-10009` peut chevaucher training si `--episodes >> 10000`** : edge case documenté, hors-scope MVP (defaults 5000 ép → jamais touchés).
5. **Eval à la diff scheduler.current uniquement** : MVP. Future extension multi-diff caractérisation curve.
```

- [ ] **Step 6 — Update "Prochaines étapes prioritaires" section**

Dans `CLAUDE.md`, localiser la section "Prochaines étapes prioritaires (post H1 confirmée 2026-05-23)". Remplacer le contenu par :

```markdown
**Prochaines étapes prioritaires (post V2-V livré 2026-05-23)** :

1. ✅ **V2-V Training Protocol Stabilization** — **LIVRÉ** (tag `v0.2.0-v`) : eval périodique greedy + best-checkpoint tracking.

2. **Re-benchmark V2-W n=5 ep=5000 AVEC V2-V activé** — validation scientifique non-bloquante :
   - Lancer 5 runs V2-W ep=5000 avec `--best-checkpoint-path checkpoints/v2v_w_best_seed{N}.pt`
   - Comparer winrate final vs best-checkpoint winrate par seed
   - Cible : seed 4 best-checkpoint ≥ 60 % (vs final 1 %) → V2-V valide son utilité

3. **Sous-projets V3+ déblocables maintenant** :
   - **V2-ZY CNN+LSTM+Double DQN** : viable car best-checkpoint protège du collapse
   - **Soft target Polyak τ=0.005** : tests propres car best-checkpoint isole l'effet vrai du timing
   - **Mazes larges (max_size=15/20)** : eval permet de tracker généralisation
   - **V2-V étendu** : early stopping + rollback + MA metrics + brancher sur V2-X/V2-Y runners
   - **Sous-projet B (mémoire persistante cross-session)** : best-checkpoint est la fondation
```

- [ ] **Step 7 — Run full suite + Aether final**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
bash aether/verify_all.sh
```

Attendu : `230 passed` + `8 OK`.

- [ ] **Step 8 — Commit doc**

```bash
git add README.md CLAUDE.md
git commit -m "docs(v2-v): add V2-V section (Training Protocol Stabilization) to README + CLAUDE.md"
```

- [ ] **Step 9 — Tag**

```bash
git tag v0.2.0-v
git tag --list | tail -7
```

Attendu : `v0.1.0`, `v0.2.0-a`, `v0.2.0-x`, `v0.2.0-y`, `v0.2.0-z`, `v0.2.0-w`, `v0.2.0-v`.

- [ ] **Step 10 — DoD final récap**

Print to stdout :

```
=== V2-V DoD CHECKLIST (livraison code) ===
[ ] pytest -q → 230 passed (211 + 19)
[ ] bash aether/verify_all.sh → 8 OK
[ ] smoke train_cnn_dqn_procedural.py --episodes 500 --device cuda --best-checkpoint-path ... OK
[ ] best_checkpoint .pt présent sur disque (~7 MB)
[ ] V2-V section dans README.md + CLAUDE.md
[ ] Tag v0.2.0-v posé
[ ] Tags antérieurs (v0.1.0, v0.2.0-a/x/y/z/w) intacts
==> Phase post-livraison : re-benchmark V2-W n=5 ep=5000 + V2-V
    pour confirmer que seed 4 best-checkpoint capture le pic ~71% winrate.
```

---

## Récapitulatif

- **8 tasks** réparties sur **7 phases**
- **19 nouveaux tests** (211 → 230, 1 de plus que prévu en spec — test bonus `test_runner_eval_disabled_no_evaluator`)
- **~9 commits** sur `main`
- **Tag livraison** : `v0.2.0-v`
- **2 nouveaux fichiers code** : `mw_ia/training/evaluator.py`, `mw_ia/training/checkpoint_tracker.py`
- **4 fichiers code modifiés** : `config.py`, `runner.py`, `scripts/train_cnn_dqn_procedural.py`, `aether_verify.yml`
- **2 nouveaux fichiers tests** : `tests/training/test_evaluator.py`, `tests/training/test_checkpoint_tracker.py`
- **DoD bloquante** : pytest 230 + Aether 8 OK + smoke E2E GPU + best.pt sur disque + tag
- **DoD non-bloquante (objectif scientifique)** : re-benchmark V2-W n=5 avec V2-V activé. Cible : seed 4 best-checkpoint ≥ 60 %.
