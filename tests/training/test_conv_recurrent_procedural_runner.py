"""Tests V2-ZY d'intégration de ConvRecurrentProceduralDQNRunner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch", reason="suite complete : requiert torch")

from mw_ia.config import (
    ConvRecurrentDQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ConvRecurrentProceduralDQNRunner, RunnerCallbacks


def _build_env() -> ProceduralGridWorld:
    cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                              min_density=0.0, max_density=0.20)
    gen = RandomObstaclesGenerator(
        rows=10, cols=10, start=(0, 0), goal=(9, 9),
        min_density=cfg.min_density, max_density=cfg.max_density,
    )
    return ProceduralGridWorld(cfg=cfg, generator=gen)


def test_runner_full_episode_with_eval_and_best_checkpoint(tmp_path: Path) -> None:
    """V2-ZY runner avec V2-V activé sauvegarde best-checkpoint."""
    env = _build_env()
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.0, max_density=0.20)
    best_path = tmp_path / "best.pt"
    dqn_cfg = ConvRecurrentDQNConfig(
        episodes=200, max_steps_per_episode=20,
        batch_size=2, min_episodes_to_learn=2, train_steps_per_episode=1,
        sequence_length=8, replay_capacity=20,
        epsilon_decay_steps=200, target_sync_steps=50,
        use_amp=False,
        eval_enabled=True,
        eval_every_episodes=50,
        eval_seeds=(10_000, 10_001, 10_002),
        eval_max_steps=10,
        eval_target_difficulty=0.10,
        best_checkpoint_path=str(best_path),
    )
    sched_cfg = SchedulerConfig(initial_difficulty=0.0)
    train_cfg = TrainingConfig(log_every_episodes=100)

    eval_count = [0]

    def evaluation_callback(**kw: Any) -> None:
        eval_count[0] += 1

    cb = RunnerCallbacks(on_eval=evaluation_callback)
    runner = ConvRecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=cb, device="cpu", seed=0,
    )
    runner.run()

    assert eval_count[0] >= 3, f"expected >=3 evals, got {eval_count[0]}"
    assert best_path.exists(), "best_checkpoint .pt manquant sur disque"
    assert runner.best_tracker is not None
    assert runner.best_tracker.best_winrate >= 0.0


def test_runner_eval_disabled_no_evaluator() -> None:
    """V2-ZY runner avec eval_enabled=False n'instancie pas evaluator."""
    env = _build_env()
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.0, max_density=0.20)
    dqn_cfg = ConvRecurrentDQNConfig(
        episodes=10, max_steps_per_episode=10,
        batch_size=2, min_episodes_to_learn=2, train_steps_per_episode=1,
        sequence_length=4, replay_capacity=10,
        epsilon_decay_steps=200, target_sync_steps=50,
        use_amp=False,
        eval_enabled=False,
    )
    sched_cfg = SchedulerConfig(initial_difficulty=0.0)
    train_cfg = TrainingConfig(log_every_episodes=100)

    runner = ConvRecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=RunnerCallbacks(), device="cpu", seed=0,
    )
    assert runner.evaluator is None
    assert runner.best_tracker is None
    runner.run()
