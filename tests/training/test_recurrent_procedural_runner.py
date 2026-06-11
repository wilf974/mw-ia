"""Tests d'intégration de RecurrentProceduralDQNRunner."""
from __future__ import annotations

import numpy as np

import pytest

pytest.importorskip("torch", reason="suite complete : requiert torch")

from mw_ia.config import (
    DRQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import RecurrentProceduralDQNRunner, RunnerCallbacks


def _make_runner(episodes: int = 10) -> RecurrentProceduralDQNRunner:
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.0, max_density=0.10)
    gen = RandomObstaclesGenerator(rows=10, cols=10, start=(0, 0), goal=(9, 9),
                                   min_density=proc_cfg.min_density,
                                   max_density=proc_cfg.max_density)
    env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
    drqn_cfg = DRQNConfig(episodes=episodes, replay_capacity=20,
                          min_episodes_to_learn=3, batch_size=4,
                          sequence_length=8, max_steps_per_episode=30,
                          use_amp=False, epsilon_decay_steps=200,
                          target_sync_steps=100)
    sched_cfg = SchedulerConfig(initial_difficulty=0.0, step=0.05, update_interval=5)
    train_cfg = TrainingConfig()
    return RecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, drqn_cfg=drqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=RunnerCallbacks(),
        device="cpu", seed=0,
    )


def test_recurrent_runner_10_episodes_no_error():
    runner = _make_runner(episodes=10)
    runner.run()
    assert runner.metrics.total_episodes == 10


def test_recurrent_runner_emits_maze_changed_callback():
    captured: list[tuple[np.ndarray, int, float]] = []
    cb = RunnerCallbacks(on_maze_changed=lambda **kw: captured.append(
        (kw["maze"], kw["episode_id"], kw["difficulty"])
    ))
    runner = _make_runner(episodes=5)
    runner.callbacks = cb
    runner.run()
    assert len(captured) == 5
