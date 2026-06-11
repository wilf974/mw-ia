"""Tests d'intégration de ProceduralDQNRunner."""
from __future__ import annotations

import numpy as np

import pytest

pytest.importorskip("torch", reason="suite complete : requiert torch")

from mw_ia.config import (
    DQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ProceduralDQNRunner, RunnerCallbacks


def _make_runner(episodes: int = 20) -> ProceduralDQNRunner:
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.10, max_density=0.30)
    gen = RandomObstaclesGenerator(rows=10, cols=10, start=(0, 0), goal=(9, 9),
                                   min_density=proc_cfg.min_density,
                                   max_density=proc_cfg.max_density)
    env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
    dqn_cfg = DQNConfig(episodes=episodes, min_replay_to_learn=32,
                        max_steps_per_episode=30, use_amp=False,
                        replay_capacity=1_000, batch_size=32)
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
