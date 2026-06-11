"""Tests V2-Z d'intégration de ConvProceduralDQNRunner."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

import pytest

pytest.importorskip("torch", reason="suite complete : requiert torch")

from mw_ia.config import (
    ConvDQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ConvProceduralDQNRunner, RunnerCallbacks


def _build_env() -> ProceduralGridWorld:
    cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                              min_density=0.0, max_density=0.20)
    gen = RandomObstaclesGenerator(
        rows=10, cols=10, start=(0, 0), goal=(9, 9),
        min_density=cfg.min_density, max_density=cfg.max_density,
    )
    return ProceduralGridWorld(cfg=cfg, generator=gen)


def _build_runner(
    *, episodes: int, callbacks: RunnerCallbacks | None = None,
) -> ConvProceduralDQNRunner:
    env = _build_env()
    proc_cfg = ProceduralEnvConfig(mode="obstacles", max_rows=10, max_cols=10,
                                   min_density=0.0, max_density=0.20)
    dqn_cfg = ConvDQNConfig(
        episodes=episodes, max_steps_per_episode=30,
        batch_size=8, min_replay_to_learn=8, train_every=1,
        epsilon_decay_steps=200, target_sync_steps=50,
        replay_capacity=500, use_amp=False,
    )
    sched_cfg = SchedulerConfig(initial_difficulty=0.0)
    train_cfg = TrainingConfig(log_every_episodes=100)
    return ConvProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=callbacks or RunnerCallbacks(),
        device="cpu", seed=0,
    )


def test_single_episode_runs() -> None:
    """1 épisode sans crash, metrics récoltés."""
    runner = _build_runner(episodes=1)
    runner.run()
    assert len(runner.metrics.episode_rewards) == 1


def test_callbacks_fired() -> None:
    """on_maze_changed, on_episode, on_step appelés au moins une fois."""
    counts: dict[str, int] = {"maze": 0, "ep": 0, "step": 0}

    def on_maze(**kw: Any) -> None:
        counts["maze"] += 1

    def on_ep(**kw: Any) -> None:
        counts["ep"] += 1

    def on_step(**kw: Any) -> None:
        counts["step"] += 1

    cb = RunnerCallbacks(on_maze_changed=on_maze, on_episode=on_ep, on_step=on_step)
    runner = _build_runner(episodes=2, callbacks=cb)
    runner.run()
    assert counts["maze"] >= 2
    assert counts["ep"] == 2
    assert counts["step"] >= 1


def test_smoke_10_episodes_no_nan() -> None:
    """10 épisodes : metrics.losses tous finis, winrate ∈ [0, 1]."""
    runner = _build_runner(episodes=10)
    runner.run()
    losses = runner.metrics.losses
    assert all(math.isfinite(l) for l in losses), f"loss non-finite : {losses}"
    wr = runner.metrics.winrate()
    assert 0.0 <= wr <= 1.0
