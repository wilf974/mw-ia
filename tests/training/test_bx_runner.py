"""Tests d'integration V2-BX runner : wiring oracle + nouveaute + logging."""
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
    runner.run()  # ne doit pas lever (obs 4-canaux coherentes act/observe)


def test_novelty_bonus_runs_end_to_end():
    runner = _make_runner(bx_novelty_beta=0.1)
    runner.run()  # reward shaping ne casse pas la boucle
