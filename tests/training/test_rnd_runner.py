"""Tests d'integration V2-C0 RND runner."""
from __future__ import annotations

from mw_ia.config import (
    ConvRecurrentDQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ConvRecurrentProceduralDQNRunner, RunnerCallbacks


def _make_runner(on_log=None, **cfg_overrides):
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
    cb = RunnerCallbacks(on_log=on_log)
    return ConvRecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg,
        sched_cfg=SchedulerConfig(update_interval=1, step=0.05),
        train_cfg=TrainingConfig(), callbacks=cb, device="cpu", seed=0,
    )


def test_rnd_disabled_no_module():
    runner = _make_runner(rnd_enabled=False)
    assert runner.rnd is None


def test_rnd_enabled_instantiates_module():
    runner = _make_runner(rnd_enabled=True, rnd_warmup_steps=0)
    assert runner.rnd is not None


def test_rnd_run_end_to_end_and_logs_result():
    logs: list[str] = []
    runner = _make_runner(
        on_log=lambda level, msg: logs.append(msg),
        rnd_enabled=True, rnd_warmup_steps=0, rnd_beta=0.2,
    )
    runner.run()
    assert any("RND_RESULT" in m and "ratio_int_ext=" in m for m in logs)


def test_rnd_disabled_emits_no_rnd_result():
    logs: list[str] = []
    runner = _make_runner(on_log=lambda level, msg: logs.append(msg), rnd_enabled=False)
    runner.run()
    assert not any("RND_RESULT" in m for m in logs)
