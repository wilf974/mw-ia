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
    bx_line = next(m for m in logs if "BX_PROBE_RESULT" in m)
    assert "diff_max=" in bx_line
    assert "ep_to_diff_0.30=" in bx_line
    assert "best_eval_0.30=" in bx_line
