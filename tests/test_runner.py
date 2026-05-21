"""Tests TrainingRunner via callbacks (sans Qt)."""
from __future__ import annotations

from mw_ia.config import DQNConfig, GridWorldConfig, QLearningConfig, TrainingConfig
from mw_ia.envs.gridworld import GridWorld
from mw_ia.training.runner import DQNRunner, RunnerCallbacks, TabularRunner


def _make_env() -> GridWorld:
    return GridWorld(GridWorldConfig(rows=4, cols=4, start=(0, 0), goal=(3, 3),
                                     obstacles=(), max_steps=30))


def test_tabular_runner_emits_events() -> None:
    env = _make_env()
    cfg_q = QLearningConfig(alpha=0.5, gamma=0.9, epsilon_start=1.0, epsilon_end=0.05,
                            epsilon_decay_episodes=20, episodes=30)
    events: list[tuple[str, dict]] = []
    cb = RunnerCallbacks(
        on_step=lambda **kw: events.append(("step", kw)),
        on_episode=lambda **kw: events.append(("episode", kw)),
        on_log=lambda level, msg: events.append(("log", {"level": level, "msg": msg})),
    )
    runner = TabularRunner(env, cfg_q, TrainingConfig(), callbacks=cb, seed=0)
    runner.run()
    eps_count = sum(1 for e in events if e[0] == "episode")
    assert eps_count == cfg_q.episodes


def test_runner_supports_stop() -> None:
    env = _make_env()
    cfg_q = QLearningConfig(episodes=200, epsilon_decay_episodes=200)
    runner = TabularRunner(env, cfg_q, TrainingConfig(), callbacks=RunnerCallbacks(), seed=0)
    runner.request_stop()
    runner.run()
    assert runner.metrics.total_episodes == 0


def test_dqn_runner_runs_short_session() -> None:
    env = _make_env()
    cfg = DQNConfig(hidden_layers=(16,), batch_size=8, replay_capacity=200,
                    min_replay_to_learn=8, target_sync_steps=20,
                    epsilon_decay_steps=100, use_amp=False, episodes=5,
                    max_steps_per_episode=30)
    losses: list[float] = []
    cb = RunnerCallbacks(on_loss=lambda step, loss: losses.append(loss))
    runner = DQNRunner(env, cfg, TrainingConfig(), callbacks=cb, device="cpu", seed=0)
    runner.run()
    assert runner.metrics.total_episodes == cfg.episodes
    assert len(losses) > 0
