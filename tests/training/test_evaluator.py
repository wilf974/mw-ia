"""Tests V2-V de PeriodicEvaluator (greedy eval sans pollution training)."""
from __future__ import annotations

import numpy as np

import pytest

pytest.importorskip("torch", reason="suite complete : requiert torch")

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
