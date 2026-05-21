"""Entraîne Q-Learning tabulaire sur GridWorld, sans GUI."""
from __future__ import annotations

import argparse
from pathlib import Path

from mw_ia.config import DEFAULT, QLearningConfig
from mw_ia.envs.gridworld import GridWorld
from mw_ia.persistence.checkpoint import dump_metrics
from mw_ia.training.runner import RunnerCallbacks, TabularRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Q-Learning tabulaire headless")
    parser.add_argument("--episodes", type=int, default=DEFAULT.qlearning.episodes)
    parser.add_argument("--alpha", type=float, default=DEFAULT.qlearning.alpha)
    parser.add_argument("--gamma", type=float, default=DEFAULT.qlearning.gamma)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/qtable"))
    parser.add_argument("--metrics", type=Path, default=Path("logs/qtable_metrics.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT.training.seed)
    args = parser.parse_args()

    env = GridWorld(DEFAULT.gridworld)
    qcfg = QLearningConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=DEFAULT.qlearning.epsilon_start,
        epsilon_end=DEFAULT.qlearning.epsilon_end,
        epsilon_decay_episodes=max(1, args.episodes // 2),
        episodes=args.episodes,
    )
    callbacks = RunnerCallbacks(on_log=lambda lvl, msg: print(f"[{lvl}] {msg}"))
    runner = TabularRunner(env, qcfg, DEFAULT.training, callbacks, seed=args.seed)
    runner.run()
    runner.agent.save(args.checkpoint)
    dump_metrics(
        args.metrics,
        {
            "winrate": runner.metrics.winrate(),
            "best_reward": runner.metrics.best_reward,
            "episodes": runner.metrics.total_episodes,
            "level": runner.metrics.level().value,
        },
    )
    print(f"Checkpoint : {args.checkpoint}.npz")
    print(f"Métriques  : {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
