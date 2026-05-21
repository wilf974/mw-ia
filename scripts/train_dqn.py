"""Entraîne DQN sur GridWorld, sans GUI."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mw_ia.config import DEFAULT, DQNConfig
from mw_ia.envs.gridworld import GridWorld
from mw_ia.persistence.checkpoint import dump_metrics
from mw_ia.training.runner import DQNRunner, RunnerCallbacks


def main() -> int:
    parser = argparse.ArgumentParser(description="DQN headless")
    parser.add_argument("--episodes", type=int, default=DEFAULT.dqn.episodes)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=DEFAULT.dqn.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT.dqn.lr)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dqn.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("logs/dqn_metrics.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT.training.seed)
    args = parser.parse_args()

    env = GridWorld(DEFAULT.gridworld)
    dcfg = DQNConfig(
        hidden_layers=DEFAULT.dqn.hidden_layers,
        batch_size=args.batch,
        lr=args.lr,
        gamma=DEFAULT.dqn.gamma,
        epsilon_start=DEFAULT.dqn.epsilon_start,
        epsilon_end=DEFAULT.dqn.epsilon_end,
        epsilon_decay_steps=DEFAULT.dqn.epsilon_decay_steps,
        replay_capacity=DEFAULT.dqn.replay_capacity,
        min_replay_to_learn=DEFAULT.dqn.min_replay_to_learn,
        target_sync_steps=DEFAULT.dqn.target_sync_steps,
        train_every=DEFAULT.dqn.train_every,
        use_amp=(not args.no_amp) and args.device == "cuda",
        episodes=args.episodes,
        max_steps_per_episode=DEFAULT.dqn.max_steps_per_episode,
    )
    callbacks = RunnerCallbacks(on_log=lambda lvl, msg: print(f"[{lvl}] {msg}"))
    runner = DQNRunner(env, dcfg, DEFAULT.training, callbacks, device=args.device, seed=args.seed)
    runner.run()
    runner.agent.save(args.checkpoint)
    dump_metrics(
        args.metrics,
        {
            "winrate": runner.metrics.winrate(),
            "best_reward": runner.metrics.best_reward,
            "episodes": runner.metrics.total_episodes,
            "level": runner.metrics.level().value,
            "device": str(runner.agent.device),
        },
    )
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Métriques  : {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
