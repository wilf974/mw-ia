"""Entraînement Recurrent DQN (DRQN/LSTM) procedural headless (CLI).

Usage :
    python scripts/train_drqn_procedural.py --episodes 5000 --mode obstacles --device cuda
"""
from __future__ import annotations

import argparse
import sys

from mw_ia.config import (
    DRQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import PerfectMazeGenerator, RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import RecurrentProceduralDQNRunner, RunnerCallbacks


def _print_log(level: str, msg: str) -> None:
    print(f"[{level:7s}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="DRQN procedural training")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--mode", choices=("obstacles", "maze"), default="obstacles")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fc-hidden", type=int, default=256)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--epsilon-decay-steps", type=int, default=200_000)
    args = parser.parse_args()

    proc_cfg = ProceduralEnvConfig(mode=args.mode)
    if args.mode == "obstacles":
        gen = RandomObstaclesGenerator(
            rows=proc_cfg.max_rows, cols=proc_cfg.max_cols,
            start=(0, 0), goal=(proc_cfg.max_rows - 1, proc_cfg.max_cols - 1),
            min_density=proc_cfg.min_density, max_density=proc_cfg.max_density,
            max_attempts=proc_cfg.max_attempts_bfs,
        )
    else:
        gen = PerfectMazeGenerator(min_size=proc_cfg.min_size, max_size=proc_cfg.max_size)

    env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
    drqn_cfg = DRQNConfig(
        episodes=args.episodes,
        fc_hidden=args.fc_hidden,
        lstm_hidden=args.lstm_hidden,
        sequence_length=args.sequence_length,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    sched_cfg = SchedulerConfig()
    train_cfg = TrainingConfig()

    cb = RunnerCallbacks(on_log=_print_log)
    runner = RecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, drqn_cfg=drqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=cb, device=args.device, seed=args.seed,
    )
    runner.run()

    final_wr = runner.metrics.winrate()
    final_diff = runner.scheduler.current
    print(f"\nFinal : winrate={final_wr:.2%}, difficulty={final_diff:.2f}")
    print("Per-bucket winrate :")
    for i, wr in enumerate(runner.bucket_tracker.winrate_per_bucket()):
        wr_str = f"{wr:.2%}" if wr is not None else "-"
        print(f"  bucket {i} ({i*0.2:.1f}-{(i+1)*0.2:.1f}) : {wr_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
