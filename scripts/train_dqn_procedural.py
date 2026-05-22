"""Entraînement DQN procedural headless (CLI).

Usage :
    python scripts/train_dqn_procedural.py --episodes 200 --mode obstacles --device cpu
"""
from __future__ import annotations

import argparse
import sys

from mw_ia.config import (
    DQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import PerfectMazeGenerator, RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ProceduralDQNRunner, RunnerCallbacks


def _print_log(level: str, msg: str) -> None:
    print(f"[{level:7s}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="DQN procedural training")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--mode", choices=("obstacles", "maze"), default="obstacles")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128],
                        help="Tailles des couches cachées du QNetwork (ex: --hidden 256 256)")
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
    dqn_cfg = DQNConfig(episodes=args.episodes, hidden_layers=tuple(args.hidden))
    sched_cfg = SchedulerConfig()
    train_cfg = TrainingConfig()

    cb = RunnerCallbacks(on_log=_print_log)
    runner = ProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
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
