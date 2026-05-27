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
    parser.add_argument("--scheduler-update-interval", type=int, default=50,
                        help="Périodicité (ép) du scheduler de difficulté (default V2-Y : 50 ; "
                             "le DRQN/LSTM oscille catastrophiquement avec update=200 V2-X)")
    parser.add_argument("--scheduler-step", type=float, default=0.05,
                        help="Pas de difficulté du scheduler (default V2-Y : 0.05)")
    parser.add_argument(
        "--polyak-tau",
        type=float,
        default=0.0,
        help="V2-U : soft Polyak target update tau. Default 0.0 = hard sync.",
    )
    parser.add_argument(
        "--per",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="V2-B0 : Prioritized Experience Replay trajectory-level (Schaul 2015 + R2D2). "
             "Default False = SequenceReplayBuffer uniforme baseline V2-Y.",
    )
    parser.add_argument("--per-alpha", type=float, default=0.6,
                        help="V2-B0 : priority exponent alpha (default 0.6, Schaul 2015).")
    parser.add_argument("--per-beta-start", type=float, default=0.4,
                        help="V2-B0 : IS exponent beta initial (default 0.4, Schaul 2015).")
    parser.add_argument("--per-beta-end", type=float, default=1.0,
                        help="V2-B0 : IS exponent beta final (default 1.0, annealing complete).")
    parser.add_argument("--per-eta", type=float, default=0.9,
                        help="V2-B0 : R2D2 priority aggregation eta (default 0.9).")
    parser.add_argument("--per-epsilon", type=float, default=1e-6,
                        help="V2-B0 : small constant epsilon (default 1e-6) garantit priority > 0.")
    parser.add_argument(
        "--max-attempts-bfs",
        type=int,
        default=100,
        help="ProceduralEnvConfig max_attempts_bfs (default 100). Recommande bench B0 : 500.",
    )
    args = parser.parse_args()

    proc_cfg = ProceduralEnvConfig(mode=args.mode, max_attempts_bfs=args.max_attempts_bfs)
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
        polyak_tau=args.polyak_tau,
        per_enabled=args.per,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_eta=args.per_eta,
        per_epsilon=args.per_epsilon,
    )
    sched_cfg = SchedulerConfig(
        update_interval=args.scheduler_update_interval,
        step=args.scheduler_step,
    )
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
