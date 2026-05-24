"""Entraînement V2-ZY headless (CLI) : Conv + LSTM + Double DQN combiné.

Usage :
    python scripts/train_cnn_lstm_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda
"""
from __future__ import annotations

import argparse
import sys

from mw_ia.config import (
    ConvRecurrentDQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import PerfectMazeGenerator, RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ConvRecurrentProceduralDQNRunner, RunnerCallbacks


def _print_log(level: str, msg: str) -> None:
    print(f"[{level:7s}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-ZY CNN+LSTM+Double DQN combiné")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--mode", choices=("obstacles", "maze"), default="obstacles")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conv-channels", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--epsilon-decay-steps", type=int, default=200_000)
    parser.add_argument("--scheduler-update-interval", type=int, default=50)
    parser.add_argument("--scheduler-step", type=float, default=0.05)
    parser.add_argument(
        "--double-dqn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Double DQN (V2-W). Default activé pour V2-ZY.",
    )
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Évaluation périodique greedy (V2-V). Default activé.",
    )
    parser.add_argument("--eval-every-episodes", type=int, default=100)
    parser.add_argument("--eval-target-difficulty", type=float, default=0.30)
    parser.add_argument("--best-checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--polyak-tau",
        type=float,
        default=0.0,
        help="V2-U : soft Polyak target update tau. Default 0.0 = hard sync. "
             "Recommande V2-ZY : 0.005 pour reduire variance inter-seed.",
    )
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
    dqn_cfg = ConvRecurrentDQNConfig(
        episodes=args.episodes,
        conv_channels=tuple(args.conv_channels),
        lstm_hidden=args.lstm_hidden,
        sequence_length=args.sequence_length,
        epsilon_decay_steps=args.epsilon_decay_steps,
        double_dqn=args.double_dqn,
        eval_enabled=args.eval,
        eval_every_episodes=args.eval_every_episodes,
        eval_target_difficulty=args.eval_target_difficulty,
        best_checkpoint_path=args.best_checkpoint_path,
        polyak_tau=args.polyak_tau,
    )
    sched_cfg = SchedulerConfig(
        update_interval=args.scheduler_update_interval,
        step=args.scheduler_step,
    )
    train_cfg = TrainingConfig()

    cb = RunnerCallbacks(on_log=_print_log)
    runner = ConvRecurrentProceduralDQNRunner(
        env=env, proc_cfg=proc_cfg, dqn_cfg=dqn_cfg, sched_cfg=sched_cfg,
        train_cfg=train_cfg, callbacks=cb, device=args.device, seed=args.seed,
    )
    runner.run()

    final_wr = runner.metrics.winrate()
    final_diff = runner.scheduler.current
    print(f"\nFinal : winrate={final_wr:.2%}, difficulty={final_diff:.2f}")
    if runner.best_tracker is not None:
        print(
            f"Best @ diff={dqn_cfg.eval_target_difficulty:.2f} : "
            f"winrate={runner.best_tracker.best_winrate:.2%} "
            f"@ ep {runner.best_tracker.best_episode}"
        )
    print("Per-bucket winrate :")
    for i, wr in enumerate(runner.bucket_tracker.winrate_per_bucket()):
        wr_str = f"{wr:.2%}" if wr is not None else "-"
        print(f"  bucket {i} ({i*0.2:.1f}-{(i+1)*0.2:.1f}) : {wr_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
