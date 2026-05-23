"""Entraînement CNN-DQN procedural headless (V2-Z, CLI).

Usage :
    python scripts/train_cnn_dqn_procedural.py --episodes 200 --mode obstacles --device cpu
"""
from __future__ import annotations

import argparse
import sys

from mw_ia.config import (
    ConvDQNConfig, ProceduralEnvConfig, SchedulerConfig, TrainingConfig,
)
from mw_ia.envs.maze_generators import PerfectMazeGenerator, RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.training.runner import ConvProceduralDQNRunner, RunnerCallbacks


def _print_log(level: str, msg: str) -> None:
    print(f"[{level:7s}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="CNN-DQN procedural training (V2-Z)")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--mode", choices=("obstacles", "maze"), default="obstacles")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conv-channels", type=int, nargs="+", default=[32, 64],
                        help="Tailles des couches conv (ex: --conv-channels 16 32)")
    parser.add_argument("--fc-hidden", type=int, default=256,
                        help="Taille de la couche FC après le bloc conv (default 256)")
    parser.add_argument("--epsilon-decay-steps", type=int, default=200_000,
                        help="Steps pour passer ε de start à end (default V2-Z : 200000)")
    parser.add_argument("--target-sync-steps", type=int, default=1_000,
                        help="Périodicité de la sync target ← online (default 1000)")
    parser.add_argument("--scheduler-update-interval", type=int, default=200,
                        help="Périodicité (ép) du scheduler (default V2-X : 200)")
    parser.add_argument("--scheduler-step", type=float, default=0.05,
                        help="Pas de difficulté du scheduler (default V2-X : 0.05)")
    parser.add_argument(
        "--double-dqn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Double DQN (Hasselt 2015) : online sélectionne, target évalue. "
             "Default V2-W. Utiliser --no-double-dqn pour reproduire V2-Z baseline.",
    )
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Évaluation périodique greedy sur seeds eval séparés (V2-V). "
             "Default activé. Utiliser --no-eval pour reproduire baseline pre-V2-V.",
    )
    parser.add_argument(
        "--eval-every-episodes",
        type=int,
        default=100,
        help="Périodicité (ép) de l'évaluation greedy (default V2-V : 100).",
    )
    parser.add_argument(
        "--eval-target-difficulty",
        type=float,
        default=0.30,
        help="Difficulty FIXE pour l'eval greedy (default 0.30). "
             "Sans diff fixe, le best capture l'agent trivial à diff=0 (winrate ~100%) "
             "et n'est jamais battu par l'agent compétent à diff supérieure.",
    )
    parser.add_argument(
        "--best-checkpoint-path",
        type=str,
        default=None,
        help="Chemin .pt du best-checkpoint (default None = pas de sauvegarde disque). "
             "Suggestion : checkpoints/v2v_best_seed{N}.pt",
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
    dqn_cfg = ConvDQNConfig(
        episodes=args.episodes,
        conv_channels=tuple(args.conv_channels),
        fc_hidden=args.fc_hidden,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_sync_steps=args.target_sync_steps,
        double_dqn=args.double_dqn,
        eval_enabled=args.eval,
        eval_every_episodes=args.eval_every_episodes,
        eval_target_difficulty=args.eval_target_difficulty,
        best_checkpoint_path=args.best_checkpoint_path,
    )
    sched_cfg = SchedulerConfig(
        update_interval=args.scheduler_update_interval,
        step=args.scheduler_step,
    )
    train_cfg = TrainingConfig()

    cb = RunnerCallbacks(on_log=_print_log)
    runner = ConvProceduralDQNRunner(
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
