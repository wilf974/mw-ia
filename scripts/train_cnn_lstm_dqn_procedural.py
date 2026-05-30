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
    parser.add_argument("--max-rows", type=int, default=10,
                        help="Hauteur max de la grille procedurale (default 10).")
    parser.add_argument("--max-cols", type=int, default=10,
                        help="Largeur max de la grille procedurale (default 10).")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Truncation per epoisode du GridWorld interne (default 200). "
                             "Normalisation d'horizon : 10x10 -> 200, 15x15 -> 400, "
                             "20x20 -> 600-800. Eviter qu'une marche aleatoire 2D ne soit "
                             "tronquee avant d'atteindre le goal sur grille plus large.")
    parser.add_argument("--replay-capacity", type=int, default=5_000,
                        help="Nombre de TRAJECTOIRES dans SequenceReplayBuffer "
                             "(default 5000). Reduire a 2000-3000 pour mazes 15x15 "
                             "afin de tenir la VRAM.")
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
    parser.add_argument(
        "--per",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="V2-B0 : Prioritized Experience Replay trajectory-level (Schaul 2015 + R2D2). "
             "Default False = SequenceReplayBuffer uniforme baseline V2-ZY.",
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
    parser.add_argument(
        "--b1a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="V2-B1a : Policy Snapshot Rehearsal - capture des trajectoires successful "
             "depuis le buffer au moment du best eval (V2-V), inject 20%% dans chaque batch. "
             "Default False = pas de rehearsal (V2-U / V2-B0 baseline).",
    )
    parser.add_argument("--b1a-snapshot-size", type=int, default=50,
                        help="V2-B1a : nb de trajectoires capturees par best (default 50).")
    parser.add_argument("--b1a-n-windows", type=int, default=3,
                        help="V2-B1a : sliding window des N derniers bests (default 3).")
    parser.add_argument("--b1a-mix-ratio", type=float, default=0.2,
                        help="V2-B1a : fraction du batch venant du snapshot (default 0.2 = 20%%).")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="V2-BX Sonde A (horizon) : discount factor. Default 0.99. "
             "Sonde A recommande : 0.997 (horizon effectif ~333, coherent max-steps 400).",
    )
    parser.add_argument(
        "--bx-repr-oracle",
        choices=("none", "scalar", "field"),
        default="none",
        help="V2-BX Sonde C (representation) : 4e canal oracle BFS. "
             "none=baseline, scalar=C1 (plan uniforme distance agent), "
             "field=C2 (champ distance par cellule). Default none.",
    )
    parser.add_argument(
        "--bx-novelty-beta",
        type=float,
        default=0.0,
        help="V2-BX Sonde B (exploration) : poids du bonus count-based "
             "beta/sqrt(visits) par cellule/episode. Default 0.0 = desactive. "
             "Point de depart recommande : 0.05 a 0.1.",
    )
    args = parser.parse_args()

    proc_cfg = ProceduralEnvConfig(
        mode=args.mode, max_rows=args.max_rows, max_cols=args.max_cols,
        max_steps=args.max_steps,
        max_attempts_bfs=args.max_attempts_bfs,
    )
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
        replay_capacity=args.replay_capacity,
        max_steps_per_episode=args.max_steps,
        eval_max_steps=args.max_steps,
        double_dqn=args.double_dqn,
        eval_enabled=args.eval,
        eval_every_episodes=args.eval_every_episodes,
        eval_target_difficulty=args.eval_target_difficulty,
        best_checkpoint_path=args.best_checkpoint_path,
        polyak_tau=args.polyak_tau,
        per_enabled=args.per,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_eta=args.per_eta,
        per_epsilon=args.per_epsilon,
        b1a_enabled=args.b1a,
        b1a_snapshot_size=args.b1a_snapshot_size,
        b1a_n_windows=args.b1a_n_windows,
        b1a_mix_ratio=args.b1a_mix_ratio,
        gamma=args.gamma,
        bx_repr_oracle=args.bx_repr_oracle,
        bx_novelty_beta=args.bx_novelty_beta,
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
