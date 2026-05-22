"""PeriodicEvaluator — évaluation périodique greedy pour V2-V.

Garantit zéro pollution du training :
- env eval distinct de l'env training (créé à l'init du PeriodicEvaluator)
- agent.act(obs, greedy=True) bypass eps-greedy et le rng training
- agent.observe() JAMAIS appelé → pas de buffer push, pas de global_step,
  pas de scheduler update
- torch.no_grad() interne à agent.act() (déjà géré par V2-Z)

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-training-protocol-stabilization-design.md §2
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from mw_ia.config import ProceduralEnvConfig
from mw_ia.envs.procedural_env import ProceduralGridWorld


class _ActableAgent(Protocol):
    """Contrat minimal qu'un agent doit respecter pour être évalué."""

    def act(self, state: np.ndarray, *, greedy: bool = False) -> int: ...


class PeriodicEvaluator:
    """Évalue un agent en mode greedy sur un set de seeds eval fixes.

    L'env eval est strictement séparé de l'env training pour garantir
    qu'aucune trajectoire d'évaluation ne pollue le replay buffer ou le
    scheduler de difficulté.
    """

    def __init__(
        self,
        *,
        eval_env: ProceduralGridWorld,
        eval_seeds: tuple[int, ...],
        max_steps: int,
        observation_encoder: Callable[..., np.ndarray],
        proc_cfg: ProceduralEnvConfig,
    ) -> None:
        if len(eval_seeds) == 0:
            raise ValueError("eval_seeds ne peut pas être vide")
        if max_steps <= 0:
            raise ValueError(f"max_steps doit être > 0, reçu {max_steps}")
        self.eval_env = eval_env
        self.eval_seeds = tuple(eval_seeds)
        self.max_steps = int(max_steps)
        self.observation_encoder = observation_encoder
        self.proc_cfg = proc_cfg

    def evaluate(
        self, agent: _ActableAgent, difficulty: float,
    ) -> dict[str, float]:
        """Lance len(eval_seeds) rollouts greedy. Retourne metrics dict.

        Args:
            agent: agent à évaluer (doit exposer act(state, greedy=True) -> int).
            difficulty: difficulty à set sur l'env eval avant les rollouts.

        Returns:
            dict avec keys : winrate, mean_reward, mean_length, n_episodes, difficulty.
        """
        self.eval_env.set_difficulty(difficulty)
        n_success = 0
        total_reward = 0.0
        total_length = 0
        for seed in self.eval_seeds:
            state, info = self.eval_env.reset(seed=seed)
            maze = info["maze"]
            goal = self.eval_env.inner.cfg.goal
            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated) and ep_len < self.max_steps:
                obs = self.observation_encoder(
                    state=state, grid=maze, goal=goal,
                    max_rows=self.proc_cfg.max_rows,
                    max_cols=self.proc_cfg.max_cols,
                )
                action = agent.act(obs, greedy=True)
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                ep_reward += reward
                ep_len += 1
            if terminated:
                n_success += 1
            total_reward += ep_reward
            total_length += ep_len

        n = len(self.eval_seeds)
        return {
            "winrate": n_success / n,
            "mean_reward": total_reward / n,
            "mean_length": total_length / n,
            "n_episodes": n,
            "difficulty": float(difficulty),
        }
