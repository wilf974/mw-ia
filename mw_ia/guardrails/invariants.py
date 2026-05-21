"""Catalogue v1 des 8 invariants RL — implémentations runtime Python.

Chaque @invariant doit avoir son équivalent Aether dans aether/invariants/.
Cohérence vérifiée par tests/guardrails/test_aether_python_sync.py.
"""
from __future__ import annotations

from typing import Optional

from mw_ia.guardrails.contracts import Severity, VariantSpec, Violation
from mw_ia.guardrails.registry import invariant


@invariant("I1", applies_to=["gamma"])
def gamma_in_open_unit(spec: VariantSpec) -> Optional[Violation]:
    """γ doit être dans l'intervalle ouvert (0, 1) pour garantir la contraction."""
    if not (0.0 < spec.gamma < 1.0):
        return Violation(
            invariant_id="I1",
            message=f"gamma={spec.gamma} hors (0,1)",
            severity=Severity.HARD,
        )
    return None


import numpy as np


def _bellman_operator(Q: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """Opérateur de Bellman optimal sur MDP tabulaire.

    Q : (S, A) — fonction Q
    P : (S, A, S') — transitions
    R : (S, A, S') — récompenses
    Returns : (S, A) tableau TQ
    """
    V_next = Q.max(axis=1)
    bellman_target = (P * (R + gamma * V_next[None, None, :])).sum(axis=2)
    return bellman_target


@invariant("I2", applies_to=["gamma"])
def bellman_contraction(spec: VariantSpec) -> Optional[Violation]:
    """∀ Q, Q' : ||TQ - TQ'||∞ ≤ γ ||Q - Q'||∞  avec γ < 1 (contraction stricte).

    γ ≥ 1 rend l'opérateur de Bellman non-contractant au sens strict :
    violation immédiate détectée analytiquement.
    Pour γ < 1, vérifié empiriquement sur 50 paires (Q, Q') sur un mini-MDP.
    """
    # Condition analytique : γ doit être strictement < 1
    if spec.gamma >= 1.0:
        return Violation(
            invariant_id="I2",
            message=f"Bellman non-contractant : γ={spec.gamma} ≥ 1 → pas de contraction stricte",
            severity=Severity.HARD,
            counter_example={
                "gamma": spec.gamma,
                "lhs": spec.gamma,
                "rhs": spec.gamma,
                "ratio": 1.0,
            },
        )

    rng = np.random.default_rng(seed=42)
    S, A = 3, 2
    P = rng.uniform(0.0, 1.0, size=(S, A, S))
    P = P / P.sum(axis=2, keepdims=True)
    R = rng.uniform(-1.0, 1.0, size=(S, A, S))

    for _ in range(50):
        Q = rng.uniform(-10.0, 10.0, size=(S, A))
        Qp = rng.uniform(-10.0, 10.0, size=(S, A))
        TQ = _bellman_operator(Q, P, R, spec.gamma)
        TQp = _bellman_operator(Qp, P, R, spec.gamma)
        lhs = float(np.abs(TQ - TQp).max())
        rhs = spec.gamma * float(np.abs(Q - Qp).max())
        if lhs > rhs + 1e-9:
            return Violation(
                invariant_id="I2",
                message=f"Bellman non-contractant : ||TQ-TQ'||∞={lhs:.6f} > γ·||Q-Q'||∞={rhs:.6f}",
                severity=Severity.HARD,
                counter_example={
                    "gamma": spec.gamma,
                    "lhs": lhs,
                    "rhs": rhs,
                    "ratio": lhs / rhs if rhs > 0 else float("inf"),
                },
            )
    return None


from mw_ia.config import TrainingConfig
from mw_ia.training.metrics import MetricsTracker


@invariant("I4", applies_to=[])
def winrate_bounds(spec: VariantSpec) -> Optional[Violation]:
    """winrate ∈ [0, 1] sur fenêtre glissante.

    Vérifié en alimentant MetricsTracker avec 200 résultats aléatoires
    et en confirmant que winrate() reste borné.
    """
    rng = np.random.default_rng(seed=42)
    tracker = MetricsTracker(TrainingConfig())
    for _ in range(200):
        success = bool(rng.integers(0, 2))
        tracker.record_episode(reward=0.0, length=1, success=success)
        wr = tracker.winrate()
        if not (0.0 <= wr <= 1.0):
            return Violation(
                invariant_id="I4",
                message=f"winrate={wr} hors [0,1]",
                severity=Severity.HARD,
                counter_example={"winrate": wr},
            )
    return None


from mw_ia.neural.replay_buffer import ReplayBuffer


@invariant("I6", applies_to=["replay_capacity"])
def replay_buffer_capacity(spec: VariantSpec) -> Optional[Violation]:
    """buffer.size ≤ capacity ∧ index < capacity, même après débordement.

    Pousse capacity * 3 transitions dans un buffer de capacity définie par
    le spec, vérifie qu'aucune borne n'est jamais franchie.
    """
    capacity = min(spec.replay_capacity, 1_000)  # cap pour rapidité du check
    obs_dim = 2
    buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, seed=42)
    rng = np.random.default_rng(seed=42)
    for _ in range(capacity * 3):
        s = rng.uniform(size=obs_dim).astype(np.float32)
        sp = rng.uniform(size=obs_dim).astype(np.float32)
        buf.push(s, action=0, reward=0.0, next_state=sp, done=False)
        if len(buf) > capacity:
            return Violation(
                invariant_id="I6",
                message=f"buffer.size={len(buf)} > capacity={capacity}",
                severity=Severity.HARD,
                counter_example={"size": len(buf), "capacity": capacity},
            )
        if buf._idx >= capacity:
            return Violation(
                invariant_id="I6",
                message=f"buffer._idx={buf._idx} >= capacity={capacity}",
                severity=Severity.HARD,
                counter_example={"idx": int(buf._idx), "capacity": capacity},
            )
    return None


@invariant("I7", applies_to=["reward_min", "reward_max"])
def reward_bounded(spec: VariantSpec) -> Optional[Violation]:
    """reward_min ≤ reward_max (cohérence interne des bornes annoncées)."""
    assert spec.reward_min is not None and spec.reward_max is not None  # garanti par applies_to
    if spec.reward_min > spec.reward_max:
        return Violation(
            invariant_id="I7",
            message=f"reward_min={spec.reward_min} > reward_max={spec.reward_max}",
            severity=Severity.HARD,
            counter_example={"reward_min": spec.reward_min, "reward_max": spec.reward_max},
        )
    return None


def _compute_epsilon(t: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """Schedule linéaire de ε(t) — référence pour I5.

    Identique à la formule utilisée par DQNAgent en V1.
    """
    if t >= decay_steps:
        return eps_end
    return eps_start + (eps_end - eps_start) * (t / decay_steps)


@invariant("I5", applies_to=["epsilon_start", "epsilon_end", "epsilon_decay_steps"])
def epsilon_schedule_decreasing(spec: VariantSpec) -> Optional[Violation]:
    """ε_{t+1} ≤ ε_t et ε_t ∈ [0,1] sur tout l'horizon."""
    prev = spec.epsilon_start
    horizon = 2 * spec.epsilon_decay_steps
    step = max(1, horizon // 100)
    for t in range(0, horizon + 1, step):
        eps = _compute_epsilon(t, spec.epsilon_start, spec.epsilon_end, spec.epsilon_decay_steps)
        if not (0.0 <= eps <= 1.0):
            return Violation(
                invariant_id="I5",
                message=f"epsilon(t={t})={eps} hors [0,1]",
                severity=Severity.HARD,
                counter_example={"t": t, "epsilon": eps},
            )
        if eps > prev + 1e-9:
            return Violation(
                invariant_id="I5",
                message=f"epsilon non décroissant : eps(t={t})={eps} > prev={prev}",
                severity=Severity.HARD,
                counter_example={"t": t, "prev": prev, "epsilon": eps},
            )
        prev = eps
    return None


def _huber(y: float, y_hat: float, delta: float = 1.0) -> float:
    """Huber loss (référence pédagogique, indépendante de torch)."""
    diff = y - y_hat
    abs_diff = abs(diff)
    if abs_diff <= delta:
        return 0.5 * diff * diff
    return delta * (abs_diff - 0.5 * delta)


@invariant("I3", applies_to=[])
def huber_nonneg(spec: VariantSpec) -> Optional[Violation]:
    """∀ y, ŷ : Huber(y, ŷ) ≥ 0.

    Vérifié empiriquement sur 100 paires (y, ŷ) uniformément échantillonnées.
    """
    rng = np.random.default_rng(seed=42)
    for _ in range(100):
        y = float(rng.uniform(-100.0, 100.0))
        y_hat = float(rng.uniform(-100.0, 100.0))
        loss = _huber(y, y_hat)
        if loss < 0:
            return Violation(
                invariant_id="I3",
                message=f"Huber loss négatif : {loss} pour y={y}, y_hat={y_hat}",
                severity=Severity.HARD,
                counter_example={"y": y, "y_hat": y_hat, "loss": loss},
            )
    return None
