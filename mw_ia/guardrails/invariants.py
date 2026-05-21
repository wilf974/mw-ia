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
