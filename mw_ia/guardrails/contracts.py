"""Dataclasses du contrat public guardrails."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional


class Severity(Enum):
    """Gravité d'une violation d'invariant."""

    HARD = "hard"   # mathématiquement faux — bloque le déploiement
    SOFT = "soft"   # atypique mais valide — signale sans bloquer (réservé v2)


@dataclass(frozen=True)
class Violation:
    """Violation d'un invariant pour un VariantSpec donné."""

    invariant_id: str
    message: str
    severity: Severity
    counter_example: Optional[dict] = None


@dataclass(frozen=True)
class VariantSpec:
    """Configuration d'un variant d'agent RL à valider contre les invariants.

    Alignement V1 : noms cohérents avec mw_ia.config.DQNConfig
    (replay_capacity, target_sync_steps).
    """

    gamma: float
    lr: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    batch_size: int
    replay_capacity: int
    target_sync_steps: int
    reward_min: Optional[float] = None
    reward_max: Optional[float] = None

    def __post_init__(self) -> None:
        # Validation structurelle : type / bornes physiques, hors invariants formels.
        if not isinstance(self.gamma, (int, float)):
            raise TypeError(f"gamma doit être float, reçu {type(self.gamma).__name__}")
        if self.lr <= 0:
            raise ValueError(f"lr doit être > 0, reçu {self.lr}")
        if not (0.0 <= self.epsilon_start <= 1.0):
            raise ValueError(f"epsilon_start doit être dans [0,1], reçu {self.epsilon_start}")
        if not (0.0 <= self.epsilon_end <= 1.0):
            raise ValueError(f"epsilon_end doit être dans [0,1], reçu {self.epsilon_end}")
        if self.epsilon_decay_steps <= 0:
            raise ValueError(f"epsilon_decay_steps doit être > 0, reçu {self.epsilon_decay_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size doit être > 0, reçu {self.batch_size}")
        if self.replay_capacity <= 0:
            raise ValueError(f"replay_capacity doit être > 0, reçu {self.replay_capacity}")
        if self.target_sync_steps <= 0:
            raise ValueError(f"target_sync_steps doit être > 0, reçu {self.target_sync_steps}")


@dataclass(frozen=True)
class VerdictReport:
    """Résultat d'un appel verify_formal(spec)."""

    passed: bool
    violations: tuple[Violation, ...]
    spec: VariantSpec
    duration_ms: float

    def to_dict(self) -> dict:
        """Sérialisation JSON-ready pour logs/debug."""
        return {
            "passed": self.passed,
            "violations": [
                {
                    "invariant_id": v.invariant_id,
                    "message": v.message,
                    "severity": v.severity.value,
                    "counter_example": v.counter_example,
                }
                for v in self.violations
            ],
            "spec": asdict(self.spec),
            "duration_ms": self.duration_ms,
        }
