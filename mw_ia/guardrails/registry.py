"""Registre interne des invariants déclarés via @invariant."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from mw_ia.guardrails.contracts import VariantSpec, Violation


CheckFn = Callable[[VariantSpec], Optional[Violation]]


@dataclass(frozen=True)
class Invariant:
    """Métadonnées d'un invariant déclaré."""

    id: str
    applies_to: tuple[str, ...]   # noms de champs VariantSpec requis non-None
    check: CheckFn


_REGISTRY: dict[str, Invariant] = {}


def invariant(id: str, applies_to: list[str]):
    """Décorateur : enregistre la fonction comme invariant `id`."""

    def decorator(fn: CheckFn) -> CheckFn:
        if id in _REGISTRY:
            raise ValueError(f"Invariant déjà enregistré : {id}")
        _REGISTRY[id] = Invariant(id=id, applies_to=tuple(applies_to), check=fn)
        return fn

    return decorator


def applicable_invariants(spec: VariantSpec) -> list[Invariant]:
    """Retourne les invariants pertinents pour ce spec.

    Un invariant est applicable si tous les champs listés dans `applies_to`
    sont présents et non-None dans le spec.
    """
    result = []
    for inv in _REGISTRY.values():
        if all(getattr(spec, fname, None) is not None for fname in inv.applies_to):
            result.append(inv)
    return result
