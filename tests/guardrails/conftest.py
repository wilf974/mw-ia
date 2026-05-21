"""Fixtures partagées pour les tests guardrails."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Générateur seedé pour tests d'invariants stochastiques."""
    return np.random.default_rng(seed=42)
