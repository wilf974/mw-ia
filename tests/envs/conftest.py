"""Fixtures partagées pour les tests envs."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Générateur seedé pour tests stochastiques."""
    return np.random.default_rng(seed=42)
