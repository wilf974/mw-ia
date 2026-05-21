"""Fixtures pytest partagées."""
from __future__ import annotations

import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    """Force la graine pour chaque test (reproductibilité)."""
    random.seed(0)
    np.random.seed(0)
