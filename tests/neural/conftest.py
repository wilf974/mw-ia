"""Fixtures partagées pour les tests neural."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cpu_device() -> torch.device:
    """Device CPU pour tests déterministes (pas de dépendance CUDA)."""
    return torch.device("cpu")
