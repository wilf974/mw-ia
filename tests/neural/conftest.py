"""Fixtures partagées pour les tests neural."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="suite complete : requiert torch")


@pytest.fixture
def cpu_device() -> torch.device:
    """Device CPU pour tests déterministes (pas de dépendance CUDA)."""
    return torch.device("cpu")
