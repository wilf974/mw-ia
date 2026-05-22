"""Tests des nouveaux dataclasses de configuration procedural."""
from __future__ import annotations

import pytest

from mw_ia.config import ProceduralEnvConfig, SchedulerConfig


def test_procedural_config_default_valid():
    cfg = ProceduralEnvConfig(mode="obstacles")
    assert cfg.mode == "obstacles"
    assert cfg.max_rows == 10
    assert cfg.max_cols == 10


def test_procedural_config_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        ProceduralEnvConfig(mode="invalid")  # type: ignore[arg-type]


def test_procedural_config_density_inverted_raises():
    with pytest.raises(ValueError, match="density"):
        ProceduralEnvConfig(mode="obstacles", min_density=0.5, max_density=0.1)


def test_procedural_config_size_inverted_raises():
    with pytest.raises(ValueError, match="size"):
        ProceduralEnvConfig(mode="maze", min_size=20, max_size=4)


def test_procedural_config_is_frozen():
    cfg = ProceduralEnvConfig(mode="obstacles")
    with pytest.raises(Exception):
        cfg.mode = "maze"  # type: ignore[misc]


def test_scheduler_config_default_valid():
    cfg = SchedulerConfig()
    assert cfg.initial_difficulty == 0.0
    assert cfg.up_threshold == 0.80
    assert cfg.down_threshold == 0.30


def test_scheduler_config_thresholds_inverted_raises():
    with pytest.raises(ValueError, match="threshold"):
        SchedulerConfig(up_threshold=0.30, down_threshold=0.80)


def test_scheduler_config_initial_out_of_range_raises():
    with pytest.raises(ValueError, match="initial"):
        SchedulerConfig(initial_difficulty=1.5)


def test_scheduler_config_is_frozen():
    cfg = SchedulerConfig()
    with pytest.raises(Exception):
        cfg.step = 0.10  # type: ignore[misc]
