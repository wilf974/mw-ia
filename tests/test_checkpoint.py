"""Tests Checkpoint."""
from __future__ import annotations

import json

from mw_ia.persistence.checkpoint import dump_metrics, load_metrics


def test_dump_and_load_metrics_roundtrip(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    payload = {"episode": 42, "reward": 0.95, "loss": 0.123, "epsilon": 0.05}
    dump_metrics(path, payload)
    loaded = load_metrics(path)
    assert loaded == payload


def test_dump_creates_parent_dirs(tmp_path) -> None:
    path = tmp_path / "subdir" / "deep" / "m.json"
    dump_metrics(path, {"k": 1})
    assert path.exists()
    assert json.loads(path.read_text())["k"] == 1
