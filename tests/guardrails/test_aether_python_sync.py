"""Garde-fou : chaque .aether d'aether/invariants/ a son @invariant Python et vice-versa."""
from __future__ import annotations

import re
from pathlib import Path

import mw_ia.guardrails.invariants  # noqa: F401  (peuple le registry)
from mw_ia.guardrails.registry import _REGISTRY


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _aether_ids() -> set[str]:
    """Extrait les IDs (I1..I8) des fichiers `iN_*.aether`."""
    aether_dir = _project_root() / "aether" / "invariants"
    ids = set()
    for aether_file in aether_dir.glob("i*_*.aether"):
        m = re.match(r"^i(\d+)_", aether_file.name, re.IGNORECASE)
        assert m, f"Nom de fichier non conforme : {aether_file.name}"
        ids.add(f"I{m.group(1)}")
    return ids


def _python_ids() -> set[str]:
    return set(_REGISTRY.keys())


def test_aether_files_have_python_invariants():
    aether = _aether_ids()
    python = _python_ids()
    missing = aether - python
    assert not missing, f"Validations Aether sans @invariant Python : {missing}"


def test_python_invariants_have_aether_files():
    aether = _aether_ids()
    python = _python_ids()
    missing = python - aether
    assert not missing, f"@invariant Python sans validation Aether : {missing}"


def test_id_extraction_is_case_insensitive():
    """Sanity : la regex matche I1, i1, etc."""
    assert re.match(r"^i(\d+)_", "i1_test.aether", re.IGNORECASE)
    assert re.match(r"^i(\d+)_", "I7_xyz.aether", re.IGNORECASE)
