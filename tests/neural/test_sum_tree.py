"""Tests V2-B0 du SumTree (structure de donnees O(log N) pour PER)."""
from __future__ import annotations

import numpy as np
import pytest

from mw_ia.neural.sum_tree import SumTree


def test_init_validates_capacity() -> None:
    """capacity doit etre >= 1."""
    with pytest.raises(ValueError, match="capacity"):
        SumTree(0)
    with pytest.raises(ValueError, match="capacity"):
        SumTree(-3)


def test_init_capacity_one() -> None:
    """Edge case capacity=1 fonctionne."""
    tree = SumTree(1)
    assert tree.capacity == 1
    assert tree.total() == 0.0


def test_update_and_total() -> None:
    """update(leaf_idx, priority) propage a la racine."""
    tree = SumTree(4)
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)
    tree.update(3, 4.0)
    assert tree.total() == pytest.approx(10.0)


def test_update_overwrites_old_priority() -> None:
    """update sur meme leaf remplace la priorite (pas accumulation)."""
    tree = SumTree(4)
    tree.update(0, 5.0)
    assert tree.total() == pytest.approx(5.0)
    tree.update(0, 2.0)
    assert tree.total() == pytest.approx(2.0)


def test_find_returns_leaf_index_and_priority() -> None:
    """find(value) descend l'arbre et retourne (leaf_idx, priority)."""
    tree = SumTree(4)
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)
    tree.update(3, 4.0)
    # value=0.5 < 1.0 -> leaf 0
    idx, prio = tree.find(0.5)
    assert idx == 0
    assert prio == pytest.approx(1.0)
    # value=1.5 in [1.0, 3.0) -> leaf 1
    idx, prio = tree.find(1.5)
    assert idx == 1
    assert prio == pytest.approx(2.0)
    # value=4.0 in [3.0, 6.0) -> leaf 2
    idx, prio = tree.find(4.0)
    assert idx == 2
    assert prio == pytest.approx(3.0)
    # value=9.9 in [6.0, 10.0) -> leaf 3
    idx, prio = tree.find(9.9)
    assert idx == 3
    assert prio == pytest.approx(4.0)


def test_find_validates_leaf_idx_range() -> None:
    """update avec leaf_idx hors [0, capacity) leve ValueError."""
    tree = SumTree(4)
    with pytest.raises(ValueError, match="leaf_idx"):
        tree.update(4, 1.0)
    with pytest.raises(ValueError, match="leaf_idx"):
        tree.update(-1, 1.0)


def test_find_distribution_converges() -> None:
    """10000 samples : frequence empirique converge vers priorite normalisee a +/-5%."""
    tree = SumTree(5)
    priorities = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, p in enumerate(priorities):
        tree.update(i, p)
    rng = np.random.default_rng(42)
    counts = np.zeros(5, dtype=int)
    n_samples = 10_000
    total = tree.total()
    for _ in range(n_samples):
        value = rng.uniform(0.0, total)
        idx, _ = tree.find(value)
        counts[idx] += 1
    expected_freq = np.array(priorities) / sum(priorities)
    empirical_freq = counts / n_samples
    assert np.allclose(empirical_freq, expected_freq, atol=0.02)


def test_capacity_5000_works() -> None:
    """capacity=5000 (default V2-ZY 10x10) fonctionne sans crash."""
    tree = SumTree(5000)
    for i in range(5000):
        tree.update(i, float(i + 1))
    expected_total = sum(range(1, 5001))
    assert tree.total() == pytest.approx(float(expected_total), rel=1e-5)
    # Sample test
    idx, _ = tree.find(0.5)
    assert idx == 0
    idx, _ = tree.find(tree.total() - 0.5)
    assert idx == 4999
