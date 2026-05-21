"""Sanity check sur la configuration."""
from mw_ia.config import DEFAULT


def test_default_gridworld_dimensions() -> None:
    assert DEFAULT.gridworld.rows == 10
    assert DEFAULT.gridworld.cols == 10


def test_default_dqn_batch_and_lr() -> None:
    assert DEFAULT.dqn.batch_size == 128
    assert DEFAULT.dqn.lr == 1e-3


def test_level_thresholds_sorted() -> None:
    th = DEFAULT.training.level_thresholds
    assert th == tuple(sorted(th))
    assert 0.0 < th[0] < th[1] < th[2] < 1.0
