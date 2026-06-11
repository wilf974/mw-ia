"""Tests du normaliseur en ligne RunningMeanStd (V2-C0 RND)."""
from __future__ import annotations

import numpy as np

from mw_ia.neural.running_mean_std import RunningMeanStd


def test_mean_converges_scalar():
    rms = RunningMeanStd(shape=())
    rng = np.random.RandomState(0)
    data = rng.normal(5.0, 2.0, size=5000)
    for i in range(0, 5000, 50):
        rms.update(data[i : i + 50])
    assert abs(rms.mean - 5.0) < 0.2
    assert abs(rms.std - 2.0) < 0.2


def test_var_converges_to_zero_on_identical_samples():
    rms = RunningMeanStd(shape=())
    for _ in range(200):
        rms.update(np.array([3.0]))
    assert rms.var < 0.01       # variance collapses toward 0
    assert rms.std >= 0.0       # std stays well-defined
    assert abs(rms.mean - 3.0) < 0.05


def test_vector_shape_tracked_elementwise():
    rms = RunningMeanStd(shape=(2,))
    batch = np.array([[1.0, 10.0], [3.0, 30.0]])
    rms.update(batch)
    assert rms.mean.shape == (2,)
    assert np.allclose(rms.mean, [2.0, 20.0], atol=1e-3)


def test_std_positive_after_init():
    rms = RunningMeanStd(shape=(3, 4, 4))
    assert np.all(rms.std > 0.0)
