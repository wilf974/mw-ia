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


def test_var_non_negative_single_samples():
    rms = RunningMeanStd(shape=())
    for x in (3.0, 3.0, 3.0):
        rms.update(np.array([x]))
    assert rms.var >= 0.0
    assert rms.std >= 0.0


def test_vector_shape_tracked_elementwise():
    rms = RunningMeanStd(shape=(2,))
    batch = np.array([[1.0, 10.0], [3.0, 30.0]])
    rms.update(batch)
    assert rms.mean.shape == (2,)
    assert rms.mean[1] > rms.mean[0]


def test_std_positive_after_init():
    rms = RunningMeanStd(shape=(3, 4, 4))
    assert np.all(rms.std > 0.0)
