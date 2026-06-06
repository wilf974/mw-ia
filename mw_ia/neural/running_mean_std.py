"""RunningMeanStd — moyenne/variance en ligne (algorithme parallele de Chan).

Utilise par V2-C0 RND pour normaliser les observations et les bonus
intrinseques. numpy float64, hors graphe torch.
"""
from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Moyenne et variance courantes mises a jour par batch.

    update(x) : x de shape (batch, *shape). Les statistiques sont suivies
    element-wise sur `shape`.
    """

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.var = m2 / tot
        self.mean = new_mean
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)
