"""4 courbes PyQtGraph : reward / loss / epsilon / winrate."""
from __future__ import annotations

from collections import deque
from typing import Deque

import pyqtgraph as pg
from PyQt6.QtWidgets import QGridLayout, QWidget

from mw_ia.gui.theme import THEME

pg.setConfigOption("background", THEME.bg)
pg.setConfigOption("foreground", THEME.fg)


class _Plot(pg.PlotWidget):
    def __init__(self, title: str, color: str, ymin: float | None = None,
                 ymax: float | None = None, maxlen: int = 1000) -> None:
        super().__init__()
        self.setTitle(title, color=THEME.accent, size="10pt")
        self.showGrid(x=True, y=True, alpha=0.25)
        pen = pg.mkPen(color=color, width=2)
        self._curve = self.plot([], [], pen=pen)
        self._xs: Deque[float] = deque(maxlen=maxlen)
        self._ys: Deque[float] = deque(maxlen=maxlen)
        if ymin is not None or ymax is not None:
            self.setYRange(ymin if ymin is not None else 0.0,
                           ymax if ymax is not None else 1.0)

    def push(self, x: float, y: float) -> None:
        self._xs.append(x)
        self._ys.append(y)
        self._curve.setData(list(self._xs), list(self._ys))

    def clear_data(self) -> None:
        self._xs.clear()
        self._ys.clear()
        self._curve.setData([], [])


class LivePlots(QWidget):
    """Widget composite : Reward, Loss, Epsilon, Winrate, Difficulty (procedural)."""

    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.reward = _Plot("Reward / épisode", THEME.accent2)
        self.loss = _Plot("Loss DQN", THEME.danger)
        self.epsilon = _Plot("Epsilon", THEME.warning, ymin=0.0, ymax=1.05)
        self.winrate = _Plot("Winrate (fenêtre)", THEME.accent, ymin=0.0, ymax=1.05)
        self.difficulty = _Plot("Difficulté (procedural)", THEME.accent2, ymin=0.0, ymax=1.05)
        layout.addWidget(self.reward, 0, 0)
        layout.addWidget(self.loss, 0, 1)
        layout.addWidget(self.epsilon, 0, 2)
        layout.addWidget(self.winrate, 1, 0)
        layout.addWidget(self.difficulty, 1, 1)

    def reset(self) -> None:
        for p in (self.reward, self.loss, self.epsilon, self.winrate, self.difficulty):
            p.clear_data()
