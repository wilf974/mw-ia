"""Affichage animé du GridWorld via QGraphicsScene."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSlot
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView

from mw_ia.envs.gridworld import GridWorld
from mw_ia.gui.theme import THEME

_CELL = 36


class GridWorldView(QGraphicsView):
    """Rendu rapide d'une grille 2D + agent + obstacles + goal."""

    def __init__(self, env: GridWorld) -> None:
        super().__init__()
        self.env = env
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QBrush(QColor(THEME.bg)))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._agent_item = None
        self._trail: list[tuple[int, int]] = []
        self._max_trail = 60
        self._draw_static()
        self._draw_agent()

    def _draw_static(self) -> None:
        pen = QPen(QColor(THEME.grid_line))
        pen.setWidth(1)
        for r in range(self.env.cfg.rows):
            for c in range(self.env.cfg.cols):
                x, y = c * _CELL, r * _CELL
                self._scene.addRect(x, y, _CELL, _CELL, pen, QBrush(QColor(THEME.bg_alt)))
        ob_brush = QBrush(QColor(THEME.obstacle))
        for r, c in self.env.cfg.obstacles:
            item = self._scene.addRect(c * _CELL, r * _CELL, _CELL, _CELL, pen, ob_brush)
            item.setData(0, "obstacle")
        gr, gc = self.env.cfg.goal
        goal_item = self._scene.addRect(
            gc * _CELL + 4, gr * _CELL + 4, _CELL - 8, _CELL - 8,
            QPen(QColor(THEME.goal)), QBrush(QColor(THEME.goal)),
        )
        goal_item.setData(0, "goal")

    def _draw_agent(self) -> None:
        r, c = self.env.state
        if self._agent_item is not None:
            self._scene.removeItem(self._agent_item)
        self._agent_item = self._scene.addEllipse(
            c * _CELL + 6, r * _CELL + 6, _CELL - 12, _CELL - 12,
            QPen(QColor(THEME.agent)), QBrush(QColor(THEME.agent)),
        )

    def update_state(self, state: tuple[int, int]) -> None:
        """Slot Qt — déplace l'agent et trace une trail courte."""
        r, c = state
        self._trail.append((r, c))
        if len(self._trail) > self._max_trail:
            self._trail.pop(0)
        for it in list(self._scene.items()):
            if it.data(0) == "trail":
                self._scene.removeItem(it)
        pen = QPen(QColor(THEME.trail))
        for tr_r, tr_c in self._trail[:-1]:
            dot = self._scene.addEllipse(
                tr_c * _CELL + 14, tr_r * _CELL + 14, _CELL - 28, _CELL - 28,
                pen, QBrush(QColor(THEME.trail)),
            )
            dot.setData(0, "trail")
        self.env._state = state
        self._draw_agent()

    def reset_view(self) -> None:
        self._trail.clear()
        for it in list(self._scene.items()):
            if it.data(0) == "trail":
                self._scene.removeItem(it)
        self._draw_agent()

    @pyqtSlot(object, int, float)
    def on_maze_changed(self, maze: np.ndarray, episode_id: int, difficulty: float) -> None:
        """Slot Qt — redessine obstacles + goal à partir d'un nouveau maze.

        Args:
            maze: np.ndarray[bool] shape (rows, cols), True = obstacle.
            episode_id: numéro d'épisode (info, pas utilisé pour le draw).
            difficulty: difficulté courante (info, pas utilisé).
        """
        # Effacer obstacles + goal précédents
        for it in list(self._scene.items()):
            if it.data(0) in ("obstacle", "goal"):
                self._scene.removeItem(it)
        # Redessiner les obstacles
        pen = QPen(QColor(THEME.grid_line))
        pen.setWidth(1)
        ob_brush = QBrush(QColor(THEME.obstacle))
        rows, cols = maze.shape
        for r in range(rows):
            for c in range(cols):
                if maze[r, c]:
                    item = self._scene.addRect(
                        c * _CELL, r * _CELL, _CELL, _CELL, pen, ob_brush,
                    )
                    item.setData(0, "obstacle")
        # Redessiner le goal (toujours coin opposé)
        gr, gc = rows - 1, cols - 1
        goal_item = self._scene.addRect(
            gc * _CELL + 4, gr * _CELL + 4, _CELL - 8, _CELL - 8,
            QPen(QColor(THEME.goal)), QBrush(QColor(THEME.goal)),
        )
        goal_item.setData(0, "goal")
        # Effacer trail
        self._trail.clear()
        for it in list(self._scene.items()):
            if it.data(0) == "trail":
                self._scene.removeItem(it)

    def sizeHint(self) -> QSize:
        return QSize(self.env.cfg.cols * _CELL + 4, self.env.cfg.rows * _CELL + 4)
