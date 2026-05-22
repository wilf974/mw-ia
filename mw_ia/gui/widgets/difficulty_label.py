"""Widget affichant 'Maze #N, diff=X.XX' pour le mode procedural."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QLabel


class DifficultyLabel(QLabel):
    """Label texte mis à jour à chaque maze_changed."""

    def __init__(self) -> None:
        super().__init__("Mode V1 (map fixe)")
        self.setStyleSheet("color: #DDD; font-family: monospace; font-size: 11pt;")

    @pyqtSlot(object, int, float)
    def on_maze_changed(self, maze, episode_id: int, difficulty: float) -> None:
        self.setText(f"Maze #{episode_id}, diff={difficulty:.2f}")
