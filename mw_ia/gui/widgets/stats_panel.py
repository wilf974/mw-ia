"""Panneau de KPIs : épisode, reward, best, winrate, niveau IA."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFormLayout, QFrame, QLabel, QVBoxLayout, QWidget

from mw_ia.gui.theme import THEME
from mw_ia.training.metrics import Level


class StatsPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Statistiques")
        title.setObjectName("title")
        outer.addWidget(title)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        form = QFormLayout(frame)

        self.lbl_ep = QLabel("0")
        self.lbl_reward = QLabel("0.00")
        self.lbl_best = QLabel("—")
        self.lbl_loss = QLabel("—")
        self.lbl_eps = QLabel("1.000")
        self.lbl_winrate = QLabel("0 %")
        self.lbl_level = QLabel(Level.BEGINNER.value)
        self.lbl_level.setStyleSheet(f"color:{THEME.warning}; font-weight:600;")

        form.addRow("Épisode", self.lbl_ep)
        form.addRow("Reward (dernier)", self.lbl_reward)
        form.addRow("Best reward", self.lbl_best)
        form.addRow("Loss", self.lbl_loss)
        form.addRow("Epsilon", self.lbl_eps)
        form.addRow("Winrate", self.lbl_winrate)
        form.addRow("Niveau IA", self.lbl_level)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignTop)
        outer.addStretch(1)

    def update_stats(
        self, *, episode: int, reward: float, best: float,
        loss: float | None, eps: float, winrate: float, level: Level,
    ) -> None:
        self.lbl_ep.setText(str(episode))
        self.lbl_reward.setText(f"{reward:+.3f}")
        self.lbl_best.setText("—" if best == float("-inf") else f"{best:+.3f}")
        self.lbl_loss.setText("—" if loss is None else f"{loss:.4f}")
        self.lbl_eps.setText(f"{eps:.3f}")
        self.lbl_winrate.setText(f"{winrate*100:.1f} %")
        self.lbl_level.setText(level.value)
        color = {
            Level.BEGINNER: THEME.warning,
            Level.INTERMEDIATE: THEME.accent2,
            Level.ADVANCED: THEME.accent,
            Level.EXPERT: THEME.goal,
        }[level]
        self.lbl_level.setStyleSheet(f"color:{color}; font-weight:600;")
