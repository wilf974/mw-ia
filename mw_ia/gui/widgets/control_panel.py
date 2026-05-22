"""Boutons Start / Pause / Reset / Save / Load + sélecteur nombre d'épisodes."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSpinBox, QWidget


class ControlPanel(QWidget):
    start_clicked = pyqtSignal()
    start_procedural_clicked = pyqtSignal()
    start_procedural_cnn_clicked = pyqtSignal()
    pause_clicked = pyqtSignal(bool)
    reset_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    load_clicked = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.lbl_episodes = QLabel("Épisodes :")
        self.spin_episodes = QSpinBox()
        self.spin_episodes.setRange(10, 50_000)
        self.spin_episodes.setSingleStep(100)
        self.spin_episodes.setValue(1_000)
        self.spin_episodes.setToolTip(
            "Nombre d'épisodes d'entraînement (10–50 000). "
            "Référence : V2-Y/Z/W baselines = 5000 ép."
        )
        self.btn_start = QPushButton("Démarrer")
        self.btn_start_procedural = QPushButton("Démarrer (procedural)")
        self.btn_start_procedural_cnn = QPushButton("Démarrer (procedural CNN)")
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setCheckable(True)
        self.btn_reset = QPushButton("Réinitialiser")
        self.btn_save = QPushButton("Sauvegarder")
        self.btn_load = QPushButton("Charger")
        layout.addWidget(self.lbl_episodes)
        layout.addWidget(self.spin_episodes)
        for b in (
            self.btn_start, self.btn_start_procedural, self.btn_start_procedural_cnn,
            self.btn_pause, self.btn_reset, self.btn_save, self.btn_load,
        ):
            layout.addWidget(b)
        layout.addStretch(1)

        self.btn_start.clicked.connect(self.start_clicked)
        self.btn_start_procedural.clicked.connect(self.start_procedural_clicked)
        self.btn_start_procedural_cnn.clicked.connect(self.start_procedural_cnn_clicked)
        self.btn_pause.toggled.connect(self._on_pause_toggled)
        self.btn_reset.clicked.connect(self.reset_clicked)
        self.btn_save.clicked.connect(self.save_clicked)
        self.btn_load.clicked.connect(self.load_clicked)

    def episodes(self) -> int:
        """Nombre d'épisodes courant sélectionné via le spinbox."""
        return int(self.spin_episodes.value())

    def _on_pause_toggled(self, checked: bool) -> None:
        self.btn_pause.setText("Reprendre" if checked else "Pause")
        self.pause_clicked.emit(checked)

    def set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_start_procedural.setEnabled(not running)
        self.btn_start_procedural_cnn.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_reset.setEnabled(not running)
        self.btn_load.setEnabled(not running)
        self.spin_episodes.setEnabled(not running)
