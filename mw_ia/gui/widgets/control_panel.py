"""Boutons Start / Pause / Reset / Save / Load."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget


class ControlPanel(QWidget):
    start_clicked = pyqtSignal()
    start_procedural_clicked = pyqtSignal()
    pause_clicked = pyqtSignal(bool)
    reset_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    load_clicked = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.btn_start = QPushButton("Démarrer")
        self.btn_start_procedural = QPushButton("Démarrer (procedural)")
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setCheckable(True)
        self.btn_reset = QPushButton("Réinitialiser")
        self.btn_save = QPushButton("Sauvegarder")
        self.btn_load = QPushButton("Charger")
        for b in (self.btn_start, self.btn_start_procedural, self.btn_pause, self.btn_reset, self.btn_save, self.btn_load):
            layout.addWidget(b)
        layout.addStretch(1)

        self.btn_start.clicked.connect(self.start_clicked)
        self.btn_start_procedural.clicked.connect(self.start_procedural_clicked)
        self.btn_pause.toggled.connect(self._on_pause_toggled)
        self.btn_reset.clicked.connect(self.reset_clicked)
        self.btn_save.clicked.connect(self.save_clicked)
        self.btn_load.clicked.connect(self.load_clicked)

    def _on_pause_toggled(self, checked: bool) -> None:
        self.btn_pause.setText("Reprendre" if checked else "Pause")
        self.pause_clicked.emit(checked)

    def set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_start_procedural.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_reset.setEnabled(not running)
        self.btn_load.setEnabled(not running)
