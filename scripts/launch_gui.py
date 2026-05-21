"""Lance la GUI MW_IA."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from mw_ia.config import DEFAULT
from mw_ia.gui.app import MainWindow
from mw_ia.gui.theme import QSS


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow(DEFAULT)
    win.show()
    run_loop = getattr(app, "exec")
    return run_loop()


if __name__ == "__main__":
    raise SystemExit(main())
