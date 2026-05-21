"""Console journal (append-only) — recolore par niveau."""
from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QPlainTextEdit

from mw_ia.gui.theme import THEME


_COLORS = {
    "info": THEME.fg,
    "debug": "#888",
    "warning": THEME.warning,
    "error": THEME.danger,
}


class LogConsole(QPlainTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def append(self, level: str, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        color = _COLORS.get(level.lower(), THEME.fg)
        html = (
            f'<span style="color:#777;">[{ts}]</span> '
            f'<span style="color:{color};">[{level.upper()}]</span> '
            f'<span style="color:{THEME.fg};">{message}</span>'
        )
        self.appendHtml(html)
        self.moveCursor(QTextCursor.MoveOperation.End)
