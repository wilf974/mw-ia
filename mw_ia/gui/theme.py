"""Palette dark moderne — couleurs partagées par tous les widgets."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    bg: str = "#1e1f22"
    bg_alt: str = "#2b2d31"
    fg: str = "#e6e6e6"
    accent: str = "#5fa8d3"
    accent2: str = "#a5dd9b"
    danger: str = "#e06c75"
    warning: str = "#e5c07b"
    grid_line: str = "#3a3d42"
    obstacle: str = "#4a4f57"
    agent: str = "#5fa8d3"
    goal: str = "#a5dd9b"
    trail: str = "#3d6478"
    font_family: str = "Segoe UI"
    font_size: int = 10
    title_size: int = 13


THEME = Theme()


QSS = f"""
QMainWindow, QWidget {{
    background-color: {THEME.bg};
    color: {THEME.fg};
    font-family: "{THEME.font_family}";
    font-size: {THEME.font_size}pt;
}}
QPushButton {{
    background-color: {THEME.bg_alt};
    border: 1px solid {THEME.grid_line};
    border-radius: 4px;
    padding: 6px 12px;
}}
QPushButton:hover {{ border-color: {THEME.accent}; }}
QPushButton:pressed {{ background-color: {THEME.accent}; color: {THEME.bg}; }}
QPushButton:disabled {{ color: #777; }}
QPlainTextEdit, QTextEdit {{
    background-color: {THEME.bg_alt};
    border: 1px solid {THEME.grid_line};
    border-radius: 4px;
    font-family: "Consolas", "Courier New", monospace;
}}
QLabel#title {{
    font-size: {THEME.title_size}pt;
    font-weight: 600;
    color: {THEME.accent};
}}
"""
