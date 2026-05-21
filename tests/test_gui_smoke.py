"""Smoke tests GUI — instancie chaque widget sans crash (offscreen)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    return app


def test_gridworld_view_smoke(qapp) -> None:
    from mw_ia.config import DEFAULT
    from mw_ia.envs.gridworld import GridWorld
    from mw_ia.gui.widgets.gridworld_view import GridWorldView
    env = GridWorld(DEFAULT.gridworld)
    view = GridWorldView(env)
    view.update_state((1, 1))
    view.reset_view()


def test_live_plots_smoke(qapp) -> None:
    from mw_ia.gui.widgets.live_plots import LivePlots
    w = LivePlots()
    w.reward.push(0, 0.5)
    w.loss.push(0, 0.2)
    w.epsilon.push(0, 0.9)
    w.winrate.push(0, 0.1)
    w.reset()


def test_stats_panel_smoke(qapp) -> None:
    from mw_ia.gui.widgets.stats_panel import StatsPanel
    from mw_ia.training.metrics import Level
    p = StatsPanel()
    p.update_stats(episode=10, reward=0.5, best=0.9, loss=0.123,
                   eps=0.5, winrate=0.42, level=Level.INTERMEDIATE)


def test_control_panel_smoke(qapp) -> None:
    from mw_ia.gui.widgets.control_panel import ControlPanel
    p = ControlPanel()
    p.set_running(False)
    p.set_running(True)


def test_log_console_smoke(qapp) -> None:
    from mw_ia.gui.widgets.log_console import LogConsole
    c = LogConsole()
    c.append("info", "hello")
    c.append("warning", "warn")
    c.append("error", "boom")


def test_main_window_smoke(qapp) -> None:
    from mw_ia.config import DEFAULT
    from mw_ia.gui.app import MainWindow
    w = MainWindow(DEFAULT)
    w.show()
    qapp.processEvents()
    w.close()
