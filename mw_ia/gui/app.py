"""Fenêtre principale MW_IA + thread Qt d'entraînement."""
from __future__ import annotations

from pathlib import Path

import torch
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from mw_ia.config import Config, DQNConfig, ProceduralEnvConfig, SchedulerConfig
from mw_ia.envs.gridworld import GridWorld
from mw_ia.envs.maze_generators import RandomObstaclesGenerator
from mw_ia.envs.procedural_env import ProceduralGridWorld
from mw_ia.gui.theme import QSS, THEME
from mw_ia.gui.widgets.control_panel import ControlPanel
from mw_ia.gui.widgets.difficulty_label import DifficultyLabel
from mw_ia.gui.widgets.gridworld_view import GridWorldView
from mw_ia.gui.widgets.live_plots import LivePlots
from mw_ia.gui.widgets.log_console import LogConsole
from mw_ia.gui.widgets.stats_panel import StatsPanel
from mw_ia.training.runner import DQNRunner, ProceduralDQNRunner, RunnerCallbacks


class TrainingThread(QThread):
    """QThread mince — émet des signaux que la MainWindow consomme."""

    step_signal = pyqtSignal(tuple, int, float, tuple)
    episode_signal = pyqtSignal(int, float, int, bool)
    loss_signal = pyqtSignal(int, float)
    epsilon_signal = pyqtSignal(int, float)
    log_signal = pyqtSignal(str, str)
    finished_signal = pyqtSignal()
    maze_changed_signal = pyqtSignal(object, int, float)
    difficulty_signal = pyqtSignal(float, int)

    def __init__(self, runner: DQNRunner) -> None:
        super().__init__()
        self.runner = runner
        self.runner.callbacks = RunnerCallbacks(
            on_step=self._on_step,
            on_episode=self._on_episode,
            on_loss=self._on_loss,
            on_epsilon=self._on_epsilon,
            on_log=self._on_log,
            on_maze_changed=self._on_maze_changed,
            on_difficulty_updated=self._on_difficulty_updated,
        )

    def _on_step(self, *, state, action, reward, next_state):
        self.step_signal.emit(state, int(action), float(reward), next_state)

    def _on_episode(self, *, ep, reward, length, success):
        self.episode_signal.emit(int(ep), float(reward), int(length), bool(success))

    def _on_loss(self, step: int, loss: float) -> None:
        self.loss_signal.emit(int(step), float(loss))

    def _on_epsilon(self, step: int, eps: float) -> None:
        self.epsilon_signal.emit(int(step), float(eps))

    def _on_log(self, level: str, msg: str) -> None:
        self.log_signal.emit(level, msg)

    def _on_maze_changed(self, *, maze, episode_id, difficulty):
        self.maze_changed_signal.emit(maze, int(episode_id), float(difficulty))

    def _on_difficulty_updated(self, *, difficulty, episode_id):
        self.difficulty_signal.emit(float(difficulty), int(episode_id))

    def run(self) -> None:
        self.runner.run()
        self.finished_signal.emit()

    def request_pause(self, paused: bool) -> None:
        self.runner.request_pause(paused)

    def request_stop(self) -> None:
        self.runner.request_stop()


class MainWindow(QMainWindow):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.setWindowTitle("MW_IA — Reinforcement Learning")
        self.resize(1280, 800)
        self.setStyleSheet(QSS)
        self.config = config
        self.env = GridWorld(config.gridworld)
        self.thread: TrainingThread | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top = QHBoxLayout()
        self.gridview = GridWorldView(self.env)
        self.stats = StatsPanel()
        self.difficulty_label = DifficultyLabel()
        self.plots = LivePlots()
        top.addWidget(self.gridview, 0)
        # Stats + label difficulty empilés verticalement
        stats_col = QVBoxLayout()
        stats_col.addWidget(self.stats)
        stats_col.addWidget(self.difficulty_label)
        stats_col.addStretch(1)
        stats_widget = QWidget()
        stats_widget.setLayout(stats_col)
        top.addWidget(stats_widget, 0)
        top.addWidget(self.plots, 1)
        root.addLayout(top, 3)

        self.controls = ControlPanel()
        root.addWidget(self.controls)
        self.log = LogConsole()
        root.addWidget(self.log, 1)

        self.controls.start_clicked.connect(self.on_start)
        self.controls.start_procedural_clicked.connect(self.on_start_procedural)
        self.controls.pause_clicked.connect(self.on_pause)
        self.controls.reset_clicked.connect(self.on_reset)
        self.controls.save_clicked.connect(self.on_save)
        self.controls.load_clicked.connect(self.on_load)

        self.log.append("info", f"Theme: {THEME.font_family} — Device CUDA: {torch.cuda.is_available()}")

    @pyqtSlot()
    def on_start(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        runner = DQNRunner(
            self.env, self.config.dqn, self.config.training,
            callbacks=RunnerCallbacks(), device=device,
            seed=self.config.training.seed,
        )
        self.thread = TrainingThread(runner)
        self.thread.step_signal.connect(self._on_step)
        self.thread.episode_signal.connect(self._on_episode)
        self.thread.loss_signal.connect(self._on_loss)
        self.thread.epsilon_signal.connect(self._on_epsilon)
        self.thread.log_signal.connect(self.log.append)
        self.thread.finished_signal.connect(self._on_finished)
        self.thread.maze_changed_signal.connect(self.gridview.on_maze_changed)
        self.thread.difficulty_signal.connect(self._on_difficulty)
        self.controls.set_running(True)
        self.thread.start()

    @pyqtSlot()
    def on_start_procedural(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc_cfg = ProceduralEnvConfig(mode="obstacles")
        gen = RandomObstaclesGenerator(
            rows=proc_cfg.max_rows, cols=proc_cfg.max_cols,
            start=(0, 0), goal=(proc_cfg.max_rows - 1, proc_cfg.max_cols - 1),
            min_density=proc_cfg.min_density, max_density=proc_cfg.max_density,
        )
        proc_env = ProceduralGridWorld(cfg=proc_cfg, generator=gen)
        # Recette V2-X gagnante : hidden=(256,256), epsilon_decay_steps=200000.
        # NE PAS utiliser self.config.dqn (defaults V1 cassés en procedural :
        # decay=50000 trop court, hidden=128x128 sous-capacitaire). Voir
        # CLAUDE.md §"V2-X — recette opérationnelle" pour la justification.
        procedural_dqn_cfg = DQNConfig(
            hidden_layers=(256, 256),
            epsilon_decay_steps=200_000,
            episodes=self.config.dqn.episodes,
        )
        runner = ProceduralDQNRunner(
            env=proc_env, proc_cfg=proc_cfg, dqn_cfg=procedural_dqn_cfg,
            sched_cfg=SchedulerConfig(), train_cfg=self.config.training,
            callbacks=RunnerCallbacks(), device=device,
            seed=self.config.training.seed,
        )
        self.thread = TrainingThread(runner)
        self.thread.step_signal.connect(self._on_step)
        self.thread.episode_signal.connect(self._on_episode)
        self.thread.loss_signal.connect(self._on_loss)
        self.thread.epsilon_signal.connect(self._on_epsilon)
        self.thread.log_signal.connect(self.log.append)
        self.thread.finished_signal.connect(self._on_finished)
        # Wiring procedural-spécifique
        self.thread.maze_changed_signal.connect(self.gridview.on_maze_changed)
        self.thread.maze_changed_signal.connect(self.difficulty_label.on_maze_changed)
        self.thread.difficulty_signal.connect(self._on_difficulty)
        self.controls.set_running(True)
        self.thread.start()

    @pyqtSlot(bool)
    def on_pause(self, paused: bool) -> None:
        if self.thread:
            self.thread.request_pause(paused)
            self.log.append("info", "Pause" if paused else "Reprise")

    @pyqtSlot()
    def on_reset(self) -> None:
        if self.thread and self.thread.isRunning():
            self.thread.request_stop()
            self.thread.wait(2000)
        self.env.reset()
        self.gridview.reset_view()
        self.plots.reset()
        self.log.append("info", "Réinitialisation OK")

    @pyqtSlot()
    def on_save(self) -> None:
        if not self.thread or self.thread.runner.agent is None:
            QMessageBox.warning(self, "Sauvegarde", "Aucun agent en mémoire.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Sauvegarder l'agent",
                                              "checkpoints/dqn.pt",
                                              "PyTorch checkpoint (*.pt)")
        if path:
            self.thread.runner.agent.save(Path(path))
            self.log.append("info", f"Sauvegardé : {path}")

    @pyqtSlot()
    def on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Charger un agent",
                                              "checkpoints/",
                                              "PyTorch checkpoint (*.pt)")
        if not path:
            return
        from mw_ia.agents.dqn import DQNAgent
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = DQNAgent(self.env, self.config.dqn, device=device,
                         seed=self.config.training.seed)
        agent.load(Path(path))
        self.log.append("info", f"Chargé : {path} (steps={agent.global_step})")

    @pyqtSlot(tuple, int, float, tuple)
    def _on_step(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        self.gridview.update_state(next_state)

    @pyqtSlot(int, float, int, bool)
    def _on_episode(self, ep: int, reward: float, length: int, success: bool) -> None:
        if self.thread is None:
            return
        m = self.thread.runner.metrics
        self.plots.reward.push(ep, reward)
        self.plots.winrate.push(ep, m.winrate())
        self.stats.update_stats(
            episode=ep, reward=reward, best=m.best_reward,
            loss=m.last_loss, eps=m.last_epsilon or 0.0,
            winrate=m.winrate(), level=m.level(),
        )

    @pyqtSlot(int, float)
    def _on_loss(self, step: int, loss: float) -> None:
        self.plots.loss.push(step, loss)

    @pyqtSlot(int, float)
    def _on_epsilon(self, step: int, eps: float) -> None:
        self.plots.epsilon.push(step, eps)

    @pyqtSlot(float, int)
    def _on_difficulty(self, difficulty: float, episode_id: int) -> None:
        self.plots.difficulty.push(episode_id, difficulty)

    @pyqtSlot()
    def _on_finished(self) -> None:
        self.controls.set_running(False)
        self.log.append("info", "Entraînement terminé")

    def closeEvent(self, event) -> None:
        if self.thread and self.thread.isRunning():
            self.thread.request_stop()
            self.thread.wait(3000)
        super().closeEvent(event)
