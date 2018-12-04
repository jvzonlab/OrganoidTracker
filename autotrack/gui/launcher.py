import sys
from functools import partial
from os import path
from typing import List, Dict, Any, Optional, Iterable, Callable

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QKeyEvent
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QMenuBar, QMenu, QAction, QVBoxLayout, QLabel, QLineEdit
from matplotlib.backend_bases import KeyEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from autotrack.core.experiment import Experiment
from autotrack.gui import dialog, APP_NAME
from autotrack.gui.application import Application
from autotrack.gui.gui_experiment import GuiExperiment
from autotrack.gui.threading import Scheduler
from autotrack.gui.window import Window


class _CommandBox(QLineEdit):
    enter_handler: Callable = None
    escape_handler: Callable = None

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_Enter or key == Qt.Key_Return:
            self.enter_handler(self.text())
            self.setText("")
        elif key == Qt.Key_Escape:
            self.escape_handler()
        else:
            super().keyPressEvent(event)


def launch_window(experiment: Experiment) -> Window:
    """Launches a window with an empty figure. Doesn't start the main loop yet. Use and activate a visualizer to add
    some interactiveness."""
    # Create matplotlib figure
    fig = Figure(figsize=(12, 12), dpi=95)

    # Create empty window
    root = QApplication.instance()
    if not root:
        root = QApplication(sys.argv)
    q_window = QMainWindow()
    q_window.setBaseSize(800, 700)
    q_window.setWindowTitle(APP_NAME)
    q_window.setWindowIcon(QIcon(path.join(path.dirname(path.abspath(sys.argv[0])), 'autotrack', 'gui', 'icon.ico')))

    # Initialize main grid
    main_frame = QtWidgets.QWidget(parent=q_window)
    q_window.setCentralWidget(main_frame)
    vertical_boxes = QVBoxLayout(main_frame)

    # Add title
    title = QLabel(parent=main_frame)
    title.setStyleSheet("font-size: 16pt; font-weight: bold")
    vertical_boxes.addWidget(title)

    # Add Matplotlib figure to frame
    mpl_canvas = FigureCanvasQTAgg(fig)  # A tk.DrawingArea.
    mpl_canvas.setParent(main_frame)
    mpl_canvas.setFocusPolicy(Qt.ClickFocus)
    mpl_canvas.setFocus()
    vertical_boxes.addWidget(mpl_canvas)

    # Add status bar
    status_box = QLabel(parent=main_frame)
    vertical_boxes.addWidget(status_box)

    # Add command box
    command_box = _CommandBox(parent=main_frame)
    vertical_boxes.addWidget(command_box)

    mpl_canvas.mpl_connect("key_release_event", partial(_commandbox_autofocus, command_box=command_box))

    toolbar = NavigationToolbar2QT(mpl_canvas, q_window)
    q_window.addToolBar(toolbar)

    application = Application(q_window)
    window = Window(q_window, application, fig, GuiExperiment(experiment), title, status_box)
    command_box.escape_handler = lambda: mpl_canvas.setFocus()
    command_box.enter_handler = partial(_commandbox_execute, window=window, main_figure=mpl_canvas)

    window.setup_menu(dict())  # This draws the menu

    q_window.show()
    return window


def _commandbox_execute(command: str, window: Window, main_figure: QWidget):
    if command.startswith("/"):
        command = command[1:]  # Strip off the command slash
    main_figure.setFocus()
    window.execute_command(command)


def _commandbox_autofocus(event: KeyEvent, command_box: QLineEdit):
    """Switches focus to command box if "/" is pressed while the figure is in focus."""
    if event.key == "/":
        command_box.setFocus()
        command_box.setText("/")
        command_box.setCursorPosition(1)


def mainloop():
    """Starts the main loop."""
    sys.exit(QApplication.instance().exec_())

