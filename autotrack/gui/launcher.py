import sys
from functools import partial
from os import path
from typing import Callable, List, Iterable
from typing import Dict, Any

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QKeyEvent
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QLineEdit
from matplotlib.backend_bases import KeyEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from autotrack.core.experiment import Experiment
from autotrack.gui import APP_NAME
from autotrack.gui.application import Plugin
from autotrack.gui.gui_experiment import GuiExperiment
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


class MainWindow(Window):
    __plugins: List[Plugin]

    def __init__(self, q_window: QMainWindow, figure: Figure, experiment: GuiExperiment,
                 title_text: QLabel, status_text: QLabel):
        super().__init__(q_window, figure, experiment, title_text, status_text)
        self.__plugins = list()

    def _get_default_menu(self) -> Dict[str, Any]:
        from autotrack.gui import action

        menu_items = {
            "File/New-New project...": lambda: action.new(self),
            "File/SaveLoad-Load images...": lambda: action.load_images(self),
            "File/SaveLoad-Load tracking data...": lambda: action.load_tracking_data(self),
            "File/SaveLoad-Save tracking data...": lambda: action.save_tracking_data(self.get_experiment()),
            "File/Export-Export detection data only...": lambda: action.export_positions_and_shapes(self.get_experiment()),
            "File/Export-Export linking data only...": lambda: action.export_links(self.get_experiment()),
            "File/Export-Export to Guizela's file format...": lambda: action.export_links_guizela(self.get_experiment()),
            "File/Plugins-Reload all plugins...": lambda: action.reload_plugins(self),
            "File/Exit-Exit (Alt+F4)": lambda: action.ask_exit(self.get_experiment()),
            "Edit/Experiment-Rename experiment...": lambda: action.rename_experiment(self),
            "View/Toggle-Toggle showing axis numbers": lambda: action.toggle_axis(self.get_figure()),
        }

        for plugin in self.get_plugins():
            menu_items.update(plugin.get_menu_items(self))

        return menu_items

    def _get_help_menu(self) -> Dict[str, Any]:
        from autotrack.gui import action

        return {
            "Help/Basic-Contents...": action.show_manual,
            "Help/Basic-About": action.about_the_program,
        }

    def install_plugins(self, plugins: Iterable[Plugin]):
        """Adds the given list of plugins to the list of active plugins."""
        for plugin in plugins:
            self.__plugins.append(plugin)

    def reload_plugins(self) -> int:
        """Reloads all plugins from disk. You should update the window after calling this. Returns the number of
        reloaded plugins."""
        for plugin in self.__plugins:
            plugin.reload()
        return len(self.__plugins)

    def get_plugins(self) -> Iterable[Plugin]:
        """Gets all installed plugins. Do not modify the returned list."""
        return self.__plugins

    def set_status(self, text: str):
        # Expand to at least six lines to avoid resizing the box so much
        line_count = text.count('\n') + 1
        while line_count < 6:
            text = "\n" + text
            line_count += 1
        super().set_status(text)

def launch_window(experiment: Experiment) -> MainWindow:
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
    main_frame = QWidget(parent=q_window)
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

    window = MainWindow(q_window, fig, GuiExperiment(experiment), title, status_box)
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

