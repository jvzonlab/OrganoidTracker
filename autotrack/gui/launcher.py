import sys
from functools import partial
from os import path
from typing import Callable, List, Iterable, Optional
from typing import Dict, Any

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QKeyEvent, QPalette, QCloseEvent
from PyQt5.QtWidgets import QMainWindow, QSizePolicy
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


class _MyQMainWindow(QMainWindow):

    command_box: _CommandBox
    title: QLabel
    status_box: QLabel
    mpl_canvas: FigureCanvasQTAgg
    close_handler: Optional[Callable[[QCloseEvent], None]] = None

    def __init__(self, figure: Figure):
        super().__init__()
        self.setBaseSize(800, 700)
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(path.join(path.dirname(path.abspath(sys.argv[0])), 'autotrack', 'gui', 'icon.ico')))

        # Initialize main grid
        main_frame = QWidget(parent=self)
        self.setCentralWidget(main_frame)
        vertical_boxes = QVBoxLayout(main_frame)

        # Add title
        self.title = QLabel(parent=main_frame)
        self.title.setStyleSheet("font-size: 16pt; font-weight: bold")
        self.title.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed))
        vertical_boxes.addWidget(self.title)

        # Add Matplotlib figure to frame
        self.mpl_canvas = FigureCanvasQTAgg(figure)  # A tk.DrawingArea.
        self.mpl_canvas.setParent(main_frame)
        self.mpl_canvas.setFocusPolicy(Qt.ClickFocus)
        self.mpl_canvas.setFocus()
        vertical_boxes.addWidget(self.mpl_canvas)

        # Set figure background color to that of the main_frame
        background_color = main_frame.palette().color(QPalette.Background)
        figure.set_facecolor((background_color.redF(), background_color.greenF(), background_color.blueF()))

        # Add status bar
        self.status_box = QLabel(parent=main_frame)
        self.status_box.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed))
        vertical_boxes.addWidget(self.status_box)

        # Add command box
        self.command_box = _CommandBox(parent=main_frame)
        vertical_boxes.addWidget(self.command_box)

        self.mpl_canvas.mpl_connect("key_release_event", partial(_commandbox_autofocus, command_box=self.command_box))
        self.command_box.escape_handler = lambda: self.mpl_canvas.setFocus()

        toolbar = NavigationToolbar2QT(self.mpl_canvas, self)
        self.addToolBar(toolbar)

    def closeEvent(self, event: QCloseEvent):
        try:
            if self.close_handler is not None:
                self.close_handler(event)
        except BaseException as e:
            from autotrack.gui import dialog
            dialog.popup_exception(e)


class MainWindow(Window):
    __plugins: List[Plugin]

    def __init__(self, q_window: QMainWindow, figure: Figure, experiment: GuiExperiment,
                 title_text: QLabel, status_text: QLabel):
        super().__init__(q_window, figure, experiment, title_text, status_text)
        self.__plugins = list()

    def _get_default_menu(self) -> Dict[str, Any]:
        from autotrack.gui import action

        menu_items = {
            "File//New-New project...": lambda: action.new(self),
            "File//SaveLoad-Load images...": lambda: action.load_images(self),
            "File//SaveLoad-Load tracking data...": lambda: action.load_tracking_data(self),
            "File//SaveLoad-Save tracking data...": lambda: action.save_tracking_data(self.get_gui_experiment()),
            "File//Export-Export detection data only...": lambda: action.export_positions_and_shapes(self.get_experiment()),
            "File//Export-Export linking data only...": lambda: action.export_links(self.get_experiment()),
            "File//Export-Export to Guizela's file format...": lambda: action.export_links_guizela(self.get_experiment()),
            "File//Plugins-Reload all plugins...": lambda: action.reload_plugins(self),
            "File//Exit-Exit (Alt+F4)": lambda: action.ask_exit(self.get_gui_experiment()),
            "Edit//Experiment-Rename experiment...": lambda: action.rename_experiment(self),
            "View//Toggle-Toggle showing axis numbers": lambda: action.toggle_axis(self.get_figure()),
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


def _window_close(window: Window, event: QCloseEvent):
    """Asks if the user wants to save before closing the program."""
    if not window.get_gui_experiment().undo_redo.has_unsaved_changes():
        return

    event.ignore()
    from autotrack.gui import dialog, action
    answer = dialog.prompt_yes_no_cancel("Confirmation", "You have unsaved changes. Do you want to save those"
                                                         " first?")
    if answer.is_yes():
        if action.save_tracking_data(window.get_gui_experiment()):
            event.setAccepted(True)
    elif answer.is_no():
        event.setAccepted(True)


def launch_window(experiment: Experiment) -> MainWindow:
    """Launches a window with an empty figure. Doesn't start the main loop yet. Use and activate a visualizer to add
    some interactiveness."""
    # Create matplotlib figure
    fig = Figure(figsize=(12, 12), dpi=95)

    # Create empty window
    root = QApplication.instance()
    if not root:
        root = QApplication(sys.argv)
    q_window = _MyQMainWindow(fig)
    window = MainWindow(q_window, fig, GuiExperiment(experiment), q_window.title, q_window.status_box)

    q_window.command_box.enter_handler = partial(_commandbox_execute, window=window, main_figure=q_window.mpl_canvas)
    q_window.close_handler = lambda close_event: _window_close(window, close_event)

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

