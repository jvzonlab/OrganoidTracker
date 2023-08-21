import sys
from functools import partial
from typing import Callable, List, Iterable, Optional
from typing import Dict, Any

from PySide2.QtCore import Qt
from PySide2.QtGui import QIcon, QKeyEvent, QPalette, QCloseEvent
from PySide2.QtWidgets import QMainWindow, QSizePolicy
from PySide2.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QLineEdit
from matplotlib import pyplot
from matplotlib.backend_bases import KeyEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import APP_NAME
from organoid_tracker.plugin.instance import Plugin
from organoid_tracker.gui.gui_experiment import GuiExperiment
from organoid_tracker.gui.icon_getter import get_icon
from organoid_tracker.gui.toolbar import Toolbar
from organoid_tracker.gui.window import Window
from organoid_tracker.plugin.plugin_manager import PluginManager


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
    toolbar: Toolbar
    status_box: QLabel
    mpl_canvas: FigureCanvasQTAgg
    close_handler: Optional[Callable[[QCloseEvent], None]] = None

    def __init__(self, figure: Figure):
        super().__init__()
        self.setBaseSize(800, 700)
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(get_icon("icon.ico"))

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

        self.toolbar = Toolbar(self.mpl_canvas, self)
        self.addToolBar(self.toolbar)

    def closeEvent(self, event: QCloseEvent):
        try:
            if self.close_handler is not None:
                self.close_handler(event)
        except BaseException as e:
            from organoid_tracker.gui import dialog
            dialog.popup_exception(e)


class MainWindow(Window):

    def __init__(self, q_window: QMainWindow, figure: Figure, experiment: GuiExperiment,
                 title_text: QLabel, status_text: QLabel):
        super().__init__(q_window, figure, experiment, title_text, status_text)

    def _get_default_menu(self) -> Dict[str, Any]:
        from organoid_tracker.gui import action

        menu_items = {
            "File//Project-New project... [Ctrl+N]": lambda: action.new(self),
            "File//SaveLoad-Load images... [Ctrl+I]": lambda: action.load_images(self),
            "File//SaveLoad-Load tracking data... [Ctrl+O]": lambda: action.load_tracking_data(self),
            "File//SaveLoad-Save tracking data [Ctrl+S]": lambda: action.save_tracking_data(self),
            "File//SaveLoad-Save tracking data as... [Ctrl+Shift+S]": lambda: action.save_tracking_data(self, force_save_as=True),
            "File//Export-Export positions//JSON, as pixel coordinates...": lambda: action.export_positions(self.get_experiment()),
            "File//Export-Export links//Guizela's file format...": lambda: action.export_links_guizela(self.get_experiment()),
            "File//Export-Export links//Cell Tracking Challenge format...": lambda: action.export_links_ctc(self.get_experiment()),
            "File//Export-Export links//TrackMate format...": lambda: action.export_links_trackmate(self.get_experiment()),
            "Edit//Experiment-Rename experiment...": lambda: action.rename_experiment(self),
            "Edit//Experiment-Set image resolution...": lambda: action.set_image_resolution(self),
            "View//Toggle-Toggle showing axis numbers": lambda: action.toggle_axis(self.get_figure()),
            "View//Statistics-View statistics...": lambda: action.view_statistics(self),
        }

        return menu_items

    def _get_plugins_menu(self) -> Dict[str, Any]:
        menu_items = dict()
        for plugin in self.plugin_manager.get_plugins():
            menu_items.update(plugin.get_menu_items(self))
        return menu_items

    def _get_last_default_menu(self) -> Dict[str, Any]:
        from organoid_tracker.gui import action

        return {
            "File//Exit-Close experiment": lambda: action.close_experiment(self),
            "File//Exit-Exit [Alt+F4]": lambda: action.ask_exit(self.get_gui_experiment()),
            "Help//Basic-Contents... [F1]": action.show_manual,
            "Help//Basic-About": action.about_the_program,
        }

    def set_status(self, text: str):
        # Expand to at least six lines to avoid resizing the box so much
        line_count = text.count('\n') + 1
        while line_count < 6:
            text = "\n" + text
            line_count += 1
        super().set_status(text)


def _window_close(window: Window, event: QCloseEvent):
    """Asks if the user wants to save before closing the program."""
    event.ignore()
    from organoid_tracker.gui import action
    if action.ask_save_unsaved_changes(window.get_gui_experiment().get_all_tabs()):
        event.setAccepted(True)


def launch_window(experiment: Experiment) -> MainWindow:
    """Launches a window with an empty figure. Doesn't start the main loop yet. Use and activate a visualizer to add
    some interactiveness."""
    pyplot.rcParams['svg.fonttype'] = 'none'
    pyplot.rcParams["font.family"] = "Arial, sans-serif"
    pyplot.rcParams["xtick.direction"] = "in"
    pyplot.rcParams["ytick.direction"] = "in"

    # Create matplotlib figure
    fig = Figure(figsize=(12, 12), dpi=95)
    fig.subplots_adjust(left=0.04, bottom=0.04, right=0.99, top=0.98)

    # Create empty window
    root = QApplication.instance()
    if not root:
        root = QApplication(sys.argv)
    q_window = _MyQMainWindow(fig)
    window = MainWindow(q_window, fig, GuiExperiment(experiment), q_window.title, q_window.status_box)

    q_window.command_box.enter_handler = partial(_commandbox_execute, window=window, main_figure=q_window.mpl_canvas)
    q_window.close_handler = lambda close_event: _window_close(window, close_event)
    _connect_toolbar_actions(q_window.toolbar, window)

    q_window.show()
    return window


def _connect_toolbar_actions(toolbar: Toolbar, window: Window):
    def new(*args):
        from organoid_tracker.gui import action
        action.new(window)
    def close(*args):
        from organoid_tracker.gui import action
        action.close_experiment(window)
    def home(*args):
        window.get_gui_experiment().execute_command("exit")
    def save(*args):
        from organoid_tracker.gui import action
        action.save_tracking_data(window)
    def load(*args):
        from organoid_tracker.gui import action
        action.load_tracking_data(window)
    def image(*args):
        from organoid_tracker.gui import action
        action.load_images(window)
    def experiment(index):
        window.get_gui_experiment().select_experiment(index)
    def update_experiment_list(*args):
        # Update the list of experiment names in the top right corner, in case experiment.name has changed
        selected_index = 0
        experiment_names = []
        for i, experiment_name in window.get_gui_experiment().get_selectable_experiments():
            if window.get_gui_experiment().is_selected(i):
                selected_index = i
            experiment_names.append(experiment_name)
        toolbar.update_selectable_experiments(experiment_names, selected_index)
    toolbar.new_handler = new
    toolbar.close_handler = close
    toolbar.home_handler = home
    toolbar.save_handler = save
    toolbar.load_handler = load
    toolbar.image_handler = image
    toolbar.experiment_select_handler = experiment
    window.get_gui_experiment().register_event_handler("any_updated_event", "toolbar", update_experiment_list)
    window.get_gui_experiment().register_event_handler("data_updated_event", "toolbar", update_experiment_list)
    window.get_gui_experiment().register_event_handler("tab_list_updated_event", "toolbar", update_experiment_list)


def _commandbox_execute(command: str, window: Window, main_figure: QWidget):
    if command.startswith("/"):
        command = command[1:]  # Strip off the command slash
    main_figure.setFocus()
    window.get_gui_experiment().execute_command(command)


def _commandbox_autofocus(event: KeyEvent, command_box: QLineEdit):
    """Switches focus to command box if "/" is pressed while the figure is in focus."""
    if event.key == "/":
        command_box.setFocus()
        command_box.setText("/")
        command_box.setCursorPosition(1)


def mainloop():
    """Starts the main loop."""
    sys.exit(QApplication.instance().exec_())

