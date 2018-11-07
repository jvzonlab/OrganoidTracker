import sys
from functools import partial
from os import path
from typing import List, Dict, Any, Optional, Iterable, Callable

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QKeyEvent
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QMenuBar, QMenu, QAction, QVBoxLayout, QLabel, QLineEdit
from matplotlib.backend_bases import KeyEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from autotrack.core.experiment import Experiment
from autotrack.gui import dialog
from autotrack.gui.threading import Scheduler

APP_NAME = "Autotrack"


class Plugin:
    """
    Represents a plugin. Plugins can add new data visualizers or provide support for more file types.
    Instead of writing a plugin, you can also write a script that uses the classes in core and imaging.
    """

    def get_menu_items(self, window: "Window")-> Dict[str, Any]:
        """
        Used to add menu items that must always be visible. Example:

            return {
                "File/Import-Import my format...": lambda: my_code_here(),
                "View/Analysis-Useful analysis screen here...": lambda: my_other_code_here()
            }
        """
        return {}


class Window:
    """The model for a window."""
    __root: QMainWindow
    __scheduler: Scheduler
    __plugins: List[Plugin]

    __fig: Figure
    __status_text: QLabel
    __title_text: QLabel
    __experiment: Experiment

    __event_handler_ids: List[int]
    __refresh_handlers: List[Any]
    __command_handlers: List[Any]
    __menu: QMenuBar

    def __init__(self, root: QMainWindow, menu: QMenuBar, figure: Figure, experiment: Experiment,
                 title_text: QLabel, status_text: QLabel):
        self.__root = root
        self.__menu = menu
        self.__fig = figure
        self.__experiment = experiment
        self.__status_text = status_text
        self.__title_text = title_text
        self.__event_handler_ids = []
        self.__refresh_handlers = []
        self.__command_handlers = []
        self.__plugins = []

        self.__scheduler = Scheduler(root)
        self.__scheduler.daemon = True
        self.__scheduler.start()

    def register_event_handler(self, event: str, action):
        """Registers an event handler. Supported events:

        * All matplotlib events.
        * "refresh_event" for when the figure needs to be redrawn.
        * "command_event" for when a command is executed
        """
        if event == "refresh_event":
            self.__refresh_handlers.append(action)
        elif event == "command_event":
            self.__command_handlers.append(action)

        self.__event_handler_ids.append(self.__fig.canvas.mpl_connect(event, action))

    def unregister_event_handlers(self):
        """Unregisters all handles registered using register_event_handler"""
        for id in self.__event_handler_ids:
            self.__fig.canvas.mpl_disconnect(id)
        self.__event_handler_ids = []
        self.__refresh_handlers.clear()
        self.__command_handlers.clear()
        self.setup_menu(dict())

    def get_figure(self) -> Figure:
        """Gets the Matplotlib figure."""
        return self.__fig

    def set_status(self, text: str):
        """Sets the small text below the figure."""

        # Expand to at least six lines to avoid resizing the box so much
        line_count = text.count('\n') + 1
        while line_count < 6:
            text = "\n" + text
            line_count += 1

        self.__status_text.setText(text)

    def set_figure_title(self, text: str):
        """Sets the big text above the main figure."""
        self.__title_text.setText(text)

    def set_window_title(self, text: Optional[str]):
        """Sets the title of the window, prefixed by APP_NAME. Use None as the title to just sown APP_NAME."""
        if text is None:
            self.__root.setWindowTitle(APP_NAME)
        else:
            self.__root.setWindowTitle(APP_NAME + " - " + text)

    def get_experiment(self) -> Experiment:
        """Gets the experiment that is being shown."""
        return self.__experiment

    def set_experiment(self, experiment: Experiment):
        """Replaces the experiment that is being shown. You'll likely want to call refresh() after calling this."""
        self.__experiment = experiment

    def refresh(self):
        """Redraws the main figure."""
        for refresh_handler in self.__refresh_handlers:
            refresh_handler()

    def setup_menu(self, extra_items: Dict[str, any]):
        """Update the main menu of the window to contain the given options."""
        menu_items = self._get_default_menu()
        for plugin in self.__plugins:
            menu_items.update(plugin.get_menu_items(self))
        menu_items.update(extra_items)
        menu_items.update(_get_help_menu())  # This menu must come last
        _update_menu(self.__root, self.__menu, menu_items)

    def get_scheduler(self) -> Scheduler:
        """Gets the scheduler, useful for registering background tasks"""
        return self.__scheduler

    def install_plugins(self, plugins: Iterable[Plugin]):
        """Adds the given list of plugins to the list of active plugins."""
        for plugin in plugins:
            self.__plugins.append(plugin)

    def _get_default_menu(self) -> Dict[str, Any]:
        from autotrack.gui import action

        return {
            "File/New-New project...": lambda: action.new(self),
            "File/SaveLoad-Load images...": lambda: action.load_images(self),
            "File/SaveLoad-Load tracking data...": lambda: action.load_tracking_data(self),
            "File/SaveLoad-Save tracking data...": lambda: action.save_tracking_data(self.get_experiment()),
            "File/Export-Export detection data only...": lambda: action.export_positions_and_shapes(self.get_experiment()),
            "File/Export-Export linking data only...": lambda: action.export_links(self.get_experiment()),
            "File/Export-Export to Guizela's file format...": lambda: action.export_links_guizela(self.get_experiment()),
            "File/Exit-Exit (Alt+F4)": lambda: action.ask_exit(self.__root),
            "Edit/Add-Add positions and shapes...": lambda: action.add_positions(self),
            "Edit/Add-Add links, scores and warnings...": lambda: action.add_links(self),
            "Edit/Add-Add positions and links from Guizela's format...": lambda: action.add_guizela_tracks(self),
            "Edit/Experiment-Rename experiment...": lambda: action.rename_experiment(self),
            "View/Toggle-Toggle showing axis numbers": lambda: action.toggle_axis(self.get_figure()),
        }

    def execute_command(self, command: str):
        """Calls all registered command handlers with the given argument. Used when a user entered a command."""
        for command_handler in self.__command_handlers:
            command_handler(command)


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


def _simple_menu_dict_to_nested(menu_items: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    menu_tree = {   # Forced order of base menus - these must go first
        "File": {}, "Edit": {}, "View": {}
    }

    for name, action in menu_items.items():
        slash_index = name.index("/")
        main_menu_name, sub_name = name[0:slash_index], name[slash_index + 1:]
        dash_index = sub_name.index("-")
        category_name, label = sub_name[0:dash_index], sub_name[dash_index + 1:]

        if main_menu_name not in menu_tree:
            menu_tree[main_menu_name] = dict()
        categories = menu_tree[main_menu_name]
        if category_name not in categories:
            categories[category_name] = dict()
        category = categories[category_name]
        category[label] = action

    return menu_tree


def _get_help_menu() -> Dict[str, Any]:
    from autotrack.gui import action

    return {
        "Help/Basic-Contents...": action.show_manual,
        "Help/Basic-About": action.about_the_program,
    }


def _update_menu(q_window: QMainWindow, menu_bar: QMenuBar, menu_items: Dict[str, Any]):
    from autotrack.gui import action

    menu_tree = _simple_menu_dict_to_nested(menu_items)

    menu_bar.clear()  # Remove old menu bar

    for menu_name, dropdown_items in menu_tree.items():
        # Create each dropdown menu
        if not dropdown_items:
            continue  # Ignore empty menus
        menu = menu_bar.addMenu(menu_name)
        first_category = True
        for category_items in dropdown_items.values():
            if not first_category:
                menu.addSeparator()
            else:
                first_category = False

            for item_name, item_action in category_items.items():
                action = QAction(item_name, q_window)
                action.triggered.connect(_with_safeguard(item_action))
                menu.addAction(action)


def _with_safeguard(action):
    """Adds an exception handler to the specified lambda action"""
    def safeguard():
        try:
            action()
        except Exception as e:
            dialog.popup_exception(e)
    return safeguard


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

    menu = q_window.menuBar()

    # Initialize main grid
    main_frame = QtWidgets.QWidget(parent=q_window)
    q_window.setCentralWidget(main_frame)
    vertical_boxes = QVBoxLayout(main_frame)

    # Add title
    title = QLabel(parent=main_frame)
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

    window = Window(q_window, menu, fig, experiment, title, status_box)
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


