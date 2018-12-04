from typing import List, Dict, Any, Optional, Iterable, Callable

from PyQt5.QtWidgets import QMainWindow, QMenuBar, QLabel, QAction
from matplotlib.figure import Figure

from autotrack.core.experiment import Experiment
from autotrack.gui import dialog, APP_NAME
from autotrack.gui.application import Application
from autotrack.gui.gui_experiment import GuiExperiment
from autotrack.gui.threading import Scheduler
from autotrack.gui.undo_redo import UndoRedo


class Window:
    """The model for a window."""
    __q_window: QMainWindow

    __fig: Figure
    __status_text: QLabel
    __title_text: QLabel
    __gui_experiment: GuiExperiment
    __application: Application
    __event_handler_ids: List[int]
    __menu: QMenuBar

    def __init__(self, q_window: QMainWindow, application: Application, figure: Figure, experiment: GuiExperiment,
                 title_text: QLabel, status_text: QLabel):
        self.__q_window = q_window
        self.__application = application
        self.__menu = q_window.menuBar()
        self.__fig = figure
        self.__gui_experiment = experiment
        self.__status_text = status_text
        self.__title_text = title_text
        self.__event_handler_ids = list()

    def _event_source(self) -> str:
        """Returns an identifier used to register and unregister events."""
        return f"window_{id(self)}"

    def register_event_handler(self, event: str, action: Callable):
        """Registers an event handler, whichs allows you to react on things like "data changed". Supported events:

        * All matplotlib events.
        * All GuiExperiment event: GuiExperiment.KNOWN_EVENTS

        Don't forget to unregister your event handlers when its no longer needed! Otherwise you'll have a memory leak.
        """
        if event in GuiExperiment.KNOWN_EVENTS:
            self.__gui_experiment.register_event_handler(event, self._event_source(), action)
        else:
            # Matplotlib events are handled differently
            self.__event_handler_ids.append(self.__fig.canvas.mpl_connect(event, action))

    def unregister_event_handlers(self):
        """Unregisters all handlers registered using register_event_handler"""
        for id in self.__event_handler_ids:
            self.__fig.canvas.mpl_disconnect(id)
        self.__event_handler_ids.clear()
        self.__gui_experiment.unregister_event_handlers(self._event_source())

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
            self.__q_window.setWindowTitle(APP_NAME)
        else:
            self.__q_window.setWindowTitle(APP_NAME + " - " + text)

    def get_experiment(self) -> Experiment:
        """Gets the experiment that is being shown."""
        return self.__gui_experiment.experiment

    def set_experiment(self, experiment: Experiment):
        """Replaces the experiment that is being shown. The undo/redo queue is cleared automatically. You'll likely want
        to call refresh() after calling this."""
        self.__gui_experiment.set_experiment(experiment)

    def redraw_data(self):
        """Redraws the main figure using the latest values from the experiment."""
        self.__gui_experiment.redraw_data()

    def redraw_image_and_data(self):
        """Redraws the image using the latest values from the experiment."""
        self.__gui_experiment.redraw_image_and_data()

    def setup_menu(self, extra_items: Dict[str, any]):
        """Update the main menu of the window to contain the given options."""
        menu_items = self._get_default_menu()
        for plugin in self.__application.get_plugins():
            menu_items.update(plugin.get_menu_items(self))
        menu_items.update(extra_items)
        menu_items.update(_get_help_menu())  # This menu must come last
        _update_menu(self.__q_window, self.__menu, menu_items)

    def get_application(self) -> Application:
        return self.__application

    def get_undo_redo(self) -> UndoRedo:
        """Gets the undo/redo handler. Any actions performed through this handler can be undone."""
        return self.__gui_experiment.undo_redo

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
            "File/Plugins-Reload all plugins...": lambda: action.reload_plugins(self),
            "File/Exit-Exit (Alt+F4)": lambda: action.ask_exit(self.get_experiment()),
            "Edit/Experiment-Rename experiment...": lambda: action.rename_experiment(self),
            "View/Toggle-Toggle showing axis numbers": lambda: action.toggle_axis(self.get_figure()),
        }

    def execute_command(self, command: str):
        """Calls all registered command handlers with the given argument. Used when a user entered a command."""
        self.__gui_experiment.execute_command(command)


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
