import re
from typing import Callable, Dict, List, Any, Optional, Union, Iterable, Tuple

from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QMainWindow, QLabel, QMenuBar, QAction, QMenu, QShortcut
from matplotlib.figure import Figure

from autotrack.core.experiment import Experiment
from autotrack.gui import APP_NAME
from autotrack.gui.gui_experiment import GuiExperiment
from autotrack.gui.threading import Scheduler
from autotrack.gui.undo_redo import UndoRedo, UndoableAction


class Window:
    """The model for a window."""
    __q_window: QMainWindow

    __fig: Figure
    __status_text: QLabel
    __title_text: QLabel
    __gui_experiment: GuiExperiment
    __event_handler_ids: List[int]
    __menu: QMenuBar
    __scheduler: Optional[Scheduler] = None

    def __init__(self, q_window: QMainWindow, figure: Figure, experiment: GuiExperiment,
                 title_text: QLabel, status_text: QLabel):
        self.__q_window = q_window
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

    def get_scheduler(self) -> Scheduler:
        if self.__scheduler is None:
            self.__scheduler = Scheduler()
            self.__scheduler.daemon = True
            self.__scheduler.start()
        return self.__scheduler

    def set_status(self, text: str):
        """Sets the small text below the figure."""
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

    def get_experiments(self) -> Iterable[Experiment]:
        return self.__gui_experiment.get_experiments()

    def get_gui_experiment(self) -> GuiExperiment:
        """Gets the GUI experiment, which stores the experiment along with undo/redo data, and some other non-saveable
        data."""
        return self.__gui_experiment

    def perform_data_action(self, action: UndoableAction):
        """Performs an action that will change the experiment data. The action will be executed, then the window will
        be redrawn with an updated status bar."""
        status = self.get_undo_redo().do(action, self.get_experiment())
        self.redraw_data()
        self.set_status(status)

    def redraw_data(self):
        """Redraws the main figure using the latest values from the experiment."""
        self.__gui_experiment.redraw_data()

    def redraw_all(self):
        """Redraws the image using the latest values from the experiment."""
        self.__gui_experiment.redraw_image_and_data()

    def setup_menu(self, extra_items: Dict[str, any], *, show_plugins: bool):
        """Update the main menu of the window to contain the given options."""
        menu_items = self._get_default_menu()
        if show_plugins:
            menu_items.update(self._get_plugins_menu())
        menu_items.update(extra_items)
        menu_items.update(self._get_help_menu())  # This menu must come last
        _update_menu(self.__q_window, self.__menu, menu_items)

    def get_undo_redo(self) -> UndoRedo:
        """Gets the undo/redo handler. Any actions performed through this handler can be undone."""
        return self.__gui_experiment.undo_redo

    def _get_default_menu(self) -> Dict[str, Any]:
        return dict()

    def _get_plugins_menu(self) -> Dict[str, Any]:
        """Returns all menu options added by plugins."""
        return dict()

    def _get_help_menu(self) -> Dict[str, Any]:
        return dict()

    def reload_plugins(self):
        """For windows that support plugins, this method reloads them all."""
        pass


_NEW_CATEGORY = "---"


class _MenuData:
    """Class to translate from the string-based menu format to the Qt menu."""

    _categories: Dict[str, Dict[str, Union[Callable, "_MenuData"]]]

    def __init__(self, *starting_names: str):
        """Initializes a menu. Use starting_names to fix the order of the submenus with the given names."""
        self._categories = dict()
        if starting_names:
            self._categories[""] = dict()
            for name in starting_names:
                self._categories[""][name] = _MenuData()

    def items(self) -> Iterable[Tuple[str, Union[Callable, "_MenuData", None]]]:
        """Gets all menu items from any category. Yields (_NEW_CATEGORY, None) in between two categories."""
        first_category = True
        for category in self._categories.values():
            if first_category:
                first_category = False
            else:
                yield _NEW_CATEGORY, None
            yield from category.items()

    def is_empty(self):
        return len(self._categories) == 0

    def to_qmenu(self, qmenu: QMenu):
        for item_label, item_action in self.items():
            if item_label == _NEW_CATEGORY:  # Separation line
                qmenu.addSeparator()
            elif isinstance(item_action, _MenuData):  # Submenu
                sub_menu = qmenu.addMenu(item_label)
                item_action.to_qmenu(sub_menu)
            else:  # Single action
                text, shortcut = self._parse_item_label(item_label)
                if shortcut:
                    qmenu.addAction(text, _with_safeguard(item_action), shortcut)
                else:
                    qmenu.addAction(text, _with_safeguard(item_action))

    def add_menu_entry(self, name: str, action: Callable):
        """Examples:

        >>> self.add_menu_entry("Lineages-Show lineage tree... [Ctrl+L]", lambda: ...)
        Adds a menu option named "Show lineage tree...". Category is "Lineages". Shortcut is CONTROL+L.

        >>> self.add_menu_entry("Analysis-Cell deaths//Distance-Distance between neighbor cells...", lambda: ...)
        Adds a menu option to the Cell deaths submenu.
        """
        split_by_slashes = name.split("//")
        part_for_this_menu = split_by_slashes[0]

        if "-" in part_for_this_menu:
            category_name, label = part_for_this_menu.split("-", maxsplit=1)
        else:
            category_name, label = "", part_for_this_menu

        # Add new category if it doesn't exist yet
        if category_name not in self._categories:
            self._categories[category_name] = dict()

        if len(split_by_slashes) > 1:
            # Add to sub menu
            category = self._categories[category_name]
            if label not in category:
                category[label] = _MenuData()
            sub_menu: _MenuData = category[label]
            sub_menu.add_menu_entry("//".join(split_by_slashes[1:]), action)
        else:
            # Add to this menu
            self._categories[category_name][label] = action

    def __repr__(self) -> str:
        return f"<_MenuData, {len(list(self.items()))} items>"

    def _parse_item_label(self, item_label: str) -> Tuple[str, Optional[str]]:
        if " [" in item_label and item_label.endswith("]"):
            index = item_label.index("[")
            return item_label[0:index - 1], item_label[index + 1:-1]
        return item_label, None


def _simple_menu_dict_to_nested(menu_items: Dict[str, Any]) -> _MenuData:
    menu_tree = _MenuData("File", "Edit", "View")  # Force order of standard menus

    for name, action in menu_items.items():
        menu_tree.add_menu_entry(name, action)

    return menu_tree


def _update_menu(q_window: QMainWindow, menu_bar: QMenuBar, menu_items: Dict[str, Any]):
    menu_tree = _simple_menu_dict_to_nested(menu_items)

    menu_bar.clear()  # Remove old menu bar

    for menu_name, dropdown_items in menu_tree.items():
        # Create each dropdown menu
        if not isinstance(dropdown_items, _MenuData) or dropdown_items.is_empty():
            continue  # Ignore empty menus
        qmenu = menu_bar.addMenu(menu_name)
        dropdown_items.to_qmenu(qmenu)


def _with_safeguard(action):
    """Adds an exception handler to the specified lambda action"""
    def safeguard():
        try:
            action()
        except Exception as e:
            from autotrack.gui import dialog
            dialog.popup_exception(e)
    return safeguard
