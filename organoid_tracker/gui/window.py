from typing import Callable, Dict, List, Any, Optional, Union, Iterable, Tuple

from PySide2.QtWidgets import QMainWindow, QLabel, QMenuBar, QMenu
from matplotlib.figure import Figure

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.gui import APP_NAME
from organoid_tracker.gui.gui_experiment import GuiExperiment
from organoid_tracker.gui.threading import Scheduler
from organoid_tracker.gui.undo_redo import UndoRedo, UndoableAction
from organoid_tracker.plugin.plugin_manager import PluginManager
from organoid_tracker.plugin.registry import Registry


class DisplaySettings:
    """Used for window-specific display settings."""
    show_next_time_point: bool
    show_images: bool
    show_splines: bool
    show_positions: bool
    show_links_and_connections: bool
    show_errors: bool
    time_point: TimePoint
    z: int
    image_channel: Optional[ImageChannel]  # Set to None to use the default image channel
    error_correction_min_time_point: Optional[TimePoint] = None
    error_correction_max_time_point: Optional[TimePoint] = None
    error_correction_min_divisions: int = 0

    def __init__(self, *, show_next_time_point: bool = False, show_images: bool = True,
                 show_data_axes: bool = True, show_positions: bool = True):
        self.show_next_time_point = show_next_time_point
        self.show_images = show_images
        self.show_splines = show_data_axes
        self.show_positions = show_positions
        self.show_errors = True
        self.show_links_and_connections = True
        self.image_channel = None
        self.time_point = TimePoint(0)
        self.z = 14

    KEY_SHOW_NEXT_IMAGE_ON_TOP = "n"
    KEY_SHOW_IMAGES = "i"


class Window:
    """The model for a window."""
    __q_window: QMainWindow

    __fig: Figure
    __status_text: QLabel
    __title_text: QLabel
    __gui_experiment: GuiExperiment
    __display_settings: DisplaySettings
    __event_handler_ids: List[int]
    __menu: QMenuBar
    __scheduler: Optional[Scheduler] = None
    __plugin_manager: PluginManager

    def __init__(self, q_window: QMainWindow, figure: Figure, experiment: GuiExperiment,
                 title_text: QLabel, status_text: QLabel):
        self.__q_window = q_window
        self.__menu = q_window.menuBar()
        self.__fig = figure
        self.__gui_experiment = experiment
        self.__status_text = status_text
        self.__title_text = title_text
        self.__event_handler_ids = list()
        self.__display_settings = DisplaySettings()
        self.__plugin_manager = PluginManager()

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
        """Gets the experiment that is being shown. Raises UserError if no particular experiment was selected."""
        return self.__gui_experiment.get_experiment()

    def get_active_experiments(self) -> Iterable[Experiment]:
        return self.__gui_experiment.get_active_experiments()

    def get_gui_experiment(self) -> GuiExperiment:
        """Gets the GUI experiment, which stores the experiment along with undo/redo data, and some other non-saveable
        data."""
        return self.__gui_experiment

    @property
    def display_settings(self) -> DisplaySettings:
        # Variable cannot be set - only read
        return self.__display_settings

    def perform_data_action(self, action: UndoableAction):
        """Performs an action that will change the experiment data. The action will be executed, then the window will
        be redrawn with an updated status bar."""
        try:
            status = self.get_undo_redo().do(action, self.get_experiment())
            self.set_status(status)
        finally:
            self.redraw_data()

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
        menu_items.update(self._get_last_default_menu())  # This menu must come last
        _update_menu(self.__q_window, self.__menu, menu_items)

    def get_undo_redo(self) -> UndoRedo:
        """Gets the undo/redo handler. Any actions performed through this handler can be undone."""
        return self.__gui_experiment.undo_redo

    def _get_default_menu(self) -> Dict[str, Any]:
        return dict()

    def _get_plugins_menu(self) -> Dict[str, Any]:
        """Returns all menu options added by plugins."""
        return dict()

    def _get_last_default_menu(self) -> Dict[str, Any]:
        """Some items that should be added last, so that they're at the end of the menu."""
        return dict()

    @property
    def plugin_manager(self) -> PluginManager:
        """Gets the plugin manager, allowing you to load, inspect and unload plugins."""
        # Protected by a @property so that you can't replace it while it is in use.
        return self.__plugin_manager

    @property
    def registry(self) -> Registry:
        """Gets the registry. Used for plugin-provided things, like new cell types."""
        return self.__plugin_manager.get_registry()

    def replace_plugin_manager(self, value: PluginManager):
        """Replaces the plugin manager. Can only be done if no plugins are loaded."""
        if self.__plugin_manager.get_plugin_count() > 0:
            raise ValueError("Plugin manager already in use, cannot replace")
        if isinstance(value, PluginManager):
            self.__plugin_manager = value
        else:
            raise ValueError("Not a plugin manager: " + repr(value))


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
            from organoid_tracker.gui import dialog
            dialog.popup_exception(e)
    return safeguard
