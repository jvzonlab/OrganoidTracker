from typing import Callable, List, Dict, Any, Iterable, Tuple

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.gui.undo_redo import UndoRedo


class _EventListeners:
    """Used to store and call event listeners."""

    _listeners: Dict[str, List[Callable[[Any], None]]]

    def __init__(self):
        self._listeners = dict()

    def add(self, source: str, action: Callable[[Any], None]):
        """Adds a new event listener. Please don't forget to remove the event handler later, otherwise we got a memory
        leak."""
        if source in self._listeners:
            self._listeners[source].append(action)
        else:
            self._listeners[source] = [action]  # No listeners yet for this source, create a list

    def remove(self, source: str):
        """Removes all event listeners that were registered with the given source."""
        if source in self._listeners:
            del self._listeners[source]

    def call_all(self, *args):
        """Calls all registered event listeners with the specified parameters."""
        for listeners in list(self._listeners.values()):
            for listener in listeners:
                listener(*args)


class SingleGuiTab:
    _experiment: Experiment
    _undo_redo: UndoRedo

    def __init__(self, experiment: Experiment):
        self._experiment = experiment
        self._undo_redo = UndoRedo()

    @property
    def experiment(self) -> Experiment:
        return self._experiment

    @property
    def undo_redo(self) -> UndoRedo:
        return self._undo_redo

    def __repr__(self) -> str:
        return "<Experiment " + self._experiment.name.get_save_name() + ">"


class GuiExperiment:
    """Used to store the experiment, along with some data that is only relevant within a GUI app, but that doesn't
    need to be saved."""

    KNOWN_EVENTS = {"data_updated_event", "any_updated_event", "tab_list_updated_event", "command_event",
                    "program_close_event"}

    _tabs: List[SingleGuiTab]
    _selected_experiment: int  # Index in self._experiments, or equal to len(self._experiments) if all are open.
    _event_handlers: Dict[str, _EventListeners]

    def __init__(self, experiment: Experiment):
        self._tabs = [SingleGuiTab(experiment)]
        self._selected_experiment = 0

        self._event_handlers = dict()
        for event_name in self.KNOWN_EVENTS:
            self._event_handlers[event_name] = _EventListeners()

    def get_undo_redo(self) -> UndoRedo:
        """Gets the UndoRedo object of the currently selected experiment. If no single experiment is selected,
        this method will raise a UserError."""
        return self.get_open_tab().undo_redo

    def register_event_handler(self, event: str, source: str, action: Callable):
        """Registers an event handler. Supported events:

        * All matplotlib events.
        * "data_updated_event" for when the figure annotations need to be redrawn.
        * "any_updated_event" for when the complete figure needs to be redrawn, including the menu bar and image.
        * "tab_list_updated_event" for when just the list of available experiments needs to be redrawn.
        * "command_event" for when a command is executed
        """
        if event in self.KNOWN_EVENTS:
            self._event_handlers[event].add(source, action)
        else:
            raise ValueError("Unknown event: " + event)

    def unregister_event_handlers(self, source_to_remove: str):
        """Unregisters all handles registered using register_event_handler"""
        for event_handlers in self._event_handlers.values():
            event_handlers.remove(source_to_remove)

    def get_registered_markers(self, type):
        raise ValueError("Moved to window.registry.get_registered_markers(...)")

    def get_marker_by_save_name(self, save_name):
        raise ValueError("Moved to window.registry.get_marker_by_save_name(...)")

    def add_experiment(self, experiment: Experiment) -> int:
        """Adds an experiment to the tab list. Returns the index that is used for the new tab, which you can then use
        to select that tab."""
        # Remove current experiment if it contains no data
        if self.get_experiment().first_time_point_number() is None:
            self._remove_experiment_without_update(self.get_experiment())

        # Add new experiment
        self._tabs.append(SingleGuiTab(experiment))
        index = len(self._tabs) - 1

        if index == self._selected_experiment:
            self._event_handlers["any_updated_event"].call_all()  # This experiment is visible, so we need to redraw
        else:
            self._event_handlers["tab_list_updated_event"].call_all()  # Only redraw tab list
        return index

    def replace_selected_experiment(self, experiment: Experiment):
        """Discards the currently selected experiment, and replaces it with a new one"""
        self._tabs[self._selected_experiment] = SingleGuiTab(experiment)
        self._event_handlers["any_updated_event"].call_all()

    def get_experiment(self) -> Experiment:
        """Gets the currently selected experiment. Raises UserError if no particular experiment has been selected."""
        return self.get_open_tab().experiment

    def get_active_experiments(self) -> Iterable[Experiment]:
        """Gets all currently active experiments. This will usually be one experiment,
        but the user has the option to open all experiments."""
        if self._selected_experiment == len(self._tabs):
            # All are open
            for gui_experiment in self._tabs:
                yield gui_experiment.experiment
        else:
            # One experiment is open
            yield self._tabs[self._selected_experiment].experiment

    def _remove_experiment_without_update(self, experiment: Experiment):
        """Removes an experiment. Does not add a new experiment in case the list becomes empty."""
        for i in range(len(self._tabs)):
            if self._tabs[i].experiment is experiment:
                del self._tabs[i]
                return

    def remove_experiment(self, experiment: Experiment):
        """Removes the given experiment from the list of loaded experiments. If no experiments are remaining, an empty
        one will be initialized."""
        self._remove_experiment_without_update(experiment)
        if len(self._tabs) == 0:  # Prevent list from being empty
            self._tabs.append(SingleGuiTab(Experiment()))
        if self._selected_experiment >= len(self._tabs):
            self._selected_experiment = len(self._tabs) - 1
        self._event_handlers["any_updated_event"].call_all()

    def redraw_data(self):
        """Redraws the main figure using the latest values from the experiment."""
        self._event_handlers["data_updated_event"].call_all()

    def redraw_image_and_data(self):
        """Redraws the image using the latest values from the experiment."""
        self._event_handlers["any_updated_event"].call_all()

    def execute_command(self, command: str):
        """Calls all registered command handlers with the given argument. Used when a user entered a command."""
        self._event_handlers["command_event"].call_all(command)

    def program_closing(self):
        """Called when the program is closed (so after saving the data - the user cannot cancel anymore at this point).
        """
        self._event_handlers["program_close_event"].call_all()

    def goto_position(self, position: Position):
        """Moves the view towards the given position. The position must have a time point set."""
        if position.time_point_number() is None:
            raise ValueError("No time point number set")
        self.execute_command(f"goto {position.x} {position.y} {position.z} {position.time_point_number()}")

    def get_selectable_experiments(self) -> Iterable[Tuple[int, str]]:
        """Gets all selectable experiment names, along with an index for self.select_experiment()"""
        for i, gui_experiment in enumerate(self._tabs):
            yield i, str(gui_experiment.experiment.name)
        if len(self._tabs) > 1:
            yield len(self._tabs), "<all experiments>"

    def is_selected(self, select_index: int) -> bool:
        """Checks if the select_index is of the currently selected experiment."""
        return select_index == self._selected_experiment

    def select_experiment(self, index: int):
        """Sets the experiment with the given index (from self.get_selectable_experiments()) as the visible
        experiment."""
        if self._selected_experiment == index:
            return  # Nothing changed
        if index < 0 or index > len(self._tabs):
            raise ValueError(f"Out of range: {index}")

        self._selected_experiment = index
        self._event_handlers["any_updated_event"].call_all()

    def get_all_tabs(self) -> List[SingleGuiTab]:
        """Gets all currently open tabs."""
        return list(self._tabs)

    def get_open_tab(self) -> SingleGuiTab:
        """Gets the currently selected tab. Raises UserError if no particular tab has been selected."""
        if self._selected_experiment == len(self._tabs):
            # Not available when all experiments are open
            raise UserError("No experiment selected", "This function only works on a single experiment. Please select"
                                                      " one in the upper-right corner of the window.")
        return self._tabs[self._selected_experiment]

    def get_active_tabs(self) -> List[SingleGuiTab]:
        """Gets the currently active tabs. Either this is a single tab, or if the "All tabs" option has
        been selected, this returns all tabs."""
        if self._selected_experiment == len(self._tabs):
            # All tabs are open
            return self.get_all_tabs()
        return [self.get_open_tab()]

