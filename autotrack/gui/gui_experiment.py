from typing import Callable, List, Dict, Any, Iterable, Optional

from autotrack.core.experiment import Experiment
from autotrack.core.position import PositionType, Position
from autotrack.gui.undo_redo import UndoRedo


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
        for listeners in self._listeners.values():
            for listener in listeners:
                listener(*args)

class _SingleGuiExperiment:
    experiment: Experiment
    undo_redo: UndoRedo

    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.undo_redo = UndoRedo()

    def __repr__(self) -> str:
        return "<Experiment " + self.experiment.name.get_save_name() + ">"

class GuiExperiment:
    """Used to store the experiment, along with some data that is only relevant within a GUI app, but that doesn't
    need to be saved."""

    KNOWN_EVENTS = {"data_updated_event", "any_updated_event", "command_event"}

    _experiments: List[_SingleGuiExperiment]
    _data_updated_handlers: _EventListeners
    _any_updated_event: _EventListeners
    _command_handlers: _EventListeners
    _position_types: Dict[str, PositionType]

    def __init__(self, experiment: Experiment):
        self._experiments = [_SingleGuiExperiment(experiment)]

        self._data_updated_handlers = _EventListeners()
        self._any_updated_event = _EventListeners()
        self._command_handlers = _EventListeners()
        self._position_types = dict()

    @property  # read-only
    def undo_redo(self) -> UndoRedo:
        return self._experiments[-1].undo_redo

    @property  # read-only
    def experiment(self) -> Experiment:
        return self._experiments[-1].experiment

    def register_event_handler(self, event: str, source: str, action: Callable):
        """Registers an event handler. Supported events:

        * All matplotlib events.
        * "data_updated_event" for when the figure annotations need to be redrawn.
        * "any_updated_event" for when the complete figure needs to be redrawn, including the menu bar and image.
        * "command_event" for when a command is executed
        """
        if event == "data_updated_event":
            self._data_updated_handlers.add(source, action)
        elif event == "any_updated_event":
            self._any_updated_event.add(source, action)
        elif event == "command_event":
            self._command_handlers.add(source, action)
        else:
            raise ValueError("Unknown event: " + event)

    def unregister_event_handlers(self, source_to_remove: str):
        """Unregisters all handles registered using register_event_handler"""
        self._data_updated_handlers.remove(source_to_remove)
        self._any_updated_event.remove(source_to_remove)
        self._command_handlers.remove(source_to_remove)

    def register_position_type(self, position_type: PositionType):
        """Registers a new position type (overwriting any existing one with the same save name)."""
        self._position_types[position_type.save_name] = position_type

    def get_position_types(self) -> Iterable[PositionType]:
        """Gets all registered position types."""
        return self._position_types.values()

    def get_position_type(self, save_name: Optional[str]) -> Optional[PositionType]:
        """Gets the position type using the given save name. Returns None if no position type exists for that save name.
        """
        if save_name is None:
            return None
        return self._position_types.get(save_name)

    def add_experiment(self, experiment: Experiment):
        # Remove current experiment if it contains no data
        if self.experiment.first_time_point_number() is None:
            self._remove_experiment_without_opdate(self.experiment)

        # Add new experiment
        self._experiments.append(_SingleGuiExperiment(experiment))
        self._any_updated_event.call_all()

    def get_experiments(self) -> Iterable[Experiment]:
        """Gets all currently loaded experiments."""
        for gui_experiment in self._experiments:
            yield gui_experiment.experiment

    def _remove_experiment_without_opdate(self, experiment: Experiment):
        """Removes an experiment. Does not add a new experiment in case the list becomes empty."""
        for i in range(len(self._experiments)):
            if self._experiments[i].experiment is experiment:
                del self._experiments[i]
                return

    def remove_experiment(self, experiment: Experiment):
        """Removes the given experiment from the list of loaded experiments. If no experiments are remaining, an empty
        one will be initialized."""
        self._remove_experiment_without_opdate(experiment)
        if len(self._experiments) == 0:  # Prevent list from being empty
            self._experiments.append(_SingleGuiExperiment(Experiment()))
        self._any_updated_event.call_all()

    def redraw_data(self):
        """Redraws the main figure using the latest values from the experiment."""
        self._data_updated_handlers.call_all()

    def redraw_image_and_data(self):
        """Redraws the image using the latest values from the experiment."""
        self._any_updated_event.call_all()

    def execute_command(self, command: str):
        """Calls all registered command handlers with the given argument. Used when a user entered a command."""
        self._command_handlers.call_all(command)

    def goto_position(self, position: Position):
        """Moves the view towards the given position. The position must have a time point set."""
        if position.time_point_number() is None:
            raise ValueError("No time point number set")
        self.execute_command(f"goto {position.x} {position.y} {position.z} {position.time_point_number()}")
