from typing import Callable, List, Dict, Any

from autotrack.core.experiment import Experiment
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


class GuiExperiment:
    """Used to store the experiment, along with some data that is only relevant within a GUI app, but that doesn't
    need to be saved."""

    KNOWN_EVENTS = {"data_updated_event", "any_updated_event", "command_event"}

    _experiment: Experiment
    _undo_redo: UndoRedo
    _data_updated_handlers: _EventListeners
    _any_updated_event: _EventListeners
    _command_handlers: _EventListeners

    def __init__(self, experiment: Experiment):
        self._experiment = experiment
        self._undo_redo = UndoRedo()

        self._data_updated_handlers = _EventListeners()
        self._any_updated_event = _EventListeners()
        self._command_handlers = _EventListeners()

    @property  # read-only
    def undo_redo(self) -> UndoRedo:
        return self._undo_redo

    @property  # read-only
    def experiment(self) -> Experiment:
        return self._experiment

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

    def set_experiment(self, experiment: Experiment):
        self._experiment = experiment
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
