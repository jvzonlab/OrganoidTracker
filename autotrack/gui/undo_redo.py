import collections
from typing import Deque

from autotrack.core.experiment import Experiment


class UndoableAction:

    def do(self, experiment: Experiment) -> str:
        """Performs the action. The scratch links in the experiment will be initialized. Returns an user-friendly
        message of what just happened."""
        raise NotImplementedError()

    def undo(self, experiment: Experiment) -> str:
        """Undoes the action. The scratch links in the experiment will be initialized. Returns an user-friendly
        message of what just happened."""
        raise NotImplementedError()


class UndoRedo:

    _undo_queue: Deque[UndoableAction]
    _redo_queue: Deque[UndoableAction]

    def __init__(self):
        self._undo_queue = collections.deque(maxlen=50)
        self._redo_queue = collections.deque(maxlen=50)

    def do(self, action: UndoableAction, experiment: Experiment) -> str:
        """Performs an action, and stores it so that we can undo it"""
        result_string = action.do(experiment)
        self._undo_queue.append(action)
        self._redo_queue.clear()
        return result_string

    def undo(self, experiment: Experiment) -> str:
        try:
            action = self._undo_queue.pop()
            result_string = action.undo(experiment)
            self._redo_queue.append(action)
            return result_string
        except IndexError:
            return "No more actions to undo."

    def redo(self, experiment: Experiment) -> str:
        try:
            action = self._redo_queue.pop()
            result_string = action.do(experiment)
            self._undo_queue.append(action)
            return result_string
        except IndexError:
            return "No more actions to redo."

    def clear(self):
        """Clears both the undo and redo queue."""
        self._undo_queue.clear()
        self._redo_queue.clear()