import collections
from typing import Deque, List

from organoid_tracker.core.experiment import Experiment


class UndoableAction:

    # Normally this is set to True, which means the user will be warned that there are unsaved changes if such an
    # action is on the undo stack.
    # But you can set this to False for your action if you don't want the user to get prompted to save their changes.
    needs_saving: bool = True

    def do(self, experiment: Experiment) -> str:
        """Performs the action. The scratch links in the experiment will be initialized. Returns an user-friendly
        message of what just happened."""
        raise NotImplementedError()

    def undo(self, experiment: Experiment) -> str:
        """Undoes the action. The scratch links in the experiment will be initialized. Returns an user-friendly
        message of what just happened."""
        raise NotImplementedError()


class ReversedAction(UndoableAction):
    """Does exactly the opposite of another action. It works by switching the do and undo methods."""
    inverse: UndoableAction

    def __init__(self, action: UndoableAction):
        """Note: there must be a link between the two positions."""
        self.inverse = action

    def do(self, experiment: Experiment):
        return self.inverse.undo(experiment)

    def undo(self, experiment: Experiment):
        return self.inverse.do(experiment)


class CombinedAction(UndoableAction):
    """Combines multiple actions into one. It works by calling the do and undo methods of the actions in the order they
    were added."""
    _actions: List[UndoableAction]
    _do_message: str
    _undo_message: str

    def __init__(self, actions: List[UndoableAction], *, do_message: str, undo_message: str):
        self._actions = list(actions)
        self._do_message = do_message
        self._undo_message = undo_message

    def do(self, experiment: Experiment):
        for action in self._actions:
            action.do(experiment)
        return self._do_message

    def undo(self, experiment: Experiment):
        for action in reversed(self._actions):
            action.undo(experiment)
        return self._undo_message


class UndoRedo:

    _undo_queue: Deque[UndoableAction]
    _redo_queue: Deque[UndoableAction]
    _unsaved_changes_count: int = 0

    def __init__(self):
        self._undo_queue = collections.deque(maxlen=50)
        self._redo_queue = collections.deque(maxlen=50)

    def has_unsaved_changes(self) -> bool:
        """Returns True if there are any unsaved changes, False otherwise."""
        return self._unsaved_changes_count != 0

    def mark_everything_saved(self):
        """Marks that at this moment in time, everything was saved. has_unsaved_changes() will return True after calling
        this method. However, after an action is done (see do(...) or redo(...), has_unsaved_changes() will return True
        again. If you then undo that action again, has_unsaved_changes() will return False gain. Clever, isn't it?"""
        self._unsaved_changes_count = 0

    def do(self, action: UndoableAction, experiment: Experiment) -> str:
        """Performs an action, and stores it so that we can undo it"""
        result_string = action.do(experiment)
        self._undo_queue.append(action)
        self._redo_queue.clear()
        if action.needs_saving:
            self._unsaved_changes_count += 1
        return result_string

    def undo(self, experiment: Experiment) -> str:
        try:
            action = self._undo_queue.pop()
            result_string = action.undo(experiment)
            self._redo_queue.append(action)
            if action.needs_saving:
                self._unsaved_changes_count -= 1
            return result_string
        except IndexError:
            return "No more actions to undo."

    def redo(self, experiment: Experiment) -> str:
        try:
            action = self._redo_queue.pop()
            result_string = action.do(experiment)
            self._undo_queue.append(action)
            if action.needs_saving:
                self._unsaved_changes_count += 1
            return result_string
        except IndexError:
            return "No more actions to redo."

    def clear(self):
        """Clears both the undo and redo queue. THis is useful if you just performed a big action that cannot be undone.
        """
        self._undo_queue.clear()
        self._redo_queue.clear()
        self.mark_unsaved_changes()

    def mark_unsaved_changes(self):
        """Forces a resave. Useful if you changed some data in the experiment that cannot be undone."""
        self._unsaved_changes_count = 1000000  # This makes sure that the save prompt will be triggered
