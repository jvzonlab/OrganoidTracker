"""Some base classes. Quick overview: Particles (usually cells, but may also be artifacts) are placed in TimePoints,
which are placed in an Experiment. A TimePoint also stores scores of possible mother-daughter cell combinations.
An Experiment also stores an ImageLoader and up to two cell links networks (stored as Graph objects)."""
import re
from typing import Optional

COLOR_CELL_NEXT = "red"
COLOR_CELL_PREVIOUS = "blue"
COLOR_CELL_CURRENT = "lime"


class UserError(Exception):
    """Used for errors that are not the fault of the programmer, but of the user."""

    title: str
    body: str

    def __init__(self, title: str, message: str):
        super().__init__(title + "\n" + message)
        self.title = title
        self.body = message


class TimePoint:
    """A single point in time."""

    _time_point_number: int

    def __init__(self, time_point_number: int):
        self._time_point_number = time_point_number

    def time_point_number(self) -> int:
        return self._time_point_number

    def __hash__(self):
        return self._time_point_number * 31

    def __eq__(self, other):
        return isinstance(other, TimePoint) and other._time_point_number == self._time_point_number

    def __repr__(self):
        return "TimePoint(" + str(self._time_point_number) + ")"


class Name:
    _automatic: bool = True  # Set to False when the user manually entered a name
    _name: Optional[str] = None

    def provide_automatic_name(self, name: Optional[str]):
        if self._automatic:
            # Allowed to replace one automatically generated name by another
            self._name = name
            return True
        return False  # Don't allow to override a manually chosen name by some automatically generated name

    def set_name(self, name: Optional[str]):
        """Forcibly sets a name."""
        self._automatic = False
        self._name = name

    def get_save_name(self):
        """Gets a name that is safe for file saving. It does not contain characters like / or \\."""
        return re.sub(r'[^A-Za-z0-9_\- ]+', '_', str(self))

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        """Checks if the name as returned by str(...) is equal. So an automatic name can match a manual name."""
        return isinstance(other, Name) and str(self) == str(other)

    def __str__(self):
        name = self._name
        return name if name is not None else "Unnamed"
