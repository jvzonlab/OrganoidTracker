"""Some base classes. Quick overview: Positions (usually cells, but may also be artifacts) are placed in TimePoints,
which are placed in an Experiment. A TimePoint also stores scores of possible mother-daughter cell combinations.
An Experiment also stores an ImageLoader and up to two cell links networks (stored as Graph objects)."""
import re
from typing import Optional, Iterable, Union

from matplotlib import colors

COLOR_CELL_NEXT = "red"
COLOR_CELL_PREVIOUS = "blue"
COLOR_CELL_CURRENT = "lime"

CM_TO_INCH = 0.393700787  # 1 cm is this many inches


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
    _name: Optional[str] = None

    def has_name(self) -> bool:
        """Returns True if there is any name stored."""
        return self._name is not None

    def set_name(self, name: Optional[str]):
        """Forcibly sets a name."""
        self._name = name

    def get_name(self) -> Optional[str]:
        """Returns the name, or None if not set.
        Note: use str(self) if you want "Unnamed" instead of None if there is no name."""
        return self._name

    def get_save_name(self):
        """Gets a name that is safe for file saving. It does not contain characters like / or \\."""
        return re.sub(r'[^A-Za-z0-9_\- ]+', '_', str(self))

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        """Checks if the name as returned by str(...) is equal."""
        return isinstance(other, Name) and str(self) == str(other)

    def __str__(self):
        """Returns the name if there is any name stored, otherwise it returns "Unnamed"."""
        name = self._name
        return name if name is not None else "Unnamed"


def min_none(numbers: Union[Optional[float], Iterable[Optional[float]]], *args: Optional[float]):
    """Calculates the minimal number. None values are ignored. Usage:

    >>> min_none(2, 3, 5, None, 5) == 2
    >>> min_none([4, None, 2, None, -1]) == -1
    >>> min_none([]) is None
    >>> min_none(None, None, None) is None
    """
    min_value = None

    if numbers is None or isinstance(numbers, float) or isinstance(numbers, int):
        numbers = [numbers] + list(args)

    for number in numbers:
        if number is None:
            continue
        if min_value is None or number < min_value:
            min_value = number
    return min_value


def max_none(numbers: Union[Optional[float], Iterable[Optional[float]]], *args: Optional[float]):
    """Calculates the minimal number. None values are ignored. Usage:

    >>> max_none(2, 3, 5, None, 5)  == 5
    >>> max_none([4, None, 2, None, -1]) == 4
    >>> max_none([]) is None
    >>> max_none(None, None, None) is None
    """
    max_value = None

    if numbers is None or isinstance(numbers, float) or isinstance(numbers, int):
        numbers = [numbers] + list(args)

    for number in numbers:
        if number is None:
            continue
        if max_value is None or number > max_value:
            max_value = number
    return max_value


def clamp(minimum: int, value: int, maximum: int):
    """Clamps the given value to the specified min and max. For example, `clamp(2, value, 4)` will return value if it's
    between 2 and 4, 2 if the value is lower than 2 and 4 if the value is higher than four."""
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value
