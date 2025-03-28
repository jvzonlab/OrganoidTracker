"""
The core classes of OrganoidTracker. The most important one is :class:`~organoid_tracker.core.experiment.Experiment`,
which holds all data of a single time-lapse movie.

Some example code to construct positions and links:

>>> from organoid_tracker.core.experiment import Experiment
>>> from organoid_tracker.core.images import ImageResolution
>>> from organoid_tracker.core.position import Position
>>> experiment = Experiment()
>>> experiment.images.set_resolution(ImageResolution(0.32, 0.32, 2, 12))
>>> experiment.name.set_name("Some name")
>>>
>>> # Add two positions and link them
>>> experiment.positions.add(Position(0, 0, 0, time_point_number=0))
>>> experiment.positions.add(Position(1, 3, 0, time_point_number=1))
>>> experiment.links.add_link(Position(0, 0, 0, time_point_number=0), Position(1, 3, 0, time_point_number=1))
>>>
>>> print(experiment.positions.of_time_point(TimePoint(1)))  # "{Position(1, 3, 0, time_point_number=1)}"
"""
import re
import typing
from collections.abc import Sequence
from typing import Optional, Iterable, Union, Tuple, Any, NamedTuple, Sized, Container

import matplotlib.colors
import numpy

from organoid_tracker.core.typing import MPLColor

COLOR_CELL_NEXT = "#d63031"
COLOR_CELL_PREVIOUS = "#74b9ff"
COLOR_CELL_CURRENT = "#00ff77"

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

    @staticmethod
    def range(first_time_point: Optional["TimePoint"], last_time_point: Optional["TimePoint"]
              ) -> Sequence["TimePoint"]:
        """Creates a range of time points, *inclusive*. Useful for iterating over time points. If any of the time points
        is None, the range will be empty. Raises a ValueError if the first time point is after the last time point."""
        if first_time_point is None or last_time_point is None:
            return []
        if first_time_point > last_time_point:
            raise ValueError(f"First time point {first_time_point} is after last time point {last_time_point}")
        return _TimePointRange(first_time_point, last_time_point)

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

    def __lt__(self, other: "TimePoint") -> bool:
        return self._time_point_number < other._time_point_number

    def __gt__(self, other: "TimePoint") -> bool:
        return self._time_point_number > other._time_point_number

    def __le__(self, other: "TimePoint") -> bool:
        return self._time_point_number <= other._time_point_number

    def __ge__(self, other: "TimePoint") -> bool:
        return self._time_point_number >= other._time_point_number

    def __add__(self, other) -> "TimePoint":
        if isinstance(other, int):
            return TimePoint(self._time_point_number + other)
        if isinstance(other, TimePoint):
            return TimePoint(self._time_point_number + other.time_point_number())
        return NotImplemented

    def __sub__(self, other) -> "TimePoint":
        if isinstance(other, int):
            return TimePoint(self._time_point_number - other)
        if isinstance(other, TimePoint):
            return TimePoint(self._time_point_number - other.time_point_number())
        return NotImplemented


class _TimePointRange(Sequence):
    """A range of time points, inclusive. Used for iterating over time points. You can create instances using
    `TimePoint.range(first_time_point, last_time_point)`."""
    min_time_point: TimePoint
    max_time_point: TimePoint
    _index: int

    def __init__(self, min_time_point: TimePoint, max_time_point: TimePoint):
        if not isinstance(min_time_point, TimePoint) or not isinstance(max_time_point, TimePoint):
            raise TypeError(f"Expected two TimePoint instances, got {min_time_point} and {max_time_point}")
        self.min_time_point = min_time_point
        self.max_time_point = max_time_point

    def __getitem__(self, item) -> TimePoint:
        time_point_number = self.min_time_point.time_point_number() + item if item >= 0\
            else self.max_time_point.time_point_number() + 1 + item
        if time_point_number < self.min_time_point.time_point_number() or time_point_number > self.max_time_point.time_point_number():
            raise IndexError(f"Time point {time_point_number} is not in range {self}")
        return TimePoint(time_point_number)

    def __len__(self) -> int:
        return self.max_time_point.time_point_number() - self.min_time_point.time_point_number() + 1


class Name:
    """The name of the experiment. Includes a flag, is_automatic, that specifies whether the name was manually set or
    generated by the computer (for example from the image file name)."""
    _name: Optional[str]
    _is_automatic: bool

    def __init__(self, name: Optional[str] = None, *, is_automatic: bool = False):
        self._name = name
        self._is_automatic = is_automatic

    def has_name(self) -> bool:
        """Returns True if there is any name stored."""
        return self._name is not None

    def set_name(self, name: Optional[str], *, is_automatic: bool = False):
        """Overwrites the name with the new name."""
        self._name = name
        self._is_automatic = is_automatic

    def provide_automatic_name(self, name: Optional[str]):
        if self._is_automatic or self._name is None:
            self._name = name
            self._is_automatic = True

    def get_name(self) -> Optional[str]:
        """Returns the name, or None if not set.
        Note: use str(self) if you want "Unnamed" instead of None if there is no name."""
        return self._name

    def get_save_name(self) -> str:
        """Gets a name that is safe for file saving. It does not contain characters like / or \\, and no whitespace at
        the start or end."""
        return re.sub(r'[^A-Za-z0-9_\- ]+', '_', str(self)).strip()

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        """Checks if the name as returned by str(...) is equal. So
        >>> Name(None) == Name("Unnamed")
        """
        return isinstance(other, Name) and str(self) == str(other)

    def __str__(self) -> str:
        """Returns the name if there is any name stored, otherwise it returns "Unnamed"."""
        name = self._name
        return name if name is not None else "Unnamed"

    def __repr__(self) -> str:
        return "Name(" + repr(self._name) + ", is_automatic=" + str(self._is_automatic) + ")"

    def is_automatic(self) -> bool:
        """Returns True if the name was set automatically (for example from an image file name), and False if the user
        set it manually."""
        return self._is_automatic

    def copy(self) -> "Name":
        """Creates a copy of this name, so that name changes don't propagate through."""
        return Name(name=self._name, is_automatic=self._is_automatic)

class Color:
    """Represents an RGB color."""

    @staticmethod
    def from_rgb(rgb: int) -> "Color":
        """Restores the color from the hexadecimal value."""
        return Color(rgb >> 16, (rgb >> 8) & 0xff, rgb & 0xff)

    @staticmethod
    def from_rgb_floats(red: float, green: float, blue: float) -> "Color":
        """Creates a color using a RGB float color, for example (1.0, 1.0, 1.0) for white."""
        return Color(int(round(red * 255)), int(round(green * 255)), int(round(blue * 255)))

    @staticmethod
    def from_matplotlib(mpl_color: MPLColor) -> "Color":
        """Creates a color using the Matplotlib library, so you can for example do Color.from_matplotlib("red")."""
        r, g, b = matplotlib.colors.to_rgb(mpl_color)
        return Color.from_rgb_floats(r, g, b)

    @staticmethod
    def white():
        """Returns a fully white color."""
        return Color(255, 255, 255)

    @staticmethod
    def black():
        """Returns a fully black color."""
        return Color(0, 0, 0)

    _rgb: int

    def __init__(self, red: int, green: int, blue: int):
        if red < 0 or red > 255 or int(red) != red\
                or green < 0 or green > 255 or int(green) != green\
                or blue < 0 or blue > 255 or int(blue) != blue:
            raise ValueError(f"Invalid color: {(red, green, blue)}")
        self._rgb = red << 16 | green << 8 | blue

    @property
    def red(self) -> int:
        """Gets the red component from 0 to 255."""
        return self._rgb >> 16

    @property
    def green(self) -> int:
        """Gets the green component from 0 to 255."""
        return (self._rgb >> 8) & 0xff

    @property
    def blue(self) -> int:
        """Gets the blue component from 0 to 255."""
        return self._rgb & 0xff

    def to_rgb(self) -> int:
        """Gets the color as a RGB number. See also Color.from_rgb()"""
        return self._rgb
    
    def to_rgb_floats(self) -> Tuple[float, float, float]:
        """Gets the color as a RGB tuple, for use with matplotlib."""
        return self.red / 255, self.green / 255, self.blue / 255

    def to_rgba_floats(self) -> Tuple[float, float, float, float]:
        """Gets the color as a RGBA tuple, for use with matplotlib."""
        return self.red / 255, self.green / 255, self.blue / 255, 1.0

    def __str__(self) -> str:
        """Returns the color as a hexadecimal value."""
        return "#" + format(self._rgb, '06x')

    def __eq__(self, other) -> bool:
        if isinstance(other, Color):
            return other._rgb == self._rgb
        return False

    def __hash__(self) -> int:
        return self._rgb

    def is_black(self) -> bool:
        """Returns True if this color is completely black."""
        return self._rgb == 0

    def to_html_hex(self):
        """Returns the color as a hexadecimal value. Same as str(...)."""
        return self.__str__()


def min_none(numbers: Union[Optional[float], Iterable[Optional[float]]], *args: Optional[float]):
    """Calculates the minimal number. None values are ignored. Usage:

    >>> min_none(2, 3, 5, None, 5) == 2
    >>> min_none([4, None, 2, None, -1]) == -1
    >>> min_none([]) is None
    >>> min_none(None, None, None) is None
    """
    min_value = None

    if numbers is None or not numpy.iterable(numbers):
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

    if numbers is None or not numpy.iterable(numbers):
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
