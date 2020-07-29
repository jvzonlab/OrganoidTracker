"""Some base classes. Quick overview: Positions (usually cells, but may also be artifacts) are placed in TimePoints,
which are placed in an Experiment. A TimePoint also stores scores of possible mother-daughter cell combinations.
An Experiment also stores an ImageLoader and up to two cell links networks (stored as Graph objects)."""
import re
from typing import Optional, Iterable, Union, Tuple, Any

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

    def __init__(self, name: Optional[str] = None):
        self._name = name

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

    def get_save_name(self) -> str:
        """Gets a name that is safe for file saving. It does not contain characters like / or \\."""
        return re.sub(r'[^A-Za-z0-9_\- ]+', '_', str(self))

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        """Checks if the name as returned by str(...) is equal."""
        return isinstance(other, Name) and str(self) == str(other)

    def __str__(self) -> str:
        """Returns the name if there is any name stored, otherwise it returns "Unnamed"."""
        name = self._name
        return name if name is not None else "Unnamed"

    def __repr__(self) -> str:
        return "Name(" + repr(self._name) + ")"

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
        """Gets the color as a RGBA tuple, for use with matplotlib."""
        return self.red / 255, self.green / 255, self.blue / 255

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
