from typing import Set, Type, Tuple, Iterable


class Marker:
    """Used to represent the type of a position, crypt axis or something else. So does this position represent a
    biological cell? And of which type? Does an axis represent a crypt-villus axis?"""

    _applies_to: Set[Type]
    _save_name: str
    _display_name: str
    _color: Tuple[int, int, int]  # RGB color, values from 0 to 255 (inclusive)
    _mpl_color: Tuple[float, float, float]  # RGB color, values from 0.0 to 1.0 (inclusive). Suitable for Matplotlib.

    def __init__(self, applies_to: Iterable[Type], save_name: str, display_name: str, color: Tuple[int, int, int]):
        self._applies_to = set(applies_to)
        self._save_name = save_name.upper()
        self._display_name = display_name
        self._color = color
        self._mpl_color = (color[0] / 255, color[1] / 255, color[2] / 255)

    @property
    def color(self) -> Tuple[int, int, int]:
        """Gets the color used to mark positions of this type."""
        return self._color

    @property
    def mpl_color(self) -> Tuple[float, float, float]:
        """Gets the color used to mark positions of this type. Color is suitable for use in Matplotlib."""
        return self._mpl_color

    @property
    def save_name(self) -> str:
        """Gets the name used to save the type. Will always be uppercase."""
        return self._save_name

    @property
    def display_name(self) -> str:
        """Gets a name used solely for display purposes."""
        return self._display_name

    def applies_to(self, type: Type) -> bool:
        """Gets whether this position type can be applied to the """
        return type in self._applies_to

    def __str__(self) -> str:
        return self._display_name
