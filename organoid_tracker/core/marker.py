from typing import Set, Type, Tuple, Iterable, Dict, Any, Optional

from matplotlib.axes import Axes

from organoid_tracker.core import Color
from organoid_tracker.core.typing import MPLColor


class Marker:
    """Used to represent the type of a position, crypt axis or something else. So does this position represent a
    biological cell? And of which type? Does an axis represent a crypt-villus axis?"""

    _applies_to: Set[Type]
    _save_name: str
    _display_name: str
    _color: Color
    _mpl_color: Tuple[float, float, float]  # RGB color, values from 0.0 to 1.0 (inclusive). Suitable for Matplotlib.
    _extra_data: Dict[str, Any]

    def __init__(self, applies_to: Iterable[Type], save_name: str, display_name: str, color: Tuple[int, int, int],
                 **extra_data):
        self._applies_to = set(applies_to)
        self._save_name = save_name.upper()
        self._display_name = display_name
        self._color = Color(color[0], color[1], color[2])
        self._mpl_color = (color[0] / 255, color[1] / 255, color[2] / 255)
        self._extra_data = extra_data

    @property
    def color(self) -> Color:
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

    def extra(self, key: str) -> Optional[Any]:
        """Gets some extra data that was passsed to the constructor."""
        return self._extra_data.get(key)

    def __str__(self) -> str:
        return self._display_name

    def __hash__(self) -> int:
        return hash(self._save_name)

    def __eq__(self, other) -> bool:
        return isinstance(other, Marker) and self._save_name == other._save_name


def draw_marker_2d(x: float, y: float, dz: int, dt: int, area: Axes, color: MPLColor, edge_color: MPLColor):
    """The default (point) representation of a shape. Implementation can fall back on this if they want."""
    if abs(dz) > 3:
        return
    marker_style = 's' if dz == 0 else 'o'
    marker_size = max(1, 7 - abs(dz) - abs(dt))
    area.plot(x, y, marker_style, color=color, markeredgecolor=edge_color, markersize=marker_size, markeredgewidth=1)
