import math
from typing import Optional, List

from matplotlib.axes import Axes
from matplotlib.patches import Ellipse


class ParticleShape:
    """Represents the shape of a particle. No absolute position data is stored here."""

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: str):
        """Draws a shape in 2d, at the given x and y. dz is how many z layers we are removed from the actual position,
        dt the same, but for time steps. Both dz and dt can be negative. area is the drawing surface, and color is the
        desired color.
        """
        raise NotImplementedError()

    def default_draw(self, x: float, y: float, dz: int, dt: int, area: Axes, color: str):
        """The default (point) representation of a shape. Implementation can fall back on this if they want."""
        marker_style = 's' if dz == 0 else 'o'
        marker_size = max(1, 7 - abs(dz) - abs(dt))
        area.plot(x, y, marker_style, color=color, markeredgecolor='black', markersize=marker_size, markeredgewidth=1)

    def raw_area(self) -> float:
        """Derived from raw detection."""
        raise NotImplementedError()

    def fitted_area(self) -> float:
        """Derived from a fitted shape."""
        raise NotImplementedError()

    def perimeter(self) -> float:
        raise NotImplementedError()

    def is_unknown(self) -> bool:
        """Returns True if there is no shape information available at all."""
        return False

    def is_eccentric(self) -> bool:
        """Returns True if the original shape does not touch the (0,0) point. This generally indicates that the shape
        is either very strange (can happen in mother cells) or mis-detected."""
        return False

    def director(self, require_reliable: bool = False) -> Optional[float]:
        """For (slightly) elongated shapes, this returns the main direction of that shape. The returned direction falls
        within the range 0 <= direction < 180. For shapes that are very spherical, the returned number becomes quite
        arbitrary. If require_reliable is set to True, then None is returned in this case."""
        if require_reliable:
            return None
        return 0

    def to_list(self) -> List:
        """Converts this shape to a list for serialization purposes. Convert back using from_list"""
        raise NotImplementedError()


class UnknownShape(ParticleShape):

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: str):
        self.default_draw(x, y, dz, dt, area, color)

    def raw_area(self) -> float:
        return 0

    def fitted_area(self) -> float:
        return 0

    def perimeter(self) -> float:
        return 0

    def is_unknown(self) -> bool:
        return True

    def to_list(self):
        return []


class EllipseShape(ParticleShape):
    """Represents an ellipsoidal shape. The shape only stores 2D information."""
    _ellipse_dx: float  # Offset from particle center
    _ellipse_dy: float  # Offset from particle center
    _ellipse_width: float  # Always smaller than height
    _ellipse_height: float
    _ellipse_angle: float  # Degrees, 0 <= angle < 180

    _original_perimeter: float
    _original_area: float
    _eccentric: bool

    def __init__(self, ellipse_dx: float, ellipse_dy: float, ellipse_width: float, ellipse_height: float,
                 ellipse_angle: float, original_perimeter: float, original_area: float, eccentric: bool = False):
        self._ellipse_dx = ellipse_dx
        self._ellipse_dy = ellipse_dy
        self._ellipse_width = ellipse_width
        self._ellipse_height = ellipse_height
        self._ellipse_angle = ellipse_angle

        self._original_perimeter = original_perimeter
        self._original_area = original_area
        self._eccentric = eccentric

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: str):
        fill = dt == 0
        edgecolor = 'white' if fill else color
        alpha = max(0.1, 0.5 - abs(dz / 6))
        area.add_artist(Ellipse(xy=(x + self._ellipse_dx, y + self._ellipse_dy),
                                width=self._ellipse_width, height=self._ellipse_height, angle=self._ellipse_angle,
                                fill=fill, facecolor=color, edgecolor=edgecolor, linestyle="dashed", linewidth=1,
                                alpha=alpha))

    def raw_area(self) -> float:
        return self._original_area

    def fitted_area(self):
        return math.pi * self._ellipse_width / 2 * self._ellipse_height / 2

    def perimeter(self) -> float:
        if self._original_perimeter is None:
            # Source: https://www.mathsisfun.com/geometry/ellipse-perimeter.html (if offline, go to web.archive.org)
            a = self._ellipse_width / 2
            b = self._ellipse_height / 2
            h = ((a - b) ** 2) / ((a + b) ** 2)
            return math.pi * (a + b) * (1 + (1/4) * h + (1/64) * h + (1/256) * h)
        return self._original_perimeter

    def director(self, require_reliable: bool = False) -> Optional[float]:
        if require_reliable and not self._ellipse_height / self._ellipse_width >= 1.2:
            return None
        return self._ellipse_angle

    def is_eccentric(self) -> bool:
        return self._eccentric

    def to_list(self) -> List:
        return ["ellipse", self._ellipse_dx, self._ellipse_dy, self._ellipse_width, self._ellipse_height,
                self._ellipse_angle, self._original_perimeter, self._original_area, bool(self._eccentric)]


def from_list(list: List) -> ParticleShape:
    if len(list) == 0:
        return UnknownShape()
    type = list[0]
    if type == "ellipse":
        return EllipseShape(*list[1:])
    raise ValueError("Cannot deserialize " + str(list))