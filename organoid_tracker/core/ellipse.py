import math
from typing import List, Tuple, Optional

import cv2
import numpy
from numpy import ndarray


def _max(a: Optional[int], b: Optional[int]) -> int:
    if a is None:
        return b
    if b is None:
        return a
    return a if a > b else b


def _min(a: Optional[int], b: Optional[int]) -> int:
    if a is None:
        return b
    if b is None:
        return a
    return a if a < b else b


class Ellipse:
    """An ellipse, with a method to test intersections. The class is immutable."""
    _x: float
    _y: float
    _width: float  # Always smaller than height
    _height: float
    _angle: float  # Degrees, 0 <= angle < 180

    _polyline: ndarray = None

    def __init__(self, x: float, y: float, width: float, height: float, angle: float):
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._angle = angle
        if height < width:
            raise ValueError("height < width, this is not allowed")

    def _get_polyline(self, n=100) -> ndarray:
        """Approximates the ellipse as n connected line segments. You can then use the intersection method on the
        returned instance to check for overlaps with other ellipses."""
        if self._polyline is None:
            # Based on https://stackoverflow.com/questions/15445546/finding-intersection-points-of-two-ellipses-python
            t = numpy.linspace(0, 2 * numpy.pi, n, endpoint=False)
            st = numpy.sin(t)
            ct = numpy.cos(t)
            a = self._width / 2
            b = self._height / 2

            angle = numpy.deg2rad(self._angle)
            sa = numpy.sin(angle)
            ca = numpy.cos(angle)
            p = numpy.empty((n, 2))
            p[:, 0] = self._x + a * ca * ct - b * sa * st
            p[:, 1] = self._y + a * sa * ct + b * ca * st
            self._polyline = p
        return self._polyline

    def get_rectangular_bounds(self) -> Tuple[int, int, int, int]:
        """Returns (minx, miny, maxx, maxy) of the ellipse."""
        polyline = self._get_polyline()
        min_x = int(polyline[:, 0].min())
        max_x = int(math.ceil(polyline[:, 0].max()))
        min_y = int(polyline[:, 1].min())
        max_y = int(math.ceil(polyline[:, 1].max()))
        return min_x, min_y, max_x, max_y

    def draw_to_image(self, target: ndarray, color, dx = 0, dy = 0, filled=False):
        thickness = -1 if filled else 2

        # PyCharm cannot recognize signature of cv2.ellipse, so the warning is a false positive:
        # noinspection PyArgumentList
        cv2.ellipse(target, ((self._x + dx, self._y + dy), (self._width, self._height), self._angle),
                    color=color, thickness=thickness)

    def get_pos(self) -> Tuple[float, float]:
        return self._x, self._y

    def __repr__(self):
        return "Ellipse(" + str(self._x) + ", " + str(self._y) + ", " + str(self._width) + ", " + str(
            self._height) + ", " + str(self._angle) + ")"

    # The properties. We cannot simply make x/y/width/height/angle public, as that would make the properties mutable.
    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def angle(self):
        return self._angle

    def area(self):
        return math.pi * self._width / 2 * self._height / 2

    def perimeter(self):
        # Source: https://www.mathsisfun.com/geometry/ellipse-perimeter.html (if offline, go to web.archive.org)
        a = self._width / 2
        b = self._height / 2
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return math.pi * (a + b) * (1 + (1 / 4) * h + (1 / 64) * h + (1 / 256) * h)

    def is_elongated(self):
        """Checks if the width is significantly larger than the height."""
        return self._width / self._height > 1.2

    def translated(self, dx, dy):
        """Gets an ellipse translated in the x/y direction. Does not modify this stack."""
        return Ellipse(self._x + dx, self._y + dy, self._width, self._height, self._angle)


class EllipseStack:
    """Multiple ellipses, each at their own z position."""
    _stack: List[Ellipse]

    def __init__(self, ellipses: List[Ellipse]):
        self._stack = ellipses

    def draw_to_image(self, target: ndarray, color, dx=0, dy=0, dz=0, filled=False):
        for z in range(len(self._stack)):
            ellipse = self._stack[z]
            if ellipse is not None:
                ellipse.draw_to_image(target[z + dz], color, dx, dy, filled)

    def __len__(self) -> int:
        return len(self._stack)

    def __getitem__(self, item: int) -> Ellipse:
        return self._stack[item]

    def get_ellipse(self, ellipse_number: int) -> Optional[Ellipse]:
        """Gets the ellipse at the given index, or None if out of bounds."""
        if ellipse_number < 0 or ellipse_number >= len(self._stack):
            return None
        return self._stack[ellipse_number]

    def intersects(self, other: "EllipseStack") -> bool:
        """Checks for an intersection on any plane."""
        total_plane_count = len(self._stack)
        intersecting_plane_count = 0
        for z in range(total_plane_count):
            ellipse = self._stack[z]
            if ellipse is None:
                continue
            other_ellipse = other._stack[z]
            if other_ellipse is None:
                continue
            if ellipse.intersects(other_ellipse):
                intersecting_plane_count += 1
        return intersecting_plane_count >= min(2, total_plane_count)

    def get_rectangular_bounds(self) -> Tuple[int, int, int, int, int, int]:
        min_x, min_y, min_z = None, None, None
        max_x, max_y, max_z = None, None, None
        for z in range(len(self._stack)):
            ellipse = self._stack[z]
            if ellipse is None:
                continue

            if min_z is None:
                min_z = z
            max_z = z
            p_min_x, p_min_y, p_max_x, p_max_y = ellipse.get_rectangular_bounds()
            min_x = _min(p_min_x, min_x)
            min_y = _min(p_min_y, min_y)
            max_x = _max(p_max_x, max_x)
            max_y = _max(p_max_y, max_y)

        return min_x, min_y, min_z, max_x, max_y, max_z

    def __str__(self):
        for z in range(len(self._stack)):
            ellipse = self._stack[z]
            if ellipse is None:
                continue
            return "Stack, first is " + str(ellipse) + " at z=" + str(z)
        return "Stack, empty"

    def get_mean_position(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Gets the mean of the positions of all ellipses. The weight of each ellipsis in the stack is equal. (So
        larger ellipses aren't weighted more.)"""
        total_x, total_y, total_z, count = 0, 0, 0, 0
        for z in range(len(self._stack)):
            ellipse = self._stack[z]
            if ellipse is None:
                continue

            x, y = ellipse.get_pos()
            total_x += x
            total_y += y
            total_z += z
            count += 1
        if count == 0:
            return None, None, None
        x = total_x / count
        y = total_y / count
        z = total_z / count
        return int(x), int(y), int(z)

    def can_be_fitted(self, image_for_intensities: ndarray) -> bool:
        """Checks if there are ellipses, and if their center of mass falls within the given image."""
        x, y, z = self.get_mean_position()
        if x is None:
            return True
        if x < 0 or x >= image_for_intensities.shape[2] \
                or y < 0 or y >= image_for_intensities.shape[1] \
                or z < 0 or z >= image_for_intensities.shape[0]:
            return False
        return True

    def translated(self, dx: float, dy: float):
        """Gets an ellipse stack translated in the x/y direction. Does not modify this stack."""
        translated_ellipses = []
        for ellipse in self._stack:
            if ellipse is None:
                translated_ellipses.append(None)
            else:
                translated_ellipses.append(ellipse.translated(dx, dy))
        return EllipseStack(translated_ellipses)
