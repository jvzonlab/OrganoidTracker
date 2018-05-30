import cv2

import numpy
from shapely.geometry.polygon import LinearRing
from numpy import ndarray
from typing import List


class Ellipse:
    """An ellipse, with a method to test intersections."""
    _x: float
    _y: float
    _width: float  # Always smaller than height
    _height: float
    _angle: float  # Degrees, 0 <= angle < 180
    _polyline: LinearRing

    def __init__(self, x, y, width, height, angle):
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._angle = angle
        self._polyline = LinearRing(self._to_polyline())

    def intersects(self, other: "Ellipse") -> bool:
        """Tests if this ellipse intersects another ellipse."""
        return self._polyline.intersects(other._polyline)

    def _to_polyline(self, n=100) -> ndarray:
        """Approximates the ellipse as n connected line segments. You can then use the intersection method on the
        returned instance to check for overlaps with other ellipses."""
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
        return p

    def draw_to_image(self, target: ndarray, color):
        cv2.ellipse(target, ((self._x, self._y), (self._width, self._height), self._angle), color=color, thickness=2)

    def __repr__(self):
        return "Ellipse(" + str(self._x) + ", " + str(self._y) + ", " + str(self._width) + ", " + str(
            self._height) + ", " + str(self._angle) + ")"


class EllipseStack:
    """Multiple ellipses, each at their own z position."""
    _stack: List[Ellipse]

    def __init__(self, ellipses: List[Ellipse]):
        self._stack = ellipses

    def draw_to_image(self, target: ndarray, color):
        for z in range(target.shape[0]):
            ellipse = self._stack[z]
            if ellipse is not None:
                ellipse.draw_to_image(target[z], color)

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

    def __str__(self):
        for z in range(len(self._stack)):
            ellipse = self._stack[z]
            if ellipse is None:
                continue
            return "Stack, first is " + str(ellipse) + " at z=" + str(z)
        return "Stack, empty"
