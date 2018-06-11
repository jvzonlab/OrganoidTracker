import math

import cv2

import numpy
from shapely.geometry.polygon import LinearRing
from numpy import ndarray
from typing import List, Tuple, Optional

from core.gaussian import Gaussian


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
    """An ellipse, with a method to test intersections."""
    _x: float
    _y: float
    _width: float  # Always smaller than height
    _height: float
    _angle: float  # Degrees, 0 <= angle < 180
    _polyline: ndarray
    _linear_ring: LinearRing

    def __init__(self, x: float, y: float, width: float, height: float, angle: float):
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._angle = angle
        self._polyline = self._to_polyline()
        self._linear_ring = LinearRing(self._polyline)

    def intersects(self, other: "Ellipse") -> bool:
        """Tests if this ellipse intersects another ellipse."""
        return self._linear_ring.intersects(other._linear_ring)

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

    def get_rectangular_bounds(self) -> Tuple[int, int, int, int]:
        """Returns (minx, miny, maxx, maxy) of the ellipse."""
        min_x = int(self._polyline[:, 0].min())
        max_x = int(math.ceil(self._polyline[:, 0].max()))
        min_y = int(self._polyline[:, 1].min())
        max_y = int(math.ceil(self._polyline[:, 1].max()))
        return (min_x, min_y, max_x, max_y)

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


class EllipseStack:
    """Multiple ellipses, each at their own z position. A stack is tagged with a numeric id, which is useful to keep
    track where an ellipse came from."""
    _stack: List[Ellipse]
    _tag: int

    def __init__(self, ellipses: List[Ellipse], tag: int):
        self._stack = ellipses
        self._tag = tag

    def get_tag(self) -> int:
        """Gets a numeric of the ellipse. Useful for keeping track where the ellipse came from."""
        return self._tag

    def draw_to_image(self, target: ndarray, color, dx=0, dy=0, dz=0, filled=False):
        for z in range(len(self._stack)):
            ellipse = self._stack[z]
            if ellipse is not None:
                ellipse.draw_to_image(target[z + dz], color, dx, dy, filled)

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

    def _get_xyz(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
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

    def to_gaussian(self, image_for_intensities: ndarray) -> Gaussian:
        """Does an initial guess at the parameters for a Gaussian. Raises ValueError if no ellipse was fitted at any
        z-layer."""
        x, y, z = self._get_xyz()
        if x is None:
            raise ValueError("No ellipses were fitted, so cannot determine starting point for Gaussian")
        if x < 0 or x >= image_for_intensities.shape[2] \
                or y < 0 or y >= image_for_intensities.shape[1] \
                or z < 0 or z >= image_for_intensities.shape[0]:
            raise ValueError("Ellipses have center of mass outside the image")
        return Gaussian(image_for_intensities[z, y, x], x, y, z, 50, 50, 2, 0, 0, 0)

    def can_be_fitted(self, image_for_intensities: ndarray) -> bool:
        """Checks if there are ellipses, and if their center of mass falls within the given image."""
        x, y, z = self._get_xyz()
        if x is None:
            return True
        if x < 0 or x >= image_for_intensities.shape[2] \
                or y < 0 or y >= image_for_intensities.shape[1] \
                or z < 0 or z >= image_for_intensities.shape[0]:
            return False
        return True


class EllipseCluster:
    """Multiple stacks of ellipses that are so close to each other that a Gaussian mixture model is necessary."""

    _stacks: List[EllipseStack]

    def __init__(self, stacks: List[EllipseStack]):
        self._stacks = list(stacks)

    def get_image_for_fit(self, original_image: ndarray, padding_xy: int = 4, padding_z: int = 1
                          ) -> Tuple[int, int, int, Optional[ndarray]]:
        """Gets all pixels in the original image that belong to the given cell(s). These pixels can then be fitted to a
        Gaussian."""
        min_x, min_y, min_z, max_x, max_y, max_z = self._get_bounds()
        if min_x is None:
            return 0, 0, 0, None
        min_x = max(0, min_x - padding_xy)
        min_y = max(0, min_y - padding_xy)
        min_z = max(0, min_z - padding_z)
        max_x = min(original_image.shape[2], max_x + padding_xy + 1)
        max_y = min(original_image.shape[1], max_y + padding_xy + 1)
        max_z = min(original_image.shape[0], max_z + padding_z + 1)

        dx, dy, dz= max_x - min_x, max_y - min_y, max_z - min_z
        threshold_image = numpy.zeros((dz, dy, dx), dtype=numpy.uint8)
        for stack in self._stacks:
            stack.draw_to_image(threshold_image, 255, -min_x, -min_y, -min_z, filled=True)
        original_image_cropped = numpy.copy(original_image[min_z:max_z, min_y:max_y, min_x:max_x])
        if original_image_cropped.shape != threshold_image.shape:
            raise ValueError("Error in shape")
        original_image_cropped &= threshold_image

        return min_x, min_y, min_z, original_image_cropped

    def _get_bounds(self):
        min_x, min_y, min_z = None, None, None
        max_x, max_y, max_z = None, None, None
        for stack in self._stacks:
            s_min_x, s_min_y, s_min_z, s_max_x, s_max_y, s_max_z = stack.get_rectangular_bounds()
            min_x = _min(s_min_x, min_x)
            min_y = _min(s_min_y, min_y)
            min_z = _min(s_min_z, min_z)
            max_x = _max(s_max_x, max_x)
            max_y = _max(s_max_y, max_y)
            max_z = _max(s_max_z, max_z)
        return min_x, min_y, min_z, max_x, max_y, max_z

    def draw_to_image(self, out: ndarray, color, filled=False):
        for stack in self._stacks:
            stack.draw_to_image(out, color, filled=filled)

    def guess_gaussians(self, image_for_intensities: ndarray) -> List[Gaussian]:
        """Gets Gaussian functions, with the intensities from the image and the positions of the ellipses as starting
        points."""
        return [stack.to_gaussian(image_for_intensities) for stack in self._stacks
                if stack.can_be_fitted(image_for_intensities)]

    def get_tags(self) -> List[int]:
        """Gets the tags of all stacks. The returned order matches the order of self.guess_gaussians(...)"""
        return [stack.get_tag() for stack in self._stacks]
