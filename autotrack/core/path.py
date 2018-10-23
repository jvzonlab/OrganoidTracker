from typing import List, Dict, Optional, Tuple

import numpy
from scipy import interpolate

from autotrack.core import TimePoint


class Path:
    """A curve (curved line) trough the particles. This can be used to measure how far the particles are along this
     curve."""

    _x_list: List[float]
    _y_list: List[float]
    _z: Optional[int] = None

    _interpolation: Optional[Tuple[List[float], List[float]]]

    def __init__(self):
        self._x_list = []
        self._y_list = []

    def add_point(self, x: float, y: float, z: float):
        """Adds a new point to the path."""
        if self._z is None:
            self._z = int(z)
        self._x_list.append(float(x))
        self._y_list.append(float(y))

        self._interpolation = None  # Invalidate previous interpolation

    def get_points_2d(self) -> Tuple[List[float], List[float]]:
        """Gets all explicitly added points (no interpolation) without the z coord."""
        return self._x_list, self._y_list

    def get_z(self) -> int:
        """Gets the Z coord of this path."""
        return self._z

    def get_interpolation_2d(self) -> Tuple[List[float], List[float]]:
        """Returns a (cached) list of x and y  values that are used for interpolation."""
        if self._interpolation is None:
            self._interpolation = self._calculate_interpolation()
        return self._interpolation

    def _calculate_interpolation(self) -> Tuple[List[float], List[float]]:
        if len(self._x_list) <= 1:
            # Not possible to interpolate
            return self._x_list, self._y_list

        k = 3 if len(self._x_list) > 3 else 1
        # noinspection PyTupleAssignmentBalance
        spline, _ = interpolate.splprep([self._x_list, self._y_list], k=k)
        points = interpolate.splev(numpy.arange(0, 1.01, 0.05), spline)
        x_values = points[0]
        y_values = points[1]
        return x_values, y_values

    def get_direction_marker(self) -> str:
        """Returns a char thar represents the general direction of this path: ">", "<", "^" or "v". The (0,0) coord
        is assumed to be in the top left."""
        if len(self._x_list) < 2:
            return ">"
        dx = self._x_list[-1] - self._x_list[0]
        dy = self._y_list[-1] - self._y_list[0]
        if abs(dx) > abs(dy):
            # More horizontal movement
            return "<" if dx < 0 else ">"
        else:
            # More vertical movement
            return "^" if dy < 0 else "v"


class PathCollection:
    """Holds the paths of all time points in an experiment."""

    _paths: Dict[TimePoint, Path]

    def __init__(self):
        self._paths = dict()

    def of_time_point(self, time_point: TimePoint) -> Optional[Path]:
        return self._paths.get(time_point)

    def set_path(self, time_point: TimePoint, path: Path):
        self._paths[time_point] = path
