from typing import List, Dict, Optional, Tuple

import numpy
from scipy import interpolate
from scipy.spatial import distance

from autotrack.core import TimePoint
from autotrack.core.particles import Particle


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

    def get_path_position_2d(self, particle: Particle):
        """Gets the closest position on the path. The position is returned in pixels from the path start."""
        x_values, y_values = self.get_interpolation_2d()

        # Find out which line segment is closest by
        min_distance_to_line_squared = None
        closest_line_index = None  # 1 for the first line, etc. Line 1 is from point 0 to point 1.
        for i in range(1, len(x_values)):
            line_x1 = x_values[i - 1]
            line_y1 = y_values[i - 1]
            line_x2 = x_values[i]
            line_y2 = y_values[i]
            distance_squared = _distance_to_line_segment_squared(line_x1, line_y1, line_x2, line_y2, particle.x, particle.y)
            if min_distance_to_line_squared is None or distance_squared < min_distance_to_line_squared:
                min_distance_to_line_squared = distance_squared
                closest_line_index = i

        # Calculate length to beginning of line segment
        combined_length_of_previous_lines = 0
        for i in range(1, closest_line_index):
            combined_length_of_previous_lines += _distance(x_values[i], y_values[i], x_values[i - 1], y_values[i - 1])

        # Calculate length on line segment
        distance_to_start_of_line_squared = _distance_squared(x_values[closest_line_index - 1], y_values[closest_line_index - 1],
                                              particle.x, particle.y)
        distance_on_line = numpy.sqrt(distance_to_start_of_line_squared - min_distance_to_line_squared)

        return combined_length_of_previous_lines + distance_on_line

    def path_position_to_xy(self, path_position: float) -> Optional[Tuple[float, float]]:
        """Given a path position, this returns the corresponding x and y coordinates. Returns None for positions outside
        of the line."""
        if path_position < 0:
            return None
        line_index = 1
        x_values, y_values = self.get_interpolation_2d()

        while True:
            line_length = _distance(x_values[line_index - 1], y_values[line_index - 1],
                                    x_values[line_index], y_values[line_index])
            if path_position < line_length:
                line_dx = x_values[line_index] - x_values[line_index - 1]
                line_dy = y_values[line_index] - y_values[line_index - 1]
                travelled_fraction = path_position / line_length
                return x_values[line_index - 1] + line_dx * travelled_fraction, \
                       y_values[line_index - 1] + line_dy * travelled_fraction

            path_position -= line_length
            line_index += 1
            if line_index >= len(x_values):
                return None

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


def _distance(x1, y1, x2, y2):
    """Distance between two points."""
    return numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _distance_squared(vx, vy, wx, wy):
    return (vx - wx)**2 + (vy - wy)**2


def _distance_to_line_segment_squared(line_x1, line_y1, line_x2, line_y2, point_x, point_y):
    """Distance from point to a line defined by the points (line_x1, line_y1) and (line_x2, line_y2)."""
    l2 = _distance_squared(line_x1, line_y1, line_x2, line_y2)
    if l2 == 0:
         return _distance_squared(point_x, point_y, line_x1, line_y1)
    t = ((point_x - line_x1) * (line_x2 - line_x1) + (point_y - line_y1) * (line_y2 - line_y1)) / l2
    t = max(0, min(1, t))
    return _distance_squared(point_x, point_y,
                             line_x1 + t * (line_x2 - line_x1), line_y1 + t * (line_y2 - line_y1))


class PathCollection:
    """Holds the paths of all time points in an experiment."""

    _paths: Dict[TimePoint, Path]

    def __init__(self):
        self._paths = dict()

    def of_time_point(self, time_point: TimePoint) -> Optional[Path]:
        return self._paths.get(time_point)

    def set_path(self, time_point: TimePoint, path: Path):
        self._paths[time_point] = path
