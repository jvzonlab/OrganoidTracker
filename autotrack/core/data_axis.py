import math
from typing import List, Dict, Optional, Tuple, Iterable

import numpy
from scipy import interpolate

from autotrack.core import TimePoint
from autotrack.core.links import Links
from autotrack.core.position import Position


class DataAxisPosition:
    axis: "DataAxis"  # The data axis at a particular time point.
    axis_id: int  # Used to identify the data axis over multiple time points.
    pos: float  # The position on the data axis.
    distance: float  # The distance from the point to the nearest point on the data axis.

    def __init__(self, axis: "DataAxis", pos: float, distance: float):
        self.axis = axis
        self.axis_id = 0
        self.pos = pos
        self.distance = distance

    def is_after_checkpoint(self) -> bool:
        """Returns True if the data axis has a checkpoint specified and this position is behind that checkpoint."""
        checkpoint = self.axis.get_checkpoint()
        if checkpoint is None:
            return False
        return self.pos > checkpoint


class DataAxis:
    """A curve (curved line) trough the positions. This can be used to measure how far the positions are along this
     curve.

     An offset specifies the zero-point of the axis. A checkpoint (relative to the offset) specifies some point after
     which a newregion starts. For example, in intestinal organoids this is used to mark the boundary between the crypt
     and the villus.
     """

    _x_list: List[float]
    _y_list: List[float]
    _z: Optional[int]

    _interpolation: Optional[Tuple[List[float], List[float]]]
    _offset: float
    _checkpoint_without_offset: Optional[float]

    def __init__(self):
        self._x_list = []
        self._y_list = []
        self._z = None
        self._interpolation = None
        self._offset = 0
        self._checkpoint_without_offset = None

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
        """Gets the Z coord of this path. Raises ValueError if the path has no points."""
        if self._z is None:
            raise ValueError("Empty path, so no z is set")
        return self._z

    def get_interpolation_2d(self) -> Tuple[List[float], List[float]]:
        """Returns a (cached) list of x and y values that are used for interpolation."""
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

    def to_position_on_axis(self, position: Position) -> Optional[DataAxisPosition]:
        """Gets the closest position on the axes and the distance to the axes, both in pixels. Returns None if the path
        has fewer than 2 points."""
        x_values, y_values = self.get_interpolation_2d()
        if len(x_values) < 2:
            return None

        # Find out which line segment is closest by
        min_distance_to_line_squared = None
        closest_line_index = None  # 1 for the first line, etc. Line 1 is from point 0 to point 1.
        for i in range(1, len(x_values)):
            line_x1 = x_values[i - 1]
            line_y1 = y_values[i - 1]
            line_x2 = x_values[i]
            line_y2 = y_values[i]
            distance_squared = _distance_to_line_segment_squared(line_x1, line_y1, line_x2, line_y2, position.x,
                                                                 position.y)
            if min_distance_to_line_squared is None or distance_squared < min_distance_to_line_squared:
                min_distance_to_line_squared = distance_squared
                closest_line_index = i

        # Calculate length to beginning of line segment
        combined_length_of_previous_lines = 0
        for i in range(1, closest_line_index):
            combined_length_of_previous_lines += _distance(x_values[i], y_values[i], x_values[i - 1], y_values[i - 1])

        # Calculate length on line segment
        distance_to_start_of_line_squared = _distance_squared(x_values[closest_line_index - 1],
                                                              y_values[closest_line_index - 1],
                                                              position.x, position.y)
        distance_on_line = numpy.sqrt(distance_to_start_of_line_squared - min_distance_to_line_squared)

        raw_path_position = combined_length_of_previous_lines + distance_on_line
        return DataAxisPosition(self, raw_path_position - self._offset, math.sqrt(min_distance_to_line_squared))

    def from_position_on_axis(self, path_position: float) -> Optional[Tuple[float, float]]:
        """Given a path position, this returns the corresponding x and y coordinates. Returns None for positions outside
        of the line."""
        if len(self._x_list) < 2:
            return None
        raw_path_position = path_position + self._offset
        if raw_path_position < 0:
            return None
        line_index = 1
        x_values, y_values = self.get_interpolation_2d()

        while True:
            line_length = _distance(x_values[line_index - 1], y_values[line_index - 1],
                                    x_values[line_index], y_values[line_index])
            if raw_path_position < line_length:
                line_dx = x_values[line_index] - x_values[line_index - 1]
                line_dy = y_values[line_index] - y_values[line_index - 1]
                travelled_fraction = raw_path_position / line_length
                return x_values[line_index - 1] + line_dx * travelled_fraction, \
                       y_values[line_index - 1] + line_dy * travelled_fraction

            raw_path_position -= line_length
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

    def __eq__(self, other):
        # Paths are only equal if they are the same instence
        return other is self

    def copy(self) -> "DataAxis":
        """Returns a copy of this path. Changes to this path will not affect the copy and vice versa."""
        copy = DataAxis()
        for i in range(len(self._x_list)):
            copy.add_point(self._x_list[i], self._y_list[i], self._z)
        copy._offset = self._offset
        copy._checkpoint_without_offset = self._checkpoint_without_offset
        return copy

    def remove_point(self, x: float, y: float):
        """Removes the point that is (within 1 px) at the given coords. Does nothing if there is no such point."""
        for i in range(len(self._x_list)):
            if abs(self._x_list[i] - x) < 1 and abs(self._y_list[i] - y) < 1:
                del self._x_list[i]
                del self._y_list[i]
                self._interpolation = None  # Interpolation is now outdated
                return

    def update_offset_for_positions(self, positions: Iterable[Position]):
        """Updates the offset of this crypt axis such that the lowest path position that is ever returned by
        get_path_position_2d is exactly 0.
        """
        if len(self._x_list) < 2:
            return  # Too small path to update

        current_lowest_position = None
        for position in positions:
            path_position = self.to_position_on_axis(position).pos
            if current_lowest_position is None or path_position < current_lowest_position:
                current_lowest_position = path_position
        if current_lowest_position is not None:  # Don't do anything if the list of positions was empty
            self._offset += current_lowest_position

    def set_offset(self, offset: float):
        """Manually sets the offset used in calls to get_path_position_2d and path_position_to_xy. See also
        update_offset_for_positions."""
        self._offset = float(offset)

    def get_offset(self) -> float:
        """Gets the offset used in calls to get_path_position_2d and path_position_to_xy. Note that these methods apply
        the offset automatically, so except for saving/loading purposes there should be no need to call this method."""
        return self._offset

    def set_checkpoint(self, checkpoint: Optional[float]):
        """Updates the checkpoint. See the class docs for what a checkpoint is. If you update the offset later on,
        the checkpoint will change relative to the offset, so that its absolute position (xyz) will stay the same."""
        if checkpoint is None:
            self._checkpoint_without_offset = None
            return
        self._checkpoint_without_offset = checkpoint - self._offset

    def get_checkpoint(self) -> Optional[float]:
        """Gets the checkpoint. See the class docs for what a checkpoint is."""
        if self._checkpoint_without_offset is None:
            return None
        return self._checkpoint_without_offset + self._offset

    def move_points(self, delta: Position):
        """Translates all points in this path with the specified amount."""
        self._x_list = [x + delta.x for x in self._x_list]
        self._y_list = [y + delta.y for y in self._y_list]
        self._z += int(delta.z)

        self._interpolation = None  # Invalidate previous interpolation

def _distance(x1, y1, x2, y2):
    """Distance between two points."""
    return numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _distance_squared(vx, vy, wx, wy):
    return (vx - wx) ** 2 + (vy - wy) ** 2


def _distance_to_line_segment_squared(line_x1, line_y1, line_x2, line_y2, point_x, point_y):
    """Distance from point to a line defined by the points (line_x1, line_y1) and (line_x2, line_y2)."""
    l2 = _distance_squared(line_x1, line_y1, line_x2, line_y2)
    if l2 == 0:
        return _distance_squared(point_x, point_y, line_x1, line_y1)
    t = ((point_x - line_x1) * (line_x2 - line_x1) + (point_y - line_y1) * (line_y2 - line_y1)) / l2
    t = max(0, min(1, t))
    return _distance_squared(point_x, point_y,
                             line_x1 + t * (line_x2 - line_x1), line_y1 + t * (line_y2 - line_y1))


class DataAxisCollection:
    """Holds the paths of all time points in an experiment."""

    _data_axes: Dict[TimePoint, List[DataAxis]]
    _min_time_point_number: Optional[int]
    _max_time_point_number: Optional[int]

    def __init__(self):
        self._data_axes = dict()
        self._min_time_point_number = None
        self._max_time_point_number = None

    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point that contains data axes, or None if there are no axes stored."""
        return self._min_time_point_number

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) that contains data axes, or None if there are no axes stored."""
        return self._max_time_point_number

    def of_time_point(self, time_point: TimePoint) -> Iterable[Tuple[int, DataAxis]]:
        """Gets the data axes of the time point along with their id, or an empty collection if that time point has no
        paths defined."""
        data_axes = self._data_axes.get(time_point)
        if data_axes is None:
            return []
        for i, data_axis in enumerate(data_axes):
            yield i + 1, data_axis

    def _to_position_on_axis(self, position: Position) -> Optional[DataAxisPosition]:
        # Find the closest axis, return position on that axis
        lowest_distance_position = None
        for axis_id, data_axis in self.of_time_point(position.time_point()):
            axis_position = data_axis.to_position_on_axis(position)
            if axis_position is None:
                continue
            axis_position.axis_id = axis_id
            if lowest_distance_position is None or axis_position.distance < lowest_distance_position.distance:
                lowest_distance_position = axis_position
        return lowest_distance_position

    def to_position_on_original_axis(self, links: Links, position: Position) -> Optional[DataAxisPosition]:
        """Gets the position on the axis that was closest in the first time point this position appeared. In this way,
        every position is assigned to a single axis, and will never switch to another axis during its lifetime."""
        first_position = links.get_first_position_of(position)
        first_axis_position = self._to_position_on_axis(first_position)
        if first_axis_position is None:
            return None
        for axis_id, axis in self.of_time_point(position.time_point()):
            if axis_id == first_axis_position.axis_id:
                position = axis.to_position_on_axis(position)
                if position is not None:
                    position.axis_id = axis_id
                return position

    def add_data_axis(self, time_point: TimePoint, path: DataAxis):
        """Adds a new data axis to the given time point. Existing axes are left untouched."""
        existing_data_axes = self._data_axes.get(time_point)
        if existing_data_axes is None:
            # Add data for a new time point
            self._data_axes[time_point] = [path]

            # Update min/max time points
            if self._max_time_point_number is None or time_point.time_point_number() > self._max_time_point_number:
                self._max_time_point_number = time_point.time_point_number()
            if self._min_time_point_number is None or time_point.time_point_number() < self._min_time_point_number:
                self._min_time_point_number = time_point.time_point_number()
        else:
            existing_data_axes.append(path)

    def remove_data_axis(self, time_point: TimePoint, path: DataAxis):
        """Removes the given data axis from the given time point. Does nothing if the data axis is not used for the
        given time point."""
        existing_paths = self._data_axes.get(time_point)
        if existing_paths is None:
            return
        try:
            existing_paths.remove(path)
            if len(existing_paths) == 0:
                # We just removed the last path of this time point
                del self._data_axes[time_point]
                if time_point.time_point_number() == self._min_time_point_number\
                        or time_point.time_point_number() == self._max_time_point_number:
                    self._recalculate_min_max_time_point()  # Removed first or last time point, calculate new min/max
        except ValueError:
            pass  # Ignore, path is not in list

    def exists(self, path: DataAxis, time_point: TimePoint) -> bool:
        """Returns True if the path exists in this path collection at the given time point, False otherwise."""
        paths = self._data_axes.get(time_point)
        if paths is None:
            return False
        return path in paths

    def has_axes(self) -> bool:
        """Returns True if there are any paths stored in this collection. """
        return len(self._data_axes) > 0

    def all_data_axes(self) -> Iterable[Tuple[DataAxis, TimePoint]]:
        """Gets all paths. Note that a single time point can have multiple paths."""
        for time_point, paths in self._data_axes.items():
            for path in paths:
                yield path, time_point

    def _recalculate_min_max_time_point(self):
        """Recalculates the min/max time point based on the current data axis map. Call this method when the last axis
        of a time point has been removed."""
        min_time_point_number = None
        max_time_point_number = None
        for time_point in self._data_axes.keys():
            if min_time_point_number is None or time_point.time_point_number() < min_time_point_number:
                min_time_point_number = time_point.time_point_number()
            if max_time_point_number is None or time_point.time_point_number() > max_time_point_number:
                max_time_point_number = time_point.time_point_number()
        self._min_time_point_number = min_time_point_number
        self._max_time_point_number = max_time_point_number

    def update_for_changed_positions(self, time_point: TimePoint, new_positions: Iterable[Position]):
        """If the positions of a time point have changed, this method must be called to update the zero point of all
        axes.
        """
        data_axes = self._data_axes.get(time_point)
        if data_axes is None:
            return
        for data_axis in data_axes:
            data_axis.update_offset_for_positions(new_positions)
