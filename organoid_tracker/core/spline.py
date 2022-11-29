import math
from typing import List, Dict, Optional, Tuple, Iterable

import numpy
from scipy import interpolate

from organoid_tracker.core import TimePoint
from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.marker import Marker
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.vector import Vector3
from organoid_tracker.imaging import angles

_REFERENCE = Vector3(0, 0, -1)


class SplinePosition:
    """Records a position projected on a spline: both the spline, its id, the position on the spline and the distance
    to the spline are recorded."""

    spline: "Spline"  # The spline at a particular time point.
    spline_id: int  # Used to identify the data axis over multiple time points.
    pos: float  # The position on the spline in pixels.
    distance: float  # The distance from the point to the nearest point on the data axis.

    def __init__(self, axis: "Spline", pos: float, distance: float):
        self.spline = axis
        self.spline_id = 0
        self.pos = pos
        self.distance = distance

    def calculate_angle(self, position: Position, resolution: ImageResolution) -> float:
        """Calculates the angle from this point on the data axis to the given position."""

        # Calculate angle from 0 to 180
        closest_position_on_axis = self.spline.from_position_on_axis(self.pos)
        if closest_position_on_axis is None:
            print(position)
        closest_position_on_axis = Position(closest_position_on_axis[0], closest_position_on_axis[1],
                                            closest_position_on_axis[2]).to_vector_um(resolution)

        vector_towards_axis = position.to_vector_um(resolution) - closest_position_on_axis
        angle = angles.angle_between_vectors(vector_towards_axis, _REFERENCE)

        # Make angle negative
        next_position_on_axis = self.spline.from_position_on_axis(self.pos + 1)
        next_position_on_axis = Position(next_position_on_axis[0], next_position_on_axis[1],
                                         next_position_on_axis[2]).to_vector_um(resolution)
        aa = _REFERENCE.cross(vector_towards_axis)
        bb = aa.dot(next_position_on_axis - closest_position_on_axis)
        angle = numpy.sign(bb) * angle

        return angle


class Spline:
    """A curve (curved line) trough the positions. This can be used to measure how far the positions are along this
     curve.

     An offset specifies the zero-point of the axis. A checkpoint (relative to the offset) specifies some point after
     which a newregion starts. For example, in intestinal organoids this is used to mark the boundary between the crypt
     and the villus.
     """

    _x_list: List[float]
    _y_list: List[float]
    _z_list: List[float]

    _interpolation: Optional[Tuple[List[float], List[float], List[float]]]
    _offset: float

    def __init__(self):
        self._x_list = []
        self._y_list = []
        self._z_list = []
        self._interpolation = None
        self._offset = 0

    def add_point(self, x: float, y: float, z: float):
        """Adds a new point to the path."""
        self._x_list.append(float(x))
        self._y_list.append(float(y))
        self._z_list.append(float(z))

        self._interpolation = None  # Invalidate previous interpolation

    def get_points_2d(self) -> Tuple[List[float], List[float]]:
        """Gets all explicitly added points (no interpolation) without the z coord."""
        return self._x_list, self._y_list

    def get_points_3d(self) -> Tuple[List[float], List[float], List[float]]:
        """Gets all explicitly added points (no interpolation) including the z coord."""
        return self._x_list, self._y_list, self._z_list

    def length(self) -> float:
        """Gets the length of the spline."""
        x_values, y_values, z_values = self.get_interpolation_3d()
        combined_length = 0
        for i in range(1, len(x_values)):
            combined_length += _distance(x_values[i], y_values[i], z_values[i], x_values[i - 1], y_values[i - 1],
                                         z_values[i - 1])
        return combined_length

    def get_z(self) -> int:
        """Gets the average Z coord of this path. Raises ValueError if the path has no points."""
        if len(self._z_list) == 0:
            raise ValueError("Empty path, so no z is set")
        return round(sum(self._z_list) / len(self._z_list))

    def get_interpolation_2d(self) -> Tuple[List[float], List[float]]:
        """Returns a (cached) list of x and y values that are used for interpolation."""
        if self._interpolation is None:
            self._interpolation = self._calculate_interpolation()
        return self._interpolation[0], self._interpolation[1]

    def get_interpolation_3d(self) -> Tuple[List[float], List[float], List[float]]:
        """Returns a (cached) list of x, y and z values that are used for interpolation."""
        if self._interpolation is None:
            self._interpolation = self._calculate_interpolation()
        return self._interpolation

    def _calculate_interpolation(self) -> Tuple[List[float], List[float], List[float]]:
        if len(self._x_list) <= 1:
            # Not possible to interpolate
            return self._x_list, self._y_list, self._z_list

        k = 3 if len(self._x_list) > 3 else 1
        # noinspection PyTupleAssignmentBalance
        spline, _ = interpolate.splprep([self._x_list, self._y_list, self._z_list], k=k)
        points = interpolate.splev(numpy.arange(0, 1.01, 0.05), spline)
        x_values = points[0]
        y_values = points[1]
        z_values = points[2]
        return x_values, y_values, z_values

    def to_position_on_axis(self, position: Position) -> Optional[SplinePosition]:
        """Interprets this spline as an axis. Gets the closest position on this axis and the distance to the axis,
         both in pixels. Returns None if this spline has fewer than 2 points."""
        x_values, y_values, z_values = self.get_interpolation_3d()
        if len(x_values) < 2:
            return None

        # Find out which line segment is closest by
        min_distance_to_line_squared = None
        closest_line_index = None  # 1 for the first line, etc. Line 1 is from point 0 to point 1.
        for i in range(1, len(x_values)):
            line_x1 = x_values[i - 1]
            line_y1 = y_values[i - 1]
            line_z1 = z_values[i - 1]
            line_x2 = x_values[i]
            line_y2 = y_values[i]
            line_z2 = z_values[i]
            distance_squared = _distance_to_line_segment_squared(line_x1, line_y1, line_z1, line_x2, line_y2, line_z2,
                                                                 position.x, position.y, position.z)
            if min_distance_to_line_squared is None or distance_squared < min_distance_to_line_squared:
                min_distance_to_line_squared = distance_squared
                closest_line_index = i

        # Calculate length to beginning of line segment
        combined_length_of_previous_lines = 0
        for i in range(1, closest_line_index):
            combined_length_of_previous_lines += _distance(x_values[i], y_values[i], z_values[i], x_values[i - 1],
                                                           y_values[i - 1], z_values[i - 1])

        # Calculate length on line segment
        distance_to_start_of_line_squared = _distance_squared(x_values[closest_line_index - 1],
                                                              y_values[closest_line_index - 1],
                                                              z_values[closest_line_index - 1],
                                                              position.x, position.y, position.z)
        distance_on_line = numpy.sqrt(distance_to_start_of_line_squared - min_distance_to_line_squared)

        raw_path_position = combined_length_of_previous_lines + distance_on_line

        return SplinePosition(self, raw_path_position - self._offset, math.sqrt(min_distance_to_line_squared))

    def from_position_on_axis(self, path_position: float) -> Optional[Tuple[float, float, float]]:
        """Given a path position, this returns the corresponding x, y and z coordinates. Returns None for positions
        outside of the line."""
        if len(self._x_list) < 2:
            return None
        raw_path_position = path_position + self._offset
        if raw_path_position < 0:
            return None
        line_index = 1
        x_values, y_values, z_values = self.get_interpolation_3d()

        while True:
            line_length = _distance(x_values[line_index - 1], y_values[line_index - 1], z_values[line_index - 1],
                                    x_values[line_index], y_values[line_index], z_values[line_index])
            if raw_path_position < line_length or line_index == len(x_values) - 1:
                # If the position is on this line segment, or it's the last line segment, then interpolate (or
                # extrapolate) the position on that line
                line_dx = x_values[line_index] - x_values[line_index - 1]
                line_dy = y_values[line_index] - y_values[line_index - 1]
                line_dz = z_values[line_index] - z_values[line_index - 1]
                travelled_fraction = raw_path_position / line_length
                return x_values[line_index - 1] + line_dx * travelled_fraction, \
                       y_values[line_index - 1] + line_dy * travelled_fraction, \
                       z_values[line_index - 1] + line_dz * travelled_fraction

            raw_path_position -= line_length
            line_index += 1

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

    def copy(self) -> "Spline":
        """Returns a copy of this path. Changes to this path will not affect the copy and vice versa."""
        copy = Spline()
        for i in range(len(self._x_list)):
            copy.add_point(self._x_list[i], self._y_list[i], self._z_list[i])
        copy._offset = self._offset
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
        """Updates the offset of this spline such that the lowest path position that is ever returned by
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

    def move_points(self, delta: Position):
        """Translates all points in this path with the specified amount."""
        self._x_list = [x + delta.x for x in self._x_list]
        self._y_list = [y + delta.y for y in self._y_list]
        self._z_list = [z + delta.z for z in self._z_list]

        self._interpolation = None  # Invalidate previous interpolation


def _distance(x1, y1, z1, x2, y2, z2) -> float:
    """Distance between two points."""
    return numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def _distance_squared(vx, vy, vz, wx, wy, wz) -> float:
    return (vx - wx) ** 2 + (vy - wy) ** 2 + (vz - wz) ** 2


def _distance_to_line_segment_squared(line_x1, line_y1, line_z1, line_x2, line_y2, line_z2, point_x, point_y, point_z):
    """Distance from point to a line defined by the points (line_x1, line_y1, line_z1) and (line_x2, line_y2, line_z2)."""
    l2 = _distance_squared(line_x1, line_y1, line_z1, line_x2, line_y2, line_z2)
    if l2 == 0:
        return _distance_squared(point_x, point_y, point_z, line_x1, line_y1, line_z1)
    t = ((point_x - line_x1) * (line_x2 - line_x1) + (point_y - line_y1) * (line_y2 - line_y1) + (point_z - line_z1) * (
                line_z2 - line_z1)) / l2
    t = max(0, min(1, t))
    return _distance_squared(point_x, point_y, point_z,
                             line_x1 + t * (line_x2 - line_x1), line_y1 + t * (line_y2 - line_y1),
                             line_z1 + t * (line_z2 - line_z1))


class SplineCollection:
    """Holds the paths of all time points in an experiment."""

    _splines: Dict[TimePoint, Dict[int, Spline]]
    _spline_markers: Dict[int, str]  # Map of spline id -> name
    _spline_is_axis: Dict[int, bool]  # Map of spline_id -> bool
    _min_time_point_number: Optional[int]
    _max_time_point_number: Optional[int]
    _reference_time_point: Optional[TimePoint]

    def __init__(self):
        self._splines = dict()
        self._spline_markers = dict()
        self._spline_is_axis = dict()
        self._min_time_point_number = None
        self._max_time_point_number = None
        self._reference_time_point = None

    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point that contains data axes, or None if there are no axes stored."""
        return self._min_time_point_number

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) that contains data axes, or None if there are no axes stored."""
        return self._max_time_point_number

    def time_points(self) -> Iterable[TimePoint]:
        """Gets all time points with data axes in them, from first to last."""
        if self._min_time_point_number is None:
            return

        for time_point_number in range(self._min_time_point_number, self._max_time_point_number + 1):
            yield TimePoint(time_point_number)

    def reference_time_point(self, number: Optional[TimePoint] = None) -> Optional[TimePoint]:
        """Gets or sets the reference time point number. The reference time point is used to define the "original"
        axis on which a point resides. Returns None when getting the time point number if there are no data axes
        provided."""
        if number is None:
            # Get the reference time point
            if self._reference_time_point is None:
                first_time_point_number = self.first_time_point_number()
                return TimePoint(first_time_point_number) if first_time_point_number is not None else None
            return self._reference_time_point

        # Set the reference time point
        self._reference_time_point = number
        if self._reference_time_point.time_point_number() == self.first_time_point_number():
            self._reference_time_point = None  # The first time point number is the default

    def of_time_point(self, time_point: TimePoint) -> Iterable[Tuple[int, Spline]]:
        """Gets the data axes of the time point along with their id, or an empty collection if that time point has no
        paths defined."""
        splines = self._splines.get(time_point)
        if splines is None:
            return []
        for spline_id, spline in splines.items():
            yield spline_id, spline

    def to_position_on_spline(self, position: Position, only_axis=False) -> Optional[SplinePosition]:
        # Find the closest axis, return position on that axis
        lowest_distance_position = None
        for axis_id, data_axis in self.of_time_point(position.time_point()):
            if only_axis and not self._spline_is_axis.get(axis_id):
                continue  # Ignore axes of this type

            axis_position = data_axis.to_position_on_axis(position)
            if axis_position is None:
                continue
            axis_position.spline_id = axis_id
            if lowest_distance_position is None or axis_position.distance < lowest_distance_position.distance:
                lowest_distance_position = axis_position
        return lowest_distance_position

    def to_position_on_original_axis(self, links: Links, position: Position) -> Optional[SplinePosition]:
        """Gets the position on the axis that was closest in the first time point this position appeared. In this way,
        every position is assigned to a single axis, and will never switch to another axis during its lifetime."""
        reference_time_point = self.reference_time_point()
        if reference_time_point is None:
            return None
        first_position = links.get_position_at_time_point(position, reference_time_point)
        if first_position is None:
            return None
        first_axis_position = self.to_position_on_spline(first_position, only_axis=True)
        if first_axis_position is None:
            return None
        for axis_id, axis in self.of_time_point(position.time_point()):
            if axis_id == first_axis_position.spline_id:
                position = axis.to_position_on_axis(position)
                if position is not None:
                    position.spline_id = axis_id
                return position

    def add_spline(self, time_point: TimePoint, path: Spline, spline_id: Optional[int]) -> int:
        """Adds a new spline to the given time point. If another spline with that id already exists in the time point,
        it is overwritten. If spline_id is None, a new id will be assigned. The spline_id is then returned."""
        existing_splines = self._splines.get(time_point)
        if existing_splines is None:
            # No splines yet for that time point
            if spline_id is None:
                spline_id = 1

            # Add data for a new time point
            self._splines[time_point] = {spline_id: path}

            # Update min/max time points
            if self._max_time_point_number is None or time_point.time_point_number() > self._max_time_point_number:
                self._max_time_point_number = time_point.time_point_number()
            if self._min_time_point_number is None or time_point.time_point_number() < self._min_time_point_number:
                self._min_time_point_number = time_point.time_point_number()
        else:
            if spline_id is None:
                # Find a free spline id
                spline_id = 1
                while spline_id in existing_splines:
                    spline_id += 1
            existing_splines[spline_id] = path
        return spline_id

    def remove_spline(self, time_point: TimePoint, spline_to_remove: Spline):
        """Removes the given data axis from the given time point. Does nothing if the data axis is not used for the
        given time point."""
        existing_paths = self._splines.get(time_point)
        if existing_paths is None:
            return

        # Find spline id
        spline_id_to_remove = None
        for spline_id, spline in existing_paths.items():
            if spline == spline_to_remove:
                spline_id_to_remove = spline_id
                break
        if spline_id_to_remove is None:
            return  # Ignore, spline is not in list

        del existing_paths[spline_id_to_remove]
        if len(existing_paths) == 0:
            # We just removed the last path of this time point
            del self._splines[time_point]
            if time_point.time_point_number() == self._min_time_point_number \
                    or time_point.time_point_number() == self._max_time_point_number:
                self._recalculate_min_max_time_point()  # Removed first or last time point, calculate new min/max

    def exists(self, path: Spline, time_point: TimePoint) -> bool:
        """Returns True if the path exists in this path collection at the given time point, False otherwise."""
        paths = self._splines.get(time_point)
        if paths is None:
            return False
        for found_path in paths.values():
            if found_path == path:
                return True
        return False

    def get_spline(self, time_point: TimePoint, spline_id: int) -> Optional[Spline]:
        """Gets the spline with the given id and time point. Returns None if not found."""
        splines_of_time_point = self._splines.get(time_point)
        if splines_of_time_point is None:
            return None
        return splines_of_time_point.get(spline_id)

    def is_axis(self, spline_id: int) -> bool:
        """Returns true if the spline with the given id is an axis."""
        return bool(self._spline_is_axis.get(spline_id))

    def has_splines(self) -> bool:
        """Returns True if there are any paths stored in this collection. """
        return len(self._splines) > 0

    def all_splines(self) -> Iterable[Tuple[int, TimePoint, Spline]]:
        """Gets all paths. Note that a single time point can have multiple paths."""
        for time_point, paths in self._splines.items():
            for spline_id, spline in paths.items():
                yield spline_id, time_point, spline

    def _recalculate_min_max_time_point(self):
        """Recalculates the min/max time point based on the current data axis map. Call this method when the last axis
        of a time point has been removed."""
        min_time_point_number = None
        max_time_point_number = None
        for time_point in self._splines.keys():
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
        splines = self._splines.get(time_point)
        if splines is None:
            return
        for spline in splines.values():
            spline.update_offset_for_positions(new_positions)

    def set_marker(self, axis_id: int, axis_marker: Optional[Marker]):
        """Sets the marker of the specified data axes across all time points. High-level version of set_marker_name.
        If the marker has the extra data "is_axis": true, then the spline will be used as an axis."""
        save_name = None
        is_data_axis = False
        if axis_marker is not None:
            if not axis_marker.applies_to(Spline):
                raise ValueError(f"Type {axis_marker} cannot be applied to a data axis")

            save_name = axis_marker.save_name
            is_data_axis = bool(axis_marker.extra("is_axis"))
        self.set_marker_name(axis_id, save_name, is_data_axis)

    def set_marker_name(self, axis_id: int, axis_marker: Optional[str], is_data_axis: bool):
        """Sets the marker of the specified data axes across all time points """
        if axis_marker is None:
            # Remove
            if axis_id in self._spline_markers:
                del self._spline_markers[axis_id]
                del self._spline_is_axis[axis_id]
            return

        # Set marker
        self._spline_markers[axis_id] = axis_marker.upper()
        self._spline_is_axis[axis_id] = is_data_axis

    def get_marker_name(self, axis_id: int) -> Optional[str]:
        """Gets the marker of the data axes with the given id."""
        return self._spline_markers.get(axis_id)

    def get_marker_names(self) -> Iterable[Tuple[int, str]]:
        """Gets all registered axis markers as (id, name)."""
        for axis_id, marker_name in self._spline_markers.items():
            yield axis_id, marker_name
