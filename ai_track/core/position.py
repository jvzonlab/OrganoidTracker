import math
from typing import Optional, Union, List

from ai_track.core import TimePoint
from ai_track.core.resolution import ImageResolution
from ai_track.core.vector import Vector3


class Position:
    """A detected position. Only the 3D + time position is stored here, see the PositionShape class for the shape.
    The position is immutable."""

    @staticmethod
    def from_vector_um(vector: Vector3, resolution: ImageResolution, *,
                       time_point: Optional[TimePoint] = None, time_point_number: Optional[int] = None) -> "Position":
        """Returns a position from the given vector, which is assumed to be in micrometers. If you want, you can
        specify a time point."""
        return Position(vector.x / resolution.pixel_size_x_um,
                        vector.y / resolution.pixel_size_y_um,
                        vector.z / resolution.pixel_size_z_um,
                        time_point=time_point, time_point_number=time_point_number)

    __slots__ = ["x", "y", "z", "_time_point_number"]  # Optimization - Google "python slots"

    x: float  # Read-only
    y: float  # Read-only
    z: float  # Read-only
    _time_point_number: Optional[int]

    def __init__(self, x: float, y: float, z: float, *,
                 time_point: Optional[TimePoint] = None, time_point_number: Optional[int] = None):
        """Constructs a new position, optionally with either a time point or a time point number."""
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        if time_point is not None:
            if time_point_number is not None:
                raise ValueError("Both time_point and time_point_number params are set; use only one of them")
            self._time_point_number = time_point.time_point_number()
        elif time_point_number is not None:
            self._time_point_number = int(time_point_number)
        else:
            self._time_point_number = None

    def distance_squared(self, other: "Position", resolution: ImageResolution) -> float:
        """Gets the squared distance in micrometers. Working with squared distances instead of normal ones gives a much
        better performance, as the expensive sqrt(..) function can be avoided."""
        dx = (self.x - other.x) * resolution.pixel_size_zyx_um[2]
        dy = (self.y - other.y) * resolution.pixel_size_zyx_um[1]
        dz = (self.z - other.z) * resolution.pixel_size_zyx_um[0]
        return dx ** 2 + dy ** 2 + dz ** 2

    def distance_um(self, other: "Position", resolution: ImageResolution) -> float:
        """Gets the distance to the other position in micrometers."""
        return math.sqrt(self.distance_squared(other, resolution))

    def time_point_number(self) -> Optional[int]:
        return self._time_point_number

    def __repr__(self):
        string = "Position(" + ("%.2f" % self.x) + ", " + ("%.2f" % self.y) + ", " + ("%.0f" % self.z) + ")"
        if self._time_point_number is not None:
            string += ".with_time_point_number(" + str(self._time_point_number) + ")"
        return string

    def __str__(self):
        string = "cell at (" + ("%.2f" % self.x) + ", " + ("%.2f" % self.y) + ", " + ("%.2f" % self.z) + ")"
        if self._time_point_number is not None:
            string += " at time point " + str(self._time_point_number)
        return string

    def __hash__(self) -> int:
        return int(self.x) ^ hash(self._time_point_number)

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        try:
            if other._time_point_number != self._time_point_number:
                return False
            if self.x != other.x:
                if abs(self.x - other.x) > 0.01:
                    return False
            if self.y != other.y:
                if abs(self.y - other.y) > 0.01:
                    return False
            if self.z != other.z:
                if abs(self.z - other.z) > 0.01:
                    return False
            return True
        except AttributeError:
            return False

    def time_point(self):
        """Gets the time point of this position. Note: getting the time point number is slightly more efficient, as
        this method requires allocating a new TimePoint instance."""
        return TimePoint(self._time_point_number)

    def is_zero(self) -> bool:
        """Returns True if the X, Y and Z are exactly zero. Time is ignored."""
        return self.x == 0 and self.y == 0 and self.z == 0

    def __sub__(self, other: "Position") -> "Position":
        """Returns a new position (without a time specified) that is the difference between this position and the other
        position. The time point of the other position is ignored, the time point of the new position will be equal to
        the time point of this position."""
        if not isinstance(other, Position):
            return NotImplemented
        return Position(self.x - other.x, self.y - other.y, self.z - other.z, time_point_number=self._time_point_number)

    def check_time_point(self, time_point: TimePoint):
        """Raises a ValueError if this position has no time point set, or if it has a time point that is not equal to
        the given time point."""
        if self._time_point_number != time_point.time_point_number():
            raise ValueError(f"Time points don't match: self is in {self._time_point_number}, other in"
                             f" {time_point.time_point_number()}")

    def __add__(self, other: "Position") -> "Position":
        """Returns a new position (without a time specified) that is the sum of this position and the other position.
        The time point of the other position is ignored, the time point of the new position will be equal to the time
        point of this position."""
        if not isinstance(other, Position):
            return NotImplemented
        if other.x == 0 and other.y == 0 and other.z == 0:
            return self  # No need to add anything
        return Position(self.x + other.x, self.y + other.y, self.z + other.z, time_point_number=self._time_point_number)

    def __mul__(self, other: Union[float, "Position"]) -> "Position":
        """Multiplication. If you multiply with a number a, this will return a position (x * a, y * a, z * a).
        If you multiply with another position, (x * a.x, y * a.y, z * a.z) will be returned. The time point number is
        unaffected."""
        if not isinstance(other, float) and not isinstance(other, int) and not isinstance(other, Position):
            return NotImplemented
        if other == 1:
            return self
        if isinstance(other, Position):
            return Position(self.x * other.x, self.y * other.y, self.z * other.z,
                            time_point_number=self.time_point_number())
        return Position(self.x * other, self.y * other, self.z * other, time_point_number=self._time_point_number)

    def __truediv__(self, other) -> "Position":
        """Scalar division. The time point number is unaffected."""
        if not isinstance(other, float) and not isinstance(other, int):
            return NotImplemented
        if other == 1:
            return self
        return Position(self.x / other, self.y / other, self.z / other, time_point_number=self._time_point_number)

    def with_time_point(self, time_point: Optional[TimePoint]) -> "Position":
        """Returns a copy of this position with the time point set to the given position."""
        return Position(self.x, self.y, self.z, time_point=time_point)

    def with_time_point_number(self, time_point_number: int):
        """Returns a copy of this position with the time point set to the given position."""
        return Position(self.x, self.y, self.z, time_point_number=time_point_number)

    def interpolate(self, to_pos: "Position") -> List["Position"]:
        """Gets a time-interpolated list of positions. If you have positions A and B, with one time point in between,
        then a list will be returned of three elements: [A, I, B], with I an interpolated position. If there are two
        time points in between, then a list of four elements will be returned, and so on.

        If there are no time points in between the two positions, then the two positions are simply returned. If both
        positions are in the same time point, then this method will raise ValueError. The returned list will always
        start with the earliest position first."""
        from_pos = self
        if to_pos.time_point_number() < from_pos.time_point_number():
            to_pos, from_pos = from_pos, to_pos
        delta_time = to_pos.time_point_number() - from_pos.time_point_number()
        if delta_time == 1:
            return [from_pos, to_pos]
        if delta_time == 0:
            raise ValueError(f"The {self} is at the same time point as {to_pos}")

        return_list = [from_pos]
        for i in range(1, delta_time):
            fraction = i / delta_time
            position = to_pos * fraction + from_pos * (1 - fraction)
            return_list.append(position.with_time_point_number(from_pos.time_point_number() + i))
        return_list.append(to_pos)
        return return_list

    def to_vector_um(self, resolution: ImageResolution) -> Vector3:
        return Vector3(self.x * resolution.pixel_size_x_um, self.y * resolution.pixel_size_y_um, self.z * resolution.pixel_size_z_um)
