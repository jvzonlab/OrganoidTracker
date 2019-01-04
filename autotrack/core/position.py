import math
from typing import Optional

from autotrack.core import TimePoint
from autotrack.core.resolution import ImageResolution


class Position:
    """A detected position. Only the 3D + time position is stored here, see the PositionShape class for the shape.
    The position is immutable."""

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

    def distance_squared(self, other: "Position", z_factor: float = 5) -> float:
        """Gets the squared distance. Working with squared distances instead of normal ones gives a much better
        performance, as the expensive sqrt(..) function can be avoided."""
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + ((self.z - other.z) * z_factor) ** 2

    def distance_um(self, other: "Position", resolution: ImageResolution) -> float:
        """Gets the distance to the other position in micrometers."""
        dx = (self.x - other.x) * resolution.pixel_size_zyx_um[2]
        dy = (self.y - other.y) * resolution.pixel_size_zyx_um[1]
        dz = (self.z - other.z) * resolution.pixel_size_zyx_um[0]
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

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

    def __hash__(self):
        if self._time_point_number is None:
            return hash(int(self.x)) ^ hash(int(self.y)) ^ hash(int(self.z))
        return hash(int(self.x)) ^ hash(int(self.y)) ^ hash(int(self.z)) ^ hash(int(self._time_point_number))

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and abs(self.x - other.x) < 0.00001 and abs(self.x - other.x) < 0.00001 and abs(self.z - other.z) < 0.00001 \
               and self._time_point_number == other._time_point_number

    def time_point(self):
        """Gets the time point of this position. Note: getting the time point number is slightly more efficient, as
        this method requires allocating a new TimePoint instance."""
        return TimePoint(self._time_point_number)

    def is_zero(self) -> bool:
        """Returns True if the X, Y and Z are exactly zero. Time is ignored."""
        return self.x == 0 and self.y == 0 and self.z == 0

    def subtract_pos(self, other: "Position") -> "Position":
        """Returns a new position (without a time specified) that is the difference between this position and the other
        position. The time point of the other position is ignored, the time point of the new position will be equal to
        the time point of this position."""
        return Position(self.x - other.x, self.y - other.y, self.z - other.z, time_point_number=self._time_point_number)

    def check_time_point(self, time_point: TimePoint):
        """Raises a ValueError if this position has no time point set, or if it has a time point that is not equal to
        the given time point."""
        if self._time_point_number != time_point.time_point_number():
            raise ValueError(f"Time points don't match: self is in {self._time_point_number}, other in"
                             f" {time_point.time_point_number()}")

    def add_pos(self, other: "Position") -> "Position":
        """Returns a new position (without a time specified) that is the sum of this position and the other position.
        The time point of the other position is ignored, the time point of the new position will be equal to the time
        point of this position."""
        if other.x == 0 and other.y == 0 and other.z == 0:
            return self  # No need to add anything
        return Position(self.x + other.x, self.y + other.y, self.z + other.z, time_point_number=self._time_point_number)

    def with_time_point(self, time_point: Optional[TimePoint]) -> "Position":
        """Returns a copy of this position with the time point set to the given position."""
        return Position(self.x, self.y, self.z, time_point=time_point)
