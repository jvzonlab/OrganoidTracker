from operator import itemgetter
from typing import Dict, AbstractSet, Optional, Iterable, Set

import math

from autotrack.core import TimePoint
from autotrack.core.resolution import ImageResolution
from autotrack.core.shape import ParticleShape, UnknownShape


class Position:
    """A detected position. Only the 3D + time position is stored here, see the PositionShape class for the shape."""

    __slots__ = ["x", "y", "z", "_time_point_number"]  # Optimization - Google "python slots"

    x: float
    y: float
    z: float
    _time_point_number: Optional[int]

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
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

    def with_time_point_number(self, time_point_number: int) -> "Position":
        if self._time_point_number is not None and self._time_point_number != time_point_number:
            raise ValueError("time_point_number was already set")
        self._time_point_number = int(time_point_number)
        return self

    def with_time_point(self, time_point: TimePoint) -> "Position":
        return self.with_time_point_number(time_point.time_point_number())

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
        """Gets the time point of this position."""
        return TimePoint(self._time_point_number)


class _PositionsAtTimePoint:
    """Holds the positions of a single point in time."""

    _positions: Dict[Position, ParticleShape]

    def __init__(self):
        self._positions = dict()
        self._mother_scores = dict()

    def positions(self) -> AbstractSet[Position]:
        return self._positions.keys()

    def positions_and_shapes(self) -> Dict[Position, ParticleShape]:
        return self._positions

    def get_shape(self, position: Position) -> ParticleShape:
        """Gets the shape of a position. Returns UnknownShape if the given position is not part of this time point."""
        shape = self._positions.get(position)
        if shape is None:
            return UnknownShape()
        return shape

    def add_position(self, position: Position, position_shape: Optional[ParticleShape]):
        """Adds a position to this time point. If the position was already added, but a shape was provided, its shape is
        replaced."""
        if position_shape is None:
            if position in self._positions:
                return  # Don't overwrite known shape with an unknown shape.
            position_shape = UnknownShape()  # Don't use None as value in the dict
        self._positions[position] = position_shape

    def detach_position(self, position: Position):
        """Removes a single position. Raises KeyError if that position was not in this time point. Does not remove a
        position from the linking graph. See also Experiment.remove_position."""
        del self._positions[position]

    def is_empty(self):
        """Returns True if there are no positions stored."""
        return len(self._positions) == 0

    def __len__(self):
        return len(self._positions)


class PositionCollection:

    _all_positions: Dict[int, _PositionsAtTimePoint]
    _min_time_point_number: Optional[int] = None
    _max_time_point_number: Optional[int] = None

    def __init__(self):
        self._all_positions = dict()

    def of_time_point(self, time_point: TimePoint) -> AbstractSet[Position]:
        """Returns all positions for a given time point. Returns an empty set if that time point doesn't exist."""
        positions_at_time_point = self._all_positions.get(time_point.time_point_number())
        if not positions_at_time_point:
            return set()
        return positions_at_time_point.positions()

    def detach_all_for_time_point(self, time_point: TimePoint):
        """Removes all positions for a given time point, if any."""
        if time_point.time_point_number() in self._all_positions:
            del self._all_positions[time_point.time_point_number()]
            self._update_min_max_time_points_for_removal()

    def add(self, position: Position, shape: Optional[ParticleShape] = None):
        """Adds a position, optionally with the given shape. The position must have a time point specified."""
        time_point_number = position.time_point_number()
        if time_point_number is None:
            raise ValueError("Position does not have a time point, so it cannot be added")

        self._update_min_max_time_points_for_addition(time_point_number)

        positions_at_time_point = self._all_positions.get(time_point_number)
        if positions_at_time_point is None:
            positions_at_time_point = _PositionsAtTimePoint()
            self._all_positions[time_point_number] = positions_at_time_point
        positions_at_time_point.add_position(position, shape)

    def _update_min_max_time_points_for_addition(self, new_time_point_number: int):
        """Bookkeeping: makes sure the min and max time points are updated when a new time point is added"""
        if self._min_time_point_number is None or new_time_point_number < self._min_time_point_number:
            self._min_time_point_number = new_time_point_number
        if self._max_time_point_number is None or new_time_point_number > self._max_time_point_number:
            self._max_time_point_number = new_time_point_number

    def _update_min_max_time_points_for_removal(self):
        """Bookkeeping: recalculates min and max time point if a time point was removed."""
        # Reset min and max, then repopulate by readding all time points
        self._min_time_point_number = None
        self._max_time_point_number = None
        for time_point_number in self._all_positions.keys():
            self._update_min_max_time_points_for_addition(time_point_number)

    def detach_position(self, position: Position):
        """Removes a position from a time point."""
        positions_at_time_point = self._all_positions.get(position.time_point_number())
        if positions_at_time_point is None:
            return

        positions_at_time_point.detach_position(position)

        # Remove time point entirely
        if positions_at_time_point.is_empty():
            del self._all_positions[position.time_point_number()]
            self._update_min_max_time_points_for_removal()

    def of_time_point_with_shapes(self, time_point: TimePoint) -> Dict[Position, ParticleShape]:
        """Gets all positions and shapes of a time point. New positions must be added using self.add(...), not using
        this dict."""
        positions_at_time_point = self._all_positions.get(time_point.time_point_number())
        if not positions_at_time_point:
            return dict()
        return positions_at_time_point.positions_and_shapes()

    def get_shape(self, position: Position) -> ParticleShape:
        positions_at_time_point = self._all_positions.get(position.time_point_number())
        if positions_at_time_point is None:
            return UnknownShape()
        return positions_at_time_point.get_shape(position)

    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point that contains positions, or None if there are no positions stored."""
        return self._min_time_point_number

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) that contains positions, or None if there are no positions stored."""
        return self._max_time_point_number

    def exists(self, position: Position) -> bool:
        """Returns whether the given position is part of the experiment."""
        positions_at_time_point = self._all_positions.get(position.time_point_number())
        if positions_at_time_point is None:
            return False
        return position in positions_at_time_point.positions()

    def __len__(self):
        """Returns the total number of positions across all time points."""
        count = 0
        for positions_at_time_point in self._all_positions.values():
            count += len(positions_at_time_point)
        return count

    def __iter__(self):
        """Iterates over all positions."""
        for positions_at_time_point in self._all_positions.values():
            yield from positions_at_time_point.positions()


