from typing import Dict, AbstractSet, Optional, Iterable, Set

from organoid_tracker.core import TimePoint, min_none, max_none
from organoid_tracker.core.position import Position


class _PositionsAtTimePoint:
    """Holds the positions of a single point in time."""

    _positions: Dict[int, Set[Position]]

    def __init__(self):
        self._positions = dict()

    def contains_position(self, position: Position) -> bool:
        at_z = self._positions.get(round(position.z))
        if at_z is None:
            return False
        return position in at_z

    def positions(self) -> Iterable[Position]:
        for positions_at_z in self._positions.values():
            yield from positions_at_z

    def add_position(self, position: Position):
        """Adds a position to this time point. If the position was already added, but a shape was provided, its shape is
        replaced."""
        at_z = self._positions.get(round(position.z))
        if at_z is None:
            at_z = set()
            self._positions[round(position.z)] = at_z
        at_z.add(position)

    def detach_position(self, position: Position) -> bool:
        """Removes a single position. Does nothing if that position was not in this time point. Does not remove a
        position from the linking graph. See also Experiment.remove_position."""
        at_z = self._positions.get(round(position.z))
        if at_z is None or position not in at_z:
            return False

        at_z.remove(position)
        if len(at_z) == 0:  # No positions at z layer, remove those too
            del self._positions[round(position.z)]
        return True

    def is_empty(self):
        """Returns True if there are no positions stored."""
        return len(self._positions) == 0

    def __len__(self) -> int:
        total = 0
        for positions in self._positions.values():
            total += len(positions)
        return total

    def copy(self) -> "_PositionsAtTimePoint":
        """Gets a deep copy of this object. Changes to the returned object will not affect this object, and vice versa.
        """
        copy = _PositionsAtTimePoint()
        for z, positions in self._positions.items():
            copy._positions[z] = positions.copy()
        return copy

    def positions_nearby_z(self, z: int) -> Iterable[Position]:
        """Yields all positions for which round(position.z) == z"""
        if z in self._positions:
            yield from self._positions[z]

    def lowest_z(self) -> Optional[int]:
        """Gets the lowest z in use, or None if there are no positions stored."""
        if len(self._positions) == 0:
            return None
        return min(self._positions.keys())

    def highest_z(self) -> Optional[int]:
        """Gets the highest z in use, or None if there are no positions stored."""
        if len(self._positions) == 0:
            return None
        return max(self._positions.keys())

    def count_nearby_z(self, z: int) -> int:
        """Returns all postions for which round(position.z) == z."""
        if z in self._positions:
            return len(self._positions[z])
        return 0

    def _move_in_time(self, time_point_offset: int):
        """Must only be called from PositionCollection, otherwise the indexing is wrong."""
        for z in list(self._positions.keys()):
            old_position_set = self._positions[z]
            new_position_set = {position.with_time_point_number(position.time_point_number() + time_point_offset)
                                for position in old_position_set}
            self._positions[z] = new_position_set


class PositionCollection:

    _all_positions: Dict[int, _PositionsAtTimePoint]
    _min_time_point_number: Optional[int] = None
    _max_time_point_number: Optional[int] = None

    def __init__(self, positions: Iterable[Position] = ()):
        """Creates a new positions collection with the given positions already present."""
        self._all_positions = dict()
        for position in positions:
            self.add(position)


    def of_time_point(self, time_point: TimePoint) -> AbstractSet[Position]:
        """Returns all positions for a given time point. Returns an empty set if that time point doesn't exist."""
        positions_at_time_point = self._all_positions.get(time_point.time_point_number())
        if not positions_at_time_point:
            return set()
        return set(positions_at_time_point.positions())

    def detach_all_for_time_point(self, time_point: TimePoint):
        """Removes all positions for a given time point, if any."""
        if time_point.time_point_number() in self._all_positions:
            del self._all_positions[time_point.time_point_number()]
            self._recalculate_min_max_time_points()

    def add(self, position: Position):
        """Adds a position, optionally with the given shape. The position must have a time point specified."""
        time_point_number = position.time_point_number()
        if time_point_number is None:
            raise ValueError("Position does not have a time point, so it cannot be added")

        self._update_min_max_time_points_for_addition(time_point_number)

        positions_at_time_point = self._all_positions.get(time_point_number)
        if positions_at_time_point is None:
            positions_at_time_point = _PositionsAtTimePoint()
            self._all_positions[time_point_number] = positions_at_time_point
        positions_at_time_point.add_position(position)

    def _update_min_max_time_points_for_addition(self, new_time_point_number: int):
        """Bookkeeping: makes sure the min and max time points are updated when a new time point is added"""
        if self._min_time_point_number is None or new_time_point_number < self._min_time_point_number:
            self._min_time_point_number = new_time_point_number
        if self._max_time_point_number is None or new_time_point_number > self._max_time_point_number:
            self._max_time_point_number = new_time_point_number

    def _recalculate_min_max_time_points(self):
        """Bookkeeping: recalculates min and max time point if a time point was removed."""
        # Reset min and max, then repopulate by readding all time points
        self._min_time_point_number = None
        self._max_time_point_number = None
        for time_point_number in self._all_positions.keys():
            self._update_min_max_time_points_for_addition(time_point_number)

    def move_position(self, old_position: Position, new_position: Position):
        """Moves a position, keeping its shape. Does nothing if the position is not in this collection. Raises a value
        error if the time points the provided positions are None or if they do not match."""
        if old_position.time_point_number() != new_position.time_point_number():
            raise ValueError("Time points are different")

        time_point_number = old_position.time_point_number()
        if time_point_number is None:
            raise ValueError("Position does not have a time point, so it cannot be added")

        positions_at_time_point = self._all_positions.get(time_point_number)
        if positions_at_time_point is None:
            return  # Position was not in collection
        if positions_at_time_point.detach_position(old_position):
            positions_at_time_point.add_position(new_position)

    def detach_position(self, position: Position):
        """Removes a position from a time point. Does nothing if the position is not in this collection."""
        positions_at_time_point = self._all_positions.get(position.time_point_number())
        if positions_at_time_point is None:
            return

        if positions_at_time_point.detach_position(position):

            # Remove time point entirely if necessary
            if positions_at_time_point.is_empty():
                del self._all_positions[position.time_point_number()]
                self._recalculate_min_max_time_points()

    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point that contains positions, or None if there are no positions stored."""
        return self._min_time_point_number

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) that contains positions, or None if there are no positions stored."""
        return self._max_time_point_number

    def first_time_point(self) -> Optional[TimePoint]:
        """Gets the first time point that contains positions, or None if there are no positions stored."""
        return TimePoint(self._min_time_point_number) if self._min_time_point_number is not None else None

    def last_time_point(self) -> Optional[TimePoint]:
        """Gets the last time point (inclusive) that contains positions, or None if there are no positions stored."""
        return TimePoint(self._max_time_point_number) if self._max_time_point_number is not None else None

    def contains_position(self, position: Position) -> bool:
        """Returns whether the given position is part of the experiment."""
        positions_at_time_point = self._all_positions.get(position.time_point_number())
        if positions_at_time_point is None:
            return False
        return positions_at_time_point.contains_position(position)

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

    def has_positions(self) -> bool:
        """Returns True if there are any positions stored here."""
        return len(self._all_positions) > 0

    def add_positions(self, other: "PositionCollection"):
        """Adds all positions and shapes of the other collection to this collection."""
        for time_point_number, other_positions in other._all_positions.items():
            if time_point_number in self._all_positions:
                # Merge positions
                self_positions = self._all_positions[time_point_number]
                for position in other_positions.positions():
                    self_positions.add_position(position)
            else:
                # Just copy in
                self._all_positions[time_point_number] = other_positions.copy()

        self._recalculate_min_max_time_points()

    def time_points(self) -> Iterable[TimePoint]:
        """Returns all time points from self.first_time_point() to self.last_time_point()."""
        return TimePoint.range(self.first_time_point(), self.last_time_point())

    def copy(self) -> "PositionCollection":
        """Creates a copy of this positions collection. Changes made to the copy will not affect this instance and vice
        versa."""
        the_copy = PositionCollection()
        for key, value in self._all_positions.items():
            the_copy._all_positions[key] = value.copy()

        the_copy._min_time_point_number = self._min_time_point_number
        the_copy._max_time_point_number = self._max_time_point_number
        return the_copy

    def nearby_z(self, z: int) -> Iterable[Position]:
        """Returns all positions (for any time point) for which round(position.z) == z."""
        for positions_at_time_point in self._all_positions.values():
            yield from positions_at_time_point.positions_nearby_z(z)

    def of_time_point_and_z(self, time_point: TimePoint, z_min: Optional[int] = None, z_max: Optional[int] = None
                            ) -> Iterable[Position]:
        """Gets all positions that are nearby the given min to max z, inclusive."""
        of_time_point = self._all_positions.get(time_point.time_point_number())
        if of_time_point is None:
            return

        if z_min is None and z_max is None:
            # Just return all positions
            yield from of_time_point.positions()
            return

        # Calculate z range
        if z_min is None:
            z_min = of_time_point.lowest_z()
        if z_max is None:
            z_max = of_time_point.highest_z()

        # Return positions within range
        for z in range(z_min, z_max + 1):
            yield from of_time_point.positions_nearby_z(z)

    def lowest_z(self) -> Optional[int]:
        """Returns the lowest z in use, or None if there are no positions in this collection."""
        return min_none(positions_at_time_point.lowest_z() for positions_at_time_point in self._all_positions.values())

    def highest_z(self) -> Optional[int]:
        """Returns the lowest z in use, or None if there are no positions in this collection."""
        return max_none(positions_at_time_point.highest_z() for positions_at_time_point in self._all_positions.values())

    def count_positions(self, *, time_point: Optional[TimePoint], z: Optional[int]):
        """Counts the number of positions at the given time point and the given z. If no z or time point is given,
        positions at all time points and/or all z will be counted."""
        if time_point is not None:
            # Specific time point
            at_time_point = self._all_positions.get(time_point.time_point_number())
            if at_time_point is None:
                return 0  # No positions at this time point

            if z is not None:
                # Specific time point, specific z
                return at_time_point.count_nearby_z(z)
            else:
                # Count at any z
                return len(at_time_point)
        else:
            if z is not None:
                # All time points, specific z
                count = 0
                for positions_at_time_point in self._all_positions.values():
                    count += positions_at_time_point.count_nearby_z(z)
                return count
            else:
                # All time points, all z
                return len(self)

    def move_in_time(self, time_point_delta: int):
        """Moves all data with the given time point delta."""
        new_positions_dict = dict()
        for time_point_number, values_old in self._all_positions.items():
            values_old._move_in_time(time_point_delta)
            new_positions_dict[time_point_number + time_point_delta] = values_old
        self._all_positions = new_positions_dict
